# Server Architecture Deep-Dive — End-to-End Audit

**Date:** 2026-04-28
**Author:** automated codebase audit + literature comparison
**Scope:** the entire HTTP inference stack — `Vendor/mlx-swift-lm` → `LLMActor` → `Features/Server/*` → wire → client.
**Sources:** every Swift file in `tesseract/Features/Server/`, key files in `tesseract/Features/Agent/` (`LLMActor.swift`, `AgentEngine.swift`, `InferenceArbiter.swift`, `AgentGeneration.swift`, `ParoQuant/`), the vendored fork `Vendor/mlx-swift-lm/Libraries/{MLXLLM,MLXLMCommon}/`, plus existing design docs in `docs/`. Industry comparison covers vLLM, llama.cpp, SGLang, TensorRT‑LLM, HuggingFace TGI, Ollama, mlx‑lm, and Anthropic's Claude API. Apple guidance pulled from WWDC sessions and developer.apple.com.

---

## 0. Executive Summary

Tesseract's server stack is a **purpose-built, single-tenant inference plane** for a local Apple-Silicon Mac. Three architectural choices distinguish it from every other open-source LLM server:

1. **Single in-flight inference** through `InferenceArbiter` — a FIFO actor that serialises GPU access. There is no continuous batching, no PagedAttention, no per-sequence interleaving.
2. **Token-level radix-tree prefix cache** with a *three-state lifecycle* (RAM → SSD → committed) and **two semantic checkpoint types** (`.system` stable-prefix and `.leaf` conversation-leaf). Built on a heavily forked **`HybridCacheSnapshot`** that captures Mamba state-space layers + attention KV in a single immutable object.
3. **A privately-forked MLX Swift LM** (`branch test/tesseract-integration-v3`, 40 commits ahead of upstream) carrying three Tesseract-only systems: **TriAttention** sparse KV pruning, **ParoQuant** for AutoAWQ Qwen3.5 PARO checkpoints, and the **HybridCacheSnapshot** machinery the prefix cache depends on.

**The verdict.** The architecture is *coherent* and *well-tested* for its target workload (one user, one device, one LLM agent at a time, with multiple coding-agent HTTP clients hitting the same model). It is *not yet production-grade* in the sense industry expects: there are concrete bugs in `HybridCacheSnapshot.restore` (drops TriAttention `runtimeState`), thirteen call sites that `fatalError` on malformed cache bytes, no `fsync`/`F_FULLFSYNC` on SSD writes, no Prometheus metrics, no per-request timeout, a force-unwrap in `ToolCallConverter`, an empty-string in `CompletionHandler:185` that should be the physical model ID, and `try?` swallowing JSON encoding errors in `HTTPResponse.json`. The four worst issues are listed in §6.1 with file:line.

**Where Tesseract leads** any local server (mlx-lm, Ollama, llama.cpp slot-based reuse): semantic stable-prefix detection (two-probe technique), tiered SSD cold-storage of KV state, hybrid Mamba+attention snapshot capture, comprehensive event-typed diagnostics, and a Marconi-style alpha tuner with a 21-candidate grid search.

**Where Tesseract trails** the GPU-server tier (vLLM, SGLang, TGI): no continuous batching, no FP8 KV quantisation, no streaming tool-call argument deltas, no grammar-constrained generation, no `/metrics` endpoint, no chunked-prefill ceiling for >128k prompts, manual HTTP/1.1 parser (no NIO/HTTP-Types), and no spec-decoding.

The remainder of this report is the data behind these claims and a prioritised work list to close the gaps.

---

## 1. The Stack at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Wire client (OpenCode, Continue, Aider, internal Agent chat)                       │
│  HTTP/1.1 + SSE     POST /v1/chat/completions         x-session-affinity (opt)     │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ TCP 127.0.0.1:8321
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  HTTPServer.swift    (905 LOC, @MainActor @Observable)                              │
│  Network.framework  NWListener → NWConnection (one detached Task per conn)          │
│  Manual HTTP/1.1 parser — accumulates into 50 MB cap (line 827)                     │
│  HTTPRequest, HTTPResponse, HTTPResponseWriter, SSEWriter, HTTPConnectionLifecycle  │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ route = /v1/chat/completions
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  CompletionHandler.swift (1224 LOC, struct Sendable)                                │
│  • JSONDecoder → OpenAI.ChatCompletionRequest                                        │
│  • ModelSelection enum: useSettings | override | unknown | notDownloaded            │
│  • HTTPRequestLogger.shared.logRequest(...) → tmp/tesseract-debug/...               │
│  • ServerGenerationLog.startRequest(...) → live activity dashboard                  │
│  • withAcquisitionTimeout(60s) → arbiter.withExclusiveGPU(.llm, …) {                │
│  •   runStreamingCompletion / runNonStreamingCompletion                             │
│  • }                                                                                 │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ inside lease
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  Glue: MessageConverter (451 LOC) + InferenceArbiter (417 LOC)                      │
│  • OpenAI msgs → systemPrompt + [LLMMessage] (reorder tool results, multimodal)     │
│  • Prefix-cache eligibility analysis (HTTPPrefixCacheEligibility)                   │
│  • Session replay via HTTPPrefixCacheSessionReplayStore (recovers reasoning)        │
│  • InferenceArbiter holds a FIFO waiter queue + isLeased flag (single in-flight)    │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ ServerInferenceService.start(req)
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  AgentEngine.generateServerTextCompletion (Features/Agent/AgentEngine.swift)        │
│   ↓ delegates to                                                                    │
│  LLMActor.generateServerTextCompletion  (Features/Agent/LLMActor.swift, ~780 LOC    │
│  per method) — the inference nucleus                                                │
│                                                                                      │
│  ┌─ inside container.perform (Metal-affine) ───────────────────────────────────┐   │
│  │ tokenize chat → detect stablePrefix (StablePrefixDetector, two-probe)       │   │
│  │ → PrefixCacheManager.lookupAndPlanCheckpoints                                │   │
│  │   (radix tree partition by (modelID,kvBits,kvGroupSize,fingerprint,         │   │
│  │    triAttentionIdentity))                                                    │   │
│  │ → if SSD-only ref: SSDSnapshotStore hydrate → HybridCacheSnapshot.restore   │   │
│  │ → slice suffix tokens → TokenIterator + generateTask()                       │   │
│  │ → mid-prefill checkpoint capture at planned offsets                          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  Stream loop (MainActor): per .text/.thinking/.toolCall event                       │
│   → ThinkingSafeguardObserver → on intervention, cancel + start continuation         │
│   → vendor-dropped tool-call buffer recovery                                         │
│   → store leaf snapshot to radix tree (HybridCacheSnapshot.capture)                  │
│   → AlphaTuner.recordRequest (Marconi feedback)                                      │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ AgentGeneration events
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  Vendor MLX Swift LM (Vendor/mlx-swift-lm, branch test/tesseract-integration-v3)    │
│  • MLXLMCommon/Evaluate.swift     TokenIterator, GenerateParameters                 │
│    threaded with checkpointAtOffsets / checkpointBaseOffset                         │
│  • MLXLMCommon/HybridCacheSnapshot.swift  capture/restore (Mamba+attn+quant+        │
│    TriAttention) — schema-versioned, safetensors-serializable                       │
│  • MLXLMCommon/TriAttention*.swift  sparse KV pruning (5 files, ~2500 LOC,          │
│    custom calibration artifact + scoring + sparse cache + GQA mask expansion)       │
│  • MLXLMCommon/ParoQuant/  ParoQuantLoader (AWQ unpack, fused proj split) +         │
│    RotateQuantizedLinear (Givens rotation Metal kernel)                             │
│  • MLXLMCommon/Tool/ToolCallProcessor + XMLFunctionParser (Qwen3.5 XML)             │
│  • MLXLLM/Models/Qwen35.swift   heterogeneous (24 Mamba + 8 attention) cache,       │
│    full-attention every 4th layer                                                   │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │ MLX kernels
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│  MLX Swift 0.31.3 → Metal → Apple Silicon unified memory                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

The system is *thirteen* Swift files in `Features/Server/`, ~10 500 LOC, plus the inference nucleus (`LLMActor.swift`, single method ~780 LOC), plus ~5 000 LOC of Tesseract-only additions in the MLX vendor fork.

---

## 2. HTTP Transport — `HTTPServer.swift`

### 2.1 What it is

A hand-rolled HTTP/1.1 server on Apple's `Network.framework`. `NWListener` runs on a `global(qos: .userInitiated)` background queue (line 717-ish). Each accepted `NWConnection` is taken in `newConnectionHandler` and handed to a `Task.detached` (line 724) that monitors the peer for disconnect, parses the request, dispatches a route, and writes the response. The `HTTPServer` itself is `@MainActor @Observable` (line 380-ish) — it surfaces the listener state to SwiftUI but the actual I/O is off-main.

The wire types are deliberately small:
- `HTTPRequest` (method, path, headers, body, `header(name)` case-insensitive lookup) — `Sendable` value.
- `HTTPResponse` (statusCode, statusText, headers, body) with factories: `.json`, `.jsonBody`, `.error`, `.badRequest`, `.modelNotFound`, `.openAIError`. Two distinct error envelopes — a legacy `OpenAIError` with `code: Int` (a pre-existing wire bug, kept so existing tests don't break — see lines 175-188) and the strict `OpenAIErrorStrict` with string `code` and explicit `param: nil` encoding (lines 198-229) used for new code paths like `model_not_found`.
- `HTTPResponseWriter` (`@unchecked Sendable` over `NWConnection`) for streaming. Bridges `NWConnection.send` callbacks via `withCheckedThrowingContinuation`.
- `SSEWriter` for the `text/event-stream` framing (`data: …\n\n`, `[DONE]`, comments for keepalives).
- `HTTPConnectionLifecycle` actor for thread-safe peer-disconnect signalling. A poller `Task` calls `connection.receive()` with 50 ms sleeps and marks the lifecycle as disconnected on EOF. Streaming consumers `await waitForDisconnect()` to learn when to cancel.

### 2.2 What's good

- **No external deps.** Network.framework is the right primitive for a local-only Apple-Silicon server. Apple positions `swift-nio` for cross-platform server-side; Tesseract ships only on macOS, so NIO would be over-engineered. `swift-nio-transport-services` exists precisely to bridge NIO to Network.framework on Apple — but its complexity isn't justified here.
- **Sandbox correctness.** Both `tesseract.entitlements` and `tesseractRelease.entitlements` carry `app-sandbox`, `network-server`, `network-client`. The 127.0.0.1 binding is permitted within the App Sandbox — `network-server` is required for *any* listening socket, even loopback.
- **Two error shapes.** Deliberately keeping the legacy buggy `OpenAIError` to avoid breaking the regression test at `CompletionHandlerTests` while routing all *new* error responses through `openAIError(...)` is a textbook example of cheap forward-compat: documented (lines 93-103), tested, and bounded.
- **`Connection: close` + `Content-Length` defaults.** No keep-alive, no HTTP/2 — sane for the workload (one long-lived SSE stream per request).
- **Per-connection `Task.detached`** is the canonical way to bridge Network.framework's queue model into Swift Concurrency on macOS today.

### 2.3 What's smelly or wrong

| Issue | File:Line | Severity | Notes |
|------|-----------|---------:|-------|
| `try? encoder.encode(value) ?? Data("{}".utf8)` swallows JSON encoding errors silently | `HTTPServer.swift:50` | Med | Client receives `{}` with no log; should `Log.server.error` and 500. |
| No request idle/read timeout | `HTTPServer.swift:825-846` | Med | Malicious or buggy client can dribble bytes for hours; the 50 MB cap doesn't bound time. |
| `try? await finish()` on streaming early returns | `HTTPServer.swift` (SSEWriter) and `CompletionHandler.swift:249` | Low | Drops cleanup errors; acceptable but worth a `Log.warning`. |
| `withCheckedThrowingContinuation` in `writeAll` (lines ~303-313) can leak if NWConnection is silently torn down without invoking the completion | rare, but real | Low | Wrap in `withTaskCancellationHandler` and call `connection.cancel()` on cancel; resume the continuation in both paths exactly once. |
| No CORS — but spec says "no browser clients planned" | `docs/HTTP_SERVER_SPEC.md:92` | None | Correct call; document the threat model in code so reviewers don't add `*` later. |
| `HTTPMethod` is closed enum — adding `OPTIONS` for browser clients later requires API churn | `HTTPServer.swift:7-9` | Cosmetic | Already includes `OPTIONS`. Fine. |

The deepest concern is the **lack of a per-request timeout**. Industry servers (vLLM, TGI) have `timeout_keep_alive`, `request_timeout`, and graceful drain semantics. Tesseract's only timeout is the 60-second *lease acquisition* timeout (`CompletionHandler.leaseTimeoutSeconds`); once generation begins, the connection can stay open as long as the model wants to talk. The disconnect monitor (the 50 ms-tick `monitorPeerDisconnect`) covers the *peer-leaves* case; it does not cover *peer wedged-but-alive*.

### 2.4 Apple guidance and what to do differently

Apple's recommendations for `Network.framework` (WWDC18 #715, WWDC20 #110, the `Network` developer docs):

- **`NWConnection.send(...)` should always have its completion bridged** to async — Tesseract does this. The only caveat is **resume-exactly-once**; `withTaskCancellationHandler` is the right pattern when the consumer can disappear (peer disconnect, request cancel). Tesseract's `HTTPConnectionLifecycle` actor does this correctly.
- **Backpressure on `send`**: Network.framework has no public API for the kernel send-buffer high-water mark. The right pattern is *await each send* before queuing the next — which Tesseract does in `SSEWriter.send`. There is no internal queue, so a slow consumer naturally throttles the producer; this couples generation throughput to network throughput, but for a single local user that is acceptable.
- **Listener queues**: Apple's docs say the listener should run on its own `DispatchQueue`; `global(qos: .userInitiated)` is fine for a single-port service.
- **TLS**: not required for `127.0.0.1`. If Tesseract ever adds remote access, `NWParameters.tls` with `.tls12+` is the recommended path; manage certs via Keychain.

**One concrete change worth landing**: keep a per-`NWConnection` deadline. When the connection is opened, schedule a `DispatchWorkItem` (or `Task` with `Task.sleep(for: …)`) that calls `connection.cancel()` if neither headers nor first SSE chunk arrives within N seconds. Configure default 30 s (parser) + ∞ (post-lease, defer to disconnect monitor). See vLLM's `timeout_keep_alive=5s` (default) and TGI's `keep_alive_timeout=20s` for prior art.

---

## 3. Request Orchestration — `CompletionHandler.swift`

### 3.1 The flow, end to end

```
handle(request, writer):
  1. Decode body → ChatCompletionRequest (400 on parse fail, 400 if messages empty)
  2. ModelSelection.resolve(requestModel, agentIDs, statuses)            (lines 145-154)
        useSettings     → fall back to selectedAgentModelID
        override(id)    → llmModelIDOverride = id
        unknown(id)     → 404 model_not_found / unknownID
        notDownloaded   → 404 model_not_found / notDownloaded
  3. HTTPRequestLogger.shared.logRequest(body, sessionAffinity)         (line 175)
  4. activityLog.startRequest(...) → TraceHandle                         (line 187)
  5. withAcquisitionTimeout(60s) {
        arbiter.withExclusiveGPU(.llm, llmModelIDOverride) {
            signal.set()
            activityLog.markLeaseAcquired
            runCompletion(...)                                           (line 202)
        }
     }
     map errors → CancellationError → 503; LeaseTimeout → 503 + Retry-After: 5;
                  AgentEngineError.modelNotDownloaded → 404; other → 503/500
```

`runCompletion` then dispatches on `request.stream`. Streaming spins up a `withTaskGroup` of *three* concurrent tasks (lines 577-624):

1. **Disconnect monitor** awaits `writer.waitForDisconnect()`; on disconnect, calls `start.cancel()` on the generation handle.
2. **Keepalive** every 250 ms: if SSE has been idle, write a `: keepalive\n\n` comment line.
3. **Event loop** consumes `start.stream` and emits SSE chunks per `AgentGeneration` event variant (`.text`, `.thinking`, `.toolCall`, `.toolCallBufferDelta`, `.thinkTruncate`, `.malformedToolCall`, `.info`).

First task to finish wins; the other two are cancelled via `group.cancelAll()`. The final SSE chunk emits `finish_reason` (`stop` / `length` / `tool_calls`) plus optional `usage` if `stream_options.include_usage = true`, plus the **vendor extension** `tesseract_thinking_safeguard` when the safeguard fired.

The *non-streaming* path simply accumulates events into local strings/arrays, encodes the response on `MainActor.run`, sends once. Both paths call `HTTPPrefixCacheSessionReplayStore.record(...)` afterwards so a future request with the same `x-session-affinity` can recover `reasoning_content` even if the client failed to round-trip it.

### 3.2 OpenAI compatibility — what the wire actually says

Read from `Features/Server/Models/OpenAITypes.swift`.

**Honoured request fields:**
- `model` — routed via `ModelSelection`, echoed in response.
- `messages` — converted to internal `LLMMessage`. `system` messages at the head are concatenated into `systemPrompt`; mid-stream `system` messages stay inline. Tool results are *reordered* (line 188-242 of `MessageConverter`) to match the preceding assistant turn's tool-call order, because Qwen3.5's chat template is positional.
- `tools` — converted to `ToolSpec` for prompt rendering.
- `stream` — controls SSE vs single response.
- `max_tokens` *and* `max_completion_tokens` — the latter takes precedence if both are present (`effectiveMaxTokens`, OpenAITypes.swift line 37-39).
- `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` — plumbed to `AgentGenerateParameters`.
- `stream_options.include_usage` — gates the trailing `usage` object in the final SSE chunk.
- `reasoning_content` and `reasoning` (legacy field) on assistant messages — `resolvedReasoningContent` prefers the new field with fallback (line 121-125).

**Silently ignored:**
- `stop` (request field exists but isn't surfaced to the generator)
- `reasoning_effort` (parsed but ignored — Qwen3.5 has fixed thinking behaviour)
- `n`, `logprobs`, `seed`, `response_format`, `tool_choice`, `parallel_tool_calls` — none of these are in `OpenAITypes`. A client sending `response_format: {"type":"json_object"}` will not get JSON-constrained generation; the server will happily produce non-JSON output. This is the **single biggest OpenAI-compat gap** for coding-agent clients.

**Vendor extensions on the wire:**
- `thinking_safeguard` request struct (enabled, max_thinking_chars, etc.) — controls `ThinkingRepetitionDetector`.
- `tesseract_thinking_safeguard` response sidecar — reports `safe_prefix_chars` if intervention fired.
- `system_fingerprint: "tesseract-1.0-mlx"` — opaque model identifier.

**Tool-call wire format:**
- Inbound: standard OpenAI `ToolDefinition` (`type:"function"`, `function:{name,description,parameters}`).
- Outbound non-streaming: `ToolCallConverter.convertToOpenAI` assigns `id: "call_<UUID>"` (lines 15-26), serialises arguments via `ToolArgumentNormalizer.encode` to a JSON string. Index field is set.
- Outbound streaming: **two chunks per tool call** — first with `name` + `id` and empty arguments, second with the full `arguments` JSON string. This is *buffered* (whole call, not per-token). Compare with Anthropic `content_block_delta` for `tool_use` which streams tokens. See §6.

### 3.3 Notable design decisions

- **Pre-lease validation.** `ModelSelection.resolve` runs *before* the queue, so an unknown model gets a 404 immediately rather than waiting up to 60 s for the lease. Models deleted between pre-lease and post-load (race) get the same 404 via `AgentEngineError.modelNotDownloaded` mapped explicitly (line 223).
- **Session affinity** (`x-session-affinity` header) partitions the `HTTPPrefixCacheSessionReplayStore` so reasoning recovered for *session A* can never bleed into *session B* even if both share a model.
- **Empty-stop diagnosis.** When generation ends with no text, no tool calls, and no info, a structured log entry captures buffer lengths, token counts, and the raw malformed-tool-call buffer (lines 641-662 / 915-929) — extremely useful for debugging Qwen3.5 anomalies.
- **`MainActor.run` for response encoding** in the non-streaming path. Sound: `JSONEncoder` is `Sendable` since 2022 and isn't blocked from off-main use, but the architectural discipline of "encode where you read state" prevents accidental cross-actor races on the response object graph.

### 3.4 What's wrong or smelly

| Issue | File:Line | Severity | Fix |
|------|-----------|---------:|-----|
| `physical: ""` passed to `echoModelID` for non-streaming | `CompletionHandler.swift:185` | **Med — almost certainly a bug** | Should pass `inferenceService.currentModelState()?.modelID ?? ""`. The streaming path (line ~564) uses the correct value. The non-streaming activity log will record an empty model name when client sent no `model`. |
| Force-unwrap `message.tool_calls!` | `ToolCallConverter.swift:51` | Low — actually safe (value-type copy on line 46) but reads as a code smell | Use `?.` or pull out a `let calls = message.tool_calls ?? []`. |
| Silent encoding-failure default to `{}` | `HTTPServer.swift:50` | Med | Log + 500. |
| `streamGenerationEvents` is 109 lines with 40+ case branches; `runStreamingCompletion` is 236 lines | `CompletionHandler.swift:530-771, 1033-1151` | Cosmetic | Split per event variant. Tests will guide decomposition. |
| Keepalive is unconditional 250 ms regardless of write latency | `CompletionHandler.swift:588-607` | Low | Track last-flush timestamp; skip keepalive if a chunk was sent in the last 200 ms. |
| Catch-all `catch let error` in stream loop logs only `localizedDescription` | `CompletionHandler.swift:1143-1148` | Low | Log the type + cause for actionable triage. |
| Unstructured `request.stop` ignored | `OpenAITypes.swift` | Low | If clients depend on it, plumb to `AgentGenerateParameters`. |
| **No structured-output / JSON-schema-constrained generation** | n/a | **Med — feature gap** | See §5 (SGLang XGrammar) and §8.P0. |

---

## 4. The Inference Spine — `InferenceArbiter`, `LLMActor`, `AgentEngine`

### 4.1 `InferenceArbiter` (417 LOC)

`@Observable @MainActor`. Holds:
- A FIFO `[(id: UUID, continuation: CheckedContinuation<Void, any Error>)]` waiter queue.
- An `isLeased: Bool` flag.
- A separate `[idleWaiters]` queue for "wait until truly idle" (background tasks).

`withExclusiveGPU(slot, llmModelIDOverride, body)` is the only entry point. It correctly handles three cancellation windows:

1. While **enqueued**: cancel removes the waiter from `waiters` and resumes with `CancellationError` (line 148-159). The `onCancel` Task hops to MainActor to mutate the queue safely.
2. After **resumed but before lease**: `Task.checkCancellation()` at line 164 throws if cancelled in the handoff race.
3. While **holding the lease**: cancellation propagates through the body; `defer` always releases.

The `defer` block (lines 170-187) is the **atomic handoff**: if waiters exist, resume the next *without* clearing `isLeased`, so the next runner inherits the lease without a window where new arrivals could bypass the queue. This is correct, well-commented, and matches the design in WWDC23 #10170 ("Beyond the basics of structured concurrency").

`ensureLoaded` (lines 283-350) is responsible for:
- Comparing requested `(modelID, visionMode, requestedTriAttention)` against `loadedLLMState` and reloading on mismatch.
- Awaiting `agentEngine.awaitPendingUnload()` — critical, because unload is fire-and-forget detached and clears `triAttentionRuntimeSelection` on its way out. Without the drain, a fresh load could see its TriAttention selection wiped.

### 4.2 `LLMActor.generateServerTextCompletion` (~780 LOC, lines 472-1253)

This is *the* function that turns an HTTP request into a token stream. It is too big — but the logic is genuinely irreducible because of the four tightly-coupled concerns it juggles:

1. **Cache lookup + checkpoint planning** (lines 479-510): tokenize the full conversation, run `StablePrefixDetector.detect()`, call `prefixCache.lookupAndPlanCheckpoints()`, hydrate any SSD-only refs.
2. **Vendor generation start**: `makeHTTPPrefixCacheGeneration` runs inside `container.perform` (Metal-affine), slices the suffix tokens, instantiates `TokenIterator` from the snapshot's restored cache, kicks off `MLXLMCommon.generateTask()`.
3. **Stream + safeguard loop** (lines 549-805): consumes `Generation` events. On `.thinking` chunks, feeds them to `ThinkingSafeguardObserver`. **If the safeguard fires** (line 772), cancels the current stream, awaits its drain, calls a continuation starter that re-tokenises with `safePrefix + injectionMessage + </think>` and resumes generation from there. Wraps the cancel/wait closures in an `OSAllocatedUnfairLock<PathAHandleBox>` so client cancellation always targets the *current* upstream generator, not the old one.
4. **Post-generation leaf store** (lines 908-1188): re-tokenise the stored conversation (`prompt + assistant turn`), pick a leaf-store mode (`canonicalUserLeaf` / `directToolLeaf` / `directLeaf`), guard against offset misalignment when normalisation trims the conversation (Mamba and TriAttention state can't be partially unwound), capture a `HybridCacheSnapshot` from the final cache, admit it to the radix tree.

There are smaller-but-important details:

- **Vendor-dropped tool-call recovery** (lines 824-854): the fork's `ToolCallProcessor` silently drops in-flight `<tool_call>…` buffers on EOS. Tesseract accumulates `libraryToolCallBufferAccum` across `.toolCallBufferDelta` events and surfaces it as a `.malformedToolCall` event so the HTTP layer can log + recover.
- **Two `Memory.clearCache()` calls** (lines 814, 930) — on cancellation and on task completion. Justified by `mlx-swift-lm-prefill-memory-research.md`: peak unified-memory during prefill can approach 2× chunk size; clearing between chunks bounds it.
- **Memory math** (lines 109-140): `cacheLimitMB = 2048` (MLX free buffer pool), `prefixCacheHeadroomBytes = 20 GiB`, `available = totalRAM − modelWeights − 20 GiB`, `prefixCacheBudget = available / 2`. On a 48 GiB Mac with a 4.8 GiB model: 23.2 GiB available, ~11.6 GiB prefix-cache budget.
- **Sampling stack** is the vendor's: `MLXLMCommon.TokenIterator` runs `temperature → top-k → top-p → repetition penalty → frequency penalty → multinomial sample` on Metal as a fused kernel sequence. Tesseract does **not** override the order. There is no logit-bias support, no `logits_processor` plugin point exposed via the HTTP API.

### 4.3 `AgentEngine` (the MainActor wrapper)

`generateServerTextCompletion` (lines 231-274) is a thin shim:
1. Verify model loaded (else `.modelNotLoaded`).
2. Try `llmActor.generateServerTextCompletion()` (the prefix-cache path).
3. If that returns nil (eligibility fail), fall back to `llmActor.startStandardChatGeneration()`.
4. Wrap the result in `startManagedHTTPGeneration` to add safeguard plumbing.

Clean separation: actor holds Metal-affine state, MainActor wrapper publishes UI state and orchestrates errors.

### 4.4 What's good

- **FIFO with cancellation correctness.** The arbiter's three cancellation windows are textbook, including the `onCancel` MainActor hop with `weak self` and the post-resumption `checkCancellation()` guard. This is *materially* harder than it looks.
- **Cache-aware unload draining.** Forcing `awaitPendingUnload()` before reload eliminates the obvious race where TriAttention runtime state survives across a model swap.
- **Bitwise-correctness gate** (`scripts/dev.sh hybrid-cache-correctness`). Not a unit test — it loads a real model, prefills 4 000 tokens, captures snapshots at 1k/2k/3k/4k, restores each into a fresh cache, generates 100 tokens, compares logits to a cold prefill at the same offset. This is the strongest correctness guarantee a system this complex can have.
- **Vendor-error escape valve.** The malformed-tool-call accumulator is exactly the kind of resilience industry servers lack; vLLM and TGI just drop the partial output.

### 4.5 What's smelly or wrong

| Issue | File:Line | Severity | Notes |
|------|-----------|---------:|-------|
| `generateServerTextCompletion` is 780 lines in one function | `LLMActor.swift:472-1253` | Cosmetic, but encourages bugs | Split: `setupCache(...)`, `runStreamLoop(...)`, `storeLeafIfEligible(...)`. Pass an inout `RequestState` struct. |
| Magic `20 GiB` headroom | `LLMActor.swift:135` | Low | Make a function of `ProcessInfo.processInfo.physicalMemory`; conservative by ratio rather than constant. |
| Magic `prefillStepSize = 1024` | `AgentGenerateParameters` | Low | Per-model preset; expose to HTTP only as advanced. |
| Magic `60 s` lease timeout | `CompletionHandler:14` | Low | Distinguish *queue* timeout (default 60 s) from *load* timeout (different path). A model load can legitimately take > 60 s on cold disk. |
| `nonisolated(unsafe)` on `ParoQuant.RotateQuantizedLinear.kernelCache` | `RotateQuantizedLinear.swift:89` | Low | Justified (single-threaded actor owns it), but the annotation is an escape hatch. Add a comment + assertion. |
| `UnsafeSendableBox` wrapping `mlxStart` for the outer Task | `LLMActor.swift:10-16` | Low | Same — justified, but auditable; verify with Swift 6.2 strict-concurrency in CI. |
| `try!` and `fatalError` in vendor `HybridCacheSnapshot.makeQuantizedCache` / `makeRotatingCache` | `Vendor/.../HybridCacheSnapshot.swift:165-300` (~13 sites) | **High — process-crash on corrupt SSD bytes** | Replace with `Result<…, SnapshotError>` return; propagate to caller; on hard fail, evict the entry and continue. |
| `ParoQuant` module uses `NSLock` for kernel cache | `RotateQuantizedLinear.swift` (per audit, line ~89 area) | Low under single-flight, **medium** if Phase-2 batching lands | Move to atomic-or-actor; lock-free reads after warm-up. |

---

## 5. Prefix Cache — the largest and most novel subsystem

The prefix cache is **8 files, 5 600 LOC** in `Features/Server/`, plus design doc (2 821 LOC), plus `HybridCacheSnapshot` in the vendor fork. It is also the most operationally novel thing Tesseract does: nothing else in the open-source landscape persists Mamba+attention KV state to NVMe with a stable-prefix detector and Marconi-style alpha tuning.

### 5.1 Three-tier hierarchy

| Tier | Backing | Lookup | Eviction | Notes |
|------|---------|--------|----------|-------|
| 1 — radix tree | `HybridCacheSnapshot` in RAM | `TokenRadixTree.findBestSnapshot` walks compressed token paths (line 76-123) | Marconi utility (recency + α·FLOPs/byte) over eligible nodes, type-protected | Primary |
| 2 — committed SSD ref | `SnapshotStorageRef` pointing to a `.safetensors` blob on disk | Same lookup; if a node's `storageRef.committed && body == nil`, hydrate from disk | `SSDSnapshotStore` writer admission-time LRU cut, type-protected | Optional, gated by `SSDPrefixCacheConfig` |
| 3 — pending SSD admission | Body still in RAM, `storageRef.pending` | Front-door `tryEnqueue` synchronously enforces `maxPendingBytes` (line 232-336) | Drop-oldest-pending under burst | Bridge between 1 and 2 |

Five-state lifecycle table (verbatim from `TieredSnapshotStore.swift`, `SnapshotManifest.swift:291-320`):

| State | RAM body | Storage ref | Committed | Meaning |
|------:|:--------:|:-----------:|:---------:|---------|
| 1 | yes | none | — | RAM-only |
| 2 | yes | pending | false | Admission in flight; body still hot |
| 3 | no | pending | false | Body evicted mid-write; hydration pending |
| 4 | yes | committed | true | RAM + SSD insurance copy |
| 5 | no | committed | true | SSD-only; lazy hydration on next hit |

State-3 and state-5 nodes are the corners that make the system tricky. Eviction must guard against removing a node with a live `storageRef` — both `evictNode()` and `collapseSingleChildNode()` carry the assertion `storageRef == nil` (`PrefixCacheManager.swift:1038-1044`, `TokenRadixTree.swift:253-259`).

### 5.2 Partition key — what isolates which caches

```swift
struct CachePartitionKey: Hashable {
    let modelID: String
    let kvBits: Int?              // nil unquantised, else 4 or 8
    let kvGroupSize: Int?         // 32, 64, or 128
    let modelFingerprint: String  // SHA-256 of weights + tokenizer
    let triAttention: TriAttentionIdentity?
}
```

(Schema bumped to v6 with the addition of TriAttention prefix-protection mode in the canonicalisation, per `SnapshotManifestSchema.swift:42-61`.)

What is **deliberately not** in the partition key:
- Tool definitions
- Template digests

The radix tree handles those implicitly — different tools produce different tokens, which produce different paths in the same partition's tree. Documented in `PrefixCacheManager.swift:82-85`. This is the right call: it lets two conversations with overlapping but non-identical tool definitions still share the system-prompt prefix.

### 5.3 Two checkpoint types in production

`.system` — the **stable-prefix boundary**, captured mid-prefill via `StablePrefixDetector`'s two-probe technique (`StablePrefixDetector.swift:22-82`). Tokenises the system message + tools + a probe-A user message, then again with probe-B; the common prefix length *is* the offset where stable content ends. Reject artefacts where common prefix < 1/3 of full prompt and full > 1 000 tokens (Jinja template non-determinism). **Type-protected** — `eligibleEvictionNodes()` (`TokenRadixTree.swift:308-312`) excludes `.system` snapshots from utility scoring.

`.leaf` — the **conversation-leaf boundary**, captured *post-generation* from the final cache. Modes (`LLMActor` lines 990-998):
- `canonicalUserLeaf` — thinking templates, captured from last-real-user-message boundary.
- `directToolLeaf` — non-thinking with tool calls; from last-message boundary.
- `directLeaf` — fallback; final cache.
**Supersession**: storing a descendant leaf evicts older ancestor leaves on the same branch in both RAM and SSD (`PrefixCacheManager.swift:637-674`). Eager — if the next request *branches* off an older ancestor instead of extending the leaf, that ancestor is gone. Acceptable for within-conversation reuse but a known footgun for branching.

(There is also a `.branchPoint` checkpoint type defined for Phase-2 speculative; captured but utility-evictable.)

### 5.4 The `HybridCacheSnapshot` itself

Lives in `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`, ~900 LOC, fully Tesseract-only.

Capture (lines 74-107) deep-copies all per-layer arrays via `state.map { $0[.ellipsis] }`. Each `LayerState` carries `className: String`, `state: [MLXArray]`, `metaState: [String]`, `offset: Int`. Supports KVCacheSimple, RotatingKVCache, QuantizedKVCache, MambaCache, TriAttentionSparseKVCache, ChunkedKVCache; returns `nil` for unsupported types (e.g. legacy `CacheList`).

Restore (lines 111-160) constructs the correct subclass per layer based on `className`, parses `metaState` into constructor args. Quantised caches get `groupSize`/`bits` from metaState; rotating caches get the 5-tuple `(keep, maxCacheSize, step, offset, idx)`; Mamba caches get `(convState, recurrentState)` directly.

#### The big bug

**Restore drops `runtimeState` on TriAttention sparse caches.** `TriAttentionSparseKVCache.runtimeState` is non-public, computed at calibration-artifact load time, and not part of the serialised payload. After `restore()`, `pruneIfNeeded()` checks `runtimeState != nil` and silently no-ops. This means **a snapshot lookup that hydrates a TriAttention partition resumes generation with sparse-attention state but no live pruning**. The model continues to use the retained K/V positions captured at snapshot time, but as new tokens append, the cache will grow unbounded until the next generation ends.

For Tesseract's current single-in-flight, single-conversation cadence, this is *probably* not catastrophic — turns are short, the next miss/eviction repopulates state. But it is a **correctness bug** waiting to bite long sessions. Documented in `triattention-upstream-deviations.md` as known. **P0 to fix**.

The fix is to either:
1. Persist the calibration-artifact identity in the snapshot, and re-attach `runtimeState` from a runtime registry on restore (preferred — already the design intent of `triAttention` in the partition key).
2. Or, deliberately disable TriAttention pruning on restored caches with a flag and log a warning. (Cheap stop-gap.)

#### `fatalError` on bad bytes

There are ~13 `fatalError` call sites in `HybridCacheSnapshot.swift` and `KVCache.swift` for malformed metaState during restore. A single corrupt byte in a `.safetensors` file on disk crashes the process — not just the request. **P0 to fix**: change the contract to `throws` and let `PrefixCacheManager` evict the entry and continue with a cold prefill.

### 5.5 SSD store — `SSDSnapshotStore.swift` (1 943 LOC)

The largest single file in the server. Combines:
- A front-door `tryEnqueue` (synchronous, MainActor-callable, never suspends; lines 232-336) that enforces `maxPendingBytes = min(4 GiB, RAM/16)` via drop-oldest-pending.
- A detached writer `Task` that drains the pending queue serially, runs admission-time type-protected LRU cuts when the on-disk budget is exceeded, calls `onDrop` on disk-full failures.
- A debounced manifest persister (`scheduleManifestPersistLocked`, default 500 ms) that writes-to-temp + atomic rename.
- Warm-start recovery: read manifest → if missing or schema-mismatched, rename to `.bak` and start cold. (Does **not** rebuild from a directory walk + per-safetensors-header descriptor parse — that recovery path is sketched in the design doc but not implemented.)

#### What's good

- **NSLock-bounded critical sections** without crossing await boundaries (line 155 area). Correct.
- **Five-state lifecycle invariants** are enforced at every transition — node mutations cross-check `storageRef.committed`.
- **Schema versioning**: bumping forces a cold start, never a partial migration; partitions store TriAttention identity since v5. `manifest.v{old}.bak` is preserved for forensics.

#### What's smelly or wrong

| Issue | File:Line area | Severity | Notes |
|------|----------|---------:|-------|
| **No `fsync`/`F_FULLFSYNC` after data write before atomic rename** | `SSDSnapshotStore.swift` writer loop ~750-850 | **Med — power-loss risk** | macOS-specific call: `fcntl(fd, F_FULLFSYNC)`. Without it, `rename(2)` succeeds while the data may not have hit physical media; manifest claims persistence, file is empty after crash. |
| **Lock held while calling `FileManager.default.removeItem` and `createDirectory`** | writer loop ~650-850 | **Med under burst** | Both can block on disk I/O; MainActor `tryEnqueue` waits on the same lock. Move I/O off the lock — capture a "to-delete" list under lock, release, perform I/O, re-acquire only for accounting. |
| **No `SSDEvictionEvent` diagnostic on writer admission-time LRU** | writer loop | Low | When the writer evicts to make room, no event is forwarded to `PrefixCacheDiagnostics`. Logs lose the cause/correlation. |
| **Stringly-typed `checkpointType` in wire format** | `SnapshotManifest.swift:117` | Low | Adding an enum case requires careful bump of schema version + filter on warm start. Document the migration policy. |
| **Lazy manifest rebuild from safetensors headers not implemented** | warm-start path | Low | Today: corrupt manifest → cold start. The design doc anticipates a header-walk fallback; safetensors headers carry partial descriptor info, but full descriptor mirror isn't there yet. |
| **O(n) lock-held drop-oldest-pending loop in `tryEnqueue`** | `SSDSnapshotStore.swift:289-296` | Low | Build a drop list and remove after release. |

### 5.6 Marconi alpha tuner — `AlphaTuner.swift`

A 314-LOC implementation of the bootstrap tuner from the [Marconi paper](https://assets.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf) and [reference repo](https://github.com/ruipeterpan/marconi).

State machine:
1. `waitingForFirstEviction` — count requests until first eviction triggers.
2. `bootstrapping` — record post-eviction requests up to `bootstrapTarget = clamp(requestsBeforeFirstEviction × 1, 10, 60)`.
3. `tuned` — replay window across α ∈ {0.0, 0.1, …, 2.0} on a sandbox cache seeded with production inventory + time-shifted recency, pick the α that maximises (FLOPs saved, hit tokens) lexicographically, write back to `EvictionPolicy.alpha`.

The 1× multiplier (vs the paper's 5×) is justified — Tesseract is single-user, samples are scarce, 1× is reachable in production whereas 5× would never converge. The 21-candidate grid is paper-faithful.

**The smell**: `AlphaTuner.replayWindow` (line 228) constructs a sandbox `PrefixCacheManager(alphaTuner: nil)` to break the recursion. There is **no compile-time guard** against a future caller passing the production tuner into the sandbox; an `assert(alphaTuner == nil)` at sandbox-init would lock that down.

### 5.7 Diagnostics — `PrefixCacheDiagnostics.swift` (680 LOC)

This is *very good* observability for a local server. Five event types: `LookupEvent`, `CaptureEvent`, `EvictionEvent`, `MemoryEvent`, `TTFTEvent`. Each carries a per-request `Context` (requestID, modelID, kvBits, kvGroupSize). Rendered to `Log.agent` as structured key=value lines and forwarded to a telemetry sink. Sample (from the audit):

```
[prefix-cache] eviction strategy=utility offset=3000 checkpointType=leaf freedBytes=150000000
  budgetBytes=12000000000 snapshotBytesAfter=11950000000
  normalizedRecency=0.2 normalizedFlopEfficiency=0.8 utility=0.36
```

The gap is **no Prometheus surface** — no `/metrics` endpoint emitting these as scrapeable counters/gauges. For a server that wants to be a backend for OpenCode / Continue / Aider in production-ish use, that's a blocker for monitoring.

---

## 6. The Vendor Fork — `Vendor/mlx-swift-lm`

Forked from `ml-explore/mlx-swift-examples`, branch `test/tesseract-integration-v3`, **40 commits ahead of upstream**. The fork carries three Tesseract-only bodies of code that block easy upstream:

### 6.1 TriAttention (5 files, ~2 500 LOC)

Sparse KV pruning for Qwen3.5 PARO. Uses a per-sampled-head frequency-domain scoring kernel (`TriAttentionRuntime.swift` ~lines 1-450), retains the top-K positions that maximise

  `score(p) = Σ_f λ_f · (r_f cos θ_f − j_f sin θ_f) + (E|Q_f| − |E[Q_f]|) · |K_f|`

(canonical Marconi-aligned form, now matching upstream PyTorch and vLLM). Calibration artefact: `TriAttention/v1/<sha256>.pt`, custom pickle decoder via ZIPFoundation. Configuration (`TriAttentionConfiguration.swift`, 110 LOC): `enabled`, `budgetTokens` (12 000), `calibrationArtifactIdentity`, `prefixProtectionMode`.

**Two open issues** flagged by the vendor audit:

1. **Restore drops `runtimeState`** — already discussed in §5.4; documented as the single highest-severity bug in the prefix-cache stack.
2. **`prefixProtectionMode` is declared but not enforced in runtime.** The default `protectStablePrefixOnly` looks like it should keep TriAttention from pruning the system+tools prefix, but `pruneIfNeeded` doesn't check the boundary. Combined with `includePrefillInBudget=true`, this lets TriAttention evict positions inside the system prefix on long-prompt prefills (12 K+).

Tests cover the scoring math, sparse cache update/trim, and quantised path; **no test covers restore-then-saturate, no test covers prefix-boundary protection, no test covers long-sequence saturation**. Coverage gap.

### 6.2 ParoQuant (2 files, ~700 LOC)

Loader for AutoAWQ-quantised Qwen3.5 PARO checkpoints:
1. `ParoQuantLoader.swift` — detect via `quant_method == "paroquant"` in config.json, AWQ unpack (`[0,2,4,6,1,3,5,7]` reorder), split fused Mamba `in_proj_ba` into `in_proj_a` + `in_proj_b`, reset `model_type` to `qwen3_5`.
2. `RotateQuantizedLinear.swift` — Givens rotations applied via a custom Metal kernel before the quantised matmul. Uses `nonisolated(unsafe) var kernelCache` guarded by `NSLock` per the latest commit (line ~89).

The recent commit `c14a307 fix(paroquant): use @ParameterInfo(key:) for channelScales` fixes a regression with mlx-swift's macro system. This integration is **fragile across mlx-swift updates** — a minor mlx-swift bump could break it. Pin the dependency tightly, or contribute the macro-friendly accessor pattern upstream.

### 6.3 `HybridCacheSnapshot`

Already covered in §5.4. The two open bugs (restore-drops-runtimeState, fatalError-on-bad-bytes) are both **upstream-blocking** for the snapshot machinery itself. Within Tesseract, fix locally; do **not** propose upstreaming until correctness is restored.

### 6.4 What's safe to keep, what should be upstreamed, what's risky

| Component | Safe to keep | Upstream candidate | Risk |
|-----------|:------------:|:------------------:|------|
| TriAttention runtime + cache | yes (with bug fix) | no — Tesseract-specific calibration artefacts and policy decisions | runtimeState drop, prefill protection underspecified |
| ParoQuant loader + rotated linear | yes | partial — could upstream as opt-in | macro-system fragility, NSLock contention if Phase-2 batching lands |
| `HybridCacheSnapshot` (capture/restore) | yes (with `Result`-typed restore) | maybe — strip TriAttention/snapshot logic, keep checkpoint-parameter threading | fatalError sites |
| `FinalizedKVCacheHandle` | yes | yes — minimal, general utility | none |
| `KVCache.copy()` | already upstream (PR #158) | n/a | n/a |
| Chunked prefill + `Memory.clearCache()` between chunks | yes | yes — general perf, opt-in flag | none |
| Tool/`ToolCallProcessor` schema threading | yes | yes — additive over upstream PR #162 | none |
| `Qwen35.swift` heterogeneous-cache plumbing | yes | unlikely — model-specific | TriAttention plumbing depends on runtimeState fix |

---

## 7. Industry Comparison

### 7.1 Prefix / KV cache

| System | Mechanism | Granularity | Cold tier |
|--------|-----------|-------------|-----------|
| **vLLM** | PagedAttention, OS-style block table | 16-token blocks | optional CPU swap |
| **llama.cpp server** | slot KV reuse, hash similarity | whole slot | manual save (`--slot-save-path`) |
| **SGLang** | RadixAttention (token-level radix) | leaf node | none |
| **TensorRT-LLM** | paged KV in TRT kernels | 16-token blocks | none |
| **TGI** | PagedAttention + FP8 KV quant | 16-token blocks | none |
| **mlx-lm** (`mlx_lm.server`) | none | n/a | n/a |
| **Anthropic Claude API** | automatic prompt caching, ephemeral (5 m) + extended (1 h) | 1 k–4 k token blocks | server-managed |
| **Tesseract** | token-level radix tree, semantic stable-prefix detector | per-snapshot (variable) | **NVMe-backed `SSDSnapshotStore`** |

Only **SGLang** has a comparable token-level radix; only **Tesseract** has SSD persistence; only **Anthropic** has a cleanly two-tier ephemeral-vs-extended scope. Tesseract's *semantic* stable-prefix detection (which finds the system+tools boundary by tokenising twice with two probes) is a real innovation beyond all of these, because it gives the cache the largest possible *type-protected* root that is *guaranteed* shared across conversations.

### 7.2 Concurrent batching

vLLM, SGLang, TGI: **continuous batching**, per-token interleaving across in-flight sequences, preemption to CPU under KV pressure.

llama.cpp: **slot batching** — N independent slots, each its own KV; if all slots full, requests queue.

Tesseract: **single in-flight** through `InferenceArbiter`. The HTTP_SERVER_SPEC v1 promised "default 4 concurrent" but the shipped reality is one. Two reasons it's actually fine for Tesseract's scope:

1. **MLX unified memory** removes the host↔device copy cost that PagedAttention is solving. There's no fragmentation to fight.
2. **Single-user local agent** — concurrency is requested by humans, who are the bottleneck. Ten parallel queries from one user is unusual.

But it leaves money on the table for the OpenCode-style use case where one HTTP client may issue **up to 4** parallel requests during agentic editing. Phase 2 of the original design contemplates this; the architecture is the right shape (replace the single-flag arbiter with a token-level scheduler) but the implementation is non-trivial.

### 7.3 OpenAI compatibility — fields / streaming / tool calls

| Field / behaviour | OpenAI | vLLM | llama.cpp | TGI | mlx-lm | **Tesseract** | Anthropic |
|------|:------:|:----:|:---------:|:---:|:------:|:-------------:|:---------:|
| `response_format: json_object` | yes | yes | yes | yes | partial | **no** | yes (tool-call) |
| `response_format: json_schema` (strict) | yes | yes (XGrammar via SGLang) | partial | partial | no | **no** | yes |
| `tool_choice: required` | yes | yes | yes | yes | partial | **partial** (parsed but not enforced) | yes |
| `seed` (deterministic sampling) | yes | yes | yes | yes | yes | **no** | yes |
| `logit_bias` | yes | yes | yes | yes | partial | **no** | yes |
| `stream_options.include_usage` | yes | yes | partial | yes | yes | **yes** | yes |
| Streaming tool-call args (per-token) | yes | partial | partial | partial | partial | **no — buffered** | **yes** |
| `reasoning_content` field | yes (gpt-5 family) | yes (DeepSeek-R1 family) | partial | partial | partial | **yes** (Qwen3.5 `<think>`) | extended thinking blocks |
| `usage.prompt_tokens_details.cached_tokens` | yes | yes | partial | yes | partial | **yes** | yes (cache stats) |

The two-cell-wide column for Tesseract reads "good wire compat for chat history, partial for tool routing, missing for structured output / determinism / logit bias". The gap that *most* hurts coding agents is structured output — see §8.P0.

### 7.4 Speculative decoding & spec sampling

vLLM, SGLang, TensorRT-LLM all ship Medusa / EAGLE. Tesseract: none. For single-in-flight, the wins are smaller — speculative decoding shines under continuous batching where the verifier can absorb spare compute. Defer until Phase 2.

### 7.5 Observability

vLLM, TGI: Prometheus `/metrics`, OpenTelemetry hooks.
SGLang: dashboard + Ray traces.
Ollama: log lines.
Tesseract: structured `os.Logger` + a SwiftUI activity dashboard + per-request artefact files in `tmp/tesseract-debug/http-completions/`.

The local-first dashboard is wonderful for a single-user app. The missing piece is the **`/metrics` endpoint**: nine lines of Prometheus exposition would let any operator pipe Tesseract into Grafana, alert on cache-hit-rate regressions, or compare two model loads. P1.

### 7.6 What Tesseract gets right that others get wrong

1. **MLX unified memory** sidesteps the entire PagedAttention problem class. Tesseract's chunked prefill + radix tree is the right shape for the underlying hardware; vLLM's paged blocks would be redundant.
2. **Two-probe stable-prefix detection** is genuinely better than vLLM/SGLang block-aligned matching for chat workloads — it doesn't depend on prompt-suffix bytes accidentally aligning to a 16-token boundary.
3. **Tiered SSD cold storage** is unique. llama.cpp slots are saveable but not automatic; nobody else persists KV state to NVMe. For long-running agent sessions where the system prompt is huge and the user comes back tomorrow, this is a real win.
4. **HybridCacheSnapshot for Mamba+attention** is necessary correctness, not optional perf. SGLang ([MambaRadixCache blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)) is the only other system that meaningfully addresses hybrid models, and it's still attention-first.
5. **Comprehensive event-typed diagnostics** (5 event variants, per-request correlation, structured rendering) is more disciplined than most.
6. **Bitwise-correctness gate** (`hybrid-cache-correctness`) — not a unit test but a loaded-model logit-equivalence gate. Industry runs perplexity regressions; Tesseract runs bitwise.
7. **Detailed empty-stop diagnosis logging** in `CompletionHandler` — the kind of thing you only get from someone who's debugged a Qwen3.5 model misbehaving in production.

### 7.7 What Tesseract gets wrong relative to industry

1. **No structured output / grammar-constrained generation.** SGLang XGrammar, OpenAI strict JSON schema, Anthropic strict tool-use are all mainstream now. Coding agents rely on it.
2. **No FP8 KV quantisation.** TGI ships it; SGLang ships it. 2× cache-size reduction with <1% perplexity loss. Apple Silicon has the bandwidth; the math fits cleanly with the existing `kvBits/kvGroupSize` partitioning.
3. **Buffered tool-call streaming.** Anthropic streams tokens; vLLM streams tokens; Tesseract emits two whole chunks per call. Latency cost on a 1 KB JSON arg: ~500 ms perceived.
4. **No `seed` / `logit_bias` / `response_format`.** Three OpenAI fields routinely used by coding agents.
5. **Single in-flight.** By design, but the spec called for 4-concurrent and the realised system is 1.
6. **No `/metrics`** for monitoring integration.
7. **No chunked-prefill ceiling** — ample for normal prompts but a 131 K-token prompt today triggers a long-prefill spike that the current `Memory.clearCache()` gating only partially mitigates.
8. **No request timeout.** vLLM's `--timeout-keep-alive`, TGI's `keep_alive_timeout`. Tesseract has only the lease timeout.
9. **No `/v1/embeddings`, no `/v1/completions`** — explicit non-goals in the spec, fine, but worth re-evaluating for coding-agent compat.
10. **No `system_fingerprint` derived from real model+config hash** — currently a static string. Should be the SHA-256 fingerprint already computed.

---

## 8. Apple Guidelines — what to do exactly

### 8.1 Network.framework

- **Use `NWListener` with `NWParameters.tcp` for TCP-only**. Tesseract does this; correct.
- **Bridge `NWConnection.send` callbacks via `withCheckedThrowingContinuation`** with `withTaskCancellationHandler` so a cancelled task can `connection.cancel()` and resume the continuation exactly once. Tesseract does this; correct.
- **Do not buffer outbound bytes on top of NWConnection's queue**; await each send before the next. Correct.
- **Apple does not provide an HTTP parser**; manual is the documented path. Correct.
- **Listener queue**: `DispatchQueue.global(qos: .userInitiated)` is appropriate for a low-volume local server. Correct.
- **TLS**: not needed for `127.0.0.1`. If you ever add remote, use `NWParameters(tls: NWProtocolTLS.Options())` and provision certs via Keychain (not a static `.cer` shipped in the bundle).

### 8.2 Swift Concurrency (Swift 6.2)

- **`@MainActor`** on `HTTPServer`, `InferenceArbiter`, `AgentEngine` is appropriate — these own UI state.
- **Custom `actor`** for `LLMActor` and `ContextManager` is appropriate — they own non-UI mutable state with strong serialisation needs.
- **`nonisolated(unsafe)`** is used in two places (`UnsafeSendableBox`, `ParoQuant` kernel cache). Both are *correct* given current actor invariants but require defensive comments + assertions because Swift 6.2's strict-concurrency cannot prove safety.
- **Task vs Task.detached**: Tesseract uses `Task.detached` for HTTP per-connection handlers (correct — they run off-main, not inheriting `MainActor`) and `Task` inside MainActor methods to spawn structured children (correct).
- **Continuations**: use `CheckedContinuation`, never `UnsafeContinuation`. Tesseract uses `CheckedContinuation`. Correct.
- **AsyncStream backpressure**: `AsyncStream` has none. Tesseract uses `AsyncThrowingStream` from `Generation` events through to the SSE writer; the `await` on each `send` is what backpressures the producer. Correct, but worth a comment in `runStreamingCompletion` clarifying this contract.
- **SE-0414 (region-based isolation)** + **SE-0466 (`@concurrent` / inferred isolation)** in Swift 6.2: the compiler now lets you pass non-`Sendable` values across isolation boundaries when control flow proves no aliasing. Tesseract should drop a few `Sendable` wrappers that exist only to satisfy 6.0 strict concurrency; rebuild on 6.2 and remove what compiles without them.
- **`OSAllocatedUnfairLock`** is the right primitive for the `PathAHandleBox` in `LLMActor` (it's used). Avoid `os_unfair_lock_t` raw, avoid `NSLock` unless interop forces it.
- **Observation framework**: Tesseract uses `@Observable` widely. For high-frequency updates (per-token activity log) the right pattern is to coalesce: throttle `version` bumps to 100 ms minimum, batch text-append events to 33 ms windows. `ServerGenerationLog.swift` lines 14-23 show this is already done. Correct.
- **Swift 6.2 `Observations` async sequence** (the new `Observations(of:)`) is preferable to `withObservationTracking` for non-view consumers; see Tesseract's own use of `Observations` in DependencyContainer and AppDelegate per CLAUDE.md.

### 8.3 MLX Swift

- **Unified memory**: MLX arrays are CPU-and-GPU-visible; no `cudaMemcpy` cost. Tesseract relies on this implicitly throughout. Correct.
- **`Memory.clearCache()`** is the published API for releasing transient buffers; not `MLX.GPU.clearCache()` (the latter is a misremembered earlier name). Tesseract calls this explicitly between prefill chunks and on cancellation. Correct.
- **`Memory.cacheLimit`**: Tesseract sets to 2 GiB at load time (`LLMActor.swift:319, 366, 408`). Reasonable.
- **Lazy evaluation**: MLX arrays are symbolic until `eval()`. Tesseract doesn't call `eval()` in the generation hot path (the vendor `TokenIterator` does it internally on yield). Correct.
- **`ModelContainer.perform`** is the Metal-affine scope. All cache capture/restore happens inside `container.perform`. Correct.
- **Quantised KV cache**: `QuantizedKVCache(groupSize:bits:)` matches `loadPromptCache` semantics. Tesseract preserves the metaState-encoded `(groupSize, bits)` round-trip in `HybridCacheSnapshot`. Correct.
- **Metal JIT entitlement**: `com.apple.security.cs.allow-jit` should be present in Release entitlements for MLX. **Verify both `.entitlements` files include this** — CLAUDE.md notes they're "currently identical" but call out this specifically; missing it would cause Metal kernel compilation to fail under hardened runtime.

### 8.4 App Sandbox & Notarisation

- `app-sandbox` + `network-server` + `network-client` — present, correct.
- `NSMicrophoneUsageDescription`, `NSAccessibilityUsageDescription` — present, correct.
- Privacy manifest declares no tracking, UserDefaults-only access — correct.
- **Verify** `com.apple.security.cs.allow-jit` is in **both** `.entitlements` files — this is the load-bearing one for MLX Metal kernel compilation under hardened runtime; if absent in Release, JIT compile fails silently and falls back to a slower path or aborts.

### 8.5 Logging & observability

- `os.Logger` via the `Log` enum (CLAUDE.md says it uses `PublicLogger` to mark all interpolations `.public`). Correct for development; **for production**, mark prompts and arguments `.private` to avoid leaking user content via `log stream`.
- `OSSignposter` for instrumentation — not currently used. Worth adding for prefill / decode / cache lookup intervals so Instruments can render the timeline. P2.
- `os_log` privacy markers for sensitive data (`%{private}@` for prompts) — verify the `Log` macros default to `.public` only for *non-sensitive* fields.

---

## 9. Critical Findings — One Triaged List

### 9.1 P0 (correctness; ship a fix this cycle)

1. **`HybridCacheSnapshot.restore` drops `runtimeState`** for `TriAttentionSparseKVCache`. After hydration, pruning silently no-ops; long sessions can drift. — `Vendor/.../HybridCacheSnapshot.swift:111-160`. Fix: persist `calibrationArtifactIdentity` in the snapshot, re-attach `runtimeState` from a runtime registry on restore. Or hard-disable TriAttention on restored caches with a logged warning as a stop-gap.
2. **`fatalError` in cache-restore parsers** — ~13 sites. A corrupt SSD byte crashes the process, not the request. — `Vendor/.../HybridCacheSnapshot.swift:165-300`, `KVCache.swift`. Fix: convert to `throws`, evict the entry and continue cold.
3. **No `fsync` / `F_FULLFSYNC` before atomic rename** in `SSDSnapshotStore` writer. Power loss → manifest persisted but data not on media. — `SSDSnapshotStore.swift` writer loop. Fix: `fcntl(fd, F_FULLFSYNC)` after data write, before close + rename.
4. **`prefixProtectionMode` declared but not enforced.** TriAttention can prune inside the system prefix on long prompts. — `TriAttentionConfiguration.swift:37`, `TriAttentionRuntime.swift` ~line 640. Fix: thread the mode through the prune kernel; gate eviction by offset boundary.
5. **`physical: ""` passed to `echoModelID`** in non-streaming activity logging — almost certainly a missed value. — `CompletionHandler.swift:185`. Fix: `physical: inferenceService.currentModelState()?.modelID ?? ""`.

### 9.2 P1 (operational gaps; ship within two cycles)

6. **No request idle timeout** on the HTTP layer; only the 60 s lease timeout. — `HTTPServer.swift:825-846`. Fix: per-connection `Task.sleep` deadline gating header + first-chunk arrival.
7. **No `/metrics` Prometheus endpoint.** Diagnostics are world-class internally but invisible to monitoring. — Add `tesseract_inference_*` counters/gauges; expose at `/metrics`.
8. **No structured output / JSON-schema-constrained generation.** Coding-agent clients (OpenCode, Continue) rely on this. Adopt SGLang's XGrammar pattern or LM Studio's grammar enforcement at sample time.
9. **Buffered tool-call streaming.** Stream argument tokens via `content_block_delta`-style events. — `CompletionHandler.swift:1090-1119`.
10. **Lock held across `FileManager` I/O** in SSDSnapshotStore writer. Burst admissions can stall MainActor `tryEnqueue`. — Fix: capture todo lists under lock, release, do I/O, re-acquire only for accounting.
11. **`try? encoder.encode(...)` swallows errors** in `HTTPResponse.json`. — `HTTPServer.swift:50`. Fix: `do { … } catch { Log.server.error(…); return error500 }`.
12. **Force-unwrap `message.tool_calls!`** in `ToolCallConverter`. — `ToolCallConverter.swift:51`. Fix: `let calls = message.tool_calls ?? []`.
13. **`system_fingerprint` is a static string.** — Use the SHA-256 already computed by `ModelFingerprint.computeFingerprint()`.
14. **No `seed`, `logit_bias`, `response_format`, `tool_choice: required`.** Wire-compat gaps with mainstream OpenAI clients. Add to `OpenAITypes` and plumb through `MessageConverter` + `AgentGenerateParameters`.

### 9.3 P2 (polish, deeper bets)

15. **Phase-2 concurrent batching.** Replace `InferenceArbiter` single-flag with a token-level scheduler. Pattern: vLLM continuous batching, scoped to 2-4 in-flight. Massive change, big payoff for OpenCode-style agentic editing.
16. **FP8 KV quantisation.** Add `kvBits = 8` and `kvBits = "fp8"` (mantissa-encoded) variants with proper canonical groupSize; benchmark against current Q4. 2× cache-size reduction, modest accuracy cost.
17. **Chunked prefill ceiling** for >128 K prompts. Today the `Memory.clearCache()` between chunks bounds peak; add an explicit chunk-size adaptation under memory pressure.
18. **OSSignposter** for prefill / decode / cache lookup. Surfaces in Instruments → System Trace.
19. **Refactor `LLMActor.generateServerTextCompletion`** (780 LOC) into `setupCache`, `runStreamLoop`, `storeLeaf`. Tests will guide the boundaries.
20. **Manifest auto-rebuild** from safetensors header walk on corruption (sketched in design doc, not implemented).
21. **Speculative decoding (Medusa).** Defer until concurrent batching lands.
22. **Persist KV across server restart** in a session-resumable format (llama.cpp slots pattern).
23. **AlphaTuner reentrancy guard** — `assert(alphaTuner == nil)` at sandbox-init.
24. **Upstream-friendly extraction.** Minimal `FinalizedKVCacheHandle` and chunked-prefill changes are ready for upstream; `HybridCacheSnapshot` and TriAttention are not.

---

## 10. Strengths Worth Preserving

1. **`InferenceArbiter` cancellation correctness.** The three-window handling (in-queue, post-resume-pre-lease, in-flight) with the MainActor-hopping `onCancel` is *materially* harder than it looks and rare in the wild.
2. **Stable-prefix two-probe detector.** Genuinely novel for chat templates; tokenisation-correct in a way that token-level radix matching alone is not.
3. **HybridCacheSnapshot and tiered SSD store.** No other open-source local server persists Mamba+attention KV to NVMe with a manifest, schema versioning, and atomic admission.
4. **Marconi alpha tuner** with bootstrap + 21-candidate grid + inventory-seeded replay. Production-faithful.
5. **Diagnostics event taxonomy.** Five typed events with per-request correlation context. Trivial to forward to a real telemetry backend when needed.
6. **Bitwise correctness gate.** Mid-prefill checkpoint restore → 100 tokens generation → logit equality to a cold prefill at the same offset. Caught real bugs (`triattention-upstream-deviations.md`).
7. **Empty-stop diagnosis logging** in `CompletionHandler`. Reads like it was designed by someone who's debugged a Qwen3.5 outage.
8. **Pre-lease validation.** `ModelSelection.resolve` returning 404 *before* the queue is the right call — many servers wait the full lease timeout to tell the client "model not found".
9. **Session replay store** (`HTTPPrefixCacheSessionReplayStore`). Recovers `reasoning_content` for clients that don't round-trip it.
10. **Two error envelopes** to preserve regression-test compatibility while fixing the wire format for new paths. Cheap forward-compat.

---

## 11. Improvement Roadmap (Concrete, Ordered)

```
Sprint 1 — correctness (≤2 weeks)
  P0.1  Persist & re-attach TriAttention runtimeState on restore
  P0.2  Convert HybridCacheSnapshot fatalError → throws + evict
  P0.3  F_FULLFSYNC in SSDSnapshotStore writer pre-rename
  P0.4  Enforce prefixProtectionMode in TriAttention prune kernel
  P0.5  CompletionHandler:185 — pass real physical model ID
  P1.6  Per-connection request timeout (30 s parser; ∞ post-lease)
  P1.11 try? → do/catch in HTTPResponse.json
  P1.12 Drop force-unwrap in ToolCallConverter
  P1.13 system_fingerprint = ModelFingerprint hash

Sprint 2 — wire-compat & observability
  P1.7  /metrics Prometheus endpoint
  P1.8  Grammar-constrained generation (XGrammar-style)
  P1.9  Token-level streaming of tool-call arguments
  P1.10 Move FileManager I/O off the SSDSnapshotStore lock
  P1.14 seed / logit_bias / response_format / tool_choice plumbing

Sprint 3 — performance bets
  P2.16 FP8 KV quantisation experiment
  P2.17 Adaptive chunked-prefill ceiling
  P2.18 OSSignposter instrumentation across prefill / decode / cache lookup
  P2.19 Refactor LLMActor.generateServerTextCompletion into 3 helpers

Backlog
  P2.15 Concurrent batching scheduler (replace single-flag arbiter)
  P2.20 Manifest auto-rebuild from safetensors headers
  P2.21 Speculative decoding (Medusa) — gated on concurrent batching
  P2.22 Session-resumable KV persistence (llama.cpp slot pattern)
  P2.23 AlphaTuner reentrancy assertion
  P2.24 Upstream-friendly extraction of FinalizedKVCacheHandle / chunked prefill
```

---

## 12. Appendix — File Reference Index

```
HTTP transport
  HTTPServer.swift                      905 LOC  — NWListener/NWConnection, parser, SSE
  HTTPRequestLogger.swift                67 LOC  — disk mirror to tmp/tesseract-debug/
  ServerGenerationLog.swift             610 LOC  — live activity dashboard

Request orchestration
  CompletionHandler.swift              1224 LOC  — /v1/chat/completions handler
  MessageConverter.swift                451 LOC  — OpenAI ↔ LLMMessage
  ToolCallConverter.swift                61 LOC  — internal ToolCall ↔ OpenAI.ToolCall
  HTTPPrefixCacheSpike.swift            411 LOC  — diagnostics structures
  HTTPPrefixCacheSessionReplay.swift    193 LOC  — reasoning recovery store
  Models/OpenAITypes.swift              ~300 LOC — Codable wire shapes
  Models/ServerInferenceTypes.swift     ~167 LOC — request / response envelopes

Inference routing
  ServerInferenceService.swift          101 LOC  — engine wrapper
  InternalInferenceRouting.swift        116 LOC  — internal Agent path
  ModelFingerprint.swift                222 LOC  — SHA-256 partition keying

Inference nucleus
  ../Agent/InferenceArbiter.swift       417 LOC  — single-flight FIFO actor
  ../Agent/AgentEngine.swift            ~LOC      — MainActor wrapper
  ../Agent/LLMActor.swift              ~few k    — generateServerTextCompletion = 780 LOC
  ../Agent/AgentGeneration.swift         121 LOC — sampling parameter struct
  ../Agent/AgentTokenizer.swift          ~LOC    — chat template + special tokens
  ../Agent/ThinkingRepetitionDetector.swift 352 LOC
  ../Agent/ThinkingSafeguardObserver.swift   76 LOC
  ../Agent/TriAttentionRuntimeSelection.swift 139 LOC

Prefix cache
  PrefixCacheManager.swift             1199 LOC  — top-level manager + planner + drain
  TokenRadixTree.swift                  598 LOC  — compressed-edge radix nodes
  StablePrefixDetector.swift            101 LOC  — two-probe technique
  EvictionPolicy.swift                  233 LOC  — Marconi utility scoring
  AlphaTuner.swift                      314 LOC  — bootstrap + 21-candidate grid
  PrefixCacheDiagnostics.swift          680 LOC  — 5 typed event surfaces

Snapshot tiering (RAM + SSD)
  SnapshotStore.swift                    82 LOC  — protocol + RAM tier
  TieredSnapshotStore.swift             308 LOC  — RAM ⇄ SSD lifecycle composition
  SSDSnapshotStore.swift               1943 LOC  — front door, writer, manifest, recovery
  SSDPrefixCacheConfig.swift             58 LOC  — sizing + toggles
  SnapshotManifest.swift                524 LOC  — schema v6, descriptors, partition meta

Vendor fork (Vendor/mlx-swift-lm/Libraries/MLXLMCommon)
  HybridCacheSnapshot.swift             ~900 LOC — capture/restore (Mamba+attn+quant+TriAtt)
  TriAttentionRuntime.swift             ~650 LOC — scoring kernel
  TriAttentionSparseKVCache.swift       ~500 LOC — sparse cache impl
  QuantizedTriAttentionSparseKVCache.swift ~400 LOC
  TriAttentionCalibrationArtifact.swift ~920 LOC — pickle decoder
  TriAttentionConfiguration.swift       ~110 LOC — policy struct
  ParoQuant/ParoQuantLoader.swift       ~500 LOC — AWQ unpack + Mamba split
  ParoQuant/RotateQuantizedLinear.swift ~200 LOC — Givens rotation Metal kernel
  Tool/ToolCallProcessor.swift           ~LOC    — XML parser routing
  Evaluate.swift                        ~ TokenIterator with checkpoint params

Vendor fork (Vendor/mlx-swift-lm/Libraries/MLXLLM/Models)
  Qwen35.swift                          ~800 LOC — heterogeneous Mamba+attn cache (Tesseract-modified)

Tests under tesseractTests/
  HTTPPrefixCacheSpikeTests
  HTTPPrefixCacheSessionReplayTests
  CompletionHandlerTests
  MessageConverterTests
  OpenAITypesTests
  HybridCacheSnapshotTests
  TokenRadixTreeTests
  StablePrefixDetectorTests
  StablePrefixDetectorNonDeterminismTests
  PrefixCacheManagerTests
  PrefixCacheIntegrationTests
  CheckpointCaptureTests
  JinjaNonDeterminismReproTests

Loaded-model verification (scripts/dev.sh)
  prefix-cache-e2e         Task 1.8 — TTFT / output equivalence proxy
  hybrid-cache-correctness Task 2.2 — bitwise logit + state equivalence
```

---

## 13. References

**Tesseract internal docs**
- `docs/HTTP_SERVER_SPEC.md`
- `docs/marconi-hybrid-prefix-cache-implementation-plan.md`
- `docs/mlx-swift-lm-kv-cache-audit.md`
- `docs/mlx-swift-lm-pr-164-review-response.md`
- `docs/mlx-swift-lm-prefill-memory-research.md`
- `docs/triattention-upstream-deviations.md`
- `docs/prefix-cache-log-analysis-2026-04-17.md`
- `docs/prefix-cache-session-2026-04-13-investigation.md`

**Apple**
- WWDC 2018 #715 — *Introducing Network.framework: A modern alternative to Sockets*
- WWDC 2021 #10132 / 10133 — *Meet async/await* / *Protect mutable state with Swift actors*
- WWDC 2023 #10170 — *Beyond the basics of structured concurrency*
- WWDC 2024 #10169 — *Migrate your app to Swift 6*
- WWDC 2025 #298 — *Explore large language models on Apple silicon with MLX*
- WWDC 2025 #315 — *Get started with MLX for Apple silicon*
- developer.apple.com/documentation/network — Network.framework
- developer.apple.com/documentation/foundation/jsonencoder
- developer.apple.com/documentation/os/oslogprivacy
- developer.apple.com/documentation/swift/checkedcontinuation
- SE-0414 (region-based isolation), SE-0406 (AsyncStream backpressure)

**Industry**
- vLLM — github.com/vllm-project/vllm, docs.vllm.ai
- llama.cpp — github.com/ggml-org/llama.cpp (KV-cache reuse discussions #13606, #20574)
- SGLang — github.com/sgl-project/sglang, sgl-project.github.io, *RadixAttention* (LMSYS blog 2024-01-17)
- TensorRT-LLM — github.com/NVIDIA/TensorRT-LLM
- HuggingFace TGI — github.com/huggingface/text-generation-inference (paged-attention, FP8-KV PR #2028)
- Ollama — docs.ollama.com, ollama.com/blog/openai-compatibility
- mlx-lm — github.com/ml-explore/mlx-lm
- Anthropic — platform.claude.com/docs/en/build-with-claude/prompt-caching, /agents-and-tools/tool-use
- Marconi paper — assets.amazon.science/96/d4/.../marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf
- Marconi reference repo — github.com/ruipeterpan/marconi
- LM Studio — lmstudio.ai/docs

---

*End of report. The codebase audit was generated from direct file reads (tesseract/Features/Server/, tesseract/Features/Agent/, Vendor/mlx-swift-lm/) supplemented by docs/ design plans. Industry comparisons sourced from authoritative project docs and recent (2025-2026) blog posts from each project. Apple guidance sourced from WWDC sessions and developer.apple.com.*
