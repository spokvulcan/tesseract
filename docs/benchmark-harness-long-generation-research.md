# Benchmark Harness vs. Long-Running Generations — Research & Next Steps

**Date:** 2026-04-14
**Trigger:** BugFind-15 reports `"Model request timed out, retrying (2/3)"` for BF-08
("Rust Integer Overflow in Release") and BF-09 ("Go Slice Aliasing"), even though the
Tesseract server is operationally healthy and the model is generating tokens at a
normal ~57 events/s rate. Forensic dumps from the M3 wall-clock watchdog confirm the
9B-paro model is doing legitimate (if very long) reasoning inside a `<think>` block
rather than deadlocking — see `tmp/tesseract-debug/hang-dumps/*-hang.json` for
~8 KB of verbatim repetition-loop reasoning content.

The user's stance: **a few minutes of think-time is acceptable.** The bug is in the
detection / contract between server and harness, not in the model. This document
collects evidence on both sides and proposes concrete next steps.

---

## TL;DR

1. **The harness's timeout is 30 seconds, hardcoded in the BugFind-15 Bench Pack** —
   not in BenchLocal core. There are three ways to override it without touching any
   code:
   - **UI:** the BenchLocal workspace tab already exposes a `Request Timeout Seconds`
     input under sampling overrides (`App.tsx:274`). Set it to e.g. `600`.
   - **Env var:** export `MODEL_REQUEST_TIMEOUT_SECONDS=600` before launching
     BenchLocal.
   - **Per-request:** any caller passing `params.request_timeout_seconds` to the
     benchpack overrides the default.
2. **Industry standard for long inference is streaming or polling, not "keep the
   non-streaming connection alive somehow."** Every OpenAI-compatible server we
   surveyed (vLLM, llama.cpp, Ollama, LMStudio) is silent on the wire during a
   non-streaming request. Anthropic's SDKs go further and *refuse* non-streaming for
   anything that might exceed 10 minutes. OpenAI's answer for o1/o3-class reasoning
   is the **Responses API background mode** — POST returns immediately, client polls
   `GET /v1/responses/{id}`.
3. **There is no HTTP-layer mechanism to send "work in progress" on a non-streaming
   `application/json` response.** The response either streams (SSE,
   `text/event-stream`) or it doesn't (a single buffered JSON body). You cannot mix
   the two without breaking standards-compliant clients.
4. **Recommended path:** the right immediate fix is harness-side (raise the BugFind
   timeout to 5–10 minutes). The right long-term fix is for clients that need long
   generation to **use `stream=true`** — Tesseract already implements SSE correctly
   on the streaming path. A "background mode" on the server is a possible future
   enhancement but not required for this bug.

---

## Part 1 — How BugFind-15 Decides A Request Is "Stuck"

### Architecture: it's the Bench Pack, not BenchLocal

BenchLocal core is intentionally streaming/timeout-agnostic. It defines a
`request_timeout_seconds?` field in the `GenerationRequest` protocol
(`packages/benchlocal-core/src/protocol.ts:217`,
`packages/benchpack-host/src/index.ts:1171`) and delegates *all* HTTP, retry, and
"stuck" logic to the Bench Pack's `runScenario()` implementation. Bench Packs are
installed as separate npm-published packages under
`~/.benchlocal/benchpacks/<pack>/versions/<version>/`.

### The actual timeout: BugFind-15 hardcodes 30 seconds

The BugFind-15 pack we're running is at
`~/.benchlocal/benchpacks/bugfind-15/versions/1.0.0-7d104110/`.

**File:** `lib/llm-client.ts:22`

```ts
const DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30;
```

**File:** `lib/llm-client.ts:77-98`

```ts
function resolveRequestTimeoutMs(params?: GenerationParams): number {
  if (params?.request_timeout_seconds !== undefined) {
    const requested = Math.trunc(params.request_timeout_seconds);
    if (Number.isFinite(requested) && requested > 0) {
      return requested * 1000;
    }
  }

  const rawTimeout = process.env.MODEL_REQUEST_TIMEOUT_SECONDS?.trim();

  if (!rawTimeout) {
    return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
  }

  const parsed = Number.parseInt(rawTimeout, 10);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
  }

  return parsed * 1000;
}
```

**Resolution order:**
1. `params.request_timeout_seconds` (per-request, set by host workspace settings)
2. `process.env.MODEL_REQUEST_TIMEOUT_SECONDS` (env var)
3. `DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30` (the literal that's biting us)

### The HTTP call: native fetch, non-streaming, AbortController

**File:** `lib/llm-client.ts:100-167`

```ts
export async function callModel(model: ModelConfig, messages: ModelMessage[], params?: GenerationParams) {
  const baseUrl = normalizeBaseUrl(model.baseUrl);
  const requestTimeoutMs = resolveRequestTimeoutMs(params);
  // ...
  const body: Record<string, unknown> = {
    model: model.model,
    messages
  };
  // temperature / top_p / top_k / min_p forwarded if set
  // NO `stream` field → OpenAI defaults to false → non-streaming

  const timeoutController = new AbortController();
  const timeoutHandle = setTimeout(() => timeoutController.abort(), requestTimeoutMs);
  // ...
  const abortFromTimeout = () => abortController.abort(new DOMException("Request timed out.", "TimeoutError"));
  // ...
  response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: abortController.signal
  });
  payload = (await response.json()) as ChatResponse;
```

**Mechanism:** plain Node `fetch`, no streaming, no SSE consumer. A single
`AbortController` is wired to a `setTimeout` that fires after
`requestTimeoutMs`. When the timer fires, the request is aborted with a
`DOMException` of name `TimeoutError`, which propagates as
`Error: Request timed out after 30s.` (line 160).

There is **no idle-vs-total distinction.** The 30 s budget covers connect, request,
*and* response — start-to-finish wall clock. The harness has no concept of "the
server is producing tokens, give it more time" because in non-streaming mode there
*are* no tokens visible until the final body arrives.

### The retry / "stuck" logic: 3 attempts, 750 ms backoff

**File:** `lib/orchestrator.ts:46-155`

```ts
const TIMEOUT_RETRY_PATTERN = /request timed out|aborted due to timeout|timeouterror|aborterror/i;
const MAX_MODEL_ATTEMPTS = 3;

async function callModelWithRetry(...) {
  for (let attempt = 1; attempt <= MAX_MODEL_ATTEMPTS; attempt += 1) {
    try {
      response = await callModel(model, messages, params);
      lastError = null;
      break;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown model execution error.");

      if (!isRetryableModelError(lastError) || attempt === MAX_MODEL_ATTEMPTS) {
        throw lastError;
      }

      traceLines.push(`retry_attempt_${attempt}=${lastError.message}`);
      const isTimeout = TIMEOUT_RETRY_PATTERN.test(lastError.message);
      await emit({
        type: "model_progress",
        modelId: model.id,
        scenarioId,
        message: isTimeout
          ? `Model request timed out, retrying (${attempt + 1}/${MAX_MODEL_ATTEMPTS})`
          : `Provider returned error, retrying (${attempt + 1}/${MAX_MODEL_ATTEMPTS})`
      });
      await sleep(750 * attempt, params?.signal);
    }
  }
}
```

**Total scenario budget:** `30 s × 3 attempts + (750 + 1500) ms backoff ≈ 92 s`
before BugFind-15 marks the scenario as "Request timed out after 30s." and moves
on. Note the regex on line 47: any error message containing `request timed out`,
`aborted due to timeout`, `timeouterror`, or `aborterror` triggers retry.

**The "stuck" detection is exactly the 30-second total-wall-clock fetch timeout —
nothing more, nothing less.** There is no separate keep-alive monitor, no
"the server is silent for too long" probe, and no streaming consumer that could
distinguish "hung" from "still producing tokens but slowly."

### BenchLocal already exposes the override in the UI

**File:** `app/src/renderer/src/App.tsx:144`, `:274`

```ts
type SamplingOverrideForm = {
  // ...
  request_timeout_seconds: string;
};

const SAMPLING_FIELDS = [
  // ...
  { key: "request_timeout_seconds", label: "Request Timeout Seconds", placeholder: "Leave blank", integer: true }
];
```

So the BenchLocal app already shows a **"Request Timeout Seconds"** field as part of
each workspace tab's sampling overrides. Whatever is entered there flows through
`tab.samplingOverrides` → `params.request_timeout_seconds` → the BugFind-15
`callModel` override path. This is a **zero-code fix** for the user's immediate
problem.

---

## Part 2 — How OpenAI And The Industry Handle Long Generations

### OpenAI HTTP API: `stream=false` is silent on the wire

For a non-streaming `/v1/chat/completions` request, OpenAI's server holds the TCP
connection open and returns HTTP headers + the full JSON body only when generation
completes. **There is literal silence on the HTTP layer in between.** No
intermediate whitespace, no chunked-encoding heartbeat, no "in progress" body
fragments. The only thing keeping the socket alive is kernel TCP keepalive packets,
which are off by default on most systems.

OpenAI documents this implicitly by recommending two things for long-running
requests:
- **`stream=true`** with Server-Sent Events for normal long generation.
- **Responses API `background: true`** for o1/o3-class reasoning models that can
  take many minutes
  ([OpenAI background mode docs](https://platform.openai.com/docs/guides/background)).

### SDK defaults: 600 seconds

| SDK | Default request timeout | Retries on timeout | Source |
|---|---|---|---|
| `openai` Python | **600 s** (10 min), connect=5 s | 2 | `DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)` in `openai/_constants.py` |
| `openai` Node | **600 000 ms** (10 min) | 2 | `static DEFAULT_TIMEOUT = 600000;` in client source |
| Anthropic SDKs (Python/TS) | Streaming-required if projected duration > 10 min; SDKs error/refuse non-streaming long requests | n/a | [Anthropic streaming docs](https://platform.claude.com/docs/en/api/messages-streaming), [Anthropic errors](https://docs.anthropic.com/en/api/errors) |

**Takeaway:** OpenAI's official SDKs allow **20× more time** by default than
BugFind-15's 30 s. BugFind-15's 30 s is *unusually* short for an OpenAI-compatible
client.

### How Anthropic and OpenAI keep streaming connections alive

Both vendors use named events / SSE comments as **periodic heartbeats** on streaming
responses, so intermediate proxies and clients don't time out the idle TCP
connection during long thinks:

- **Anthropic Messages streaming**: emits explicit `event: ping\ndata: {"type":"ping"}\n\n`
  events. Clients are told to ignore them. Quote from the docs: *"Event streams may
  also include any number of `ping` events."*
- **OpenAI** uses SSE comment lines (`:\n\n`) — a colon-prefixed line is a no-op per
  the SSE spec. Cited in
  [Claude Code #45224](https://github.com/anthropics/claude-code/issues/45224) as the
  precedent.
- **Recommended cadence**: 15–30 s, chosen to beat the shortest proxy timeout in
  the chain (see "Anti-patterns" below).

### Long-running options ranked

1. **SSE with periodic heartbeats** — industry default. Used by every major
   provider. Works because the cadence resets every read-timeout in the chain.
2. **Create-then-poll ("background mode")** — the only pattern that's actually
   robust to >10 minutes. POST returns a job id, client GETs `/v1/responses/{id}`
   in a loop. OpenAI's solution for o3-deep-research; Anthropic's `Message Batches
   API` is the equivalent.
3. **Non-streaming with a long client timeout** — what BugFind-15 is doing now,
   only worse because the timeout is 30 s instead of 600 s. Works for short
   generations, breaks for anything else.

### Other OpenAI-compatible servers

| Server | Non-streaming behavior on long requests | Notes |
|---|---|---|
| **vLLM** | Silence until completion. No heartbeat. [vLLM #19268](https://github.com/vllm-project/vllm/issues/19268) (feature request for non-streaming heartbeat) was closed "not planned" in Oct 2025. vLLM also can't detect non-streaming client disconnects. |
| **llama.cpp `server`** | `Transfer-Encoding: chunked` + `Keep-Alive: timeout=5, max=100`, but emits one chunk at the end. No app-layer heartbeat. |
| **Ollama** | Known to hang indefinitely on long non-streaming requests. [Ollama #15258](https://github.com/ollama/ollama/issues/15258) reports `/v1/chat/completions` hanging forever on Apple Silicon. Community guidance is "use streaming." |
| **LMStudio** | OpenAI-compatible wrapper around llama.cpp. Same silence pattern. |
| **Anthropic Messages API** | Hard 10-minute ceiling on non-streaming. SDKs refuse longer requests. |

**No OpenAI-compatible server sends an application-layer keep-alive on a
`stream=false` request.** Tesseract is in good company being silent — it would be
*unusual* for it to behave otherwise.

### Anti-patterns and why proxy timeouts kill long requests

Even if the model server is silent and patient, somewhere in the chain a proxy will
typically close an idle connection. Defaults:

| Component | Default idle timeout |
|---|---|
| nginx `proxy_read_timeout` | **60 s** |
| AWS ALB idle timeout | **60 s** |
| Cloudflare Free/Pro HTTP idle | **100 s** (not configurable below Enterprise) |
| HAProxy default | **60 s** |
| Kubernetes ingress-nginx | **60 s** typically |
| Azure OpenAI (deployment-side ceiling) | **600 s** |

Tesseract serves on `127.0.0.1:8321` without a proxy chain, so this class of
problems doesn't apply directly — but any future deployment over a tunnel/proxy
will hit it instantly. The mitigation is the same in both cases: **streaming with
periodic heartbeats**.

### Sources

- [openai-python `_constants.py`](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py) — 600 s default
- [OpenAI background mode guide](https://platform.openai.com/docs/guides/background) — create-then-poll pattern
- [Anthropic Messages streaming docs](https://platform.claude.com/docs/en/api/messages-streaming) — `ping` events
- [vLLM #19268](https://github.com/vllm-project/vllm/issues/19268) — non-streaming heartbeat feature request, closed
- [Ollama #15258](https://github.com/ollama/ollama/issues/15258) — non-streaming hang on Apple Silicon
- [Claude Code #45224](https://github.com/anthropics/claude-code/issues/45224) — SSE keep-alive precedent
- [Cherry Studio #11965](https://github.com/CherryHQ/cherry-studio/issues/11965) — configurable SSE timeout request
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [BenchLocal README](file:///Users/owl/projects/BenchLocal/README.md)
- BugFind-15 source at `~/.benchlocal/benchpacks/bugfind-15/versions/1.0.0-7d104110/lib/{llm-client.ts,orchestrator.ts}`

---

## Part 3 — Why You Cannot "Send Work-In-Progress" On A Non-Streaming Response

This is the question the user asked: *"maybe the server should respond at least
something like work in progress so the harness sees that work is in progress and
we did not stuck."*

The answer is: **HTTP doesn't have a mechanism for that on a `application/json`
response.** The standards say:

1. **You commit to a content type at header time.** The server sends `Content-Type:
   application/json` (or `text/event-stream`) before the body starts. Once that
   header is on the wire, the client knows how to parse the body and will fail
   loudly if the body shape doesn't match.
2. **A JSON body must be a single complete document.** The client calls
   `await response.json()` (BugFind-15's `lib/llm-client.ts:153`). That
   buffers the *entire* body until EOF, then parses. If you send anything before
   the final JSON object — whitespace, an "in progress" sentinel, a partial
   object — `JSON.parse` throws.
3. **Chunked transfer encoding doesn't help.** `Transfer-Encoding: chunked` lets
   the server send the body in pieces, but each piece is still part of the same
   single JSON document. You can't slip non-JSON heartbeats between chunks; they're
   concatenated and parsed together.
4. **You cannot upgrade a non-streaming response to SSE mid-flight.** Once you've
   committed to `application/json`, you can't change the content-type. SSE requires
   `Content-Type: text/event-stream` from the very first header.

**The two legal shapes are:**
- **`stream=false`** → `Content-Type: application/json` → silent until the full
  body, then deliver one JSON object. **No heartbeats possible.**
- **`stream=true`** → `Content-Type: text/event-stream` → SSE frames as soon as
  available, with `:keepalive\n\n` comments to fill silence. **Heartbeats are how
  you keep this alive.**

If a client wants progress signals, the *contract* is "pass `stream=true`."
There is no escape hatch.

The only other option — sometimes called "long polling" or "background mode" — is
to *split* the operation: POST starts the job and immediately returns a job id;
GET polls for the result. That's what OpenAI's Responses API `background: true`
does. It's a *different endpoint*, not a sneaky modification of the existing one.

---

## Part 4 — Suggested Next Steps

The fixes split cleanly into three buckets. **The harness fix is mandatory; the
server changes are optional and only help if Tesseract grows new client types.**

### Bucket A — Harness fixes (do these now)

#### A1. Set `MODEL_REQUEST_TIMEOUT_SECONDS=600` for BenchLocal (zero code)

Launch BenchLocal with the env var set. On macOS, the simplest way is to start
the app from a terminal:

```bash
MODEL_REQUEST_TIMEOUT_SECONDS=600 open -a "BenchLocal"
```

This raises the default for *every* benchpack that respects the env var (currently
just BugFind-15, but the convention is shared). The 600 s value matches OpenAI's
own SDK default and gives Tesseract enough headroom for any sane reasoning length
short of o1-style 15-minute thinks.

**Verify:** the next BugFind-15 run on BF-08 should let the model run until either
it produces the `<solution>` tag or the wall-clock watchdog (currently 60 s
server-side) fires. Either way you'll get a result, not a 30 s retry storm.

#### A2. Set the workspace tab's "Request Timeout Seconds" override (UI)

The BenchLocal app already shows this field — `App.tsx:274`. Open the BugFind-15
tab in BenchLocal, find the sampling-overrides section, and set
**Request Timeout Seconds = 600** (or higher). This overrides the env var and the
hardcoded default per-workspace, so it's the cleanest persistent fix.

**This is the recommended primary fix.** It needs no code changes anywhere, lives
in the harness UI exactly where it belongs, and survives benchpack updates.

#### A3. Server-side wall-clock watchdog limit needs to be at least as high as the harness timeout

Tesseract currently caps generation at `generationWallClockLimitSeconds = 60`
(`tesseract/Features/Server/CompletionHandler.swift:24`). If the harness timeout
is 600 s, the server should also be willing to run that long. **Raise the server
wall-clock to match** — say `300 s` (5 min) as a first cut, or `600 s` to fully
match the OpenAI SDK convention.

**File:** `tesseract/Features/Server/CompletionHandler.swift:24-ish`

```swift
static let generationWallClockLimitSeconds: UInt64 = 300  // was 60
```

The silence watchdog (`generationSilenceTimeoutSeconds = 60`) should *not* be
raised — its job is to catch true MLX/Metal deadlocks where no events flow at all,
and 60 s of complete silence on a healthy generation is still clearly broken.

### Bucket B — Server-side improvements (consider, not required)

These improve robustness for *future* clients but aren't needed to fix the
current BugFind-15 issue.

#### B1. Document the streaming contract

Add a one-paragraph note to `docs/HTTP_SERVER_SPEC.md` saying:

> Clients that expect generations to take longer than 30 seconds should pass
> `stream: true`. Non-streaming requests are subject to the server's wall-clock
> generation watchdog (`generationWallClockLimitSeconds`, currently 300 s) and
> will return `503 Service Unavailable` if exceeded. There is no application-layer
> keep-alive on non-streaming responses; this matches OpenAI, vLLM, llama.cpp,
> Ollama, and Anthropic Messages.

This sets expectations for any future client integration.

#### B2. Always stream internally, buffer for non-streaming clients

Tesseract already does this via `AgentEngine.generateServerTextCompletion`, but
the implementation is split between the streaming and non-streaming code paths.
Consolidating to a single internal pipeline that *always* streams from `LLMActor`
and only differs at the response-encoding step would simplify the watchdog story
and eliminate any path where `stream=false` could regress to a synchronous
generation call.

**Status:** mostly already done; would be a small refactor.

#### B3. Add SSE keep-alive to the streaming path

When `stream=true` and the model is in a long think block, the server should emit
periodic SSE heartbeat frames so intermediate proxies (when Tesseract is eventually
deployed behind one) and slow clients don't time out. Two acceptable forms:

- **SSE comments** (`: heartbeat\n\n`) — invisible to standards-compliant clients.
- **OpenAI-style empty deltas** (`data: {"choices":[{"delta":{}}]}\n\n`) — visible
  but harmless.

The `streamGenerationEvents` helper at
`tesseract/Features/Server/CompletionHandler.swift:573-ish` is the right place to
add this. It already runs inside a task group with the keepalive task — wire the
heartbeat there.

#### B4. Add a Responses-API-style background mode (large)

For genuinely long reasoning (10+ minutes), the right pattern is create-then-poll.
The InferenceArbiter already models a single in-flight request, so a job table
keyed by response id is a small addition:

- `POST /v1/responses` with `background: true` → returns
  `{"id": "resp_xxx", "status": "in_progress"}` immediately, kicks off a detached
  generation task.
- `GET /v1/responses/{id}` → returns current status; final completed body once
  done.

This is the only pattern that's actually robust to multi-minute thinks without
relying on client SDK timeouts. **It's a future enhancement, not a fix for this
bug.**

### Bucket C — Bench Pack improvements (file upstream)

The BugFind-15 30-second default is genuinely *too short* for any local-model
benchmark. This is worth filing as an upstream change request to the benchpack
authors:

#### C1. Bump `DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS` to match OpenAI SDK

**Suggested upstream PR:** change
`~/.benchlocal/benchpacks/bugfind-15/versions/<v>/lib/llm-client.ts:22` from `30`
to `600`, matching `openai-python`/`openai-node`. The env var override and per-tab
override semantics stay unchanged.

Rationale to put in the PR: every other OpenAI-compatible client uses 600 s as
the default, BugFind-15 benchmarks specifically include "Hard" scenarios where
reasoning models can plausibly take 1–5 minutes, and the current 30 s default
turns those scenarios into infrastructure failures rather than capability
failures.

#### C2. Switch the BugFind-15 client to streaming consumption

A bigger upstream PR: change `lib/llm-client.ts:callModel` to use `stream: true`
and consume SSE chunks. Pros: gets the harness "still alive" signal for free, no
matter how long the model thinks; future-proofs against proxy timeouts. Cons:
needs a Server-Sent Events parser, more state to manage. Worth doing as a follow-up
once C1 unblocks the immediate problem.

---

## Recommended sequence

1. **Right now (zero code):** Set the workspace tab's "Request Timeout Seconds" to
   600 in BenchLocal, then re-run BugFind-15 (Bucket A2).
2. **Same session (one-line code change):** Bump
   `generationWallClockLimitSeconds` in `CompletionHandler.swift` from 60 to 300
   (Bucket A3) and rebuild. This avoids the server-side watchdog firing during
   legitimate long thinks now that the client is willing to wait.
3. **Verify:** run the full BugFind-15 benchmark. BF-08 and BF-09 may still get
   `score=0` for content reasons (the 9B model can't solve them), but the
   *infrastructure* path should be clean — no `Model request timed out` events,
   every request gets a response, every lease gets released.
4. **Soon after:** add the streaming-contract doc note (B1) and file the upstream
   benchpack PR for the 600 s default (C1).
5. **Eventually:** SSE keep-alive on the streaming path (B3) and possibly
   background mode (B4) once a real user case appears.

---

## Appendix — Verification commands

After applying A1/A2/A3:

```bash
# Confirm the BenchLocal env var is set (if you used A1)
echo $MODEL_REQUEST_TIMEOUT_SECONDS  # should print 600

# Confirm BugFind-15's resolved timeout
grep -A 3 'DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS' \
  ~/.benchlocal/benchpacks/bugfind-15/versions/*/lib/llm-client.ts

# Confirm Tesseract's server-side limit
grep generationWallClockLimitSeconds \
  /Users/owl/projects/tesseract/tesseract/Features/Server/CompletionHandler.swift
```

After running the benchmark:

```bash
# Look for any "timed out, retrying" events in the harness output (should be 0)
grep -c "Model request timed out" <benchmark-output.log>

# Look for any wall-clock watchdog fires (should be 0 if the cap is high enough)
grep "exceeded wall-clock" /tmp/tesseract-bench-*.log

# Confirm every lease was released
grep -c "lease acquired" /tmp/tesseract-bench-*.log
grep -c "lease released" /tmp/tesseract-bench-*.log
# These two counts should match
```

If both watchdog fires are zero and the lease-acquired / lease-released counts
match, the harness↔server contract is healthy and any remaining failed scenarios
are content/model issues — not infrastructure.
