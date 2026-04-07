# HTTP Inference Server Specification

Tesseract Agent local HTTP server — serves the on-device LLM via an OpenAI-compatible API for coding agents (OpenCode/Crush, Aider, Continue) and internal use by the Agent chat and cron agents.

**Status**: Ready for Development
**Date**: 2026-04-07

---

## 1. Goals

1. **Serve local LLM to coding agents** — OpenAI-compatible HTTP API on `127.0.0.1` so external tools can use Tesseract's loaded model for inference.
2. **Advanced prompt caching** — Block-based paged KV cache with chain-hash prefix matching, segment-aware priority eviction, and SSD cold tier. Shared between HTTP clients and the internal Agent chat.
3. **Concurrent inference** — Multiple coding agents (configurable, default 4) plus the internal Agent chat, all generating concurrently via interleaved token-level scheduling.
4. **Statistics dashboard** — Dedicated app page showing server status, cache hit rates, active requests, token throughput, and per-client usage for both HTTP and Agent chat.

## 2. Non-Goals (for initial release)

- Authentication / API keys (local-only, `127.0.0.1`)
- Model switching via API (serves whatever is loaded)
- `/v1/completions` (text completions) — chat only
- `/v1/embeddings`
- Remote/non-localhost access
- True batched inference with `[B, H, L, D]` KV caches (phase 2)
- Anthropic `/v1/messages` format (phase 2)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tesseract Agent App                             │
├─────────────────────────────────────────────────────────────────────────┤
│  UI Layer                                                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ Agent Chat   │  │ Server Stats     │  │  Settings                 │  │
│  │ (existing)   │  │ Page (new)       │  │  (server toggle, port)    │  │
│  └──────┬───────┘  └──────────────────┘  └───────────────────────────┘  │
├─────────┼───────────────────────────────────────────────────────────────┤
│  Server │Layer                                                          │
│  ┌──────┴───────┐  ┌──────────────────────────────────────────────────┐  │
│  │ Agent        │  │  HTTPServer (@Observable, @MainActor)            │  │
│  │ Coordinator  │  │  ┌────────────────────────────────────────────┐  │  │
│  │              │  │  │  NWListener (port 8321)                    │  │  │
│  │              │  │  │  Route: POST /v1/chat/completions          │  │  │
│  │              │  │  │  Route: GET  /v1/models                    │  │  │
│  │              │  │  │  Route: GET  /health                       │  │  │
│  │              │  │  └────────────────────────────────────────────┘  │  │
│  └──────┬───────┘  └──────────┬───────────────────────────────────────┘  │
│         │                     │                                          │
├─────────┼─────────────────────┼──────────────────────────────────────────┤
│  Inference Scheduler                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  InferenceScheduler (actor)                                       │   │
│  │  - Manages multiple TokenIterators (1 per active request)         │   │
│  │  - Round-robin interleaved token generation                       │   │
│  │  - Configurable concurrency: max_concurrent_sequences (default 4) │   │
│  │  - Interfaces with CacheManager for KV cache allocation           │   │
│  └───────────────────────────────┬───────────────────────────────────┘   │
│                                  │                                       │
├──────────────────────────────────┼───────────────────────────────────────┤
│  Cache Layer                     │                                       │
│  ┌───────────────────────────────┴───────────────────────────────────┐   │
│  │  CacheManager (actor)                                             │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │   │
│  │  │  PagedKVCache   │  │  PrefixHashIndex  │  │  SSDCacheTier   │  │   │
│  │  │  (hot blocks)   │  │  (chain hashing)  │  │  (cold blocks)  │  │   │
│  │  └─────────────────┘  └──────────────────┘  └─────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
├──────────────────────────────────┼───────────────────────────────────────┤
│  Model Layer (existing)          │                                       │
│  ┌───────────────────────────────┴───────────────────────────────────┐   │
│  │  LLMActor  →  ModelContainer  →  TokenIterator (per sequence)     │   │
│  │  (model weights are immutable, safe for concurrent read)          │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. HTTP Server

### 4.1 Transport

- **Framework**: `Network.framework` (`NWListener` + `NWConnection`) — no external dependencies.
- **Entitlement**: Requires `com.apple.security.network.server` in **both** `tesseract.entitlements` (Debug) and `tesseractRelease.entitlements` (Release). The existing `com.apple.security.network.client` is not sufficient for listening sockets, even on loopback. This must be added as part of Phase 1.
- **Default port**: `8321` (configurable in Settings).
- **Bind address**: `127.0.0.1` only. No `0.0.0.0`.
- **CORS**: No CORS headers. Desktop and CLI coding agents (OpenCode, Aider, Continue) do not make browser-origin requests, so permissive CORS is unnecessary. Omitting `Access-Control-Allow-Origin` prevents arbitrary websites from reaching the unauthenticated API via the user's browser. If browser-based clients are needed in the future, add an opt-in setting with a restrictive origin allowlist.
- **Toggle**: Settings switch "Enable API Server". Off by default.

### 4.2 Endpoints

#### `GET /health`

```json
{ "status": "ok" }
```

#### `GET /v1/models`

Returns the currently loaded model. Required by OpenCode for local provider discovery.

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-9b-paro",
      "object": "model",
      "type": "llm",
      "owned_by": "tesseract",
      "max_context_length": 131072,
      "loaded_context_length": 131072,
      "state": "loaded"
    }
  ]
}
```

Fields:
- `id` — model directory name or configuration name from Tesseract
- `max_context_length` — model's maximum context window
- `loaded_context_length` — effective context window (may differ if user configured a limit)
- `state: "loaded"` — signals to OpenCode that this is the active model

If no model is loaded, return an empty `data` array.

#### `POST /v1/chat/completions`

Primary inference endpoint. Both streaming and non-streaming.

**Request body:**

```json
{
  "model": "qwen3.5-9b-paro",
  "messages": [
    { "role": "system", "content": "You are a coding assistant." },
    { "role": "user", "content": "Read main.swift" },
    {
      "role": "assistant",
      "content": "I'll read that file for you.",
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "read",
            "arguments": "{\"path\": \"main.swift\"}"
          }
        }
      ]
    },
    { "role": "tool", "tool_call_id": "call_abc123", "content": "file contents..." },
    { "role": "user", "content": [
        { "type": "text", "text": "What about this image?" },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute shell commands",
        "parameters": {
          "type": "object",
          "properties": {
            "command": { "type": "string" }
          },
          "required": ["command"]
        }
      }
    }
  ],
  "stream": true,
  "max_tokens": 4096,
  "max_completion_tokens": 4096,
  "temperature": 0.6,
  "top_p": 0.95,
  "reasoning_effort": "medium",
  "stream_options": { "include_usage": true }
}
```

**Parameter handling:**

| Parameter | Behavior |
|---|---|
| `model` | Ignored (serves loaded model), but validated for response echo |
| `messages` | Required. Supports `system`, `user`, `assistant`, `tool` roles |
| `tools` | Optional. Converted to Qwen3.5 XML tool format in chat template |
| `stream` | Default `false`. When `true`, SSE response |
| `max_tokens` | Maps to `AgentGenerateParameters.maxTokens` |
| `max_completion_tokens` | Same as `max_tokens` (OpenCode sends this for "reasoning" models) |
| `temperature` | Maps to `AgentGenerateParameters.temperature` |
| `top_p` | Maps to `AgentGenerateParameters.topP` |
| `reasoning_effort` | Accepted but ignored for now (Qwen3.5 uses fixed thinking behavior) |
| `stream_options.include_usage` | When `true`, send usage in final SSE chunk |
| `stop` | Optional stop sequences |
| Unknown params | Silently ignored |

**Non-streaming response:**

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "model": "qwen3.5-9b-paro",
  "created": 1712345678,
  "system_fingerprint": "tesseract-1.0-mlx",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Here is the file content...",
        "tool_calls": [
          {
            "id": "call_<uuid>",
            "type": "function",
            "function": {
              "name": "bash",
              "arguments": "{\"command\": \"ls\"}"
            }
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 42,
    "total_tokens": 192,
    "prompt_tokens_details": {
      "cached_tokens": 120
    }
  }
}
```

**Streaming response (SSE):**

Content chunks:
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

```

Tool call chunks (buffered, not per-token):
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_<uuid>","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\": \"ls\"}"}}]},"finish_reason":null}]}

```

Final chunk with finish reason:
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"qwen3.5-9b-paro","created":1712345678,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":150,"completion_tokens":42,"total_tokens":192,"prompt_tokens_details":{"cached_tokens":120}}}

data: [DONE]

```

Keepalive during prefill (SSE comment, ignored by clients):
```
: keepalive 1024/4096

```

**`finish_reason` values:**
- `"stop"` — natural stop or stop sequence hit
- `"length"` — hit `max_tokens` / `max_completion_tokens`
- `"tool_calls"` — model produced tool calls

### 4.3 Tool Calling

The server handles tool calling end-to-end:

1. **Client sends `tools`** in the request — standard OpenAI function-calling format.
2. **Server converts to Qwen3.5 chat template format** — tools are injected into the system prompt via `apply_chat_template` (Jinja). Qwen3.5 uses XML function calling: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`.
3. **Server parses model output** — the existing `ToolCallParser` detects `<tool_call>` tags in the streaming output.
4. **Server converts back to OpenAI format** — parsed tool calls become `tool_calls` array in the response with `id`, `type: "function"`, `function.name`, `function.arguments` (JSON string).
5. **Client sends tool results** — as `role: "tool"` messages with `tool_call_id`.
6. **Server converts tool results for next turn** — maps `tool` role messages back to the format expected by the chat template.

Tool call IDs: Generated by the server as `call_<uuid>`. Mapped back when the client sends tool results.

### 4.4 Message Conversion

OpenAI message roles map to the internal `LLMMessage` types:

| OpenAI Role | Internal Type | Notes |
|---|---|---|
| `system` | System prompt parameter | Extracted from messages, passed as system prompt |
| `user` | `.user(content)` | Text or multipart (text + images) |
| `assistant` | `.assistant(content, toolCalls?)` | Content may include thinking blocks |
| `tool` | `.toolResult(id, content)` | Matched by `tool_call_id` |

Image handling: `image_url` content parts with `data:` URIs are decoded and passed as `ImageAttachment` to the model.

### 4.5 Thinking/Reasoning Blocks

Qwen3.5 models produce `<think>...</think>` blocks. The server handles these transparently:

- **In streaming**: Thinking tokens are streamed as regular `content` delta. The `<think>` and `</think>` tags appear in the content stream. OpenCode treats all content as displayable text.
- **In non-streaming**: Thinking content is included in `message.content`. Optionally, if the model produced thinking, it can be split into a separate `message.reasoning` field (this is what mlx_lm.server does, and OpenCode reads it).

---

## 5. Prompt Cache System

### 5.1 Design: Block-Based Paged KV Cache with Chain Hashing

The cache system is the core performance differentiator. It is shared between the HTTP server and internal Agent chat — any prompt prefix computed for an Agent turn can be reused by an HTTP client, and vice versa.

**Design principles:**
- Maximize cache hit rate through prefix sharing
- Minimize memory via block-level granularity and reference counting
- Two-tier hot/cold storage: RAM blocks + SSD overflow
- Actor-isolated for thread safety (Swift concurrency native)

### 5.2 Block Structure

Fixed-size blocks of **256 tokens**. Each block stores the KV cache tensors for exactly 256 tokens across all model layers.

```swift
struct CacheBlock: Sendable {
    let blockID: Int
    let layerCaches: [(key: MLXArray, value: MLXArray)]  // one pair per layer

    var refCount: Int          // number of active sequences referencing this block
    var blockHash: Data        // SHA-256 chain hash
    var segmentType: SegmentType  // .system, .user, .assistant
    var lastAccess: ContinuousClock.Instant

    // Doubly-linked list pointers for free queue
    var prevFree: Int?
    var nextFree: Int?
}

enum SegmentType: Int, Comparable {
    case assistant = 0   // lowest priority (evicted first)
    case user = 1
    case system = 2      // highest priority (evicted last)
}
```

**Memory per block** (Qwen3.5-9B, bfloat16, 40 layers, 8 KV heads, 128 head_dim):
- Per token: `2 directions * 40 layers * 8 heads * 128 dim * 2 bytes = 163,840 bytes ≈ 160 KB`
- Per block (256 tokens): `256 * 160 KB ≈ 40 MB`
- With KV quantization (8-bit): `≈ 20 MB per block`
- With KV quantization (4-bit): `≈ 10 MB per block`

### 5.3 Chain Hashing for Prefix Matching

Each block's hash depends on all previous blocks, forming a chain:

```swift
func computeBlockHash(
    parentHash: Data?,   // nil for the first block
    tokens: [Int32],     // exactly 256 tokens
    modelID: String
) -> Data {
    var hasher = SHA256()
    if let parentHash { hasher.update(data: parentHash) }
    tokens.withUnsafeBytes { hasher.update(bufferPointer: $0) }
    hasher.update(data: Data(modelID.utf8))
    return Data(hasher.finalize())
}
```

**Lookup algorithm:**

```
Input: prompt tokens [t0 ... tn]
Split into blocks: B0=[t0..t255], B1=[t256..t511], ...

h0 = hash(nil, B0, model)
h1 = hash(h0, B1, model)
h2 = hash(h1, B2, model)
...

For each hi:
  1. Check hot tier (hashIndex[hi])
  2. If miss, check SSD tier (ssdIndex[hi])
  3. If miss, STOP — this is the divergence point

Return: (matched_blocks: [block0..blockK], remaining_tokens: [t_{K*256}..tn])
```

**Complexity**: O(N/256) hash computations + O(1) dictionary lookups per block. For a 120K token prompt, that's ~469 hash lookups — trivially fast.

**Why chain hashing over a trie:**
- O(N/256) vs O(N) for trie walks
- Hash computation is vectorizable on Apple Silicon
- Dictionary lookup is O(1) amortized vs tree traversal
- No tree rebalancing or pointer chasing
- Block-level granularity matches MLX's preferred tensor alignment

### 5.4 Segment-Aware Priority LRU Eviction

Blocks are tagged with their segment type based on where they fall in the message sequence:
- **System blocks** — tokens from system prompt. These are shared across ALL conversations (coding agents typically send the same system prompt). Evicted last.
- **User blocks** — tokens from user messages and tool results. Shared across turns within a conversation. Evicted second.
- **Assistant blocks** — tokens from assistant responses. Least likely to be reused. Evicted first.

Three separate free lists (doubly-linked), one per segment type. Eviction order: assistant → user → system. Within each tier, LRU (oldest first).

### 5.5 Reference Counting and Copy-on-Write

Multiple active sequences can share blocks (e.g., two coding agents with the same system prompt):
- Each block has a `refCount`. Incremented when a sequence claims a block, decremented when the sequence completes or is evicted.
- Blocks with `refCount > 0` are pinned — never evicted.
- Blocks with `refCount == 0` go to the free list.
- **COW**: If a sequence needs to modify a shared block (shouldn't happen in normal flow since KV caches grow append-only), the block is copied first.

### 5.6 SSD Cold Tier

Blocks evicted from the hot tier are written to SSD before being freed:

- **Location**: `~/Library/Caches/Tesseract Agent/kv-cache/`
- **Format**: One `.safetensors` file per block, stored in hash-bucketed directories: `{hash[0:2]}/{hash[2:4]}/{hash}.safetensors`
- **Write path**: Extract raw bytes from MLXArray on the inference actor (required for Metal), then write to disk from a background `Task.detached` (no Metal API calls).
- **Read path**: Load safetensors from disk, reconstruct MLXArray on the inference actor.
- **Index**: In-memory `[Data: SSDBlockEntry]` dictionary rebuilt from directory scan on startup.
- **Eviction**: LRU by last access time. Configurable max size (default: 10 GB). Uses `URLResourceKey.totalFileAllocatedSizeKey` for tracking.
- **Benefit**: A common system prompt (e.g., OpenCode's ~4K token system prompt = 16 blocks) is loaded from SSD in milliseconds instead of re-computing ~1-2s of prefill.

### 5.7 Partial Block Handling

The last chunk of tokens in a prompt may not fill a complete 256-token block. This "tail" is handled specially:
- NOT stored in the block cache (wastes a full block on a partial).
- Kept as a **pending buffer** attached to the active sequence.
- When the next turn extends the conversation, the pending buffer is incorporated: if it completes a full block, that block is hashed and stored.
- This avoids wasting memory on partial blocks while still capturing all full prefixes.

### 5.8 CacheManager Actor

```swift
actor CacheManager {
    // Hot tier
    private var blocks: [CacheBlock]           // pre-allocated pool
    private var hashIndex: [Data: Int]         // chain hash -> block ID
    private var freeLists: [SegmentType: FreeBlockQueue]

    // Cold tier
    private var ssdTier: SSDCacheTier

    // Statistics
    private(set) var stats: CacheStats

    // Core operations
    func findPrefix(tokens: [Int32], modelID: String) -> PrefixMatch
    func storeBlocks(tokens: [Int32], caches: [[KVCachePair]], segmentTypes: [SegmentType]) -> [Int]
    func claimBlocks(blockIDs: [Int], sequenceID: String)
    func releaseBlocks(sequenceID: String)

    // Memory management
    func evictToFit(bytesNeeded: Int)
    func totalMemoryUsage() -> Int
}

struct PrefixMatch {
    let blocks: [Int]               // block IDs of matched prefix
    let cachedTokenCount: Int       // how many tokens were cached
    let remainingTokens: [Int32]    // tokens that need prefill
    let source: PrefixSource        // .hot, .ssd, .none
}

struct CacheStats: Sendable {
    var totalLookups: Int = 0
    var hotHits: Int = 0
    var ssdHits: Int = 0
    var misses: Int = 0
    var hitRate: Double { Double(hotHits + ssdHits) / Double(max(totalLookups, 1)) }
    var hotBlocksUsed: Int = 0
    var hotBlocksTotal: Int = 0
    var ssdBlockCount: Int = 0
    var ssdBytesUsed: Int64 = 0
}
```

### 5.9 Integration with Agent Chat

The existing Agent chat currently clears the KV cache between tool rounds (`Memory.clearCache()` in `LLMActor`). With the new cache system:

1. After each generation turn, the KV state is stored as blocks in the `CacheManager`.
2. On the next turn (after tool execution), the `CacheManager` finds the longest cached prefix — typically the entire conversation history up to the new tool result.
3. Only the new tokens (tool result + any new user message) need prefill.
4. **Expected speedup**: For a 30K token conversation, if 29K tokens are cached, prefill drops from ~3s to ~0.1s.

This replaces the current full-context re-evaluation on every turn with incremental prefill.

---

## 6. Concurrent Inference

### 6.1 Approach: Interleaved Token-Level Scheduling

MLX Swift's `ModelContainer` allows concurrent reads of model weights after prefill. Each `TokenIterator` has its own private KV cache. The inference scheduler manages multiple iterators and alternates between them.

```
Time →
Seq A: [prefill aaaa] [decode a][        ][decode a][        ][decode a] ...
Seq B:                 [        ][decode b][        ][decode b][        ] ...
                       ←── round-robin token generation ──→
```

**Why not true batched (phase 1)?**
- MLX Swift has no `BatchKVCache` with `merge/filter/extract` operations.
- Swift KV caches are `[1, H, L, D]` — no batch dimension.
- Porting the Python `BatchGenerator` requires ~2000 lines and model-level changes.
- Interleaved scheduling works TODAY with no model changes. The performance gap is modest for <=4 concurrent sequences since Apple Silicon's unified memory eliminates PCIe transfer overhead.

### 6.2 InferenceScheduler Actor

Replaces the current single-sequence `LLMActor.generate()` flow.

```swift
actor InferenceScheduler {
    private let modelContainer: ModelContainer
    private var activeSequences: [SequenceID: ActiveSequence]
    private let cacheManager: CacheManager
    private let maxConcurrent: Int  // from Settings, default 4

    struct ActiveSequence {
        let id: SequenceID
        let iterator: TokenIterator
        var continuation: AsyncStream<Generation>.Continuation
        var pendingTokens: [Int32]  // partial block buffer
        var state: SequenceState    // .prefilling, .generating, .complete
    }

    /// Submit a new generation request. Returns a stream of tokens.
    func submit(
        input: UserInput,
        parameters: GenerateParameters,
        source: RequestSource  // .http(requestID) or .agentChat
    ) async throws -> AsyncThrowingStream<Generation, Error>

    /// Cancel an active sequence.
    func cancel(sequenceID: SequenceID) async

    /// The main scheduling loop — runs continuously while sequences are active.
    private func schedulingLoop() async
}

enum RequestSource: Sendable {
    case http(requestID: String)
    case agentChat
    case cronAgent(agentID: String)
}
```

### 6.3 Scheduling Algorithm

```
loop:
  for each activeSequence in roundRobinOrder:
    if sequence.state == .prefilling:
      // Prefill runs to completion (holds model lock)
      // During prefill, send keepalive SSE comments to HTTP clients
      prefill(sequence)
      sequence.state = .generating

    if sequence.state == .generating:
      // Generate one token
      let token = sequence.iterator.next()
      sequence.continuation.yield(token)

      if token.isStop || token.isMaxLength:
        sequence.state = .complete
        // Store KV blocks in CacheManager
        storeBlocks(sequence)
        // Release resources
        cleanup(sequence)

  // Check for new submissions
  if let newSeq = pendingQueue.dequeue(), activeSequences.count < maxConcurrent:
    // Look up cached prefix
    let match = cacheManager.findPrefix(tokens: newSeq.tokens, modelID: model.id)
    // Create iterator with cached KV state + remaining tokens
    startSequence(newSeq, prefixMatch: match)
```

### 6.4 Prefill Priority

Prefill is expensive and holds the model exclusively. Policy:
- **Internal Agent chat** gets priority over HTTP requests (the user is watching).
- Among HTTP requests: FIFO ordering.
- During prefill, other sequences are paused. HTTP clients receive keepalive SSE comments (`: keepalive N/total\n\n`) so they don't time out.
- Prefill is chunked (e.g., 2048 tokens per step) so it can yield between chunks for latency-sensitive sequences.

### 6.5 Memory Budget for Concurrent Sequences

Each active sequence holds its own KV cache in memory. With KV quantization at 8-bit:

| Model | Per-Token KV | 8K Context | 32K Context |
|---|---|---|---|
| Qwen3.5-9B | ~80 KB | ~640 MB | ~2.5 GB |
| Qwen3.5-27B | ~160 KB | ~1.3 GB | ~5 GB |

**4 concurrent sequences at 8K context:**
- 9B model: 4 * 640 MB = 2.5 GB KV + 5 GB weights = 7.5 GB (fits in 20 GB)
- 27B model: 4 * 1.3 GB = 5.2 GB KV + 14 GB weights = 19.2 GB (tight fit in 32 GB)

The scheduler should monitor `os_proc_available_memory()` and refuse new sequences when memory pressure is high, returning HTTP 503 with `Retry-After`.

### 6.6 Configurable Concurrency

Setting in `SettingsManager`:

```swift
// Settings
var serverMaxConcurrentSequences: Int = 4  // 1-8, configurable in UI
```

The internal Agent chat always has a reserved slot (not counted against the HTTP limit). So `maxConcurrent = 4` means up to 4 HTTP + 1 Agent chat + N cron agents.

---

## 7. Statistics Page

A new top-level page in the app sidebar (alongside Chat, Settings).

### 7.1 Server Section

- **Status**: Running / Stopped, with toggle
- **Address**: `http://127.0.0.1:8321` (copyable)
- **Uptime**: Time since server started
- **Active connections**: Count of open HTTP connections
- **Active sequences**: Count / max (e.g., "2 / 4")

### 7.2 Inference Section

- **Model**: Name + size of loaded model
- **Total requests**: Count since server start
- **Tokens generated**: Total across all requests
- **Throughput**: Current tokens/sec (rolling average), peak tokens/sec
- **Average TTFT**: Time to first token (rolling average)
- **Queue depth**: Number of waiting requests

### 7.3 Cache Section

- **Hot tier**: Blocks used / total, memory usage
- **SSD tier**: Block count, disk usage
- **Hit rate**: Overall, and split by hot/SSD/miss
- **Cache hits saved**: Estimated prefill time saved (tokens_cached * avg_ms_per_token)
- **Top prefixes**: List of most-referenced block chains (shows what's being cached most — e.g., "OpenCode system prompt: 14 blocks, 12 hits")

### 7.4 Per-Client Section

Table of recent/active clients:

| Client | Requests | Tokens In | Tokens Out | Cache Hit % | Last Active |
|---|---|---|---|---|---|
| OpenCode (HTTP) | 47 | 125K | 18K | 87% | 2s ago |
| Agent Chat | 12 | 45K | 8K | 92% | Active |
| Cron: daily-review | 3 | 15K | 4K | 78% | 1h ago |

Clients are identified by: HTTP `User-Agent` header, or internal source type.

---

## 8. Integration Points

### 8.1 DependencyContainer

New services to wire:

```swift
// In DependencyContainer
lazy var cacheManager = CacheManager(
    hotBlockCount: dynamicBlockCount(),  // based on available memory
    ssdCachePath: appSupportURL.appending(path: "kv-cache"),
    ssdMaxBytes: 10_737_418_240  // 10 GB default
)

// NOTE: agentEngine.modelContainer is not currently public.
// Phase 2a research spike must resolve KV cache access before
// the scheduler can manage caches directly. Phase 1 routes
// through LLMActor's existing generate() interface.
lazy var inferenceScheduler = InferenceScheduler(
    llmActor: agentEngine.llmActor,  // requires exposing LLMActor
    cacheManager: cacheManager,
    maxConcurrent: settingsManager.serverMaxConcurrentSequences
)

lazy var httpServer = HTTPServer(
    inferenceScheduler: inferenceScheduler,
    agentEngine: agentEngine,
    settingsManager: settingsManager
)
```

### 8.2 AgentEngine Changes

`AgentEngine.generate()` is refactored to route through `InferenceScheduler`:

```swift
// Before: direct LLMActor call
func generate(...) -> AsyncThrowingStream<AgentGeneration, Error>

// After: through scheduler
func generate(...) -> AsyncThrowingStream<AgentGeneration, Error> {
    let stream = inferenceScheduler.submit(input: input, parameters: params, source: .agentChat)
    return stream.map { generation in
        // Convert Generation -> AgentGeneration (parse tool calls, thinking)
    }
}
```

The `ToolCallParser` remains in `AgentEngine` — the scheduler deals in raw token streams, not parsed tool calls. This keeps the HTTP server and Agent chat using the same inference path but different post-processing.

### 8.3 InferenceArbiter Redesign

The existing `InferenceArbiter` is a single-authority serializer — it grants exclusive GPU leases for the full duration of a generation run. `AgentCoordinator` holds the lease for the entire agent turn, and background agents queue behind it via `BackgroundAgentFactory`. This model is fundamentally incompatible with concurrent HTTP + chat inference.

**Required changes (Phase 4 prerequisite):**

1. **Lease granularity**: Replace the single exclusive lease with a capacity-based admission model. The LLM slot becomes a pool (N concurrent sequences) rather than a mutex. TTS and ImageGen remain exclusive (they evict LLM sequences or wait).
2. **Preemption policy**: When TTS/ImageGen needs the GPU, the scheduler must drain or checkpoint active LLM sequences. Define a drain timeout — if sequences don't complete within N seconds, they are suspended (KV state saved to cache) and resumed after TTS/ImageGen finishes.
3. **Priority bands**: Agent chat (foreground, user-visible) > cron agents > HTTP requests. Within HTTP, FIFO. Priority affects both admission and preemption order.
4. **AgentCoordinator integration**: `AgentCoordinator` currently acquires and holds the arbiter lease for the full run. It must be refactored to acquire a scheduler slot instead, and release it between tool rounds (allowing other sequences to progress during tool execution).
5. **BackgroundAgentFactory**: Currently queues behind the arbiter. Must be updated to submit through the scheduler like any other client.

This is a significant architectural change — not a wrapper. It should be designed and reviewed as a standalone sub-spec before Phase 4 implementation begins.

### 8.4 Settings

New settings in `SettingsManager`:

```swift
var isServerEnabled: Bool = false
var serverPort: Int = 8321
var serverMaxConcurrentSequences: Int = 4
var cacheSSDMaxGB: Int = 10
```

---

## 9. Implementation Phases

### Phase 1: HTTP Server + Basic Inference (Foundation)

- Add `com.apple.security.network.server` to both `tesseract.entitlements` and `tesseractRelease.entitlements`
- `HTTPServer` with `NWListener`, routing, SSE streaming
- `POST /v1/chat/completions` (streaming + non-streaming)
- `GET /v1/models`, `GET /health`
- Message conversion: OpenAI format ↔ internal `LLMMessage`
- Tool call parsing (Qwen3.5 XML → OpenAI JSON format) via existing `ToolCallParser`
- Single-sequence inference serialized through the existing `InferenceArbiter`. Each HTTP request acquires an exclusive `.llm` lease via `InferenceArbiter.withExclusiveGPU(.llm)` — the same path used by `AgentCoordinator` and `BackgroundAgentFactory`. If the lease is held (e.g., Agent chat is generating), the HTTP request waits in line. If an HTTP request holds the lease, Agent chat waits. No separate queue or bypass — one shared admission point for all LLM access.
- Settings toggle + port config
- **Validation**: OpenCode connects and completes a multi-turn tool-calling conversation. Verify that an HTTP request during an active Agent chat turn queues correctly (no overlap, no deadlock).

### Phase 2a: KV Cache Access Research Spike (GATE)

The entire cache system depends on the ability to extract, save, and restore KV cache state from `TokenIterator` / `ModelContainer`. Currently:
- Swift `TokenIterator` creates KV caches privately — no public API to extract or inject them.
- `LLMActor` keeps `ModelContainer` private — no external access.
- The `DependencyContainer` integration sketch references `agentEngine.modelContainer` which does not exist as a public property.

**This phase is a research gate.** No cache implementation work should begin until it is resolved.

Deliverables:
- Determine whether `mlx-swift-lm` can be extended without forking (e.g., subclass `TokenIterator`, expose cache via protocol extension).
- If forking is required, scope the minimal changes needed: public `KVCache` accessor on `TokenIterator`, a `generate(withCache:)` entry point on `ModelContainer`, or similar.
- If upstream contribution is viable, open a PR/issue on `ml-explore/mlx-swift-lm`.
- Produce a technical design doc with the chosen approach, API surface, and estimated effort.

**Exit criteria**: A working prototype that (a) runs a generation, (b) extracts the KV cache as MLXArrays, (c) starts a new generation seeded with that cache, and (d) produces identical output to a full re-prefill.

### Phase 2b: Prompt Cache System

Gated on Phase 2a completion.

- `CacheManager` actor with block-based paged KV cache
- Chain-hash prefix matching
- Segment-aware priority LRU eviction
- Integration with `LLMActor` — store/restore KV blocks via the API surface established in Phase 2a
- `usage.prompt_tokens_details.cached_tokens` in responses
- Agent chat uses the cache (replace `Memory.clearCache()` between tool rounds)
- **Validation**: Cache hit rate > 80% on typical coding agent workloads. Agent chat TTFT drops from seconds to <200ms on cache hit.

### Phase 3: SSD Cold Tier

- `SSDCacheTier` with safetensors persistence
- Background write from inference actor
- Startup index rebuild
- LRU eviction with configurable max size
- **Validation**: Cold start (after app restart) loads system prompt from SSD in <100ms

### Phase 4: Concurrent Inference

- `InferenceScheduler` actor with round-robin scheduling
- Multiple `TokenIterator` instances
- Prefill priority (Agent chat > HTTP)
- Memory pressure monitoring
- Configurable concurrency limit
- InferenceArbiter redesign (see section 8.3 — requires its own sub-spec)
- **Validation**: 4 OpenCode instances running simultaneously with acceptable per-client throughput

> **Note**: Parallel agent tool execution (running multiple tools concurrently within a single agent turn) is **out of scope** for this spec. The current `AgentLoop` executes tools sequentially and uses that ordering for steering and skip semantics. Changing that behavior is a separate feature with its own design considerations.

### Phase 5: Statistics Page

- SwiftUI statistics page
- Real-time metrics via `@Observable` `ServerStats`
- Per-client tracking
- Cache visualization
- Server controls (start/stop, port)

### Phase 6: Batched Inference (Future)

- Port Python `BatchKVCache` (`merge/filter/extract`) to Swift
- `BatchGenerator` with true `[B, H, L, D]` tensor batching
- Continuous batching (add/remove sequences mid-generation)
- **Expected improvement**: ~2-3x throughput for concurrent sequences vs interleaved

---

## 10. Open Questions

1. **MLXLMCommon KV cache access**: The current Swift `TokenIterator` creates KV caches privately. We need to either (a) expose cache save/restore on `ModelContainer`, or (b) fork/extend `mlx-swift-lm` to support external cache management. Need to investigate what's possible without model changes.

2. **Block size tuning**: 256 tokens is the vLLM/omlx standard. May need tuning for MLX's Metal shader tile sizes. Profile with 128, 256, 512 to find the sweet spot.

3. **KV cache quantization interaction**: The block cache stores quantized KV tensors (8-bit or 4-bit per `AgentGenerateParameters.kvBits`). Blocks must be tagged with their quantization config — a block cached at 8-bit can't be used if the next request specifies 4-bit. Since the server uses a fixed config, this should be consistent, but needs explicit handling.

4. **Chat template compatibility**: The server applies the Jinja chat template via `ModelContainer`. When an HTTP client sends `tools`, the template must format them correctly. Qwen3.5's template handles tools natively. Other models may not — need graceful degradation.

5. **Context window management for HTTP clients**: The Agent chat has compaction (summarize old messages at 120K tokens). HTTP clients manage their own context. The server should return clear errors when context is exceeded rather than silently truncating.

6. **OpenCode → Crush migration**: OpenCode has been archived and continued as [Crush](https://github.com/charmbracelet/crush) by Charmbracelet. The API requirements are identical (same OpenAI Go SDK). We should test against both.

7. **Cron agent integration**: The statistics page mentions cron agents. The cron agent system itself is a separate feature — this spec assumes they will submit requests through the `InferenceScheduler` like any other client. The cron agent design is out of scope for this spec.

---

## 11. Implementation Tasks

Sequential task list for implementation across sessions. Each task is designed to be completable in a single session. Tasks within a phase are ordered by dependency — do not skip ahead.

**Notation**: `[Blocked by: T##]` means a task cannot start until the referenced task is done. Tasks without a blocker can start as soon as the previous task in the list is complete.

---

### Phase 1: HTTP Server + Basic Inference

#### T01 — Entitlements + Settings Model

Add `com.apple.security.network.server` to both entitlement files. Add server settings to `SettingsManager`.

**Files to modify:**
- `tesseract/tesseract.entitlements` — add `com.apple.security.network.server`
- `tesseract/tesseractRelease.entitlements` — same
- `tesseract/Features/Settings/SettingsManager.swift` — add `isServerEnabled: Bool`, `serverPort: Int`

**Verify:** App builds and launches. Confirm entitlement in both files with `plutil -p`.

---

#### T02 — OpenAI API Types (Codable Models)

Define the request and response types for the OpenAI-compatible API. These are pure data types with no logic — foundation for all HTTP handling.

**New file:** `tesseract/Features/Server/Models/OpenAITypes.swift`

**Types to define:**
- `ChatCompletionRequest` — messages, tools, model, stream, max_tokens, max_completion_tokens, temperature, top_p, reasoning_effort, stream_options, stop
- `ChatCompletionResponse` — id, object, model, created, system_fingerprint, choices, usage
- `ChatCompletionChunk` — same structure but with `delta` instead of `message`
- `ChatMessage` — role (system/user/assistant/tool), content (string or content parts array), tool_calls, tool_call_id
- `ContentPart` — type (text/image_url), text, image_url
- `ToolCall` — id, type, function (name + arguments JSON string), index (streaming only)
- `ToolDefinition` — type, function (name, description, parameters)
- `Usage` — prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details
- `ModelListResponse` — object, data array
- `ModelObject` — id, object, type, owned_by, max_context_length, loaded_context_length, state

**Verify:** Types compile. Write a unit test encoding/decoding a sample OpenCode request and response JSON (from spec section 4.2).

---

#### T03 — Schema-Only Generate Overload on AgentEngine

Add a `generate()` overload on `AgentEngine` that accepts raw `[ToolSpec]?` (i.e., `[[String: any Sendable]]?`) instead of `[AgentToolDefinition]?`. The HTTP server uses this path because client-supplied tools are schemas for prompt rendering only — the server never executes them.

**Modify:** `tesseract/Features/Agent/AgentEngine.swift`

**Implementation:**
- Add overload: `func generate(systemPrompt: String, messages: [LLMMessage], toolSpecs: [ToolSpec]?, parameters: AgentGenerateParameters) throws -> AsyncThrowingStream<AgentGeneration, Error>`
- The existing `generate(..., tools: [AgentToolDefinition]?)` already converts to `ToolSpec` at line 122 (`tools?.map { $0.toolSpec }`), then passes to `UserInput(chat:tools:)`. The new overload skips that conversion and passes `toolSpecs` directly to `UserInput`.
- The existing overload can be refactored to call the new one: `generate(systemPrompt:, messages:, toolSpecs: tools?.map { $0.toolSpec }, parameters:)`

**Verify:** Existing Agent chat still works (calls the `[AgentToolDefinition]?` overload unchanged). New overload compiles and produces identical output when given the same tool specs. Unit test: pass a hand-built `ToolSpec` dictionary → generation includes tools in the prompt.

---

#### T04 — Message Conversion Layer

Convert between OpenAI message format and internal `LLMMessage` types. This is the bridge between the HTTP API and the existing inference engine.

**New file:** `tesseract/Features/Server/MessageConverter.swift`

**Functions:**
- `convertMessages(_ messages: [ChatMessage]) -> (systemPrompt: String?, messages: [LLMMessage])` — extract system message, convert user/assistant/tool roles
- `convertToolDefinitions(_ tools: [ToolDefinition]?) -> [[String: any Sendable]]?` — OpenAI function schema → `ToolSpec` (raw schema dictionaries). **Do NOT convert to `AgentToolDefinition`** — that type carries an `execute` closure and represents tools Tesseract itself can run. HTTP clients only send schemas for prompt rendering; the model output tells the *client* which tools to call, the server never executes them. Pass `ToolSpec` arrays directly to `AgentEngine.generate()` via the overload added in T03.
- `convertImageContent(_ part: ContentPart) -> ImageAttachment?` — decode `data:` URI base64 images
- Handle multipart user messages (text + images)
- Map `role: "tool"` messages with `tool_call_id` → internal `.toolResult`

**Depends on:** `LLMMessage` enum in `Core/AgentMessage.swift`, T03 (schema-only generate overload).

**Verify:** Unit tests converting OpenCode-style multi-turn conversation (system → user → assistant with tool_calls → tool result → user) to internal types and back. Verify tool definitions pass through as schema-only — no `execute` closures created.

---

#### T05 — Tool Call Format Conversion

Convert Qwen3.5 XML tool calls from model output into OpenAI JSON format for the HTTP response, and convert OpenAI tool results back into the format expected by the chat template.

**New file:** `tesseract/Features/Server/ToolCallConverter.swift`

**Functions:**
- `convertToOpenAI(_ toolCalls: [ToolCall]) -> [OpenAIToolCall]` — internal parsed tool calls (from `ToolCallParser`) → OpenAI format with generated `call_<uuid>` IDs
- `mapToolCallIDs(_ messages: [ChatMessage], idMap: [String: String]) -> [ChatMessage]` — when client sends back `tool_call_id`, map it to internal IDs if needed

**Key behavior:**
- Tool call IDs are generated server-side as `call_<UUID>`. The `ToolCallParser` doesn't produce IDs (Qwen3.5 XML format has no IDs), so the server assigns them.
- Arguments must be JSON-stringified in the response (`function.arguments` is a string, not an object).

**Verify:** Unit test: feed raw `<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>` through `ToolCallParser` → `convertToOpenAI` → verify correct OpenAI JSON format.

---

#### T06 — HTTP Transport Layer (NWListener)

Build the core HTTP server using `Network.framework`. This handles TCP connections, HTTP request parsing, and response writing. No endpoint logic yet — just the transport.

**New file:** `tesseract/Features/Server/HTTPServer.swift`

**Implementation:**
- `HTTPServer` class: `@Observable @MainActor`
- `NWListener` on configurable port, bound to `127.0.0.1`
- `start()` / `stop()` lifecycle methods
- HTTP/1.1 request parser: extract method, path, headers, body from raw TCP bytes
- Response writer: status line + headers + body, chunked transfer for SSE
- Route dispatch: match (method, path) → handler
- Error responses: 400 (bad request), 404 (not found), 405 (method not allowed), 500 (internal), 503 (model not loaded)
- Connection lifecycle: accept → parse request → dispatch → respond → close (or keep-alive)

**Verify:** Server starts on port 8321. `curl http://127.0.0.1:8321/health` returns 404 (no routes wired yet). `curl` to random path returns 404. Server stops cleanly.

---

#### T07 — SSE Streaming Support

Add Server-Sent Events streaming capability to the HTTP transport layer.

**Modify:** `tesseract/Features/Server/HTTPServer.swift`

**Implementation:**
- SSE response mode: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`
- `SSEWriter` helper: formats `data: {json}\n\n` lines, sends keepalive comments (`: keepalive\n\n`), sends `data: [DONE]\n\n` sentinel
- Streaming lifecycle: open connection → write headers → yield chunks → `[DONE]` → close
- Back-pressure: if the client disconnects mid-stream, detect and cancel the generation

**Verify:** Manual test with `curl -N http://127.0.0.1:8321/test-sse` (temporary test endpoint) that streams 5 JSON lines with 100ms delay, then `[DONE]`.

---

#### T08 — GET Endpoints (/health, /v1/models)

Wire the simple read-only endpoints.

**Modify:** `tesseract/Features/Server/HTTPServer.swift`

**Implementation:**
- `GET /health` → `{"status": "ok"}`
- `GET /v1/models` → query `AgentEngine` for loaded model info. Return `ModelListResponse` with the loaded model's name, context window, state. Return empty `data` array if no model loaded.
- Need to read model info from `AgentEngine` / `LLMActor` — model name, context window size

**Depends on:** `AgentEngine` must expose model metadata (name, context length). Check what's already public.

**Verify:** `curl http://127.0.0.1:8321/health` returns `{"status":"ok"}`. `curl http://127.0.0.1:8321/v1/models` returns model list (or empty data if model not loaded). Verify with OpenCode's expected response shape.

---

#### T09 — InferenceArbiter Integration + CompletionHandler

Route HTTP inference through the existing `InferenceArbiter` lease system. This MUST be done before any completions endpoint work, since all generation calls require the lease.

**New file:** `tesseract/Features/Server/CompletionHandler.swift`

**Implementation:**
- `CompletionHandler` receives references to `InferenceArbiter`, `AgentEngine` (injected via `HTTPServer`)
- Core method: `handleCompletion(request:connection:)` — parses request, acquires lease, runs generation, writes response
- Before calling `AgentEngine.generate()`, acquire `.llm` lease: `try await arbiter.withExclusiveGPU(.llm) { ... }`
- The entire generation (prefill + decode) runs inside the lease scope
- If lease acquisition is cancelled (e.g., client disconnects while waiting), return HTTP 503
- Add timeout for lease acquisition (configurable, default 60s) — return 503 with `Retry-After` if exceeded

**Verify:** Start Agent chat generation → send HTTP request → HTTP request waits → Agent chat finishes → HTTP request proceeds. No overlap in `LLMActor` access.

---

#### T10 — POST /v1/chat/completions (Non-Streaming)

Implement the core inference endpoint in non-streaming mode. Arbiter integration from T09 is already in place.

**Modify:** `tesseract/Features/Server/CompletionHandler.swift`

**Implementation:**
- Parse `ChatCompletionRequest` from JSON body
- Call `MessageConverter` to convert messages → internal format
- Call `MessageConverter.convertToolDefinitions()` to get `ToolSpec` arrays (schema-only, no execute closures)
- Inside arbiter lease (from T09), call `AgentEngine.generate()` with converted messages, system prompt, tool specs
- Accumulate `AgentGeneration` events from the `AsyncThrowingStream` directly — **do NOT re-parse tool calls from raw text**. `AgentEngine.generate()` already returns `.toolCall(ToolCall)` events via its internal `ToolCallParser`. Collect `.text`, `.toolCall`, `.thinking`, and `.info` events into the response.
- Convert accumulated `ToolCall` objects to OpenAI format via `ToolCallConverter.convertToOpenAI()`
- Build `ChatCompletionResponse` with content, tool_calls, finish_reason, usage
- Return as JSON

**Key details:**
- `finish_reason`: "stop" (normal), "length" (max tokens), "tool_calls" (tool calls present)
- `usage.prompt_tokens` and `usage.completion_tokens` from `AgentGeneration.info`
- `usage.prompt_tokens_details.cached_tokens` = 0 (no cache in Phase 1)
- Thinking blocks: include in `message.content` and optionally split into `message.reasoning`

**Verify:** `curl -X POST http://127.0.0.1:8321/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"test","messages":[{"role":"user","content":"Say hello"}]}'` returns valid response.

---

#### T11 — POST /v1/chat/completions (Streaming)

Add streaming mode to the completions endpoint.

**Modify:** `tesseract/Features/Server/CompletionHandler.swift`

**Implementation:**
- When `request.stream == true`, switch to SSE response mode
- For each `AgentGeneration` event from the stream (already parsed by `AgentEngine`):
  - `.text(str)` → emit `ChatCompletionChunk` with `delta.content = str`
  - `.toolCall(tc)` → convert via `ToolCallConverter`, emit as `delta.tool_calls` chunks with index
  - `.thinkStart` / `.thinking` / `.thinkEnd` → emit as content (thinking tags in stream)
  - `.info(info)` → emit final chunk with `finish_reason` + `usage` (if `include_usage`)
- Send `data: [DONE]\n\n` after final chunk
- Send keepalive comments every 5s during prefill
- Handle client disconnect: cancel the per-request `Task` that wraps the generation. `CancellationToken` is for tool execution only and is not used by `AgentEngine.generate()`. The generation stream is task-based — cancelling the wrapping `Task` triggers cooperative cancellation through Swift structured concurrency, which terminates the `TokenIterator` and releases the arbiter lease. If the SSE connection drops (detected via `NWConnection` state change), cancel the task immediately to avoid holding the lease for a disconnected client.

**Verify:** `curl -N -X POST http://127.0.0.1:8321/v1/chat/completions -d '{"model":"test","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'` streams SSE chunks, ends with `[DONE]`.

---

#### T12 — Settings UI + DependencyContainer Wiring

Add server toggle and port configuration to Settings UI. Wire `HTTPServer` into `DependencyContainer`.

**Modify:**
- `tesseract/App/DependencyContainer.swift` — add `httpServer` lazy property
- `tesseract/Features/Settings/SettingsView.swift` (or relevant settings view) — add server section with toggle + port field
- `tesseract/Core/ViewModifiers.swift` — inject `HTTPServer` if needed for stats page

**Implementation:**
- `DependencyContainer.httpServer` = `HTTPServer(agentEngine:, inferenceArbiter:, settingsManager:)`
- Server starts/stops reactively based on `settingsManager.isServerEnabled` (observe with `Observations`)
- Port changes require server restart — warn in UI or auto-restart
- Show server address `http://127.0.0.1:{port}` when enabled (copyable)

**Verify:** Toggle server on in Settings → `curl /health` works. Toggle off → connection refused. Change port → server restarts on new port.

---

#### T13 — End-to-End Validation with OpenCode

Full integration test with OpenCode (or Crush) as the client.

**No new code** — this is a validation session.

**Test plan:**
1. Install OpenCode: `go install github.com/opencode-ai/opencode@latest` (or Crush)
2. Configure: `LOCAL_ENDPOINT=http://127.0.0.1:8321`
3. Verify model discovery: OpenCode detects the loaded model via `GET /v1/models`
4. Simple conversation: "What files are in this directory?" → tool call (ls/glob) → tool result → response
5. Multi-turn with tool calls: "Read main.swift and explain it" → read tool → explanation
6. Streaming verification: Watch SSE stream in real-time
7. Concurrent access: Start Agent chat turn → send OpenCode request → verify queuing
8. Error cases: Request with no model loaded (503), oversized context (400)
9. Edge cases: Empty messages array, missing tools field, unknown parameters (should be ignored)

**Fix any issues found** — this task may span multiple iterations.

---

### Phase 2a: KV Cache Access Research Spike

#### T14 — Audit mlx-swift-lm KV Cache Internals [Blocked by: T13]

Research-only task. No code changes to Tesseract.

**Goal:** Map the complete KV cache lifecycle in `mlx-swift-lm` and identify the narrowest intervention point for external cache management.

**Investigation:**
1. Read `ModelContainer.swift` — how `generate()` creates `TokenIterator`, how `SerialAccessContainer` manages access
2. Read `KVCache.swift` — `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache` internals. What state do they hold? What's the shape of stored arrays?
3. Read `TokenIterator.swift` — how it creates caches via `model.newCache()`, how caches grow during generation
4. Read `Generation.swift` — the `prepare()` and `generate()` paths on `ModelContainer`
5. Check if `TokenIterator.cache` is accessible (public/internal?)
6. Check if `ModelContainer` has any API for external cache injection
7. Check the `test/tesseract-integration` branch for any custom modifications already made

**Deliverable:** Written summary (in this spec or separate doc) with:
- Exact types and shapes of KV cache arrays per layer
- Which access modifiers need to change (private → public)
- Whether forking is required or if extension/subclass works
- Proposed minimal API: which methods/properties need to be exposed
- Risk assessment: does exposing the cache break any invariants?

---

#### T15 — Prototype KV Cache Extract + Restore [Blocked by: T14]

Build a minimal proof-of-concept that extracts KV state after generation and uses it to seed a new generation.

**Implementation approach (depends on T13 findings):**
- Option A (if cache is accessible): Extract `[KVCache]` array from `TokenIterator` after generation. Create a new `TokenIterator` with pre-populated cache. Generate continuation.
- Option B (if fork needed): Fork `mlx-swift-lm`, add `public var cache: [any KVCache]` on `TokenIterator` + `generate(withCache:)` on `ModelContainer`. Update `Package.swift` to point to fork.
- Option C (snapshot approach): Use `KVCache.copy()` (already exists) to snapshot state, then investigate how to inject the copy into a new iterator.

**Verify (exit criteria):**
1. Generate response for prompt "The capital of France is" — capture KV state
2. Start new generation with same prompt using restored KV — verify it skips prefill and produces identical continuation
3. Start new generation with extended prompt "The capital of France is Paris. The capital of Germany is" using restored KV — verify only the new tokens are prefilled
4. Measure time difference: full prefill vs restored cache

**Deliverable:** Working code (can be throwaway/test harness), timing measurements, and the chosen approach documented for Phase 2b.

---

### Phase 2b: Prompt Cache System

#### T16 — CacheBlock + Chain Hash Data Structures [Blocked by: T15]

Implement the core data structures for the block-based cache, with no integration yet.

**New file:** `tesseract/Features/Server/Cache/CacheBlock.swift`
**New file:** `tesseract/Features/Server/Cache/ChainHash.swift`

**Implementation:**
- `CacheBlock` struct (per spec section 5.2): blockID, layerCaches, refCount, blockHash, segmentType, lastAccess, free-list pointers
- `SegmentType` enum: `.assistant`, `.user`, `.system` with `Comparable`
- `computeBlockHash(parentHash:tokens:modelID:)` using `CryptoKit.SHA256`
- `tokenizeIntoBlocks(_ tokens: [Int32]) -> [[Int32]]` — split token array into 256-token blocks + remainder
- `computeBlockChain(tokens:modelID:) -> [Data]` — compute chain hashes for all complete blocks

**Verify:** Unit tests:
- Hash determinism: same tokens → same hash every time
- Chain property: changing one early token changes all subsequent hashes
- Block splitting: 600 tokens → 2 full blocks + 88 remainder
- Empty input → no blocks

---

#### T17 — FreeBlockQueue (Doubly-Linked List) [Blocked by: T16]

Implement the O(1) doubly-linked free list used for LRU eviction within each segment tier.

**New file:** `tesseract/Features/Server/Cache/FreeBlockQueue.swift`

**Implementation:**
- `FreeBlockQueue` — doubly-linked list of block IDs using array-based storage (indices as pointers)
- `pushBack(_ blockID: Int)` — add to tail (most recently freed)
- `popFront() -> Int?` — remove from head (least recently used)
- `remove(_ blockID: Int)` — remove from middle (when block is re-claimed)
- `count` property
- `isEmpty` property

**Verify:** Unit tests: push 5 items, pop returns FIFO order. Remove from middle. Push after remove. Empty queue returns nil.

---

#### T18 — CacheManager Actor (Core Operations) [Blocked by: T17]

Implement the `CacheManager` actor with prefix lookup, block storage, and eviction.

**New file:** `tesseract/Features/Server/Cache/CacheManager.swift`

**Implementation:**
- Pre-allocated block pool (size based on available memory)
- `hashIndex: [Data: Int]` — chain hash → block ID
- Three `FreeBlockQueue` instances (one per `SegmentType`)
- `findPrefix(tokens:modelID:) -> PrefixMatch` — compute chain hashes, look up in hashIndex, return longest match
- `storeBlocks(tokens:caches:segmentTypes:) -> [Int]` — allocate blocks from free list, store KV arrays, index by hash
- `claimBlocks(blockIDs:sequenceID:)` — increment refCount, remove from free list
- `releaseBlocks(sequenceID:)` — decrement refCount, add to free list if refCount reaches 0
- `evictToFit(bytesNeeded:)` — evict from assistant tier first, then user, then system
- `CacheStats` tracking (per spec section 5.8)
- Partial block handling: do NOT store the tail (< 256 tokens), return it as `pendingTokens` in `PrefixMatch`

**Verify:** Unit tests:
- Store 3 blocks → findPrefix with same tokens → returns all 3 (hot hit)
- Store blocks → findPrefix with first 2 matching + 1 different → returns 2 blocks + remaining tokens
- Fill to capacity → evict → assistant blocks evicted first
- Claim + release → refCount lifecycle
- Stats: lookups, hits, misses tracked correctly

---

#### T19 — LLMActor + AgentEngine Cache Integration [Blocked by: T18]

Connect the `CacheManager` to the inference pipeline so that KV state is stored after generation and restored on the next call.

**Modify:**
- `tesseract/Features/Agent/LLMActor.swift` — after generation completes, extract KV cache and pass to `CacheManager.storeBlocks()`. Before generation, call `CacheManager.findPrefix()` and seed the iterator with cached state.
- `tesseract/Features/Agent/AgentEngine.swift` — pass `CacheManager` reference to `LLMActor`, expose `CacheManager` for stats

**Key changes:**
- Replace `Memory.clearCache()` between tool rounds with cache store/restore cycle
- The generate path becomes: tokenize → findPrefix → create iterator with cached KV → prefill only remaining tokens → decode → store new blocks
- Wire through `DependencyContainer`

**Verify:**
- Agent chat multi-turn: first turn prefills fully, second turn shows reduced prefill time (cache hit)
- HTTP request: `usage.prompt_tokens_details.cached_tokens` reports non-zero when prefix matches
- Log cache hit/miss stats

---

#### T20 — Agent Chat Cache Integration [Blocked by: T19]

Ensure the Agent chat's multi-turn conversations benefit from the prompt cache. This replaces the current `Memory.clearCache()` pattern with incremental prefill.

**Modify:**
- `tesseract/Features/Agent/Core/AgentLoop.swift` — remove or gate `Memory.clearCache()` calls. The cache manager now handles KV lifecycle.
- Verify that compaction (context summarization at 120K tokens) correctly invalidates cached blocks when the conversation is rewritten.

**Verify:**
- 5-turn Agent chat conversation: measure TTFT on each turn. Turns 2-5 should show significant TTFT reduction vs turn 1.
- Compaction trigger: when context is compacted, the next turn correctly cache-misses on the rewritten prefix and re-prefills.
- No memory leaks: blocks are released when conversations are cleared.

---

### Phase 3: SSD Cold Tier

#### T21 — SSD Cache Read/Write [Blocked by: T20]

Implement the SSD persistence layer for cold KV cache blocks.

**New file:** `tesseract/Features/Server/Cache/SSDCacheTier.swift`

**Implementation:**
- Directory: `~/Library/Caches/Tesseract Agent/kv-cache/`
- Hash-bucketed paths: `{hash[0:2]}/{hash[2:4]}/{hash}.safetensors`
- `writeBlock(hash:layerCaches:)` — extract raw bytes from MLXArray on calling actor, write safetensors file from `Task.detached` (no Metal API calls from background thread)
- `readBlock(hash:) -> [(key: MLXArray, value: MLXArray)]?` — load safetensors, reconstruct MLXArrays
- `SSDBlockEntry` struct: hash, file path, byte size, last access time
- In-memory index: `[Data: SSDBlockEntry]`

**Verify:** Write a block to SSD → read it back → compare MLXArray contents are identical. Verify file is at expected hash-bucketed path.

---

#### T22 — SSD Index Rebuild + Eviction [Blocked by: T21]

Add startup index rebuild from existing files and LRU eviction.

**Modify:** `tesseract/Features/Server/Cache/SSDCacheTier.swift`

**Implementation:**
- `rebuildIndex()` — scan cache directory on startup, rebuild `[Data: SSDBlockEntry]` from filenames
- LRU eviction: track access times, evict oldest when total size exceeds `cacheSSDMaxGB` setting
- `trimToSize(maxBytes:)` — called after writes and periodically
- Use `URLResourceKey.totalFileAllocatedSizeKey` for accurate disk usage tracking

**Modify:** `tesseract/Features/Settings/SettingsManager.swift` — add `cacheSSDMaxGB: Int = 10`

**Verify:** Fill SSD cache beyond limit → oldest blocks are evicted. Kill and restart app → index is rebuilt → previously cached blocks are found.

---

#### T23 — CacheManager Two-Tier Integration [Blocked by: T22]

Connect the SSD tier to the `CacheManager` so lookups check both hot and cold tiers, and eviction spills to SSD.

**Modify:** `tesseract/Features/Server/Cache/CacheManager.swift`

**Implementation:**
- `findPrefix()`: after hot miss, check `ssdTier.readBlock(hash:)`. If SSD hit, promote block to hot tier.
- `evictToFit()`: before freeing a block's memory, write it to SSD via `ssdTier.writeBlock()`.
- Background write: eviction writes to SSD asynchronously — don't block the eviction path.
- Stats: track `ssdHits` separately from `hotHits`.

**Verify:**
- Load blocks into hot cache → evict (memory pressure) → blocks appear on SSD → new request with same prefix → SSD hit → blocks promoted to hot
- Cold start test: quit app → restart → first request finds system prompt blocks on SSD → faster TTFT than full prefill (target: <100ms for 4K token system prompt)

---

### Phase 4: Concurrent Inference

#### T24 — InferenceArbiter Redesign Sub-Spec [Blocked by: T23]

Design-only task. Produce a sub-spec for the arbiter redesign before writing code.

**Deliverable:** A section in this spec (or separate doc) covering:
1. Capacity-based admission model: N LLM slots (configurable) vs exclusive TTS/ImageGen
2. Preemption protocol: what happens to active LLM sequences when TTS needs GPU? Drain timeout, KV checkpoint, resumption.
3. Priority bands: foreground Agent chat > cron agents > HTTP. How priority affects admission and preemption order.
4. `AgentCoordinator` refactor: acquire scheduler slot (not arbiter lease), release between tool rounds.
5. `BackgroundAgentFactory` update: submit through scheduler.
6. API surface: what does the new arbiter/scheduler interface look like to callers?
7. Migration path: how to swap from current arbiter to new system without breaking existing behavior.

**Verify:** Review with user before proceeding to implementation.

---

#### T25 — InferenceScheduler Actor [Blocked by: T24]

Implement the core scheduler that manages multiple concurrent generation sequences.

**New file:** `tesseract/Features/Server/InferenceScheduler.swift`

**Implementation:**
- `InferenceScheduler` actor (per spec section 6.2)
- `submit(input:parameters:source:) -> AsyncThrowingStream<Generation, Error>` — queue or start a sequence
- `cancel(sequenceID:)` — stop a running sequence
- `ActiveSequence` struct: iterator, continuation, state, source
- Pending queue for requests waiting for a slot
- Slot management: track active count vs `maxConcurrent`
- Integration with new arbiter (from T23 design): acquire LLM pool slot, release on completion

**Verify:** Submit 2 sequences → both start (if slots available). Submit more than `maxConcurrent` → excess queues. Cancel a running sequence → slot freed → queued sequence starts.

---

#### T26 — Round-Robin Scheduling Loop [Blocked by: T25]

Implement the interleaved token generation loop.

**Modify:** `tesseract/Features/Server/InferenceScheduler.swift`

**Implementation:**
- `schedulingLoop()` — runs as a `Task` while sequences are active
- Round-robin: generate 1 token from each active sequence in turn
- Prefill: when a new sequence starts, prefill runs to completion (or chunked in 2048-token steps with yields)
- Use `CacheManager.findPrefix()` to seed new sequences with cached KV state
- During prefill, other sequences are paused — send keepalive to their HTTP clients
- After generation completes: store blocks in `CacheManager`, release slot

**Verify:** 2 concurrent sequences: both produce tokens interleaved (visible in logs). Throughput per sequence is ~50% of single-sequence (expected with round-robin). Prefill of sequence B doesn't cause sequence A to time out.

---

#### T27 — Arbiter Capacity Model + AgentCoordinator Refactor [Blocked by: T26]

Implement the new arbiter design and refactor existing callers.

**Modify:**
- `tesseract/Features/Agent/InferenceArbiter.swift` — replace exclusive LLM lease with pool-based admission
- `tesseract/Features/Agent/AgentCoordinator.swift` — acquire scheduler slot instead of arbiter lease, release between tool rounds
- `tesseract/Features/Agent/BackgroundAgentFactory.swift` — submit through scheduler

**Implementation:**
- LLM pool: N slots (from `serverMaxConcurrentSequences` setting). Each `submit()` claims a slot.
- TTS/ImageGen: exclusive. When requested, drain active LLM sequences (with timeout), run exclusively, then resume.
- Foreground priority: Agent chat `submit()` with `.agentChat` source gets next available slot, preempting queued HTTP requests.
- `AgentCoordinator`: hold slot only during generation, release during tool execution, re-acquire for next turn.

**Verify:**
- Agent chat works as before (no regression)
- Background agents work through scheduler
- 4 HTTP requests + Agent chat = 5 concurrent sequences (Agent chat has reserved slot)
- TTS request during active HTTP generation: HTTP sequences drain, TTS runs, HTTP resumes

---

#### T28 — Memory Pressure + HTTP 503 [Blocked by: T27]

Add memory-aware admission control and proper HTTP error responses for overload.

**Modify:**
- `tesseract/Features/Server/InferenceScheduler.swift` — check `os_proc_available_memory()` before admitting new sequences
- `tesseract/Features/Server/CompletionHandler.swift` — return 503 with `Retry-After` header when overloaded

**Implementation:**
- Before admitting a new sequence, estimate its KV memory cost: `context_length * per_token_kv_bytes * (1 + safety_margin)`
- If estimated cost exceeds available memory, reject with 503
- If all slots are full, reject with 503 + `Retry-After: 5`
- If model is not loaded, reject with 503 + clear error message

**Modify:** `tesseract/Features/Settings/SettingsManager.swift` — add `serverMaxConcurrentSequences: Int = 4`

**Verify:** Fill all slots → next request gets 503. Load large model with low memory → new requests rejected based on memory. `Retry-After` header present.

---

### Phase 5: Statistics Page

#### T29 — ServerStats Observable Model [Blocked by: T12]

Note: This task depends only on T12 (basic server exists), not on cache/concurrency phases. Can start any time after Phase 1.

**New file:** `tesseract/Features/Server/ServerStats.swift`

**Implementation:**
- `ServerStats`: `@Observable @MainActor`
- Server stats: isRunning, address, uptime, activeConnections
- Inference stats: totalRequests, tokensGenerated, currentThroughput (tok/s rolling 10s window), peakThroughput, averageTTFT
- Cache stats: hotBlocksUsed, hotBlocksTotal, ssdBlockCount, ssdBytesUsed, hitRate (overall + hot/ssd/miss split)
- Per-client stats: `[ClientID: ClientStats]` where `ClientStats` has requestCount, tokensIn, tokensOut, cacheHitRate, lastActive
- Client identification: HTTP `User-Agent` header, or internal source type string
- Updated by `HTTPServer` via direct method calls (both `@MainActor`). `CacheManager` is a separate `actor` (not `@MainActor`), so stats from the cache layer must be pulled asynchronously: either (a) `CacheManager` pushes a `CacheStats` snapshot to `ServerStats` after each operation via `await serverStats.updateCacheStats(snapshot)`, or (b) `ServerStats` polls `await cacheManager.stats` on a timer (e.g., every 1s). Option (a) is preferred — avoids polling and keeps stats current. Cache stats will show zeros until Phase 2b lands, but the `ServerStats` API should accept the updates from day one.

**Verify:** Stats object compiles and tracks basic counts. Unit test: increment counters → verify computed properties.

---

#### T30 — Statistics Page UI [Blocked by: T29]

Build the SwiftUI statistics page.

**New file:** `tesseract/Features/Server/ServerStatsView.swift`

**Implementation:**
- Four sections matching spec section 7: Server, Inference, Cache, Per-Client
- Server section: status badge (green/red), address (copyable), uptime, active connections/sequences
- Inference section: model name, total requests, tokens generated, throughput gauge, TTFT, queue depth
- Cache section: hot/SSD tier bars, hit rate pie/ring chart, top cached prefixes list
- Per-client section: table with columns per spec section 7.4
- Auto-refreshes via `@Observable` bindings (no polling needed)
- Server start/stop toggle in the page header

**Verify:** Page renders with live data. Toggle server on/off from the page. Stats update in real-time during inference.

---

#### T31 — Sidebar Navigation [Blocked by: T30]

Add the statistics page to the app's sidebar navigation.

**Modify:**
- `tesseract/Models/NavigationItem.swift` — add `.server` case to the `NavigationItem` enum. Add it to `mainPages` array (e.g., after `.scheduled`). Implement `name` ("Server"), `symbolName` ("server.rack"), and `destinationView` (`ServerStatsView()`) in each switch.
- `tesseract/Features/Dictation/Views/SidebarView.swift` — no changes needed if it iterates `NavigationItem.mainPages` (the new case will appear automatically)
- `tesseract/Features/Dictation/Views/ContentView.swift` — add a `.server` case to the `injectedDestinationView(for:)` switch. The current app uses an explicit switch there (not `NavigationItem.destinationView`), so this file MUST be updated or the new page won't compile/route.
- Wire `ServerStats` through `DependencyContainer` and view modifier injection

**Verify:** Server page appears in sidebar. Navigation works. Page shows live stats. Other navigation (Chat, Settings) still works.

---

### Phase 6: Batched Inference (Future — Tasks TBD)

This phase requires porting ~2000 lines of Python batch infrastructure from `mlx-lm` to Swift. Tasks will be defined after Phase 4 is complete and real-world concurrency data is available to justify the investment.

Placeholder tasks:
- T32 — Port `BatchKVCache` to Swift (merge/filter/extract/extend on `[B,H,L,D]` tensors)
- T33 — Port `PromptProcessingBatch` (batched prefill with padding)
- T34 — Port `GenerationBatch` (batched decode with per-sequence sampling)
- T35 — Port `BatchGenerator` (continuous batching orchestrator)
- T36 — Model forward pass audit (ensure `[B,...]` inputs work through attention layers)
- T37 — Integration + benchmarks vs interleaved approach

---

### Task Dependency Graph

```
Phase 1 (HTTP MVP):
T01 → T02 → T03 (engine overload) → T04 (message conversion) → T05 → T06 → T07 → T08 → T09 (arbiter) → T10 (non-stream) → T11 (stream) → T12 → T13

Phase 2a (Research Gate):
T13 → T14 → T15

Phase 2b (Cache):
T15 → T16 → T17 → T18 → T19 → T20

Phase 3 (SSD):
T20 → T21 → T22 → T23

Phase 4 (Concurrency):
T23 → T24 → T25 → T26 → T27 → T28

Phase 5 (Stats UI — can start after Phase 1):
T12 → T29 → T30 → T31

Phase 6 (Future):
T28 → T32 ... T37
```

**Parallelism opportunity**: Phase 5 (T29–T31) can be worked on in parallel with Phases 2–4, since it only depends on T12 (basic server wired). Cache and per-client stats will show zeros until those phases land, but the UI and basic server/inference stats work independently.
