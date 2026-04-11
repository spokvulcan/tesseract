# Marconi-Style Hybrid Prefix Cache — Implementation Plan

**Date:** 2026-04-11 (v9 — paper/repo aligned, development-ready)
**Status:** Ready for development
**Prerequisite reading:** `docs/mlx-swift-lm-prefill-memory-research.md` (§4.7, §7.5–7.6)
**References:** [Marconi paper](https://assets.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf), [Marconi reference repo](https://github.com/ruipeterpan/marconi), [SGLang MambaRadixCache](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/), [mlx-lm #1072](https://github.com/ml-explore/mlx-lm/pull/1072)

---

## Review Blockers Addressed

### v2 blockers (resolved)

| # | Blocker | Resolution |
|---|---------|-----------|
| 1 | Attention-only Phase 1 cannot resume divergent suffixes on Qwen3.5 | Checkpoints store full hybrid cache (all layers). No attention-only path. |
| 2 | Phase 2 defined checkpoint as layer outputs instead of SSM state | `HybridCacheSnapshot` stores actual `MambaCache.state` (conv + recurrent arrays) and `KVCacheSimple.state` / `QuantizedKVCache.state`. Matches Marconi and mlx-lm #1072. |
| 3 | Partial-prefix hits don't define cache slicing to match length | Exact-prefix matching only at checkpointed offsets. No mid-stream trimming. |
| 4 | Normalization treated as diagnostics-only | Pipeline-critical. normalize → tokenize → radix lookup. |
| 5 | Storage model duplicates full cache per node | Selective admission at checkpoint boundaries only. Type-based eviction. |

### v3 blockers (resolved)

| # | Blocker | Resolution |
|---|---------|-----------|
| 6 | Phase 1 `store()` creates earlier snapshots from final cache — impossible for Mamba | Snapshots captured **during prefill**, not post-hoc. Checkpoint offsets known before prefill starts. Prefill loop captures snapshots as it passes each offset. |
| 7 | Checkpoint capture only fires on chunk boundaries, not arbitrary segment offsets | Prefill loop dynamically adjusts chunk size to land exactly on checkpoint offsets. Variable-size chunks near boundaries. |
| 8 | Snapshot restore assumes KVCacheSimple but runtime uses QuantizedKVCache | Restore mirrors `loadPromptCache()` pattern — stores `className` per layer, creates correct type on restore. Handles KVCacheSimple, QuantizedKVCache, RotatingKVCache, MambaCache. QuantizedKVCache constructor receives groupSize/bits parsed from metaState. |
| 9 | Three-block segmentation model (system/user/assistant) doesn't match real alternating prompt structure | Replaced with two concrete checkpoint types: **stable-prefix boundary** (cross-conversation reuse) and **conversation leaf** (within-conversation reuse, like current HTTPPrefixCacheSpike). No coarse block model. |

### v4 blockers (this revision)

| # | Blocker | Resolution |
|---|---------|-----------|
| 10 | Phase 1 integration calls `model.prepare()` directly, but `TokenIterator.init()` already owns prefill (`Evaluate.swift:607-609`) — would double-prefill or require non-trivial TokenIterator refactor | Checkpoint offsets flow INTO `TokenIterator` via `GenerateParameters`. TokenIterator passes them to `model.prepare()` inside its own init. Captured snapshots stored as `TokenIterator.capturedSnapshots` property. **No call to prepare() outside TokenIterator.** LLMActor reads snapshots after iterator creation. |
| 11 | Checkpoint capture loop exits at `y.tokens.size <= prefillStepSize`, so checkpoints in the final tail remainder are never captured | After main while-loop, a second drain loop processes any checkpoints that fall within the remaining tail before returning `.tokens(y)`. |
| 12 | System-boundary detection tokenizes system-only messages, but real shared prefix is system + tool definitions (tools passed as separate param to `applyChatTemplate(messages:tools:)` via `Tokenizer.swift:16-20`) | Renamed to `StablePrefixDetector`. Uses **two-probe technique**: tokenize messages with two different dummy user contents, find common prefix length = system + tools boundary. No sentinel token search (tokenization is context-dependent). |
| 13 | Token-only radix tree dropped model/config partitioning — snapshot from kvBits=8 could be returned for kvBits=nil request | `PrefixCacheManager` holds `[CachePartitionKey: TokenRadixTree]` dictionary. Partition key = `(modelID, kvBits, kvGroupSize)`. Matches existing `HTTPPrefixCacheKey` fields. Tool/template digests implicit in token sequences. |

### v5 blockers (this revision)

| # | Blocker | Resolution |
|---|---------|-----------|
| 14 | Checkpoint offsets are in full-prompt coordinates but suffix prefill starts at offset 0 — checkpoints never fire on cache hit | `prepare()` gains `checkpointBaseOffset: Int` (default 0). Set to `snapshot.tokenOffset` on cache hit. Inside the loop, checks use `checkpointBaseOffset + currentOffset`. Snapshot stored with the correct absolute offset. |
| 15 | Leaf snapshot stored under pre-generation `fullTokens` but `finalCache` includes the generated response — offset mismatch | After generation, re-tokenize the stored conversation (prompt + generated assistant turn) → `storedTokens`. Store leaf under `storedTokens` with offset = `storedTokens.count`. Mirrors existing `measureHTTPPrefixCacheTokenCount` pattern. |
| 16 | `LanguageModel.prepare()` signature change breaks 10+ VLM overrides, SpeculativeTokenIterator, WiredMemoryUtils | Don't change the existing protocol method. Add a NEW protocol extension method with the extended signature that delegates to the old one by default. Only `LLMModel` and `Qwen35` override the new method. All other conformers unchanged. |
| 17 | `StablePrefixDetector` searches for sentinel tokens in token stream, but existing `AgentEngine.formatRawPrompt()` operates on decoded text — tokenization is context-dependent, token-level search is unreliable | Replaced with **two-probe technique**: tokenize the same messages with two different dummy user messages, find common prefix length. No sentinel encoding needed. Robust for any template. |

### v6 blockers (resolved — consistency fixes)

| # | Blocker | Resolution |
|---|---------|-----------|
| 18 | `CachePartitionKey` drops `toolDefinitionsDigest`/`templateContextDigest` without explaining the routing change vs `HTTPPrefixCacheKey` | Added explicit documentation: digests dropped intentionally because radix tree handles them implicitly (different tools → different tokens → different paths). Existing mismatch diagnostics retained for migration logging. |
| 19 | Design Decision §1 still describes `model.prepare()` returning snapshots, contradicting Task 1.2's `prepareWithCheckpoints()` protocol extension | Removed stale `prepare()` signature change from §1. Single contract: `prepareWithCheckpoints()` as protocol extension. |
| 20 | `planCheckpoints()` documented as covering "stable-prefix + leaf" but logic says leaf NOT planned + test expects "always includes leaf" | One rule: `planCheckpoints()` returns mid-prefill checkpoints only (stable prefix, branch points). Leaf captured post-generation via `storeLeaf()`. Test updated to `planCheckpointsNeverIncludesLeaf`. |
| 21 | `StablePrefixDetector` type comment still says "sentinel technique" despite §1.3 implementing two-probe | All sentinel references replaced with two-probe. |

### v7 blockers (resolved)

| # | Blocker | Resolution |
|---|---------|-----------|
| 22 | Leaf capture omits the normalization-offset alignment step. Current code (`LLMActor.swift:331-357`) trims attention KV when re-tokenized count < actual cache offset because assistant whitespace normalization shortens the stored conversation. Without this, leaf snapshot offset doesn't match its cache state. | Added explicit **offset-alignment step** between re-tokenization and snapshot capture. Attention layers trimmed by `(actualOffset - storedTokens.count)`. Mamba divergence accepted and documented (same as current prototype). |
| 23 | Token-path extraction does `tokens.asArray(Int.self)` but Qwen3.5 VLM produces 2D `[batch, seq]` tensors. Current code (`LLMActor.swift:525-547`) handles both shapes. | Added `extractTokenSequence()` helper that extracts 1D sequence from either 1D or 2D token tensors via `.dim(-1)` for count and `tokens[0, ...]` or `tokens` for flat access. |
| 24 | `StablePrefixDetector` call passes `conversation.systemMessages` but `HTTPPrefixCacheConversation` exposes `systemPrompt: String?`, not a messages collection. | Changed detector to accept `systemPrompt: String?` and construct `[Chat.Message.system(prompt)]` internally. Matches existing conversation shape. |

### v8 blockers (this revision)

| # | Blocker | Resolution |
|---|---------|-----------|
| 25 | Cache-hit suffix slicing uses generic `sliceInput()` but live code (`LLMActor.swift:537-553`) handles 1D/2D differently and drops the mask. | Replaced `sliceInput()` with explicit `sliceSuffix()` spec: 1D → `tokens[offset...]`, 2D → `tokens[0..., offset...]`, mask set to `nil`. Matches existing shape-handling contract. |
| 26 | Accepted Mamba divergence on normalized leaf hits conflicts with the bitwise-logit-equality correctness gate in Phase 2 and benchmark PC8. | Logit-equivalence tests and PC8 benchmark explicitly scoped to **mid-prefill checkpoint restore only** (where no normalization occurs). Leaf hits from normalized conversations excluded — they use the same accepted-divergence contract as the current prototype. New test validates the divergence is bounded. |
| 27 | "Open Questions (Resolve Before Phase 1)" section lists unresolved prerequisites that are actually answerable now. | Section rewritten as "Implementation Notes" with resolved answers for each, or downgraded to "verify during implementation" with concrete verification steps. No open blockers remain. |

### v9 blockers (resolved — paper/repo alignment)

| # | Blocker | Resolution |
|---|---------|-----------|
| 28 | Snapshot size estimates were hand-waved and inconsistent with the actual local Qwen3.5 cache shapes, so memory-budget guidance was not trustworthy. | Replaced them with shape-derived sizing from the local model config: 4K unquantized snapshot ≈ 202 MiB, 8K ≈ 330 MiB, 16K ≈ 586 MiB. |
| 29 | Phase 2 speculative admission allows exact-path extensions and multiple branch-point candidates, but Marconi admits at most one speculative intermediate checkpoint per sequence. | Task 2.1 now matches speculative insertion: create at most one `.branchPoint` candidate, only when insertion would split an existing edge and create a new intermediate radix node. Exact-path extensions rely on the leaf checkpoint only. |
| 30 | Phase 2 eviction is underspecified versus Marconi: no recency transform, no min-max normalization, no parent-relative FLOP delta, no eligible-node filter, no single-child collapse rule. | Task 2.3 now specifies the full utility-scored eviction policy: candidates are snapshot nodes with `<= 1` child, utility uses normalized recency plus `alpha *` normalized FLOP efficiency, and single-child nodes collapse after snapshot eviction to preserve radix compression. |
| 31 | Task 2.4 replaced Marconi's adaptive `alpha` tuning with a manual sweep, so the plan no longer aligned with the paper/reference repo. | Restored a paper/repo-aligned bootstrap tuner: start with `alpha = 0`, wait for first eviction, use a bootstrap window of `5x` the first-eviction request count (repo default within the paper's 5-15x range), then grid-search `alpha in {0.0, 0.1, ..., 2.0}` by replaying the window. |

**API:** `Memory.clearCache()` (not `MLX.GPU.clearCache()`).

---

## Goal

Replace `HTTPPrefixCacheSpike` with a token-level radix tree supporting full hybrid cache snapshots captured during prefill. Two checkpoint types: stable-prefix boundary (system + tools, shared across conversations) and conversation leaf (shared within a conversation's tool loop). Trees partitioned by `(modelID, kvBits, kvGroupSize)` to prevent cross-config contamination.

**Target:** 98% cache hit rate on agentic coding workloads (per mlx-lm #1006 benchmarks on Qwen3.5), ~3x TTFT reduction on steady-state turns.

---

## Architecture Overview

```
   Wire Request
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Normalization Pipeline (HTTPPrefixCacheMessage)      │
│  - Assistant content: whitespace trim                 │
│  - Reasoning: whitespace trim, nil if empty           │
│  - Tool args: JSON sort + canonicalize                │
│  - System prompt: whitespace trim                     │
└──────────┬───────────────────────────────────────────┘
           │ normalized conversation (historyMessages)
           ▼
┌──────────────────────────────────────────────────────┐
│  Tokenizer (processor.prepare → LMInput.text.tokens)  │
│  Jinja chat template renders full message sequence     │
└──────────┬───────────────────────────────────────────┘
           │ token sequence: [Int]
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PrefixCacheManager                             │
│                                                                  │
│  ┌──────────────────────┐   ┌──────────────────────────────┐    │
│  │   TokenRadixTree      │   │  EvictionPolicy               │    │
│  │                        │   │  Phase 1: type-based LRU      │    │
│  │  Node {                │   │  Phase 2: Marconi utility     │    │
│  │    edgeTokens          │   │  over eligible nodes          │    │
│  │    children            │   │  score = norm(R)+a*norm(F/B)  │    │
│  │    snapshot?           │   └──────────────────────────────┘    │
│  │    checkpointType      │                                      │
│  │    tokenOffset         │   ┌──────────────────────────────┐    │
│  │  }                     │   │  CheckpointPrefill            │    │
│  │                        │   │  (modified prepare() loop)    │    │
│  │  findBestSnapshot()    │   │                                │    │
│  │  insertPath()          │   │  Adjusts chunk size to land   │    │
│  │  storeSnapshot()       │   │  on checkpoint offsets.        │    │
│  └──────────────────────┘   │  Captures snapshot mid-prefill. │    │
│                              └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions (with code references)

### 1. Snapshots captured DURING prefill, not post-hoc — through TokenIterator

MambaCache state is updated in-place during the forward pass (`Qwen35.swift` GatedDeltaNet lines 664-666, 697-699). Once prefill advances past offset K, the Mamba state at offset K is overwritten and irrecoverable. Therefore:

- **Before prefill:** Determine checkpoint offsets (stable prefix = system + tools boundary, branch points)
- **During prefill:** When the chunking loop reaches a checkpoint offset, snapshot ALL layer states
- **After prefill:** Store captured snapshots in the radix tree

**Critical: checkpoint capture flows through TokenIterator, not around it.** `TokenIterator.init()` owns the prefill lifecycle — it calls `model.prepare()` inside its initializer (`Evaluate.swift:607-609`). Calling `model.prepare()` separately would either double-prefill or require a non-trivial refactor of TokenIterator.

**The correct integration:**
1. Checkpoint offsets are added to `GenerateParameters` (new fields: `checkpointAtOffsets: Set<Int>`, `checkpointBaseOffset: Int`)
2. `TokenIterator.prepare()` calls `model.prepareWithCheckpoints()` — a NEW protocol extension method (see Task 1.2)
3. `prepareWithCheckpoints()` returns `(PrepareResult, [HybridCacheSnapshot])`
4. TokenIterator stores captured snapshots in a new public property: `capturedSnapshots`
5. After creating `TokenIterator`, the caller (LLMActor) reads `iterator.capturedSnapshots`

This preserves the TokenIterator contract: one prefill call inside init, cache owned by iterator. The existing `LanguageModel.prepare()` protocol method is **untouched** — all existing conformers, SpeculativeTokenIterator, and WiredMemoryUtils continue calling it. The only changes are:
- `GenerateParameters` gains `checkpointAtOffsets` and `checkpointBaseOffset` fields (defaults: `[]`, `0`)
- New `prepareWithCheckpoints()` protocol extension method (default delegates to `prepare()`)
- `TokenIterator.prepare()` calls the new method instead of the base one
- `TokenIterator` gains a `capturedSnapshots` property populated during init

**Leaf snapshot:** Captured AFTER generation from the final cache via `FinalizedKVCacheHandle.takeFinalCache()`. The leaf represents the terminal state — no rollback needed, so post-hoc capture is correct for this type only.

### 2. Variable chunk sizes to land on checkpoint offsets, including in the tail

The prefill loop advances in `prefillStepSize` chunks (default 256). Checkpoint offsets are **absolute** (in full-prompt coordinates). On a cache hit, the suffix prefill starts at offset 0 of the sliced input, but checkpoints are at positions like 4000 (absolute). The `checkpointBaseOffset` parameter bridges this gap.

```swift
func prepare(
    _ input: LMInput,
    cache: [any KVCache],
    windowSize: Int?,
    checkpointAtOffsets: Set<Int>,
    checkpointBaseOffset: Int = 0       // absolute offset of input[0] in the full prompt
) throws -> (PrepareResult, [HybridCacheSnapshot]) {
    let prefillStepSize = windowSize ?? 512
    var y = input.text
    var currentOffset = 0               // relative to THIS input (starts at 0)
    var snapshots: [HybridCacheSnapshot] = []
    
    // Convert absolute checkpoint offsets to relative (within this input)
    let relativeCheckpoints = checkpointAtOffsets
        .map { $0 - checkpointBaseOffset }
        .filter { $0 > 0 }             // only checkpoints within this input
        .sorted()

    while y.tokens.size > prefillStepSize {
        var chunkSize = prefillStepSize
        
        // Shrink to land on the nearest checkpoint within this chunk
        if let next = relativeCheckpoints.first(where: {
            $0 > currentOffset && $0 < currentOffset + chunkSize
        }) {
            chunkSize = next - currentOffset
        }
        
        let input = y[.newAxis, ..<chunkSize]
        _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)
        currentOffset += chunkSize
        
        if relativeCheckpoints.contains(currentOffset) {
            // Store snapshot with ABSOLUTE offset for correct radix tree placement
            snapshots.append(HybridCacheSnapshot.capture(
                cache: cache, offset: checkpointBaseOffset + currentOffset, type: ...
            ))
        }
        
        y = y[chunkSize...]
        Memory.clearCache()
    }

    // *** TAIL DRAIN: capture checkpoints in the final remainder ***
    // Without this, checkpoints in the last <= prefillStepSize tokens are missed.
    while let nextCP = relativeCheckpoints.first(where: {
        $0 > currentOffset && $0 < currentOffset + y.tokens.size
    }) {
        let chunkSize = nextCP - currentOffset
        guard chunkSize > 0 else { break }
        let input = y[.newAxis, ..<chunkSize]
        _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
        eval(cache)
        currentOffset += chunkSize
        snapshots.append(HybridCacheSnapshot.capture(
            cache: cache, offset: checkpointBaseOffset + currentOffset, type: ...
        ))
        y = y[chunkSize...]
        Memory.clearCache()
    }

    return (.tokens(y), snapshots)
}
```

**Offset rebasing example:**
- Full prompt = 8000 tokens. Stable-prefix checkpoint at absolute offset 4000.
- Cache hit: snapshot restored at offset 3000. Suffix = tokens [3000..8000].
- `checkpointBaseOffset = 3000`. Input has 5000 tokens (relative 0..5000).
- Relative checkpoint = 4000 - 3000 = 1000.
- Loop captures snapshot at relative 1000, stores with absolute offset 3000 + 1000 = 4000. Correct.

**Key invariant:** All checkpoint offsets strictly before `checkpointBaseOffset + currentOffset + y.tokens.size` are captured. Snapshots are stored with absolute offsets for correct radix tree placement. `y` contains the final remainder for `TokenIterator.step()`.

Chunk size bounded: `1 ≤ chunkSize ≤ prefillStepSize`. Small chunks near checkpoints are fine.

### 3. Restore mirrors loadPromptCache() exactly

`HybridCacheSnapshot` stores per-layer `(className, state, metaState)` — the same triple that `savePromptCache()`/`loadPromptCache()` uses (`KVCache.swift:1244-1388`).

Restore creates the correct class based on `className`:

| className | Constructor | State | MetaState |
|-----------|-------------|-------|-----------|
| `"KVCache"` / `"KVCacheSimple"` | `KVCacheSimple()` | `[keys, values]` — offset auto-set from `keys.dim(2)` on state setter (`KVCache.swift:391`) | `[""]` (empty, ignored) |
| `"QuantizedKVCache"` | `QuantizedKVCache(groupSize: parsed, bits: parsed)` | 4 or 6 arrays (wq, scales, [biases]) | `[step, offset, groupSize, bits]` — parse `[2]` and `[3]` for constructor |
| `"RotatingKVCache"` | `RotatingKVCache(maxSize: parsed)` | `[keys, values]` | `[keep, maxCacheSize, step, offset, idx]` — parse `[1]` for constructor, setter restores all 5 |
| `"ChunkedKVCache"` | `ChunkedKVCache()` | `[keys, values]` (inherited from KVCacheSimple) | `[chunkSize, startPosition]` |
| `"MambaCache"` | `MambaCache()` | `[convState, gatedDeltaState]` | `[""]` (empty) |

**QuantizedKVCache note:** The current `loadPromptCache()` creates `QuantizedKVCache()` with defaults (groupSize=64, bits=8) — which happens to match Tesseract's runtime config. Our snapshot restore is stricter: parse groupSize/bits from metaState and pass to constructor. This prevents silent corruption if config ever changes.

**Dynamic quantization note:** Qwen3.5 starts with `KVCacheSimple` for attention layers (`Qwen35.swift:585`), then `maybeQuantizeKVCache()` converts to `QuantizedKVCache` after `quantizedKVStart` tokens (`KVCache.swift:1637-1661`). Early checkpoints (e.g., system boundary at offset 100) will have `KVCacheSimple`; later checkpoints will have `QuantizedKVCache`. The className per layer in the snapshot handles this transparently.

### 4. Two checkpoint types: stable prefix + conversation leaf

Real HTTP conversations are alternating sequences (ChatML format from `Chat.swift:47-100`):
```
<|im_start|>system\n{system}<|im_end|>\n
[tool definitions rendered by Jinja from `tools` parameter — Tokenizer.swift:16-20]
<|im_start|>user\n{user_1}<|im_end|>\n
<|im_start|>assistant\n{assistant_1_with_tool_calls}<|im_end|>\n
<|im_start|>tool\n{tool_result}<|im_end|>\n
<|im_start|>user\n{user_2}<|im_end|>\n
<|im_start|>assistant\n
```

Tool definitions are NOT inside the system message — they're passed as a separate `tools` parameter to `applyChatTemplate(messages:tools:)` (`Tokenizer.swift:16-20`) and rendered by the Jinja template as their own section. The high-value shared prefix is **system prompt + tool definitions**, not system prompt alone.

**Type A — Stable-prefix boundary:** The token offset where system message(s) + tool definitions end. This is the longest prefix shared across ALL requests that use the same system prompt AND the same tools. Detectable via two-probe technique (see Task 1.3). Highest eviction priority in **Phase 1**.

When tools change between requests, the token sequences diverge at the tool-definition section. The radix tree handles this correctly — different tokens = different paths = separate snapshots. The stable-prefix checkpoint only applies when both system prompt and tools match.

**Type B — Conversation leaf:** The full token count after generation completes. Equivalent to current `HTTPPrefixCacheSpike` entry — reusable when the next request extends the same conversation prefix. Shared within a conversation's tool loop.

Phase 2 adds **Type C — Branch-point checkpoint:** captured speculatively when the radix tree detects divergence from stored paths.

### 5. Normalization BEFORE tokenization

Pipeline from `CompletionHandler.swift:109-119`:
1. `sessionReplayStore.repair()` — recover dropped messages
2. `HTTPPrefixCacheMessage` construction — normalize assistant whitespace, canonicalize tool args JSON
3. `historyMessages` — reconstruct `[Chat.Message]` from normalized conversation (`HTTPPrefixCacheSpike.swift:263-283`)
4. `processor.prepare(input: UserInput(chat: historyMessages, tools: toolSpecs))` — tokenize

Normalization ensures:
- Assistant content `"\n\nNow let me read…\n\n\n\n"` and echoed `"\n\nNow let me read…"` → same normalized form → same tokens
- Tool args `{"path":"main.swift","limit":100}` vs `{ "path": "main.swift", "limit": 100 }` → same canonical JSON → same tokens
- Whitespace-only content `"\n\n"` and empty `""` → same normalized form → same tokens

Without normalization, identical-intent requests produce different tokens → systematic cache misses (regressions documented in `HTTPPrefixCacheSpikeTests.swift:331-653`).

---

## Constraints

- **Qwen3.5 hybrid layout:** 24 MambaCache + 8 KVCacheSimple layers (3:1, `fullAttentionInterval=4`). Every 4th layer (3, 7, 11, ...) is attention.
- **MambaCache state:** `[0]` conv state `[B, 3, 14336]`, `[1]` gated delta state `[B, 64, 128, 192]`. Fixed-size, non-trimmable.
- **KVCacheSimple.state setter** auto-sets offset from `keys.dim(2)` (`KVCache.swift:391`). No need to store offset separately.
- **QuantizedKVCache** needs groupSize/bits in constructor before state injection. Parse from metaState.
- **RotatingKVCache** needs maxSize in constructor. Parse from metaState.
- **Token tensor shape:** Qwen3.5 VLM produces 2D `[batch, seq]` token tensors; LLM-only produces 1D `[seq]`. All radix tree operations need flat `[Int]` sequences. Use `extractTokenSequence()` helper: `ndim == 1 → tokens.asArray(Int.self)`, `ndim >= 2 → tokens[0].asArray(Int.self)`. Token count always via `.dim(-1)`. Mirrors the existing shape handling in `LLMActor.swift:525-547`.
- **Normalization-offset alignment:** After generation, the normalized stored conversation may be shorter than the actual cache offset (assistant whitespace trimming). Attention layers must be trimmed by the difference before leaf snapshot capture. Mamba layers are non-trimmable — accepted divergence matching current prototype (`LLMActor.swift:331-357`).
- **No concurrent writers.** `@MainActor` isolation sufficient.
- **Memory budget:** ≤20 GB unified memory. Model ~10 GB. Cache budget ~8-12 GB.

---

## Phase 0 — Prerequisites (P0 fixes from research doc)

### Task 0.1: Add `Memory.clearCache()` between prefill chunks

**Files:**
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift:22-37` — add `Memory.clearCache()` after `eval(cache)` in chunking loop
- `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift:1104-1165` — add chunking to VLM `prepare()` (currently ignores `windowSize`, processes entire input in one shot). Vision/multimodal path: pre-compute `inputEmbeddings` for the full vision prefix, then chunk only the text-token portion that follows.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:645-663` — add `Memory.clearCache()` after `prepare()` returns in `TokenIterator.prepare`

**Test 0.1:** Manual — build Release, replay 19K-token prefill, confirm peak memory drops ~40%, no OOM.

### Task 0.2: Verify Memory.cacheLimit

Grep all `Memory.cacheLimit` writes in `LLMActor.swift`. Defaults: `cacheLimitMB = 2048` (line 62), applied at lines 127, 192. Confirm no override.

### Task 0.3: Audit setWiredLimit

Grep `Vendor/mlx-swift-lm/` for `setWiredLimit`, `set_wired_limit`. Lower to 50% or remove if present.

---

## Phase 1 — Hybrid-Safe Radix Tree with Checkpoint-During-Prefill

Replaces `HTTPPrefixCacheSpike`. Config-partitioned token-level radix tree with full hybrid snapshots captured DURING prefill (via TokenIterator → model.prepare()) at stable-prefix boundary (system + tools) and conversation leaf.

### Task 1.1: `HybridCacheSnapshot` — correct multi-type restore

**New file:** `tesseract/Features/Server/HybridCacheSnapshot.swift`

```swift
/// Full snapshot of all per-layer cache state at a specific token offset.
/// Mirrors the savePromptCache/loadPromptCache serialization contract.
/// Immutable after creation.
struct HybridCacheSnapshot: Sendable {
    let tokenOffset: Int

    /// Per-layer cache state. Mirrors savePromptCache's serialization format.
    struct LayerState: Sendable {
        let className: String          // "KVCache", "QuantizedKVCache", "RotatingKVCache",
                                       // "ChunkedKVCache", "MambaCache", "ArraysCache"
        let state: [MLXArray]          // cache.state (deep-copied on capture)
        let metaState: [String]        // cache.metaState
    }

    let layers: [LayerState]
    let checkpointType: CheckpointType
    let memoryBytes: Int               // pre-computed for eviction
    let createdAt: ContinuousClock.Instant

    enum CheckpointType: Comparable {
        case system         // stable-prefix reuse; highest priority in Phase 1 only
        case leaf           // standard conversation-prefix reuse
        case branchPoint    // Phase 2: speculative Marconi checkpoint
    }

    /// Capture from live cache during prefill. Deep-copies all state arrays.
    static func capture(
        cache: [any KVCache],
        offset: Int,
        type: CheckpointType
    ) -> HybridCacheSnapshot

    /// Restore into a live cache array. Creates correct class per layer.
    /// Mirrors loadPromptCache() reconstruction logic from KVCache.swift:1340-1378.
    func restore(kvBitsHint: Int?, kvGroupSizeHint: Int?) -> [any KVCache]
}
```

**`capture()` implementation:**
1. For each layer in the cache array:
   - Determine className via type check (ChunkedKVCache before KVCacheSimple — subclass check order matters, matching `savePromptCache` at `KVCache.swift:1252-1271`)
   - Deep-copy state: `layer.state.map { $0[.ellipsis] }` (new backing arrays)
   - Copy metaState: `layer.metaState` (strings, immutable)
2. Sum tensor byte sizes for `memoryBytes`

**`restore()` implementation:**
1. For each `LayerState`:
   - Create the correct class:
     - `"KVCache"` / `"KVCacheSimple"` → `KVCacheSimple()` — offset auto-set from `keys.dim(2)` on state setter
     - `"QuantizedKVCache"` → parse `metaState[2]` as groupSize, `metaState[3]` as bits → `QuantizedKVCache(groupSize:bits:)`
     - `"RotatingKVCache"` → parse `metaState[1]` as maxSize → `RotatingKVCache(maxSize:)` — metaState setter restores keep, step, offset, idx
     - `"MambaCache"` → `MambaCache()` — state setter injects [convState, gatedDeltaState]
     - `"ChunkedKVCache"` → `ChunkedKVCache()` — metaState setter restores chunkSize, startPosition
   - Inject state: `cache.state = layerState.state`
   - Inject metaState: `cache.metaState = layerState.metaState`
2. Return `[any KVCache]`

**Tests 1.1 — `HybridCacheSnapshotTests.swift`:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `captureStoresAllLayerStates` | 32-layer Qwen3.5 hybrid → 24 MambaCache + 8 attention layer states captured |
| 2 | `captureDeepCopiesArrays` | Mutating source cache after capture doesn't affect snapshot |
| 3 | `captureRecordsCorrectClassName` | MambaCache layers → `"MambaCache"`, KVCacheSimple → `"KVCache"`, QuantizedKVCache → `"QuantizedKVCache"` |
| 4 | `captureWithMixedCacheTypes` | After dynamic quantization: some layers KVCacheSimple, some QuantizedKVCache → both captured correctly |
| 5 | `memoryBytesMatchesSumOfTensorSizes` | Sum of all state array nbytes == memoryBytes |
| 6 | `restoreCreatesKVCacheSimple` | className `"KVCache"` → KVCacheSimple with offset auto-set from keys.dim(2) |
| 7 | `restoreCreatesQuantizedKVCache` | className `"QuantizedKVCache"` + metaState `["0", "512", "64", "8"]` → QuantizedKVCache(groupSize:64, bits:8), offset=512 |
| 8 | `restoreCreatesRotatingKVCache` | className `"RotatingKVCache"` + metaState with maxSize → RotatingKVCache(maxSize:), all 5 properties restored |
| 9 | `restoreCreatesMambaCache` | className `"MambaCache"` → MambaCache with conv+recurrent state injected |
| 10 | `restoreCreatesChunkedKVCache` | className `"ChunkedKVCache"` → ChunkedKVCache with chunkSize+startPosition from metaState |
| 11 | `roundTripCaptureRestorePreservesState` | capture → restore → compare state arrays: bitwise equal |
| 12 | `restoredCacheIsIsolatedFromSnapshot` | Mutating restored cache doesn't affect snapshot |
| 13 | `restoredKVCacheSimpleOffsetMatchesKeys` | After restore, `cache.offset == cache.state[0].dim(2)` (auto-set invariant) |
| 14 | `restoredQuantizedKVCacheHasCorrectBits` | After restore, `cache.bits == 8`, `cache.groupSize == 64` (parsed from metaState, not default) |

### Task 1.2: Checkpoint-capable `prepare()` via protocol extension, threaded through TokenIterator

**Files:**
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/LanguageModel.swift` — add protocol extension with default impl
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModel.swift` — override extended method with checkpoint logic + tail drain
- `Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` — override extended method with chunking + capture
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` — three changes:
  1. `GenerateParameters` gains `checkpointAtOffsets: Set<Int>` (default `[]`) and `checkpointBaseOffset: Int` (default `0`)
  2. `TokenIterator.prepare()` calls the extended method instead of the base one
  3. `TokenIterator` gains `public private(set) var capturedSnapshots: [HybridCacheSnapshot]`

**Non-breaking protocol extension strategy:**

The existing `LanguageModel` protocol method (`LanguageModel.swift:160`) is:
```swift
func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult
```

This is called by 10+ VLM model overrides (Pixtral, Mistral3, Paligemma, Qwen2VL, etc.), `SpeculativeTokenIterator` (`Evaluate.swift:826,839`), and `WiredMemoryUtils.prefillOnly` (`WiredMemoryUtils.swift:98`). **Changing this signature breaks all of them.**

Instead, add a NEW method via protocol extension:

```swift
// LanguageModel.swift — protocol extension, NOT protocol requirement
extension LanguageModel {
    /// Extended prepare with checkpoint capture support.
    /// Default implementation delegates to the base prepare() and returns no snapshots.
    /// Only LLMModel and Qwen35 override this.
    func prepareWithCheckpoints(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize: Int?,
        checkpointAtOffsets: Set<Int>,
        checkpointBaseOffset: Int
    ) throws -> (PrepareResult, [HybridCacheSnapshot]) {
        let result = try prepare(input, cache: cache, windowSize: windowSize)
        return (result, [])
    }
}
```

**What changes, what doesn't:**

| Component | Changes? | Why |
|-----------|----------|-----|
| `LanguageModel` protocol definition | **No** | Base `prepare()` signature untouched |
| `LLMModel.swift` | **Yes** | Overrides `prepareWithCheckpoints()` with chunking + checkpoint + tail drain |
| `Qwen35.swift` | **Yes** | Overrides `prepareWithCheckpoints()` with VLM chunking + capture |
| `TokenIterator.prepare()` | **Yes** | Calls `prepareWithCheckpoints()` instead of `prepare()` |
| `GenerateParameters` | **Yes** | Gains `checkpointAtOffsets` + `checkpointBaseOffset` fields |
| Pixtral, Mistral3, Paligemma, etc. | **No** | Inherit default extension (delegates to their existing `prepare()`) |
| `SpeculativeTokenIterator` | **No** | Still calls base `prepare()` directly |
| `WiredMemoryUtils.prefillOnly` | **No** | Still calls base `prepare()` directly |

**Data flow through TokenIterator:**

```
LLMActor                          TokenIterator                    model.prepareWithCheckpoints()
   │                                  │                                  │
   │  params.checkpointAtOffsets={4K} │                                  │
   │  params.checkpointBaseOffset=3K  │  (set to snapshot.tokenOffset    │
   │                                  │   on cache hit, 0 on miss)       │
   │──────►  TokenIterator(input,     │                                  │
   │           model, cache, params)  │                                  │
   │                                  │──► self.cache = cache ?? new     │
   │                                  │──► prepareWithCheckpoints(       │
   │                                  │      input, cache, windowSize,   │
   │                                  │      {4K}, baseOffset=3K)        │
   │                                  │                 │──► rebase: 4K-3K=1K │
   │                                  │                 │──► chunk loop       │
   │                                  │                 │    + tail drain     │
   │                                  │                 │──► snapshots[4K]    │
   │                                  │◄── (.tokens(y), snapshots)       │
   │                                  │──► self.capturedSnapshots = snaps│
   │                                  │──► step(previous: y) // 1st tok  │
   │◄────── iterator ready            │                                  │
   │  read iterator.capturedSnapshots │                                  │
```

**Modified `TokenIterator.prepare()` (`Evaluate.swift:645-663`):**

```swift
mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
    processor?.prompt(input.text.tokens)
    
    // Call EXTENDED method with checkpoint offsets from parameters
    let (result, snapshots) = try model.prepareWithCheckpoints(
        input, cache: cache, windowSize: windowSize,
        checkpointAtOffsets: checkpointAtOffsets,
        checkpointBaseOffset: checkpointBaseOffset
    )
    self.capturedSnapshots = snapshots
    
    switch result {
    case .tokens(let tokens):
        y = tokens
        let token = step(previous: y)
        y = .init(tokens: token)
        asyncEval(y.tokens)
    case .logits(let result):
        y = .init(tokens: convertToToken(logits: result.logits))
        asyncEval(y.tokens)
    }
}
```

**Chunking loop with checkpoint + tail drain:** See §2 above for the full loop code. Key addition from v5: `checkpointBaseOffset` converts between relative (input) and absolute (full-prompt) coordinates.

**Qwen35 VLM override:** Must also add chunking (Phase 0) and checkpoint capture. Vision path pre-computes `inputEmbeddings` for the vision prefix, then text-token chunking operates on the remainder.

**Tests 1.2 — checkpoint capture:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `emptyCheckpointOffsetsReturnsNoSnapshots` | `checkpointAtOffsets: []` → standard prepare, empty snapshots |
| 2 | `snapshotCapturedAtAlignedOffset` | prefillStepSize=256, checkpoint at 512 → snapshot at 512 |
| 3 | `snapshotCapturedAtMisalignedOffset` | prefillStepSize=256, checkpoint at 300 → chunk splits: 256+44, snapshot at 300 |
| 4 | `multipleCheckpointsInSinglePrefill` | checkpoints at {300, 600, 1000} → three snapshots |
| 5 | `checkpointBeyondInputSizeIgnored` | checkpoint at 10000 on 5000-token input → no snapshot |
| 6 | `chunkSizeNeverExceedsPrefillStepSize` | All chunks ≤ prefillStepSize |
| 7 | `chunkSizeNeverZero` | Checkpoint at offset 0 → no zero-length chunk |
| 8 | `checkpointDoesNotAlterFinalPrepareResult` | PrepareResult.tokens identical with/without checkpoints |
| 9 | `clearCacheCalledBetweenAllChunks` | Memory.clearCache() after each chunk (mock or log) |
| 10 | `checkpointInTailCaptured` | prefillStepSize=256, input=400, checkpoint at abs 300 → captured in tail drain |
| 11 | `checkpointAtLastTokenBeforeTailCaptured` | prefillStepSize=256, input=500, checkpoint at abs 490 → captured in tail drain |
| 12 | `tokenIteratorExposesSnapshots` | After `TokenIterator(params: {checkpoints: {100}})`, `iterator.capturedSnapshots` has 1 entry |
| 13 | `tokenIteratorWithNoCheckpointsHasEmptySnapshots` | Default params → empty `capturedSnapshots` |
| 14 | `checkpointRebasedOnCacheHit` | baseOffset=3000, absolute checkpoint at 4000 → fires at relative 1000, stored at absolute 4000 |
| 15 | `checkpointBeforeBaseOffsetIgnored` | baseOffset=3000, checkpoint at 2000 → filtered out (already covered by snapshot) |
| 16 | `defaultExtensionDelegatesToBasePrepare` | Non-LLM model (e.g., Pixtral) → `prepareWithCheckpoints()` delegates to `prepare()`, returns empty snapshots |
| 17 | `specIteratorStillCallsBasePrepare` | SpeculativeTokenIterator unaffected (still calls base `prepare()` at Evaluate.swift:826,839) |

### Task 1.3: Stable-prefix boundary detection (system + tools)

**New file:** `tesseract/Features/Server/StablePrefixDetector.swift`

The high-value shared prefix is system prompt + tool definitions — NOT system alone. Tools are passed as a separate `tools` parameter to `applyChatTemplate(messages:tools:)` (`Tokenizer.swift:16-20`) and rendered by the Jinja template in their own section after the system message. Tokenizing `[system_only]` misses the tool section and produces a shorter (incorrect) boundary.

```swift
/// Detects the stable-prefix boundary: the token offset where
/// system message(s) + tool definitions end in the tokenized prompt.
/// Uses the two-probe technique (NOT sentinel token search — see implementation below).
/// (AgentEngine.swift:198-223).
@MainActor
struct StablePrefixDetector {

    /// Returns the token offset where the stable prefix (system + tools) ends.
    /// nil if detection fails or no system prompt.
    ///
    /// Accepts systemPrompt: String? to match HTTPPrefixCacheConversation's API
    /// (which exposes .systemPrompt, not a .systemMessages collection).
    /// Constructs [Chat.Message.system(prompt)] internally for the two probes.
    static func detect(
        systemPrompt: String?,
        toolSpecs: [ToolSpec]?,
        fullTokens: [Int],
        tokenizer: /* Tokenizer type */
    ) async throws -> Int?
}
```

**Implementation (two-probe technique):**

The existing `AgentEngine.formatRawPrompt()` (`AgentEngine.swift:198-223`) uses a sentinel but operates on **decoded text**, not token streams — it calls `formatRawPromptWithCount()`, gets `(text: String, tokenCount: Int)`, and finds the sentinel string in the decoded text via `text.range(of: placeholder)`. Token-level sentinel search is unreliable because tokenization is context-dependent (the same string tokenizes differently depending on surrounding template markup).

Instead, use a **two-probe technique** that avoids encoding issues entirely:

1. If `systemPrompt` is nil, return nil
2. Let `sysMsg = [Chat.Message.system(systemPrompt!)]`
3. Tokenize probe A: `sysMsg + [.user("__PROBE_A__")]` WITH `toolSpecs` → `tokensA`
4. Tokenize probe B: `sysMsg + [.user("__PROBE_B__")]` WITH `toolSpecs` → `tokensB`
5. Find common prefix: `commonLength = zip(tokensA, tokensB).prefix(while: ==).count`
6. Verify against full tokens: `fullTokens[0..<commonLength] == tokensA[0..<commonLength]`
7. Return `commonLength`

**Why two-probe works:**
- Both probes share the same system + tools prefix (rendered identically by the template)
- They diverge at the user message content ("PROBE_A" vs "PROBE_B")
- The common prefix length = exactly where system + tools end and user content begins
- No sentinel encoding needed — we compare token-by-token between two known-different inputs
- Robust for ANY template format (ChatML, Llama, custom) — no template-specific assumptions

**Edge cases:**
- Template interleaves tools after user message → probes diverge earlier than expected → common prefix is shorter → still correct (just misses tool tokens in stable prefix, detected by step 5)
- No tools → stable prefix = system-only boundary (shorter, still valid)
- Probes produce identical tokens (e.g., template ignores user content) → commonLength = min(len(A), len(B)) → step 5 catches mismatch if wrong

**Tests 1.3:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `detectsSystemPlusToolsBoundary` | System + 5 tools → two-probe common prefix covers system + tool tokens |
| 2 | `noSystemMessageReturnsNil` | Empty system → nil |
| 3 | `stablePrefixIsPrefixOfFullSequence` | Verification passes: probeA[0..<boundary] == fullTokens[0..<boundary] |
| 4 | `noToolsDetectsSystemOnlyBoundary` | System, no tools → offset covers system tokens only |
| 5 | `longSystemPrompt8KTokens` | 8K-token system + tools → correct boundary |
| 6 | `differentToolsProduceDifferentBoundaries` | Same system, different tools → different offsets |
| 7 | `twoProbesDivergeAtUserContent` | probeA and probeB share exact prefix, diverge at user message |
| 8 | `probeRobustToSpecialCharsInSystem` | System prompt with quotes, newlines, template-like content → still correct |

### Task 1.4: `TokenRadixTree`

**New file:** `tesseract/Features/Server/TokenRadixTree.swift`

```swift
final class RadixTreeNode {
    var edgeTokens: [Int]
    var children: [Int: RadixTreeNode]          // keyed by first token of child edge
    var snapshot: HybridCacheSnapshot?           // nil for nodes without checkpoint
    var tokenOffset: Int                         // cumulative tokens from root
    var lastAccessTime: ContinuousClock.Instant
    weak var parent: RadixTreeNode?
}

@MainActor
final class TokenRadixTree {
    private let root: RadixTreeNode
    private(set) var nodeCount: Int = 1
    private(set) var totalSnapshotBytes: Int = 0

    /// Find the deepest node with a snapshot whose offset ≤ the shared prefix length.
    ///
    /// Walk the tree matching tokens. Track the deepest snapshot-bearing node.
    /// When walk diverges or ends, return tracked node.
    /// Caller gets (snapshot, snapshotOffset). Tokens [snapshotOffset..] must be re-prefilled.
    func findBestSnapshot(tokens: [Int]) -> (node: RadixTreeNode, sharedPrefixLength: Int)?

    /// Insert path for the token sequence. Does NOT store a snapshot.
    func insertPath(tokens: [Int])

    /// Attach a snapshot to the node at the given offset.
    /// Node must already exist (via insertPath).
    func storeSnapshot(_ snapshot: HybridCacheSnapshot, atOffset offset: Int)

    /// Remove a node's snapshot. Node structure stays.
    func evictSnapshot(node: RadixTreeNode)

    /// Remove a leaf node and clean up empty ancestors.
    func evictNode(node: RadixTreeNode)

    /// Snapshot-bearing nodes eligible for Marconi eviction scoring.
    /// Candidate rule: node has a snapshot AND childCount <= 1.
    func eligibleEvictionNodes() -> [RadixTreeNode]

    /// Collapse a snapshot-less node with exactly one child to preserve radix compression.
    /// Concatenates the node's edgeTokens into the child edge and re-links parent/child.
    func collapseSingleChildNode(_ node: RadixTreeNode)
}
```

`findBestSnapshot()` behavior:
- Tokens `[0..5000]` match, snapshot at offset 4000: → return snapshot at 4000. Re-prefill 1000 tokens.
- Tokens `[0..5000]` match, snapshot at offset 5000: → return snapshot at 5000. Re-prefill 0 tokens.
- Tokens `[0..3000]` match, only snapshot at offset 4000: → 4000 > 3000, NOT usable. Return nil (or shallower snapshot).

**Access-time rule:** On lookup hit, update `lastAccessTime` on the returned node only. Do **not** touch ancestors. This matches Marconi's recency policy and avoids making shared ancestors look artificially hot.

**Collapse rule:** If eviction removes a snapshot from a node that now has no snapshot and exactly one child, collapse the node into the child so the radix tree stays compressed. In this design, collapse changes only path structure; the child snapshot already contains the full cache state for its own `tokenOffset`.

**Tests 1.4 — `TokenRadixTreeTests.swift`:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `emptyTreeReturnsNil` | Fresh tree → nil |
| 2 | `insertAndExactMatch` | Insert [1..100], store snapshot at 100, lookup [1..100] → snapshot at 100 |
| 3 | `snapshotAtPrefixReturnedOnDivergence` | Snapshot at 50, lookup [1..50, 200..210] → snapshot at 50 |
| 4 | `deeperSnapshotPreferred` | Snapshots at 50 and 80, lookup [1..100] → snapshot at 80 |
| 5 | `snapshotBeyondSharedPrefixNotReturned` | Snapshot at 100, shared prefix only 80 → returns nil or shallower |
| 6 | `compressedEdges` | Insert [1..5] → single edge, not 5 nodes |
| 7 | `splitEdgeOnBranch` | Insert [1,2,3,4] then [1,2,5,6] → split at [1,2] |
| 8 | `evictSnapshotKeepsNode` | Evict snapshot → node stays, just no snapshot |
| 9 | `evictLeafCleansAncestors` | Evict only leaf → empty ancestors removed |
| 10 | `evictLeafPreservesSiblings` | Evict one sibling → other sibling and shared ancestor preserved |
| 11 | `totalSnapshotBytesAccurate` | After insert/evict, counter matches actual |
| 12 | `findBestSnapshotOnlyUpdatesReturnedNode` | Lookup updates returned node's timestamp, not ancestors |
| 13 | `50KTokenSequence` | Large insert/lookup works |
| 14 | `eligibleEvictionNodesExcludeMultiChildNodes` | Shared-prefix nodes with 2+ children are protected from Phase 2 scoring |
| 15 | `collapseSingleChildNodeMergesEdges` | Snapshot-less intermediate node with one child collapses cleanly |

### Task 1.5: `PrefixCacheManager` (public API)

**New file:** `tesseract/Features/Server/PrefixCacheManager.swift`

```swift
/// Partition key for isolating radix trees by runtime configuration.
///
/// INTENTIONAL ROUTING CHANGE from HTTPPrefixCacheKey:
/// The existing HTTPPrefixCacheKey (HTTPPrefixCacheSpike.swift:10-15) has 5 fields:
///   modelID, kvBits, kvGroupSize, toolDefinitionsDigest, templateContextDigest
///
/// CachePartitionKey keeps only the first 3. The last 2 (tool/template digests) are
/// DROPPED because the radix tree handles them implicitly:
///   - Different tools → applyChatTemplate renders different tool sections → different tokens
///   - Different template context → different template variables → different tokens
///   - Different tokens → different radix paths → naturally isolated, no explicit key needed
///
/// This is a deliberate simplification: HTTPPrefixCacheKey needed digests because it
/// did message-level prefix matching (same text could mean same cache). The radix tree
/// does token-level matching, so anything that changes tokens is automatically separated.
///
/// The existing mismatch diagnostics (HTTPPrefixCacheSpike.swift:142-170) are retained
/// as secondary logging during the migration period, but do not gate cache routing.
struct CachePartitionKey: Hashable {
    let modelID: String
    let kvBits: Int?          // nil = unquantized, 4, 8
    let kvGroupSize: Int      // typically 64
}

@MainActor
final class PrefixCacheManager {
    /// One radix tree per (model, kvBits, kvGroupSize) configuration.
    /// Prevents a snapshot captured with kvBits=8 (QuantizedKVCache) from being
    /// returned for a kvBits=nil request (KVCacheSimple) — same tokens, incompatible caches.
    private var trees: [CachePartitionKey: TokenRadixTree] = [:]
    private var memoryBudgetBytes: Int

    struct LookupResult {
        let snapshot: HybridCacheSnapshot?
        let restoredCache: [any KVCache]?
        let snapshotTokenOffset: Int               // 0 on miss
        let sharedPrefixLength: Int
        let reason: LookupReason
    }

    enum LookupReason: CustomStringConvertible {
        case hit(snapshotOffset: Int, totalTokens: Int, type: HybridCacheSnapshot.CheckpointType)
        case missNoEntries
        case missNoSnapshotInPrefix
    }

    /// Lookup: returns best usable snapshot + restored cache.
    /// Routes to the tree for the given partition key.
    func lookup(
        tokens: [Int],
        partitionKey: CachePartitionKey
    ) -> LookupResult

    /// Determine checkpoint offsets for the upcoming prefill.
    /// Phase 1: stable-prefix boundary only (if known and not already stored).
    /// Phase 2: + at most one speculative branch point from radix tree analysis,
    /// matching Marconi's speculative insertion rule.
    ///
    /// NOTE: the leaf checkpoint is NOT planned here. It is captured post-generation
    /// from the final cache and stored separately via storeLeaf(). Only mid-prefill
    /// checkpoints (whose Mamba state would be lost after prefill advances) are planned.
    func planCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        partitionKey: CachePartitionKey
    ) -> [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]

    /// Store mid-prefill snapshots under the prompt token path.
    /// Snapshots captured DURING prefill by the modified prepareWithCheckpoints() loop.
    func storeSnapshots(
        promptTokens: [Int],
        capturedSnapshots: [HybridCacheSnapshot],
        partitionKey: CachePartitionKey
    )

    /// Store the leaf snapshot under the post-response token path.
    /// storedTokens = re-tokenized (prompt + generated response) — represents
    /// the conversation state that the NEXT request will include as prefix.
    /// The leaf's cache state includes the generated response.
    func storeLeaf(
        storedTokens: [Int],
        leafSnapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey
    )

    /// Evict lowest-priority snapshots across ALL partitions until under budget.
    func evictToFitBudget()

    var stats: CacheStats
}
```

**`planCheckpoints()` logic (Phase 1):**
1. If `stablePrefixOffset` is known and no existing `.system` snapshot covers it in this partition → include as `.system` checkpoint
2. The leaf checkpoint is NOT planned here — it's captured post-generation from the final cache (leaf = terminal state, post-hoc is correct for this type only)
3. Exclude offsets that already have snapshots in the tree (no re-capture)

**`storeSnapshots()` logic:**
1. Get or create tree for `partitionKey`
2. `tree.insertPath(promptTokens)` — ensure path exists
3. For each captured snapshot: `tree.storeSnapshot(snapshot, atOffset: snapshot.tokenOffset)`
4. `evictToFitBudget()` if over memory

**`storeLeaf()` logic:**
1. Get or create tree for `partitionKey`
2. `tree.insertPath(storedTokens)` — extends the prompt path with generated response tokens
3. `tree.storeSnapshot(leafSnapshot, atOffset: storedTokens.count)` — at the leaf node
4. `evictToFitBudget()` if over memory

**Eviction priority (Phase 1 only):** Type-based LRU. Evict `.leaf` first, then `.branchPoint` (Phase 2), then `.system` last. Within the same type, evict least-recently-accessed. **Phase 2 replaces this heuristic** with Marconi's global utility score over eligible nodes (`childCount <= 1`).

**Tests 1.5 — `PrefixCacheManagerTests.swift`:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `lookupEmptyCacheReturnsMiss` | Cold start → `.missNoEntries` |
| 2 | `storeAndExactLookupReturnsHit` | Store + lookup same tokens → hit |
| 3 | `systemSnapshotSharedAcrossConversations` | Store sys+userA, lookup sys+userB → hit on system snapshot |
| 4 | `leafSnapshotMatchesWithinConversation` | Store sys+user1+asst1, lookup sys+user1+asst1+tool1 → hit on leaf |
| 5 | `lookupReturnsDeepCopiedCache` | Two lookups get independent cache objects |
| 6 | `mutatingRestoredCacheDoesNotAffectSnapshot` | Modify restored cache → snapshot unchanged |
| 7 | `evictionRemovesLeafBeforeSystem` | Type-priority honored |
| 8 | `memoryBudgetEnforced` | Overflow → eviction triggered |
| 9 | `planCheckpointsIncludesSystemBoundary` | Known system offset → included in plan |
| 10 | `planCheckpointsExcludesExistingSnapshots` | System snapshot already stored → not re-planned |
| 11 | `planCheckpointsNeverIncludesLeaf` | Leaf NOT in plan (captured post-generation via storeLeaf, not mid-prefill) |
| 12 | `snapshotBeyondDivergenceNotReturned` | Snapshot at 100, divergence at 80 → fallback to shallower |
| 13 | `storeSnapshotsFromPrefillAtCorrectOffsets` | Mid-prefill snapshots stored at absolute offsets |
| 14a | `storeLeafUnderPostResponseTokens` | Leaf stored under storedTokens (prompt + response), NOT under promptTokens |
| 14b | `leafSnapshotOffsetMatchesStoredTokenCount` | leafSnapshot.tokenOffset == storedTokens.count |
| 14c | `nextRequestHitsLeafViaExtendedPrefix` | Store leaf under [sys,user1,asst1]. Next request [sys,user1,asst1,tool1,user2] → leaf hit |
| 14 | `statsReflectState` | nodeCount, snapshotCount, totalSnapshotBytes accurate |
| 15 | `differentKvBitsIsolated` | Store with kvBits=8, lookup with kvBits=nil → miss (separate partitions) |
| 16 | `sameKvBitsShared` | Store with kvBits=8, lookup with kvBits=8 → hit (same partition) |
| 17 | `differentModelIDsIsolated` | Store with modelA, lookup with modelB → miss |
| 18 | `evictionCrossesPartitions` | Budget pressure evicts from any partition |

### Task 1.6: Integrate into `LLMActor` and `CompletionHandler`

**Files:**
- `tesseract/Features/Agent/LLMActor.swift` — replace `HTTPPrefixCacheSpike` with `PrefixCacheManager`
- `tesseract/Features/Server/CompletionHandler.swift` — unchanged (normalization stays)

**New flow in `LLMActor.makeHTTPPrefixCacheGeneration()` (inside `container.perform`):**

```swift
// 1. Normalize (existing pipeline — unchanged, happens in CompletionHandler)
//    conversation = HTTPPrefixCacheConversation (already normalized)

// 2. Tokenize (moved BEFORE cache lookup — was after in current code)
let history = conversation.historyMessages
let fullInput = try await context.processor.prepare(input: UserInput(chat: history, tools: toolSpecs))

// Extract flat token sequence from potentially 2D VLM tensor.
// Current LLMActor.swift:525-547 handles both 1D [seq] and 2D [batch, seq].
// Qwen3.5 VLM produces 2D; LLM-only produces 1D.
let fullTokens: [Int] = extractTokenSequence(fullInput.text.tokens)
// where extractTokenSequence handles:
//   ndim == 1: tokens.asArray(Int.self)
//   ndim == 2: tokens[0].asArray(Int.self)   (first batch element)
// and fullTokenCount = fullInput.text.tokens.dim(-1)  (last dimension, works for both)

// 3. Build partition key (replaces HTTPPrefixCacheKey for cache routing)
let partitionKey = CachePartitionKey(
    modelID: modelID,
    kvBits: parameters.kvBits,
    kvGroupSize: parameters.kvGroupSize
)

// 4. Detect stable prefix boundary (system + tools) via two-probe technique
//    HTTPPrefixCacheConversation exposes systemPrompt: String?, not a messages array.
//    Detector constructs [Chat.Message.system(prompt)] internally.
let stablePrefixOffset = try await StablePrefixDetector.detect(
    systemPrompt: conversation.systemPrompt,
    toolSpecs: toolSpecs,
    fullTokens: fullTokens,
    tokenizer: context.tokenizer
)  // Returns common prefix length of two probes with different dummy user content

// 5. Radix tree lookup
let lookupResult = prefixCache.lookup(tokens: fullTokens, partitionKey: partitionKey)

// 6. Determine what to prefill + what checkpoints to capture
let inputForGeneration: LMInput
let cacheToUse: [any KVCache]?
var checkpointPlan = prefixCache.planCheckpoints(
    tokens: fullTokens,
    stablePrefixOffset: stablePrefixOffset,
    partitionKey: partitionKey
)

let checkpointBaseOffset: Int
if let snapshot = lookupResult.snapshot, snapshot.tokenOffset > 0 {
    // HIT: restore cache, prefill only suffix.
    // Slice tokens on last dimension, drop mask (downstream recreates from cache offset).
    // Mirrors LLMActor.swift:537-553 which handles both 1D and 2D VLM shapes.
    cacheToUse = lookupResult.restoredCache
    let slicedTokens: MLXArray
    if fullInput.text.tokens.ndim <= 1 {
        slicedTokens = fullInput.text.tokens[snapshot.tokenOffset...]          // 1D: [seq]
    } else {
        slicedTokens = fullInput.text.tokens[0..., snapshot.tokenOffset...]    // 2D: [batch, seq]
    }
    inputForGeneration = LMInput(text: LMInput.Text(tokens: slicedTokens, mask: nil))
    checkpointBaseOffset = snapshot.tokenOffset
    // Only capture checkpoints in the SUFFIX (ones before snapshot are already stored)
    checkpointPlan = checkpointPlan.filter { $0.offset > snapshot.tokenOffset }
} else {
    // MISS: full prefill
    cacheToUse = nil
    inputForGeneration = fullInput
    checkpointBaseOffset = 0
}

// 7. Set checkpoint offsets on parameters — flows into TokenIterator → prepareWithCheckpoints()
//    Offsets are ABSOLUTE (full-prompt coordinates). checkpointBaseOffset tells prepare()
//    where input[0] sits in the full prompt, so it can rebase to relative coordinates.
var genParams = parameters
genParams.checkpointAtOffsets = Set(checkpointPlan.map(\.offset))
genParams.checkpointBaseOffset = checkpointBaseOffset

// 8. Create TokenIterator — this calls model.prepare() internally with checkpoints
//    NO separate prepare() call. TokenIterator owns prefill. (Evaluate.swift:607-609)
let iterator = try TokenIterator(
    input: inputForGeneration,
    model: context.model,
    cache: cacheToUse,         // nil on miss → TokenIterator creates fresh cache
    parameters: genParams
)

// 9. Read captured snapshots (populated by prepare() inside TokenIterator.init)
let capturedSnapshots = iterator.capturedSnapshots

// 10. Generate tokens via existing generateTaskWithFinalCache() flow
//     ... async generation loop using iterator ...

// 11. After generation: build stored conversation (prompt + generated assistant turn)
//     and re-tokenize to get the post-response token sequence.
//
//     WHY: finalCache has advanced through the generated response, so its offset is
//     beyond fullTokens.count. The leaf snapshot represents the post-response state.
//     The NEXT request will include the generated response in its history, so the
//     next request's token sequence starts with storedTokens as a prefix.
//     This mirrors the existing measureHTTPPrefixCacheTokenCount pattern.
let storedConversation = buildStoredConversation(conversation, generatedResponse)
let storedInput = try await context.processor.prepare(
    input: UserInput(chat: storedConversation.historyMessages, tools: toolSpecs)
)
let storedTokens: [Int] = extractTokenSequence(storedInput.text.tokens)  // handles 2D VLM

// 12. Offset-alignment: reconcile normalized token count with actual cache state.
//
//     The generated assistant content may have trailing whitespace that gets
//     normalized (trimmed) in the stored conversation. This means:
//       storedTokens.count < finalCache's actual offset
//     because the cache advanced through the full (un-normalized) response,
//     but storedTokens reflects the trimmed version.
//
//     Current code (LLMActor.swift:331-357) handles this by trimming attention KV.
//     We must do the same before capturing the leaf snapshot.
let finalCache = try await cacheHandle.takeFinalCache()
let actualCacheOffset = finalCache.compactMap { $0.offset }.max() ?? 0

if actualCacheOffset > storedTokens.count {
    let trimAmount = actualCacheOffset - storedTokens.count
    for i in 0..<finalCache.count {
        if finalCache[i].isTrimmable {
            finalCache[i].trim(trimAmount)  // KVCacheSimple, QuantizedKVCache: offset -= trimAmount
        }
        // MambaCache: isTrimmable == false → skip.
        // Accepted divergence: Mamba recurrent state includes the un-normalized
        // trailing tokens. On restore, these extra tokens are baked into the state.
        // This matches the current prototype's behavior (LLMActor.swift:349-355).
        // Impact: negligible for trailing whitespace (whitespace tokens have minimal
        // effect on recurrent state). If this proves problematic, capture a separate
        // post-normalization Mamba state by re-running the trimmed suffix — but that
        // is expensive and not needed unless quality regressions are observed.
    }
}

let leafSnapshot = HybridCacheSnapshot.capture(
    cache: finalCache, offset: storedTokens.count, type: .leaf
)

// 13. Store: mid-prefill snapshots under fullTokens path, leaf under storedTokens path
//   - Mid-prefill snapshots (e.g., stable prefix at offset 4K): stored at their absolute
//     offsets in the radix tree. fullTokens[0..4K] is the path, snapshot at node 4K.
//   - Leaf snapshot: stored under storedTokens path (which extends fullTokens with the
//     generated response). Next request's tokens will share this prefix.
prefixCache.storeSnapshots(
    promptTokens: fullTokens,
    capturedSnapshots: capturedSnapshots,     // from mid-prefill, under fullTokens path
    partitionKey: partitionKey
)
prefixCache.storeLeaf(
    storedTokens: storedTokens,
    leafSnapshot: leafSnapshot,              // under post-response path
    partitionKey: partitionKey
)
Memory.clearCache()
```

**Key changes from current code:**
- Tokenization moved BEFORE cache lookup (was after)
- Stable-prefix detection replaces system-only detection
- `CachePartitionKey` replaces `HTTPPrefixCacheKey` for cache routing
- Checkpoint offsets flow through `GenerateParameters` → `TokenIterator` → `model.prepare()` — **no separate prepare() call, no double-prefill**
- Snapshots read from `iterator.capturedSnapshots` after init
- Leaf snapshot captured post-generation from final cache (correct for terminal state)

**Backward compat:** Keep `HTTPPrefixCacheConversation.diagnosePrefixMismatch()` as secondary diagnostic logging on cache miss. Remove when radix tree is stable.

**Tests 1.6 — `PrefixCacheIntegrationTests.swift`:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `endToEndNormalizeTokenizeLookupHit` | Wire message → normalize → tokenize → store → new request → normalize → tokenize → lookup → hit |
| 2 | `suffixSlicedAtSnapshotOffset` | Suffix tokens start at snapshotTokenOffset |
| 3 | `cacheMissTriggersFullPrefillWithCheckpoints` | Miss → full prefill → system and leaf snapshots captured |
| 4 | `hybridCacheRestoredWithCorrectLayerTypes` | Restored cache has MambaCache at 0,1,2,4,5,6,... and attention at 3,7,11,... |
| 5 | `restoredCacheOffsetMatchesSuffix` | `cache[faIdx].offset == snapshotTokenOffset` (critical alignment invariant) |
| 6 | `normalizationProducesStableTokens` | Same message, different whitespace → same tokens after normalize |
| 7 | `dynamicQuantizationHandledInSnapshot` | Early snapshot has KVCacheSimple, later has QuantizedKVCache → both restore correctly |
| 8 | `snapshotsCapturedDuringPrefillNotPostHoc` | Stable-prefix snapshot captured at correct offset DURING prefill, not from final cache |
| 9 | `noDoublePrefill` | TokenIterator.init() calls prepareWithCheckpoints() exactly once |
| 10 | `partitionKeyIsolatesDifferentKvBits` | Store at kvBits=8, lookup at kvBits=nil → miss despite same tokens |
| 11 | `stablePrefixIncludesToolTokens` | Stable prefix checkpoint covers system + tool tokens (two-probe boundary) |
| 12 | `checkpointBaseOffsetRebasesCorrectlyOnHit` | Cache hit at 3K, checkpoint at abs 4K → fires at relative 1K in suffix prefill |
| 13 | `leafStoredUnderPostResponsePath` | After generation, leaf snapshot stored under re-tokenized (prompt + response) path |
| 14 | `nextTurnHitsLeafFromPreviousTurn` | Turn N stores leaf. Turn N+1 (extending conversation) hits leaf on lookup |
| 15 | `leafOffsetAlignedAfterNormalization` | Generated response with trailing whitespace → trim aligns attention offset to storedTokens.count |
| 16 | `mambaLayersNotTrimmedInLeafAlignment` | MambaCache.isTrimmable==false → skip in trim loop (accepted divergence) |
| 17 | `vlm2DTokensExtractedCorrectly` | 2D `[batch, seq]` tensor → flat `[Int]` sequence from first batch element |
| 18 | `detectorAcceptsSystemPromptString` | `conversation.systemPrompt` (String?) → constructs Chat.Message internally |

### Task 1.7: Migrate existing `HTTPPrefixCacheSpikeTests`

**Keep as-is** (normalization — still needed):
- `assistantContentIsTrimmed`, `userAndToolContentIsPreservedVerbatim`, `subagentTrailingWhitespaceContentMatchesEcho`, `storedTurnWithWhitespaceContentMatchesEmptyEcho`, `reasoningIsTrimmedAtConstruction`, `storedAssistantTurnMatchesOpenCodeEcho`, `storedAssistantRoundTripWithEmptyContentAndWhitespaceReasoning`, `assistantToolCallTurnsCanBeRestoredAsPrefixes`

**Migrate:** `storeRestoresDeepCopiedCaches` → `lookupReturnsDeepCopiedCache`

**Retire:** Whole-conversation keying tests (replaced by token-level matching)

### Task 1.8: E2E verification (requires model)

**Scenario: HybridPrefixCacheE2E**

```
1. Load Qwen3.5-4B-paro

2. Request A: system(4K) + tools + user "list files in /tmp"
   → Full prefill with checkpoint capture at system boundary + leaf
   → Record TTFT_A (baseline)

3. Request B: SAME system + SAME tools + DIFFERENT user "read file /tmp/foo.txt"
   → Should hit stable-prefix snapshot (system + tools)
   → Record TTFT_B
   → Assert: TTFT_B < TTFT_A × 0.6

4. Logit equivalence (mid-prefill checkpoint restore):
   → Run request B WITHOUT cache (full prefill)
   → Compare logits at first generated token: cached vs uncached must be BITWISE IDENTICAL
   → This gate applies to MID-PREFILL checkpoint restores (stable prefix, branch points)
     where no normalization-offset trimming occurs — state is captured and restored exactly.
   → For LEAF hits (normalized conversation reuse), the Mamba normalization divergence
     means bitwise equality is not guaranteed. Instead, verify bounded divergence:
     max |logit_cached - logit_uncached| < 0.01 (same threshold as current prototype)

5. Normalization round-trip:
   → Send request B with trailing whitespace in assistant echo
   → Must still hit cache

6. QuantizedKVCache path:
   → If kvBits=8 is active, verify restored attention layers are QuantizedKVCache
   → Output correctness unchanged
```

**Manual test 1.8:**
1. Run OpenCode subagent flow that triggered original OOM
2. Verify logs: `prefixCache.hit` on subsequent requests
3. TTFT drops after first request
4. No quality regression

---

## Phase 2 — Marconi Extensions: Branch-Point Checkpointing & Utility-Scored Eviction

### Task 2.1: Speculative insertion at admission time

**File:** `tesseract/Features/Server/PrefixCacheManager.swift`

Extend `planCheckpoints()` to perform **Marconi speculative insertion**:
1. Walk the tree with the new token sequence as if inserting it.
2. If insertion would **split an existing edge** (the shared prefix continues inside a compressed edge, then diverges), mark the split offset as a `.branchPoint` checkpoint candidate.
3. If the new sequence only **extends an existing path at a node boundary**, do **not** create a `.branchPoint` checkpoint. The leaf checkpoint already covers this case.
4. Admit **at most one** `.branchPoint` candidate per request. Combined with the leaf checkpoint, this preserves Marconi's "max 2 checkpoints per sequence" rule.

```swift
// In planCheckpoints() — Phase 2 addition:
if let splitOffset = tree.findIntermediateSplitOffsetForInsertion(tokens: tokens) {
    if !tree.hasSnapshotAt(offset: splitOffset) {
        plan.append((offset: splitOffset, type: .branchPoint))
    }
}
```

**Tests 2.1:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `divergenceInsideCompressedEdgeCreatesCandidate` | [1,2,3,4] stored, [1,2,5,6] → split at offset 2 |
| 2 | `exactPathExtensionDoesNotCreateBranchPoint` | [1,2,3] stored, [1,2,3,4,5] → no speculative checkpoint |
| 3 | `nodeBoundaryDivergenceDoesNotCreateIntermediateCheckpoint` | Existing node [1,2], new child [1,2,9] → leaf only, no `.branchPoint` |
| 4 | `coldTreeNoSpeculativeCandidates` | Empty tree → no branch-point candidates |
| 5 | `existingSnapshotNotReCandidate` | Split offset already has snapshot → not re-planned |
| 6 | `atMostOneBranchPointPerSequence` | Complex tree, single insertion → at most one speculative candidate |

### Task 2.2: Logit-equivalence verification harness

**The most critical correctness gate.** Required before any further phases.

**Scope:** Logit-equivalence tests apply to **mid-prefill checkpoint restores** (stable prefix, branch points) where no normalization-offset trimming occurs. These checkpoints capture and restore state exactly — bitwise logit match is required.

For **leaf hits from normalized conversations**, the Mamba normalization divergence (blocker #22) means bitwise equality is not guaranteed. Leaf hits use a weaker **bounded-divergence** contract: `max |logit_cached - logit_uncached| < 0.01`. This matches the current prototype's accepted divergence (`LLMActor.swift:349-355`).

**Tests 2.2 — `HybridCacheCorrectnessTests.swift` (requires model):**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `CRITICAL_midPrefillRestoreMatchesFullPrefill` | Mid-prefill snapshot (no normalization): full prefill logits == restore-at-K + suffix logits. K = N/4, N/2, 3N/4. **Bitwise match required.** |
| 2 | `restoreAtExactMatch` | Restore at full length → no suffix → logits match |
| 3 | `divergentSuffixAfterRestore` | Checkpoint at K, different suffix → model processes correctly |
| 4 | `mambaStateRestoredExactly` | Mid-prefill: MambaCache.state arrays bitwise match pre-checkpoint |
| 5 | `attentionKVRestoredExactly` | After restore: offset, keys, values match |
| 6 | `quantizedKVCacheRestoredExactly` | After restore: wq, scales, biases match; groupSize/bits correct |
| 7 | `multipleRestoresFromSameSnapshot` | Two restores → identical logits (isolation) |
| 8 | `longContext16KRestore` | Checkpoint at 8K, suffix 8K → logits match full 16K |
| 9 | `leafHitWithNormalizationDivergenceBounded` | Leaf hit where normalization trimmed attention but not Mamba: `max\|logit_diff\| < 0.01` |
| 10 | `leafHitWithoutNormalizationMatchesBitwise` | Leaf hit where no trimming occurred (0 trim amount): logits match exactly |

### Task 2.3: FLOP-aware eviction

**New file:** `tesseract/Features/Server/EvictionPolicy.swift`

```swift
struct EvictionScore: Comparable {
    let rawRecency: Double
    let rawFlopEfficiency: Double
    let normalizedRecency: Double
    let normalizedFlopEfficiency: Double
    let utility: Double              // normalizedRecency + alpha * normalizedFlopEfficiency
    static var alpha: Double = 0.0   // pure recency until Task 2.4 tunes alpha
}
```

**Candidate set (Marconi rule):** score **only** nodes that (a) have a snapshot and (b) have `childCount <= 1`. Nodes with `2+` children represent shared prefixes and are protected from eviction scoring.

**Per-node FLOP formulas (Appendix A / `utils.py` in the reference repo):**
- `F_attn(L, D) = 8 * L * D^2 + 4 * L^2 * D`
- `F_mlp(L, D) = 16 * L * D^2`
- `F_ssm(L, D, N) = 12 * L * D^2 + 16 * L * D * N + 10 * L * D`

**Per-node state-size formulas:**
- Generic Marconi attention formula: `M_attn(L, D_kv) = 4 * L * D_kv` bytes, where `D_kv = kvHeads * headDim`
- For local Qwen3.5-4B attention: `kvHeads = 8`, `headDim = 128`, so `D_kv = 1024` and `M_attn(L) = 4 * L * 1024`
- Local Qwen3.5 GatedDeltaNet stores two arrays in `MambaCache.state`:
  - conv state: `[B, convKernel - 1, convDim]`
  - recurrent state: `[B, numVHeads, headVDim, headKDim]`
- Therefore `M_ssm = 2 * ((convKernel - 1) * convDim + numVHeads * headVDim * headKDim)` bytes per layer for `B = 1`

**Parent-relative FLOPs saved (Marconi rule):**
- Let `L_total = node.tokenOffset`
- Let `L_parent = node.parent?.tokenOffset ?? 0`
- Let `deltaL = L_total - L_parent`
- `deltaF(node) =`
  `numSSMLayers * F_ssm(deltaL, D, N)`
  `+ numAttentionLayers * (F_attn(L_total, D) - F_attn(L_parent, D))`
  `+ numMLPLayers * (F_mlp(L_total, D) - F_mlp(L_parent, D))`
- `rawFlopEfficiency(node) = deltaF(node) / snapshot.memoryBytes`

`snapshot.memoryBytes` is the correct denominator for this design because each snapshot stores the **full hybrid cache** at that node's offset, not a delta from its parent. FLOP scoring should use the measured bytes from the captured snapshot; the formulas above are for reasoning and test expectations.

**Recency transform:**
- `ageSeconds = max(now - node.lastAccessTime, epsilon)`
- `rawRecency(node) = 1 / ageSeconds`

**Normalization (repo-aligned min-max):**

```swift
func normalize(_ values: [Double]) -> [Double] {
    guard values.count > 1 else { return Array(repeating: 1.0, count: values.count) }
    guard let minValue = values.min(), let maxValue = values.max(), minValue != maxValue else {
        return Array(repeating: 1.0, count: values.count)
    }
    return values.map { ($0 - minValue) / (maxValue - minValue) }
}
```

For each eviction pass:
1. Gather eligible nodes across **all partitions**
2. Compute `rawRecency` and `rawFlopEfficiency`
3. Min-max normalize both vectors independently
4. Compute `utility = normalizedRecency + alpha * normalizedFlopEfficiency`
5. Evict the **minimum-utility** node

**Eviction mechanics:**
- `childCount == 0`: delete leaf node and clean empty ancestors
- `childCount == 1`: evict the node's snapshot, then collapse the snapshot-less node into its child to preserve radix compression

This replaces the Phase 1 type-priority heuristic. Checkpoint type still exists for diagnostics and Phase 1 policy, but it is **not** part of the Marconi Phase 2 utility score.

**Tests 2.3:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `candidateSetExcludesMultiChildNodes` | Shared-prefix nodes with `2+` children are never scored |
| 2 | `parentRelativeFlopsUsed` | Same total length, longer unique suffix after parent → higher `deltaF` |
| 3 | `minMaxNormalizationHandlesDegenerateCase` | All-equal inputs normalize to `1.0` |
| 4 | `recentAccessBoostsUtility` | Newer node gets higher normalized recency |
| 5 | `higherFlopEfficiencyBoostsUtility` | At equal recency, higher FLOPs/byte wins |
| 6 | `lowestUtilityEvictedAcrossPartitions` | Global minimum utility is evicted, not just within one tree |
| 7 | `singleChildEvictionCollapsesNode` | Snapshot eviction on a 1-child node preserves compressed radix structure |
| 8 | `memoryBudgetRespected` | Repeated lowest-utility eviction brings usage under budget |

### Task 2.4: Adaptive `alpha` tuning (paper/repo-aligned)

**New file:** `tesseract/Features/Server/AlphaTuner.swift`

Marconi starts with pure recency and tunes `alpha` retrospectively:

1. Start with `alpha = 0.0`
2. On the **first** call to `evictToFitBudget()`, record `requestsBeforeFirstEviction`
3. Set `bootstrapWindowSize = 5 * requestsBeforeFirstEviction`
   - The paper describes a `5-15x` bootstrap range
   - The reference repo uses `bootstrap_multiplier = 5`
4. Snapshot the radix-tree state at the start of the bootstrap window
5. Collect request history for the window as `(inputTokens, outputTokens)` tuples
6. Replay the window for `alpha in {0.0, 0.1, ..., 2.0}`
7. Choose the `alpha` with the highest **total FLOPs saved**
   - Tie-breaker: higher token hit rate
8. Adopt the tuned `alpha` for subsequent evictions

For this on-device design, one tuning pass is enough for development readiness. Continuous retuning can be added later if workload drift becomes visible in benchmarks.

**Tests 2.4:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `startsAtZeroBeforeFirstEviction` | Cache behaves like pure-recency scoring before tuning |
| 2 | `bootstrapWindowUsesFiveTimesFirstEvictionCount` | Tuning window matches repo default multiplier |
| 3 | `gridSearchChoosesBestAlphaByFlopsSaved` | Replayed window selects the top-FLOPs-saved `alpha` |
| 4 | `tunedAlphaUsedForSubsequentEvictions` | After tuning, eviction utility uses the chosen `alpha` |

**Manual validation:** Run benchmark scenarios with the tuned `alpha` and compare against fixed `alpha = 0.0` and `alpha = 2.0`. Confirm the tuned value is near the knee of the TTFT/hit-rate curve on the target workload.

---

## Phase 3 — Two-Pass Prefill & Production Hardening

### Task 3.1: Two-pass prefill for checkpoint alignment

When the best snapshot is at offset K but the shared prefix extends to M > K, tokens [K..M] are re-prefilled even though they match the tree. Two-pass optimization:

1. Restore from snapshot at K
2. Prefill [K..M] and capture a NEW snapshot at M
3. Continue prefilling [M..N]

Future requests sharing [0..M] will hit the M-offset snapshot directly.

**Heuristic:** Only trigger two-pass if gap (M - K) > 256 tokens. Smaller gaps not worth the snapshot overhead.

**Tests 3.1:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `alignedSnapshotSinglePass` | Snapshot at exact divergence → one pass |
| 2 | `largeGapTriggersTwoPass` | Gap > 256 → two-pass, new snapshot created |
| 3 | `smallGapSinglePass` | Gap ≤ 256 → no extra snapshot |
| 4 | `twoPassLogitsMatchFullPrefill` | **CRITICAL** — correctness gate |
| 5 | `newSnapshotFromTwoPassReusable` | Subsequent request benefits from mid-gap snapshot |

### Task 3.2: prefillStepSize benchmark

With full system: benchmark `prefillStepSize ∈ {256, 512, 1024, 2048, 4096}` for cold and warm prefill. Consider adaptive: smaller for cold (memory safety), larger for warm (throughput).

### Task 3.3: Memory budget auto-sizing

```swift
let total = ProcessInfo.processInfo.physicalMemory
let modelMem: UInt64 = /* from loading stats */
let budget = (total - modelMem - 4_GB_headroom) / 2
```

48 GB Mac, 10 GB model: `(48 - 10 - 4) / 2 = 17 GB` cache budget.

**Tests 3.3:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `budgetScalesWithRAM` | More RAM → higher budget |
| 2 | `budgetCapped` | Max reasonable limit even on large machines |
| 3 | `budgetAccountsForModel` | Bigger model → smaller budget |

### Task 3.4: Diagnostic logging

Structured logging via `Log.agent`:
- Hit/miss reason + token counts + snapshot offset + checkpoint type
- Snapshot capture events (offset, type, bytes, during-prefill confirmation)
- Eviction events (type, score, freed bytes)
- TTFT breakdown (lookup, restore, prefill, first-token)
- Memory (snapshots total, model, free pool)

### Task 3.5: E2E benchmark suite

**New file:** `tesseract/Features/Agent/Benchmark/PrefixCacheBenchmark.swift`

| Scenario | Description | Expected |
|----------|-------------|----------|
| PC1 | Cold: 16K, no cache | Baseline TTFT |
| PC2 | Warm: repeat exact 16K | Near-zero TTFT |
| PC3 | Shared system: same 4K system + different user | ~50-70% TTFT reduction |
| PC4 | Tool change: same system, different tools | Hit on system snapshot only |
| PC5 | Long conversation: 20 turns | TTFT stable (prefix reuse) |
| PC6 | Subagent switch: main→sub→main | Hit on return |
| PC7 | Memory pressure: fill + new request | Graceful eviction |
| PC8 | **Logit equivalence: mid-prefill restore vs full** | **Bitwise match** (mid-prefill checkpoints only; leaf hits with normalization trimming use bounded divergence < 0.01) |
| PC9 | Normalization: whitespace variants | Hits despite differences |
| PC10 | Two-pass: misaligned snapshot | New snapshot created |
| PC11 | QuantizedKVCache round-trip | Correct bits/groupSize after restore |

---

## Phase Dependency Graph

```
Phase 0 (Memory.clearCache fix)
    │
    ▼
Phase 1 (partitioned radix tree + checkpoint-during-prefill via TokenIterator + stable prefix)
    │
    ▼
Phase 2 (Marconi: speculative branch-point + utility-scored eviction + alpha tuning)
    │
    ▼
Phase 3 (two-pass prefill + production hardening)
```

All phases are sequential. Phase 1 includes the modified `prepare()` loop (previously deferred to Phase 2) because Mamba state capture during prefill is a prerequisite for any hybrid checkpoint.

---

## Test Summary

- Phase 0: 3 manual validation steps for memory behavior.
- Phase 1: 72 new unit tests across snapshot/prepare/detector/radix/manager, 18 integration tests for the end-to-end cache path, migration of the existing normalization-focused `HTTPPrefixCacheSpikeTests`, 1 model-backed E2E scenario.
- Phase 2: 28 new unit tests across speculative admission, correctness gating, Marconi utility scoring, and adaptive `alpha` tuning, plus 1 manual benchmark validation pass.
- Phase 3: 8 unit tests for two-pass prefill and prefill-step heuristics, 11 model-backed benchmark scenarios, and 1 manual validation pass.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mid-prefill snapshot restore produces wrong logits | Medium | **Critical** | Task 2.2 tests #1-8: bitwise-match gate on mid-prefill restores (no normalization). Fall back to full prefill on any mismatch. |
| Normalized leaf hit divergence exceeds bound | Low | Medium | Task 2.2 test #9: bounded divergence < 0.01. If exceeded, disable leaf caching for hybrid models (fall back to stable-prefix-only). Current prototype already accepts this divergence. |
| Variable chunk sizes hurt prefill throughput | Low | Low | Small chunks (< prefillStepSize) only at checkpoint boundaries. Vast majority of chunks are full-size. |
| QuantizedKVCache restore with wrong groupSize/bits | Low | High | Parse from metaState, not defaults. Test 1.1#14 validates. |
| Stable-prefix detection fails for non-ChatML templates | Medium | Medium | Two-probe technique + verification step (`fullTokens[0..<boundary] == probeA[0..<boundary]`) catches mismatches. Falls back to no stable-prefix checkpoint. |
| Snapshot memory exceeds budget | Medium | Medium | Budgeting now uses corrected sizing formulas from the local Qwen3.5 cache shapes: a 4K unquantized snapshot is ~202 MiB. Auto-sizing and Marconi utility-scored eviction must use measured `snapshot.memoryBytes`, not rough MB estimates. |
| Adaptive `alpha` tuning picks an unstable value on a short trace | Low | Medium | Start with `alpha = 0`, require a bootstrap window of `5x` the first-eviction request count, tie-break on token hit rate, and allow fallback to `alpha = 0` if the bootstrap trace is too small to tune confidently. |
| Upstream mlx-swift-lm changes conflict | Medium | Medium | Minimize vendor changes. `prepareWithCheckpoints()` is a protocol extension (existing `prepare()` untouched). `GenerateParameters.checkpointAtOffsets`/`checkpointBaseOffset` are new optional fields with defaults. Only `LLMModel`, `Qwen35`, and `TokenIterator.prepare()` modified. |
| Cross-config contamination (wrong kvBits snapshot returned) | Low | High | `CachePartitionKey` isolates trees by `(modelID, kvBits, kvGroupSize)`. Tests 1.5#15-17 validate. |
| TokenIterator contract breaks in future mlx-swift-lm updates | Low | Medium | `capturedSnapshots` property is additive. Protocol extension is non-breaking — all existing conformers keep working via default impl. |
| Re-tokenization of stored conversation is expensive | Low | Low | One extra tokenization per generation. Current code already does this via `measureHTTPPrefixCacheTokenCount`. Same cost. |
| Two-probe detection produces wrong boundary for unusual templates | Low | Medium | Verification step (probe prefix == fullTokens prefix) catches mismatches. Falls back to no stable-prefix checkpoint. |

---

## Implementation Notes (previously Open Questions — all resolved or downgraded)

1. **eval(cache) completeness:** `eval(cache)` in the prefill loop forces the lazy graph for cache state arrays (both KV and Mamba state) before snapshot capture. After `eval(cache)`, it is safe to deep-copy `cache.flatMap(\.state)`. **Verify during implementation:** capture a snapshot immediately after `eval(cache)`, call `Memory.clearCache()`, and confirm the copied arrays remain valid and bitwise-stable. Do not rely on a debug-only `isTracing` API.

2. **Snapshot size profiling:** Derive sizes from the **actual local cache shapes**, not only from the paper's generic symbols. Local Qwen3.5-4B config constants from `Qwen35.swift` are `hiddenSize = 4096`, `attentionHeads = 32`, `kvHeads = 8`, `linearNumValueHeads = 64`, `linearNumKeyHeads = 16`, `linearValueHeadDim = 128`, `linearKeyHeadDim = 192`, `linearConvKernelDim = 4`, with `32` total layers = `24` SSM + `8` attention:
   - Attention KV state per layer uses GQA, so size is based on `kvHeads * headDim`, **not** `hiddenSize`
   - `headDim = hiddenSize / attentionHeads = 128`
   - `D_kv = kvHeads * headDim = 8 * 128 = 1024`
   - Attention KV bytes per layer at sequence length `L`: `M_attn(L) = 4 * L * D_kv = 4 * L * 1024`
   - At `L = 4096`: `4 * 4096 * 1024 = 16,777,216` bytes = `16 MiB` per attention layer
   - `8` attention layers: `4K ≈ 128 MiB`, `8K ≈ 256 MiB`, `16K ≈ 512 MiB`
   - GatedDeltaNet conv cache shape: `[B, convKernel - 1, convDim] = [1, 3, 14336]`
   - Conv bytes per SSM layer: `1 * 3 * 14336 * 2 = 86,016` bytes
   - GatedDelta recurrent state shape from `gatedDeltaUpdate()`: `[B, Hv, Dv, Dk] = [1, 64, 128, 192]`
   - Recurrent bytes per SSM layer: `1 * 64 * 128 * 192 * 2 = 3,145,728` bytes
   - Total SSM bytes per layer: `3,231,744` bytes ≈ `3.08 MiB`
   - `24` SSM layers: ≈ `73.97 MiB`, independent of sequence length
   - **Total unquantized snapshot:** `4K ≈ 202 MiB`, `8K ≈ 330 MiB`, `16K ≈ 586 MiB`
   - With `kvBits = 8`, only the attention portion should shrink materially; use measured `snapshot.memoryBytes` as the source of truth because quantized packing overhead depends on runtime representation.
   - **Verify during implementation:** log `snapshot.memoryBytes` on first capture and compare against these shape-derived estimates.

3. **Chat template determinism:** The Jinja chat template is deterministic for identical inputs — no random or time-dependent elements. The `processor.prepare()` pipeline (tokenize → template render → re-tokenize) is stateless. **No risk** for the radix tree as long as normalization is applied before tokenization (which it is).

4. **TokenIterator cache injection contract:** When passing restored cache to `TokenIterator`:
   - `Qwen35Language.LanguageModel.__call__` reads `cache[model.faIdx].offset` at line 930-932 for RoPE positioning
   - KVCacheSimple: offset auto-set from `keys.dim(2)` on state setter (`KVCache.swift:391`) — correct after restore
   - QuantizedKVCache: offset set from `metaState[1]` via metaState setter (`KVCache.swift:942`) — correct after restore
   - RotatingKVCache: offset set from `metaState[3]` via metaState setter (`KVCache.swift:635`) — correct after restore
   - **No action needed** — the existing state/metaState injection handles offset correctly for all types.

5. **GenerateParameters mutability:** `GenerateParameters` is a struct with `var` properties (`Evaluate.swift:67-130`). Adding `var checkpointAtOffsets: Set<Int> = []` and `var checkpointBaseOffset: Int = 0` is a backward-compatible extension. Default values mean no existing callers break. **Verify during implementation:** compile and run existing tests after adding the fields.

6. **Stored conversation construction:** The current code already builds the stored conversation in `LLMActor.generateServerTextCompletion()` (`LLMActor.swift:298-318`) by appending the generated assistant turn to the conversation. `historyMessages` reconstructs `[Chat.Message]` from this. The new flow uses the same construction for re-tokenization. **Verify during implementation:** confirm `storedConversation.historyMessages` produces the same messages as what the next request will send.

7. **Normalization trim magnitude:** Whitespace normalization typically removes 1-5 tokens (for example trailing newlines on assistant content). Treat the impact on Mamba state as an **empirical assumption**, not a guaranteed mathematical property. Test 2.2#9 is the gate: if `max |logit_cached - logit_uncached|` ever exceeds `0.01`, disable leaf caching for hybrid models and keep only stable-prefix / branch-point checkpoints.
