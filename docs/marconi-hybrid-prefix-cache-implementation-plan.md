# Marconi-Style Hybrid Prefix Cache — Implementation Plan

**Date:** 2026-04-11 (v9 — paper/repo aligned, development-ready)
**Status:** Ready for development
**Prerequisite reading:** `docs/mlx-swift-lm-prefill-memory-research.md` (§4.7, §7.5–7.6)
**References:** [Marconi paper](https://assets.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf), [Marconi reference repo](https://github.com/ruipeterpan/marconi), [SGLang MambaRadixCache](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/), [mlx-lm #1072](https://github.com/ml-explore/mlx-lm/pull/1072)

---

## Review Blockers Addressed

### v2 blockers (resolved)

| #   | Blocker                                                            | Resolution                                                                                                                                                               |
| --- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Attention-only Phase 1 cannot resume divergent suffixes on Qwen3.5 | Checkpoints store full hybrid cache (all layers). No attention-only path.                                                                                                |
| 2   | Phase 2 defined checkpoint as layer outputs instead of SSM state   | `HybridCacheSnapshot` stores actual `MambaCache.state` (conv + recurrent arrays) and `KVCacheSimple.state` / `QuantizedKVCache.state`. Matches Marconi and mlx-lm #1072. |
| 3   | Partial-prefix hits don't define cache slicing to match length     | Exact-prefix matching only at checkpointed offsets. No mid-stream trimming.                                                                                              |
| 4   | Normalization treated as diagnostics-only                          | Pipeline-critical. normalize → tokenize → radix lookup.                                                                                                                  |
| 5   | Storage model duplicates full cache per node                       | Selective admission at checkpoint boundaries only. Type-based eviction.                                                                                                  |

### v3 blockers (resolved)

| #   | Blocker                                                                                                | Resolution                                                                                                                                                                                                                                                     |
| --- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6   | Phase 1 `store()` creates earlier snapshots from final cache — impossible for Mamba                    | Snapshots captured **during prefill**, not post-hoc. Checkpoint offsets known before prefill starts. Prefill loop captures snapshots as it passes each offset.                                                                                                 |
| 7   | Checkpoint capture only fires on chunk boundaries, not arbitrary segment offsets                       | Prefill loop dynamically adjusts chunk size to land exactly on checkpoint offsets. Variable-size chunks near boundaries.                                                                                                                                       |
| 8   | Snapshot restore assumes KVCacheSimple but runtime uses QuantizedKVCache                               | Restore mirrors `loadPromptCache()` pattern — stores `className` per layer, creates correct type on restore. Handles KVCacheSimple, QuantizedKVCache, RotatingKVCache, MambaCache. QuantizedKVCache constructor receives groupSize/bits parsed from metaState. |
| 9   | Three-block segmentation model (system/user/assistant) doesn't match real alternating prompt structure | Replaced with two concrete checkpoint types: **stable-prefix boundary** (cross-conversation reuse) and **conversation leaf** (within-conversation reuse, like current HTTPPrefixCacheSpike). No coarse block model.                                            |

### v4 blockers (this revision)

| #   | Blocker                                                                                                                                                                                                            | Resolution                                                                                                                                                                                                                                                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10  | Phase 1 integration calls `model.prepare()` directly, but `TokenIterator.init()` already owns prefill (`Evaluate.swift:607-609`) — would double-prefill or require non-trivial TokenIterator refactor              | Checkpoint offsets flow INTO `TokenIterator` via `GenerateParameters`. TokenIterator passes them to `model.prepare()` inside its own init. Captured snapshots stored as `TokenIterator.capturedSnapshots` property. **No call to prepare() outside TokenIterator.** LLMActor reads snapshots after iterator creation. |
| 11  | Checkpoint capture loop exits at `y.tokens.size <= prefillStepSize`, so checkpoints in the final tail remainder are never captured                                                                                 | After main while-loop, a second drain loop processes any checkpoints that fall within the remaining tail before returning `.tokens(y)`.                                                                                                                                                                               |
| 12  | System-boundary detection tokenizes system-only messages, but real shared prefix is system + tool definitions (tools passed as separate param to `applyChatTemplate(messages:tools:)` via `Tokenizer.swift:16-20`) | Renamed to `StablePrefixDetector`. Uses **two-probe technique**: tokenize messages with two different dummy user contents, find common prefix length = system + tools boundary. No sentinel token search (tokenization is context-dependent).                                                                         |
| 13  | Token-only radix tree dropped model/config partitioning — snapshot from kvBits=8 could be returned for kvBits=nil request                                                                                          | `PrefixCacheManager` holds `[CachePartitionKey: TokenRadixTree]` dictionary. Partition key = `(modelID, kvBits, kvGroupSize)`. Matches existing `HTTPPrefixCacheKey` fields. Tool/template digests implicit in token sequences.                                                                                       |

### v5 blockers (this revision)

| #   | Blocker                                                                                                                                                                                                          | Resolution                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 14  | Checkpoint offsets are in full-prompt coordinates but suffix prefill starts at offset 0 — checkpoints never fire on cache hit                                                                                    | `prepare()` gains `checkpointBaseOffset: Int` (default 0). Set to `snapshot.tokenOffset` on cache hit. Inside the loop, checks use `checkpointBaseOffset + currentOffset`. Snapshot stored with the correct absolute offset.                 |
| 15  | Leaf snapshot stored under pre-generation `fullTokens` but `finalCache` includes the generated response — offset mismatch                                                                                        | After generation, re-tokenize the stored conversation (prompt + generated assistant turn) → `storedTokens`. Store leaf under `storedTokens` with offset = `storedTokens.count`. Mirrors existing `measureHTTPPrefixCacheTokenCount` pattern. |
| 16  | `LanguageModel.prepare()` signature change breaks 10+ VLM overrides, SpeculativeTokenIterator, WiredMemoryUtils                                                                                                  | Don't change the existing protocol method. Add a NEW protocol extension method with the extended signature that delegates to the old one by default. Only `LLMModel` and `Qwen35` override the new method. All other conformers unchanged.   |
| 17  | `StablePrefixDetector` searches for sentinel tokens in token stream, but existing `AgentEngine.formatRawPrompt()` operates on decoded text — tokenization is context-dependent, token-level search is unreliable | Replaced with **two-probe technique**: tokenize the same messages with two different dummy user messages, find common prefix length. No sentinel encoding needed. Robust for any template.                                                   |

### v6 blockers (resolved — consistency fixes)

| #   | Blocker                                                                                                                                          | Resolution                                                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 18  | `CachePartitionKey` drops `toolDefinitionsDigest`/`templateContextDigest` without explaining the routing change vs `HTTPPrefixCacheKey`          | Added explicit documentation: digests dropped intentionally because radix tree handles them implicitly (different tools → different tokens → different paths). Existing mismatch diagnostics retained for migration logging. |
| 19  | Design Decision §1 still describes `model.prepare()` returning snapshots, contradicting Task 1.2's `prepareWithCheckpoints()` protocol extension | Removed stale `prepare()` signature change from §1. Single contract: `prepareWithCheckpoints()` as protocol extension.                                                                                                       |
| 20  | `planCheckpoints()` documented as covering "stable-prefix + leaf" but logic says leaf NOT planned + test expects "always includes leaf"          | One rule: `planCheckpoints()` returns mid-prefill checkpoints only (stable prefix, branch points). Leaf captured post-generation via `storeLeaf()`. Test updated to `planCheckpointsNeverIncludesLeaf`.                      |
| 21  | `StablePrefixDetector` type comment still says "sentinel technique" despite §1.3 implementing two-probe                                          | All sentinel references replaced with two-probe.                                                                                                                                                                             |

### v7 blockers (resolved)

| #   | Blocker                                                                                                                                                                                                                                                                                                             | Resolution                                                                                                                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22  | Leaf capture omits the normalization-offset alignment step. Current code (`LLMActor.swift:331-357`) trims attention KV when re-tokenized count < actual cache offset because assistant whitespace normalization shortens the stored conversation. Without this, leaf snapshot offset doesn't match its cache state. | Added explicit **offset-alignment step** between re-tokenization and snapshot capture. Attention layers trimmed by `(actualOffset - storedTokens.count)`. Mamba divergence accepted and documented (same as current prototype). |
| 23  | Token-path extraction does `tokens.asArray(Int.self)` but Qwen3.5 VLM produces 2D `[batch, seq]` tensors. Current code (`LLMActor.swift:525-547`) handles both shapes.                                                                                                                                              | Added `extractTokenSequence()` helper that extracts 1D sequence from either 1D or 2D token tensors via `.dim(-1)` for count and `tokens[0, ...]` or `tokens` for flat access.                                                   |
| 24  | `StablePrefixDetector` call passes `conversation.systemMessages` but `HTTPPrefixCacheConversation` exposes `systemPrompt: String?`, not a messages collection.                                                                                                                                                      | Changed detector to accept `systemPrompt: String?` and construct `[Chat.Message.system(prompt)]` internally. Matches existing conversation shape.                                                                               |

### v8 blockers (this revision)

| #   | Blocker                                                                                                                                     | Resolution                                                                                                                                                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 25  | Cache-hit suffix slicing uses generic `sliceInput()` but live code (`LLMActor.swift:537-553`) handles 1D/2D differently and drops the mask. | Replaced `sliceInput()` with explicit `sliceSuffix()` spec: 1D → `tokens[offset...]`, 2D → `tokens[0..., offset...]`, mask set to `nil`. Matches existing shape-handling contract.                                                                                                                                |
| 26  | Accepted Mamba divergence on normalized leaf hits conflicts with the bitwise-logit-equality correctness gate in Phase 2 and benchmark PC8.  | Logit-equivalence tests and PC8 benchmark explicitly scoped to **mid-prefill checkpoint restore only** (where no normalization occurs). Leaf hits from normalized conversations excluded — they use the same accepted-divergence contract as the current prototype. New test validates the divergence is bounded. |
| 27  | "Open Questions (Resolve Before Phase 1)" section lists unresolved prerequisites that are actually answerable now.                          | Section rewritten as "Implementation Notes" with resolved answers for each, or downgraded to "verify during implementation" with concrete verification steps. No open blockers remain.                                                                                                                            |

### v9 blockers (resolved — paper/repo alignment)

| #   | Blocker                                                                                                                                                                                   | Resolution                                                                                                                                                                                                                                                                               |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 28  | Snapshot size estimates were hand-waved and inconsistent with the actual local Qwen3.5 cache shapes, so memory-budget guidance was not trustworthy.                                       | Replaced them with shape-derived sizing from the local model config: 4K unquantized snapshot ≈ 202 MiB, 8K ≈ 330 MiB, 16K ≈ 586 MiB.                                                                                                                                                     |
| 29  | Phase 2 speculative admission allows exact-path extensions and multiple branch-point candidates, but Marconi admits at most one speculative intermediate checkpoint per sequence.         | Task 2.1 now matches speculative insertion: create at most one `.branchPoint` candidate, only when insertion would split an existing edge and create a new intermediate radix node. Exact-path extensions rely on the leaf checkpoint only.                                              |
| 30  | Phase 2 eviction is underspecified versus Marconi: no recency transform, no min-max normalization, no parent-relative FLOP delta, no eligible-node filter, no single-child collapse rule. | Task 2.3 now specifies the full utility-scored eviction policy: candidates are snapshot nodes with `<= 1` child, utility uses normalized recency plus `alpha *` normalized FLOP efficiency, and single-child nodes collapse after snapshot eviction to preserve radix compression.       |
| 31  | Task 2.4 replaced Marconi's adaptive `alpha` tuning with a manual sweep, so the plan no longer aligned with the paper/reference repo.                                                     | Restored a paper/repo-aligned bootstrap tuner: start with `alpha = 0`, wait for first eviction, use a bootstrap window of `5x` the first-eviction request count (repo default within the paper's 5-15x range), then grid-search `alpha in {0.0, 0.1, ..., 2.0}` by replaying the window. |

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

| className                       | Constructor                                         | State                                                                                       | MetaState                                                                                      |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `"KVCache"` / `"KVCacheSimple"` | `KVCacheSimple()`                                   | `[keys, values]` — offset auto-set from `keys.dim(2)` on state setter (`KVCache.swift:391`) | `[""]` (empty, ignored)                                                                        |
| `"QuantizedKVCache"`            | `QuantizedKVCache(groupSize: parsed, bits: parsed)` | 4 or 6 arrays (wq, scales, [biases])                                                        | `[step, offset, groupSize, bits]` — parse `[2]` and `[3]` for constructor                      |
| `"RotatingKVCache"`             | `RotatingKVCache(maxSize: parsed)`                  | `[keys, values]`                                                                            | `[keep, maxCacheSize, step, offset, idx]` — parse `[1]` for constructor, setter restores all 5 |
| `"ChunkedKVCache"`              | `ChunkedKVCache()`                                  | `[keys, values]` (inherited from KVCacheSimple)                                             | `[chunkSize, startPosition]`                                                                   |
| `"MambaCache"`                  | `MambaCache()`                                      | `[convState, gatedDeltaState]`                                                              | `[""]` (empty)                                                                                 |

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

| #   | Test                                     | What it validates                                                                                                         |
| --- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1   | `captureStoresAllLayerStates`            | 32-layer Qwen3.5 hybrid → 24 MambaCache + 8 attention layer states captured                                               |
| 2   | `captureDeepCopiesArrays`                | Mutating source cache after capture doesn't affect snapshot                                                               |
| 3   | `captureRecordsCorrectClassName`         | MambaCache layers → `"MambaCache"`, KVCacheSimple → `"KVCache"`, QuantizedKVCache → `"QuantizedKVCache"`                  |
| 4   | `captureWithMixedCacheTypes`             | After dynamic quantization: some layers KVCacheSimple, some QuantizedKVCache → both captured correctly                    |
| 5   | `memoryBytesMatchesSumOfTensorSizes`     | Sum of all state array nbytes == memoryBytes                                                                              |
| 6   | `restoreCreatesKVCacheSimple`            | className `"KVCache"` → KVCacheSimple with offset auto-set from keys.dim(2)                                               |
| 7   | `restoreCreatesQuantizedKVCache`         | className `"QuantizedKVCache"` + metaState `["0", "512", "64", "8"]` → QuantizedKVCache(groupSize:64, bits:8), offset=512 |
| 8   | `restoreCreatesRotatingKVCache`          | className `"RotatingKVCache"` + metaState with maxSize → RotatingKVCache(maxSize:), all 5 properties restored             |
| 9   | `restoreCreatesMambaCache`               | className `"MambaCache"` → MambaCache with conv+recurrent state injected                                                  |
| 10  | `restoreCreatesChunkedKVCache`           | className `"ChunkedKVCache"` → ChunkedKVCache with chunkSize+startPosition from metaState                                 |
| 11  | `roundTripCaptureRestorePreservesState`  | capture → restore → compare state arrays: bitwise equal                                                                   |
| 12  | `restoredCacheIsIsolatedFromSnapshot`    | Mutating restored cache doesn't affect snapshot                                                                           |
| 13  | `restoredKVCacheSimpleOffsetMatchesKeys` | After restore, `cache.offset == cache.state[0].dim(2)` (auto-set invariant)                                               |
| 14  | `restoredQuantizedKVCacheHasCorrectBits` | After restore, `cache.bits == 8`, `cache.groupSize == 64` (parsed from metaState, not default)                            |

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

| Component                           | Changes? | Why                                                                          |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `LanguageModel` protocol definition | **No**   | Base `prepare()` signature untouched                                         |
| `LLMModel.swift`                    | **Yes**  | Overrides `prepareWithCheckpoints()` with chunking + checkpoint + tail drain |
| `Qwen35.swift`                      | **Yes**  | Overrides `prepareWithCheckpoints()` with VLM chunking + capture             |
| `TokenIterator.prepare()`           | **Yes**  | Calls `prepareWithCheckpoints()` instead of `prepare()`                      |
| `GenerateParameters`                | **Yes**  | Gains `checkpointAtOffsets` + `checkpointBaseOffset` fields                  |
| Pixtral, Mistral3, Paligemma, etc.  | **No**   | Inherit default extension (delegates to their existing `prepare()`)          |
| `SpeculativeTokenIterator`          | **No**   | Still calls base `prepare()` directly                                        |
| `WiredMemoryUtils.prefillOnly`      | **No**   | Still calls base `prepare()` directly                                        |

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

| #   | Test                                              | What it validates                                                                                            |
| --- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | `emptyCheckpointOffsetsReturnsNoSnapshots`        | `checkpointAtOffsets: []` → standard prepare, empty snapshots                                                |
| 2   | `snapshotCapturedAtAlignedOffset`                 | prefillStepSize=256, checkpoint at 512 → snapshot at 512                                                     |
| 3   | `snapshotCapturedAtMisalignedOffset`              | prefillStepSize=256, checkpoint at 300 → chunk splits: 256+44, snapshot at 300                               |
| 4   | `multipleCheckpointsInSinglePrefill`              | checkpoints at {300, 600, 1000} → three snapshots                                                            |
| 5   | `checkpointBeyondInputSizeIgnored`                | checkpoint at 10000 on 5000-token input → no snapshot                                                        |
| 6   | `chunkSizeNeverExceedsPrefillStepSize`            | All chunks ≤ prefillStepSize                                                                                 |
| 7   | `chunkSizeNeverZero`                              | Checkpoint at offset 0 → no zero-length chunk                                                                |
| 8   | `checkpointDoesNotAlterFinalPrepareResult`        | PrepareResult.tokens identical with/without checkpoints                                                      |
| 9   | `clearCacheCalledBetweenAllChunks`                | Memory.clearCache() after each chunk (mock or log)                                                           |
| 10  | `checkpointInTailCaptured`                        | prefillStepSize=256, input=400, checkpoint at abs 300 → captured in tail drain                               |
| 11  | `checkpointAtLastTokenBeforeTailCaptured`         | prefillStepSize=256, input=500, checkpoint at abs 490 → captured in tail drain                               |
| 12  | `tokenIteratorExposesSnapshots`                   | After `TokenIterator(params: {checkpoints: {100}})`, `iterator.capturedSnapshots` has 1 entry                |
| 13  | `tokenIteratorWithNoCheckpointsHasEmptySnapshots` | Default params → empty `capturedSnapshots`                                                                   |
| 14  | `checkpointRebasedOnCacheHit`                     | baseOffset=3000, absolute checkpoint at 4000 → fires at relative 1000, stored at absolute 4000               |
| 15  | `checkpointBeforeBaseOffsetIgnored`               | baseOffset=3000, checkpoint at 2000 → filtered out (already covered by snapshot)                             |
| 16  | `defaultExtensionDelegatesToBasePrepare`          | Non-LLM model (e.g., Pixtral) → `prepareWithCheckpoints()` delegates to `prepare()`, returns empty snapshots |
| 17  | `specIteratorStillCallsBasePrepare`               | SpeculativeTokenIterator unaffected (still calls base `prepare()` at Evaluate.swift:826,839)                 |

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

| #   | Test                                       | What it validates                                                          |
| --- | ------------------------------------------ | -------------------------------------------------------------------------- |
| 1   | `detectsSystemPlusToolsBoundary`           | System + 5 tools → two-probe common prefix covers system + tool tokens     |
| 2   | `noSystemMessageReturnsNil`                | Empty system → nil                                                         |
| 3   | `stablePrefixIsPrefixOfFullSequence`       | Verification passes: probeA[0..<boundary] == fullTokens[0..<boundary]      |
| 4   | `noToolsDetectsSystemOnlyBoundary`         | System, no tools → offset covers system tokens only                        |
| 5   | `longSystemPrompt8KTokens`                 | 8K-token system + tools → correct boundary                                 |
| 6   | `differentToolsProduceDifferentBoundaries` | Same system, different tools → different offsets                           |
| 7   | `twoProbesDivergeAtUserContent`            | probeA and probeB share exact prefix, diverge at user message              |
| 8   | `probeRobustToSpecialCharsInSystem`        | System prompt with quotes, newlines, template-like content → still correct |

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

| #   | Test                                          | What it validates                                                         |
| --- | --------------------------------------------- | ------------------------------------------------------------------------- |
| 1   | `emptyTreeReturnsNil`                         | Fresh tree → nil                                                          |
| 2   | `insertAndExactMatch`                         | Insert [1..100], store snapshot at 100, lookup [1..100] → snapshot at 100 |
| 3   | `snapshotAtPrefixReturnedOnDivergence`        | Snapshot at 50, lookup [1..50, 200..210] → snapshot at 50                 |
| 4   | `deeperSnapshotPreferred`                     | Snapshots at 50 and 80, lookup [1..100] → snapshot at 80                  |
| 5   | `snapshotBeyondSharedPrefixNotReturned`       | Snapshot at 100, shared prefix only 80 → returns nil or shallower         |
| 6   | `compressedEdges`                             | Insert [1..5] → single edge, not 5 nodes                                  |
| 7   | `splitEdgeOnBranch`                           | Insert [1,2,3,4] then [1,2,5,6] → split at [1,2]                          |
| 8   | `evictSnapshotKeepsNode`                      | Evict snapshot → node stays, just no snapshot                             |
| 9   | `evictLeafCleansAncestors`                    | Evict only leaf → empty ancestors removed                                 |
| 10  | `evictLeafPreservesSiblings`                  | Evict one sibling → other sibling and shared ancestor preserved           |
| 11  | `totalSnapshotBytesAccurate`                  | After insert/evict, counter matches actual                                |
| 12  | `findBestSnapshotOnlyUpdatesReturnedNode`     | Lookup updates returned node's timestamp, not ancestors                   |
| 13  | `50KTokenSequence`                            | Large insert/lookup works                                                 |
| 14  | `eligibleEvictionNodesExcludeMultiChildNodes` | Shared-prefix nodes with 2+ children are protected from Phase 2 scoring   |
| 15  | `collapseSingleChildNodeMergesEdges`          | Snapshot-less intermediate node with one child collapses cleanly          |

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

| #   | Test                                         | What it validates                                                                         |
| --- | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | `lookupEmptyCacheReturnsMiss`                | Cold start → `.missNoEntries`                                                             |
| 2   | `storeAndExactLookupReturnsHit`              | Store + lookup same tokens → hit                                                          |
| 3   | `systemSnapshotSharedAcrossConversations`    | Store sys+userA, lookup sys+userB → hit on system snapshot                                |
| 4   | `leafSnapshotMatchesWithinConversation`      | Store sys+user1+asst1, lookup sys+user1+asst1+tool1 → hit on leaf                         |
| 5   | `lookupReturnsDeepCopiedCache`               | Two lookups get independent cache objects                                                 |
| 6   | `mutatingRestoredCacheDoesNotAffectSnapshot` | Modify restored cache → snapshot unchanged                                                |
| 7   | `evictionRemovesLeafBeforeSystem`            | Type-priority honored                                                                     |
| 8   | `memoryBudgetEnforced`                       | Overflow → eviction triggered                                                             |
| 9   | `planCheckpointsIncludesSystemBoundary`      | Known system offset → included in plan                                                    |
| 10  | `planCheckpointsExcludesExistingSnapshots`   | System snapshot already stored → not re-planned                                           |
| 11  | `planCheckpointsNeverIncludesLeaf`           | Leaf NOT in plan (captured post-generation via storeLeaf, not mid-prefill)                |
| 12  | `snapshotBeyondDivergenceNotReturned`        | Snapshot at 100, divergence at 80 → fallback to shallower                                 |
| 13  | `storeSnapshotsFromPrefillAtCorrectOffsets`  | Mid-prefill snapshots stored at absolute offsets                                          |
| 14a | `storeLeafUnderPostResponseTokens`           | Leaf stored under storedTokens (prompt + response), NOT under promptTokens                |
| 14b | `leafSnapshotOffsetMatchesStoredTokenCount`  | leafSnapshot.tokenOffset == storedTokens.count                                            |
| 14c | `nextRequestHitsLeafViaExtendedPrefix`       | Store leaf under [sys,user1,asst1]. Next request [sys,user1,asst1,tool1,user2] → leaf hit |
| 14  | `statsReflectState`                          | nodeCount, snapshotCount, totalSnapshotBytes accurate                                     |
| 15  | `differentKvBitsIsolated`                    | Store with kvBits=8, lookup with kvBits=nil → miss (separate partitions)                  |
| 16  | `sameKvBitsShared`                           | Store with kvBits=8, lookup with kvBits=8 → hit (same partition)                          |
| 17  | `differentModelIDsIsolated`                  | Store with modelA, lookup with modelB → miss                                              |
| 18  | `evictionCrossesPartitions`                  | Budget pressure evicts from any partition                                                 |

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

| #   | Test                                          | What it validates                                                                                |
| --- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | `endToEndNormalizeTokenizeLookupHit`          | Wire message → normalize → tokenize → store → new request → normalize → tokenize → lookup → hit  |
| 2   | `suffixSlicedAtSnapshotOffset`                | Suffix tokens start at snapshotTokenOffset                                                       |
| 3   | `cacheMissTriggersFullPrefillWithCheckpoints` | Miss → full prefill → system and leaf snapshots captured                                         |
| 4   | `hybridCacheRestoredWithCorrectLayerTypes`    | Restored cache has MambaCache at 0,1,2,4,5,6,... and attention at 3,7,11,...                     |
| 5   | `restoredCacheOffsetMatchesSuffix`            | `cache[faIdx].offset == snapshotTokenOffset` (critical alignment invariant)                      |
| 6   | `normalizationProducesStableTokens`           | Same message, different whitespace → same tokens after normalize                                 |
| 7   | `dynamicQuantizationHandledInSnapshot`        | Early snapshot has KVCacheSimple, later has QuantizedKVCache → both restore correctly            |
| 8   | `snapshotsCapturedDuringPrefillNotPostHoc`    | Stable-prefix snapshot captured at correct offset DURING prefill, not from final cache           |
| 9   | `noDoublePrefill`                             | TokenIterator.init() calls prepareWithCheckpoints() exactly once                                 |
| 10  | `partitionKeyIsolatesDifferentKvBits`         | Store at kvBits=8, lookup at kvBits=nil → miss despite same tokens                               |
| 11  | `stablePrefixIncludesToolTokens`              | Stable prefix checkpoint covers system + tool tokens (two-probe boundary)                        |
| 12  | `checkpointBaseOffsetRebasesCorrectlyOnHit`   | Cache hit at 3K, checkpoint at abs 4K → fires at relative 1K in suffix prefill                   |
| 13  | `leafStoredUnderPostResponsePath`             | After generation, leaf snapshot stored under re-tokenized (prompt + response) path               |
| 14  | `nextTurnHitsLeafFromPreviousTurn`            | Turn N stores leaf. Turn N+1 (extending conversation) hits leaf on lookup                        |
| 15  | `leafOffsetAlignedAfterNormalization`         | Generated response with trailing whitespace → trim aligns attention offset to storedTokens.count |
| 16  | `mambaLayersNotTrimmedInLeafAlignment`        | MambaCache.isTrimmable==false → skip in trim loop (accepted divergence)                          |
| 17  | `vlm2DTokensExtractedCorrectly`               | 2D `[batch, seq]` tensor → flat `[Int]` sequence from first batch element                        |
| 18  | `detectorAcceptsSystemPromptString`           | `conversation.systemPrompt` (String?) → constructs Chat.Message internally                       |

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

## Phase 1 — Completion Status (2026-04-12)

**Status:** shipped. All 8 tasks (1.1–1.8) merged. Integration validated
end-to-end against production OpenCode workloads on Qwen3.5-4B-paro.

### Final commit chain (chronological)

| Task | Commit         | Summary                                                                                                              |
| ---- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1.1  | `c787f1dd`     | `HybridCacheSnapshot` with multi-type KV/Mamba restore                                                               |
| 1.2  | `99c67e0e`     | Checkpoint capture tests (17 plan items + 6 edge cases)                                                              |
| 1.3  | `ca8d625b`     | `StablePrefixDetector` (two-probe)                                                                                   |
| 1.4  | `aecd1e1a`     | `TokenRadixTree` (compressed prefix lookup)                                                                          |
| 1.5  | `988f625a`     | `PrefixCacheManager` with partitioned radix lookup                                                                   |
| 1.6  | `7ac40d3c`     | Integration into `LLMActor` + `CompletionHandler`                                                                    |
| 1.6b | `5acd5039`     | Cross-turn reuse + swift-jinja determinism fix (completed in-place on 1.6's integration)                             |
| 1.7  | `851032aa`     | `HTTPPrefixCacheSpikeStore` and supporting types deleted; tests migrated to normalization + `isPrefix` coverage only |
| 1.8  | (this session) | `HybridPrefixCacheE2E` loaded-model verification — `PrefixCacheE2ERunner`, invoked via `--prefix-cache-e2e` CLI flag |

### What the final session (1.8) actually fixed

Two independent bugs were masking Phase 1's benefits in production:

#### Bug A — swift-jinja 2.3.2 non-deterministic `tojson`

`tesseract.xcodeproj/.../Package.resolved` pinned **swift-jinja 2.3.2** while
`Vendor/mlx-audio-swift/Package.resolved` pinned **2.3.5**. SPM resolution
used 2.3.2, which has:

- `Value.encode(to:)` copies `OrderedDictionary<String, Value>` into a plain
  `[String: Value]` before encoding → insertion order lost.
- `Filters.tojson` does **not** set `JSONEncoder.outputFormatting.sortedKeys`.

Result: the same tool dict rendered to different JSON key orderings on
successive `template.render()` calls within a single process. Every
request's `fullTokens` diverged from the previous request's stored path
somewhere inside the first tool's JSON.

**Fix:** bump `Package.resolved` to swift-jinja **2.3.5** (`revision:
0aeefadec459ce8e11a333769950fb86183aca43`). 2.3.5 uses `value.keys.sorted()`
in `Value.encode` AND sets `.sortedKeys` in `tojson`.

**Defense-in-depth (kept even though the library is fixed):**

- `LLMActor.canonicalizeToolSpecs` round-trips tools through
  `JSONSerialization(options: [.sortedKeys])` before passing to the
  tokenizer, so if the library ever regresses again the radix tree stays
  consistent.

#### Bug B — leaf snapshots unreachable across turns (Qwen3.5 template rewriting)

Qwen3.5's chat template strips `<think>...</think>` blocks from
**non-latest** assistant messages via its reverse-walk
`last_query_index` computation
(`z-lab_Qwen3.5-4B-PARO/chat_template.jinja` lines 67–104). Turn N generates
`assistant1 = <think>X</think>Y` and stores a leaf at the post-response
offset. Turn N+1 re-renders the same history with `assistant1 = Y` (think
block stripped, because it is no longer the most recent assistant). Token
paths diverge at the assistant position, so the stored leaf is unreachable.

**Fix:** capture a **second** mid-prefill checkpoint at the _last-message
boundary_ — the offset where the final history message ends, right before
the `<|im_start|>assistant\n<think>\n` generation prompt. Unlike the leaf,
this offset sits **before** any re-renderable assistant content, so it is
stable across turns: turn N+1's first `lastMessageBoundaryOffset` tokens
match turn N's stored tokens byte-for-byte.

**Implementation details:**

1. `LLMActor` detects the boundary by encoding the known generation-prompt
   string (`<|im_start|>assistant\n<think>\n` for thinking models,
   `<|im_start|>assistant\n` otherwise) and subtracting its length from
   `fullTokens.count`. This avoids needing an `addGenerationPrompt=false`
   path, which the `MLXLMCommon.Tokenizer` protocol doesn't expose.
2. `PrefixCacheManager.planCheckpoints` gained a
   `lastMessageBoundaryOffset: Int?` parameter. Both `stablePrefixOffset`
   and `lastMessageBoundaryOffset` are planned as `.system`-type
   checkpoints when not already stored, with automatic dedup if the two
   offsets coincide.
3. The existing `alreadyStored` logic was tightened to match the
   **requested** checkpoint type, not just any snapshot at the offset, so a
   pre-existing `.leaf` doesn't suppress a needed `.system` checkpoint.

#### Ancillary robustness additions

- `StablePrefixDetector` rejects suspiciously-short common prefixes on
  large prompts (`commonLength < fullTokens.count / 3 && fullTokens.count
  > 1000`). Prevents tree poisoning from any future per-request rendering
  > non-determinism.
- `HTTPRequestLogger` writes every `/v1/chat/completions` request body to
  `tmp/tesseract-debug/http-completions/{HH-mm-ss}-{seq:04d}-request.json`
  for offline investigation.
- `PrefixCacheManager.lookup` on `.missNoSnapshotInPrefix` now returns the
  **actual** tree walk depth (via `TokenRadixTree.findSharedPrefixLength`)
  instead of hardcoded 0, giving meaningful miss diagnostics.

### Measured end-to-end performance

Validated on Qwen3.5-4B-paro, Mac15,9 / 48 GB, running a 51-turn OpenCode
agentic research session (bodyBytes growing from 69 KB to 273 KB as
context accumulated):

| Turn | promptTokens | skippedPrefillTokens | Skip %    | newTokensToPrefill |
| ---- | ------------ | -------------------- | --------- | ------------------ |
| 2    | 15,726       | 0 (cold)             | 0%        | 15,726             |
| 3    | 15,751       | 15,721               | 99.8%     | 30                 |
| 4    | 16,770       | 15,746               | 93.9%     | 1,024              |
| 5    | 16,873       | 15,746               | 93.3%     | 1,127              |
| 10   | 25,049       | 17,021               | 68.0%     | 8,028              |
| 20   | 42,985       | 42,204               | 98.2%     | 781                |
| 30   | 56,391       | 54,194               | 96.1%     | 2,197              |
| 40   | 67,180       | 67,023               | 99.8%     | 157                |
| 50   | 75,057       | 73,723               | 98.2%     | 1,334              |
| 51   | 76,919       | 76,711               | **99.7%** | **208**            |

**Before 1.6b**: every turn was either a full miss (0 skipped) or hit only
the 22-token bare system header. **After 1.6b**: steady-state cache hit
rate ~98%, with one-digit percentages of new tokens prefilled per turn —
matching the Marconi paper's ~98% hit-rate target for agentic workloads.

### Task 1.8 — Loaded-model verification

`PrefixCacheE2ERunner` (`--prefix-cache-e2e` CLI flag) implements the
HybridPrefixCacheE2E scenario from the plan against a real loaded model
(Qwen3.5-4B PARO on Mac15,9). Runs independently of the scenario-turn
benchmark suite because it validates correctness, not tool accuracy.

**7 checks, all passing (2026-04-12 run):**

| Check                                      | Value                                                     | Pass criterion                                                 |
| ------------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------- | ----------- | ------------- |
| `requestA_cold_start`                      | `cachedTokens=0`                                          | Must be 0                                                      |
| `requestB_hits_stable_prefix`              | `cachedTokens=432`                                        | Must be > 0                                                    |
| `requestB_ttft_dropped`                    | `ttftB/ttftA=0.153` (464ms → 71ms, **6.5× speedup**)      | Must be < 0.6                                                  |
| `requestB2_cold_after_reload`              | `cachedTokens=0`                                          | Must be 0 after unload/reload                                  |
| **`greedy_output_equivalence`**            | **byte-identical 126 chars** between warm and cold runs   | Must match the full common-prefix length under greedy decoding |
| `normalization_roundtrip_hits_cache`       | `cachedTokens=440` on second identical request            | Must be > 0                                                    |
| `checkpoint_skips_more_than_system_header` | `cachedTokens=440` (covers full system+tools, not just `< | im_start                                                       | >system\n`) | Must be > 100 |

The `greedy_output_equivalence` check is the critical correctness gate.
Under greedy decoding (`temperature=0, topK=1`), byte-identical outputs
prove the logit argmax at each step matched between the cached and cold
prefill paths, which is the sufficient-by-proxy bitwise logit equality
requirement in the plan. Because we can't reach the raw logit tensor
through `AgentEngine`'s public API, greedy output comparison is the
tightest public-API gate available. Any drift in `HybridCacheSnapshot`
capture/restore would almost immediately flip an argmax within the
first few generated tokens.

Report format: JSON at `tmp/tesseract-debug/benchmark/prefix-cache-e2e/e2e_YYYY-MM-DD_HH-mm-ss.json`
with per-check `pass`/`detail` and full measurement dump (TTFT, cached
tokens, generated text prefix). Log file alongside.

Run time: ~6 seconds (2 model loads, 4 generation passes, short 32-token
max). Run manually before releases or after any changes to `LLMActor`,
`PrefixCacheManager`, `HybridCacheSnapshot`, or `StablePrefixDetector`.

### Test coverage added in 1.6b

- `JinjaNonDeterminismReproTests.swift` (7 tests) — exercise swift-jinja's
  `Value(any:)` + `tojson` directly, including the production flow:
  JSON → `OpenAI.ToolDefinition` → `MessageConverter.convertToolDefinitions`
  → `LLMActor.canonicalizeToolSpecs` → `Jinja.Value` → `JSONEncoder`. Would
  have caught the 2.3.2 bug if we'd had them pre-session. Also covers a
  `Template.render()` path that matches what production executes.
- `StablePrefixDetectorNonDeterminismTests.swift` (7 tests) — reproduce
  non-determinism with a deliberately-flaky mock tokenizer and verify the
  ratio threshold (`fullTokens.count / 3`) rejects poisoned results while
  still accepting legitimate small-prompt detections.
- `PrefixCacheManagerTests.swift` — 3 new tests for multi-checkpoint
  planning: `planCheckpointsIncludesLastMessageBoundary`,
  `planCheckpointsDedupesIdenticalOffsets`,
  `planCheckpointsSkipsExistingLastMessageBoundary`.
- `PrefixCacheIntegrationTests.swift` — 19 tests (all 18 from the plan
  plus `missNoSnapshotReportsActualTreeMatchDepth` added during reviewer
  feedback).

### Known limitations (baseline into Phase 2)

1. **Mamba divergence on leaf-alignment trim is accepted.** When assistant
   whitespace normalization shortens `storedTokens` vs the actual cache
   offset, attention layers are trimmed but Mamba layers are left as-is.
   Impact: negligible in practice (trim amounts are 1–17 tokens, mostly
   whitespace), but it means leaf snapshots can't be bitwise-verified
   against a fresh re-prefill. Phase 2 logit-equivalence tests explicitly
   scope to mid-prefill checkpoints only — leaf hits keep the current
   accepted-divergence contract.

2. **Unit-test coverage is component-level; loaded-model coverage lives
   in the benchmark runner.** The `tesseractTests` suite uses synthetic
   token sequences and validates the contracts (`PrefixCacheManager`,
   `HybridCacheSnapshot`, `StablePrefixDetector`) that the `LLMActor`
   wiring depends on. True end-to-end coverage of the HTTP actor path
   with a real model lives in `PrefixCacheE2ERunner` (Task 1.8),
   invoked via `--prefix-cache-e2e` on the Tesseract CLI. Not a unit
   test — runs for ~6 seconds with real inference, writes a JSON
   report, and exits with non-zero status on any failed check. Run
   manually before releases or after cache-related changes.

3. **Phase 1 uses only type-based LRU eviction.** `.leaf` is evicted before
   `.branchPoint` before `.system`, LRU within a type. No Marconi utility
   scoring yet — that's Task 2.3.

4. **Only one branch-point-style checkpoint per request.** Phase 1 captures
   the stable-prefix boundary and the last-message boundary (both stored
   as `.system`). Speculative `.branchPoint` admission at divergence
   points is Task 2.1.

5. **No FLOP-aware eviction.** All `.system` snapshots are treated equally
   by the eviction policy. Phase 2 adds per-node FLOP cost tracking and
   the `alpha * norm(F/B) + norm(R)` utility score.

6. **Main-agent / subagent cross-eviction (observed in production).**
   Main agent and subagent share a partition (partition key is
   `(modelID, kvBits, kvGroupSize)` — same model → same partition). When
   a subagent runs a long deep-research loop (40+ turns, ~25–30K tokens
   each), its own checkpoints hit frequently and stay fresh, while the
   main agent's much taller 76K-token checkpoint sits untouched and its
   LRU timestamp goes stale. Type-based LRU eviction then picks the
   stale-but-valuable main-agent snapshot over the fresh-but-small
   subagent snapshots.

   **Observed symptom (production, 2026-04-12 session):**
   Main agent builds up to `skippedPrefillTokens=76711` over 50 turns
   (99.7% hit rate), then delegates to a subagent with a different
   toolset (`toolDefinitions=11` vs subagent's `toolDefinitions=5`).
   After the subagent's 40-turn research loop completes and main agent
   resumes, the next main-agent request reports
   `sharedPrefixLength=24` — only the bare `<|im_start|>system\n# Tools
\n\n<tools>\n` header matches. Full 76K-token re-prefill required.

   **Why this is a Phase 1 limitation, not a bug:**
   LRU cannot distinguish "this checkpoint is worth 76K × N_layers × D
   FLOPs" from "this checkpoint is worth 5K × N_layers × D FLOPs". The
   cache evicts the larger-value checkpoint because it hasn't been
   accessed recently. **Marconi's utility score fixes this exactly**:
   `utility = norm(R) + alpha * norm(F/B)` where F is the FLOP savings
   per hit. The 76K checkpoint's F is ~15× the subagent's F, so its
   utility stays high even when R drops. Task 2.3 implements this.

   **Important for parallel subagents (future use case):**
   The target workload is 3–4 subagents running simultaneously alongside
   a long main-agent conversation (80K+ tokens). Under Phase 1 LRU,
   whichever partition is actively hot at any given moment squeezes the
   others out. Under Phase 2 utility-scored eviction, all the tall
   prefixes coexist because their F values dominate the score. This is
   the motivating scenario for Task 2.3 and should be the acceptance
   criterion for Phase 2 completion.

   Workarounds available **within** Phase 1 for users hitting this:
   - Raise `Defaults.prefixCacheMemoryBudgetBytes` in `LLMActor.swift`
     from 3 GiB to 6–8 GiB if unified memory allows. Each additional GiB
     buys roughly 2–5 more snapshot slots, enough for main + 3 subagents
     to coexist without eviction pressure. Cost: proportional increase
     in steady-state RSS.
   - Keep it at 3 GiB and accept that main-agent prefixes are lost after
     long subagent work. Phase 2 is the real fix.

### File inventory (Phase 1 final state)

Production code:

- `tesseract/Features/Agent/LLMActor.swift` — integration entry point, canonicalization, last-message-boundary detection
- `tesseract/Features/Server/PrefixCacheManager.swift` — `@MainActor` store wrapper with multi-checkpoint planCheckpoints
- `tesseract/Features/Server/TokenRadixTree.swift` — compressed radix with `findBestSnapshot` + `findSharedPrefixLength`
- `tesseract/Features/Server/StablePrefixDetector.swift` — two-probe detector + ratio threshold
- `tesseract/Features/Server/HTTPRequestLogger.swift` — file-based request body logging
- `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift` — Task 1.8 loaded-model verification
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` — multi-type cache snapshot

Legacy file **retained** for the normalization layer, not the cache itself:

- `tesseract/Features/Server/HTTPPrefixCacheSpike.swift` — after Task 1.7, this file only contains the normalization types (`HTTPPrefixCacheConversation`, `HTTPPrefixCacheMessage`, `HTTPPrefixCacheToolCall`, `HTTPPrefixCacheAssistantSignature`) and the cache-state inspection helpers (`httpPrefixCacheReportedTokenCount`, `httpPrefixCacheHasReusableState`, `httpPrefixCacheOffsets`). These are still actively used by `LLMActor` (post-generation normalization of the stored assistant turn, cache-offset math for the trim-to-align step) and by `HTTPPrefixCacheSessionReplayStore` (reasoning recovery on cross-request client mismatches). The `HTTPPrefixCacheSpikeStore` actor, `HTTPPrefixCacheKey`/`Match`/`Lookup`/`Entry` types, `HTTPPrefixCacheMismatchReport`, and the `diagnosePrefixMismatch` method were all deleted in Task 1.7 — they were dead code after Task 1.6 replaced them with the radix tree. File name kept for git history continuity.

Tests:

- `tesseractTests/HybridCacheSnapshotTests.swift` — Task 1.1
- `tesseractTests/TokenRadixTreeTests.swift` — Task 1.4
- `tesseractTests/StablePrefixDetectorTests.swift` — Task 1.3
- `tesseractTests/PrefixCacheManagerTests.swift` — Task 1.5 + multi-checkpoint additions
- `tesseractTests/PrefixCacheIntegrationTests.swift` — Task 1.6 (19 tests)
- `tesseractTests/StablePrefixDetectorNonDeterminismTests.swift` — Task 1.8 reproduction
- `tesseractTests/JinjaNonDeterminismReproTests.swift` — Task 1.8 root-cause exploration

### Handoff to Phase 2

Phase 2 starts from a working, production-validated Phase 1. The radix
tree is populated correctly, hit rates are high, and eviction is stable.
Phase 2's job is to make eviction **smart** (utility-scored per Marconi)
and to admit `.branchPoint` checkpoints at speculative divergence points
rather than only at the two Phase 1 boundaries. The file layout and APIs
established in Phase 1 are expected to be stable — Phase 2 should extend
`planCheckpoints` and `findEvictionCandidate` rather than rewriting them.

---

## Phase 2 — Marconi Extensions: Branch-Point Checkpointing & Utility-Scored Eviction

### Task 2.1: Speculative insertion at admission time

**File:** `tesseract/Features/Server/PrefixCacheManager.swift`

Extend `planCheckpoints()` to perform **Marconi speculative insertion**:

1. Walk the tree with the new token sequence as if inserting it.
2. If insertion would **split an existing edge** (the shared prefix continues inside a compressed edge, then diverges), mark the split offset as a `.branchPoint` checkpoint candidate.
3. If the new sequence only **extends an existing path at a node boundary**, do **not** create a `.branchPoint` checkpoint. The leaf checkpoint already covers this case.
4. Admit **at most one** `.branchPoint` candidate per request — independent of the Phase 1 stable-prefix and last-message-boundary checkpoints, which are kept as-is.

**Checkpoint budget — deliberate divergence from the paper.** Marconi's strict reading is "max 2 checkpoints per sequence" (one mid-prefill + one leaf). Tesseract's Phase 1 already ships **two** mid-prefill checkpoints (`stablePrefixOffset` for system+tools reuse, `lastMessageBoundaryOffset` for cross-turn reuse on Qwen3.5's template) plus the leaf, so the actual Phase 1 budget is **3 captures per request**. Task 2.1 adds at most one further `.branchPoint` candidate when a true mid-edge divergence occurs, raising the worst-case Phase 2 budget to **4 captures per request**:

| Phase                        | Mid-prefill | Leaf | Total worst-case | Triggered by                          |
| ---------------------------- | ----------- | ---- | ---------------- | ------------------------------------- |
| Marconi paper                | 1           | 1    | 2                | always                                |
| Tesseract Phase 1            | 2           | 1    | 3                | non-empty system + multi-message body |
| Tesseract Phase 2 (Task 2.1) | up to 3     | 1    | 4                | + mid-edge divergence on the body     |

The extra captures are conditional and rare in practice — branch points only fire on true mid-edge divergence (not node-boundary extensions, not cold cache, not exact-path reuse). Each capture adds one chunk split in `chunkedPrefill` (negligible) and one snapshot allocation (`evictToFitBudget` enforces the global memory budget immediately after store, so steady-state usage stays bounded). The trade is two extra cross-conversation hit paths in exchange for the enlarged budget — the design considers it worth it for Tesseract's primary subagent + Qwen3.5 workload.

```swift
// In planCheckpoints() — Phase 2 addition:
if let splitOffset = tree.findIntermediateSplitOffsetForInsertion(tokens: tokens),
   splitOffset > 0,
   splitOffset < tokens.count,
   !plan.contains(where: { $0.offset == splitOffset })
{
    plan.append((offset: splitOffset, type: .branchPoint))
}
```

**Tests 2.1:**

| #   | Test                                                        | What it validates                                                                               |
| --- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | `divergenceInsideCompressedEdgeCreatesCandidate`            | [1,2,3,4] stored, [1,2,5,6] → split at offset 2                                                 |
| 2   | `exactPathExtensionDoesNotCreateBranchPoint`                | [1,2,3] stored, [1,2,3,4,5] → no speculative checkpoint                                         |
| 3   | `nodeBoundaryDivergenceDoesNotCreateIntermediateCheckpoint` | Existing node [1,2], new child [1,2,9] → leaf only, no `.branchPoint`                           |
| 4   | `coldTreeNoSpeculativeCandidates`                           | Empty tree → no branch-point candidates                                                         |
| 5   | `existingSnapshotNotReCandidate`                            | Re-running planner after a real prior capture (intermediate node materialized) → not re-planned |
| 6   | `atMostOneBranchPointPerSequence`                           | Complex tree, single insertion → at most one speculative candidate                              |
| 7   | `branchPointCoexistsWithSystemCheckpoint`                   | Stable-prefix + branch-point at distinct offsets → both planned                                 |
| 8   | `branchPointSkippedIfSameOffsetAsSystem`                    | Branch-point offset coincides with stable-prefix offset → no duplicate                          |

### Task 2.2: Logit-equivalence verification harness

**The most critical correctness gate.** Required before any further phases. Implemented as `HybridCacheCorrectnessRunner` (loaded-model harness driven via `scripts/dev.sh hybrid-cache-correctness`); not a unit test.

**Scope:** Logit-equivalence tests apply to **mid-prefill checkpoint restores** (stable prefix, branch points) where no normalization-offset trimming occurs. These checkpoints capture and restore state exactly — bitwise logit match is required.

For **leaf hits from normalized conversations**, the Mamba normalization divergence (blocker #22) means bitwise equality is not guaranteed. The original spec proposed a `max|logit_diff| < 0.01` bound; **this turned out to be empirically wrong on Qwen3.5**: the Mamba state mismatch perturbs raw logits by ~10 even at `trim = 1`, while leaving the argmax stable. Argmax stability is sufficient for greedy decoding but not for sampled decoding, and the HTTP server propagates the request's `temperature`/`top_p` which may be > 0.

**Production response (`LLMActor.swift` around line 337):** the offset-alignment block now **skips the leaf store entirely** when normalization would require any non-zero `trimAmount`. The trade-off is lost cache hits on whitespace-normalized conversations in exchange for sampler-agnostic correctness — Tesseract's primary workload is greedy by default and trim is rare in normal ChatML round-trips, so the hit-rate impact is empirically minimal (verified by `PrefixCacheE2ERunner` Step 5, which still passes the cross-turn cache-hit assertion after the guard). Test 9 in this harness now serves as a **diagnostic** showing the divergence math that justifies the guard — it simulates the (now-unreachable in production) trim-and-restore path and characterizes the drift envelope on the current model.

**Tests 2.2 — `HybridCacheCorrectnessRunner` (requires model, ~80s on Qwen3.5-4B):**

| #   | Test                                        | What it validates                                                                                                                                                                                                                                                                                                                                                                |
| --- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `midPrefillRestoreMatchesFullPrefill`       | **CRITICAL.** Mid-prefill snapshot (no normalization): full prefill logits == restore-at-K + suffix logits. K = N/4, N/2, 3N/4. Bitwise match required.                                                                                                                                                                                                                          |
| 2   | `restoreAtExactMatch`                       | Capture at K = N (full prompt length). Live and restored caches forward the same sentinel token; logits must match bitwise. Validates the K = N round-trip path independent of any suffix prefill.                                                                                                                                                                               |
| 3   | `divergentSuffixAfterRestore`               | Checkpoint at K, prefill a divergent suffix on the restored cache; resulting logits must be finite and span vocab (smoke test only — no reference comparison since prompts diverge).                                                                                                                                                                                             |
| 4   | `mambaStateRestoredExactly`                 | Mid-prefill: MambaCache.state arrays bitwise match pre-checkpoint                                                                                                                                                                                                                                                                                                                |
| 5   | `attentionKVRestoredExactly`                | After restore: offset, keys, values match                                                                                                                                                                                                                                                                                                                                        |
| 6   | `quantizedKVCacheRestoredExactly`           | After restore: wq, scales, biases match; groupSize/bits correct                                                                                                                                                                                                                                                                                                                  |
| 7   | `multipleRestoresFromSameSnapshot`          | Two restores → identical logits (isolation)                                                                                                                                                                                                                                                                                                                                      |
| 8   | `longContext16KRestore`                     | 16K-token prompt, checkpoint at 8K, suffix 8K → logits match full 16K bitwise                                                                                                                                                                                                                                                                                                    |
| 9   | `leafHitWithNormalizationDivergenceBounded` | **Diagnostic only — production no longer reaches this path** (LLMActor skips leaf store on `trimAmount > 0`). Simulates the trim-and-restore math, sweeps `trim ∈ {1, 2, 4}`, logs measured `maxAbsDiff` and argmax stability. Pass: argmax stable at `trim = 1` (Phase 1 historical assumption). The original `max\|diff\| < 0.01` bound was unreachable; see scope note above. |
| 10  | `leafHitWithoutNormalizationMatchesBitwise` | Leaf hit where no trimming occurred (0 trim amount): logits match exactly                                                                                                                                                                                                                                                                                                        |

**Possible follow-up (deferred):** the current production guard is conservative — it skips leaf store on **any** trim regardless of the future request's sampling mode. A more permissive variant would tag the snapshot with `requiresGreedy` and only skip the lookup when the new request is sampled, preserving cache hits for greedy clients on normalized conversations. Tag-based gating requires a `HybridCacheSnapshot` schema change; not worth implementing until measured hit-rate data shows the conservative skip is hurting real workloads.

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

| #   | Test                                       | What it validates                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `candidateSetExcludesMultiChildNodes`      | Shared-prefix nodes with `2+` children are never scored                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2   | `parentRelativeFlopsUsed`                  | Same total length, longer unique suffix after parent → higher `deltaF`                                                                                                                                                                                                                                                                                                                                                                                                         |
| 3   | `minMaxNormalizationHandlesDegenerateCase` | All-equal inputs normalize to `1.0`                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 4   | `recentAccessBoostsUtility`                | Newer node gets higher normalized recency                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 5   | `higherFlopEfficiencyBoostsUtility`        | At equal recency, higher FLOPs/byte wins                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 6   | `lowestUtilityEvictedAcrossPartitions`     | Global minimum utility is evicted, not just within one tree                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 7   | `singleChildEvictionCollapsesNode`         | Snapshot eviction on a 1-child node preserves compressed radix structure                                                                                                                                                                                                                                                                                                                                                                                                       |
| 8   | `memoryBudgetRespected`                    | Repeated lowest-utility eviction brings usage under budget                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 9   | `tallMainAgentSurvivesSubagentChurn`       | **Regression test for Phase 1 limitation #6.** Simulate a main-agent checkpoint at offset 76K coexisting with many subagent checkpoints at offset 20–30K. Subagent checkpoints are accessed frequently (recent), main-agent checkpoint is accessed rarely (stale). Under utility scoring with `alpha > 0`, the main-agent checkpoint must survive eviction because its F/B ratio dominates. Under Phase 1 LRU, it would be evicted; Phase 2 must not regress to that behavior. |
| 10  | `parallelSubagentsCoexistWithMainAgent`    | **Acceptance criterion for Phase 2.** Simulate 1 main agent (80K checkpoint) + 3 subagents (25K checkpoints each) all under a shared budget just tight enough to require eviction. All 4 tall-prefix checkpoints must remain in the cache (shorter leaf-like entries are evicted first). Validates that `norm(F/B)` weighting is sufficient to preserve tall context prefixes across sessions.                                                                                 |

**Loaded-model E2E coverage (extends `PrefixCacheE2ERunner`):**

Task 2.1's planner-only landing left a coverage gap: the existing `prefix-cache-e2e` runner only exercises Request A → Request B with same system + different user, which diverges at the _node boundary_ of the system snapshot, not mid-edge. The branch-point capture path is exercised by unit and integration tests but never by a real loaded model.

Task 2.3 closes that gap because utility-scored eviction must distinguish branch-point snapshots from system snapshots in production conditions. Add a 3-request scenario to `PrefixCacheE2ERunner` that:

1. **Request A** — system + user₁ → fully cold; captures stable-prefix + last-message-boundary + leaf
2. **Request B** — same system + user₁ + assistant₁ + user₂ → diverges _mid-edge_ of A's leaf path; planner emits a `.branchPoint` candidate at the divergence offset; the captured snapshot must land tagged `.branchPoint` (verifiable via `tmp/tesseract-debug/http-completions/` request mirror or via `PrefixCacheManager.stats`)
3. **Request C** — same system + user₁ + assistant₁ + user₃ → re-hits the branch point captured by Request B; reported `cachedTokens` should match the branch-point offset, _not_ the stable-prefix offset

Assertions to add alongside the existing 7 checks:

- `requestB_captures_branch_point` — at least one `.branchPoint`-tagged snapshot exists in the cache after Request B
- `requestC_hits_branch_point` — Request C's `cachedTokens` equals the offset of the `.branchPoint` snapshot, not the stable-prefix offset
- `branch_point_survives_under_pressure` — repeat Request C N times; under utility scoring with `alpha > 0`, the branch-point snapshot must outlive shorter siblings even if they're more recent (folds the `tallMainAgentSurvivesSubagentChurn` and `parallelSubagentsCoexistWithMainAgent` invariants into the loaded-model harness)

This bundles the Task 2.1 deferred E2E coverage with Task 2.3's natural verification needs — one harness, one runtime envelope (~6 seconds plus the extra requests), instead of two one-shot runners.

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

| #   | Test                                             | What it validates                                      |
| --- | ------------------------------------------------ | ------------------------------------------------------ |
| 1   | `startsAtZeroBeforeFirstEviction`                | Cache behaves like pure-recency scoring before tuning  |
| 2   | `bootstrapWindowUsesFiveTimesFirstEvictionCount` | Tuning window matches repo default multiplier          |
| 3   | `gridSearchChoosesBestAlphaByFlopsSaved`         | Replayed window selects the top-FLOPs-saved `alpha`    |
| 4   | `tunedAlphaUsedForSubsequentEvictions`           | After tuning, eviction utility uses the chosen `alpha` |

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

| #   | Test                             | What it validates                                 |
| --- | -------------------------------- | ------------------------------------------------- |
| 1   | `alignedSnapshotSinglePass`      | Snapshot at exact divergence → one pass           |
| 2   | `largeGapTriggersTwoPass`        | Gap > 256 → two-pass, new snapshot created        |
| 3   | `smallGapSinglePass`             | Gap ≤ 256 → no extra snapshot                     |
| 4   | `twoPassLogitsMatchFullPrefill`  | **CRITICAL** — correctness gate                   |
| 5   | `newSnapshotFromTwoPassReusable` | Subsequent request benefits from mid-gap snapshot |

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

| #   | Test                     | What it validates                           |
| --- | ------------------------ | ------------------------------------------- |
| 1   | `budgetScalesWithRAM`    | More RAM → higher budget                    |
| 2   | `budgetCapped`           | Max reasonable limit even on large machines |
| 3   | `budgetAccountsForModel` | Bigger model → smaller budget               |

### Task 3.4: Diagnostic logging

Structured logging via `Log.agent`:

- Hit/miss reason + token counts + snapshot offset + checkpoint type
- Snapshot capture events (offset, type, bytes, during-prefill confirmation)
- Eviction events (type, score, freed bytes)
- TTFT breakdown (lookup, restore, prefill, first-token)
- Memory (snapshots total, model, free pool)

### Task 3.5: E2E benchmark suite

**New file:** `tesseract/Features/Agent/Benchmark/PrefixCacheBenchmark.swift`

| Scenario | Description                                        | Expected                                                                                                              |
| -------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| PC1      | Cold: 16K, no cache                                | Baseline TTFT                                                                                                         |
| PC2      | Warm: repeat exact 16K                             | Near-zero TTFT                                                                                                        |
| PC3      | Shared system: same 4K system + different user     | ~50-70% TTFT reduction                                                                                                |
| PC4      | Tool change: same system, different tools          | Hit on system snapshot only                                                                                           |
| PC5      | Long conversation: 20 turns                        | TTFT stable (prefix reuse)                                                                                            |
| PC6      | Subagent switch: main→sub→main                     | Hit on return                                                                                                         |
| PC7      | Memory pressure: fill + new request                | Graceful eviction                                                                                                     |
| PC8      | **Logit equivalence: mid-prefill restore vs full** | **Bitwise match** (mid-prefill checkpoints only; leaf hits with normalization trimming use bounded divergence < 0.01) |
| PC9      | Normalization: whitespace variants                 | Hits despite differences                                                                                              |
| PC10     | Two-pass: misaligned snapshot                      | New snapshot created                                                                                                  |
| PC11     | QuantizedKVCache round-trip                        | Correct bits/groupSize after restore                                                                                  |

---

## Phase 4 — Draft Ideas (exploratory, not development-ready)

**Status:** Brainstorm. None of these are scoped, blocker-reviewed, or sequenced. They are candidates for enhancement _after_ Phases 0–3 ship and we have a production trace to optimize against. Each idea lists an estimated win, the open questions that would need answers before scoping, and the code areas most likely to change. **Do not implement from this section directly.**

The ordering reflects rough return-on-effort for Tesseract's single-user Apple Silicon workload — SSD persistence is the first candidate because it directly attacks the biggest remaining failure mode (cold restart throws away every snapshot, including the `stablePrefix` + tool-definition snapshot that is the most expensive to rebuild and the most universally shared across conversations).

### 4.1 — SSD persistence tier

**Status:** Scoped draft. Supersedes the prior brainstorm block. Awaits review against the "Decisions assumed" and "Open questions" sections at the end before greenlighting tasks. Sequenced against Phase 3 (see "Phase ordering" under decisions).

**Motivation.** Phase 1–3's cache lives entirely in RAM. Every app restart, model swap, or OOM-triggered unload drops the full radix tree — including the stable-prefix snapshot, which is the single most valuable entry because it is reused by every conversation against the same system prompt + tools. On the first post-restart request the user pays a full cold prefill for system + tools (~4K tokens on the current agent prompt) even though nothing about that prefix has changed. Every time the user hot-swaps the model for a benchmark run (common in Tesseract's own dev loop) the cost repeats. oMLX (`jundot/omlx`) addresses this with a write-through safetensors tier on SSD (`PagedSSDCacheManager`, `BoundarySnapshotSSDStore`) and retains cache hits across process restarts. The same shape fits Marconi, with modifications to account for our token-level radix tree (vs oMLX's fixed-size chained-hash blocks) and our Apple Silicon unified-memory footing (vs oMLX's CUDA-patterned write path).

**Target win.**

- Post-restart stable-prefix hit: cold → ~40–80 ms NVMe read + snapshot reconstruct on Apple Silicon for a ~200 MiB `.system` snapshot, versus ~1–3 s cold prefill today. Net win ~1–3 s on the first request after every launch and after every hot model swap.
- Capacity relief: RAM budget can stay conservative (low single-digit GiB fraction of the unified-memory headroom) while the SSD tier absorbs overflow, extending the effective working-set size into the tens of GiB without touching the unified-memory envelope or fighting the Metal allocator.
- Preserves leaf snapshots across short app restarts, short model unloads, and Xcode rebuild cycles during active development.

**Non-goals (Phase 4.1).**

- Snapshot quantization at capture or serialization time (Idea 4.2).
- Pre-hydration of top-K snapshots on launch (Idea 4.3).
- VLM image feature cache (Idea 4.4).
- Predictive tool-loop prefill (Idea 4.5).
- Cross-model sharing (Idea 4.6).
- Cross-process access. This tier is single-process only, matching oMLX and the existing single-user scope. Benchmarks that run while the app is live use a per-PID subdirectory.
- A web UI for cache contents. Diagnostics live in `PrefixCacheDiagnostics` and the existing logs only.

**Prior art imported (with the specific lesson).**

- **oMLX `PagedSSDCacheManager`** — the working implementation closest to ours. Lessons:
  - _Metal-deadlock rule._ oMLX has a regression test (`test_load_no_executor_deadlock`) asserting that `mx.load()` is **never** dispatched to a background thread. A prior executor-based approach deadlocked against the inference path because Metal command-buffer submission is thread-pinned. The fix is: **SSD reads happen on the inference thread** (inside `container.perform { ... }`), synchronously, accepting the ~40 ms read latency. Our design adopts this rule verbatim.
  - _Two-step write._ The inference thread runs `eval(cache)` on the live MLX arrays, then extracts raw `Data` from each array (Metal-safe after eval), then hands the raw bytes to a background writer. Only the raw-bytes handoff crosses threads, not the `MLXArray` itself. This keeps Metal resources unpinned and lets the writer thread do pure file I/O.
  - _Atomic temp-then-rename._ Writes go to `{snapshotID}.tmp.safetensors`, fsync, rename to `{snapshotID}.safetensors`. The rename is atomic on APFS, so crash-during-write leaves the old file intact (or no file at all). We copy this verbatim.
  - _Bounded queue + coalescing._ Writer queue depth scales with total RAM (`max(32, min(256, total_ram_gb / 2))`). If a new write arrives for a snapshot whose write is already in-flight, drop the older write. Queue-full → drop-oldest-pending. No blocking of the producer. We adopt this.
  - _oMLX's weak spots we **fix** in our design:_
    - No weight fingerprint in the partition key → silent stale hits when the user replaces weights under the same model ID. We fold a `modelFingerprint` into `CachePartitionKey`.
    - Scan-every-safetensors-header at startup → O(N) parse of every cache file's metadata block on launch. We persist a sidecar `manifest.json` so warm start is a single file read.
    - Write-back mode (their "hot cache" path) has a latent deadlock when the queue fills while the hot cache evicts. We ship **write-through only** in Phase 4.1 and revisit write-back only if write amplification becomes a measurable problem in Phase 4.1.b.
    - One file per block with no indirection → 100k-file directory on large caches (bad for APFS, Spotlight, Time Machine). Our unit is a whole 50–600 MiB snapshot, not a 64-token block, so file count is structurally bounded at ≲ cache-budget / min-snapshot-size — low hundreds at most — and the problem does not bite us.
- **SGLang HiCache (Strata / LMSYS 2025-09)** — the direct shape-match for a radix-tree-with-disk-tier. Lesson: **selective write-through on reuse**. Do not persist a snapshot the first time it is captured; persist only after it has been hit at least once from RAM. This filters ephemeral branch points (a heavy Phase 2 workload could generate hundreds of MiB of speculative captures per minute; most are useless) and is strictly stronger than the existing plan's "survive one eviction pass" rule because it measures usefulness directly instead of using age as a proxy. We adopt this.
- **DuckDB Adaptive Radix Tree persistence (2022 blog post)** — canonical header-scan-then-lazy-hydrate pattern. Post-order serialization with swizzled child pointers; single container file. We do not adopt the single-container-file + swizzled pointers (our unit is a whole-snapshot blob, not a small node; the post-order append-only constraint is wrong for our churn). We **do** adopt the shape of "manifest is authoritative for tree structure; bodies are lazy." Our manifest is a JSON file, not a post-ordered binary, because our churn is low and JSON is human-debuggable.
- **LMCache + CacheGen (SIGCOMM 2024)** — layered KV store + wire format. Lesson: **custom binary beats safetensors for chunked reads**, but our reads are whole-blob (one snapshot per file), so `savePromptCache()`'s existing safetensors format is fine. We keep safetensors as the on-disk format and wrap with a metadata header. Do not over-engineer.
- **Mooncake (FAST 2025)** — `cache_salt` for partition isolation. Lesson: fold the weight identity into the hash seed. We do the same via `modelFingerprint`.
- **KVSwap (arXiv 2511.11907, on-device inference on Jetson)** — the most directly applicable on-device measurement: **effective NVMe bandwidth collapses below ~6% at 512-byte reads** due to controller read-amplification. Read granularity must be multi-MB. Our 50–600 MiB snapshots read as whole files, so we are naturally on the favorable side of this curve. Lesson: **do not chunk internally** in Phase 4.1 — one snapshot = one file = one read.
- **AttentionStore / CachedAttention (ATC 2024)** — layer-by-layer overlap of write and compute so the forward pass never stalls. We do not need this in Phase 4.1 because Apple Silicon unified memory eliminates the GPU→CPU staging copy that dominates the hot path on CUDA systems; the whole snapshot is already in writable memory immediately after `eval(cache)`. Phase 4.1.b can revisit if per-layer streaming turns into a measurable win.

**Apple Silicon advantage (why Phase 4.1 is simpler than any datacenter paper).** The CUDA literature spends most of its complexity budget on pinned host memory, dedicated CUDA streams, per-layer overlap, and PCIe bandwidth analysis because on a discrete GPU the GPU→CPU copy is the bottleneck. On Apple Silicon unified memory there is no explicit staging copy — after `eval(cache)`, the tensors are already in a region the CPU can read. Our write path therefore collapses to: `eval → memcpy to `Data` → enqueue → write`. Our hydration path collapses to: `mx.load → reconstruct snapshot → return`. The entire datacenter layered-allocator / stream-management apparatus drops out. Keep this in mind while reviewing — the simplicity is load-bearing, not a sign of missing rigor.

---

**Design note — how Phase 4.1 extends Marconi (read this before the write-path section).**

Tesseract's current eviction policy (Phase 2) is a faithful implementation of _Marconi: Prefix Caching for the Era of Hybrid LLMs_ (Pan et al., MLSys 2025, [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)). The utility score at `EvictionPolicy.swift:151-193` matches the paper's Eq. 1/2 verbatim: `S(n) = recency(n) + α · flop_efficiency(n)`, with min-max normalization across the candidate set and α tuned by the adaptive bootstrap (Phase 2 alpha tuner). Phase 4.1 extends Marconi with an SSD tier using a **deliberately simpler** scoring policy that is equivalent to the RAM tier at today's production α=0 default. It is not a claim that the same formula runs on both tiers — an earlier draft made that claim and review flagged it as stronger than the descriptor schema supports.

**What Marconi specifies, and how we honor it on the RAM tier:**

- _Utility formula._ Unchanged on RAM. Phase 4.1 does not touch `EvictionPolicy.computeScores`. The full `S(n) = recency + α · flop_efficiency` formula continues to drive RAM eviction with the alpha tuner's live α.
- _Topological protection._ The paper protects nodes with ≥2 children. Phase 2 filters the RAM-tier eligible set by `childCount <= 1` (`TokenRadixTree.eligibleEvictionNodes()`). Phase 4.1 does not change this filter.
- _Selective admission._ Marconi §4.1 admits "purely-input" prefixes only on the second occurrence, and "input-and-output" prefixes unconditionally at end of decode. Tesseract's current Phase 2 approximates this with its two-checkpoint scheme (stable-prefix + last-message-boundary). Phase 4.1 inherits this admission policy without modification — the SSD tier writes through whatever Phase 2 admits to RAM, using the same admission decisions. **We do not introduce a second, tier-specific admission gate that would filter what gets persisted based on type — every persist-eligible capture is enqueued to SSD at capture time, see decision 15.**
- _α adaptive tuning._ The Phase 2 alpha tuner is untouched. Phase 4.1's Task 4.1.7 adds a precondition: the tuner's sandbox `PrefixCacheManager` is constructed with SSD disabled so trace replay never touches the real SSD directory.

**Where we extend Marconi (with the paper's blessing, since §6 cites CachedAttention and Pensieve as legitimate multi-tier directions Marconi chose not to build):**

1. **Tiering.** Marconi is RAM-only. Phase 4.1 adds an SSD persistence tier whose own eviction policy is a simpler LRU-with-type-protection — see the next bullet for why this is not "Marconi on SSD" verbatim.
2. **`.system` type protection (on both tiers).** Marconi's protection is topological: ≥2 children. On a single-user workload Tesseract does not reliably produce branch topology at the system-prompt boundary (there is only one active session most of the time, so the stable-prefix node never gets a sibling that would turn it into a branch). We substitute an explicit `.system` type tag that has the same semantic effect. On the RAM tier this is already implemented as a filter in `eligibleEvictionNodes()`. On the SSD tier, the writer filter checks `descriptor.checkpointType != "system"` before selecting a victim, **and in the degenerate case where all non-`.system` residents are exhausted, the writer drops a non-`.system` incoming rather than evicting any `.system` resident** (see Eviction + demotion bullet 4 for the full rule, and P1 of 2026-04-14 for the bug fix). The substitution is semantic equivalence, not verbatim match — if Tesseract ever grew to multiple active sessions with shared system prompts, Marconi's topology protection would fire naturally and the type tag would become redundant.
3. **SSD-tier eviction is type-protected LRU, which equals Marconi at α=0 pre-tuning.** This is the place to be honest: the SSD tier does **not** run the full `EvictionPolicy.computeScores` formula. The writer processes descriptors from an actor-isolated state and does not have access to live radix-tree inputs (`parentTokenOffset` for the FLOP term, `childCount` for topological eligibility). A previous draft claimed "same formula on both tiers, incoming is a scored candidate" — review flagged that as stronger than the descriptor schema supports, because `PersistedSnapshotDescriptor` does not carry `parentTokenOffset` or `childCount`. Phase 4.1's honest answer is: **the SSD tier runs recency-only LRU with `.system` type protection, which is exactly what `EvictionPolicy.computeScores` produces at α=0.** At startup, before the Phase 2 alpha tuner runs its grid search, `EvictionPolicy.alpha` is `0.0` (`EvictionPolicy.swift:68`, reset in `LLMActor.swift:666`), and the two policies are behaviorally equivalent. **After the alpha tuner writes `bestAlpha` into `EvictionPolicy.alpha`** (`AlphaTuner.swift:193`), the RAM tier moves to the full formula while the SSD tier stays at LRU. Phase 4.1 **accepts** this divergence: RAM evicts based on the tuned Marconi utility, SSD evicts based on pure recency, and the two tiers run subtly different policies until Phase 4.1.b extends the descriptor schema + promotes the writer's scoring to the full formula. The practical impact of the divergence is expected to be small because (a) the recency term typically dominates the utility score in workloads where α is modest, (b) the `.system` type protection holds on both tiers regardless of α, and (c) the divergence is **scoped**: RAM scoring governs only which entries stay RAM-resident; SSD persistence is decided at capture time by the write-through-at-capture path, so post-tuning RAM scoring cannot retroactively change which entries are on disk. Upgrade path: if the alpha tuner raises α above 0 AND production traces show a measurable RAM-vs-SSD divergence penalty, Phase 4.1.b extends the descriptor schema (`parentTokenOffset`, stored `childCount`) and promotes the writer's scoring to the full formula. Decision 21 locks in the Phase 4.1 shipping behavior; decision 27 locks in the post-tuning divergence acceptance.

**Where the paper is silent and we make a deliberate engineering choice:**

- _Caps on specific entry types._ Marconi has no quota system. Phase 4.1 **rejects** a per-type cap (e.g., the earlier draft's "N=4 most-recent `.lastMessageBoundary` on SSD") because (a) it violates Marconi's "one formula, one knob" convention, (b) the recency term already penalizes stale boundaries, and (c) the top-level `prefixCacheSSDBudgetBytes` (20 GiB default) is a single cap that bounds total growth without discriminating by type. If production traces ever show the recency term is insufficient for bounding `.lastMessageBoundary` growth, Phase 4.1.b can add a per-type admission filter — which Marconi's authors would call an admission gate, matching the paper's admission-first philosophy.
- _Cost-aware utility._ Marconi treats hits as free. A true tier-aware formula would weight `F/B` by `1 / readLatency(tier)` so a RAM hit (~1 ms) counts differently from an SSD hit (~40 ms). Phase 4.1 does **not** do this — we use recency-only on SSD because (a) we don't have measured tier-hit latencies yet, (b) the descriptor schema does not carry the FLOP inputs, so any tier-aware formula on SSD would require the same schema extension as full Marconi, and (c) the RAM tier's formula is unchanged so there is nothing asymmetric on the RAM side either. Phase 4.1.b is the natural home for both upgrades (schema extension + cost-aware weighting) and they should be scoped together.

**What this means for reviewers.** Three specific claims to anchor against:

1. The RAM tier is **unchanged** — full Phase 2 Marconi with the alpha tuner's live α. The eviction loop in `PrefixCacheManager.evictToFitBudget` is untouched except for the victim disposition (body-drop if storageRef is present, hard-delete otherwise).
2. The SSD tier is **type-protected recency-only LRU**, which is Marconi at α=0. At startup (and before the alpha tuner raises α) the two policies produce the same eviction decisions; once the tuner writes `bestAlpha > 0`, they diverge subtly. Phase 4.1 accepts the divergence as a deliberate trade (decision 27). Any "same formula across both tiers" language in older drafts has been replaced with this narrower framing.
3. The type protection invariant holds **regardless of α**: `.system` entries are never evicted on either tier while any non-`.system` candidate remains. The tuner cannot produce an α value that unprotects `.system` — the filter is upstream of the scoring.

---

**Architecture overview.**

**Initialization flow (once per model load):**

```
SettingsManager (@MainActor, @Observable)
  │   prefixCacheSSDEnabled, prefixCacheSSDBudgetBytes, prefixCacheSSDDirectoryOverride
  │
  ▼ settingsManager.makeSSDPrefixCacheConfig() -> SSDPrefixCacheConfig?
  │   (synchronous; returns nil if disabled; snapshot is immutable and Sendable)
  │
AgentEngine.loadModel(from:visionMode:)  [@MainActor]
  │
  ▼ llmActor.loadModel(from: dir, visionMode: vm, ssdConfig: snapshot)
  │
LLMActor                                 [its own actor]
  │
  ├─ self.ssdConfig = snapshot           (actor-isolated stored property,
  │                                       cleared on unloadModel())
  │
  ▼ later: ensurePrefixCache()
  │
  └─ if let config = self.ssdConfig {
       constructs SSDSnapshotStore(config: config) once, attaches to
       PrefixCacheManager via its new ssdStore: parameter. Writer reads
       rootURL / budgetBytes / maxPendingBytes from the captured config.
     }
     else {
       PrefixCacheManager is constructed with ssdStore: nil. All SSD code
       paths collapse to RAM-only. No disk I/O, no writer task, no
       extraction memcpy. This is the production fallback when SSD is
       disabled in settings AND the fallback for all benchmark runners
       that construct `AgentEngine()` without an ssdConfig (see the call
       site table under AgentEngine.swift in Files-to-change).
     }
```

**Request flow (per capture / lookup):**

```
LLMActor (non-MainActor, holds ModelContainer + self.ssdConfig)
  │
  ▼ container.perform { context in ... }       ← Metal-affine scope
  │   │
  │   ├─ extractSnapshotPayloads() guards on `self.ssdConfig?.enabled == true`
  │   │     (synchronous actor-isolated read; no await, no MainActor hop)
  │   │
  │   ├─ chunkedPrefill captures HybridCacheSnapshot (RAM deep-copy, still MLX)
  │   ├─ asData() on every LayerState array → SnapshotPayload (Data + dtype + shape)
  │   │     • pure Sendable value types, no MLX references escape the scope
  │   └─ (hydrate path, on SSD hit) mx.load(file) → reconstruct HybridCacheSnapshot
  │
  ▼ MainActor.run { prefixCache.storeSnapshots/storeLeaf }   ← SYNCHRONOUS
  │                                                           ← no `await` inside
PrefixCacheManager (@MainActor, all methods non-async)
  │
  ├─ RAM tier (existing TokenRadixTree, nodes grow a storageRef sibling)
  │     ← runs current Phase 2 Marconi eviction unchanged
  │
  └─ sync under lock → SSDSnapshotStore.tryEnqueue(payload, desc, node)
       │                (nonisolated final class; NSLock-protected front door;
       │                 byte-bounded via maxPendingBytes; drops oldest pending
       │                 on overflow; attaches pending storageRef to node;
       │                 returns TryEnqueueResult synchronously)
       │
       ▼ writer runs in its own detached Task:
       SSDSnapshotStore (final class, lock-protected queue)
         ├─ writerLoop: drains pending under lock, processes one at a time
         ├─ admission cut: type-protected LRU with asymmetric fallback
         │     (α=0 Marconi; evict oldest non-.system; if only .system left,
         │      drop non-system incoming OR laterally evict .system on system incoming)
         ├─ writes safetensors.tmp → fsync → atomic rename
         ├─ fires Task { @MainActor in markStorageRefCommitted(id:) } on success
         ├─ fires Task { @MainActor in markStorageRefDropped(id:) } on failure
         ├─ manifest: in-memory index + 500ms debounced persist
         └─ directory under SSDRoot (from config.rootURL, captured at init)
```

The tree is still a `TokenRadixTree`. `RadixTreeNode.snapshot: HybridCacheSnapshot?` grows a sibling field `storageRef: SnapshotStorageRef?` that points at the SSD file once the write-through lands. Lookup prefers resident, falls back to SSD-only, and SSD-only hits are resolved synchronously by the caller (`LLMActor`) inside `container.perform` via `ssdStore.loadSync(storageRef:, expectedFingerprint:)` — a **nonisolated instance method** on `SSDSnapshotStore` (not a static helper; it needs `self.rootURL` to derive the file URL, see Read/hydration path bullet 4). RAM eviction is a body-drop with a cleanup guard: `node.snapshot = nil` leaves the `storageRef` intact, and the eviction loop skips `evictNode` / `collapseSingleChildNode` when `node.storageRef != nil`, so the next lookup matches and hydrates from disk.

**On-disk layout.**

```
{SSDRoot}/
  manifest.json                               # authoritative tree-structure index; debounced writes
  manifest.json.tmp                           # atomic-rename staging
  partitions/
    {partitionDigest}/                        # 8-hex-char FNV of CachePartitionKey including modelFingerprint
      _meta.json                              # partition-level fingerprint + human-readable key
      snapshots/
        {shardByte}/                          # 0-f hex bucket for APFS load balance
          {snapshotID}.safetensors            # whole HybridCacheSnapshot, one file
          {snapshotID}.safetensors.tmp        # atomic staging
```

`SSDRoot` defaults to `FileManager.default.url(for: .cachesDirectory, ...)` — which the app-sandbox redirects to `~/Library/Containers/{bundleID}/Data/Library/Caches/prefix-cache`. Rationale: caches are "recoverable if deleted," macOS purges them under disk pressure (free safety valve), and the semantic fit matches the data (derivative from weights + tokens, not user-owned).

**Data model.**

```swift
// A single persisted snapshot. One file on disk, one entry in manifest.
//
// IMPORTANT — what the descriptor does and does not carry.
// The descriptor carries only the inputs the SSD-tier eviction policy
// actually consumes. Phase 4.1 uses type-protected LRU on SSD (see
// decision 21: SSD-tier scoring is Marconi at alpha=0, which reduces to
// LRU within the eligible set). That means we need checkpointType for
// type protection and lastAccessAt for recency — nothing else. In
// particular, we deliberately do NOT store parentTokenOffset or
// childCount here, because those inputs are only meaningful with live
// radix-tree state. The writer actor cannot inspect the tree from its
// own isolation domain, so a full Marconi eviction on SSD would require
// either (a) crossing into MainActor for every scoring call or (b) a
// descriptor extension that snapshots tree structure at write time.
// Both are Phase 4.1.b concerns gated on production traces that show
// the alpha tuner raising alpha above 0.
struct PersistedSnapshotDescriptor: Codable, Sendable {
    let snapshotID: String               // UUID-shaped, stable across restarts
    let partitionDigest: String          // 8-hex from CachePartitionKey
    let pathFromRoot: [Int]              // radix tree path, enables tree rebuild
    let tokenOffset: Int
    let checkpointType: String           // "system" | "lastMessageBoundary" | "leaf" | "branchPoint"
    let bytes: Int                       // on-disk file size
    let createdAt: Double                // Date since reference
    var lastAccessAt: Double             // Date since reference — bumped to .now by SSDSnapshotStore.recordHit(id:) on every hit (RAM state-4 lookup OR SSD state-5 hydration); sole eviction input under alpha=0
    let fileRelativePath: String         // "partitions/abc12345/snapshots/f/{id}.safetensors"
    let schemaVersion: Int               // current = 1
}

// Per-partition side metadata.
struct PartitionMeta: Codable, Sendable {
    let modelID: String
    let modelFingerprint: String         // hex SHA-256 over config.json bytes + tokenizer.json bytes + sorted [(filename, size, mtime)] for every *.safetensors in the model dir (see decision 5)
    let kvBits: Int?                     // nil for unquantized
    let kvGroupSize: Int
    let sessionAffinity: String?
    let createdAt: Double
    let schemaVersion: Int               // current = 1
}

// The full manifest, persisted to manifest.json, rebuilt from scratch if corrupt.
struct SnapshotManifest: Codable, Sendable {
    var schemaVersion: Int               // current = 1; mismatch → wipe
    var partitions: [String: PartitionMeta]            // keyed by partitionDigest
    var snapshots: [String: PersistedSnapshotDescriptor]  // keyed by snapshotID
}

// In-memory reference attached to RadixTreeNode when an SSD write is
// in flight (committed == false) or on disk (committed == true).
// Lifecycle in "Storage ref lifecycle" section below.
struct SnapshotStorageRef: Sendable {
    let snapshotID: String
    let partitionDigest: String
    let tokenOffset: Int
    let checkpointType: HybridCacheSnapshot.CheckpointType
    let bytesOnDisk: Int
    var lastAccessTime: ContinuousClock.Instant
    /// false = write enqueued but not yet committed to disk;
    /// true  = file exists, fsync'd, and validated against the expected fingerprint.
    /// Lookups that land on a ref with committed == false treat the node
    /// as a miss (the file does not yet exist), so no race can surface
    /// a half-written file to a lookup caller.
    var committed: Bool
}
```

The safetensors header for each snapshot file carries the critical invariants for verification on load:

```
metadata: {
    "schema_version": "1",
    "snapshot_id": "...",
    "model_id": "...",
    "model_fingerprint": "...",
    "kv_bits": "8",                      // or "none"
    "kv_group_size": "64",
    "token_offset": "4123",
    "checkpoint_type": "system",
    "created_at": "2026-04-14T12:34:56Z",
    "path_length": "42",
    "path.0": "151643",                  // token ids as decimal strings
    "path.1": "9707",
    ...
    "path.41": "27"
}
```

The path ids live in the safetensors metadata (not only the manifest) so that if the manifest is corrupt and we need to rebuild it, a directory walk + header parse is enough — no data loss. The manifest is the fast path; the safetensors headers are the recovery path.

---

**Write path (write-through at capture, non-suspending admission).**

The core invariant: **the SSD write is initiated at the moment the snapshot is captured inside `container.perform`, not at the moment it is later evicted from RAM.** This is load-bearing — it means (a) the `asData()` extraction always happens on the inference thread with clear Metal affinity, (b) demotion at eviction time is a trivial body-drop because the SSD copy already exists, (c) the MainActor admission call stays non-suspending so the existing `MainActor.run { ... }` closures in `LLMActor.swift:337`, `:487`, and `:1383` keep their current shape.

There is **no selective-write-through gate** in Phase 4.1. Every persist-eligible snapshot is extracted and enqueued to SSD at capture time. The earlier draft had a gate deferring `.branchPoint` captures until first reuse (SGLang HiCache pattern), but we rejected it because: (1) the gate would require re-extraction on hit, which crosses the Metal-affinity boundary; (2) write amplification on a 20 GiB cap on consumer NVMe is a non-issue — Apple SSDs have multi-PB endurance and the budget rotates data naturally; (3) the gate added `hitCount` tracking on `RadixTreeNode` and an orphan-extraction code path that Phase 4.1.b can add back if traces ever show amplification biting.

1. **Capture + extraction must happen inside `container.perform`, AND the three store call sites are NOT symmetric.** The plan's previous "inside the same container.perform scope, before the hop to MainActor" framing was overly simplistic — verified against the actual code on 2026-04-14, only the stripped-leaf path is shaped that way. Each of the three store call sites needs its own sub-plan. The shared infrastructure is a new helper:

   ```swift
   // LLMActor-isolated method. Must be called from a context that has a
   // live [KVCache] or [HybridCacheSnapshot] whose MLXArrays are eval'd and
   // Metal-affine — in practice, from inside `container.perform` or a
   // method nested within one.
   private func extractSnapshotPayloads(
       _ snapshots: [HybridCacheSnapshot]
   ) -> [SnapshotPayload] {
       guard self.ssdConfig?.enabled == true else { return [] }
       // For each LayerState, for each MLXArray in .state:
       //   call MLXArray.asData() → (Data, dtype, shape)
       // Bundle into SnapshotPayload (Sendable value type).
       // Same helper used by all three sub-plans below.
   }
   ```

   `self.ssdConfig?.enabled` is a synchronous actor-isolated read (see decision 26). If disabled, return `[]` immediately with no memcpy cost. The `asData()` API is the same one `tesseract/Features/Agent/UserInput.swift:128` uses on the VLM hot path, so no new MLX Swift surface is needed.

   **Spike (Task 4.1.4):** run `asData()` on a real mid-prefill `HybridCacheSnapshot` with `QuantizedKVCache` to confirm that calling it _after_ `eval(cache)` does not trigger a second Metal command-queue submission. Fallback: route the whole write through `savePromptCache(url:cache:metadata:)` inside `container.perform` — Task 4.1.3 ships the vendor wrappers so the fallback is ready on day one.

1a. **Sub-plan for call site 1 — mid-prefill snapshots (`LLMActor.swift:337` `storeSnapshots`).**

- **Capture location:** line ~1023 inside `makeHTTPPrefixCacheGeneration`, which opens `container.perform` at line 875 and closes it before returning.
- **Problem:** by the time execution reaches the `MainActor.run { prefixCache.storeSnapshots(...) }` hop at line 337, `makeHTTPPrefixCacheGeneration`'s `container.perform` has already closed. Payload extraction at line 337 would be **outside** Metal-affine scope.
- **Fix:** extract the payloads **inside `makeHTTPPrefixCacheGeneration`**, immediately after `let capturedSnapshots = iterator.capturedSnapshots` at line 1023. Still inside the outer `container.perform` block at line 875. Bundle the result into the `HTTPPrefixCacheGeneration` return type as a new `let capturedPayloads: [SnapshotPayload]` field alongside `capturedSnapshots`. The caller at line 337 then reads `mlxStart.capturedPayloads` and passes it to `storeSnapshots(promptTokens: ..., capturedSnapshots: ..., snapshotPayloads: mlxStart.capturedPayloads, partitionKey: ..., requestID: ...)`.
- **Net change:** `HTTPPrefixCacheGeneration` gains one field; `makeHTTPPrefixCacheGeneration` gains one line (the `extractSnapshotPayloads` call) inside its existing `container.perform` closure; the call site at `:337` gains one parameter but stays inside the same synchronous `MainActor.run { ... -> StoreDiagnostics in ... }` closure.

1b. **Sub-plan for call site 2 — unstripped leaf (`LLMActor.swift:487` `storeLeaf`).**

- **Capture location:** line 461, using `HybridCacheSnapshot.capture(cache: finalCache, offset: ..., type: .leaf)`. `finalCache` comes from `mlxStart.finalCacheHandle.takeFinalCache()` at line 379, which runs on LLMActor's own scope **outside any `container.perform`**.
- **Problem:** the existing capture at `:461` is not Metal-affine because it's not inside `container.perform`. The existing `HybridCacheSnapshot.capture` call works today because it only deep-copies (`array[.ellipsis]`) and does not call `asData()`. Phase 4.1 needs both the capture AND the payload extraction to happen inside a Metal-affine scope.
- **Fix:** wrap the leaf capture at `:461` in a new `container.perform` block that performs BOTH the capture AND the payload extraction, returning the `(HybridCacheSnapshot, SnapshotPayload?)` tuple to the outer scope. Roughly:

  ```swift
  // Before (line 461):
  guard let leafSnapshot = HybridCacheSnapshot.capture(
      cache: finalCache, offset: storedTokens.count, type: .leaf
  ) else { ... }

  // After:
  let (leafSnapshot, leafPayload): (HybridCacheSnapshot, SnapshotPayload?) =
      try await container.perform { _ in
          guard let snap = HybridCacheSnapshot.capture(
              cache: finalCache, offset: storedTokens.count, type: .leaf
          ) else { return (nil, nil) }   // adjusted to `throws` / optional pair
          let payload = self.extractSnapshotPayloads([snap]).first
          return (snap, payload)
      }
  guard let leafSnapshot else { /* existing skip path */ }
  ```

- This is a small extra actor hop per successful leaf store (one `container.perform` round-trip for the capture+extract block), but it is the correct Metal-affinity shape. Leaf stores are one-per-generation on the success path, so the overhead is bounded.
- **Net change:** `capture(...)` at `:461` moves inside a new `container.perform` block; `extractSnapshotPayloads([...]).first` is added alongside; the `MainActor.run { ... storeLeaf(... leafPayload: ...) }` closure at `:487` gains one parameter but stays synchronous. The 3-tuple coalesced return at `:487` is preserved.

1c. **Sub-plan for call site 3 — stripped leaf (`LLMActor.swift:1383` `storeLeaf`).**

- **Capture location:** line 1352, inside `container.perform` at line 1275 (the stripped-leaf prefill block). The `MainActor.run { ... storeLeaf(...) }` hop at `:1383` runs inside the same `container.perform` closure because the closure is `async` and awaits MainActor inside it.
- **Problem:** none, structurally. The capture AND the store are both inside the stripped-leaf `container.perform` at `:1275`. Payload extraction fits naturally between them.
- **Fix:** immediately after `guard let strippedLeaf = HybridCacheSnapshot.capture(...)` at `:1352`, insert `let strippedLeafPayload = self.extractSnapshotPayloads([strippedLeaf]).first`. Pass this to the `storeLeaf` call at `:1383` as a new `leafPayload: SnapshotPayload?` parameter. The closure signature stays `() -> (StoreDiagnostics, Int, Int)`.
- **Net change:** one line of extraction inside the existing `container.perform`; one parameter on the `storeLeaf` call.

1d. **Why the sub-plans differ (summary for reviewers).** The three call sites are structurally different because they capture from three different cache sources at three different points in the request lifecycle: (1) mid-prefill snapshots come from the iterator populated inside `makeHTTPPrefixCacheGeneration`'s `container.perform`, (2) the unstripped leaf comes from `mlxStart.finalCacheHandle` on LLMActor's outer scope, and (3) the stripped leaf is captured during a dedicated stripped-prefill inside its own `container.perform`. A single "extract inside container.perform before the store hop" pattern only fits (3). The plan's earlier simpler framing missed this and would have put the mid-prefill extraction outside Metal-affine scope and the unstripped-leaf extraction in the wrong place entirely.

2. **MainActor hop — unchanged shape at all three call sites.** The existing `MainActor.run { ... }` synchronous closures at `LLMActor.swift:337`, `:487`, and `:1383` stay synchronous. Their signatures gain an extra parameter carrying `[SnapshotPayload]` (for `storeSnapshots`) or `SnapshotPayload?` (for `storeLeaf`), but the closure type stays `() -> Tuple`, not `() async -> Tuple`. No Swift concurrency refactor on the call sites.
3. **Admission on MainActor — non-suspending, lock-protected front door (not spawn-and-return).** Inside `PrefixCacheManager.storeSnapshots(promptTokens:capturedSnapshots:snapshotPayloads:partitionKey:requestID:)`:
   - Store the `HybridCacheSnapshot` on its radix node exactly as today (RAM tier).
   - For each `(snapshot, payload)` pair, call `ssdStore?.tryEnqueue(payload:descriptor:node:)`. **This method is critical — read the next bullet.**
   - Run the existing `evictToFitBudget(requestID:preferredPartitionKey:)` RAM-tier pass. No change to its shape.
   - Return `StoreDiagnostics` synchronously. The call sites at `LLMActor.swift:337`, `:487`, `:1383` receive the tuple they already expect.
4. **`tryEnqueue` is nonisolated, lock-protected, byte-bounded, and synchronous.** It is **not** a spawn-and-return wrapper — an earlier draft had it as `Task { await self._acceptEnqueue(...) }`, which would retain the full payload inside a detached task heap allocation before the actor's queue could apply back-pressure. Under a burst of captures, this would accumulate multiple GiB of pending bytes outside the queue cap (this was flagged as a P1 by review on 2026-04-14 and fixed in this revision). The correct shape:

   ```swift
   // Inside SSDSnapshotStore (a final class, not a Swift actor for the front door):
   private let frontDoorLock = NSLock()
   private var pending: [PendingWrite] = []        // FIFO
   private var pendingBytes: Int = 0
   private let maxPendingBytes: Int                // decision 22
   private let wakeup: AsyncStream<Void>.Continuation

   nonisolated func tryEnqueue(
       payload: SnapshotPayload,
       descriptor: PersistedSnapshotDescriptor,
       node: RadixTreeNode                         // MainActor-owned; see note below
   ) -> TryEnqueueResult {
       frontDoorLock.lock()
       defer { frontDoorLock.unlock() }

       // Drop oldest pending until the new one fits, under the byte cap.
       while pendingBytes + payload.totalBytes > maxPendingBytes,
             let oldest = pending.first {
           pending.removeFirst()
           pendingBytes -= oldest.payload.totalBytes
           // Fire an async MainActor callback to strip the dropped entry's
           // pending storageRef from its node. Safe because markDropped is
           // itself non-suspending on MainActor.
           Task { @MainActor in prefixCache.markStorageRefDropped(id: oldest.snapshotID) }
       }
       if payload.totalBytes > maxPendingBytes {
           return .rejectedTooLargeForBudget    // single payload exceeds cap — extremely rare
       }
       pending.append(PendingWrite(payload: payload, descriptor: descriptor))
       pendingBytes += payload.totalBytes
       wakeup.yield()                           // non-blocking signal to the writer
       return .accepted
   }
   ```

   - **Why no Swift actor for this layer.** Swift `actor` methods are implicitly `async`, and that forces the caller into `await`. The front door must be callable from a synchronous MainActor closure (the three call sites at `LLMActor.swift:337`/`:487`/`:1383`), so we drop down to a `final class` + `NSLock` for this specific hot path. The writer loop that drains `pending` runs as a detached `Task` and does not need locks to be an actor — the lock is what provides the cross-thread safety. **This is a deliberate departure from `actor` for memory-bound reasons and should not be "fixed" by a future refactor.** The non-suspending admission invariant would break.
   - **Node reference.** `tryEnqueue` takes the `RadixTreeNode` because on `.accepted` it attaches a new `SnapshotStorageRef(committed: false)` to the node — this is what establishes the lifecycle's "pending" state (decision 23). The node reference is MainActor-isolated; the front door holds it only during the call under the lock, and writes it to the node via `node.storageRef = ref` directly on MainActor (the call site is a MainActor closure). It is not captured by the writer loop and never crosses the actor boundary.
   - **On drop-oldest-pending eviction**: the dropped entry's node needs its pending ref cleared. The front door fires an async MainActor callback (`Task { @MainActor in markStorageRefDropped(id:) }`) that runs non-blocking on MainActor — the whole operation is still drop-and-return from the caller's perspective.
   - **Memory bound math.** `maxPendingBytes` defaults to `min(4 GiB, physicalMemoryBytes / 16)`. On a 64 GB Mac that's 4 GiB (enough for ~8–20 mid-prefill snapshots in flight); on a 16 GB Mac it's 1 GiB. This is the actual cap on memory held outside the tree by the writer pipeline — not a depth cap, which is inappropriate when payloads range 50–600 MiB. See decision 22.

5. **SSD admission decision (inside the writer, using the descriptor-only LRU filter).** The writer task drains the front-door queue serially. For each `PendingWrite`:
   - Compute `spaceNeeded = descriptor.bytes`. If `currentSSDBytes + spaceNeeded <= prefixCacheSSDBudgetBytes`, proceed to write.
   - Otherwise, run **type-protected LRU** over the SSD residents. The rule is:
     1. Enumerate non-`.system` residents, sorted by `lastAccessAt` ascending.
     2. Evict oldest non-`.system` residents one at a time, updating `currentSSDBytes` after each, until the new entry fits.
     3. If non-`.system` residents are exhausted and the new entry still doesn't fit, branch on the incoming's type:
        - **Incoming is `.system`**: fall through to evicting oldest `.system` residents. This is a lateral move — a fresh system prompt is replacing an older one, and protection is preserved across the set. Continue until fit.
        - **Incoming is NOT `.system` (`.leaf` / `.lastMessageBoundary` / `.branchPoint`)**: **drop the incoming write**. Do NOT evict `.system` residents. The incoming is less valuable than the `.system` entries we'd have to destroy to make room, and the correct Marconi-faithful answer is to protect the high-value resident set. Fire `PrefixCacheDiagnostics.ssdAdmit(id:, outcome: .droppedSystemProtectionWins)` and return. The in-flight `storageRef` on the node is cleared via the normal `markStorageRefDropped` callback.
   - This rule preserves the stated protection goal: `.system` entries on SSD are never destroyed to make room for lower-value entries. The earlier draft's "fallback to oldest including `.system`" phrasing was a regression — it allowed non-system incomings to evict `.system` residents in the degenerate case, which broke type protection. (P1 flagged on 2026-04-14 fixed in this revision.)
   - This is **not** full Phase 2 Marconi — it is Marconi at α=0, which reduces to LRU within the eligible set. The rationale is in decision 21, decision 27, and the "Marconi extension" design note above: the writer actor cannot inspect live radix-tree state (parent offsets, childCount) from its own isolation domain, and running full Marconi on SSD would require either a descriptor schema extension or a cross-actor hop per scoring call. **At startup before the alpha tuner runs**, the RAM tier is also at α=0 (`LLMActor.swift:666` resets `EvictionPolicy.alpha = 0.0`), so the two tiers' policies are behaviorally equivalent. **After the alpha tuner writes `bestAlpha` at `AlphaTuner.swift:193`**, the RAM tier moves to the full formula while the SSD tier stays at LRU — the policies diverge subtly, and Phase 4.1 accepts the divergence (decision 27). Phase 4.1 ships the simpler form and leaves the schema extension as a Phase 4.1.b concern, gated on both the alpha tuner raising α above 0 AND production traces showing that the divergence costs real hit rate.
   - **Incoming usually wins but NOT always.** At α=0, recency is the sole eviction-scoring input and the incoming's `lastAccessAt = .now` beats every existing resident. So in the common case (SSD has at least one non-`.system` resident) the writer evicts the oldest non-`.system` resident and admits the incoming. The **exception** is the degenerate case documented in step 5 above and in Eviction + demotion bullet 4: if the non-`.system` eligible set is exhausted and the incoming is itself non-`.system`, the incoming is **dropped**. This preserves `.system` type protection, which is why the rule is asymmetric. A `.system` incoming in the same degenerate state still wins (lateral eviction of the oldest `.system` resident). This is the only case where "incoming loses" — not because of scoring, but because type protection for `.system` is a hard rule on SSD.
6. **Writer: commit or drop.** Once SSD space is reserved:
   - Write to `{snapshotID}.safetensors.tmp` via `FileHandle.write` (safetensors is flat; no internal pagination), `fsync`, rename atomically, update the in-memory manifest, schedule a debounced manifest persist (500 ms idle → write `manifest.json.tmp` → rename).
   - On success: fire `Task { @MainActor in prefixCache.markStorageRefCommitted(id: snapshotID) }`. This flips `committed: false → true` on the node's ref. Subsequent lookups can now hydrate.
   - On failure (`ENOSPC`, `EDQUOT`, safetensors write error, other I/O): log, evict oldest eligible resident, retry once. On retry failure, drop the write AND fire `markStorageRefDropped(id:)` to clear the pending ref from the node. The node reverts to RAM-only (state 1) or, if its body was already evicted from RAM, gets hard-deleted (because it has neither a body nor a committed ref).
   - The writer also owns the release of bytes from `pendingBytes` — once the item is fully processed (committed or dropped), the front door's `pendingBytes` counter is decremented. The decrement is done via another lock-acquire inside the writer (lock is held only briefly).
7. **The inference path never waits on any of this.** LLMActor captures → extracts → hands off via synchronous `tryEnqueue` → resumes decoding. MainActor stores in RAM → fires `tryEnqueue` (returns in microseconds under the lock) → returns synchronously. The writer task runs at its own pace. The only MainActor-visible latency is the payload _extraction_ inside `container.perform` — which is a `memcpy` of already-`eval`'d tensors on Apple Silicon unified memory, bounded by the snapshot size (50–600 MiB at ~20+ GB/s memory bandwidth = 2–30 ms). Instrumented via `PrefixCacheDiagnostics.captureDuration`; regression if p95 > 100 ms.

**Storage ref lifecycle (explicit five-state machine).**

Every radix node falls into exactly one of five storage states at any moment. This replaces any implicit assumption that "storageRef != nil means SSD-available":

| State                   | Body    | StorageRef | Committed | Semantics                                                                                |
| ----------------------- | ------- | ---------- | --------- | ---------------------------------------------------------------------------------------- |
| 1. RAM-only             | present | none       | —         | No SSD. Either SSD disabled, extraction skipped, or write was dropped.                   |
| 2. Pending(body)        | present | present    | false     | RAM body live; SSD write queued or in-flight. Lookups hit RAM; SSD not yet available.    |
| 3. Pending(body-drop)   | absent  | present    | false     | RAM body evicted while write was still in flight. Lookups treat as miss until commit.    |
| 4. Committed(body)      | present | present    | true      | RAM body live AND SSD copy fsync'd. Lookups hit RAM; SSD copy is insurance for eviction. |
| 5. Committed(body-drop) | absent  | present    | true      | RAM body evicted; SSD copy lives. Lookups hit SSD (hydrate on LLMActor).                 |

**Transitions:**

- `1 → 2` on accepted `tryEnqueue`.
- `2 → 4` on writer commit callback (`markStorageRefCommitted`).
- `2 → 3` on RAM eviction before commit lands. Body-drop the RAM copy; pending ref stays.
- `2 → 1` on writer drop callback (`markStorageRefDropped`) before RAM eviction. Ref cleared; body stays.
- `3 → 5` on writer commit callback after RAM was already evicted.
- `3 → removed-from-tree` on writer drop callback while body is absent. Node has nothing; hard-delete.
- `4 → 5` on RAM eviction; SSD copy is the insurance.
- `5 → 4` on SSD hit + hydration; LLMActor loads the body back into RAM via `loadSync`.

**Lookup rules:**

- RAM body present (states 1, 2, 4) → return from RAM.
- RAM body absent AND committed ref present (state 5) → SSD hit; LLMActor hydrates.
- RAM body absent AND pending ref present (state 3) → miss. The file does not yet exist; returning a hit would race the writer. Subsequent lookups after the writer commits will succeed.
- RAM body absent AND no ref → miss (no node reached; standard radix tree miss).

**RAM eviction rules (unchanged from the previous section but spelled out here for completeness):**

- Eligible victim candidates: states 1, 2, 4 (body present).
- Victim disposition:
  - State 1 → hard delete.
  - State 2 → body-drop; transition to state 3.
  - State 4 → body-drop; transition to state 5.
  - `.system` type protection applies in all three cases — filtered out at candidate enumeration, per `eligibleEvictionNodes()`.

**MainActor callback surface (on `TieredSnapshotStore` or `PrefixCacheManager`):**

- `markStorageRefCommitted(id: UUID)` — looks up `pendingRefsByID[id]` (a MainActor-isolated `[UUID: RadixTreeNode]` map), flips `committed = true`, removes from the pending map.
- `markStorageRefDropped(id: UUID)` — looks up the node, clears `storageRef`, removes from the pending map. If the node is in state 3 (no body, no ref), hard-delete from the tree.
- Both are non-async, non-throwing. Callers hop to MainActor via `Task { @MainActor in ... }`.
- `pendingRefsByID` holds nodes strongly so the commit/drop callback can always find them — even if RAM eviction would otherwise have dropped them. The map entry is removed on commit or drop, at which point the node can be collected by the tree's normal eviction flow if it was a state-3 orphan.

**Read / hydration path.**

1. A request lookup enters `PrefixCacheManager.lookup(tokens:, partitionKey:)`. The match traverses the radix tree as today.
2. **RAM hit — state 1 / 2 / 4 (`snapshot != nil`).** Return the resident snapshot as today. `TokenRadixTree.findBestSnapshot` already bumps `node.lastAccessTime = .now` for the RAM-tier Marconi recency term. **New step in `PrefixCacheManager.lookup`**: if the matched node is in state 4 (`snapshot != nil && storageRef?.committed == true`), also call `ssdStore?.recordHit(id: node.storageRef!.snapshotID)` synchronously under the SSD store's NSLock. This bumps the descriptor's `lastAccessAt` in the manifest so that a hot RAM entry does not appear stale on the SSD eviction path when its body is eventually dropped to state 5. Calling this from the lookup hot path is cheap — the store's lock is contended only by `tryEnqueue` and the writer loop, and the update is an O(1) dictionary touch.
3. **SSD hit — state 5 only (`snapshot == nil && storageRef != nil && storageRef.committed == true`).** Return a new `LookupResult.reason = .ssdHit(storageRef: SnapshotStorageRef, ssdStore: SSDSnapshotStore, node: RadixTreeNode)` carrying (a) the storage ref for the `snapshotID` / `partitionDigest` inputs needed by `loadSync`, (b) an unowned reference to the `SSDSnapshotStore` so LLMActor can call `loadSync` and `recordHit` without reaching back into `PrefixCacheManager`, (c) the matched `RadixTreeNode` so LLMActor can pass it to `promote` in the MainActor hop below. The `ssdStore` reference is MainActor-owned (it's held by `PrefixCacheManager` which is `@MainActor`), so embedding it inside `LookupResult.reason` and handing it back to the LLMActor-isolated caller is safe because `SSDSnapshotStore` is a `final class` whose public methods (`loadSync`, `recordHit`) are `nonisolated`. Note the explicit `committed == true` guard — see step 3a.
   3a. **State-3 (pending + body-dropped) is a miss, not a hit.** If the matched node has `snapshot == nil && storageRef != nil && storageRef.committed == false`, the file does not yet exist on disk (the write is still in the writer's queue or in flight). The lookup must treat this as a miss — returning an SSD hit and hydrating from the in-flight descriptor would race the writer and surface a half-written or absent file to the caller. This was flagged as P2 on 2026-04-14: the lifecycle table said pending refs are miss but the read-path summary used `storageRef != nil && snapshot == nil` without the `committed` guard. This revision makes the guard explicit at every lookup bullet.
4. **LLMActor materializes the SSD hit — concrete API.** The `LookupResult` returned by `PrefixCacheManager.lookup` carries either the RAM body or enough state for LLMActor to hydrate from disk. For SSD hits, `LookupResult` now also carries an `sssdStoreRef: SSDSnapshotStore` reference (weak or unowned — the store is held by `PrefixCacheManager` for the life of the manager). LLMActor's existing lookup code path (at `LLMActor.swift:950`, `lookupAndPlanCheckpoints` hop) detects an `.ssdHit` reason and branches:

   ```swift
   // Inside LLMActor, after the existing lookupAndPlanCheckpoints hop returns.
   // lookupResult is awaited from MainActor, now we're back on LLMActor.
   if case .ssdHit(let storageRef, let ssdStore) = lookupResult.reason {
       // Synchronous load inside container.perform — Metal-affine.
       let materialized: HybridCacheSnapshot? = try await container.perform { _ in
           ssdStore.loadSync(storageRef: storageRef,
                             expectedFingerprint: self.modelFingerprint!)
       }
       guard let snapshot = materialized else {
           // File missing, corrupt, or fingerprint mismatch. loadSync already
           // fired markStorageRefDropped on MainActor internally; treat as miss.
           return noHitPath()
       }
       // Both callbacks are MainActor-isolated, so they go through MainActor.run.
       await MainActor.run {
           ssdStore.recordHit(id: storageRef.snapshotID)    // non-suspending under lock
           prefixCache.promote(node: lookupResult.node,     // state 5 → state 4
                               snapshot: snapshot)
       }
       // Continue into the existing "warm" branch with `snapshot` as the resumed state.
   }
   ```

   **`SSDSnapshotStore.loadSync` is a `nonisolated` instance method, NOT a static helper.** The earlier draft said "static helper" but that is wrong — the method needs access to the store's immutable `rootURL` (captured at init from `SSDPrefixCacheConfig.rootURL`) to resolve the file URL. As a `nonisolated func` on a `final class`, the method is safe to call from any actor context (including LLMActor inside `container.perform`) without `await`, because it only touches immutable let-bound state on `self`. Signature:

   ```swift
   extension SSDSnapshotStore {
       // Metal-affine: callers must invoke inside container.perform.
       nonisolated func loadSync(
           storageRef: SnapshotStorageRef,
           expectedFingerprint: String
       ) -> HybridCacheSnapshot? {
           let url = fileURL(for: storageRef)             // uses self.rootURL
           guard FileManager.default.fileExists(atPath: url.path) else {
               Task { @MainActor in prefixCache.markStorageRefDropped(
                   id: storageRef.snapshotID) }
               return nil
           }
           do {
               let (caches, metadata) = try loadPromptCache(url: url)
               guard metadata["model_fingerprint"] == expectedFingerprint else {
                   try? FileManager.default.removeItem(at: url)
                   Task { @MainActor in prefixCache.markStorageRefDropped(
                       id: storageRef.snapshotID) }
                   return nil
               }
               return HybridCacheSnapshot.reconstruct(from: caches, metadata: metadata)
           } catch {
               try? FileManager.default.removeItem(at: url)
               Task { @MainActor in prefixCache.markStorageRefDropped(
                   id: storageRef.snapshotID) }
               return nil
           }
       }

       // File URL derivation — uses immutable rootURL captured at init.
       private nonisolated func fileURL(for ref: SnapshotStorageRef) -> URL {
           let shardByte = String(ref.snapshotID.prefix(1))   // 0–f bucket
           return self.rootURL
               .appendingPathComponent("partitions")
               .appendingPathComponent(ref.partitionDigest)
               .appendingPathComponent("snapshots")
               .appendingPathComponent(shardByte)
               .appendingPathComponent("\(ref.snapshotID).safetensors")
       }
   }
   ```

   **Why this resolves the "fileRelativePath missing from `SnapshotStorageRef`" finding:** the `SnapshotStorageRef` does **not** carry `fileRelativePath` directly — that field lives on `PersistedSnapshotDescriptor` (which is persisted to the manifest). The `loadSync` method derives the file URL from three pieces: (a) the immutable `self.rootURL` captured at store init from the config, (b) `ref.partitionDigest`, (c) `ref.snapshotID`. Together they reconstruct the canonical on-disk path using the same sharding rule used at write time. No field addition needed on `SnapshotStorageRef`.

5. **The `promote` call is an explicit `await MainActor.run { ... }` hop.** `PrefixCacheManager` is `@MainActor`, so calling `prefixCache.promote(node:, snapshot:)` from LLMActor (its own actor) requires `await MainActor.run { ... }`. The earlier draft's "synchronously on LLMActor" phrasing was wrong — it's synchronous _within_ the MainActor hop, but the hop itself is async-awaited from LLMActor's perspective. The code snippet above makes this explicit: `await MainActor.run { ssdStore.recordHit(...); prefixCache.promote(...) }` coalesces both MainActor operations into a single hop, minimizing the actor-switching overhead on the hot path.

6. **`ssdStore.recordHit(id:)` is MainActor-safe but nonisolated.** It's a nonisolated instance method on the `SSDSnapshotStore` final class — it acquires the front-door `NSLock`, updates `manifest.snapshots[id]?.lastAccessAt = .now`, schedules a debounced persist, and releases. Because it's nonisolated, it can be called from either LLMActor directly or from inside `MainActor.run`. Both are safe; the coalesced hop above puts it inside the MainActor hop for logical grouping with `promote`, but the call itself does not require MainActor isolation.

7. **Rationale for synchronous load.** The oMLX regression (`test_load_no_executor_deadlock`) proves that moving `mx.load` to a background worker thread deadlocks Metal command-queue submission. A 200 MiB NVMe read at 5 GB/s is ~40 ms, which amortizes against the 800+ ms prefill time saved — the trade-off is firmly positive. This is the single most important load-bearing constraint in the design.
8. **`loadSync` failure cleanup handles all three error modes** (file missing, fingerprint mismatch, safetensors parse error) by firing `Task { @MainActor in prefixCache.markStorageRefDropped(id:) }` before returning `nil`. This transitions the node out of state 5 back to "no ref" (state 1 if a body is somehow resident, hard-delete otherwise), so a subsequent lookup on the same path does not re-attempt hydration on the same broken file. `markStorageRefDropped` is MainActor-isolated; the `Task { @MainActor in ... }` is fire-and-forget from loadSync's nonisolated context.

**Warm start.**

1.  `PrefixCacheManager.init()` is unchanged (still cheap) except for the new `ssdStore:` parameter. A new `warmStart(ssdRoot:) async throws` method performs the restoration.
2.  `LLMActor.ensurePrefixCache()` (currently at `LLMActor.swift:664`) awaits `warmStart` the first time it lazily instantiates the manager for the current `partitionKey`.
3.  `warmStart` reads `manifest.json`:
    - Missing or schema mismatch → initialize empty, `currentSSDBytes = 0`, no error.
    - Corrupt JSON → rename `manifest.json` → `manifest.corrupt.{timestamp}.json`, rebuild from directory walk (parse safetensors headers), write fresh manifest, re-seed `currentSSDBytes` from the rebuilt manifest (see step 4a).
4.  For each partition in the manifest:
    - Validate `modelFingerprint` against the partition currently loaded in `LLMActor`. Mismatch → mark the entire partition as invalid, schedule async cleanup of its directory, skip.
    - For each `PersistedSnapshotDescriptor` in the partition: - Call `PrefixCacheManager.restoreStorageRef(path: desc.pathFromRoot, storageRef: SnapshotStorageRef(...), partitionKey: ..., lastAccessTime: ...)`. This is a new method parallel to the existing `restoreSnapshot(path:snapshot:...)` (at `PrefixCacheManager.swift:383`) that attaches a `storageRef` instead of a body. The existing `TokenRadixTree.insertPath` + storage hook already does the structural work.
      4a. **Seed `SSDSnapshotStore.currentSSDBytes` from the restored manifest.** This is a load-bearing step that the earlier draft omitted. The writer's admission loop uses `currentSSDBytes` to decide when to run the type-protected LRU cut (bullet 4 of the Eviction + demotion section). Without seeding, the post-restart store thinks the SSD is empty and would over-admit on the first few writes, blowing past `prefixCacheSSDBudgetBytes` until a sync-up pass happened to recompute the total. **Explicit seeding rule**: after step 4 completes, compute


        ```swift
        currentSSDBytes = manifest.snapshots.values
            .filter { partition(for: $0) != .invalidated }
            .reduce(0) { $0 + $1.bytes }
        ```
        and write the result into the store under the front-door NSLock before the store is ready to accept admissions. The filter excludes entries in partitions whose fingerprint mismatched in step 4 — those entries are scheduled for cleanup and their bytes should not count against the budget during the post-restart window. The seeding is synchronous with the warm-start `async throws` method (it's a pure Swift computation over the Codable manifest; no file I/O).
        **Regression test**: a `WarmStartTests` case that writes a manifest with three descriptors of known sizes, calls `warmStart`, and asserts `store.currentSSDBytes == totalOfThree`. A follow-up test forces a partition-fingerprint mismatch on one of the three and asserts the count excludes that partition's bytes.
5.  No bodies are loaded at warm start. First lookup on each node pays the one-time ~40 ms NVMe read.
6.  Pre-hydration of top-K `.system` snapshots is **not** part of Phase 4.1 — that is Idea 4.3, a separate follow-up measured against a real trace.

**Eviction + demotion (Phase 2 Marconi on RAM, type-protected LRU on SSD, no cascade complexity).**

Because SSD writes happen at capture time (not at eviction time), RAM eviction is pure body-drop and does not trigger any new SSD work. This is the biggest simplification compared to the earlier draft and is what resolves the "new RAM demotions can be dropped while lower-utility SSD entries survive" inconsistency — there is no "new RAM demotion" path for SSD admission, because SSD already has the copy (or has explicitly rejected it at write time).

1. **RAM eviction (`PrefixCacheManager.evictToFitBudget`) — almost unchanged from Phase 2.** `findEvictionCandidate` runs `EvictionPolicy.computeScores` over the RAM-tier eligible set (`TokenRadixTree.eligibleEvictionNodes()`, already filters `.system` + `childCount <= 1`). Picks the lowest-utility victim. The Phase 2 alpha tuner continues to govern α on the RAM tier. This is the full Marconi formula with live radix-tree inputs. **The only change to the eviction loop at `PrefixCacheManager.swift:499-516` is the post-eviction cleanup branch (see bullet 2).**
2. **Victim disposition — with an explicit guard against cleaning up SSD-backed nodes.** The current loop at `PrefixCacheManager.swift:499-516` reads:
   ```swift
   candidate.tree.evictSnapshot(node: candidate.node)     // clears node.snapshot
   events.append(...)
   if candidate.node.isLeaf {
       candidate.tree.evictNode(node: candidate.node)     // removes node from tree
   } else if candidate.node.childCount == 1, candidate.node.snapshot == nil {
       candidate.tree.collapseSingleChildNode(candidate.node)  // merges degenerate edge
   }
   ```
   **This cleanup path would orphan any SSD reference** in the common `.leaf` / `.lastMessageBoundary` case, because those entries are almost always leaf nodes in the radix tree — `evictNode` at `TokenRadixTree.swift:220` removes the node (and its `storageRef`) from the tree entirely, and `collapseSingleChildNode` at `TokenRadixTree.swift:333` would also drop the pointer when the now snapshot-less node collapses into its child's edge. The SSD copy would remain on disk but become unreachable from any radix path, silently leaking disk bytes until warm-start cleanup.
   **Fix — two explicit guards on the SSD-backed case:**
   1. **In the eviction loop**: after `evictSnapshot`, check `candidate.node.storageRef != nil` AND `candidate.node.storageRef?.committed == true`. If so, skip BOTH the `evictNode` and the `collapseSingleChildNode` branches — the node is now in state 5 (SSD-only, committed), and it must remain in the tree as a lookup target for the persisted copy. (Pending refs — state 3 — are also skipped; the writer's commit/drop callback will clean up if needed.)
   2. **In `TokenRadixTree.collapseSingleChildNode(_:)`** at `TokenRadixTree.swift:333`: update the guard from `node.snapshot == nil && node.childCount == 1` to `node.snapshot == nil && node.storageRef == nil && node.childCount == 1`. A node with a storageRef is structurally meaningful (it pins a radix path to a persisted file) and must not be collapsed even if it has no in-memory snapshot. This is a defense-in-depth fix — the eviction-loop guard in (1) already suppresses the call from the normal eviction path, but external callers of `collapseSingleChildNode` (tests, future refactors) could still reach it.
   3. **In `TokenRadixTree.evictNode(_:)`** at `TokenRadixTree.swift:220`: the function is currently called only from the eviction loop, but add a debug assertion (`assert(node.storageRef == nil, "evictNode must not be called on SSD-backed leaves")`) at function entry so a future caller gets a clear diagnostic if they break the invariant. The actual suppression is at the call site in `evictToFitBudget`, not inside `evictNode` itself — `evictNode` correctly removes a tree leaf when asked, and the caller is responsible for not asking when an SSD ref is present.
      **State-to-disposition table** (the cleanup branch above runs inside the loop for each victim):
      | Pre-eviction state | Post `evictSnapshot(node)` | `node.storageRef` | Cleanup branch taken | Post-cleanup state |
      |---|---|---|---|---|
      | State 1 (RAM-only) | `snapshot = nil`, no ref | nil | `isLeaf → evictNode` OR `childCount == 1 → collapseSingleChildNode` | node removed OR collapsed |
      | State 2 (pending, body) | `snapshot = nil`, pending ref | present, not committed | **skip both branches** | state 3 (pending, no body) |
      | State 4 (committed, body) | `snapshot = nil`, committed ref | present, committed | **skip both branches** | state 5 (SSD-only) |
   - `.system` type protection applies at candidate enumeration — the filter is in `eligibleEvictionNodes()` and doesn't change, so a `.system` node is never selected as a victim except in the degenerate RAM-mandatory-fallback case.
3. **Why this works without a cascade.** The SSD admission decision was already made at capture time (by the writer, using type-protected LRU — see bullet 4 below). By the time we get to RAM eviction, there is no pending SSD decision — either the SSD write landed, is on its way, or was rejected at write time. RAM eviction is a pure RAM-tier decision; the SSD-tier state is whatever the writer already settled it into. The P1 inconsistency flagged on 2026-04-14 ("new demotions dropped while lower-utility SSD entries survive") is prevented by construction: there is no "new demotion" at RAM eviction time — only a body-drop on an entry whose SSD fate is already decided.
4. **SSD-tier eviction — type-protected LRU with asymmetric protection, inside the writer.** When the writer processes a pending item and `currentSSDBytes + descriptor.bytes > prefixCacheSSDBudgetBytes`, it runs **type-protected LRU** over the SSD-resident descriptor set. Rule (see the Write path section for the full loop):
   - Evict oldest non-`.system` residents first, one at a time, until the new entry fits.
   - If non-`.system` residents are exhausted and the entry still doesn't fit:
     - If incoming is `.system` → evict oldest `.system` residents (lateral move; protection preserved across the set).
     - If incoming is NOT `.system` → **drop the incoming write**. Do NOT evict any `.system` resident. This is what makes type protection real: a fresh leaf / boundary / branch-point never destroys a system resident, regardless of how little non-system budget is available. The earlier draft's "fallback to oldest including `.system`" phrasing would have allowed non-system incomings to evict protected `.system` residents — that bug was flagged on 2026-04-14 (P1) and is fixed here.
   - **Incoming is NOT guaranteed to win admission** under this rule — the edge case where incoming is non-system AND all non-system residents are already gone drops the incoming. In practice this case is rare because SSD budget (20 GiB default) far exceeds typical `.system` usage (~200 MiB per entry, low single-digit count), but the rule must handle it correctly for protection to be meaningful.
5. **Why LRU on SSD instead of full Marconi.** The previous draft claimed "same formula on both tiers, incoming is a scored candidate." That claim is **stronger than the descriptor schema can support** (the P1 flagged on 2026-04-14). Full Marconi scoring needs `parentTokenOffset` (for the parent-relative FLOPs term) and `childCount` (for topological eligibility) — both of which live on live `RadixTreeNode` instances and cannot be inspected from the writer's isolation domain without a cross-actor hop per scoring call. Two legitimate responses: (a) enrich `PersistedSnapshotDescriptor` with those inputs (schema extension + staleness concerns on `childCount`); (b) run the cut on MainActor with live tree access before handing the pending item to the writer. Both add complexity to solve a problem that **is small at startup** (both tiers run at α=0 until the tuner's grid search finishes) and **becomes a real but bounded divergence** once `AlphaTuner.swift:193` sets `EvictionPolicy.alpha = bestAlpha` and the RAM tier starts factoring FLOP efficiency into its choices while the SSD tier stays at LRU. Phase 4.1 ships the simpler form and accepts the divergence as a deliberate trade (decision 27). Upgrade path: if the alpha tuner raises α above 0 AND production traces justify it, Phase 4.1.b extends the descriptor schema + promotes the writer's scoring to full Marconi. This is documented as decision 21 (policy choice) and decision 27 (explicit acceptance of post-tuning divergence).
6. **Type protection holds on both tiers, for the same reason.** The RAM tier filters `.system` via `TokenRadixTree.eligibleEvictionNodes()`. The SSD tier filters `.system` by checking `descriptor.checkpointType != "system"` before picking the oldest victim. A `.system` entry on either tier is never selected while any non-`.system` candidate remains. In the degenerate case where the non-`.system` eligible set is empty: on RAM, the existing fallback at `PrefixCacheManager.swift:643-655` picks the oldest snapshot anywhere (including `.system`) because RAM eviction is mandatory once the budget is exceeded; on SSD, the writer's rule is different — if the incoming is itself `.system` it may evict older `.system` residents (lateral move), but if the incoming is non-`.system` it is **dropped** rather than eviction falling through to a protected entry (see Eviction + demotion bullet 4 and the P1 fix on 2026-04-14). The asymmetry exists because on RAM we have no choice (we must fit the new RAM snapshot or the budget is violated), while on SSD we do have a choice (reject the incoming write) and the Marconi-faithful answer is to preserve the high-value `.system` residents.
7. **No cost-aware utility extension in Phase 4.1.** A tier-aware formula that weights recency or FLOP efficiency by `1/readLatency(tier)` would let us use a different α per tier, or adjust the utility for tier-hit latency. We do not do this in Phase 4.1 because (a) we have no measured SSD-hit vs RAM-hit latencies yet, (b) on pure-attention Qwen3.5 at α=0 the FLOP efficiency term is inert anyway, and (c) a cost-aware term depends on the same descriptor extension as full Marconi on SSD, so both belong in the same Phase 4.1.b package. This is the single most obvious place where Phase 4.1.b can extend Phase 4.1 once production traces are available.
8. **Cap semantics — explicit resolution of the P2 concern flagged on 2026-04-14.** The only cap is `prefixCacheSSDBudgetBytes` (default 20 GiB). There are **no per-type caps**. An earlier draft of this plan proposed "per-partition cap of N=4 most-recent `.lastMessageBoundary` snapshots on SSD" to bound long-conversation accumulation. That cap is **rejected** in this version for three reasons: (a) it violates Marconi's "one formula, one knob" convention (the research summary cited earlier confirms the paper has no quota system — everything is governed by `S(n)` plus topological protection); (b) LRU already penalizes stale `.lastMessageBoundary` entries, so under real workloads they are the first non-`.system` victims when SSD pressure bites; (c) the top-level 20 GiB budget is a sufficient single cap. If production traces ever show that `.lastMessageBoundary` growth overwhelms the recency term (e.g., a user runs 50+ turn conversations with steadily shifting context), Phase 4.1.b can add a per-type admission filter — which Marconi's authors would call an "admission gate" per §4.1 and matches the paper's admission-first philosophy. This is a deliberate trade: we prefer the paper's philosophy over defensive scaffolding until a trace shows the scaffolding is earning its keep.

**Invalidation.**

- **Model weight swap** → `modelFingerprint` mismatch at partition load time. Entire partition directory scheduled for cleanup. Cheap because the fingerprint is computed once at model load.
- **Schema version mismatch on the manifest** → rename `manifest.json` → `manifest.v{old}.bak`, start fresh, let the next directory walk rebuild if body files exist.
- **Schema version mismatch on a safetensors header** → delete the file, remove from manifest.
- **File missing (deleted externally)** → reader returns miss, manifest cleaned up opportunistically.
- **Corrupt file (safetensors parse error)** → same as missing, plus log at `error` level.
- **Partial write survived crash** → the `.tmp` suffix makes recovery trivial: on warm start, any `*.tmp` files are deleted.
- **Leaf-store-guard (Task 2.2)** — unchanged. Snapshots with `trimAmount > 0` are never stored on any tier, so they cannot appear on disk.

**Partitioning + model fingerprint.**

- `CachePartitionKey` grows a new optional field `modelFingerprint: String?`. Backward-compat: all existing call sites can pass `nil`, and the disk tier simply won't persist for those (RAM tier behavior is unchanged).
- `LLMActor` computes the fingerprint once per model load via a new helper `Model.computeFingerprint(at: modelDir) throws -> String`:
  - SHA-256 over: `config.json` bytes + `tokenizer.json` bytes + sorted list of `(filename, size, mtime)` for every `*.safetensors` in the model directory.
  - Deliberately does **not** hash the weight bytes themselves (too expensive at load time — ~5–10 seconds for a 4B model). The `(size, mtime)` tuple is cryptographically weak but matches APFS behavior: replacing a weight file always changes mtime, and any real retraining pipeline produces different sizes or different mtimes.
  - Stored on `LLMActor` alongside the `ModelContainer`; passed into every `CachePartitionKey` constructed on that actor.
- The fingerprint is written into the safetensors header for every snapshot AND into the partition meta file AND into the partition digest (`partitionDigest = fnv8(modelID || kvBits || kvGroupSize || sessionAffinity || modelFingerprint)`). A mismatch at any level invalidates without any chance of cross-contamination.

**Concurrency (load-bearing — read carefully).**

- **`SSDSnapshotStore` is a `final class`, not a Swift `actor`**, for the front-door admission layer. This is a deliberate choice — Swift `actor` methods are implicitly async, and the non-suspending admission invariant requires the MainActor caller to complete the call synchronously under a lock. The writer loop that drains the front-door queue runs as a detached `Task` and does its own serial I/O work. Cross-thread safety comes from an `NSLock` guarding the queue + byte counter, not from actor isolation. Any future refactor that converts this into a Swift actor will re-introduce the spawn-and-return memory-safety bug flagged on 2026-04-14 and must be rejected.
- **Admission is non-suspending from MainActor.** `SSDSnapshotStore.tryEnqueue(payload:descriptor:node:)` is a `nonisolated func` that acquires the front-door lock, enforces the byte budget (drop-oldest-pending on overflow), pushes the item to the internal queue, releases the lock, and yields a wakeup signal to the writer via an `AsyncStream.Continuation.yield()` (non-blocking). Returns a synchronous `TryEnqueueResult`. No `await` anywhere on the caller side.
- **`PrefixCacheManager` stays `@MainActor`.** Its public API (`storeSnapshots`, `storeLeaf`, `lookup`, `evictToFitBudget`) remains non-async. Internally, the SSD interactions go through the synchronous `tryEnqueue` above — no `await` inside these methods.
- **Writer loop runs in a detached `Task`.** After `SSDSnapshotStore.init`, it spawns a single `Task.detached { await self.writerLoop() }`. The loop does `for await _ in wakeupStream { drainAllPending() }`, and for each drained item runs the admission-time type-protected LRU cut, writes the file, commits, and fires the MainActor callback to mark the node's ref committed. The writer never holds the front-door lock during I/O — it acquires the lock only to pop items (fast path) and to release byte-count on completion.
- **MainActor commit/drop callbacks** from the writer loop use `Task { @MainActor in prefixCache.markStorageRefCommitted(id: ...) }` / `markStorageRefDropped(id: ...)`. These hops are non-blocking on the writer side (writer proceeds to the next item immediately after firing the task) and execute synchronously on MainActor when they land. The MainActor callbacks look up the node in `pendingRefsByID: [UUID: RadixTreeNode]`, flip `committed` or clear the ref, and remove the map entry. If the lookup misses (node already evicted without the write ever landing), the callback logs at `debug` and returns — the file cleanup is the writer's responsibility, not the callback's.
- **The synchronous `loadSync` entry point for hydration** is called on `LLMActor` inside `container.perform`. It bypasses actor isolation because the read path must be thread-pinned to the inference thread (oMLX Metal-deadlock rule). Implemented as a `nonisolated func` (instance method, **not** static) on `SSDSnapshotStore`: the method needs `self.rootURL` (immutable let-bound state captured from `SSDPrefixCacheConfig` at init) to resolve the file URL from `storageRef.partitionDigest` + `storageRef.snapshotID`. Because `rootURL` is immutable after init, reading it from a `nonisolated` method is safe without actor isolation. On failure paths (file missing, fingerprint mismatch, safetensors error), `loadSync` fires a fire-and-forget `Task { @MainActor in prefixCache.markStorageRefDropped(id:) }` to clean up the node's pending ref before returning `nil`.
- **Warm start is the only async entry point the manager exposes.** `PrefixCacheManager.warmStart(ssdRoot:) async throws` is awaited once, at model load time, inside `LLMActor.ensurePrefixCache`. Everything else stays sync.
- **Multi-process.** Benchmark runs (`scripts/dev.sh prefix-cache-e2e`) run in a subprocess. To avoid contention, the benchmark CLI overrides `SSDRoot` to a per-PID subdirectory: `{SSDRoot}/_benchmark/{pid}/`. The main app uses the shared root. No fcntl locking.
- **The three LLMActor store call sites — exact refactor surface.** Here are the exact changes required at each call site:
  - **`LLMActor.swift:337` (mid-prefill snapshots).** The closure body `prefixCache.storeSnapshots(promptTokens:capturedSnapshots:partitionKey:requestID:)` gains a new `snapshotPayloads: [SnapshotPayload]` parameter. The surrounding `MainActor.run { ... -> StoreDiagnostics in ... }` closure stays synchronous. The payloads are extracted by the new `LLMActor.extractSnapshotPayloads(_:)` helper _before_ the MainActor hop, inside the same `container.perform` scope that captured the snapshots.
  - **`LLMActor.swift:487` (unstripped leaf store, coalesced with stats read).** The closure body gains `leafPayload: SnapshotPayload?` alongside `leafSnapshot`. The closure signature stays `() -> (StoreDiagnostics, Int, Int)` — the 3-tuple coalesced pattern is preserved. The payload is extracted inside the `container.perform` scope that earlier produced the leaf snapshot.
  - **`LLMActor.swift:1383` (stripped leaf store, same coalesced pattern).** Identical change to `:487`. Same 3-tuple, same synchronous closure, new `leafPayload:` parameter.
- **Why not use Swift's `sending` / `@_unsafeInheritExecutor` or similar.** We considered letting `storeSnapshots` become `async` and converting the closures to `MainActor.run { () async -> ... }` form, but Swift does not have a clean `MainActor.run` variant that accepts async closures and returns a typed result — the workaround is to spawn a `Task` and `await` it, which is a real refactor at every call site and introduces task-cancellation considerations that are not currently handled at those sites. Non-suspending `tryEnqueue` under a lock is strictly simpler and has bounded memory regardless of burst patterns.

**Settings + config flow (P1 resolution 2026-04-14).**

Three new UserDefaults-backed keys are added to `SettingsManager.swift` following the existing `Int` + `Bool` pattern (Key enum → var with didSet → register(defaults:) → reset):

- `prefixCacheSSDEnabled: Bool` (default `true`).
- `prefixCacheSSDBudgetBytes: Int64` (default `20 * 1024 * 1024 * 1024` = 20 GiB). Stored as `Int` in UserDefaults; validated at load.
- `prefixCacheSSDDirectoryOverride: String?` (default `nil`; normal path is the sandbox `.cachesDirectory`). Mostly for tests and benchmarks.

**The config flow is load-bearing.** `SettingsManager` is `@MainActor @Observable` (`tesseract/Features/Settings/SettingsManager.swift:11`), so neither `LLMActor` (its own Swift actor) nor `SSDSnapshotStore` (nonisolated final class) can consult it from their isolation domains without a `MainActor.run` hop — and the hot path inside `container.perform` cannot afford to suspend. The fix: snapshot the settings into an **immutable `Sendable` struct on MainActor at model load time**, and pass the snapshot through AgentEngine into LLMActor as a constructor/load-time argument. LLMActor stores it in an actor-isolated property and reads synchronously. `SSDSnapshotStore` is then constructed with the snapshot inside `LLMActor.ensurePrefixCache`.

```
SettingsManager (@MainActor, observable)
  │
  │  settingsManager.makeSSDPrefixCacheConfig() -> SSDPrefixCacheConfig?
  │      // synchronous call; AgentEngine is @MainActor so this is direct.
  ▼
AgentEngine.loadModel(from:visionMode:)         [@MainActor]
  │
  │  llmActor.loadModel(from: dir, visionMode: vm, ssdConfig: snapshot)
  │      // crosses actor boundary; SSDPrefixCacheConfig is Sendable.
  ▼
LLMActor.loadModel(from:visionMode:ssdConfig:)  [LLMActor-isolated]
  │
  │  self.ssdConfig = ssdConfig   // actor-isolated stored property
  ▼
LLMActor.extractSnapshotPayloads(_ snapshots:)  [called inside container.perform]
  │
  │  guard self.ssdConfig?.enabled == true else { return [] }
  │      // synchronous property read on LLMActor's own state; no await.
  ▼
LLMActor.ensurePrefixCache()                    [constructs the manager once]
  │
  │  let ssdStore: SSDSnapshotStore? = self.ssdConfig.map {
  │      SSDSnapshotStore(config: $0)
  │  }
  │  let manager = PrefixCacheManager(
  │      memoryBudgetBytes: budget,
  │      alphaTuner: tuner,
  │      ssdStore: ssdStore               // nil if config was nil
  │  )
  ▼
SSDSnapshotStore writer loop + tryEnqueue       [only runs if ssdStore != nil]
```

**Config refresh semantics.** The snapshot is captured **once per model load**. If the user changes a setting while a model is loaded, the change does not take effect until the next `unloadModel()` + `loadModel()` cycle. This is a deliberate trade-off: (a) hot-path reads must be synchronous, (b) propagating a mid-run config change through LLMActor + SSDSnapshotStore is a real refactor for marginal benefit, (c) settings changes in active development are bracketed by model reloads anyway. Documented in `SettingsManager.swift` adjacent to the setters with a one-line comment.

**Why not use `Observation` async sequences to react to setting changes?** Swift 6.2's `Observations<...>` sequence is available and already used elsewhere in the codebase for non-view observation. We could have `LLMActor` subscribe to `settingsManager` changes and update its own `ssdConfig` property. Rejected for Phase 4.1 because (1) an in-flight `container.perform` capture would race the update, (2) the `SSDSnapshotStore` already holds a captured copy from its init and would need a separate invalidation path, (3) the resulting invariants ("what config was in effect at the moment of this specific capture?") become harder to reason about. Reload is the simpler semantic.

No UI surface in Phase 4.1. A Debug settings panel can expose these later if needed — `SettingsManager` is the source of truth, the UI is optional. Effective settings are logged once at `LLMActor` load via `Log.agent.info("prefix-cache ssd enabled=\(enabled) budget=\(budgetBytes) root=\(rootURL) maxPendingBytes=\(maxPendingBytes)")` so operator visibility does not require a UI (see Task 4.1.12 diagnostics).

---

**Task breakdown.** Sequenced, each task stands alone enough to be reviewed/merged independently.

- **Task 4.1.0 — Prerequisites: fingerprint + SSD config plumbing + SettingsManager surface.** Blocker for every later task. Bundles three things that all subsequent tasks depend on. Split into three clearly-labeled subsections in the PR description:
  - **4.1.0.a — Model fingerprint plumbing.** Compute + plumb `modelFingerprint` through `LLMActor` load + `CachePartitionKey`. No SSD interaction. Unit tests for fingerprint stability across reloads and sensitivity to weight swaps.
  - **4.1.0.b — SettingsManager SSD surface.** Add three UserDefaults-backed keys following the existing `Int`/`Bool` pattern in `tesseract/Features/Settings/SettingsManager.swift`: `prefixCacheSSDEnabled: Bool` (default `true`), `prefixCacheSSDBudgetBytes: Int64` (default `20 * 1024 * 1024 * 1024`), `prefixCacheSSDDirectoryOverride: String?` (default `nil`). Includes the `Key` enum additions, the `didSet` persistence, `register(defaults:)` call, and `resetToDefaults` entries. Also adds a new **MainActor-isolated** factory method:
    ```swift
    // In SettingsManager.swift, @MainActor
    func makeSSDPrefixCacheConfig() -> SSDPrefixCacheConfig? {
        guard prefixCacheSSDEnabled else { return nil }
        let rootURL: URL = prefixCacheSSDDirectoryOverride.flatMap(URL.init(string:))
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
                .appendingPathComponent("prefix-cache", isDirectory: true)
        let maxPendingBytes = min(
            4 * 1024 * 1024 * 1024,                                    // 4 GiB hard ceiling
            Int(ProcessInfo.processInfo.physicalMemory) / 16           // 1/16 of physical RAM
        )
        return SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: prefixCacheSSDBudgetBytes,
            maxPendingBytes: maxPendingBytes
        )
    }
    ```
    Tests: default values round-trip via `UserDefaults`; `makeSSDPrefixCacheConfig()` returns `nil` when disabled; `maxPendingBytes` scales correctly with `physicalMemory`.
  - **4.1.0.c — `SSDPrefixCacheConfig` struct + AgentEngine plumbing.** Define the immutable Sendable value type in a new file `tesseract/Features/Server/SSDPrefixCacheConfig.swift`:
    ```swift
    struct SSDPrefixCacheConfig: Sendable, Equatable {
        let enabled: Bool                  // mirror of SettingsManager.prefixCacheSSDEnabled at snapshot time
        let rootURL: URL                   // resolved: override OR sandbox Caches/prefix-cache
        let budgetBytes: Int               // prefixCacheSSDBudgetBytes
        let maxPendingBytes: Int           // front-door byte cap, see decision 22
    }
    ```
    Plumb through three code surfaces:
    1. **`AgentEngine.loadModel(from:visionMode:)` at `AgentEngine.swift:65`** — snapshots the config via `settingsManager.makeSSDPrefixCacheConfig()` on MainActor (AgentEngine is `@MainActor` per line 41, so the call is synchronous). Passes the result into the LLMActor load.
    2. **`LLMActor.loadModel(from:visionMode:ssdConfig:)` (new parameter)** — stores the config as an actor-isolated property `private var ssdConfig: SSDPrefixCacheConfig?`. The property is set once per load and cleared on `unloadModel()`.
    3. **`AgentEngine`'s initializer** — takes a `SettingsManager` reference if it does not already have one. Inspect `DependencyContainer` for the existing wiring; if AgentEngine is already constructed with a settings dependency elsewhere, route through that.
       Tests: the config propagates from SettingsManager → AgentEngine → LLMActor on model load; the LLMActor property is nil before load and after unload; a settings change between loads produces a new config snapshot on the next load.

  **Why this is bundled into 4.1.0.** Every subsequent task reads `self.ssdConfig` or instantiates `SSDSnapshotStore(config:)`. Splitting the plumbing across three different tasks would leave later tasks referencing symbols that don't exist yet. Shipping the plumbing in one PR under 4.1.0 keeps the prerequisite graph clean.

  **Why it's safe to ship 4.1.0 without the SSD tier code.** Nothing in 4.1.0 creates `SSDSnapshotStore` or touches disk. The `SSDPrefixCacheConfig` is a dormant data type; `LLMActor.ssdConfig` is set but nothing reads it. The existing behavior is unchanged. Tasks 4.1.1 through 4.1.9 progressively activate the config.

- **Task 4.1.1 — Data model + schema.** Define `PersistedSnapshotDescriptor`, `PartitionMeta`, `SnapshotManifest`, `SnapshotStorageRef`, `SnapshotPayload`. Pure value types + `Codable`. Round-trip tests. No wiring yet.
- **Task 4.1.2 — `SSDSnapshotStore` skeleton (final class, NSLock-protected front door).** `tryEnqueue(payload:descriptor:node:)` nonisolated synchronous entry point with byte-bounded back-pressure (drop oldest pending on overflow, reject single payloads larger than the cap). Internal `Task.detached { await writerLoop() }` that drains the queue serially. Writer runs the admission-time type-protected LRU cut (see Task 4.1.6), writes safetensors via atomic temp-rename, handles `ENOSPC` / `EDQUOT`, manages the in-memory manifest + debounced persist. **Not a Swift `actor`** — the front door is a lock-protected final class specifically to keep admission non-suspending (decision 22). Unit tests via `FileManager` against a `fs.tempDirectory` subdir: writer loop ordering, coalesce, back-pressure byte budget enforcement, drop-oldest-pending semantics under burst, atomic-rename under simulated crash, manifest debounce, `ENOSPC` recovery, commit/drop callback ordering.
- **Task 4.1.3 — Serialize / deserialize.** Add `HybridCacheSnapshot.serialize(to url: URL, metadata: [String: String]) throws` and `static func deserialize(from url: URL, expectedFingerprint: String) throws -> HybridCacheSnapshot` in `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`. Thin wrapper over `savePromptCache` / `loadPromptCache`. **Thread-affinity contract (documented loudly in the doc comment):** both `serialize` and `deserialize` MUST be called from inside `container.perform` on `LLMActor` (or another Metal-affine context). Swift has no runtime Metal-context detection, so the contract is doc-enforced and trust-the-caller. A `#if DEBUG` precondition at function entry calls `eval(arr[0])` on the first layer's first state array as a smoke check — if the caller is outside a Metal context this will fail-fast in debug builds with a clear message. Bitwise round-trip test: capture → write → read → capture again → compare every `LayerState.state` array byte-for-byte. Also add a test asserting the doc comment exists (grep for the exact contract phrase in the file), so future edits do not silently drop the contract.
- **Task 4.1.4 — Payload extraction on LLMActor + wiring into the three store call sites.** Ships the `LLMActor.extractSnapshotPayloads(_:)` helper AND updates each of the three store call sites per its specific sub-plan (Write path bullets 1a, 1b, 1c). Contract: `extractSnapshotPayloads` must only be called from a Metal-affine context (inside `container.perform`). **Config gating:** reads `self.ssdConfig?.enabled` (actor-isolated synchronous property access, set at model load per Task 4.1.0.c); if `false` or `nil`, returns `[]` without touching the snapshots. **Three separate code changes in this task:**
  1. **Mid-prefill (Write path bullet 1a):** `HTTPPrefixCacheGeneration` gains `let capturedPayloads: [SnapshotPayload]`. `makeHTTPPrefixCacheGeneration` populates it at line ~1023 inside its existing `container.perform` closure. Call site at `:337` passes `mlxStart.capturedPayloads` to `storeSnapshots`.
  2. **Unstripped leaf (Write path bullet 1b):** introduce a new `container.perform { ... }` block around the existing `HybridCacheSnapshot.capture(...)` at line 461. The block captures the leaf AND extracts the payload, returning a `(HybridCacheSnapshot?, SnapshotPayload?)` tuple to the outer scope. Call site at `:487` passes `leafPayload` to `storeLeaf`.
  3. **Stripped leaf (Write path bullet 1c):** add one line after line 1352 inside the existing `container.perform` at line 1275 to extract the payload. Call site at `:1383` passes `strippedLeafPayload` to `storeLeaf`.
     **Spike (before merge):** run on a real mid-prefill `HybridCacheSnapshot` with `QuantizedKVCache` to confirm `asData()` after `eval(cache)` does not trigger a second Metal command-queue submission. If it does, fall back to routing through `savePromptCache(url:cache:metadata:)` inside `container.perform` — Task 4.1.3 ships the wrappers so this fallback is trivial. **Tests:** mock `HybridCacheSnapshot` → extract → verify bytes round-trip; verify the extraction returns `[]` when `ssdConfig?.enabled != true`; three integration tests asserting that each call site correctly plumbs the payload through to `storeSnapshots` / `storeLeaf`.
- **Task 4.1.5 — `SnapshotStore` protocol + `InMemorySnapshotTier`.** Extract the current in-memory behavior behind a protocol. No functional change yet — both tiers point at the same thing. Regression gate: the existing prefix cache test suite must still pass.
- **Task 4.1.6 — `TieredSnapshotStore` composition + storage ref lifecycle + admission-time LRU cut.** Compose the existing RAM tier with the new `SSDSnapshotStore`. **No selective-write-through gate** — every captured snapshot is enqueued to SSD at capture time via the synchronous `SSDSnapshotStore.tryEnqueue`. The writer runs the admission-time type-protected LRU cut (see "Eviction + demotion" bullet 4 and decision 21) when the budget would be exceeded. Implement the full five-state storage ref lifecycle (see "Storage ref lifecycle" section): `pendingRefsByID: [UUID: RadixTreeNode]` map on `TieredSnapshotStore`, `markStorageRefCommitted(id:)` / `markStorageRefDropped(id:)` MainActor callbacks, lookup rules that treat pending (uncommitted) refs as miss, eviction rules that transition states 2→3 and 4→5 on RAM eviction. Unit tests: `.system` writes through immediately and is type-protected on SSD; fresh capture is admitted under budget pressure by evicting oldest non-`.system` SSD resident; lookup on state-3 (pending + body-dropped) returns miss; state-3 correctly transitions to state-5 when commit callback fires; state-3 correctly hard-deletes the node when drop callback fires; admission is non-suspending (synthetic test holds MainActor busy and measures `tryEnqueue` latency — should be microseconds under lock acquisition); byte-budget back-pressure drops oldest pending and clears its ref via `markStorageRefDropped`.
- **Task 4.1.7 — Wire `TieredSnapshotStore` into `PrefixCacheManager` without touching LLMActor call-site shapes.** Replace direct `TokenRadixTree` calls in `PrefixCacheManager.storeSnapshots` / `storeLeaf` / `lookup` / `evictToFitBudget` with `TieredSnapshotStore` calls. Add `restoreStorageRef(path:storageRef:partitionKey:lastAccessTime:)` (mirror of existing `restoreSnapshot` at `PrefixCacheManager.swift:383`). **Explicit non-refactor:** the three LLMActor call sites at `:337`, `:487`, `:1383` keep their synchronous `MainActor.run { } -> StoreDiagnostics` (and coalesced 3-tuple) shapes. The only changes at those call sites are (a) an extra parameter on the closure body (`snapshotPayloads:` / `leafPayload:`) and (b) a new helper call inside the enclosing `container.perform` scope that produces the payloads before the MainActor hop. Assert this with a line-count regression test: the closure bodies at those line numbers should remain synchronous (no `await`) — the test greps the file for "await" inside the specific lines and fails if one appears. Regression gate: full prefix cache test suite passes.
- **Task 4.1.8 — Eviction body-drop + cleanup-suppression guards.** `findEvictionCandidate` returns a victim as today. Three concrete code changes:
  1. **`PrefixCacheManager.swift:512-516` eviction-loop cleanup**: wrap the existing `if isLeaf → evictNode` / `else if childCount == 1 && snapshot == nil → collapseSingleChildNode` branches in a guard that skips both when `candidate.node.storageRef != nil`. When the ref is present (state 2, 3, 4, or 5), leave the node in the tree: the `evictSnapshot(node:)` call already dropped the RAM body; the storageRef keeps the node reachable as an SSD-backed entry. This is the core fix for the "RAM eviction orphans SSD refs" bug flagged on 2026-04-14.
  2. **`TokenRadixTree.swift:333` `collapseSingleChildNode(_:)` guard**: extend the precondition from `node.snapshot == nil && node.childCount == 1` to `node.snapshot == nil && node.storageRef == nil && node.childCount == 1`. Defense-in-depth against external callers (tests, future refactors) that might call `collapseSingleChildNode` without knowing about the SSD tier.
  3. **`TokenRadixTree.swift:220` `evictNode(_:)` debug assertion**: add `assert(node.storageRef == nil, "evictNode on SSD-backed leaf orphans storage ref")` at function entry. The actual suppression is at the call site, but the assertion catches regressions in debug builds.
     **Tests:**
  - Victim with a committed storageRef is body-dropped; the node remains in the tree; a subsequent lookup on the same path returns `.ssdHit` with the storageRef.
  - Victim with a pending storageRef is body-dropped; the node remains in the tree; a subsequent lookup returns miss (because pending refs are not hit targets per lookup rules); a follow-up `markStorageRefCommitted` transitions the node to state 5.
  - Victim without a storageRef is hard-deleted as today — regression check that the existing behavior is preserved for RAM-only nodes.
  - `collapseSingleChildNode` is called on a snapshot-less node with a storageRef → early-return (no collapse happens).
  - A leaf node with a committed storageRef is the selected eviction victim: `evictNode` is NOT called (per the new guard); the node stays in the tree; a subsequent radix lookup on the same path still reaches the node.
  - `.system` type protection is still respected — the `.system` filter in `eligibleEvictionNodes()` is unchanged, so a `.system` victim is never selected while any non-`.system` candidate remains.
- **Task 4.1.9 — Warm start + lazy hydration.** `PrefixCacheManager.warmStart(ssdRoot:)` + `LLMActor.ensurePrefixCache` wiring. Extends `LookupResult.reason` with a new case `.ssdHit(storageRef: SnapshotStorageRef, ssdStore: SSDSnapshotStore, node: RadixTreeNode)`. LLMActor's branch on `.ssdHit` enters `container.perform`, calls `ssdStore.loadSync(storageRef:, expectedFingerprint: self.modelFingerprint!)`, and on success hops to MainActor via `await MainActor.run { ssdStore.recordHit(id:); prefixCache.promote(node:, snapshot:) }`. On `loadSync` failure, LLMActor treats the lookup as a miss (loadSync already fired `markStorageRefDropped` internally via a fire-and-forget MainActor task). Also: initial SSD byte accounting (see Warm start section step 4 for the exact seeding rule). Tests: manifest-only restart produces correct tree structure; first lookup materializes via `loadSync` and promotes to state 4; mismatched fingerprint fires `markStorageRefDropped` and returns miss; file-missing on hydration similarly returns miss; `currentSSDBytes` on the store matches the sum of restored descriptor bytes before any new admission happens.
- **Task 4.1.10 — subsumed into Task 4.1.0.b and 4.1.0.c.** The original scope (three new UserDefaults keys + defaults + reset path) was merged into Task 4.1.0 when the reviewer flagged on 2026-04-14 that the `SettingsManager` wiring had to exist before any task that read `self.ssdConfig` could land. This slot is retained in the numbering so that renaming cascades in downstream tracking don't happen — treat it as "no additional work" and skip ahead to Task 4.1.11.
- **Task 4.1.11 — Benchmark: restart scenario as a separate Step X with its own engine instance.** Extend `PrefixCacheE2ERunner` (`tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift`) with a new Step X that runs **after** the existing Step 4 completes. **Critical constraint**: the existing Steps 1–4 must keep their current behavior, in particular the `requestB2_cold_after_reload` assertion at `:134-138` which requires `requestB2.cachedTokens == 0` after unload/reload. If Step X enabled SSD on the same engine used by Steps 1–4, warm-start would rebuild the tree from the manifest on the reload at `:121-122`, the reload request would hit SSD, and the cold-after-reload assertion would break. **Fix: Step X uses a separate engine instance** with its own SSD config pointing at a per-PID temp directory:
  1. **After Step 4 finishes**, unload the original engine: `await engine.unloadModel()`.
  2. **Construct a second engine** with explicit SSD config:
     ```swift
     let ssdTempDir = FileManager.default.temporaryDirectory
         .appendingPathComponent("tesseract-e2e-ssd")
         .appendingPathComponent(String(getpid()))
     try? FileManager.default.removeItem(at: ssdTempDir)   // fresh start
     try FileManager.default.createDirectory(at: ssdTempDir, withIntermediateDirectories: true)
     let ssdEngine = AgentEngine(ssdConfig: SSDPrefixCacheConfig(
         enabled: true,
         rootURL: ssdTempDir,
         budgetBytes: 4 * 1024 * 1024 * 1024,    // 4 GiB for the test
         maxPendingBytes: 1 * 1024 * 1024 * 1024 // 1 GiB front-door cap
     ))
     defer {
         Task { await ssdEngine.unloadModel() }
         try? FileManager.default.removeItem(at: ssdTempDir)
     }
     try await ssdEngine.loadModel(from: modelDir, visionMode: false)
     ```
  3. **Issue Request X1** with the same system prompt + user message as Request B. Expect `cachedTokens == 0` (cold — first run on this engine). Capture `requestX1.generatedText` as the baseline.
  4. **Unload and reload the SSD engine** (same directory, same config): `await ssdEngine.unloadModel(); try await ssdEngine.loadModel(from: modelDir, visionMode: false)`. Warm-start reads the manifest from the per-PID temp directory and restores the radix tree with `storageRef`-only nodes. **Critical invariant**: `currentSSDBytes` is seeded from the manifest during warm start (see Warm start section step 4).
  5. **Issue Request X2** with the same prompt as X1. Assert:
     - `requestX2.cachedTokens > 0` — warm-start + hydration hit the SSD-resident stable-prefix snapshot.
     - `requestX2.generatedText == requestX1.generatedText` — byte-identical output proves hydration preserved state correctness.
     - `requestX2.ttftSeconds < coldTtftThreshold` — TTFT is within a constant of the measured NVMe read time, not the cold prefill time. Specifically: `requestX2.ttftSeconds < max(requestX1.ttftSeconds * 0.5, 0.200)` (50% of cold or 200 ms, whichever is larger).
  6. **Issue Request X3** (new user message on the same system prompt) to verify the SSD-resident stable-prefix is reused across user message variations. Assert `cachedTokens > 0`.
  7. **Cleanup** the per-PID temp directory via the `defer` block regardless of success/failure.

  Gated by the existing `--prefix-cache-e2e` CLI flag. This is the new top-level correctness gate for Phase 4.1. **The existing Steps 1–4 and the `requestB2_cold_after_reload` check are not touched** — they continue to validate RAM-only correctness on the default `AgentEngine()` engine. Step X runs alongside to validate SSD persistence on the second engine.

- **Task 4.1.12 — Diagnostics + logging.** `PrefixCacheDiagnostics` event cases (aligned with the corrected design — LRU on SSD, not full Marconi): `ssdAdmit(id:bytes:outcome:)` with `SSDAdmitOutcome = .accepted | .droppedByteBudget | .droppedTooLargeForBudget | .droppedDiskFull | .droppedSystemProtectionWins`, `ssdEvictAtAdmission(victimId:, incomingId:)` for visibility into the type-protected LRU cut at write-time, `ssdHit(id:hydrateMs:)`, `ssdMiss(id:reason:)`, `ssdRecordHit(id:)` for every lookup that bumps a committed descriptor's `lastAccessAt` (fires on both state-4 RAM hits and state-5 SSD hydrations), `ssdBodyDrop(id:)` for the RAM-to-SSD transitions (states 2→3 and 4→5), `storageRefCommit(id:)` for the writer commit callback, `storageRefDropCallback(id:, reason:)` for the writer drop callback, `warmStartComplete(partitionCount:snapshotCount:durationMs:)`, `fingerprintMismatch(partition:)`. Wire into the existing logger + the `Log.agent` subsystem. Tests: events emitted in the right order for each code path; `.droppedSystemProtectionWins` fires exactly when the non-`.system` eligible set is empty and the incoming is non-`.system`; `ssdRecordHit` fires on every hit that lands on a committed ref; an effective-settings snapshot is logged at `LLMActor` load via `Log.agent.info("prefix-cache ssd enabled=\(...) budget=\(...) root=\(...) maxPendingBytes=\(...)")` so operator visibility does not require a debug panel (per decision 6).

**Files to change / create.**

Create:

- `tesseract/Features/Server/SnapshotStore.swift` — `SnapshotStore` protocol + `InMemorySnapshotTier` (extracted from current direct usage).
- `tesseract/Features/Server/TieredSnapshotStore.swift` — RAM/SSD composition, `pendingRefsByID: [UUID: RadixTreeNode]` MainActor-isolated map, `markStorageRefCommitted(id:)` / `markStorageRefDropped(id:)` callbacks, body-drop-on-eviction lifecycle rules.
- `tesseract/Features/Server/SSDSnapshotStore.swift` — **`final class`, not a Swift actor.** Holds `let rootURL: URL`, `let budgetBytes: Int`, `let maxPendingBytes: Int` captured from `SSDPrefixCacheConfig` at init. Nonisolated `tryEnqueue(payload:descriptor:node:) -> TryEnqueueResult` under `NSLock` with byte-bounded front door, detached writer `Task`, type-protected LRU admission cut (including the non-system-drop rule in the degenerate case), atomic temp-rename writes, manifest management, `currentSSDBytes` seeded at warm-start from the restored manifest. Also a nonisolated `recordHit(id: UUID)` instance method under the same lock that bumps `manifest.snapshots[id]?.lastAccessAt = .now` and schedules a debounced manifest persist — called from `PrefixCacheManager.lookup` on state-4 RAM hits and from LLMActor's SSD hydration flow on state-5 hits. Also a **nonisolated `loadSync(storageRef:, expectedFingerprint:) -> HybridCacheSnapshot?` instance method** (not a static helper — it reads the immutable `self.rootURL` to derive the file URL) called from LLMActor inside `container.perform` for synchronous `mx.load`; on failure paths it fires `Task { @MainActor in prefixCache.markStorageRefDropped(id:) }` before returning `nil`. Also a private `fileURL(for ref:) -> URL` helper that constructs `{rootURL}/partitions/{partitionDigest}/snapshots/{shardByte}/{snapshotID}.safetensors` from the ref + the store's `rootURL`. The class shape is deliberate — see decision 22.
- `tesseract/Features/Server/SnapshotManifest.swift` — `Codable` types (`PersistedSnapshotDescriptor`, `PartitionMeta`, `SnapshotManifest`, `SnapshotStorageRef`, `SnapshotPayload`, `TryEnqueueResult`).
- `tesseract/Features/Server/ModelFingerprint.swift` — `computeFingerprint(modelDir:) throws -> String`.
- `tesseract/Features/Server/SSDPrefixCacheConfig.swift` — immutable `Sendable` struct with `enabled: Bool`, `rootURL: URL`, `budgetBytes: Int`, `maxPendingBytes: Int`. Consumed by `SSDSnapshotStore.init(config:)`, `LLMActor.loadModel(from:visionMode:ssdConfig:)`, and the extraction gate inside `container.perform`. All field values are derived on MainActor at model load time via `SettingsManager.makeSSDPrefixCacheConfig()`.

Modify:

- `tesseract/Features/Server/PrefixCacheManager.swift` — **`init` signature extended** to `init(memoryBudgetBytes:, alphaTuner: AlphaTuner? = nil, ssdStore: SSDSnapshotStore? = nil)` — the new `ssdStore` parameter is nil-defaulted so `AlphaTuner.replayWindow` (`AlphaTuner.swift:228`) does not need to change. When `ssdStore == nil`, all SSD code paths collapse to RAM-only behavior. Also: add `storageRef` awareness to `LookupResult` (treat state-3 pending refs as miss; state-5 committed refs as `.ssdHit(storageRef:ssdStore:node:)`; state-4 committed-with-body as RAM hit that ALSO fires `ssdStore?.recordHit(id:)` synchronously under the store's lock to bump the descriptor's `lastAccessAt`), add `restoreStorageRef(path:storageRef:partitionKey:lastAccessTime:)`, add `warmStart(ssdRoot:) async throws`, add `promote(node:snapshot:)` for post-hydration promotion, add `markStorageRefCommitted(id:)` and `markStorageRefDropped(id:)` MainActor callbacks, wire `TieredSnapshotStore`, accept new `snapshotPayloads:` / `leafPayload:` parameters on `storeSnapshots` / `storeLeaf`. Post-eviction cleanup path gains the SSD-backed-node guard documented in Task 4.1.8 (skip `evictNode` and `collapseSingleChildNode` when `storageRef != nil`). **Does not become `async`.** Admission stays non-suspending.
- `tesseract/Features/Server/TokenRadixTree.swift` — `RadixTreeNode` grows `storageRef: SnapshotStorageRef?` alongside the existing `snapshot`. A node is "snapshot-bearing" for eviction-eligibility purposes if it has a resident body (states 1, 2, or 4); nodes in states 3 or 5 have no body to drop and are filtered out of `eligibleEvictionNodes()`. `totalSnapshotBytes` tracks **RAM-resident bytes only** (not SSD bytes) — disk-only nodes do not count against the RAM budget. SSD bytes are tracked separately in `SSDSnapshotStore.currentSSDBytes`. **Cleanup-path guards** (per Task 4.1.8):
  - `collapseSingleChildNode(_:)` at line 333 adds `node.storageRef == nil` to its guard, so a storage-ref-pinned node is never collapsed.
  - `evictNode(_:)` at line 220 adds `assert(node.storageRef == nil, ...)` at function entry as a debug-build regression trap.
  - Neither method is allowed to silently drop a `SnapshotStorageRef`.
- `tesseract/Features/Server/EvictionPolicy.swift` — **no formula change, no new enum, no new helper.** The RAM tier continues to use the existing `computeScores` + `selectVictim` flow. The SSD tier runs its own trivial LRU in `SSDSnapshotStore` — no cross-tier helper is needed because at α=0 the two are equivalent and the SSD tier has no access to the tree inputs anyway (see decision 21).
- `tesseract/Features/Server/PrefixCacheDiagnostics.swift` — new event cases: `ssdAdmit(id:bytes:outcome:)` with `SSDAdmitOutcome = .accepted | .droppedByteBudget | .droppedTooLargeForBudget | .droppedDiskFull | .droppedSystemProtectionWins`, `ssdHit(id:hydrateMs:)`, `ssdMiss(id:reason:)`, `ssdBodyDrop(id:)`, `ssdEvictAtAdmission(victimId:, incomingId:)`, `ssdRecordHit(id:)` (fires on every lookup that touches a committed ref — useful for correlating LRU age progression with workload traces), `warmStartComplete(partitionCount:snapshotCount:durationMs:)`, `fingerprintMismatch(partition:)`, `storageRefCommit(id:)`, `storageRefDropCallback(id:, reason:)`.
- `tesseract/Features/Server/CachePartitionKey` (in `PrefixCacheManager.swift`) — add `modelFingerprint: String?`, update `Comparable`.
- `tesseract/Features/Agent/LLMActor.swift` — fingerprint computation at load time, new `ssdConfig: SSDPrefixCacheConfig?` actor-isolated stored property set via a new `loadModel(from:visionMode:ssdConfig:)` parameter (Task 4.1.0.c) and cleared on `unloadModel()` at `LLMActor.swift:637`, `ensurePrefixCache` at `LLMActor.swift:664` awaits `warmStart` on first manager instantiation — if `ssdConfig != nil` it constructs an `SSDSnapshotStore` from the config and passes it into the manager's new `ssdStore:` init parameter; otherwise passes nil (RAM-only). Partition key construction updated for `modelFingerprint` (existing site at `LLMActor.swift:900-905`). **Three separate payload-extraction sites** per Write path bullets 1a/1b/1c: (1) `HTTPPrefixCacheGeneration` gains a `capturedPayloads: [SnapshotPayload]` field populated inside `makeHTTPPrefixCacheGeneration`'s container.perform at line ~1023; (2) unstripped leaf capture at line 461 wrapped in a new container.perform block that captures + extracts atomically; (3) stripped leaf at line 1352 extracts inline inside the existing container.perform at line 1275. The three `MainActor.run { ... }` closure bodies at `:337`/`:487`/`:1383` keep their synchronous shape and existing tuple return types, each gaining an extra parameter for the payload(s). **SSD hit branch**: on `.ssdHit(storageRef:, ssdStore:, node:)` from lookup, LLMActor enters `container.perform`, calls `ssdStore.loadSync(storageRef:, expectedFingerprint: self.modelFingerprint!)`, and on success hops to MainActor via `await MainActor.run { ssdStore.recordHit(id:); prefixCache.promote(node:, snapshot:) }`. Both `loadSync` and `recordHit` are nonisolated instance methods on the store (not static) — they access the store's immutable let-bound state without actor isolation.
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` — `serialize(to:metadata:)` + `deserialize(from:expectedFingerprint:)` wrappers. Additive + non-breaking; no changes to `capture` / `restore` / `chunkedPrefill`. Thread-affinity contract documented in the doc comment + smoke-checked in `#if DEBUG`.
- `tesseract/Features/Settings/SettingsManager.swift` — three new keys (`prefixCacheSSDEnabled: Bool`, `prefixCacheSSDBudgetBytes: Int64`, `prefixCacheSSDDirectoryOverride: String?`) with the existing enum Key → didSet → register(defaults:) → resetToDefaults pattern, plus a new `@MainActor` factory method `makeSSDPrefixCacheConfig() -> SSDPrefixCacheConfig?` that returns the snapshot consumed by AgentEngine on model load. All of this is scoped under Task 4.1.0.b, not Task 4.1.10 (which is now a no-op marker referring back here).
- `tesseract/Features/Agent/AgentEngine.swift` — **init signature changes to `init(settingsManager: SettingsManager? = nil, ssdConfig: SSDPrefixCacheConfig? = nil)`**, both parameters optional, both defaulting to `nil`. `loadModel(from:visionMode:)` at line 65 resolves the config at the start of each load via the following precedence:
  1. If `self.ssdConfig != nil` (set at init) → use it directly. Caller is providing an explicit config and wants no SettingsManager lookup.
  2. Else if `self.settingsManager != nil` → call `self.settingsManager!.makeSSDPrefixCacheConfig()` synchronously on MainActor (AgentEngine is `@MainActor`). Settings changes between loads are picked up here.
  3. Else both are nil → SSD is disabled for the lifetime of the engine. `LLMActor` receives `ssdConfig: nil`, never extracts payloads, never instantiates `SSDSnapshotStore`.
     This resolution lets each of the six existing `AgentEngine()` call sites opt in explicitly without breaking the default behavior. Concrete per-call-site strategy (all six sites must be updated as part of Task 4.1.0.c):

| Call site                                                                  | New call                                                                                                                      | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tesseract/App/DependencyContainer.swift:35`                               | `AgentEngine(settingsManager: settingsManager)`                                                                               | Production path. Picks up the user's live settings via `SettingsManager`.                                                                                                                                                                                                                                                                                                                                                                                           |
| `tesseract/Features/Agent/Benchmark/BenchmarkRunner.swift:27`              | `AgentEngine()` (unchanged; both params nil)                                                                                  | Scenario benchmarks; SSD is disabled to keep measurements reproducible and to avoid mutating `~/Library/Caches` state between runs.                                                                                                                                                                                                                                                                                                                                 |
| `tesseract/Features/Agent/Benchmark/HybridCacheCorrectnessRunner.swift:31` | `AgentEngine()` (unchanged; both params nil)                                                                                  | Bitwise correctness runner; SSD disabled so the test exercises only the RAM path.                                                                                                                                                                                                                                                                                                                                                                                   |
| `tesseract/Features/Agent/Benchmark/PrefillStepBenchmarkRunner.swift:285`  | `AgentEngine()` (unchanged; both params nil)                                                                                  | Prefill-step timing; SSD disabled for reproducibility.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift:40`         | `AgentEngine()` (unchanged; SSD disabled)                                                                                     | **Steps 1–4 keep SSD off.** The existing Step 4 at `PrefixCacheE2ERunner.swift:134-138` asserts `requestB2.cachedTokens == 0` after unload/reload — this is the logit-equivalence proxy that requires a genuinely cold cache on reload. If SSD were enabled on this engine, warm-start would rebuild the tree from the manifest and the reload request would hit SSD, breaking the assertion.                                                                       |
| **NEW: Step X in `PrefixCacheE2ERunner.swift`** (added in Task 4.1.11)     | `AgentEngine(ssdConfig: SSDPrefixCacheConfig(enabled: true, rootURL: perPIDTempDir, budgetBytes: ..., maxPendingBytes: ...))` | **Second engine instance**, constructed AFTER Step 4 finishes. Uses a per-PID temp directory (`{FileManager.default.temporaryDirectory}/tesseract-e2e-ssd/{getpid()}/`) to isolate from production state and from concurrent benchmark runs. The original engine from line 40 is unloaded before Step X's engine is loaded to avoid holding two models in unified memory simultaneously. Step X cleans up the temp directory on exit regardless of success/failure. |
| `tesseractTests/AgentEngineToolSpecTests.swift:59`                         | `AgentEngine()` (unchanged; both params nil)                                                                                  | Unit test of tool-spec construction; has no cache interactions. SSD disabled by default.                                                                                                                                                                                                                                                                                                                                                                            |

**Why two optional parameters instead of a single required one.** The benchmark runners do not want to consult `SettingsManager` (reproducibility), the E2E runner wants an explicit override (per-PID isolation), and production wants the live settings path. Two orthogonal optional parameters cover all three shapes without requiring a `SettingsManager` construction at benchmark time (which would pull in AppKit + ServiceManagement dependencies the CLI paths don't otherwise need). Precedence (ssdConfig over settingsManager) is documented in the init doc comment.
**Migration note.** All six call sites are updated in a single PR (Task 4.1.0.c) — this keeps the build green at every commit. The existing `AgentEngine()` signature semantically becomes "SSD disabled" which is the correct pre-rollout default for everything except the two call sites that explicitly opt in (DependencyContainer and PrefixCacheE2ERunner).

- `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift` — restart scenario (Step X).

**Test topology.**

Unit tests (`tesseractTests/`, Swift Testing framework):

- `SSDConfigPlumbingTests` — covers decision 26's config flow: `SettingsManager.makeSSDPrefixCacheConfig()` returns `nil` when `prefixCacheSSDEnabled == false`; returns a populated snapshot otherwise; `maxPendingBytes` scales correctly with `physicalMemory`; the snapshot round-trips through a mock `AgentEngine` → `LLMActor.loadModel` → `LLMActor.ssdConfig` stored property; mutating a `SettingsManager` setting between two model loads produces two different snapshots on the two loads; mutating during a load does NOT change the in-flight `LLMActor.ssdConfig` (the snapshot is immutable after the load completes). Runs purely against in-memory mocks; no model load required.
- `SnapshotManifestTests` — `Codable` round-trip, schema version mismatch handling, partition digest stability.
- `SSDSnapshotStoreTests` — FIFO writer queue ordering, `tryEnqueue` returns synchronously in microseconds under lock acquisition, byte-budget back-pressure drops oldest-pending and fires `markStorageRefDropped` callback, `.rejectedTooLargeForBudget` on a single-payload-too-large admission attempt, `ENOSPC` handling, atomic `.tmp` → final rename under simulated crash, manifest debounce, **`recordHit(id:)` bumps descriptor.lastAccessAt synchronously under lock** (this is the P2 fix — without it, hot disk-resident entries would keep looking old to the LRU cut).
- `SSDRecordHitIntegrationTests` — end-to-end test that the descriptor's `lastAccessAt` actually moves to the top of the LRU ordering after a lookup: fill SSD with a mix of types, hit one of the older non-`.system` entries, assert it is NOT the next eviction victim when a new entry is admitted. Also: a state-4 RAM hit (body present + committed ref) bumps the SSD descriptor, so that when the body is eventually dropped to state 5 the recency carries forward correctly.
- `TieredSnapshotStoreTests` — composition + lifecycle only, **admission edge cases are deferred to `SSDAdmissionLRUTests`** which owns the asymmetric-fallback rule (decision 24). This suite covers: write-through at capture for every persist-eligible type (`.system`, `.leaf`, `.lastMessageBoundary`, `.branchPoint`); admission-time type-protected LRU cut evicts the oldest non-`.system` SSD resident when the budget would be exceeded in the common case; promote-on-SSD-hit installs body on node; demote-on-RAM-eviction is a body-drop that preserves the `storageRef` in whichever state it already holds; `.system` is type-protected on both tiers during normal operation; non-suspending admission confirmed via a synchronous holder test on MainActor (MainActor queue is held busy and `tryEnqueue` latency is measured — should be microseconds under lock). Does **not** assert that "incoming is always admitted" — that claim was wrong (decision 24 introduces the `.droppedSystemProtectionWins` outcome for non-`.system` incomings under degenerate pressure), and the correct assertion lives in `SSDAdmissionLRUTests`.
- `WarmStartTests` — manifest rebuild after corruption, fingerprint mismatch invalidation, missing file recovery, partial tree rebuild with the existing `PrefixCacheIntegrationTests` fixtures.
- `ModelFingerprintTests` — stability across reloads, sensitivity to weight swap (simulated by writing a dummy weight file with a new mtime).
- `HybridCacheSnapshotSerializationTests` — bitwise round-trip capture → serialize → deserialize → compare every layer's state arrays byte-for-byte; schema version mismatch; expected-fingerprint mismatch.
- `SSDAdmissionLRUTests` — new suite covering the writer-side type-protected LRU cut: `.system` never evicted while any non-`.system` resident remains; oldest-`lastAccessAt` is always the victim among eligible; non-`.system` incoming writes are **dropped** (not admitted) when only `.system` residents remain; `.system` incoming writes may evict older `.system` residents (lateral move) when no non-system candidates remain; budget arithmetic correct across sequential admissions; dropped incomings fire `markStorageRefDropped` via the writer's callback; `.system`-protection holds across a stress scenario that fills SSD with mixed types and then drains it. Runs against an in-memory fake `PersistedSnapshotDescriptor` collection, no real file I/O needed.
- `StorageRefLifecycleTests` — state-machine tests covering all eight transitions (1→2, 2→4, 2→3, 2→1, 3→5, 3→removed, 4→5, 5→4). Each test constructs a `TieredSnapshotStore` in a scratch directory, drives the transition via the public API, and asserts the post-state via lookup + node introspection. Also covers the MainActor callback ordering (commit vs drop vs eviction races).
- `FrontDoorBackPressureTests` — byte-budget back-pressure: enqueue a burst totalling > `maxPendingBytes`, assert oldest-pending is dropped and its `markStorageRefDropped` fires; assert `tryEnqueue` returns `.rejectedTooLargeForBudget` when a single payload exceeds the cap; assert the writer processes exactly the retained set.
- `PrefixCacheManagerSSDTests` — integration of the above: admit → RAM full → demote → lookup → hydrate → promote → verify node state across tiers.

Loaded-model tests (not unit):

- `PrefixCacheE2ERunner.swift` Step X — the restart scenario. New top-level gate for Phase 4.1.
- `HybridCacheCorrectnessRunner.swift` — extended with a serialize/deserialize pass inside the existing bitwise-logit-equivalence check: capture → serialize → deserialize → restore → compare logits against the uncached path. This is the strong gate; the E2E runner is the pipeline-regression gate.

Expected test counts:

- ~60 unit tests across 7 new suites.
- ~15 additional tests on existing suites (regression + extension).
- 2 loaded-model scenarios (E2E restart, serialized correctness).

**Risks (additions to the existing register in this document).**

| Risk                                                                                                                                          | Likelihood           | Impact                 | Mitigation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mx.load` on a background thread deadlocks Metal                                                                                              | **Observed in oMLX** | **Critical**           | Synchronous load on LLMActor inside `container.perform`. Regression test asserts the hydration path never dispatches to a background queue.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `asData()` on a cache-state array triggers a second Metal command-queue submission and stalls inference                                       | Low                  | Medium                 | 30-minute spike in Task 4.1.4 on a real `QuantizedKVCache` snapshot. Fallback: route through `savePromptCache(url:cache:metadata:)` inside `container.perform` — Task 4.1.3 ships the wrappers so the fallback is ready on day one.                                                                                                                                                                                                                                                                                                                                                                                 |
| Write-through-at-capture adds extraction latency to every prefill                                                                             | Medium               | Low                    | Extraction is a `memcpy` of already-`eval`'d tensors on Apple Silicon unified memory, ~2–30 ms for snapshot sizes we've measured (50–600 MiB at >20 GB/s memory bandwidth). This happens inside `container.perform` so the MainActor hop is not affected. `PrefixCacheDiagnostics` records the extraction time per-capture so regressions are visible.                                                                                                                                                                                                                                                              |
| Front-door queue fills during a capture burst                                                                                                 | Medium               | Low                    | **Byte-bounded** front door (`maxPendingBytes = min(4 GiB, physicalMemoryBytes / 16)`, decision 22) enforced synchronously under `NSLock`. Drop-oldest-pending on overflow, with a `markStorageRefDropped` MainActor callback to clear the dropped entry's pending ref. The byte cap is the hard memory bound — total memory held by the pipeline outside the tree is at most `maxPendingBytes`, regardless of burst patterns. Phase 4.1.b can add a byte-per-minute rate cap if real traces show sustained amplification.                                                                                          |
| Spawn-and-return enqueue pattern reappears in a future refactor, breaking the memory bound                                                    | Low                  | High                   | Decision 22 commits to `tryEnqueue` as a synchronous `final class` method under `NSLock`, **explicitly not** a Swift `actor` method. Task 4.1.6 test suite asserts `tryEnqueue` returns synchronously in microseconds under lock acquisition. A future refactor to `actor` would reintroduce the P1 flagged on 2026-04-14 — the risk-register entry + decision 22 serve as the audit trail against silent regression.                                                                                                                                                                                               |
| `storageRef` committed callback races RAM eviction                                                                                            | Medium               | Low                    | Five-state lifecycle (decision 23) makes every race explicit. Lookups treat uncommitted refs as miss (no half-written file ever surfaces). Commit/drop callbacks use `pendingRefsByID` MainActor-isolated map; a node evicted before commit still has a live map entry, so the commit callback always finds its target and can either upgrade to state 5 (if body already dropped) or to state 4 (if body still present).                                                                                                                                                                                           |
| Writer fires commit/drop callback for a node that was already hard-deleted                                                                    | Low                  | Low                    | Callbacks check `pendingRefsByID[id]` and log-and-return if the map entry is missing. The writer is also responsible for deleting the file in the drop path, so no file leak. Commit path in this case is a no-op: the file exists on disk but no node points at it, and the next warm-start directory walk will reconcile (either rebuild a pointer if a matching node exists, or delete the orphan).                                                                                                                                                                                                              |
| Pending ref orphaned on abrupt shutdown (app crash between enqueue and commit)                                                                | Medium               | Low                    | On abrupt shutdown the pending ref is lost (in-memory only), the writer task is killed, the partial file remains as `{id}.safetensors.tmp`. Warm start deletes all `*.tmp` files on init (standard atomic-rename recovery). No committed state is lost because the manifest was only updated after a successful rename.                                                                                                                                                                                                                                                                                             |
| Non-`.system` incoming evicts a protected `.system` resident in the degenerate case                                                           | Low                  | **High**               | Type-protected LRU's asymmetric fallback rule (decision 24): if non-`.system` residents are exhausted AND the incoming is non-`.system`, the writer **drops the incoming** rather than evicting a `.system` resident. Unit test `SSDAdmissionLRUTests` covers this exact scenario — fill SSD with `.system` entries, attempt to admit a fresh `.leaf`, assert the leaf is dropped with `.droppedSystemProtectionWins` outcome. Without this rule, type protection would be ineffective in the degenerate case.                                                                                                      |
| Hot disk-resident entry evicted prematurely because its recency was never updated on SSD hit                                                  | Medium               | Medium                 | `SSDSnapshotStore.recordHit(id:)` (decision 25) fires on every lookup that lands on a committed ref — both state-4 RAM hits and state-5 SSD hydrations. Under the NSLock, O(1). Integration test `SSDRecordHitIntegrationTests` asserts that a hit on an older entry moves it to the top of the LRU ordering. Without this, the LRU cut would pick hot entries as victims.                                                                                                                                                                                                                                          |
| Settings change (e.g. toggle SSD off, raise budget) does not take effect without a model reload                                               | Medium               | Low                    | **Deliberate design** per decision 26. The config is snapshotted at model load and held as an immutable `ssdConfig` property on `LLMActor` for the lifetime of the load. This is a trade-off against the complexity of propagating a live config change through the actor boundary mid-run. Documented in the "Settings + config flow" section and in a one-line comment adjacent to the `SettingsManager` setters. A user who changes the setting can trigger the refresh by unloading and reloading the model (the app already supports this via the standard load path).                                         |
| `SettingsManager` is MainActor but `LLMActor` is its own actor, and the plan's hot path inside `container.perform` cannot await the MainActor | Medium               | **High** if unresolved | Resolved by decision 26 via the `SSDPrefixCacheConfig` snapshot pattern: MainActor-isolated factory (`SettingsManager.makeSSDPrefixCacheConfig()`) → plumbed through `AgentEngine.loadModel` → `LLMActor.loadModel(...ssdConfig:)` → actor-isolated stored property → synchronous read from inside `container.perform` via `self.ssdConfig?.enabled`. Task 4.1.0.c lands the plumbing before any task that reads it. A new unit test under `SSDConfigPlumbingTests` asserts the flow end-to-end.                                                                                                                    |
| Weight swap under same modelID serves stale snapshots                                                                                         | Low (manual ops)     | **Critical**           | `modelFingerprint` folded into `CachePartitionKey` + per-file safetensors metadata. Validated on every hydration, on warm start, and at partition digest level.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Manifest corruption loses the cache                                                                                                           | Low                  | Medium                 | Safetensors headers carry the full `pathFromRoot` so a directory walk can rebuild the manifest. Worst case: ~second-long rebuild on first launch, zero data loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Sandbox Caches directory is purged by macOS under disk pressure                                                                               | Medium               | Low                    | **Deliberate design.** If files disappear, reader returns miss, manifest self-heals on next write. Never treat the SSD tier as authoritative.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| APFS rename non-atomic across filesystems                                                                                                     | Low                  | Medium                 | Temp file lives in the same sandbox Caches directory as the final file, so rename is always intra-volume and thus atomic.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Disk fills during `.system` persist, user pays cold prefill forever                                                                           | Low                  | Medium                 | `ENOSPC` → log + evict oldest SSD entry to make room + retry once. If still fails, drop the write and log. Subsequent lookups miss cleanly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Fingerprint is too weak (same size + mtime, different content)                                                                                | Very low             | High                   | Manual escape hatch: delete the sandbox cache directory (`~/Library/Containers/{bundleID}/Data/Library/Caches/Tesseract Agent/prefix-cache/`) — the next model load rebuilds the manifest from scratch and any stale persisted entries are gone. This is a documented operator procedure, not a new `SettingsManager` API — Phase 4.1 does not ship a dedicated "clear prefix cache" method. Phase 4.1.b can upgrade the fingerprint to a streaming SHA-256 of the first weight file (adds ~100 ms to load) and/or add an explicit `SettingsManager.clearPrefixCacheSSD()` action if this ever bites in production. |
| Admission goes async in a future refactor and synchronous call sites break silently                                                           | Medium               | High                   | Task 4.1.7 includes a line-count regression test that greps the three call sites at `LLMActor.swift:337`, `:487`, `:1383` for any `await` keyword appearing inside the closure body. If anyone adds an `await` there, the test fails with a pointer at this decision record. Documents the invariant in a way that survives edits.                                                                                                                                                                                                                                                                                  |
| `.lastMessageBoundary` accumulation overwhelms recency term on long conversations                                                             | Medium               | Low                    | Pure Marconi scoring with the 20 GiB top-level cap is the Phase 4.1 choice (see decision 20). If real traces show the recency term is insufficient, Phase 4.1.b can add a per-type admission filter (Marconi's "admission gate" pattern from §4.1). The E2E runner's 20-turn variant surfaces this early.                                                                                                                                                                                                                                                                                                           |

---

**Decisions (resolved in review, 2026-04-14; revised 2026-04-14 after P1/P2 findings).**

1. **Phase ordering — resolved.** All of Phases 0–3 are already shipped (verified against the Phase Dependency Graph above). Phase 4.1 follows naturally after them and does not re-sequence anything.
2. **Write policy — resolved.** Strict write-through at capture time, no write-back, no hot-cache mode. See "Write path" section for the full flow. Revisit in Phase 4.1.b only if write amplification shows up in real traces.
3. **Sync reads on the inference thread — resolved.** Synchronous `mx.load` inside `container.perform` on `LLMActor`. The oMLX Metal-deadlock regression test is sufficient evidence; we do not re-prove it on MLX Swift. A regression test in our own suite asserts the hydration path never dispatches to a background queue.
4. **Cache directory — resolved.** Sandbox `~/Library/Caches/Tesseract Agent/prefix-cache/` (redirected through the app-sandbox container). Free macOS purging safety valve; semantic fit for derivative data.
5. **Model fingerprint — resolved.** SHA-256 over `config.json` bytes + `tokenizer.json` bytes + sorted list of `(filename, size, mtime)` for every `*.safetensors` in the model directory. ~1 ms at load. Full weight-bytes hashing is an upgrade path only if this weaker form actually bites.
6. **Settings UI — resolved.** Hidden UserDefaults keys only. No Settings window surface in Phase 4.1. **Operator visibility** is handled by logging the effective SSD settings once at `LLMActor` load via `Log.agent` — path, budget, enabled flag — so an operator can grep the log without a debug panel. Full UI is deferred indefinitely.
7. **Restart benchmark — resolved.** In-process tear-down via the existing `engine.unloadModel()` + `engine.loadModel(...)` machinery (`LLMActor.swift:637`, `PrefixCacheE2ERunner.swift:105`). This is the blocking correctness gate for Phase 4.1. A cross-process smoke test is added only when we flip the default-on flag for non-dev builds (not applicable in active development, see decision 10).
8. **No pre-hydration of top-K snapshots on launch — resolved.** Idea 4.3 is deferred. The first hit on each `.system` snapshot pays the ~40 ms NVMe read; after that it's in RAM.
9. **One file per snapshot — resolved.** No container file / segment packing. Snapshot count is bounded by `SSDBudget / minSnapshotSize` (low hundreds on the default 20 GiB budget). Phase 4.1.c can add segment packing if file count ever becomes a measurable problem.
10. **Rollout — resolved.** Ship immediate, default-on, no feature-flag scaffolding, no `#if FEATURE_SSD_PREFIX_CACHE`. The codebase is in active development phase and the user will manually validate each task as it lands. The restart benchmark (Task 4.1.11) is the blocking correctness gate; merging proceeds as soon as it passes.
11. **Marconi across both tiers — resolved, revised 2026-04-14 (second revision).** The RAM tier runs the Phase 2 Marconi formula (`EvictionPolicy.computeScores`) with the alpha tuner's live α. The SSD tier runs **type-protected LRU with an asymmetric-incoming fallback** (decisions 21 + 24): the scoring function is recency-only (equivalent to `computeScores` at α=0), and the degenerate-case fallback drops non-`.system` incomings rather than evicting protected `.system` residents. Rationale: (a) at startup before the tuner runs, both tiers are effectively at α=0 and behaviorally equivalent in the common case; (b) once the tuner writes `bestAlpha` above 0, the RAM tier moves to the full formula while the SSD tier stays at LRU, and Phase 4.1 accepts the divergence (decision 27); (c) the degenerate-case fallback behavior differs by design — RAM's fallback cascades to `.system` eviction because RAM eviction is mandatory, SSD's fallback drops the incoming because SSD admission is optional, and the SSD rule is the stronger type protection; (d) extending the descriptor with tree-structure metadata is a real complexity cost we don't need today; (e) the upgrade path to full Marconi on SSD is documented (Phase 4.1.b, gated on the alpha tuner raising α above 0 AND production traces justifying it). Type protection for `.system` holds on both tiers regardless of α. See the "Eviction + demotion" and "Marconi extension" sections for the full rationale.
12. **MLXArray → Data extraction — resolved pending spike.** `MLXArray.asData()` exists and is used on the VLM hot path at `tesseract/Features/Agent/UserInput.swift:128`. Task 4.1.4 does a 30-minute spike on a real `HybridCacheSnapshot` to confirm that calling `asData()` on a cache-state array _after_ `eval(cache)` does not trigger a second Metal command-queue submission. If the spike fails, fall back to routing through `savePromptCache(url:cache:metadata:)` inside `container.perform` (Task 4.1.3 already ships the wrappers that make this fallback trivial).
13. **Vendor change to `HybridCacheSnapshot` — approved.** Task 4.1.3 adds `serialize(to:metadata:)` + `deserialize(from:expectedFingerprint:)` to `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`. Additive only; no changes to `capture` / `restore` / `chunkedPrefill`. **Explicit thread-affinity contract:** both helpers must be called from inside `container.perform` on `LLMActor`. Documented loudly in the doc comment, smoke-checked in `#if DEBUG` via an `eval` on the first layer's first array at function entry, and guarded by a test that greps for the contract phrase in the file to prevent silent edits.
14. **`hitCount` on `RadixTreeNode` — withdrawn.** The earlier draft added a `hitCount` field to support a selective-write-through gate for `.branchPoint` captures. The gate is **dropped** in this revision (see decision 15), so the `hitCount` field is no longer needed in Phase 4.1. Drop the field. Phase 4.1.b can reintroduce it if the paper's speculative-admission (§4.1 purely-input) rule becomes applicable to Tesseract's single-user workload.
15. **Selective-write-through gate — dropped.** The earlier draft had a gate deferring `.branchPoint` captures until first reuse (SGLang HiCache pattern). It is dropped because: (a) re-extraction on hit would require crossing the Metal-affinity boundary from MainActor back to LLMActor, which is a layer violation; (b) keeping a resident payload alongside the RAM snapshot for the full residency period doubles per-snapshot memory; (c) write amplification on a 20 GiB cap on consumer NVMe is a non-issue — Apple SSDs have multi-PB endurance and the budget rotates data naturally; (d) the logic complexity (orphan extraction, promotion tracking, tier-boundary state machines) exceeded the expected value on a single-user workload. Every persist-eligible snapshot is written through at capture time. Phase 4.1.b can reintroduce a gate if production traces show amplification biting.
16. **Test `ssdRoot` override — resolved.** Constructor parameter on the `SSDSnapshotStore` class (`init(rootURL:)`). `SettingsManager` produces the root URL from either the optional override or the default; tests construct the store directly and never touch `SettingsManager`.
17. **Alpha tuner sandbox — resolved with an explicit API change to `PrefixCacheManager.init`.** `PrefixCacheManager.init` at `PrefixCacheManager.swift:77` currently has signature `init(memoryBudgetBytes: Int, alphaTuner: AlphaTuner? = nil)`. Phase 4.1 **extends it with a new optional parameter**:
    ```swift
    init(
        memoryBudgetBytes: Int,
        alphaTuner: AlphaTuner? = nil,
        ssdStore: SSDSnapshotStore? = nil           // new; nil = RAM-only tier
    )
    ```
    Semantics: when `ssdStore == nil`, all SSD-related code paths inside `PrefixCacheManager` (write-through at capture, the `markStorageRefCommitted` / `markStorageRefDropped` callbacks, the `.ssdHit` lookup branch, the body-drop vs hard-delete disposition) are no-ops or collapse to the RAM-only behavior. This is exactly what the sandbox replay path needs.
    **`AlphaTuner.replayWindow()` at `AlphaTuner.swift:228`** currently constructs `PrefixCacheManager(memoryBudgetBytes: simBudget, alphaTuner: nil)`. Phase 4.1 leaves this call **unchanged** — it already omits the new `ssdStore:` parameter, so the default `nil` kicks in and the sandbox manager is RAM-only. **No changes to `AlphaTuner.swift` are required for SSD isolation.** The fix is entirely on the `PrefixCacheManager` side: adding the parameter with a safe default.
    **Regression test** added to `AlphaTunerTests`: construct a real `SSDSnapshotStore` against a scratch directory, pass it into a "production" `PrefixCacheManager` with SSD enabled, run `AlphaTuner.runGridSearch()` on a synthetic trace, assert that `SSDSnapshotStore.tryEnqueue` is **never** called during replay (instrumented via a test hook or a spy on the store), assert that the scratch directory remains empty after replay, and assert that `EvictionPolicy.alpha` is restored to its pre-replay value after the grid search completes.
18. **P1 resolution: drop-vs-cascade inconsistency — resolved by construction.** The P1 flagged 2026-04-14 ("new RAM demotions can be dropped while lower-utility SSD entries survive") is prevented by the write-through-at-capture invariant: RAM eviction becomes a body-drop on an already-persisted entry, so there is no "new demotion to admit" that competes with existing SSD residents. The only SSD admission decision happens at capture time, inside the writer, using type-protected LRU (decision 21). No cascade, no late admission, no inconsistency. This is also the simplest mental model for reviewers: one write point, one eviction-decision point, one formula.
19. **P1 resolution: async store integration — resolved by non-suspending admission.** The P1 flagged 2026-04-14 ("LLMActor store call sites cannot stay as-is if admission suspends") is prevented by keeping admission non-suspending: `PrefixCacheManager.storeSnapshots` and `storeLeaf` stay synchronous functions, they call `SSDSnapshotStore.tryEnqueue` (nonisolated, lock-protected, synchronous — **not** a spawn-and-return wrapper), and the three `MainActor.run { ... }` closures at `LLMActor.swift:337`, `:487`, `:1383` keep their existing synchronous shape. Task 4.1.7 includes a line-count regression test asserting no `await` appears inside those closure bodies.
20. **P2 resolution: cap semantics — resolved, explicit choice of pure LRU (α=0 Marconi).** The P2 flagged 2026-04-14 ("earlier draft rationale about bounding long-conversation accumulation is now gone") is resolved by an explicit documented choice. There are **no per-type caps** in Phase 4.1. The only cap is the top-level `prefixCacheSSDBudgetBytes` (20 GiB default), applied uniformly across entry types. The earlier draft's proposed `N=4 most-recent .lastMessageBoundary` cap is rejected because (a) it violates Marconi's "one formula, one knob" convention per the paper's §4.2/§4.3, (b) LRU already penalizes stale boundaries under real workloads, (c) the top-level budget is a sufficient single cap. If production traces show the recency term is insufficient, Phase 4.1.b can add a per-type admission filter — which Marconi's authors would call an admission gate and which matches the paper's admission-first philosophy. Full reasoning is in the "Eviction + demotion" section bullet 8.
21. **P1 resolution (third-revision P1): SSD-tier scoring inputs — resolved by simplification to type-protected LRU.** The P1 flagged on 2026-04-14 ("the 'same Marconi formula' claim is stronger than the stored data supports") is resolved by scoping the SSD-tier eviction to **type-protected LRU**: sort by `lastAccessAt` ascending, filter `checkpointType != .system`, evict oldest non-`.system` until the new entry fits. In the degenerate case where non-`.system` residents are exhausted, the fallback rule is asymmetric on the incoming's type: `.system` incoming falls through to evicting older `.system` residents (lateral move); non-`.system` incoming is dropped with outcome `.droppedSystemProtectionWins` (see decision 24 and Eviction + demotion bullet 4). This is the simplest correct reading of "Marconi on SSD" given the descriptor schema — and it **matches current production behavior** because α=0 is the live default (LRU is `EvictionPolicy.computeScores` at α=0). The descriptor schema does NOT carry `parentTokenOffset` or `childCount` — these inputs are only meaningful with live radix-tree state, which the writer's isolation domain cannot access without a cross-actor hop per scoring call. Extending the schema to support those inputs is a real complexity cost (staleness on `childCount`, schema versioning for `parentTokenOffset`) that buys no measurable benefit at α=0. Upgrade path: if the alpha tuner raises α above 0 AND production traces show a measurable difference between full Marconi and LRU on the SSD tier, Phase 4.1.b extends the descriptor schema and promotes the writer's scoring to full Marconi. Until then, type-protected LRU with the asymmetric-incoming fallback is the correct shipped behavior.
22. **P1 resolution (third-revision P1): front-door byte-bound admission — resolved with NSLock + byte cap, not spawn-and-return.** The P1 flagged on 2026-04-14 ("fire-and-forget enqueue bypasses the intended memory bound") is resolved by making `SSDSnapshotStore.tryEnqueue` a **nonisolated synchronous `final class` method protected by `NSLock`**, not a Swift `actor` method wrapped in a detached `Task`. The earlier draft's `enqueuePersist` spawned a `Task { await self._acceptEnqueue(payload) }` which would retain the full payload inside the task's heap allocation before the actor's queue could apply back-pressure — under a burst of captures this could accumulate multiple GiB of pending bytes outside the queue cap. The corrected shape enforces byte-based back-pressure **at the front door**, synchronously, before the payload can escape the caller's scope: `tryEnqueue` acquires the lock, checks `pendingBytes + payload.totalBytes <= maxPendingBytes`, drops the oldest pending write under overflow (with a MainActor callback to clear the dropped entry's pending `storageRef`), admits the new payload if it fits, and signals the writer via `AsyncStream.Continuation.yield()`. `maxPendingBytes` defaults to `min(4 GiB, physicalMemoryBytes / 16)` — concretely 4 GiB on a 64 GB Mac, 1 GiB on a 16 GB Mac. This is a bytes cap, not a depth cap, because payloads range 50–600 MiB and a depth cap would not bound memory. The front-door code is written once and is 30 lines; it should never be refactored into a Swift `actor` because the implicit `async` on actor methods would break the non-suspending admission invariant.
23. **P1 resolution (third-revision P1): `storageRef` lifecycle — resolved with a five-state machine + commit/drop callbacks.** The P1 flagged on 2026-04-14 ("the plan relies on `storageRef`, but never defines when it becomes committed") is resolved by an explicit five-state machine on each `RadixTreeNode`, spelled out in the "Storage ref lifecycle" section above. Key invariants: (a) a `SnapshotStorageRef` is attached to the node at `tryEnqueue` acceptance with `committed: false`; (b) the writer fires `Task { @MainActor in prefixCache.markStorageRefCommitted(id:) }` on successful fsync/rename, flipping the flag to `true`; (c) the writer fires `markStorageRefDropped(id:)` on failure or drop, clearing the ref entirely; (d) lookups treat `committed: false` refs as misses, so no race can surface a half-written file to a caller; (e) RAM eviction on a node with a pending ref body-drops the body but leaves the ref in pending state (transitions to state 3); (f) commit/drop callbacks look up nodes via a `pendingRefsByID: [UUID: RadixTreeNode]` MainActor-isolated map, so the writer never holds a tree reference. The map holds nodes strongly so the callback can always find them even after the tree's normal eviction flow would have collected them. Map entry is removed on commit or drop.
24. **P1 resolution (fourth-revision P1): SSD type protection does not fall through to `.system` eviction for non-`.system` incomings.** The P1 flagged on 2026-04-14 ("SSD LRU still lets a new non-system entry evict the protected system snapshot") is resolved by making the admission fallback **asymmetric** with respect to the incoming's type: in the degenerate case where all non-`.system` residents have been evicted and the incoming still does not fit, the writer checks the incoming's type. If incoming is `.system`, the writer falls through to evicting older `.system` residents (lateral move; protection is preserved across the set). If incoming is non-`.system`, the writer **drops the incoming write** and fires `ssdAdmit(outcome: .droppedSystemProtectionWins)`. This is the Marconi-faithful answer — a fresh leaf or boundary is less valuable than an existing `.system` resident, so the correct behavior is to preserve the high-value resident rather than admit a low-value incoming. The earlier draft's "fallback to oldest including `.system` in the degenerate case" was a regression that would have let non-system incomings evict protected `.system` entries, and is fixed here. The asymmetry between RAM and SSD fallback — RAM always falls through to `.system` eviction in its mandatory eviction path, SSD drops non-system incomings instead — exists because on RAM the eviction is mandatory (the budget MUST be satisfied or the manager is in violation) while on SSD the admission is optional (rejecting the incoming is a clean return path). See Eviction + demotion bullet 4 for the full rule.
25. **P2 resolution (fourth-revision P2): SSD `lastAccessAt` is updated on every hit via `recordHit(id:)`.** The P2 flagged on 2026-04-14 ("SSD LRU depends on `lastAccessAt`, but the hit path never updates it") is resolved by an explicit new method `SSDSnapshotStore.recordHit(id: UUID)`: nonisolated, lock-protected, synchronous, O(1). It bumps `manifest.snapshots[id]?.lastAccessAt = .now` in the in-memory manifest and schedules a debounced persist on the same 500 ms timer as writes. It is called from two places: (a) `PrefixCacheManager.lookup` on a state-4 RAM hit (body present + committed ref), so that a hot RAM entry's SSD copy inherits recency for the eventual body-drop transition; (b) LLMActor's SSD hydration flow after `loadSync` returns successfully, so that a state-5 hit actually bumps the descriptor's recency. Without this method, hot disk-resident entries would keep looking old to the LRU cut and would get evicted prematurely despite recent reuse. The call is fast (single dictionary touch under an already-taken lock) and the lookup hot path can absorb it. See Read/hydration path bullets 2 and 5.
26. **P1 resolution (fifth-revision P1): Settings flow — resolved with an immutable config snapshot + load-time plumbing.** The P1 flagged on 2026-04-14 ("settings surface is still not wired into the runtime plan") is resolved by introducing `SSDPrefixCacheConfig`: an immutable `Sendable` value type snapshotted on MainActor by `SettingsManager.makeSSDPrefixCacheConfig()` and passed through `AgentEngine.loadModel(...)` → `LLMActor.loadModel(from:visionMode:ssdConfig:)` → stored as an actor-isolated property on `LLMActor` → consumed synchronously from inside `container.perform` via `self.ssdConfig?.enabled` (actor-isolated property read, no await). `SSDSnapshotStore` receives the same snapshot via its init and captures `rootURL`, `budgetBytes`, and `maxPendingBytes` as immutable properties. All three symbols (`SSDPrefixCacheConfig`, `makeSSDPrefixCacheConfig`, and the new `ssdConfig:` parameter on `LLMActor.loadModel`) ship together in Task 4.1.0.c — _before_ any task that reads them — so the prerequisite graph is clean. **Config refresh requires a model reload.** Settings changes in flight do not propagate to the currently-loaded model; the next `unloadModel()` + `loadModel()` cycle picks up the new values. Rationale: (a) hot-path reads must be synchronous from inside `container.perform`, (b) propagating a mid-run config change through LLMActor + SSDSnapshotStore is a real refactor for marginal benefit, (c) settings changes in active development are bracketed by model reloads anyway. An `Observations<...>`-based live-update path was considered and rejected for Phase 4.1; see the "Settings + config flow" section above for the full rationale. **`AgentEngine` gains two optional init parameters** (`settingsManager: SettingsManager? = nil`, `ssdConfig: SSDPrefixCacheConfig? = nil`) with the precedence rule `ssdConfig > settingsManager > disabled`, and all six existing `AgentEngine()` call sites are updated per the table in the `AgentEngine.swift` Modify entry of the Files-to-change section: `DependencyContainer.swift:35` passes `settingsManager`, `PrefixCacheE2ERunner.swift:40` is left **unchanged** (default `AgentEngine()`, SSD disabled) so the existing `requestB2_cold_after_reload` assertion at `:134-138` keeps working, and the five other benchmark/test callers keep the default (both nil = SSD disabled) because reproducibility matters more than SSD coverage for their scenarios. **A new Step X added in Task 4.1.11** constructs a second `AgentEngine` instance (also inside `PrefixCacheE2ERunner`) with an explicit per-PID `ssdConfig` to exercise the SSD persistence path without touching Steps 1–4; see Task 4.1.11 for the full scenario.
27. **Accepted divergence: RAM = tuned Marconi, SSD = LRU after tuner runs.** The P2 flagged on 2026-04-14 ("SSD/RAM policy 'equivalence today' is still overstated once the alpha tuner runs") is resolved by explicitly accepting the post-tuning divergence rather than papering over it. Facts: `LLMActor.swift:666` resets `EvictionPolicy.alpha = 0.0` at cache creation; `AlphaTuner.swift:193` writes `bestAlpha` when the grid search completes; `AlphaTuner.swift:227` is the tuner's alpha setter on the hot path. Between startup and `bestAlpha` landing, both tiers run at α=0 and are behaviorally equivalent. After `bestAlpha` lands, the RAM tier factors FLOP efficiency into its victim selection while the SSD tier stays at pure recency. Phase 4.1 **accepts** this divergence because (a) the descriptor schema can't carry the inputs needed to mirror the RAM formula (decision 21), (b) the divergence is expected to be small in practice — the recency term typically dominates utility for workloads where α is modest, and the `.system` type protection is upstream of the scoring so it holds regardless of α, (c) the divergence is **asymmetric in scope by design**: RAM scoring governs only which entries stay resident in RAM; SSD persistence was already decided at capture time by the write-through-at-capture path (Write path bullet 1), so post-tuning RAM scoring cannot retroactively promote or demote anything on the SSD tier. What it CAN influence is which entries have a resident RAM body at the moment they're hit (state 4 vs state 5), which affects the p99 latency profile but not which entries are persistent — (d) the upgrade path is clean (Phase 4.1.b extends the descriptor schema + promotes the writer's scoring to the full formula). The plan has been swept to replace "observationally equivalent today" / "literally equivalent" language with pre/post-tuning-scoped claims. `SSDRecordHitIntegrationTests` and the production E2E runner's 20-turn variant are the empirical checks — if they reveal a measurable divergence penalty, Phase 4.1.b is the landing pad. **Explicit correction of an earlier draft:** an earlier version of this decision said "RAM eviction choices still propagate to SSD via the demote-on-eviction path" — that was a stale reference to a write-back-at-eviction design we no longer use. Under write-through-at-capture, demote-on-eviction is a body-drop only; it does not initiate any SSD work, and the SSD tier's admitted set is entirely determined by write-time decisions.

**Remaining spikes and in-development verification (non-blocking).**

These are not open design questions — they are empirical checks that happen during implementation, each scoped to a single task. None of them block the Phase 4.1 greenlight; they are listed here so the owner of each task knows what to verify before landing.

1. **Task 4.1.4 — `asData()` Metal-sync spike.** Described in decision 12 above. Run on a real mid-prefill `HybridCacheSnapshot` with the `QuantizedKVCache` path (the most common in production) and confirm no observable stall on the inference path. Expected outcome: passes cleanly. Fallback outcome: switch the extraction step to route through `savePromptCache(url:cache:metadata:)` directly (wrapped inside `container.perform`) — Task 4.1.3 ships the wrappers so the fallback is a ~30-line change.
2. **Task 4.1.4 — Extraction latency profiling.** Write-through at capture adds a per-capture `memcpy` step inside `container.perform`. The expected cost is 2–30 ms for snapshot sizes we ship (50–600 MiB at >20 GB/s Apple Silicon memory bandwidth). Instrument via `PrefixCacheDiagnostics.captureDuration` and confirm the p95 is in the expected range. Regression if p95 > 100 ms on the default config.
3. **Task 4.1.2 — Manifest debounce window.** Default to 500 ms; instrument via `PrefixCacheDiagnostics` during the E2E runner's restart scenario; tune based on measured I/O load. Not a correctness concern, a cost concern.
4. **Task 4.1.9 — `InferenceArbiter` interaction.** Hydration adds ~40 ms to the prefill path. Verify that the arbiter's single-in-flight invariant and timeout logic absorb this without surprise. Probably fine (40 ms << cold prefill), but worth one manual run.
5. **Task 4.1.11 — `.lastMessageBoundary` accumulation under long conversations.** The Marconi scoring handles this correctly by design (old boundaries are low-utility, evicted first on SSD pressure), but the E2E runner should include a 20-turn conversation variant that exercises the SSD eviction path, so we have a concrete trace of the behavior to sanity-check against the alpha-tuner's existing workload model. This is the direct empirical check on decision 20 (pure Marconi without the per-type cap).
6. **Task 4.1.7 — Non-suspending admission regression test.** Line-count regression test that greps `LLMActor.swift` for any `await` keyword inside the three `MainActor.run` closure bodies at `:337`, `:487`, `:1383`. Fails fast if a future refactor breaks the non-suspending admission invariant that these call sites depend on. Pairs with risk-register entry "Admission goes async in a future refactor and synchronous call sites break silently."

---

**Rollout.**

- **Ship immediate, default-on.** The codebase is in active development phase; the user manually validates each task as it lands.
- **No feature flag scaffolding.** No `#if FEATURE_SSD_PREFIX_CACHE`, no `prefixCacheSSDEnabled = false` default. The setting defaults to `true`; a user who wants to turn it off does so via `defaults write`.
- **Merge order** is the task order (4.1.0 → 4.1.12). Each task passes the existing prefix cache test suite before merging, and the restart benchmark (Task 4.1.11) is the final blocking correctness gate.
- **Manual validation** at each task: the user drives the app through a typical session (start → issue a request → kill → relaunch → issue another request → observe SSD hit in logs) to sanity-check each increment.
- **Telemetry floor.** `PrefixCacheDiagnostics` events (Task 4.1.12) give enough visibility to diagnose any surprise without a debug panel. Effective settings are logged at `LLMActor` load time. Grep is the UI.

**Expected review asks (anticipated).**

- "Why write-through at capture instead of write-back at eviction?" — Three reasons, in rough order of importance. (1) **Metal affinity.** Payload extraction must run inside `container.perform` on `LLMActor` (the oMLX regression). A write-back design triggers extraction at eviction time, which happens on MainActor during the eviction loop — crossing back to LLMActor for each demote is a layer violation, adds two actor hops per demote, and makes the eviction loop implicitly async. (2) **Non-suspending admission.** Write-through at capture lets the MainActor admission call (inside `prefixCache.storeSnapshots`) be a non-suspending fire-and-forget enqueue, which preserves the three `MainActor.run { } -> Tuple` synchronous closures at `LLMActor.swift:337`, `:487`, `:1383`. A write-back design forces those closures to become async. (3) **Drop-vs-cascade cleanup.** Write-through means the SSD admission decision is made once, at capture, with Marconi scoring. RAM eviction is a pure body-drop on already-persisted data — no competing admission at eviction time, no cascade question.
- "Why no selective-write-through gate like SGLang HiCache?" — The gate (defer `.branchPoint` captures until first reuse) was in the earlier draft. Dropped because re-extraction on hit crosses the Metal-affinity boundary the wrong way, and write amplification on a 20 GiB cap on consumer NVMe is a non-issue. See decision 15.
- "Can we prove the fingerprint is strong enough?" — Yes, with a unit test that writes a new config.json and asserts the fingerprint changes, plus a brief write-up in the test comments.
- "What happens under simulated crash?" — The atomic rename + recovery-from-headers flow handles it. Add a unit test that kills the writer mid-flight and asserts warm start recovers cleanly.
- "Why not use `FileHandle`'s async APIs instead of an actor?" — Actor gives us ordered serial writes + queue back-pressure + coalescing + manifest ownership in one unit. `FileHandle` async is lower-level and doesn't solve the coordination problems.
- "Why JSON for the manifest instead of a binary index?" — Human debuggable, easy to version, and the size is small (~few KB per hundred snapshots). No measurable cost.
- "Why type-protected LRU on SSD instead of the full Marconi formula?" — Three reasons. (1) **Startup α=0.** At startup and before the alpha tuner finishes its grid search, `EvictionPolicy.alpha` is `0.0` (`EvictionPolicy.swift:68`, reset in `LLMActor.swift:666`), at which point `EvictionPolicy.computeScores` collapses to LRU within the eligible set — so LRU on SSD is behaviorally equivalent to Marconi on SSD at startup. **After the tuner writes `bestAlpha` at `AlphaTuner.swift:193`, the policies diverge** and Phase 4.1 accepts the divergence (decision 27). (2) **Descriptor schema honesty.** Full Marconi needs `parentTokenOffset` (for the parent-relative FLOPs term) and `childCount` (for topological eligibility), both of which live on live `RadixTreeNode` instances and cannot be inspected from the writer's isolation domain without a cross-actor hop per scoring call. The earlier draft claimed "same formula on both tiers" without carrying those inputs in `PersistedSnapshotDescriptor` — that claim was stronger than the data. This revision is honest about what the schema supports. (3) **Upgrade path is clean.** If the alpha tuner raises α above 0 and production traces show a measurable divergence penalty, Phase 4.1.b extends the descriptor schema and promotes the writer to full Marconi. Until then, shipping LRU is the right call. See decisions 21 + 27 and Eviction + demotion bullets 4 and 5.
- "Why not make `SSDSnapshotStore` a Swift actor?" — Swift actor methods are implicitly `async`, which forces the admission path into `await`. That would either (a) force the three synchronous `MainActor.run` closures at `LLMActor.swift:337`/`:487`/`:1383` to become async (real refactor, task-cancellation handling that isn't there today) or (b) use spawn-and-return (`Task { await ... }`), which retains the full payload inside the task heap before the actor's queue can apply back-pressure — and on a burst of captures that would accumulate multiple GiB of pending bytes outside the cap. Neither is acceptable. The lock-protected `final class` + detached writer pattern is the correct shape. See decision 22.

### Idea 4.2 — Snapshot quantization for RAM efficiency

The snapshot size table in Implementation Note 2 shows a 16K unquantized snapshot at ~586 MiB. Most of that is attention KV; SSM state is fixed at ~74 MiB regardless of sequence length. Runtime already uses `QuantizedKVCache` with `kvBits = 8`, but the snapshots we capture inherit whatever type the live cache is using at the checkpoint offset — early mid-prefill checkpoints may still be `KVCacheSimple` unquantized.

**Idea:** optionally re-quantize attention state to `int8` (or even `int4`) at capture time, independently of the live cache. Halves or quarters the RAM footprint of the largest snapshots. Leaves SSM state untouched.

**Concern:** this is a correctness trade. Re-quantization is lossy; Task 2.2's bitwise-logit-equality gate would need to be relaxed to a bounded-drift gate for quantized-at-capture snapshots. Not worth pursuing until we have a real RAM-pressure trace showing that the budget is binding.

### Idea 4.3 — Warm-start pre-hydration of the top-K stable prefixes

Pairs with 4.1. Even with the SSD tier, the first lookup pays a ~100 ms disk read. On launch, pre-hydrate the top-K most-recently-accessed `.system` snapshots into RAM speculatively, before the first request arrives. K ≈ 1–3 on typical workloads because stable-prefix snapshots are rare and tall.

**Implementation:** after warm-start header scan, kick off a background `Task` that loads the K highest-ranked `.system` snapshots by last-access time. Gate behind a user preference; costs ~200–600 MiB of RAM at launch that might not be needed.

### Idea 4.4 — VLM image feature cache

oMLX ships `vision_feature_cache.py` — a content-addressed cache of encoded image embeddings keyed by image hash. Orthogonal to the text prefix cache but high-value for any VLM workload (Qwen3.5 VLM is the current target): image encoding is a fixed-cost prefill step that dominates first-token latency when the same image is referenced across turns.

**Shape.** Hash-by-content (`SHA-256` of raw image bytes) → cached `inputEmbeddings` tensor. Lives outside the text radix tree because its key space is different. Invalidated when the vision encoder weights change (same fingerprint story as 4.1).

**Defer until:** there is a real VLM agent workload. The current `/v1/chat/completions` path is LLM-only for the agent; VLM is wired but untested per `CLAUDE.md`.

### Idea 4.5 — Predictive prefetch of likely tool-loop continuations

During the gap between a tool call returning and the next user turn, the agent's next prompt shape is _mostly_ known: the existing conversation + the tool result. If we speculatively tokenize and prefill that continuation in the background (behind the arbiter, canceled on user input), the next user turn lands on a warm cache with zero wait.

**Concern.** This breaks the `InferenceArbiter` single-in-flight invariant and competes with the actual tool execution for Metal bandwidth. Only worth doing if benchmarks show the tool-result → next-turn gap is a dominant UX issue, and only with a clean cancellation path so the prefetch can't starve the real request.

### Idea 4.6 — Cross-model stable-prefix sharing for fine-tunes

If two models share identical embeddings and identical first N transformer layers (common for LoRA fine-tunes), the first N layers' snapshot state is identical. In theory the stable-prefix snapshot could be shared across partitions.

**Concern.** Tesseract ships one model. This is speculative architecture for a user pattern we do not have. Park it unless multi-model serving becomes a real requirement.

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
    │
    ▼
Phase 4 (draft — SSD tier, warm start, snapshot quantization, VLM feature cache — not scoped)
```

All phases 0–3 are sequential. Phase 1 includes the modified `prepare()` loop (previously deferred to Phase 2) because Mamba state capture during prefill is a prerequisite for any hybrid checkpoint. **Phase 4 is a draft idea list, not a sequenced plan** — each idea is independently scopable after 0–3 ship and a production trace is available.

---

## Test Summary

- Phase 0: 3 manual validation steps for memory behavior.
- Phase 1: 72 new unit tests across snapshot/prepare/detector/radix/manager, 18 integration tests for the end-to-end cache path, migration of the existing normalization-focused `HTTPPrefixCacheSpikeTests`, 1 model-backed E2E scenario.
- Phase 2: 28 new unit tests across speculative admission, correctness gating, Marconi utility scoring, and adaptive `alpha` tuning, plus 1 manual benchmark validation pass.
- Phase 3: 8 unit tests for two-pass prefill and prefill-step heuristics, 11 model-backed benchmark scenarios, and 1 manual validation pass.
- Phase 4: **draft only — no test scope yet.** Each idea carries a rough test topology but is not sequenced. Test counts will be assigned at scoping time, after a production trace justifies the specific enhancement.

---

## Risk Register

| Risk                                                              | Likelihood | Impact       | Mitigation                                                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------- | ---------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mid-prefill snapshot restore produces wrong logits                | Medium     | **Critical** | Task 2.2 tests #1-8: bitwise-match gate on mid-prefill restores (no normalization). Fall back to full prefill on any mismatch.                                                                                                                                                          |
| Normalized leaf hit divergence exceeds bound                      | Low        | Medium       | Task 2.2 test #9: bounded divergence < 0.01. If exceeded, disable leaf caching for hybrid models (fall back to stable-prefix-only). Current prototype already accepts this divergence.                                                                                                  |
| Variable chunk sizes hurt prefill throughput                      | Low        | Low          | Small chunks (< prefillStepSize) only at checkpoint boundaries. Vast majority of chunks are full-size.                                                                                                                                                                                  |
| QuantizedKVCache restore with wrong groupSize/bits                | Low        | High         | Parse from metaState, not defaults. Test 1.1#14 validates.                                                                                                                                                                                                                              |
| Stable-prefix detection fails for non-ChatML templates            | Medium     | Medium       | Two-probe technique + verification step (`fullTokens[0..<boundary] == probeA[0..<boundary]`) catches mismatches. Falls back to no stable-prefix checkpoint.                                                                                                                             |
| Snapshot memory exceeds budget                                    | Medium     | Medium       | Budgeting now uses corrected sizing formulas from the local Qwen3.5 cache shapes: a 4K unquantized snapshot is ~202 MiB. Auto-sizing and Marconi utility-scored eviction must use measured `snapshot.memoryBytes`, not rough MB estimates.                                              |
| Adaptive `alpha` tuning picks an unstable value on a short trace  | Low        | Medium       | Start with `alpha = 0`, require a bootstrap window of `5x` the first-eviction request count, tie-break on token hit rate, and allow fallback to `alpha = 0` if the bootstrap trace is too small to tune confidently.                                                                    |
| Upstream mlx-swift-lm changes conflict                            | Medium     | Medium       | Minimize vendor changes. `prepareWithCheckpoints()` is a protocol extension (existing `prepare()` untouched). `GenerateParameters.checkpointAtOffsets`/`checkpointBaseOffset` are new optional fields with defaults. Only `LLMModel`, `Qwen35`, and `TokenIterator.prepare()` modified. |
| Cross-config contamination (wrong kvBits snapshot returned)       | Low        | High         | `CachePartitionKey` isolates trees by `(modelID, kvBits, kvGroupSize)`. Tests 1.5#15-17 validate.                                                                                                                                                                                       |
| TokenIterator contract breaks in future mlx-swift-lm updates      | Low        | Medium       | `capturedSnapshots` property is additive. Protocol extension is non-breaking — all existing conformers keep working via default impl.                                                                                                                                                   |
| Re-tokenization of stored conversation is expensive               | Low        | Low          | One extra tokenization per generation. Current code already does this via `measureHTTPPrefixCacheTokenCount`. Same cost.                                                                                                                                                                |
| Two-probe detection produces wrong boundary for unusual templates | Low        | Medium       | Verification step (probe prefix == fullTokens prefix) catches mismatches. Falls back to no stable-prefix checkpoint.                                                                                                                                                                    |

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
