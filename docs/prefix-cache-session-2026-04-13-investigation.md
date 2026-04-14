# Prefix Cache Investigation — 2026-04-13 Session Changes

**Status:** RESEARCH — every change here is provisional. Each section ends with a
**Keep / Discard / Evaluate** verdict so we can decide later which changes to
ship and which to revert.

**Context.** This session started as an analysis of the
[Marconi-style hybrid prefix cache](marconi-hybrid-prefix-cache-implementation-plan.md)
in production, then evolved into a series of incremental fixes for cache
pathologies observed under real OpenCode workloads on Qwen3.5-9B-paro:

1. Late-turn cold prefills (~5 minutes) when the main agent returned after a
   long subagent run.
2. Memory pressure causing 14 GB of swap on a 48 GB Mac.
3. Cross-new-user-turn divergence where the post-generation leaf was unreachable.
4. Long-conversation pathology where every turn's freshly stored leaf was
   immediately evicted in its own admission cycle.

The fixes interact, so order matters when reading. Changes are listed roughly
chronologically.

---

## Quick reference table

| # | Change | Verdict | Reverts cleanly? | Critical |
|---|---|---|---|---|
| 1 | `.system`-only eviction protection (Option A) | **Keep** (as modified) | Yes | Yes |
| 2 | AlphaTuner `bootstrapMultiplier 5 → 1` + min/max bounds | **Evaluate** | Yes | Medium |
| 3 | Session-scoped partitioning (`sessionAffinity` in `CachePartitionKey`) | **Keep** | Yes | Yes |
| 4 | Memory headroom `4 GiB → 20 GiB` | **Evaluate** | Yes | Yes for 9B |
| 5 | Stripped leaf capture (think-stripping fix) | **Keep** | Yes | Important |
| 6 | `.lastMessageBoundary` checkpoint type split | **Keep** | Risky (vendor file) | **The actual fix** |
| 7 | `capturedThenEvicted` admission diagnostic | **Keep** | Yes | Low (observability) |

**Critical takeaway**: Change #6 is the load-bearing fix for the long-conversation
pathology. Changes #1, #3, #5 are also keepers but each has a narrower job. #2 and
#4 are tunables that should be re-evaluated against more workloads.

---

## 1. `.system`-only eviction protection (Option A)

**Date in session:** Early — first fix attempted.

**Problem.**
In the original v9 plan, `TokenRadixTree.eligibleEvictionNodes()` only excluded
`childCount > 1` nodes (the multi-child rule from Marconi). In single-conversation
workloads, the stable-prefix node has `childCount == 1` (one assistant chain), so
it was eligible for eviction. Under pure-recency scoring (alpha=0 during bootstrap),
the stable-prefix snapshot was the LRU victim every drain pass — turning new-user-turn
requests into multi-minute cold prefills because the stable prefix was gone.

**What was changed.**
`tesseract/Features/Server/TokenRadixTree.swift` `collectEligible(node:into:)`
gained a third condition:

```swift
if let snapshot = node.snapshot,
   node.childCount <= 1,
   snapshot.checkpointType != .system    // ← new
{
    result.append(node)
}
```

Hard budget invariant is preserved by `PrefixCacheManager.findEvictionCandidate`'s
fallback path (still drops oldest from `allSnapshotNodes()` when eligible is empty).

**Why it was wrong (initially) and how change #6 fixed it.**
At the time, both **stable-prefix** (`system + tools`) and **last-message-boundary**
checkpoints were stored as `.system`. Excluding `.system` therefore protected
*both*. In long conversations, every turn captured a new last-message-boundary, so
the protected set grew unboundedly and the only thing eligible to evict was
freshly-stored leaves → admission storms. Change #6 (the type split) is what makes
this filter work correctly: only the stable prefix is `.system` now, boundaries are
`.lastMessageBoundary` (not protected).

**Files.**
- `tesseract/Features/Server/TokenRadixTree.swift` (the filter + comment)

**Tests.**
- `tesseractTests/TokenRadixTreeTests.swift::eligibleEvictionNodesExcludeSystemSnapshots`
- `tesseractTests/TokenRadixTreeTests.swift::eligibleEvictionNodesAllowsBranchPointAndLeaf`
- `tesseractTests/PrefixCacheManagerTests.swift::systemSnapshotProtectedFromUtilityEviction`

**Verdict: Keep (as modified by #6).**
The protection itself is correct and important — without it the stable prefix
gets evicted under recency. The bug was conflating two checkpoint types under
one name; #6 fixes that.

**To revert:** restore the two-condition filter
(`snapshot != nil && childCount <= 1`). Stable prefix would become evictable
again and we'd see new-user-turn cold prefills return.

---

## 2. AlphaTuner tuned for single-user (`bootstrapMultiplier 5 → 1`)

**Date in session:** Right after Option A.

**Problem.**
Marconi's adaptive `alpha` tuner needs a "bootstrap window" of post-first-eviction
requests to grid-search the alpha parameter. Original
`bootstrapMultiplier = 5` (paper-aligned) meant a window of 55 requests for typical
9B sessions where first eviction fires after ~11 requests. **No realistic Tesseract
session reaches 55 post-bootstrap requests in one process lifetime**, so the tuner
*never completed* and `alpha` stayed at its `0.0` default. FLOP-aware utility
scoring (the entire point of Phase 2) was inert.

**What was changed.**
`tesseract/Features/Server/AlphaTuner.swift`:
- `bootstrapMultiplier = 1` (was 5).
- New `maximumBootstrapWindow = 60` cap (prevents pathological cases where first
  eviction fires after hundreds of requests).
- `minimumBootstrapWindow = 10` (was 5) — small floor of distinct hit/miss
  outcomes for the grid search to score against.
- `notifyFirstEviction` clamps to `[min, max]`.

**Files.**
- `tesseract/Features/Server/AlphaTuner.swift` (constants + clamp logic).
- `tesseractTests/AlphaTunerTests.swift::bootstrapWindowUsesMultiplierTimesFirstEvictionCount` (renamed from `bootstrapWindowUsesFiveTimesFirstEvictionCount`, generalised to use the constant).
- `tesseractTests/AlphaTunerTests.swift::bootstrapTargetHonorsMaximumWindow` (new).

**What we observed in logs after the change.**
A 9B session with `requestsBeforeFirstEviction=11` produced
`AlphaTuner: bootstrap started — windowSize=11`, then completed tuning in ~11
post-bootstrap requests (~6 minutes). Tuned `alpha = 0.0` because the bootstrap
window had no eviction conflicts to give the FLOP term any signal. So tuning
worked but didn't affect behaviour for that workload.

**Verdict: Evaluate.**
The change is harmless for normal workloads — it lets the tuner finish, which it
should. But the question of whether `alpha = 0.0` is the right answer for our
workloads is still open. If we never see non-zero tuned alphas in practice, the
whole tuner is effectively dead weight and we should consider hardcoding alpha
or removing it entirely.

**To revert:** set `bootstrapMultiplier = 5`, drop the cap, restore
`minimumBootstrapWindow = 5`. Tuning becomes unreachable on typical sessions.

---

## 3. Session-scoped partitioning (main agent vs subagents)

**Date in session:** After noticing that long subagent runs evicted main-agent
snapshots in the production log files.

**Problem.**
The `CachePartitionKey` was `(modelID, kvBits, kvGroupSize)` — meaning every
client of the HTTP server shared the same partition. When OpenCode's main agent
delegated to a subagent, the subagent ran for ~10 minutes and its many fresh
snapshots evicted the idle main-agent snapshots under recency. When the main
agent resumed, it cold-prefilled 91K tokens (~5 minutes).

The disk request files showed the main agent and subagent had completely
different `x-session-affinity` HTTP headers (different `ses_*` UUIDs), so we
already had the discriminator — it just wasn't flowing into the cache.

**What was changed.**
1. `CachePartitionKey` gained `let sessionAffinity: String?` as an optional
   fourth field. `Comparable` and `Hashable` conformances updated.
2. `nonisolated init(...)` to keep the explicit initializer callable from non-MainActor
   closures (a Swift 6 strict-concurrency wrinkle — the synthesized
   memberwise init was MainActor-isolated by inference once we added members).
3. `CompletionHandler.startGeneration` now passes `sessionAffinity` (already
   extracted from the `x-session-affinity` header for the session replay
   store and HTTP logger) through to `engine.generateServerTextCompletion`.
4. `AgentEngine.generateServerTextCompletion` and
   `LLMActor.generateServerTextCompletion` gained `sessionAffinity: String?`
   parameters.
5. `LLMActor.makeHTTPPrefixCacheGeneration` builds the partition key with the
   session affinity included.
6. `PrefixCacheManager.findEvictionCandidate` was rewritten as a 4-tier strategy:
   - **Preferred utility** (writing partition's eligible nodes, lowest utility).
   - **Global utility** (other partitions' eligible nodes, lowest utility).
   - **Preferred fallback** (oldest snapshot in writing partition, including
     multi-child / `.system`).
   - **Global fallback** (oldest snapshot anywhere — preserves hard-budget invariant).
7. `evictToFitBudget(preferredPartitionKey:)` resolves the partition to a tree
   and passes it as `preferredTree` to the candidate finder.
8. `storeSnapshots` and `storeLeaf` pass `partitionKey` to
   `evictToFitBudget(preferredPartitionKey:)` so each store drains its own
   partition first before touching others.
9. Benchmark runners (`PrefixCacheE2ERunner`, `PrefillStepBenchmarkRunner`)
   pass `sessionAffinity: nil` to preserve their single-partition semantics.

**Files.**
- `tesseract/Features/Server/PrefixCacheManager.swift` (key, eviction strategy)
- `tesseract/Features/Server/CompletionHandler.swift` (plumbing)
- `tesseract/Features/Agent/AgentEngine.swift` (plumbing)
- `tesseract/Features/Agent/LLMActor.swift` (plumbing + partition key build)
- `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift` (compile fix)
- `tesseract/Features/Agent/Benchmark/PrefillStepBenchmarkRunner.swift` (compile fix)

**Tests.**
- `tesseractTests/PrefixCacheManagerTests.swift::differentSessionsIsolated`
- `tesseractTests/PrefixCacheManagerTests.swift::nilSessionAffinityShared`
- `tesseractTests/PrefixCacheManagerTests.swift::idleMainAgentSurvivesSubagentChurn`
   (acceptance test for the original pathology)
- `tesseractTests/PrefixCacheManagerTests.swift::preferredPartitionSpillsToGlobalWhenExhausted`

**Verdict: Keep.**
This change addresses a real, observed production failure mode. Without it, any
multi-session workload (main agent + subagents) is at risk of cross-eviction.

**To revert:** drop the `sessionAffinity` field, drop the `preferred*` argument
chain, restore the original 2-tier `findEvictionCandidate`. Subagent churn
would once again evict idle main-agent snapshots.

---

## 4. Memory headroom `4 GiB → 20 GiB`

**Date in session:** After observing 14 GB of swap on a 48 GB Mac running the 9B
model with peak MLX usage at ~36 GB.

**Problem.**
`LLMActor.autoSizedPrefixCacheMemoryBudgetBytes` formula:
`(physicalMemory - modelWeightBytes - headroom) / 2`. With `headroom = 4 GiB` on a
48 GB Mac with the 4.8 GB model, the budget was ~19.7 GB. Peak active MLX
(model + cache + activations + transient prefill buffers) hit ~36 GB, leaving
only ~12 GB OS headroom and triggering 14 GB of swap.

**What was changed.**
`tesseract/Features/Agent/LLMActor.swift` `Defaults.prefixCacheHeadroomBytes`
changed from `4 * 1024^3` to `20 * 1024^3`. Doc comment updated to explain the
math: 12–14 GiB peak active MLX above the cache budget on Qwen3.5 9B, +6–8 GiB
for macOS + WindowServer + other apps, +slack for typical dev environments.

New formula on a 48 GB Mac with 4.8 GB model: `(48 - 4.8 - 20) / 2 ≈ 11.6 GB`
budget. Peak MLX should land around 25 GB instead of 36 GB.

**Files.**
- `tesseract/Features/Agent/LLMActor.swift` (`Defaults.prefixCacheHeadroomBytes` + doc comment).
- `tesseractTests/EvictionPolicyTests.swift` (4 budget tests updated with new expected values).

**Trade-off.**
- Pro: No swap on 9B sessions on 48 GB Macs.
- Pro: Holds for smaller machines too — clamps cleanly to 0 on 16 GB Macs that
  can't even fit a thinking model.
- Con: Smaller cache budget overall. Fewer simultaneous snapshots. Lower
  cross-conversation hit rates if user switches between many distinct
  conversations.
- Con: Hardcoded for the 9B working set. The 4B model only needs ~6–8 GiB
  headroom; with 20 GiB headroom, 4B sessions get a smaller budget than they
  could otherwise use.

**Verdict: Evaluate.**
The current value works for the 9B-on-48GB scenario but is probably too
conservative for 4B or for 64 GB Macs. A better fix would be to make the
headroom proportional to model size (e.g., `model + 4 * model + 4 GiB`) or to
read the working-set requirement from the model config. For now: keep at 20 GiB
because the alternative (swap) is much worse, but treat as a placeholder for a
proper solution.

**To revert:** set back to `4 GiB`. 9B sessions on 48 GB Macs would swap again.

---

## 5. Stripped leaf capture (Qwen3.5 think-stripping fix)

**Date in session:** After session scoping and headroom were in place but
new-user-turn requests were still cold-prefilling.

**Problem.**
Qwen3.5's chat template strips `<think>...</think>` blocks from any assistant
message that is not the *latest assistant* (determined by `last_query_index`
reverse-walk). Concretely:

- Turn N stores its leaf via `measureStoredTokenSequence` after generation. The
  re-tokenization sees the just-generated assistant as the LAST message, so the
  template keeps its `<think>` block. The leaf is keyed by tokens that include
  the think content.
- Turn N+1 arrives with a new user message. The template now sees that
  assistant as not-latest and strips its think block. Turn N+1's tokens diverge
  from Turn N's stored tokens at the start of the assistant's content, so the
  unstripped leaf is unreachable.

**What was changed.**
`tesseract/Features/Agent/LLMActor.swift` gained two new private helpers:

1. **`computeStrippedStoredTokens`** — two-probe technique:
   ```
   probeA = baseHistory + .user("Aqkz_strip_probe")
   probeB = baseHistory + .user("Zqkz_strip_probe")
   ```
   Tokenize both. Both probes have a dummy user as the latest message, so the
   template strips the just-generated assistant's think block in both. Find the
   first divergence between probeA and probeB tokens (where the dummy user
   content differs), then subtract the encoded length of `"<|im_start|>user\n"`
   to land at the end of the stripped assistant. Defense-in-depth: verify the
   user-opener tokens actually precede the divergence point.

2. **`captureStrippedLeaf`** — orchestrator that:
   - Runs the probes.
   - Restores a fresh `[KVCache]` from the `lastMessageBoundary` snapshot in
     `iterator.capturedSnapshots` (captured mid-prefill during this turn).
   - Prefills the stripped assistant residual on top via
     `context.model.prepareWithCheckpoints(checkpoints: [:])` — same chunked
     prefill loop the production code uses, with the leftover-token drain
     pattern from `HybridCacheCorrectnessRunner.prefill`.
   - Captures a `.leaf` snapshot from the resulting cache.
   - Stores it under `storedTokensStripped` via `prefixCache.storeLeaf`.
   - Logs `Stripped leaf captured — offset=X residualTokens=Y prefillMs=Z ...`.
   - Catches all errors, logs them via `logSkip(...)`, and never blocks the
     normal request path.

3. `HTTPPrefixCacheGeneration` gained two new fields:
   - `lastMessageBoundaryOffset: Int?` — plumbed out of `container.perform` so the
     post-generation task can find the matching snapshot in `capturedSnapshots`.
   - `prefillStepSize: Int` — plumbed from `GenerateParameters` so the residual
     prefill uses the same chunk size as the main path.

4. The post-generation `Task` in `makeHTTPPrefixCacheGeneration` calls
   `captureStrippedLeaf` immediately after the existing unstripped-leaf store
   block. Both leaves are captured per turn (the unstripped one is the canonical
   path for tool-loop continuations where the assistant stays latest; the
   stripped one is for cross-new-user-turn lookups).

5. Skip conditions (logged via `logSkip(stage: "strippedLeafStore", ...)`):
   - `!promptStartsThinking` — silent (non-thinking model, nothing to strip).
   - `thinkingContent < 50 chars` — silent (prefill cost > expected savings).
   - `lastMessageBoundaryOffset == nil` — logged.
   - No matching snapshot in `capturedSnapshots` — silent (prior turn handled it).
   - Probe divergence failed / opener mismatch — logged.
   - `strippedLen <= boundaryOffset` (no residual) — logged.
   - `strippedLen >= storedTokens.count` (template didn't actually strip
     anything, e.g. non-Qwen3.5) — logged warning.
   - Residual prefill threw — logged warning.
   - Snapshot capture returned nil — logged.

**Files.**
- `tesseract/Features/Agent/LLMActor.swift` (helpers + post-gen wiring + struct fields)

**Tests.**
- `tesseractTests/PrefixCacheIntegrationTests.swift::strippedLeafHitOnNewUserTurn`
   (acceptance test: store both leaves, verify new-user-turn lookup hits the
   stripped one and tool-loop continuation hits the unstripped one)
- `tesseractTests/PrefixCacheIntegrationTests.swift::unstrippedLeafPreferredWhenDeeperOnSamePath`
   (regression guard: when both leaves exist on the same linear path, the
   deeper one is preferred)

**Cost analysis.**
- Two extra `processor.prepare` calls per turn for the probes (~5–15 ms total).
- One extra forward-pass prefill over the residual (`assistant_opener +
  response + <|im_end|>` tokens). Typically 100–2000 tokens. On Qwen3.5-9B at
  ~350 tok/s prefill: 0.3–5 seconds wall time per turn.
- Memory: one additional `.leaf` snapshot per turn (~200 MiB to 2.5 GiB depending
  on conversation length). Bounded by the eviction budget.
- Amortization: each successful cross-new-user-turn hit avoids a full cold
  prefill. Break-even after one cross-turn reuse; every additional reuse is
  pure win. For typical thinking-model agentic sessions, this is net positive.

**Verdict: Keep.**
Without this fix, cross-new-user-turn requests on thinking models cold-prefill
the entire conversation (5+ minutes on 9B at 80K tokens). The stripped leaf
gives them a deep cache hit instead. The capture overhead is small relative to
the savings.

**Important caveat.** This change *alone* didn't fix the user's failing 12:31
session — see #6 below. The stripped leaf was being captured correctly but
*immediately evicted* in the same admission cycle because of the type-protection
issue. #5 + #6 together are what's needed.

**To revert:** delete the two helper functions, drop the post-generation
invocation, drop the struct fields, drop the tests. Cross-new-user-turn lookups
go back to falling through to the (shallower) `.lastMessageBoundary` checkpoint.

---

## 6. `.lastMessageBoundary` checkpoint type split (THE actual fix)

**Date in session:** Last fix in the session, after analyzing the 12:31 failure.

**Problem.**
The 12:31:00 lookup in the production log showed the deep stripped leaf at
offset 74981 was unreachable from the next request, even though the next
request's first 74981 tokens should have matched. The smoking gun was at
12:31:05–12:31:06 in the same logs:

```
event=capture offset=75105 type=leaf source=leaf
event=eviction utility offset=75105 type=leaf utility=1.0   ← evicted in same cycle
event=capture offset=75073 type=leaf source=strippedLeaf
event=eviction utility offset=75073 type=leaf utility=1.0   ← also evicted in same cycle
```

`utility=1.0` doesn't mean "high quality" — `EvictionPolicy.normalize([x])`
returns `[1.0]` for a single candidate. So `utility=1.0` means **"this was the
only eligible candidate."** The fresh leaf was forced to be its own victim
because no other eligible nodes existed.

**Why no other eligible nodes existed.**
Both **stable-prefix** and **last-message-boundary** checkpoints were typed
`.system`. After Option A (#1) protected `.system` from utility scoring, the
filter excluded both. In a long conversation, every turn captured a new
`.system` boundary, so the protected set grew to dozens of snapshots. The
freshly-stored leaf was the only thing eligible to evict → it died on
admission, every turn.

**What was changed.**
1. `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` — added
   a fourth case to `CheckpointType`:
   ```swift
   public enum CheckpointType: Comparable, Sendable {
       case system               // stable-prefix reuse (system + tools)
       case lastMessageBoundary  // end-of-last-message reuse across turns
       case leaf                 // standard conversation-prefix reuse
       case branchPoint          // Phase 2: speculative Marconi checkpoint
   }
   ```

2. `tesseract/Features/Server/PrefixCacheManager.swift` `planCheckpoints` — emit
   `.lastMessageBoundary` for the boundary checkpoint instead of `.system`.
   The `alreadyStored(...)` deduplication looks for the new type when checking
   if the boundary is already in the tree.

3. `tesseract/Features/Server/TokenRadixTree.swift` `collectEligible` — the
   filter (`!= .system`) is unchanged, but now `.lastMessageBoundary`
   automatically passes through it and becomes evictable.
   Doc comment updated to explain: *"Only `.system` (stable prefix) snapshots
   are type-protected — they sit linearly on the path from root in
   single-conversation usage and represent the cross-conversation hot prefix
   that the entire tree is built on. `.lastMessageBoundary` snapshots are
   explicitly NOT protected because long conversations accumulate one per turn;
   protecting all of them would fill the budget with stale boundaries from old
   turns and cause the freshly-stored leaf for the current turn to be the only
   eligible eviction candidate, killing it on admission."*

4. `tesseract/Features/Server/PrefixCacheDiagnostics.swift::checkpointType(_:)`
   — added the `.lastMessageBoundary` case to the renderer (`switch must be
   exhaustive` Swift 6 error).

**Files.**
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` (vendor
  enum — risky to modify)
- `tesseract/Features/Server/PrefixCacheManager.swift` (planCheckpoints)
- `tesseract/Features/Server/TokenRadixTree.swift` (doc comment, no logic change)
- `tesseract/Features/Server/PrefixCacheDiagnostics.swift` (renderer case)

**Tests.**
- `tesseractTests/PrefixCacheManagerTests.swift::planCheckpointsIncludesLastMessageBoundary`
  (updated assertion: `boundary.type == .lastMessageBoundary`)
- `tesseractTests/PrefixCacheManagerTests.swift::planCheckpointsSkipsExistingLastMessageBoundary`
  (updated to pre-store a `.lastMessageBoundary` instead of `.system`)
- `tesseractTests/PrefixCacheManagerTests.swift::lastMessageBoundaryEvictableButStablePrefixProtected`
  (new — regression test: stable prefix + boundary on different paths under
  tight budget; the boundary is the eviction victim, the stable prefix survives)
- `tesseractTests/PrefixCacheManagerTests.swift::olderLastMessageBoundaryEvictedBeforeFreshOne`
  (new — two boundaries from different turns under tight budget; the older one
  ages out via LRU, the fresh one survives)

**Verdict: Keep.**
This is THE fix for the long-conversation pathology. Without it, every fresh
leaf in a long conversation gets immediately evicted, which means the entire
post-generation leaf store is a no-op past ~5–10 turns. With it, old boundaries
age out and fresh leaves survive.

**Risk.**
- The change touches a vendored file
  (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift`).
  Future upstream merges will conflict on this enum. The conflict is trivial
  (one new case) but needs to be re-applied on every vendor sync.
- We're trusting that boundaries can age out without breaking anything. In
  theory, a request that needs a *very old* boundary (e.g., looking up a
  conversation prefix that hasn't been touched in many turns) could miss what
  it would have hit before. In practice, the freshest boundary is the only one
  that matters for tool-loop continuation, and the stable-prefix snapshot
  covers cross-conversation reuse — so the loss is tiny.

**To revert:** drop the new enum case, switch boundary back to `.system` in
`planCheckpoints`, drop the new tests, drop the renderer case. The pathology
returns: long conversations will see fresh leaves immediately evicted.

---

## 7. `capturedThenEvicted` admission diagnostic

**Date in session:** Last change in the session, alongside #6. Pure observability.

**Problem.**
The "fresh leaf evicted in its own admission cycle" bug was hiding in the logs.
We could see it in retrospect by reading the `event=eviction` lines and
comparing offsets, but there was no explicit log line saying "the snapshot you
just stored was just evicted." When debugging future regressions, this would be
useful to spot immediately.

**What was changed.**
`tesseract/Features/Agent/LLMActor.swift` — both leaf store sites now check
whether the just-stored snapshot's offset appears in the `evictions` list
returned by `storeLeaf`. If yes, they log:

```
event=skip stage=leafAdmission|strippedLeafAdmission reason=capturedThenEvicted
  offset=X bytes=Y budgetBytes=B snapshotBytesAfter=A
```

The `budgetBytes` and `snapshotBytesAfter` are read inside the same
`MainActor.run` block as the `storeLeaf` call (single hop). For the unstripped
leaf, the diagnostic also gates `leafStoreForTuner` — if the leaf was evicted,
we don't tell the alpha tuner about it (otherwise the tuner replays a leaf
that doesn't actually exist in the production cache).

**Files.**
- `tesseract/Features/Agent/LLMActor.swift` (both leaf store call sites + the
  helper inside `captureStrippedLeaf`)

**Verdict: Keep.**
Pure observability. Cheap to maintain. If the fix in #6 starts regressing or
the budget gets too tight again, this log line will tell us immediately.

**To revert:** delete the `unstrippedAdmissionEvicted` and `admissionEvicted`
checks. We lose the early-warning signal for admission failures.

---

## What we explicitly did NOT do (and why)

### Drop the unstripped leaf entirely

I almost did this in the session — the unstripped leaf has a known alignment
issue with trailing gen-prompt tokens that probably makes it unreachable in
many cases, AND the stripped leaf covers the new-user-turn case correctly.
But the user pointed out:

1. The unstripped leaf is still needed for tool-loop continuations where the
   assistant remains the latest message in the next request's history. The
   template doesn't strip it in that case, so the stripped path's tokens
   diverge from the actual next-request tokens.
2. Tests in `PrefixCacheIntegrationTests.swift` (`strippedLeafHitOnNewUserTurn`
   and `unstrippedLeafPreferredWhenDeeperOnSamePath`) explicitly verify both
   leaves coexist and the deeper one is preferred when reachable.
3. Even if we dropped it, the user's specific 12:31 failure wouldn't be fixed
   because the issue was eviction, not leaf design.

So both leaves still coexist. Future research could investigate fixing the
unstripped leaf's alignment (subtract `genPromptTokens.count` from
`storedTokens.count` and trim the cache) but that's a separate task with its
own risk profile.

### Hard-pin just-stored leaves (admission control)

The "freshly-stored leaf as its own victim" pattern could be fixed more directly
by passing the just-stored node identifier into `evictToFitBudget` and
excluding it from the eligible set. I considered this but didn't implement it
because:

1. Fix #6 (the type split) addresses the root cause — old boundaries are now
   evictable, so there are always older candidates to pick before the fresh
   leaf.
2. Hard-pinning the new leaf would force fallback evictions of `.system`
   stable-prefix snapshots in extreme cases, which is worse.
3. The diagnostic in #7 will tell us if we still need this.

Re-evaluate if `event=skip stage=*Admission reason=capturedThenEvicted` lines
appear in production logs after this session.

### Fix the unstripped leaf's `storedTokens.count` offset

The unstripped leaf is stored at `storedTokens.count` where
`storedTokens` comes from `measureStoredTokenSequence` →
`processor.prepare(input: UserInput(chat:, tools:))`. The `processor.prepare`
default is `add_generation_prompt = true`, which appends the gen prompt to
the tokenized output. So `storedTokens` ends with `[..., assistant_message,
<|im_start|>assistant\n<think>\n]` and the leaf's offset is at the END of that
trailing gen prompt — a position that doesn't naturally appear in any future
request's tokens. This is probably why the unstripped leaf rarely hits in
production.

Fixing this requires either subtracting `genPromptTokens.count` from
`storedTokens.count` before storing the leaf, or finding a way to call
`processor.prepare` without `add_generation_prompt = true`. The
`MLXLMCommon.Tokenizer` protocol doesn't expose the latter. The former is a
1-line change but needs careful test coverage to verify cache state alignment.

Out of scope for this session. The cross-turn reuse path now goes through the
`.lastMessageBoundary` checkpoint and the stripped leaf, so the unstripped
leaf's brokenness is masked. Worth fixing as a follow-up if/when we want
maximum tool-loop optimization.

---

## How to verify these changes in the next session

### Run the test suites

All eleven prefix-cache test suites should pass:

```bash
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests/HybridCacheSnapshotTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/EvictionPolicyTests \
  -only-testing:tesseractTests/AlphaTunerTests \
  -only-testing:tesseractTests/PrefixCacheDiagnosticsTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/StablePrefixDetectorNonDeterminismTests \
  -only-testing:tesseractTests/JinjaNonDeterminismReproTests
```

Last verified passing on this date with all changes applied.

### Build Release

```bash
xcodebuild build -project tesseract.xcodeproj -scheme tesseract -configuration Release -destination 'platform=macOS' -skipPackagePluginValidation
```

Last verified `BUILD SUCCEEDED` on this date.

### Run a real session

```bash
scripts/dev.sh dev-release
```

Reproduce the failing 9B-paro flow via OpenCode:
1. Long thinking-model conversation with multiple tool calls (tool loop).
2. Generate a final response (`finishReason=stop`).
3. Type a new user message (e.g. "nice").
4. Watch `scripts/dev.sh log` during the next request.

**Expected log signals (good):**
- `event=capture ... checkpointType=lastMessageBoundary` on every turn
  (was `system` before #6).
- `Stripped leaf captured — offset=X residualTokens=Y prefillMs=Z ...` on
  thinking-model turns with non-trivial reasoning.
- `event=eviction strategy=utility checkpointType=lastMessageBoundary` as
  the conversation grows past the budget — old boundaries aging out.
- On the new-user-turn request: deep `cachedTokens=...` close to the
  previous turn's stripped leaf offset, sub-second `prefillMs`.

**Expected log signals (bad — flag for follow-up):**
- `event=skip stage=*Admission reason=capturedThenEvicted` — the fresh leaf
  was still evicted on admission. Probably means the budget is too tight
  even after the type split. Either lower `prefixCacheHeadroomBytes` further
  (more budget) or implement the hard-pin admission control.
- 60+ second prefill on a new-user-turn request — same conclusion.
- `event=eviction strategy=fallback checkpointType=system` repeatedly — the
  stable prefix is being fallback-evicted. Means the budget is so tight
  that even type-protected snapshots are being dropped. The 20 GiB headroom
  is too aggressive for this workload.

### Read the on-disk request files

```bash
ls /Users/owl/Library/Containers/app.tesseract.agent/Data/tmp/tesseract-debug/http-completions/
```

Each file starts with `// session=ses_*` and contains the OpenAI-format
JSON body. Useful for offline diagnosis — see the analysis path used in
this session: load files via Python `json.load` (after skipping the first
comment line), inspect message arrays, compare across requests to find
divergence points.

---

## Summary of files touched

### Production code

- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` — new
  `.lastMessageBoundary` enum case (vendor file, will conflict on upstream
  merges).
- `tesseract/Features/Server/PrefixCacheManager.swift` — `CachePartitionKey`
  + `sessionAffinity`, 4-tier eviction, `planCheckpoints` emits new type.
- `tesseract/Features/Server/TokenRadixTree.swift` — `.system`-only filter in
  `collectEligible` (and updated docs).
- `tesseract/Features/Server/PrefixCacheDiagnostics.swift` — renderer case for
  the new type.
- `tesseract/Features/Server/CompletionHandler.swift` — `sessionAffinity` plumbing.
- `tesseract/Features/Server/AlphaTuner.swift` — `bootstrapMultiplier` +
  min/max bounds.
- `tesseract/Features/Agent/AgentEngine.swift` — `sessionAffinity` plumbing.
- `tesseract/Features/Agent/LLMActor.swift` — `prefixCacheHeadroomBytes`,
  `HTTPPrefixCacheGeneration` extra fields, `computeStrippedStoredTokens`,
  `captureStrippedLeaf`, `capturedThenEvicted` admission diagnostic at both
  leaf store sites.
- `tesseract/Features/Agent/Benchmark/PrefixCacheE2ERunner.swift` — compile fix
  (`sessionAffinity: nil`).
- `tesseract/Features/Agent/Benchmark/PrefillStepBenchmarkRunner.swift` —
  compile fix (`sessionAffinity: nil`).

### Tests

- `tesseractTests/PrefixCacheManagerTests.swift` — many updates and new tests
  for session scoping, type split, admission. Notable new tests:
  `differentSessionsIsolated`, `idleMainAgentSurvivesSubagentChurn`,
  `preferredPartitionSpillsToGlobalWhenExhausted`,
  `lastMessageBoundaryEvictableButStablePrefixProtected`,
  `olderLastMessageBoundaryEvictedBeforeFreshOne`,
  `systemSnapshotProtectedFromUtilityEviction`.
- `tesseractTests/PrefixCacheIntegrationTests.swift` —
  `strippedLeafHitOnNewUserTurn`, `unstrippedLeafPreferredWhenDeeperOnSamePath`.
- `tesseractTests/TokenRadixTreeTests.swift` —
  `eligibleEvictionNodesExcludeSystemSnapshots`,
  `eligibleEvictionNodesAllowsBranchPointAndLeaf`. `makeSnapshot` helper now
  takes an optional `type:` parameter.
- `tesseractTests/EvictionPolicyTests.swift` — budget tests updated for the
  20 GiB headroom; `singleChildEvictionCollapsesNode` switched to
  `.branchPoint` to avoid the type-protection guard.
- `tesseractTests/AlphaTunerTests.swift` —
  `bootstrapWindowUsesMultiplierTimesFirstEvictionCount` (renamed),
  `bootstrapTargetHonorsMaximumWindow` (new),
  `managerDefersBootstrapBoundaryUntilFullRequestCompletes` updated to use
  `.branchPoint` snapshot types so the test isn't accidentally protected by
  the `.system` filter.

### Documentation

- `docs/marconi-hybrid-prefix-cache-implementation-plan.md` — not modified in
  this session, but referenced extensively. Phase 1 limitation #6 is the
  predicted symptom we ran into; this session implemented several of its
  proposed mitigations (session scoping = "separate trees per workload",
  type-aware eligibility = the `.lastMessageBoundary` split).
- `docs/prefix-cache-session-2026-04-13-investigation.md` — this file.

---

## Open questions for the next session

1. **Does `alpha = 0.0` ever flip to non-zero in real workloads?** If never,
   the AlphaTuner is dead code and can be deleted entirely. Run several long
   sessions and check `AlphaTuner: tuned alpha=...` log lines.

2. **Is the 20 GiB headroom too conservative for 4B / 64 GB Macs?** Make it
   model-aware (e.g., `headroom = max(8 GiB, model_bytes * 5)` or similar) and
   measure actual peak vs cache budget on each model.

3. **Should we fix the unstripped leaf's gen-prompt offset bug?** Currently
   masked by the `.lastMessageBoundary` checkpoint. A fix would give us
   ~200–2000 tokens of extra cache hit per tool-loop continuation. Worth
   measuring impact before committing.

4. **Should we strip the unstripped leaf entirely** and rely solely on the
   stripped leaf + boundary checkpoint? Pros: simpler, less memory. Cons: loses
   the tool-loop optimization the unstripped leaf was supposed to provide
   (assuming the alignment bug gets fixed). Decide after measuring real
   tool-loop hit rates.

5. **Long-term: should the cache budget be split per partition?** Currently
   one global budget shared by all sessions. With session scoping (#3), this
   means a single hot session can still monopolise the budget. Per-partition
   budgets would give each session a guaranteed slice. Adds complexity but
   would be the "next level" of session isolation.

6. **Mid-session vendor file modification (#6) needs to be tracked.** The
   `.lastMessageBoundary` enum case is in
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/HybridCacheSnapshot.swift` —
   re-apply on every vendor sync. Consider upstreaming to mlx-swift-lm to avoid
   the conflict.
