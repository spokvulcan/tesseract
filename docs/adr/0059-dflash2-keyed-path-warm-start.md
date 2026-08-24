# ADR-0059: DFlash2 on the keyed path — one prefill authority, warm-start handoff

- Status: Accepted
- Date: 2026-08-24
- Relates to: ADR-0057/0058 (DFlash2), ADR-0056 (MTP cold path — its
  amendment is superseded for the DFlash2 arm), ADR-0009 (speculative
  canonical prefill), ADR-0019 (leaf home guarantee)

## Context

ADR-0056's amendment parked speculation whenever the predicted leaf store
mode was not `directLeaf`, and both drafters only engaged on the *cold*
branch of the server completion. On the product's primary workload — coding
agents over HTTP, tools defined, thinking enabled — every turn predicts
`directToolLeaf` or `canonicalUserLeaf` and every turn after the first is
warm, so speculative decoding never ran at all. The two features the server
exists to combine (radix prefix cache, DFlash2 decode) were mutually
exclusive in exactly the traffic that matters.

The root cause was ownership, not physics: both speculative iterators ran
their *own* prompt prefill, so engaging them forfeited the mid-prefill
boundary snapshots (lastUser/lastMessage) the canonical and tool leaf modes
are synthesized from — the 2026-08-18 partition-starvation incident. The
reference stacks (SGLang runs DFlash drafts with its radix cache on) treat
speculation as a decode-time concern layered over whatever cache state
exists; nothing about the draft requires owning the prefill.

DFlash2 specifically needs only two things at decode time: a KV cache
covering the prompt, and the target's layer-5/19/33/47/61 hidden states for
at most the last `dflashContextKeepCount` (2047) prompt rows — captured
with absolute RoPE positions, so rows are position-stable no matter where a
prefix/tail split falls.

## Decision

The app's `PrefillExecutor` stays the single prefill authority; the DFlash2
iterator warm-starts behind it and capture-prefills only the tail it needs.

1. **Vendor.** `DFlash2SpeculativeIterator` gains
   `prefilledPrefixTokens: Int = 0`: `prepare` starts its hidden-state
   capture prefill at that offset instead of 0, preconditioned on the main
   cache's offset matching. At 0 the code path is bitwise-identical to the
   parent (gate: A/B below). The drafter's window is built from tail rows
   only; hidden states are never persisted alongside KV prefixes.
2. **App.** Engagement becomes a keyed-path decision, upstream of the
   restore switch: `DFlash2Support.shouldEngage(hasDrafter:
   textOnlyIdentityKeySpace:kvBits:)`. No leaf-mode gate, no cold-only
   gate. When it engages, the normal keyed pipeline runs unchanged —
   restore, chunked prefill, *all* checkpoint captures — but stops at
   `splitOffset = min(max(executionBaseOffset, lastCaptureOffset,
   fullTokenCount - 1 - dflashContextKeepCount), fullTokenCount - 1)`, and
   the iterator is built with `prefilledPrefixTokens: splitOffset`. Every
   leaf store mode keeps its boundary snapshots; the iterator's own capture
   prefill covers at most the 2047-row window minus whatever the split
   already left it.
3. **KV-quant gate.** `kvBits != nil` refuses engagement: the stage-2
   `commitPipelined` force-casts `KVCacheSimple`, and quantized-KV requests
   key a different partition anyway.
4. **MTP unchanged.** The MTP head still requires one hidden row per prompt
   token from an unchunked vendor prefill, so ADR-0056's cold +
   `directLeaf` gate stands for that arm; it now also yields to DFlash2
   whenever both drafters are loaded.

## Evidence

- **Live end-to-end** (qwen3.8-27b, tools + thinking, temp 0): turn 1 —
  `lookup=hit(branchPoint at 40/328)`, DFlash2 `acceptance=62.3%`,
  `directToolLeaf captured — offset=383`; turn 2 (tool result) —
  `cached_tokens 383/428`, `restoreMs=6.2`, `acceptance=50.6%`,
  `canonicalLeaf captured — offset=490`, total 2.35s. Speculation and the
  prefix cache compounding on the exact traffic the amendment had parked.
- **Bitwise identity** (ledger R54): cold `--dflash2-bench --bench-blocks
  8f` gate, same session, parent binary vs integrated binary — both
  `accepted=125/469`, both `output-identity: MATCH`, bit-stable across
  repeat runs. The banked 147/322 was not reproduced by *either* binary:
  an environmental acceptance re-roll (R44 trajectory class), not a code
  perturbation.
- Unit: `DFlash2SupportTests` (engagement axes), vendor
  `testDFlash2IteratorWarmStartMatchesColdStream` (warm-started stream ==
  cold stream, window rows re-based).

## Consequences

- Speculative decoding now engages on warm keyed turns, tool turns, and
  thinking turns — the primary workload — with the cache intact. The
  cold-only era's "speculation OR cache" trade is gone for DFlash2.
- The drafter's hidden window is rebuilt per request from the tail capture
  prefill (≤2047 rows); on a full cache hit that tail is small and the
  rebuild rides the same forward that primes decode.
- The MTP arm remains cold + `directLeaf` until someone builds a
  boundary-preserving unchunked prefill; with the DFlash2 draft installed
  it is effectively a fallback for non-DFlash2 checkpoints.
