# ADR-0061: DFlash2 reduced to its fast path and reshaped for upstream

- Status: Accepted
- Date: 2026-09-04
- Relates to: ADR-0057 (DFlash2 speculative decoding), ADR-0058 (mma8 +
  pipeline + adaptive width), ADR-0059 (keyed-path warm start),
  `docs/mlx-swift-lm-fork.md` (the carry ledger)

## Context

The DFlash2 series on the `spokvulcan/mlx-swift-lm` fork grew as research:
13 commits, ~5,000 lines, a dozen `DFLASH2_*` environment knobs, an
n-gram-advised host selector, lattice dumps, accept logs, profile timelines,
a fused q/k-norm scan kernel variant, an elementwise verify conv, a verify
stream stride, an adaptive-width bandit, eager escape hatches and a
passthrough mode. Every one of those was measured (experiments ledger
R11–R57) and every one either lost or tied against the path that shipped:
pipelined round, fixed width 8, compiled draft/verify segments, masked GDN
replay, same-input QMM stacking. The research is done; the fork carry is now
the whole DFlash2 series plus the pin and C25, and upstreaming it is the next
filing (`docs/mlx-swift-lm-fork.md`, "closed 2026-09-03").

Upstream also fixed the shape it wants for drafters while the series was in
flight: `MTPDrafterModel` (stateless, target per call), a typed
factory/registry/container, `generate` overloads, `CompiledTrace`, and a
processor discipline in `SpeculativeTokenIterator` that runs logit
processors row by row on a copy.

## Decision

One PR, one commit on top of upstream `e3d4a20`, carrying only the fast
path, in upstream's own shapes:

- **Typed seams instead of state keys.** `DFlash2TargetModel` (layer count,
  embedding, head, `dflash2SupportsCache`, `dflash2Prefill`, `dflash2Verify`)
  and `DFlash2DrafterModel` (block size, mask id, target layer ids and depth,
  context window, `makeState`, `propose`). The `LMOutput.State` key protocol
  (`dflash2CaptureLayerIdsKey` and friends) is gone.
- **Verify computes, the iterator commits.** A verify pass writes attention
  rows at a lazy position (`KVCacheSimple.writeRows`) without moving the
  offset and returns `GatedDeltaCapture`s; the iterator assigns replayed
  recurrent state and commits offsets once the accept count is known. The
  old "roll back after the fact" path (`rollbackSpeculativeHybridCaches`) is
  gone, and with it a latent bug: an all-drained final round left the next,
  unsynced verify's GDN end state in the caches. `finalizeGeneration` now
  replays the last committed round to what the consumer drained.
- **One round shape.** Greedy and sampled rounds share the pipelined
  build; acceptance is lazy (`cumprod` for greedy, rejection sampling
  against the selector's candidate distribution otherwise) and the host
  syncs once per round on `[drafts, accepted, bonus]`. Width is fixed at
  the drafter's block size (8), narrowed only by `maxTokens`.
- **Stateless drafter.** The drafter borrows the target's embedding and
  head per proposal, so one instance serves every session; per-stream
  context caches live in `DFlash2DrafterState`. The context cache keeps
  placeholder rows with positions and validity, resolved after the sync.
- **Losslessness through the upstream processor discipline.** Verify rows
  are processed sequentially on a processor copy with `didSample` per draft,
  so penalties see exactly the history the target's own decoding would.
- **Stacking as a protocol.** `SameInputProjectionStacking` +
  `stackSameInputProjections(in:)` replace `dflash2StackGateUpProjections`;
  the exact-class `plainQuantizedLinear` guard stays (ADR-0058's PARO
  incident).
- **Deleted.** All `DFLASH2_*` and `MLX_OP_CENSUS` knobs, the advised
  selector and n-gram index, lattice dump, accept log, profile timelines,
  the fused q/k-norm kernel variants (`GatedDelta.swift` back to upstream),
  elementwise conv, verify stream stride, adaptive width, eager escape
  hatches, passthrough, parity tests, `FileHandle` writes and static mutable
  counters. `MLX_DYNSLICE_INPLACE` is moot: the KV write is `putAlong`,
  measured flat against the fork's in-place dynamic slice update (A/B
  2026-09-04: 31.8/29.8 vs 30.9/29.8 tok/s, identity MATCH).

The app keeps its policy layer (`DFlash2Support`: detection, 4-bit load,
geometry check, engagement predicates, penalty-stripping iterator factory)
and the bench (`--bench-blocks 8`; the `f` suffix is accepted and ignored).
`MLX_MAX_ACTIVE_TASKS=40` is still set only by the bench runner.

## What the gates caught

The first full build passed identity MATCH and every unit test but
accepted 108/577 instead of the banked 115/532. A per-round trace against
the pre-reshape build diverged at round 1: the prompt-window rows entered
the drafter's context cache as placeholders (the pipelined append shape)
and nothing resolved them, so from round 1 the visibility mask hid the
whole prompt and the drafter proposed from the last block alone. Identity
cannot see this — the target verifies every proposal — only acceptance
can, which is why the bit-identical acceptance count is a gate and not a
nice-to-have. `prepare` now resolves the window rows right after the
first round appends them, and the mock-driven iterator test asserts the
committed-row count seen by each proposal.

## Consequences

- The upstream PR is reviewable: five new files, a contained Qwen3.5 diff,
  no environment-driven behaviour, tests through the two protocols.
- Anything that wants back in (adaptive width, an advisor) re-enters as a
  measured change against this baseline, not as a knob.
- Gates for this reshape: identity MATCH and acceptance 115/532 bit-identical
  on the docs prompt, per-round drafts identical to the pre-reshape build,
  tok/s ABBA against it (new 32.8 / 30.1 vs old 28.8 / 27.8 tok/s medians (per-run 29.5–32.8 vs 26.9–28.8), AR flat at ~21), the vendor unit suite (740/740), and
  the full CI replica (swift-format 603, verify-docs, xcodebuild tests).
