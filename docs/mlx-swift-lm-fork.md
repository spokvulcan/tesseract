# mlx-swift-lm fork ledger

The state of `spokvulcan/mlx-swift-lm` — what we carry on top of upstream
`ml-explore/mlx-swift-lm`, what has been contributed back, and how to re-pin.
Governing decision: ADR-0006 (amended) — the vendor is the
frontier-experimentation surface; every vendor change must be general and
upstreamable, shaped as an upstream PR from the start, and the pin re-converges
on vanilla as PRs merge.

**Keep this file current**: update it whenever the pin moves, a PR opens or
merges, or a new fork-only commit lands.

## How the fork is consumed

`Vendor/mlx-swift-lm` is a git submodule of the fork. The pinned commit rides
branch **`pin-upstream-mlx-swift`**, which is rebuilt (not merged) on every
re-pin: base = upstream `main` (or an open PR branch that already contains it)
plus the carried commits cherry-picked on top. The fork's `main` mirrors
upstream `main` exactly and carries nothing.

Old pin branches (`feat/paro-moe-220`, `pin-2026-07-15-upstream-f1573a9`,
`pin-2026-07-23-upstream-eaefe75`, `pin-gemma4-12b-358`, …) are kept so
historical tesseract commits' gitlinks stay reachable — never delete them, and
never force-push a branch an old gitlink points into without checking
reachability. `pin-gemma4-12b-358` is
the parked Gemma 4 12B multimodal stack (audio encoder + encoder-free
`gemma4_unified` processor + suppress_tokens) that tesseract draft PR #359
pins; it rejoins this table's carry list only if that experiment is revived.

## Current pin (2026-09-03)

Base: upstream `main` @ `e3d4a20` — 44 commits past the 2026-08-17 base
(`d7dc03d`). Headline upstream content: **our #471 ParoQuant MoE batch merged
(`e23300b`, byte-identical to the review tip `c39c560`)**; Qwen3.5 norm-shift
detection from the conv1d layout #598 (the fix we carried as `f90e3eb`);
fused GDN input projections #572 (the same four-way qkv|z|b|a stack our
DFlash2 lever built post-load, now done by the model lifecycle with the same
exact-class guard); direct expert reduction #573 and shared fused router
top-k #567/#568; compiled decode segments generalized to Qwen3-Next #569 and
module weights declared as compile state (`CompiledTrace`) #589; downstream
specialization hooks for the Qwen3.5 GDN/MoE blocks and an `open` SwitchGLU
#511; variance-normalized KV cache #329; byte-balanced parallel weight
loading #575; prompt-cache reuse for text-only inputs #549 and its report in
`GenerateCompletionInfo` #559; reranker API #375; Helium #555; LoRA dropout
#541; VLM processor loading rules #565; Gemma4 LoRA layers #602.

The pin branch is built directly on upstream `main` — no PR branch of ours
is in flight any more. Two carries were re-expressed against the new base
rather than picked verbatim (`perf(dflash2): model-side verify prebuild +
same-input QMM stacking` and `fix(dflash2): stack only plain QuantizedLinear
projections`):

- The GDN in-projection stack (`stackInProjections`) is gone — #572's
  `prepareFusedInputProjection` fuses the same rows in the same order under
  the same exact-class guard, so `dflash2StackGateUpProjections` now counts
  upstream's fusion for that group and keeps gate|up, attention q|k|v and
  the drafter's stacks. Bitwise-neutral by construction (one concatenation
  along the output axis; per-row K-accumulation unchanged). Rollback switch
  for the upstream half: `MLX_QWEN_FOUR_GDN=0`.
- Every DFlash2 compiled function (`compiledVerifySegments`, the drafter's
  context/segment traces, the greedy selector) is a `CompiledTrace` with its
  weights declared as compile state, as #589 requires — upstream's `compile`
  shadow rejects a `[unowned self]` capture. The attention-pre trace keeps
  the rope offset as its second trace input.

Dropped from the carry list as merged upstream: the ten #471 commits
(`2ee084d`…`98076af`, plus the 08-18/09-03 review-round commits that only
ever lived on the PR branch) and `fix(qwen3_5): detect the raw-HF norm
convention from conv1d layout` (#598). The three `pin mlx-swift` commits
collapsed into one. The previous tip (`ddc1f66`) stays reachable through
the old gitlink history; old pin branches stay per the policy above.

mlx-swift needs no move: upstream requires `0.31.6` up-to-next-minor, no
newer tag exists, and the fork pin `24779d5` sits on the `0.31.6` tag.
mlx-core stays at v0.31.1 (thread-local command encoders block the move;
`docs/mlx-core-fork.md`).

Carried on top, in order:

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `fix: pin mlx-swift to the spokvulcan fork at 24779d5` | Exact-revision pin on `spokvulcan/mlx-swift` `pin-tesseract` (0.31.6 base + provenance + the Cmlx gitlink bumps carrying the C-series, qmv_wide, affine_qmm_mma8, SDPA mma8, multi-query SDPA and round-5 kernels + `dynamicSliceUpdated`). SwiftPM cannot mix revision and version requirements for one package, so this must match mlx-audio-swift and tesseract-speech exactly | Permanent local; never upstream |
| `feat(tokenizers): ChatTemplateRendering protocol + adaptor forwarding (C25)` | Exposes the render half of `applyChatTemplate` at the MLXLMCommon layer. Enables tesseract's render+token cache (experiments-ledger C25). Requires `renderChatTemplate` on the swift-transformers side — `spokvulcan/swift-transformers` `pin-tesseract` @ `63edf42` (`docs/swift-transformers-fork.md`) | Not filed (queued — owner go-ahead) |
| `feat(speculative): expose GenerationFinalizingTokenIterator` | Makes the finalize protocol (and the two upstream conformances) public so the app's own token loop (`TokenGenerationLoop`) can rewind speculative lookahead the way `generateLoopTask` does | Permanent local unless upstream wants it; kept out of the DFlash2 PR |
| `feat(speculative): DFlash2 block-parallel speculative decoding for Qwen3.5 (ADR-0061)` | The whole DFlash2 series reshaped into one commit in upstream's own shapes: `DFlash2DrafterModel` / `DFlash2TargetModel` protocols, `DFlash2SpeculativeTokenIterator`, factory/registry/container, `generate` overloads, Qwen3.5 target side (verify pass, `writeRows`, gated-delta captures), `SameInputProjectionStacking`. Fast path only — no environment knobs, no research arms | **Upstream PR ready** (branch `dflash2-upstream-clean` = `e3d4a20` + this commit; issue + PR drafts in the 2026-09-04 status entry) |

Earlier pin branches carried one `chore: pin mlx-swift to <rev>` commit per
accepted Cmlx experiment (C4–C13 and the 2026-07-24 review round). That
lockstep bookkeeping is collapsed into the single pin commit above as of the
2026-07-27 re-pin; `pin-2026-07-23-upstream-eaefe75` still has the long form.

## Upstream filing queue — closed 2026-09-03

The 2026-07-18 → 07-25 inference-perf loop's four units and the balanced
chunking are all upstream: #467 (compiled decode schedule), #468 (GDN
decode conv1d as fused multiply-adds), #469 (fused router top-k), #470
(balanced prompt chunking) and #471 (ParoQuant MoE batch, merged
2026-09-03). The mlx-core-side wins from the same loop (C1/C13/C8+C9 filed
as mlx#3918/#3919/#3920; C4/C5/C7 deferred) are tracked in
`docs/mlx-core-fork.md`. The DFlash2 series (ADR-0057/0058/0059) was
reshaped into a single upstreamable commit on 2026-09-04 (ADR-0061); the
issue and PR drafts are ready for the owner to post (status entry below).

### Status log

**Filed 2026-07-26** — umbrella issue
[#466](https://github.com/ml-explore/mlx-swift-lm/issues/466); unit 1 =
PR #467, unit 2 = #468 (stacked on #467), unit 3 = #469, unit 4 = #471,
balanced chunking = #470. mlx-core side: C1 = mlx#3918, C13 = mlx#3919,
C8+C9 = mlx#3920; C4/C5/C7 deferred (upstream restructured the Metal
command-buffer machinery — DeviceStream merged into CommandEncoder,
thread-local encoders — so the port needs a rebase + re-measure), C6
half-superseded by mlx#3869 (regex removal), re-measure before filing.

**Status 2026-07-29** — #460/#467/#469 merged upstream; the three open
PRs (#468, #470, #471) were each rebased onto the fresh main (861649b),
full CI replica green per branch, force-pushed — all three MERGEABLE,
awaiting review. #470's merge commit was linearized away in the rebase.

**Status 2026-07-31** — #468 merged upstream 2026-07-30 (0321f28). #470
rebased onto the fresh main (a2736d4): the #448 Qwen2/2.5-VL windowed
prefill collided with the PrefillParameters reshape, so those models'
new `prepareContinuation` loops adopt `resolvedStepSize()` +
`forEachChunk` with the reserved tail position (mirroring the Qwen35
treatment), and upstream's new ChatSession/Nanbeige/Qwen25VLContinuation
tests migrated to `prepare(prefill:)`. The standalone lint-fixup commit
folded into the signature commit; full CI replica green, force-pushed —
MERGEABLE, awaiting review. #471 unaffected by the new main, still
MERGEABLE.

**Status 2026-08-17** — #470 merged upstream 2026-08-06 (4c7874b). #471
rebased onto the fresh main (d7dc03d) for the 2026-08-17 re-pin — clean
rebase, full CI replica green (lint, verify-docs, build-for-testing, 498
tests), plus a new commit `fix(paroquant): resolve chat conventions when
the caller passes none` adapting the loader to the post-#471 conventions
scheme; force-pushed, MERGEABLE, awaiting review.

**Status 2026-08-18** — davidkoski reported two Prepared Checkpoint tests
failing on upstream's self-hosted macOS runner while green locally. Cause:
`write()`'s advisory free-space guard read
`volumeAvailableCapacityForImportantUsage`, which resolves through
cache-management machinery CI hosts lack, so it reported nothing and the
write was silently skipped. Fixed by `fix(paroquant): don't skip the
checkpoint write when free space is unreadable` — fall back to the
statfs-backed capacity, and unknown capacity never vetoes the write. Same
day the branch was rebased onto upstream main `7871b09`; force-pushed, tip
`f2dd7dc`.

**Status 2026-08-28** — #471 rebased onto upstream main `37688d2` (26 new
commits: reranker API #375, variance-normalized KV cache #329, parallel
byte-balanced weight loading #575, fused/shared MoE router top-k #567/#568,
compiled decode segments generalized to Qwen3-Next #569, direct expert
reduction #573, fused GDN input projections #572, Helium #555, LoRA dropout
#541, VLM processor loading rules #565). Two conflicts, both mechanical:

- `ParoQuantLoader` step 12 — upstream replaced `eval(model)` with
  `materializeModelForInference(model)`; kept upstream's call and our
  `markPhase("eval")` around it.
- `SwitchLayers` — upstream's #573 split `callAsFunction`'s dataflow out
  into `projectExperts` (shared with the new `callAndWeightedReduce`), so
  the `transformInput`/`transformHidden` hooks now sit on `projectExperts`.
  That is the better seam: the PARO rotations reach both the plain call and
  the new fused reduction. `weightedExpertUnsort` runs downstream of
  `down_proj`, so it composes with the rotations either way.

Full CI replica green (pre-commit/swift-format 603, `build-for-testing`,
verify-docs, 565 XCTest + 722 Swift Testing). Force-pushed, tip `3ae4a12`,
MERGEABLE; workflow runs sit at `action_required` pending maintainer
approval. Pre-rebase tip `f2dd7dc` kept on local branch
`backup/paroquant-moe-pre-rebase-20260828`. **The Vendor pin
(`pin-upstream-mlx-swift`) still carries the pre-08-18 #471 commits — pick
the rebased ten from `3ae4a12` at the next re-pin.**

**Status 2026-09-03** — davidkoski's first review of #471 (nine inline
comments, 2026-09-02) verified and answered. Branch rebased onto upstream
main `5694a2f` (9 new commits, clean; #511 made `SwitchGLU` `open`, #598
landed upstream the same conv1d-layout norm-shift fix carried below — drop
that carry at the next re-pin). One review-round commit `c39c560`
`fix(paroquant): address review round — frozen rotations, sized generic
kernel, resolved tool formats`: `PairwiseRotation` freezes (direct weighted
reduction requires no trainable params); the generic rotation kernel is
templated on groupSize/krot/element type (any even groupSize ≤ 2048, bf16
on both kernels) with geometry checked once (typed
`ParoQuantError.unsupportedRotationGeometry` at load + init preconditions);
`convertAutoAWQ` casts scales/biases to the checkpoint float dtype read
from the rotation tensors; `loadParoQuantModel` resolves tool formats via
`ToolCallFormat.resolved(forTokenizerDirectory:)`. Two comments answered
without code by measurement: the artifact carries the vision tower (667 MB /
17.5% on the 4B, 893 MB / 4.3% on the 35B — kept, one artifact serves both
containers; subset artifact offered as a follow-up) and f16 cos/sin
derivation (matches the z-lab MLX reference bitwise; max |c²+s²−1| 6.8e-4
f16 vs 1.2e-7 f32 over the 4B's theta tensors — kept). Values-identical on
every shipped checkpoint (all-f16). Full CI replica: lint, build-for-testing,
verify-docs, 722/722 Swift Testing, 576/577 XCTest — the one failure
(`TurboQuantIntegrationTests.testRawKeyModeBFloat16MatchesReference`, cos
0.951 < 0.97) is untouched by the PR and passes 4/4 in isolation
(order-dependent random state). Force-pushed, tip `c39c560`. **Vendor pin
not yet moved to this tip** — next re-pin should take it.

**Status 2026-09-03 (re-pin)** — #471 merged upstream as `e23300b`
(byte-identical to `c39c560` in the ParoQuant files). Pin branch rebuilt on
upstream `main` `e3d4a20`: 15 carried commits (pin + C25 + 13 DFlash2), two
re-expressed against #572/#589 as described under "Current pin". Fork `main`
fast-forwarded to `e3d4a20`. The post-review ParoQuant polish that was
sitting uncommitted in the fork clone (rotations frozen at init, MLX
template-argument kernels, test rewrite) is parked on
`feat/paroquant-templated-kernels` (`2b71167`, WIP, unbuilt) — not carried.
Gates (2026-09-04): fork build + swift-format 603 + focused suites green;
app Release build with zero source changes, server + agent group green;
Prepared Checkpoint parity PASS on Qwen3.8-27B PARO and Qwen3.6-35B-A3B
PARO; DFlash2 bs8f acceptance 115/532 bit-identical old vs new pin, speed at
parity in an interleaved A/B (experiments-ledger R55).

**Status 2026-09-04 (DFlash2 reshaped for upstream, ADR-0061)** — the 13
DFlash2 carry commits (~5,000 lines, a dozen `DFLASH2_*` knobs, the
advised selector, lattice dump, accept log, profile timelines, fused
q/k-norm kernel, elementwise conv, verify stride, adaptive width,
passthrough, parity fixtures) collapse into one commit that keeps only the
measured fast path: pipelined round, fixed width 8, compiled draft/verify
segments, masked GDN replay, same-input QMM stacking. Shape follows
upstream's MTP drafter: `DFlash2DrafterModel` + `DFlash2TargetModel`
protocols instead of `LMOutput.State` keys, verify computes / iterator
commits (`KVCacheSimple.writeRows` + `GatedDeltaCapture`), stateless
drafter with per-stream `DFlash2ContextCache`, processor-copy losslessness,
`SameInputProjectionStacking`. `GatedDelta.swift` is back to upstream.
One bug surfaced by the acceptance gate and fixed before banking: the
prompt-window rows entered the drafter's context cache as placeholders and
were never resolved, so from round 1 the mask hid the whole prompt
(identity still MATCHed — the target verifies everything — but acceptance
fell to 108/577; a per-round trace against the old build diverged at round
1). Gates: 740/740 vendor tests, swift-format 603, verify-docs, full ABBA
vs the pre-reshape build on the docs prompt (`cd4da088…`): identity MATCH,
acceptance 115/532 bit-identical, per-round drafts identical to the old
build for the whole run, tok/s new 32.8 / 30.1 vs old 28.8 / 27.8 tok/s medians (per-run 29.5–32.8 vs 26.9–28.8), AR flat at ~21. Upstream branch
`dflash2-upstream-clean` (= `e3d4a20` + the commit, without the pin, C25
and the finalize-public carry) builds against upstream `mlx-swift` 0.31.6
(740/740 tests, verify-docs, swift-format). `MLX_MAX_ACTIVE_TASKS=40` is still set only by the
bench runner.

## Contributed back

| PR | What | Status |
| --- | --- | --- |
| [#147](https://github.com/ml-explore/mlx-swift-lm/pull/147) | GPU-only penalty processors, TopPSampler optimization | Merged 2026-03-27 |
| [#164](https://github.com/ml-explore/mlx-swift-lm/pull/164) | ParoQuant (pairwise rotation quantization) support | Merged 2026-05-11 |
| [#170](https://github.com/ml-explore/mlx-swift-lm/pull/170) | TokenRing.loadPrompt 2D-prompt fix | Merged 2026-05-11 |
| [#411](https://github.com/ml-explore/mlx-swift-lm/pull/411) | Qwen3VL sRGB tone curve in image preprocess | Merged 2026-07-13 |
| [#418](https://github.com/ml-explore/mlx-swift-lm/pull/418) | Qwen3 embedder: honor attentionMask | Merged 2026-07-13 |
| [#399](https://github.com/ml-explore/mlx-swift-lm/pull/399) | Qwen3.5/3.6 windowed prefill + state-threaded warm continuation (multi-turn M-RoPE drift fix) | Merged 2026-07-14 |
| [#398](https://github.com/ml-explore/mlx-swift-lm/pull/398) | Qwen3VL default per-image 1,280 vision-token budget | Merged 2026-07-15 |
| [issue #420](https://github.com/ml-explore/mlx-swift-lm/issues/420) | Qwen2/2.5/3-VL drop cross-turn state (same class as #399) | Filed; follow-up PR offered |
| [#460](https://github.com/ml-explore/mlx-swift-lm/pull/460) | Nanbeige4.2 looped-transformer model support | **Merged 2026-07-29** (3697686) |
| [issue #466](https://github.com/ml-explore/mlx-swift-lm/issues/466) | Umbrella: July 2026 inference-perf batch (map + totals) | Filed 2026-07-26 |
| [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467) | Qwen3.5/3.6 compiled decode step (C11+C12+leak fix+C14+review round, lifecycle tests) | **Merged 2026-07-29** (0bd3da4); 2026-07-29 simplify pass (51882f9, traced bodies deduped into shared `forward`) + review fix 5304b23 — NSLock around every lazy `compile` assignment (davidkoski: class properties are only thread-safe settable at init, weights not loaded yet) |
| [#468](https://github.com/ml-explore/mlx-swift-lm/pull/468) | GDN decode conv1d as fused multiply-adds (C16 + contract test) | Filed 2026-07-26; 2026-07-29 rebased onto deduped #467 (2ba11d5): fused conv extracted as `decodeConv` vs `generalConv`, test pins one against the other. **f32-input discovery: FMA ≠ Convolution kernel for f32 (102/256 channels) — fused branch gated to unmasked f16/bf16 S==1**; Vendor copy still carries the ungated C16 form (fine in practice: models run f16/bf16) — align at next re-pin. After #460/#467/#469 merged, re-rebased onto main (ee026ba, 2 own commits). **Merged 2026-07-30** (0321f28) |
| [#469](https://github.com/ml-explore/mlx-swift-lm/pull/469) | Fused router top-k kernel (C18, uint32 indices, contract test) | **Merged 2026-07-29** (861649b); review round 2026-07-29 — MLXFast import/dep removed (deprecated, lives in MLX) |
| [#470](https://github.com/ml-explore/mlx-swift-lm/pull/470) | Balanced prompt chunking (~9% prefill) | **Merged 2026-08-06** (4c7874b), landed as `PrefillParameters` with balanced chunking as the default |
| [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) | ParoQuant MoE batch: MoE path, Prepared Checkpoint, E1/E2/E6b (#164 follow-up); review round 2026-09-02/03 (frozen rotations, templated generic kernel, resolved tool formats) | **Merged 2026-09-03** (e23300b) |

Earlier fork-era contributions (#167 ToolCallProcessor schema plumbing, #168
TokenRing fix) predate the submodule pin scheme; see ADR-0006 for that history.

## Upstream candidates outside this fork (mlx-core)

Findings from the inference-optimization loop whose fix lives in mlx-core
(Cmlx). Since 2026-07-23 Cmlx **is** forked — `spokvulcan/mlx` +
`spokvulcan/mlx-swift`, scheme and per-iteration workflow in
`docs/mlx-core-fork.md`. The measured opportunity list is
`docs/mlx-core-optimization-roadmap.md` (M1–M8); evidence per experiment in
`benchmarks/experiments-ledger.md`. Two are ripe for filing as
evidence-backed issues against `ml-explore/mlx` — owner's call:

Filed 2026-07-26 as PRs (not issues): C1 tile geometry =
[mlx#3918](https://github.com/ml-explore/mlx/pull/3918), C13 fused causal
softmax = [mlx#3919](https://github.com/ml-explore/mlx/pull/3919), C8+C9
eval-path overhead = [mlx#3920](https://github.com/ml-explore/mlx/pull/3920).
C4/C5 (commit pipeline) and C7 (runtime commit-limit API) deferred: measured
on v0.31.1, and upstream has since merged DeviceStream into CommandEncoder
with thread-local encoders, so the port is a re-implementation that needs a
re-measure first. C6 is half-superseded by upstream mlx#3869.

- **Thread-unsafe stream binding** (mlx-c + mlx-swift, not mlx). mlx made
  command encoders thread-local (mlx#3281, #3348, both in v0.32.0), so
  mlx-swift's process-wide default `Stream` throws the moment a
  Swift-concurrency thread hop evaluates on it. The escape hatch,
  `new_thread_unsafe_stream`, has no mlx-c binding — and mlx-c#121 ("Bump to
  MLX 0.32.0") does not add one. Two small contributions: the mlx-c binding,
  then mlx-swift adopting it for `Device.defaultStream`. **This is what blocks
  the whole Swift stack from moving past mlx v0.31.1** — see
  `docs/mlx-core-fork.md`. Not filed.
- **M1** — `gather_qmm_rhs` tile geometry at small rows-per-expert:
  occupancy loss, not a bandwidth roofline (tesseract #256, ledger E4);
  worth ~12–15% of 35B MoE prefill. Not filed.
- **M2** — decode command-buffer segmentation: ~22% of MoE decode is
  inter-buffer idle (ledger E10). Not filed.

## Evidence asset branches — never delete

Orphan branches on the fork hosting images embedded (by raw URL) in upstream
issues/PR comments. Deleting them breaks the embeds:

- `assets/qwen3vl-srgb-evidence` (issue #410, tesseract PR #242)
- `assets/qwen3vl-budget-evidence` (PR #398 review reply)

## Re-pin procedure

1. In the fork clone (`~/projects/mlx-swift-lm`): `git fetch upstream origin`,
   fast-forward `main` to `upstream/main`, push.
2. `git checkout -B pin-upstream-mlx-swift <base>` where `<base>` is
   `upstream/main`, or the open PR branch that already contains it if one is
   still in flight.
3. Cherry-pick the carried commits from the previous pin, dropping any that
   merged upstream. Update the table above.
4. Build (`swift build`), push the branch (force push is expected — the branch
   is rebuilt each time).
5. In tesseract: fetch + checkout the new tip in `Vendor/mlx-swift-lm`, build
   the app (`scripts/dev.sh dev-release`), run the server/agent suites, commit
   the gitlink bump.

Gotcha: the fork's pre-commit hook formats the **whole repo** with the PATH
`swift-format` (602.x), which fights the CI-pinned 603 on import sorting.
Format touched files with `xcrun swift-format` (CI-matching) and commit with
`SKIP=swift-format git commit`.
