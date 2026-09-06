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
newer tag exists, and the fork pin (`6058402` since 2026-09-05, `24779d5`
before) sits on the `0.31.6` tag.
mlx-core stays at v0.31.1 (thread-local command encoders block the move;
`docs/mlx-core-fork.md`).

Carried on top, in order:

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `fix: pin mlx-swift to the spokvulcan fork at 24779d5` | Exact-revision pin on `spokvulcan/mlx-swift` `pin-tesseract` (0.31.6 base + provenance + the Cmlx gitlink bumps carrying the C-series, qmv_wide, affine_qmm_mma8, SDPA mma8, multi-query SDPA and round-5 kernels + `dynamicSliceUpdated`). SwiftPM cannot mix revision and version requirements for one package, so this must match mlx-audio-swift and tesseract-speech exactly | Permanent local; never upstream |
| `feat(tokenizers): ChatTemplateRendering protocol + adaptor forwarding (C25)` | Exposes the render half of `applyChatTemplate` at the MLXLMCommon layer. Enables tesseract's render+token cache (experiments-ledger C25). Requires `renderChatTemplate` on the swift-transformers side — `spokvulcan/swift-transformers` `pin-tesseract` @ `63edf42` (`docs/swift-transformers-fork.md`) | Not filed (queued — owner go-ahead) |
| `feat(speculative): expose GenerationFinalizingTokenIterator` | Makes the finalize protocol (and the two upstream conformances) public so the app's own token loop (`TokenGenerationLoop`) can rewind speculative lookahead the way `generateLoopTask` does | Permanent local unless upstream wants it; kept out of the DFlash2 PR |
| `feat(speculative): DFlash2 block-parallel speculative decoding for Qwen3.5 (ADR-0061)` | The whole DFlash2 series reshaped into one commit in upstream's own shapes: `DFlash2DrafterModel` / `DFlash2TargetModel` protocols, `DFlash2SpeculativeTokenIterator`, factory/registry/container, `generate` overloads, Qwen3.5 target side (verify pass, `writeRows`, gated-delta captures), `SameInputProjectionStacking`. Fast path only — no environment knobs, no research arms | Upstream PR #607 (branch `dflash2-upstream-clean` = `e3d4a20` + this commit) |
| `chore(deps): pin mlx-swift to 6058402 (dynamicSlice op, mlx b6a5f3b6)` (`fa7012a`) | Moves the pin to the 2026-09-05 loop's mlx-swift/mlx commits (`dynamicSlice`, QMM tile diet + v2 default, 1-pass SDPA, fast-math custom kernels, profiler probes) | Permanent local; collapses into the pin row at the next re-pin |
| `perf(qwen35): fuse the GDN conv + norm, the gated output norm and the scan's gate tables` (`552fd61`) | `GatedDeltaConvNorm.swift`, `GatedDeltaNormGate.swift`; in-kernel gate tables and output-only / state-after-valid scan variants in `GatedDelta.swift`; `GatedDeltaCapture` carries gates. All bitwise with the ops chains | Follow-up PR candidate on #607 (2026-09-05 loop) |
| `perf(qwen35): fuse the q/k RMSNorm + RoPE and fold the attention scale into the query norm` (`fc20fec`) | `AttentionNormRope.swift` (`fastmath_` kernel name: bitwise with the AOT `rope` kernel only under the fork's fast-math compile), `PlainRoPEParameters` from the config, folded power-of-two query scale, head-major gate | Follow-up PR candidate; the `fastmath_` compile needs an mlx-side change first |
| `perf(qwen35): fuse each residual add into the RMSNorm that follows it` (`9f5f43e`) | `RMSNormResidual.swift` (bitwise with Add then RMSNorm over both norm geometries), next-norm plumbing through the decode/verify segments and the drafter's | Follow-up PR candidate |
| `perf(dflash2): fused drafter dynamic conv, greedy walk, top-k and a head over the vocabulary prefix` (`0647cf9`) | `DFlash2DynamicConv.swift`, `DFlash2GreedyWalk.swift`, `TopKIndices.swift`, the 98304-row head prefix (`DFLASH2_DRAFT_VOCAB`, `observeCommitted`), context-cache slack rows via `dynamicSliceUpdated`, traces declaring their modules | Follow-up PR candidate; the slack-row write pays off only with the fork's `MLX_DYNSLICE_INPLACE` |
| `Keep the text before a possible tool-call tag when the tag does not complete in the chunk` (`a1bd36d`, cherry-pick of `ede8b3f`) | `ToolCallProcessor.processChunk` returned `nil` while buffering a possible `<tool_call>` start and lost the text split off before the `<` (" a `<memory>`" → " a<memory>`", `i < n` → `i< n`); the fix returns that text from the call that split it. Found because the mangled client echo broke **Live Leaf Capture**'s live-path equality. Four tests in `ToolTests` | Upstream issue [#609](https://github.com/ml-explore/mlx-swift-lm/issues/609) and PR [#610](https://github.com/ml-explore/mlx-swift-lm/pull/610) opened 2026-09-06 from fork branch `fix/tool-call-processor-leading-text` (`ede8b3f` = upstream `e3d4a20` + fix). Drop from the carry when it merges |
| `Emit only the new scalars when a token extends the previous character` (`ed74418`, cherry-pick of `c3c12ed`) | `NaiveStreamingDetokenizer.next()` measured the common prefix between the previous decode and the new one in `Character`s, so a token that appended a combining scalar (U+FE0F, a zero-width joiner, an accent) to the previous character re-emitted the whole merged cluster: `🏳️‍🌈` streamed as `🏳🏳️🏳️‍🏳️‍🌈`, `'️` as `''️`. The prefix is now measured in Unicode scalars. Found because the duplicated characters reached Pi's tool arguments and the file it wrote, and broke **Live Leaf Capture**'s live-path equality on every emoji turn. Four tests in `StreamingDetokenizerTests` | Upstream issue [#612](https://github.com/ml-explore/mlx-swift-lm/issues/612) and PR [#613](https://github.com/ml-explore/mlx-swift-lm/pull/613) opened 2026-09-06 from fork branch `fix/streaming-detokenizer-grapheme` (`c3c12ed` = upstream `e3d4a20` + fix, CI replica green locally). Drop from the carry when it merges |

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

**Status 2026-09-05 (optimization loop landed)** — the 10-hour DFlash2
throughput loop (travel 68.3-69.8 -> 54.2-54.4 ms/round, 69.4 tok/s at the
same acceptance; ledger `benchmarks/dflash2/FINDINGS.md`) is committed on
`pin-upstream-mlx-swift` as `fa7012a` (pin to mlx-swift `6058402`) +
four per-family `perf(qwen35|dflash2)` commits, tip `0647cf9`; mlx-side
in spokvulcan/mlx `b6a5f3b6` and mlx-swift `6058402` (pushed). Section
"2026-09-05 optimization loop — landed" below has the map. Upstream:
these are follow-up PR candidates on #607, not filed.

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
| [#610](https://github.com/ml-explore/mlx-swift-lm/pull/610) (fork branch `fix/tool-call-processor-leading-text`, `ede8b3f`) | `ToolCallProcessor.processChunk` drops the text before a `<` that might start a tool call (issue [#609](https://github.com/ml-explore/mlx-swift-lm/issues/609)) | OPEN 2026-09-06, CI replica green locally; fork-PR CI waits for maintainer approval |
| [#613](https://github.com/ml-explore/mlx-swift-lm/pull/613) (fork branch `fix/streaming-detokenizer-grapheme`, `c3c12ed`) | `NaiveStreamingDetokenizer` repeats the previous character when a token adds a combining scalar (issue [#612](https://github.com/ml-explore/mlx-swift-lm/issues/612)) | OPEN 2026-09-06, CI replica green locally; fork-PR CI waits for maintainer approval |

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

## 2026-09-05 optimization loop — landed

Final numbers (verification 08:30, `--bench-check`, cool GPU): travel
54.24 / 54.36 ms/round 140/356 identity MATCH (69.4 tok/s, from 54-55 at
00:20), code 54.48 / 54.57 141/349 MATCH (70.4 tok/s), math 54.50 / 54.57
158/249 DIVERGED +8 as before the loop (97.8 tok/s). The measured ledger
is `benchmarks/dflash2/FINDINGS.md`; the reference fixtures after the loop
are travel 140/356, code 141/349, math 158/249 (192 tokens each, streams
identical to the pre-loop streams; the draft acceptance re-rolled once,
when the small-M QMM tile switched to scale-after-accumulate).

Everything the loop kept is committed and pushed (11:40-12:30, see the
commit split below); the refuted knobs and the `sdpa_mma_prefetch` kernel
variant were stripped first and the three fixtures re-verified on the
stripped text (AR and speculative fingerprints equal to the 06:55 v2
references, identity MATCH on travel and code). A clean Release build on
the re-resolved package graph (12:09-12:16) confirmed the landed stack:
travel 54.1/54.3 ms/round, 69.5 tok/s, 140/356; code 54.0/54.2 ms/round,
70.9 tok/s, 141/349; identity MATCH on both, AR and speculative
fingerprints equal to the v2 references.

### Where the mlx-side changes live

- spokvulcan/mlx `pin-tesseract` `b2fcc671` -> `b6a5f3b6`: `7e1110cc8`
  feat(metal) fast-math compile for custom kernels named `fastmath_*`;
  `122e60c46` feat(metal) `MLX_KERNEL_PROFILE` / `MLX_CB_PROFILE` probes;
  `9c1dfe5b1` perf(metal) small-M 4-bit QMM tile diet, `full_tiles`
  variant and v2 scale-after-accumulate default (`MLX_QMM_MMA8_V2=0`
  kill-switch, `MLX_QUANTIZED_KERNEL_FILE` dev override, `MLX_QMM_DEBUG`);
  `b6a5f3b61` perf(metal) 1-pass SDPA for any qL <= 8 (gqa <= 32) and
  32/64 partitions for the 2-pass MMA kernel + the unit-scale skip in
  `fast.cpp`.
- spokvulcan/mlx-swift `pin-tesseract` `24779d5` -> `6058402`: `a9e589f`
  feat `dynamicSlice` op (`Ops+DynamicSlice.swift`); `6058402` the gitlink
  bump with `mlx-generated/quantized.cpp` regenerated from the submodule
  (cmake `-DMLX_METAL_JIT=ON`, `make quantized`, as `tools/update-mlx.sh`
  does — the regenerated text differed from the hand-carried region
  replace only by a trailing separator comment). `mlx-generated/metal/
  sdpa_vector.h` is back at its base (the prefetch variant is gone).
- The three `Package.swift` pins (`Vendor/mlx-swift-lm`,
  `Vendor/mlx-audio-swift`, `Vendor/tesseract-speech`) moved to `6058402`
  in lockstep; the DerivedData checkout re-resolved clean at `6058402` /
  `b6a5f3b6` (SwiftPM refuses the submodule update while the checkout
  carries local edits — reset the submodule to the pushed commit first).

### Environment knobs the loop left behind

| Knob | Read in | Default | Disposition |
|---|---|---|---|
| `DFLASH2_DRAFT_VOCAB` | `MLXLLM/Models/DFlash2.swift` (`headLogits`) | 98304 rows, `0` = full head | product default; keep documented |
| `MLX_QMM_MMA8_MMIN` | fork `quantized.cpp` | 5 (mma8 tile serves M = 5..8; below that the qmv family) | keep or strip to the constant; the tile's M gate also requires `group_size == 64`, `K % 512 == 0`, `N >= 2048`, bf16 activations, so other models' odd K (11008, 8960) fall through to the generic route |
| `MLX_QMM_MMA8_V2` | fork `quantized.cpp` | `1` (v2 tile) | keep: `0` is the kill-switch to the per-element tile (bitwise with the pre-loop logits; v2 differs in logit low bits, -6% per QMM) |
| `MLX_QMM_MMA8_N16` | fork `quantized.cpp` | on | keep: kill-switch to the 8-wide tile |
| `MLX_DYNSLICE_INPLACE` | `TesseractApp.swift`, bench runner | set to 1 by the app | keep until the fork defaults it |
| `MLX_SDPA_1PASS_ANY_QL` | fork `scaled_dot_product_attention.cpp` | on | DONE: the fork default (`use_fallback` serves any qL <= 8 with gqa <= 32), knob gone |
| `GDN_SCAN_RPT` | `MLXLMCommon/GatedDelta.swift` | 2 (falls back to 1 unless it divides `Dv`) | DONE: the constant `gatedDeltaRowsPerThread = 2` |
| `GDN_CONV_NORM_VARIANT`, `ATTN_NORM_ROPE_VARIANT` | `GatedDeltaConvNorm.swift`, `AttentionNormRope.swift` | v2 / plain fast-math form | DONE: variant structs, mode template args and the losing kernel forms gone; the bench runner's two microbenches check the one production kernel |
| `MLX_SDPA_2PASS_MIN_N`, `MLX_SDPA_2PASS_MIN_KL`, `MLX_SDPA_MMA_PREFETCH` | fork | off | DONE: stripped with the `sdpa_mma_prefetch` kernel variant (function constant 27) |
| `MLX_SDPA_MMA_BLOCKS`, `MLX_QMM_MMA8_N32` | fork | override off / off | KEPT: both pre-date the loop (in `b2fcc671`); `MLX_SDPA_MMA_BLOCKS` now only overrides the 32/64 default. Strip in a follow-up if wanted |
| `MLX_QMM_DEBUG`, `MLX_KERNEL_PROFILE`, `MLX_CB_PROFILE`, `MLX_QUANTIZED_KERNEL_FILE`, `MLX_MAX_ACTIVE_TASKS` | fork | off | keep: diagnostics / dev loop |
| `DFLASH2_QMM_MICROBENCH`, `DFLASH2_QMM_SHAPES`, `DFLASH2_GDN_ITERATIONS`, `DFLASH2_BW_MICROBENCH`, `DFLASH2_SDPA_MICROBENCH` | `DFlash2BenchRunner.swift` | off | keep: bench-only microbenches |

### Vendor test pass (07:29-07:50)

`xcodebuild test -scheme mlx-swift-lm-Package -destination platform=macOS
-skipPackagePluginValidation -only-testing:MLXLMTests`, at the time with
the `spokvulcan/mlx-swift` dependency temporarily swapped for
`.package(path: <DerivedData checkout>)` because the pinned revision
lacked `dynamicSlice` (no longer needed: the pin is `6058402`). Two rules
the tests taught, both applied:

- MLX binds custom-kernel inputs under 8 elements in the `constant` address
  space, so `const device T* p = input + ...` in an `MLXFast.metalKernel`
  source fails to JIT on tiny test arrays (the fused-gate scan, greedy
  walk, dynamic conv, conv+norm and norm+gate kernels all did, taking the
  xctest process down). Every such pointer is `auto` now.
- The GDN scan's rows per thread must divide `Dv` (the grid is `Dv / RPT`
  simdgroups per head); the launch falls back to one row per thread
  otherwise (the Kahan test's `Dv = 1` state ran nothing before).
- When a Swift Testing case crashes the xctest process the log cannot name
  it (every case runs in parallel); read the faulting thread of the newest
  `~/Library/Logs/DiagnosticReports/xctest-*.ips` instead. To run Swift
  Testing functions alone: `-only-testing:MLXLMTests/<function>()` (the
  parentheses are part of the identifier; without them nothing matches
  and xcodebuild reports success over zero tests).
- Xcode's parallel test runner hung at 0% CPU in two of four full runs of
  this GPU-heavy target (~165 of ~800 Swift Testing cases finished). With
  `-parallel-testing-enabled NO -test-timeouts-enabled YES
  -default-test-execution-time-allowance 120` the whole target passes:
  XCTest 592 (1 skipped), Swift Testing 743 cases in 55 suites, 2.5 min
  total. Use those flags when replicating CI.
- CI replication on the committed tree (12:03-12:10, run on the first-cut
  tip `ac3a96d`, whose tree the final tip `0647cf9` reproduces exactly):
  swift-format 6.3 over the tree (no changes outside the loop's files),
  `scripts/verify-docs.sh` green after two DocC link fixes (`RoPE` is an
  MLXNN symbol; a `GateLayout` doc linked its parent's package-level init),
  `build-for-testing` green, serialized `MLXLMTests` green (XCTest 592,
  1 skipped, 0 failures; Swift Testing 743 cases in 55 suites, 2.5 min).
  Every intermediate commit (`fa7012a`, `552fd61`, `fc20fec`, `9f5f43e`)
  builds for testing on its own; the first cut of the residual commit
  (`913db53`) carried the top-k test and did not, so the two top commits
  were rewritten (12:16-12:18) and the branch force-pushed with lease.

### Commit split (executed 2026-09-05, 11:40-12:30)

Bottom of the stack first, each replicating CI (lint, verify-docs, the
serialized xctest run above) before a push. What was executed differs
from the plan below in five places: the fast-math compile for `fastmath_*`
custom kernels got its own fork commit (`7e1110cc8`) ahead of the probes;
the four test repairs were made inside the family commits that introduced
the code they repair (no separate `fix(tests)`); the loop-only knobs were
stripped before anything was committed, so no `chore: strip` commit exists;
`MLX_SDPA_MMA_BLOCKS` / `MLX_QMM_MMA8_N32` stayed (pre-existing, see the
table); and the vendor branch was force-pushed once, after the residual
and drafter commits were re-cut so that the top-k test lands with the
top-k kernel (the tree is unchanged). Vendor tip `0647cf9` = `fa7012a`
pin + `552fd61` GDN + `fc20fec` norm+RoPE + `9f5f43e` residual+norm +
`0647cf9` drafter. The plan as approved:

1. spokvulcan/mlx `pin-tesseract` (submodule under the DerivedData
   checkout, 10 files): `perf(metal): small-M 4-bit QMM tile diet,
   full-tile variant and v2 scale-after-accumulate default` (`kernels/
   quantized.h`, `quantized.cpp`, `jit_kernels.cpp`, `quantized.metal`),
   `perf(metal): 1-pass SDPA for any qL <= 8 and 32/64 partitions for the
   2-pass MMA kernel` (`scaled_dot_product_attention.cpp`, `fast.cpp`,
   `sdpa_vector.h`), `feat(metal): MLX_KERNEL_PROFILE and MLX_CB_PROFILE
   probes` (`device.cpp/.h`, `eval.cpp`). Strip the refuted knobs and the
   `sdpa_mma_prefetch` function-constant variant (82 lines, measured
   neutral to slower) before the first of these.
2. spokvulcan/mlx-swift (the checkout, 3 files): `feat: dynamicSlice op`
   (`Ops+DynamicSlice.swift`), plus the regenerated `mlx-generated/
   quantized.cpp` preamble and `mlx-generated/metal/sdpa_vector.h`, which
   must be re-derived from the submodule commit rather than hand-carried.
   Then move the vendor's `Package.swift` pin to the new revision.
3. spokvulcan/mlx-swift-lm (vendor, 7 modified + 7 new files): one
   `perf(dflash2): ...` commit per fused kernel family keeps upstream
   review possible (GDN conv+norm / norm+gate / gate tables; q/k norm +
   RoPE; drafter dynamic conv, greedy walk, top-k, head vocabulary
   prefix; RMSNorm+residual), a `fix(tests): ...` commit for the four test
   repairs, and a `chore: strip loop-only env knobs` commit per the table
   above. Upstream PR #607 stays the drafter/iterator reshape; these are
   follow-up PRs on top of it.
4. tesseract: `feat(bench): DFlash2 fast bench, fixtures and ledger`
   (`scripts/dflash2-bench.sh`, `scripts/dflash2-compare.py`,
   `benchmarks/dflash2/`, `DFlash2BenchRunner.swift`, `scripts/bench.sh`),
   `fix(app): set MLX_DYNSLICE_INPLACE at launch` (`TesseractApp.swift`),
   and the docs commit for this file; the vendor pin moves last.

### Follow-ups the loop measured but did not build

- Per-group activation row sums precomputed once per QMM for the v2 tile:
  BUILT AND REVERTED. -2..-3% per QMM in the microbench, nothing
  measurable in ms/round, and the `simd_sum` order flipped target argmaxes
  (travel identity DIVERGED at +38, code at +146). A retry must reproduce
  the tile's own shuffle order bit for bit (ledger, 07:20).
- Long context: the 2-pass MMA SDPA runs ~4x off the K/V bandwidth floor at
  16k keys (summary fixture 78-84 ms/round at 6k prompt tokens).
- Tree / multi-candidate verification (acceptance lever; algorithmic).
- Prefill, not decode: the prompt chunks (M = 53 / 81 rows) run MLX's
  generic quantized GEMM at 9-11 TFLOPS (gate_up 3.1 ms at M = 81, 1.2 ms
  at M = 53 for 100 MB of weights) — a separate program, outside this
  loop's per-round ruler.
