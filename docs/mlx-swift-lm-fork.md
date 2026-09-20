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
reachability. Since `pin-upstream-mlx-swift` itself is force-pushed on every
re-pin, **every outgoing tip is tagged first** as `pin-tip/<date>-<sha>` and
the tag pushed (`pin-tip/2026-08-17-ddc1f66`, `pin-tip/2026-09-07-921c676`,
…); a branch tip nothing else points at is otherwise gone from the fork the
moment the branch moves, which is what had happened to `ddc1f66` before the
2026-09-15 re-pin backfilled its tag from a local clone. `pin-gemma4-12b-358` is
the parked Gemma 4 12B multimodal stack (audio encoder + encoder-free
`gemma4_unified` processor + suppress_tokens) that tesseract draft PR #359
pins; it rejoins this table's carry list only if that experiment is revived.

## Load memory carry (2026-09-20, #550)

The #550 app branch advances the gitlink from `f177464` to `a3c1776` on
`fix/550-load-memory-retention`, five commits on top of the #533 carry
(fast-forward; `pin-upstream-mlx-swift` and every historical tip
unchanged). `5e353f1` only cites the upstream reports in the sibling
probes: the assignment case is fixed by ml-explore/mlx#4453 (merged
2026-09-11, not in the pinned mlx), the compiled split case is
ml-explore/mlx#3932 (open). `a3c1776` writes the trace's state and body
closures in the labeled form the upstream branch uses (swift-format's lint
flags a closure argument beside a trailing closure). The three that change
code:

- `4bbca60` `feat(load): read only the indexed shard a key prefix maps to`.
  `WeightFileSelection.indexedKeyPrefix` reads the files the safetensors
  index maps a key prefix to; the MTP drafter factory asks for `mtp.`, so
  the head loads from `model-mtp-head.safetensors` alone instead of
  materializing the whole target checkpoint its `sanitize` then dropped.
- `9ad90d7` `fix(qwen35): declare the fused GDN projection as compile
  state`. The fused four-way projection is not a registered child, so the
  decode, segment and verify traces read its arrays as tape constants and
  MLX kept them alive after the traces were erased: 2.3 GB of Qwen3.8-27B
  resident per load, growing on every reload (0 → 3.0 → 5.3 → 7.6 → 9.9 GB
  across the e2e's reloads). Every trace that runs a GDN layer now declares
  the fused module as compile state (`fusedProjectionTraceState`,
  `traceState(forLayers:)`). Regression: a dropped fused model leaves 4
  bytes resident (18,852 before). `SiblingCycleTests` documents the two
  upstream causes as expected failures: erasing a compiled function whose
  tape `split`s a captured constant keeps the constant alive
  (ml-explore/mlx#3932), and assigning over a multi-output sibling (what
  `MLXArray._updateInternal` does through mlx-c `mlx_array_set`) skips
  MLX's cycle break in `~array` (ml-explore/mlx#4453, fixed upstream);
  the second leaks the lazy init-quantize graph of every `QuantizedLinear`
  (descriptors and 4-byte scalars, ~2.5 MB per load).
- `f8b4827` `fix(stacking): free each block's originals before packing the
  next`. `stackSameInputProjections(in:)` iterated `modules()`, whose array
  holds every projection, so all originals lived until the loop ended: a
  7 GB transient over the 15.9 GB model on Qwen3.8-27B (peak 22.97 GB),
  the swap spike on the 48 GB machine. The loop keeps only the stacking
  modules; the test bounds the transient at two blocks (8 blocks of 5.2 MB
  held 41.9 MB before).

Validation: `LoadWeightsTests` 25, `Qwen35FusedGDNProjectionTests` 16 (1
skipped), `SiblingCycleTests` 10 (2 expected failures),
`CompiledDecodeWeightUpdateTests` 6, `CompiledTraceTests` 8,
`HadamardQuantizedTests` 17 and the two `DFlash2Tests` stacking tests pass;
formatter-clean on the touched files. The serialized `MLXLMTests` run is
green but for `TokenIteratorClearCacheTests.testFirstTokenClearsBufferCache`
(the 256 MB seed buffer does not land in the MLX cache: 309 KB cached), which
fails the same way on the pristine `f177464`, alone and on an idle machine;
it predates this carry and is not understood yet. Upstream: prepared on 2026-09-20 as
three branches on the fork, each built and tested against upstream's
`mlx-swift` 0.31.6 pin and formatter-clean under the CI-pinned swift-format
603.0.0. Filed by the owner on 2026-09-20: the fused-projection fix as
upstream PR [#631](https://github.com/ml-explore/mlx-swift-lm/pull/631), the
loader selection as [#632](https://github.com/ml-explore/mlx-swift-lm/pull/632);
the stacking fix rides on #607's branch. The texts as posted are below.

- `upstream/fused-projection-compile-state` (`b05e0ba` = vanilla `c6446cf`
  + the fix, re-applied by hand: the fork's commit conflicts with the
  carried verify traces). Upstream PR
  [#631](https://github.com/ml-explore/mlx-swift-lm/pull/631); drop from
  the carry when it merges. `Qwen35FusedGDNProjectionTests` 16 (1 skipped) on
  vanilla; the regression test retains 18,848 bytes on plain `main` and
  passes with the fix. `SiblingCycleTests` stays fork-only (it probes mlx).
- `upstream/indexed-key-prefix-selection` (`6756dd8` = `c6446cf` +
  `4bbca60`, cherry-picked clean). `LoadWeightsTests` 25 on vanilla.
  Upstream PR [#632](https://github.com/ml-explore/mlx-swift-lm/pull/632);
  drop from the carry when it merges.
- `dflash2-upstream-clean-stacking` (`580ef7b` = `56a21b2`, the branch
  behind PR #607, + `f8b4827`, cherry-picked clean): folds into #607 by
  fast-forwarding `dflash2-upstream-clean` to it. Both stacking tests pass
  there. Upstream `main` has no `SameInputProjectionStacking.swift`, so
  this cannot stand alone.

Related upstream reports: ml-explore/mlx#3932 (open) is the compiled
multi-output capture leak the fused-projection fix works around;
ml-explore/mlx#4453 (merged 2026-09-11, after the mlx `ce45c52` that
mlx-swift 0.31.6 ships) fixes the assignment-over-siblings case.

*PR* (`upstream/fused-projection-compile-state` → `main`, title "Declare
the fused GDN projection as compile state so it frees with the model"):

```markdown
## Proposed changes

Unloading a Qwen3.5 model that ran the compiled decode path leaves its fused GDN input projections resident, so memory grows on every reload.

The fused projection `prepare()` builds is not a registered child of `Qwen35GatedDeltaNet`, so `CompiledTrace`'s default state does not include it and the per-layer and decode-segment traces read its arrays as tape constants. MLX keeps a constant captured by a compiled function whose tape holds a multi-output primitive alive after the function is erased (ml-explore/mlx#3932). The fix declares the fused module as compile state wherever a trace runs a GDN layer, which is what the compile-state contract from #589 intends anyway.

`testCompiledDecodeReleasesFusedProjectionWithTheModel` runs three compiled decode steps, drops the model and asserts active memory returns to within 64 bytes of the baseline: 18,848 bytes retained on `main`, 4 with this change.

Measured in the app that embeds this library, Qwen3.5-27B 4-bit, MLX active memory in GB:

| phase | before | after |
|---|---|---|
| resident at load begin, reload 1 / 2 / 3 | 3.0 / 5.3 / 7.6 | 0.76 / 0.76 / 0.76 (a second small model the app keeps) |
| resident after load | 17.86 | 17.86 |
```

*PR* (`upstream/indexed-key-prefix-selection` → `main`, title "Load only
the shard the safetensors index maps a key prefix to"):

```markdown
## Proposed changes

The MTP drafter factory loads with the default file selection, so for a checkpoint that keeps the head in its own shard (`model-mtp-head.safetensors`, mapped by `model.safetensors.index.json`) it read and evaluated the whole checkpoint and then dropped every target tensor in `sanitize(weights:)`.

`WeightFileSelection.indexedKeyPrefix(prefix)` selects only the files the index maps a weight named `prefix…` to, falling back to `automatic` when there is no usable index or nothing matches. The MTP factory asks for `mtp.` unless the configuration sets an explicit selection. Two tests in `LoadWeightsTests` cover the selection and the fallback.

Qwen3.5-27B 4-bit with its MTP head, measured in the app that embeds this library:

| | before | after |
|---|---|---|
| bytes read and evaluated for the head | 15.1 GB (every shard) | 0.85 GB (the head shard) |
```

*Comment on #607* (its branch was fast-forwarded to `580ef7b` on
2026-09-20 with the owner's approval; posted as
[issuecomment-5747123420](https://github.com/ml-explore/mlx-swift-lm/pull/607#issuecomment-5747123420)):

```markdown
One more commit. `stackSameInputProjections(in:)` iterated `modules()`, whose array holds every projection module, so each block's originals stayed alive until the loop ended and the transient over a load was the sum of every stacked block instead of one. The loop now keeps only the stacking modules. `testSameInputStackingReleasesEachBlockBeforeTheNext` stacks eight quantized MLP blocks and bounds the peak at two; it held all eight before.

Qwen3.5-27B 4-bit in the app that embeds this library, MLX peak memory in GB:

| phase | before | after |
|---|---|---|
| peak during projection stacking | 22.97 | 17.84 (flat over the 15.9 GB model) |
```

Both PRs take the vendor template's checklist and AI-usage block; the
disclosure line is the owner's to write.

## Capacity reservation carry (2026-09-19, #533)

The #533 app branch advances the gitlink from `51542c4` to `f177464` on
`codex/533-cache-capacity-reservation`. This is a fast-forward carry; the shared
`pin-upstream-mlx-swift` branch and every historical tip remain unchanged while
the app PR is under review. `f177464` adds `KVCache.reserveCapacity(_:)`, honored
by simple and quantized attention caches and forwarded by CacheList. Allocation
increments double from 256 rows to 4096; prompt reservations use the initial
granule. Logical state, metadata and persistence formats stay unchanged.

The same commit is prepared on vanilla upstream `c6446cf` as `a7162cd`, branch
`codex/533-upstream-cache-capacity`, with no other fork carries. Upstream PR:
**pending the owner's read-and-approve attestation** required by the vendor's
`CONTRIBUTING.md` and PR template. The app PR remains draft until this acceptance
item is complete. Add the upstream PR link here once filed; drop the carry when
it merges and the app re-pins to that upstream revision (ADR-0006).

Validation: the carried revision passes the six new capacity tests and nine
existing serialization/copy/empty-cache tests; the clean upstream branch passes
`pre-commit run --all-files` with the Xcode formatter and nine selected capacity/serialization/copy tests
against upstream dependency pins. The app's final 795 tests in 58 prefix-cache
and touched suites pass, including canonical, speculative and raw prefill
reservation regressions. Both Standards and Spec review are clear after the
missing-path fix. Test-host model prewarms are disabled.
No loaded-model, long-context or model-reload campaign was run.

## Current pin (2026-09-15, second cut)

Base: upstream `main` @ `c6446cf` — 9 commits past the 2026-09-03 base
(`e3d4a20`). The same-day second cut adds #620 (clear the MLX buffer cache
on the first generated token, as mlx-lm does, so short generations no
longer leave their prefill and decode buffers cached — a two-line move in
`TokenIterator.next()`, which the app's autoregressive path drives; the
DFlash2 iterator clears between prefill chunks and has no per-token
cadence to move). The first cut's tip `e5fec88` is tagged
`pin-tip/2026-09-15-e5fec88`; the rebase onto `c6446cf` was clean, every
carry re-hashed. The 8 commits before #620: **our #613 merged** (`4c3d793`, the streaming-detokenizer
scalar fix carried as `ed74418`); bounded cross-dialect tool-call recovery
and schema validation #548 (a rewrite of `ToolCallProcessor`, +3.4k lines,
which also covers the leading-text case of our #610 — its four tests pass
unchanged on the new main, so #610 was closed as superseded on 2026-09-15);
KV-cache reuse for append-only media turns #515; wrap-aware
`RotatingKVCache.trim` #584; configuration-based LoRA discovery #597; the
generation loop moved onto a private serial executor so `asyncEval`
backpressure no longer blocks Swift cooperative workers #611 (the app's own
`TokenGenerationLoop` is unaffected); fused float32 logit softcapping for
the Gemma 4 VLM #615; a differentiable gated-delta recurrence for training
#616.

Two carries were re-expressed against the new base:

- `perf(qwen35): fuse the GDN conv + norm…` conflicted with #616's
  `useKernel` flag on the fused-kernel gate in `gatedDeltaUpdate`. Resolved
  as the conjunction: `useKernel && usesFusedKernel(keyDimension:)`, with
  `useKernel` threaded through the `GatedDeltaGates` overload. Training
  (`useKernel: false`) takes the differentiable ops path as upstream
  intends; inference keeps the fused kernel.
- `perf(tools): scan only the chunk for a collecting call's end tag` was
  rewritten over #548's processor: the native path now finds the frame end
  through `ToolCallFrameScanner.frameEnd` (a whole-buffer structural scan
  per chunk) and the new recovery scanner, which the app's declared tools
  enable, does the same over its own buffer. Both call sites now gate the
  scan on `ToolCallFrameScanner.marker(_:mayHaveArrivedIn:appendedByteCount:)`
  (the appended bytes plus the tag's overlap). The linear-time test failed
  on plain upstream main at 13.7 s against its 3 s bound before the rewrite;
  a fifth test covers the declared-tools path.

2026-09-17: one commit added on top, fast-forward (no rebase, no
re-hash): `676080b`, the Hadamard-rotated ternary checkpoint loader
(ADR-0067). Tip moves from `46f0356` to `676080b`; the outgoing tip is
tagged `pin-tip/2026-09-17-46f0356`. 2026-09-18: its tidy-up `602a101`
added the same way (shared `HadamardQuantizedCheckpoint`, the embedding
over the stock `QuantizedEmbedding`); tip moves to `602a101`. Later the same
day `51542c4` (same-input stacking of Hadamard-rotated siblings, the GDN
`qkv|z` stack) added the same way; tip moves to `51542c4`.

Everything else cherry-picked clean. Dropped as merged or superseded:
`ed74418` (#613) and `a1bd36d` (#610). Outgoing tips are tagged
`pin-tip/2026-09-07-921c676` and `pin-tip/2026-09-15-e5fec88`.

mlx-swift needs no move: upstream still requires `0.31.6` up-to-next-minor,
no newer tag exists, and the fork pin stays at `6058402`. mlx-core stays at
v0.31.1; the blocker's first step landed upstream (mlx-c #122), and the move
is tracked as tesseract issue #513 (`docs/mlx-core-fork.md`).

Carried on top, in order:

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `536b670` `fix: pin mlx-swift to the spokvulcan fork at 24779d5` | Exact-revision pin on `spokvulcan/mlx-swift` `pin-tesseract` (0.31.6 base + provenance + the Cmlx gitlink bumps carrying the C-series, qmv_wide, affine_qmm_mma8, SDPA mma8, multi-query SDPA and round-5 kernels + `dynamicSliceUpdated`). SwiftPM cannot mix revision and version requirements for one package, so this must match mlx-audio-swift and tesseract-speech exactly | Permanent local; never upstream |
| `4662ad9` `feat(tokenizers): ChatTemplateRendering protocol + adaptor forwarding (C25)` | Exposes the render half of `applyChatTemplate` at the MLXLMCommon layer. Enables tesseract's render+token cache (experiments-ledger C25). Requires `renderChatTemplate` on the swift-transformers side — `spokvulcan/swift-transformers` `pin-tesseract-2026-09-15` (`docs/swift-transformers-fork.md`) | Not filed (queued — owner go-ahead) |
| `a3cb04d` `DFlash2 block-parallel speculative decoding for Qwen3.5` (ADR-0061) | The whole DFlash2 series reshaped into one commit in upstream's own shapes: `DFlash2DrafterModel` / `DFlash2TargetModel` protocols, `DFlash2SpeculativeTokenIterator`, factory/registry/container, `generate` overloads, Qwen3.5 target side (verify pass, `writeRows`, gated-delta captures), `SameInputProjectionStacking`. Fast path only — no environment knobs, no research arms | Upstream PR #607 (branch `dflash2-upstream-clean` = `3e6ea1e` + this commit, rebased and force-pushed 2026-09-15 as `56a21b2`, MERGEABLE; one unrelated upstream commit behind since #620) |
| `6498a02` `feat(speculative): expose GenerationFinalizingTokenIterator` | Makes the finalize protocol (and the two upstream conformances) public so the app's own token loop (`TokenGenerationLoop`) can rewind speculative lookahead the way `generateLoopTask` does | Permanent local unless upstream wants it; kept out of the DFlash2 PR |
| `8449a52` `chore(deps): pin mlx-swift to 6058402 (dynamicSlice op, mlx b6a5f3b6)` | Moves the pin to the 2026-09-05 loop's mlx-swift/mlx commits (`dynamicSlice`, QMM tile diet + v2 default, 1-pass SDPA, fast-math custom kernels, profiler probes) | Permanent local; collapses into the pin row at the next re-pin |
| `1f83de0` `perf(qwen35): fuse the GDN conv + norm, the gated output norm and the scan's gate tables` | `GatedDeltaConvNorm.swift`, `GatedDeltaNormGate.swift`; in-kernel gate tables and output-only / state-after-valid scan variants in `GatedDelta.swift`; `GatedDeltaCapture` carries gates. All bitwise with the ops chains. Re-expressed on #616 (see above) | Follow-up PR candidate on #607 (2026-09-05 loop) |
| `eadbf6c` `perf(qwen35): fuse the q/k RMSNorm + RoPE and fold the attention scale into the query norm` | `AttentionNormRope.swift` (`fastmath_` kernel name: bitwise with the AOT `rope` kernel only under the fork's fast-math compile), `PlainRoPEParameters` from the config, folded power-of-two query scale, head-major gate | Follow-up PR candidate; the `fastmath_` compile needs an mlx-side change first |
| `35c95ea` `perf(qwen35): fuse each residual add into the RMSNorm that follows it` | `RMSNormResidual.swift` (bitwise with Add then RMSNorm over both norm geometries), next-norm plumbing through the decode/verify segments and the drafter's | Follow-up PR candidate |
| `dc9bd1e` `perf(dflash2): fused drafter dynamic conv, greedy walk, top-k and a head over the vocabulary prefix` | `DFlash2DynamicConv.swift`, `DFlash2GreedyWalk.swift`, `TopKIndices.swift`, the 98304-row head prefix (`DFLASH2_DRAFT_VOCAB`, `observeCommitted`), context-cache slack rows via `dynamicSliceUpdated`, traces declaring their modules | Follow-up PR candidate; the slack-row write pays off only with the fork's `MLX_DYNSLICE_INPLACE` |
| `4ad84b2` `fix(dflash2): compute in the drafter's dtype whatever the target hands over` (tidied in `530e84e`) | `DFlash2DraftModel.propose` casts the target's block embedding and captured hidden states to the drafter's checkpoint dtype (`computeDType`), runs the target's head in the target's dtype and scores in its own — no-ops on the bfloat16 pairing. The float16 ParoQuant Qwen3.8-27B beside the bfloat16 drafter promoted every mixed matmul to float32, and the fused residual norm's dtype precondition crashed the server on every `qwen3.8-27b-paro` request since 2026-09-05. Test: `testDFlash2CompiledProposalMatchesEager` takes the pairing as an argument | Follow-up PR candidate on #607 (fold into the drafter commit) |
| `46f0356` `perf(tools): scan only the chunk for a collecting call's end tag` | Both scanners (the native `processTaggedChunk` and `TextToolCallRecoveryScanner`'s native-frame context) gate the whole-buffer frame scan on the appended bytes plus the tag's overlap holding the end tag, through `ToolCallFrameScanner.marker(_:mayHaveArrivedIn:appendedByteCount:)`. Five tests in `ToolCallProcessorLongCallTests` (linear time with and without declared tools; the tag split across chunks, in the start-tag-completing chunk, and followed by another call) | Upstream issue [#624](https://github.com/ml-explore/mlx-swift-lm/issues/624) and PR [#625](https://github.com/ml-explore/mlx-swift-lm/pull/625) (branch `perf/tool-call-end-tag-scan`, `31fcc22` = upstream `3e6ea1e` + this commit, one unrelated upstream commit behind since #620), posted 2026-09-15 with the owner's attestation. Drop from the carry when it merges |
| `676080b` `feat: load Hadamard-rotated ternary checkpoints (prism_hadamard_qwen35)` (ADR-0067, tidied in `602a101`) | `HadamardQuantized.swift` in MLXLMCommon: `SignedBlockHadamard` (float32 blockwise signed Hadamard, forward + inverse), `castingUnpackedWeights` (the pack stores norms, conv taps and the small GDN projections in float32; cast to the manifest activation dtype at load, `A_log` excepted, as mlx-vlm's converter does), `HadamardQuantizedLinear` / `HadamardQuantizedEmbedding` over the stock `QuantizedLinear` / `QuantizedEmbedding` (rotate the input before the packed matmul, un-rotate looked-up rows; `signs` is a frozen parameter so the pack's `.signs` tensors load through the ordinary weight path), `HadamardQuantizedManifest` (decode + validate the pack config), `substituteHadamardQuantizedModules` (replace exactly the manifest's `Linear` / `Embedding` leaves before the weights load) and `HadamardQuantizedCheckpoint` (the validated manifest with its path prefix and activation dtype: one validate / substitute / sanitize sequence for both model classes). `PrismHadamardQwen35Model: Qwen35Model` (MLXLLM) and `PrismHadamardQwen35: Qwen35` (MLXVLM) substitute in `init` and cast in `sanitize` through it; both registries map `prism_hadamard_qwen35` to them. Stock ops only, no kernels; 12 tests in `HadamardQuantizedTests`. Loads PrismML's Bonsai 2 27B (Qwen3.8-27B, ternary in affine 2-bit) | Upstream PR [#630](https://github.com/ml-explore/mlx-swift-lm/pull/630) (branch `upstream/hadamard-rotated-checkpoints`, `bac826f` = upstream `c6446cf` + this commit and `602a101` squashed), posted 2026-09-19 with the owner's attestation. Rebuilt on vanilla upstream: cherry-picked clean, builds and tests against upstream's `mlx-swift` 0.31.6 pin (12 `HadamardQuantizedTests`, 66 `ChatSessionTests`), formatter-clean under the CI-pinned swift-format 603.0.0. Drop from the carry when it merges |
| `51542c4` `perf(stacking): fold Hadamard-rotated siblings that share a sign vector` (ADR-0067 consequences, amended) | `stackedSameInputProjection` in `SameInputProjectionStacking.swift` takes the projections as `Linear`s: exact `QuantizedLinear`s stack as before; exact `HadamardQuantizedLinear`s stack when their `rotation` is equal and their `signs` are array-equal (`arrayEqual`, once, at load) into one `HadamardQuantizedLinear` — one rotation, one packed matmul, the callers' split unchanged; any other subclass is left alone. `Qwen35GatedDeltaNet` conforms with a `qkv|z` stack (`qkvzStacked`, a third branch in `projectInputs`) for the case the four-way fusion cannot take — a pack whose `in_proj_b` / `in_proj_a` are unquantized, as the rotated one's are; it defers to the fused row when that is prepared and to the `MLX_QWEN_FOUR_GDN` switch. Bitwise with the separate layers, tested at Bonsai 2 27B's projection shapes (5120-wide input; 12288\|1024\|1024, 10240\|6144, 17408\|17408) for a decode row and a prefill block. 5 tests added to `HadamardQuantizedTests` (17 total); serialized `MLXLMTests` green (XCTest 664, 2 skipped; Swift Testing 889 in 65 suites). Measured null on the 48 GB M3 Max: plain decode inside the pass-to-pass drift, decode being bandwidth-bound | Not filed — held back from #630: it edits `SameInputProjectionStacking.swift`, which reaches upstream only with #607, so it cannot build on plain `main`. Folds in as a follow-up once #607 merges; #630's description says so and records the null measurement |

## Pin of 2026-09-03 (superseded 2026-09-15)

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

The carry table of this pin is superseded by the 2026-09-15 one above; the
rows were the same carries at their pre-rebase SHAs, plus the two dropped
since (`a1bd36d` #610, `ed74418` #613).

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

**Status 2026-09-15 (second cut, #620)** — upstream merged #620 the same
evening; the pin was rebased onto `c6446cf` (clean, tip `46f0356`), the
first cut's tip tagged `pin-tip/2026-09-15-e5fec88`. Gates re-run: vendor CI
replica green (647 XCTest with upstream's new `TokenIteratorClearCacheTests`,
889 Swift Testing); app Release build; full Debug suite with the corpus
gates 2941 tests / 0 failures, 66/66 boundaries registered, 0 fidelity
rejections; travel DFlash2 fixture 140/356 identity MATCH on both streams
at 54.1 ms/round (69.6 tok/s, the 2026-09-05 reference speed on a quiet
machine).

**Status 2026-09-15 (re-pin)** — pin rebuilt on upstream `main`
`3e6ea1e`, tip `e5fec88` (section "Current pin (2026-09-15)" above). Same
batch: #610 closed as superseded by #548; #607 rebased onto the new main and
force-pushed (`56a21b2`, MERGEABLE, builds); the end-tag scan rewritten over
both scanners and pushed as `perf/tool-call-end-tag-scan` (`31fcc22`, built
from upstream `main`, not from the pin); outgoing tips tagged
`pin-tip/2026-08-17-ddc1f66` and `pin-tip/2026-09-07-921c676`. The
issue and PR below were posted the same day as
[#624](https://github.com/ml-explore/mlx-swift-lm/issues/624) and
[#625](https://github.com/ml-explore/mlx-swift-lm/pull/625), with the owner's
attestation given in the session (both upstream templates open with an "I
have read this and approve it as my own" checkbox, so an agent posts them
only on that go-ahead).

Gates:

- Fork (`e5fec88`): `swift-format` in place over the tree changes nothing; `verify-docs`; `build-for-testing`; serialized `MLXLMTests` green — XCTest 646 (2 skipped, 0 failures), Swift Testing 889 cases in 65 suites.
- App: Release build on the new pins; full Debug suite serialized with the emitted-path corpus gates on — 2941 tests, 0 failures, 14 skipped; corpus: 85 recordings, 66 boundaries, 66 registered live, 0 fidelity rejections, next-request misses none, slowest tail 84.6 ms.
- The first full-suite run had 2 failures, both the app's snapshot restore rejecting upstream's new seven-value `RotatingKVCache` metaState (#584 records the ring's wrapped layout). The restorability check now mirrors the setter's real precondition (5 to 7 values, origin tag, boolean flag); two tests added (six-value shape keeps restoring, non-boolean flag throws instead of reaching the vendor's fatalError). Second run green.
- Tokenizer cache bench (C25/C27/C28): PASS — 0 token mismatches, 0 parity failures, 0 path failures. Prefix-cache end-to-end: PASS.
- DFlash2 fixtures on Qwen3.8-27B 4-bit, block 8, `--bench-check`: travel 140/356 identity MATCH (58.1 tok/s, 64.8 ms/round), code 141/349 MATCH (62.2 tok/s, 61.7 ms/round), math 158/249 with the reference's known DIVERGED at +8 (84.3 tok/s). Same counts as the 2026-09-05 references.

*Issue* (template "Bug report"):

```markdown
- [ ] I have read this issue in full and approve it as my own, however it was
      drafted.

**Describe the bug**

`ToolCallProcessor` gets slower with every chunk of a long tool call. While it is collecting a tagged call it looks for the closing tag in the whole buffered call on every chunk, so a call streamed token by token costs time quadratic in its length. The cross-dialect recovery scanner that declared tools enable does the same over its own buffer. A `write` call of about 13k tokens (Qwen3.5 format, `<tool_call>` / `</tool_call>`) spends seconds in that scan alone, on the thread that drives sampling, and anything that replays the stream pays it again.

**To Reproduce**

```swift
import MLXLMCommon

let content = String(repeating: "x", count: 40_000)
let call = "<tool_call>\n<function=write>\n<parameter=content>\n\(content)\n</parameter>\n</function>\n</tool_call>"
let processor = ToolCallProcessor(format: .qwen35)
let start = Date()
for character in call { _ = processor.processChunk(String(character)) }
print(Date().timeIntervalSince(start))
// 13.7 s on an M-series laptop; the same loop with `tools:` declared takes as long
```

**Expected behavior**

Time linear in the call's length: well under a second for the loop above. Only the text a chunk appends can complete the closing tag, because an earlier occurrence was already in the buffer when the previous chunk was scanned, so the scan only needs to cover the chunk plus the tag's overlap with what preceded it.

**Desktop (please complete the following information):**
 - OS Version: macOS 26
 - Device: <fill in>
 - Version: main
```

*PR* (`perf/tool-call-end-tag-scan` → `ml-explore/mlx-swift-lm:main`, title
"Scan only the chunk for a collecting tool call's end tag"):

```markdown
## Proposed changes

Fixes #<issue number>.

While collecting a tagged tool call, `ToolCallProcessor` and the text recovery scanner searched the whole buffered call for its closing tag on every chunk, which is quadratic in the call's length: a 13k-token `write` call spent seconds in that scan alone. Only the text a chunk appends can complete the closing tag, because an earlier occurrence was already in the buffer when the previous chunk was scanned. Both scanners now check the appended bytes plus the tag's overlap with what preceded them, through one helper on `ToolCallFrameScanner`, and run the structural frame scan only when that window holds the tag. The chunk that opens the frame still scans the whole buffer, which then holds at most the start tag and that chunk. Five tests in `ToolCallProcessorLongCallTests`: a 40k-character call streamed one character at a time parses in linear time, with and without declared tools, and the closing tag is found when it is split across chunks, when it arrives in the chunk that completes the start tag, and when another call follows it in the same chunk.

## Checklist

Put an `x` in the boxes that apply.

- [ ] I have read the [CONTRIBUTING](https://github.com/ml-explore/mlx-swift-lm/blob/main/CONTRIBUTING.md) document
- [x] I have run `pre-commit run --all-files` to format my code / installed pre-commit prior to committing changes
- [x] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the necessary documentation (if needed)

## AI usage

- [ ] I have read this PR description in full and approve it as my own, and it
      accurately describes the code changes.
- AI usage disclosure: <fill in>
```

**Status 2026-09-07 (end-tag scan)** — `perf(tools): scan only the chunk
for a collecting call's end tag` (`921c676`) is on `pin-upstream-mlx-swift`
and pushed; tesseract's gitlink moves to it in the same batch. Upstream
filing is deferred by choice, so the text is banked here for whoever posts
it.

*Issue* — "ToolCallProcessor rescans the whole collected call for its end
tag on every chunk". While `state == .collectingToolCall`,
`processTaggedChunk` evaluates `toolCallBuffer.contains(endTag)` per chunk
over the entire buffered call, so a call costs time quadratic in its
length. A tagged `write` call of ~13k tokens streamed one token per chunk
(qwen35 format, `<tool_call>`/`</tool_call>`) spends seconds in that scan
alone, on the thread that drives sampling; the same cost is paid again by
anything that replays a stream. Repro: `ToolCallProcessorLongCallTests`'s
`longCallStreamedByCharacterIsLinear` without the fix.

*PR* — only the text a chunk appends can complete the end tag, because an
earlier occurrence would have left the collecting state when it arrived, so
the scan covers the chunk plus `endTag.count - 1` characters of overlap
with what preceded it. The fall-through from a partial start tag keeps the
whole-buffer scan, which then holds at most the start tag and that chunk.
Four tests: the 40k-character call above, an end tag split across chunks,
an end tag in the chunk that completes the start tag, and a second call
following the end tag in one chunk. Branch it from upstream `main` — the
fork branch of the same name sits on the pin and carries everything else
the pin carries.

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
| [#613](https://github.com/ml-explore/mlx-swift-lm/pull/613) | Streaming detokenizer: emit only the new scalars when a token extends the previous character (issue #612) | **Merged 2026-09-10** (4c3d793) |
| [#610](https://github.com/ml-explore/mlx-swift-lm/pull/610) | ToolCallProcessor: keep the text before a `<` that turns out not to be a tool-call tag (issue #609) | Closed 2026-09-15 — superseded by #548, whose rewrite passes the PR's four tests unchanged |
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
1a. Tag the outgoing pin tip — `git tag -a pin-tip/<date>-<sha> <sha>` and
   push the tag — so it stays fetchable after the force push below.
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
