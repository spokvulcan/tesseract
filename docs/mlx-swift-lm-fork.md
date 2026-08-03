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

## Current pin (2026-07-27)

Base: upstream `main` @ `3cbf928` — the 2026-07-23 re-pin's base (`eaefe75`:
Qwen3.5 interleaved M-RoPE #442, Qwen3VL per-image fused SDPA #455, TurboQuant
KV cache #232, Gemma 4 MTP speculative decoding #415, tool-schema `$defs`
hoisting #434) plus seven newer commits: end-to-end video input for Gemma 4
#391, Hunyuan dense V1 #347, DeepSeek-V2 #379, a DeepSeek-V3 kvHeads / KV-cache
fix #457, multi-round tool calling in MLXFoundationModels #456, and two CI
changes (#458, #464).

The previous tip (`8519cf3`) is preserved as
`pin-2026-07-23-upstream-eaefe75`.

This re-pin also **collapses the eleven per-experiment `chore: pin mlx-swift to
<rev>` bookkeeping commits into the single pin declaration** below — they were
lockstep noise, and the pin is a single fact.

mlx-core stays at v0.31.1: the move to upstream mlx main is built and green on
`pin-tesseract-2026-07-27` in both mlx forks, but is blocked on thread-local
command encoders. Full diagnosis and re-attempt checklist:
`docs/mlx-core-fork.md`.

Carried on top, in order:

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `fix: pin mlx-swift to the spokvulcan fork at 457a0d6d` | Exact-revision pin on `spokvulcan/mlx-swift` `pin-tesseract` (0.31.6 base + .gitmodules provenance + the Cmlx gitlink bumps carrying C1/C4–C9/C13). SwiftPM cannot mix revision and version requirements for one package, so this must match mlx-audio-swift and tesseract-speech exactly | Permanent local; never upstream |
| `fix(paroquant): convert every AWQ prefix and cast scales to f16` | AWQ→PARO conversion correctness | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `refactor(paroquant): extract PairwiseRotation from RotateQuantizedLinear` | Shared rotation core for the MoE path | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `feat(paroquant): MoE PARO path — RotateSwitchGLU + loader passes` | PARO quantization for MoE models (Qwen3.6-35B-A3B) | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `perf(prefill): balance the prompt chunks instead of leaving a remainder` | Equal prefill chunks; kills the degenerate remainder forward (~9% prefill, tesseract #258) | **Filed as [#470](https://github.com/ml-explore/mlx-swift-lm/pull/470)** (2026-07-26) |
| `feat(paroquant): Prepared Checkpoint + O(1) AWQ conversion matching` | Prepared Checkpoint artifact + O(1) matcher (ADR-0032) | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `feat(models): Nanbeige looped-transformer support` | Nanbeige4.2 model (`nanbeige`): shared-weight layer loops, per-loop KV caches, xmlFunction tool calls, `<think>` reasoning config | **Filed as [#460](https://github.com/ml-explore/mlx-swift-lm/pull/460)** (2026-07-23, branch `feat/nanbeige-looped-transformer` — cherry-pick on upstream `main` @ `1032402`); Python-side counterpart is MercuriusDream/mlx-lm `add-nanbeige-model` |
| `perf(paroquant): rotate gate_up before the MoE expert gather/sort` | Rotate L token rows pre-gather instead of L×topK rows post-gather (bitwise-identical); +3–4.5% MoE prefill at 8K–32K (tesseract experiments-ledger E1) | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `perf(paroquant): compile-fuse the GatedDelta decay gate chain` | One compiled kernel for the 6-kernel elementwise g chain per GDN layer per step (bitwise-identical); +3.1% MoE decode, +1.4% dense decode at ctx=128 (tesseract experiments-ledger E2) | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26) |
| `perf(paroquant): simdgroup-resident rotation kernel — no CTA barriers` | 32-lane simdgroup CTAs, compile-time krot, row-major tile, float4 IO for groupSize 128; generic pre-E6b kernel restored as the fallback for other group sizes (shared `dispatchPairwiseRotation`); bitwise-identical; kernel 1.7–2× at prefill shapes; +1.8–2.5% MoE prefill, +1.3–2.1% dense prefill, +3.4–5% dense decode (tesseract experiments-ledger E6b) | Filed in [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) (2026-07-26; `017086e` fallback restore included) |
| `perf(qwen35): compile the MoE block during decode (C11)` | Per-instance compiled MoE block closure, decode (L==1) only; router/shared-expert/residual elementwise fuse; +3–7% MoE decode (tesseract experiments-ledger C11) | Filed in [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467) (2026-07-26) |
| `perf(qwen35): compile the GDN decode step with explicit state (C12)` | Compiled GDN decode step, conv/recurrent state as explicit I/O; +1.75% dense 128 decode, +0.94% MoE (ledger C12). The module-local compiled wrapper was subsumed by C14 and removed in the PR #427 review round; the compiled body (`decodeForward`) is the surviving artifact | Filed in [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467) (2026-07-26) |
| `fix(qwen35): unowned captures for the compiled decode closures` | Breaks the module→closure→module retain cycle C11/C12 shipped — the cycle leaked each block, its weights, and the compiled mlx tape on every model release; + `Qwen35CompiledDecodeLifecycleTests` red/green regression test (2026-07-24 review round, tesseract PR #425 follow-up) | Filed in [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467) (2026-07-26) |
| `perf(qwen35): whole-step compiled decode schedule (C14)` | The whole decode step as traced segments (11 MoE / 9 dense), split only where the KV cache is written, rather than one compiled closure per block; +1.33% MoE 128 decode, +0.67% MoE 8K, +1.73% dense 8K (ledger C14) | Filed in [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467) (2026-07-26) |
| `perf(qwen35): C16 — decode conv1d as fused multiply-adds` | At S==1 the GDN depthwise conv is a fixed 4-term dot per channel; written as elementwise multiply-adds it folds into the surrounding compiled segment, deleting a dispatch *and* a hazard barrier per GDN layer. Bitwise only with **f32 accumulation** — native-dtype accumulation differs in ~47% of channels. 2026-07-29 addendum: for **f32 input** the FMA does NOT match the kernel either (102/256 channels — its f32 path accumulates in a different order); upstream #468 gates the fused branch to f16/bf16, this Vendor form is ungated. +1.77% median MoE 8K decode (ledger C16) | Filed as [#468](https://github.com/ml-explore/mlx-swift-lm/pull/468) (2026-07-26, stacked on #467); **merged 2026-07-30** (0321f28) in the f16/bf16-gated form — drop this cherry-pick at the next re-pin |
| `perf(qwen35): C18 — fused router top-k kernel` | `ArgPartition::eval_gpu` delegates to `gpu_merge_sort`, so the router fully sorted 256 experts to name 8; one custom kernel replaces the sort and the gather/sum/divide tail. Bit-identical by construction (the sort is stable, so the selection order is reproducible; the 8-wide reduce accumulates sequentially in the output dtype). Decode only — prefill keeps the block sort. +1.91% MoE 128 decode (ledger C18) | Filed as [#469](https://github.com/ml-explore/mlx-swift-lm/pull/469) (2026-07-26; MLXVLM copy flagged in the PR) |
| `fix(qwen35): PR #427 review round — uint32 router indices, dead C12 wrapper, bitwise-contract tests` | C18 kernel emits `uint32` indices (argPartition's dtype); removes the C12 wrapper C14 obsoleted; adds `Qwen35BitwiseContractTests` (C16 conv contract + C18 router contract, NaN ordering included) and a quantized-cache lifecycle variant; also carried an MLXFast product dep — self-inflicted by C18's `import MLXFast` and removed in the #469 review round 2026-07-29 (the `MLXFast` enum in MLX covers the symbols; the standalone module is a deprecated shim) | Filed 2026-07-26, split across [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467)/[#468](https://github.com/ml-explore/mlx-swift-lm/pull/468)/[#469](https://github.com/ml-explore/mlx-swift-lm/pull/469) |
| `feat(tokenizers): ChatTemplateRendering protocol + adaptor forwarding (C25)` | Exposes the render half of `applyChatTemplate` at the MLXLMCommon layer (new `ChatTemplateRendering` protocol; the macro-generated bridge forwards, same `missingChatTemplate` mapping). Enables tesseract's render+token cache (experiments-ledger C25 — tokenize only a verified suffix of a previously rendered prompt). Requires `renderChatTemplate` on the swift-transformers side — `spokvulcan/swift-transformers` `pin-tesseract` @ `63edf42` (scheme: `docs/swift-transformers-fork.md`) | Not filed (queued — owner go-ahead) |

Earlier pin branches carried one `chore: pin mlx-swift to <rev>` commit per
accepted Cmlx experiment (C4–C13 and the 2026-07-24 review round). That
lockstep bookkeeping is collapsed into the single pin commit above as of the
2026-07-27 re-pin; `pin-2026-07-23-upstream-eaefe75` still has the long form.

## Upstream filing queue (2026-07-25)

The 2026-07-18 → 07-25 inference-perf loop is closed (ledger: C1–C19 run,
decode/prefill program re-priced and exhausted under the zero-output-change
constraint), so the accepted survivors are ready to shape into upstream
PRs. Four units, each self-contained and parity-evidence-backed; file
against `ml-explore/mlx-swift-lm` `main` with the #460 cherry-pick
procedure. Owner go-ahead is the trigger — nothing below is filed yet.

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

1. **Compiled decode schedule for Qwen3.5/3.6** — C11 + C12 + the unowned-
   captures fix + C14 + the PR #427 review-round commit, squashed into one
   PR: per-layer decode traces, the whole-step segment schedule split at
   the KV write, the MoE block closure for non-plain cache kinds, and both
   test files (`Qwen35CompiledDecodeLifecycleTests`,
   `Qwen35BitwiseContractTests`). Quantization-agnostic — none of it
   depends on PARO. Evidence: +3–7% MoE decode (C11), +1.75% dense (C12),
   +1.33%/+0.67%/+1.73% on top (C14), 108 A/B pairs token-identical.
2. **GDN decode conv1d as fused multiply-adds (C16)** — depends on unit 1
   (the FMA form pays by folding into the compiled segment, and lives in
   `decodeForward`). Small diff; general to every Qwen3Next-family GDN
   model upstream ships. Evidence: bitwise gate (f32 accumulation, 8192
   channels, f16+bf16) now CI-pinned by the contract tests; +1.77% median
   MoE 8K decode.
3. **Fused router top-k kernel (C18)** — independent of units 1–2
   (`routerTopK` works in the uncompiled block too). Offer upstream the
   generalization to every `SwitchGLU` router, or Qwen3.5-only first with
   the MLXVLM copy flagged as the follow-up. Evidence: bit-identical by
   construction with the contract tests as proof, +1.91% MoE 128 decode
   (10/10 pairs).
4. **PARO batch** — the existing #164 follow-up: AWQ conversion fixes,
   `PairwiseRotation` extraction, the MoE PARO path, Prepared Checkpoint,
   E1 (pre-gather rotation, +3–4.5% MoE prefill), E2 (decay-gate compile
   fuse, +3.1% MoE decode), E6b (simdgroup-resident rotation kernel,
   +1.8–5% by shape). One batched PR, queued since the 2026-07-23 review
   round (tesseract PR #424).

Also carried but filed separately when touched next: `perf(prefill):
balance the prompt chunks` (standalone, model-agnostic, ~9% prefill).
The mlx-core-side wins from the same loop (C4 caps, C9, C13 fused
causal-softmax, gather identity-index cache, …) are tracked in
`docs/mlx-core-fork.md`, and the two ripe evidence-backed `ml-explore/mlx`
issues (M1 tile geometry, M2 command-buffer segmentation) in the section
below — different upstream, different queue.

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
| [#470](https://github.com/ml-explore/mlx-swift-lm/pull/470) | Balanced prompt chunking (~9% prefill) | Filed 2026-07-26; 2026-07-31 rebased over the #448 Qwen2/2.5-VL windowed prefill (their continuations adopt PrefillParameters/forEachChunk) — MERGEABLE, awaiting review |
| [#471](https://github.com/ml-explore/mlx-swift-lm/pull/471) | ParoQuant MoE batch: MoE path, Prepared Checkpoint, E1/E2/E6b (#164 follow-up) | Filed 2026-07-26 |

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
