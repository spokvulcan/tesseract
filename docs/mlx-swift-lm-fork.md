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
`pin-gemma4-12b-358`, …) are kept so historical tesseract commits' gitlinks
stay reachable — never delete them, and never force-push a branch an old
gitlink points into without checking reachability. `pin-gemma4-12b-358` is
the parked Gemma 4 12B multimodal stack (audio encoder + encoder-free
`gemma4_unified` processor + suppress_tokens) that tesseract draft PR #359
pins; it rejoins this table's carry list only if that experiment is revived.

## Current pin (2026-07-23)

Base: upstream `main` @ `eaefe75` (adds Qwen3.5 interleaved M-RoPE
optimization #442, Qwen3VL per-image fused SDPA #455, TurboQuant KV cache
#232, Gemma 4 MTP speculative decoding #415, tool-schema `$defs` hoisting
#434). Carried on top, in order:

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `fix: pin upstream ml-explore/mlx-swift at 0.31.6; drop retained-CB fork` | Exact-revision mlx-swift pin, matching mlx-audio-swift — SwiftPM cannot mix revision and version requirements for one package | Permanent local; never upstream |
| `fix: pin mlx-swift to the spokvulcan fork (Cmlx experiment loop)` | mlx-swift pin moves to `spokvulcan/mlx-swift` @ `54ca1ec` (upstream 0bb916c + .gitmodules provenance only) so Cmlx is writable via `spokvulcan/mlx` — scheme: `docs/mlx-core-fork.md` | Permanent local; never upstream |
| `fix(paroquant): convert every AWQ prefix and cast scales to f16` | AWQ→PARO conversion correctness | Not filed — candidate follow-up to #164 |
| `refactor(paroquant): extract PairwiseRotation from RotateQuantizedLinear` | Shared rotation core for the MoE path | Not filed — candidate (prerequisite of the MoE commit) |
| `feat(paroquant): MoE PARO path — RotateSwitchGLU + loader passes` | PARO quantization for MoE models (Qwen3.6-35B-A3B) | Not filed — candidate follow-up to #164 |
| `perf(prefill): balance the prompt chunks instead of leaving a remainder` | Equal prefill chunks; kills the degenerate remainder forward (~9% prefill, tesseract #258) | Not filed — candidate |
| `feat(paroquant): Prepared Checkpoint + O(1) AWQ conversion matching` | Prepared Checkpoint artifact + O(1) matcher (ADR-0032) | Not filed — candidate follow-up to #164 |
| `feat(models): Nanbeige looped-transformer support` | Nanbeige4.2 model (`nanbeige`): shared-weight layer loops, per-loop KV caches, xmlFunction tool calls, `<think>` reasoning config | **Filed as [#460](https://github.com/ml-explore/mlx-swift-lm/pull/460)** (2026-07-23, branch `feat/nanbeige-looped-transformer` — cherry-pick on upstream `main` @ `1032402`); Python-side counterpart is MercuriusDream/mlx-lm `add-nanbeige-model` |
| `perf(paroquant): rotate gate_up before the MoE expert gather/sort` | Rotate L token rows pre-gather instead of L×topK rows post-gather (bitwise-identical); +3–4.5% MoE prefill at 8K–32K (tesseract experiments-ledger E1) | Not filed — candidate (fold into the MoE PARO commit when #164-follow-up opens) |
| `perf(paroquant): compile-fuse the GatedDelta decay gate chain` | One compiled kernel for the 6-kernel elementwise g chain per GDN layer per step (bitwise-identical); +3.1% MoE decode, +1.4% dense decode at ctx=128 (tesseract experiments-ledger E2) | Not filed — candidate (general to all GDN models, e.g. Qwen3Next) |
| `perf(paroquant): simdgroup-resident rotation kernel — no CTA barriers` | 32-lane simdgroup CTAs, compile-time krot, row-major tile, float4 IO for groupSize 128; generic pre-E6b kernel restored as the fallback for other group sizes (shared `dispatchPairwiseRotation`); bitwise-identical; kernel 1.7–2× at prefill shapes; +1.8–2.5% MoE prefill, +1.3–2.1% dense prefill, +3.4–5% dense decode (tesseract experiments-ledger E6b) | Not filed — candidate (fold into #164 follow-up; also fixes the latent bf16 compile failure) |

The three perf carries above (E1/E2/E6b) are queued for one batched
upstream PR folded into the #164 follow-up; filing deferred pending owner
go-ahead (2026-07-23 review round, tesseract PR #424).

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
| [#460](https://github.com/ml-explore/mlx-swift-lm/pull/460) | Nanbeige4.2 looped-transformer model support | Filed 2026-07-23 |

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
