# ADR-0036: TTS v2 inference stands on a re-ported vendor of upstream mlx-audio-swift

- Status: Accepted
- Date: 2026-07-13
- Relates to: map #334 (Voice engine v2), tickets #335 (field survey), #336 (model family), #337 (re-port spike), #338 (budgets), #339 (benchmarks), #342 (v1 autopsy), #346 (speech-swift, out of scope); ADR-0003 (speech seams), ADR-0005 (vendor as frontier-experimentation surface), ADR-0006 (mlx-swift-lm fork migration)

## Context

Voice engine v2 needs an inference substrate. The vendored `mlx-audio-swift` snapshot (cut 2026-02-06 from the PR-era branch) is dead on arrival for v2: it cannot load quantized checkpoints at all (autopsy D3, `Qwen3TTSModel.swift:1254-1331` — no `quantize(model:)` path), while the #338 budgets force all-quantized shipping (bf16 breaks the ≤ 3 GB envelope at 4.7 GB RSS, #339). Meanwhile upstream `Blaizzy/mlx-audio-swift` rewrote the Qwen3-TTS internals since our cut — any sync is a re-port, not a rebase (#335).

Three candidates were weighed against seven acceptance criteria (all evidence-backed: #338 budgets on the M3 Max baseline, quantized loading, full v1 parity surface, quality ≥ v1 by ear per #339's three-layer method, compatibility with the autopsy's ten boundary constraints, bounded maintenance with a written sync policy, measured effort):

1. **Upstream via remote SPM** — killed by the #337 spike on hard facts: the five vendor-only features (voice anchor, word alignment, seed, instant-cancel parity, first-chunk cadence) require source patches, and upstream's version-range requirements cannot coexist with our revision-pinned dependency graph (SwiftPM: one requirement kind per package identity).
2. **In-house Qwen3-TTS inference inside Tesseract** — the charter's stated lean; rejected on evidence. It passes none of the seven criteria yet and would re-derive what two active codebases maintain: the stateful streaming decoder (upstream #70/#82), quantized loading (#61), two perf passes (#81/#82), cancel-on-drop (#157) — weeks to reach the starting line the re-port already stands on, then permanent solo maintenance of a moving model family (~200 upstream PRs / 5 months). The lean's real content — ownership — is delivered by re-vendoring anyway: the source lives in our tree, builds against our vendored `mlx-swift-lm` fork, and upstream cannot break us; if upstream dies, the fallback is absorption of code we already ship, not a rewrite.
3. **Re-ported vendor** — re-vendor current upstream and apply the spike's feature port. Passes all seven criteria, measured: every #338 budget passes (warm TTFA 104–310 ms, sustained RTF 0.23–0.39; 1.6× vendor at bf16, ~2.5× quantized — #339), quantized loading works today, all five vendor-only features fit at +283/−5 LOC across 3 files, runtime-smoke-tested with real VoiceDesign weights (#337), 8-bit ≈ bf16 by ear (#339), and the effort is a measured 2.5–3.5 focused days.

`soniqo/speech-swift` as a dependency was separately ruled out of scope (#346): we won't pull a whole library to run one model. It stays a read-only design/performance reference.

## Decision

v2's inference stands on the **re-ported vendor**: upstream `Blaizzy/mlx-audio-swift` at tag **v0.1.3 (`d302a5c`, 2026-07-09)** — byte-identical to the revision the #337 spike built and smoke-tested against — re-vendored as a **full 1:1 tree** at `Vendor/mlx-audio-swift`, carrying the spike's ~283 LOC feature port.

Shape decisions:

- **Full tree, not a slimmed subset.** With a 1:1 tree the delta against the upstream tag is exactly our patches, so every future re-port starts from a clean three-way diff. A subset would make the delta "patches + deletions" and upstream has already shown it moves files around. The unused targets cost only checked-in source text, because—
- **The app links the `MLXAudioTTS` product**, not the legacy combined `MLXAudio` product it links today (`project.pbxproj:788`), so the unused STT/STS/UI targets never build.
- **Location stays `Vendor/mlx-audio-swift`** — the established convention alongside `mlx-swift-lm`, `mlx-image-swift`, `tesseract-highlight`.

Maintenance policy (the "bounded maintenance" criterion made concrete):

1. **Pin to tags, never `main`** — upstream issue #120 promises a TTS loading-layer refactor; main is churn by announcement. Provenance (repo, tag, SHA, date) recorded in the vendor tree.
2. **Committed delta ledger** — every divergence from the tag lives in `Vendor/mlx-audio-swift/TESSERACT-PATCHES.md`: what changed, why, which seam. A divergence not in the ledger is a bug. (The spike's patch files live in git-ignored `research/` and cannot serve.)
3. **Sync on need, never on cadence** — a re-port happens only when a concrete upstream change earns it (new Qwen-family checkpoint support, a perf pass we want, a bug we've hit). Every sync costs a re-port of the ledger plus listening validation under #339's method; skipping releases is the default posture.
4. **Upstream the portable patches, opportunistically, after v2 ships** — warm-cache mask fix first (a genuine upstream bug, load-bearing for the voice anchor, silently regressable), then seed, `tokenizeForAlignment`, cancel parity — shrinking the permanent delta from ~283 toward ~200 LOC. Independent of acceptance, every load-bearing patch gets a pinning test in our tree so a future re-port cannot silently drop it.

## Consequences

- The v2 API boundary (#343) designs against upstream's rewritten internals: stateful streaming decoder (continuous by construction — vendor's segmentation/crossfade machinery is obsolete), cancellation surfacing as `CancellationError`, quantization as a load parameter.
- The old vendor snapshot is replaced wholesale; nothing in it survives except via the ported five-feature surface.
- **Accepted one-time caveat: the seed re-roll.** Upstream's sampler differs, so a voice tied to an old seed re-rolls once at migration. Accepted without mitigation — #339 established a seed doesn't pin a voice across checkpoints or run lengths anyway; seed semantics (reproducibility knob vs voice identity) get settled in #343.
- The "if in-house wins" questions (porting scope, package layout, maintenance policy) are answered for the winning option by this ADR; in-house remains available later only as absorption of the re-vendored tree, not as a fresh effort premise.

## Accepted costs

- Three-way pin coupling (mlx-audio patch ↔ vendored `mlx-swift-lm` fork ↔ mlx-swift revision `dc43e62d` = tag 0.31.4) must move together on any sync, against a fast-moving upstream.
- The warm-cache mask fix is Tesseract-only until upstreamed — the pinning test is the guard.
- Unused upstream source (STT/STS/UI, other TTS models, codec families) is checked in but never built.
