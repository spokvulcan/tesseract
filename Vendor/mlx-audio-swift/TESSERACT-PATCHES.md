# Tesseract patch ledger — mlx-audio-swift

**Rule: a divergence from the upstream tag that is not in this ledger is a bug.**
(ADR-0036. Sync policy: tags only, on need, never cadence. Re-port = re-apply this
ledger onto the new tag, then re-run the pinning tests and the #339 bench gates.)

## Provenance

- Upstream: `https://github.com/Blaizzy/mlx-audio-swift`
- Base: tag **v0.1.3** = `d302a5c6080d2bb97bae38c7418f82abb76013b6` (2026-07-09)
- Re-vendored: 2026-07-13 (`feat/voice-engine-v2`), from the #337 spike patches
  (`research/re-port-spike-337/patches/`) + #339 bench patches
  (`research/model-bench-339/patches/`), scratch-verified: full release build green,
  runtime-smoke-tested with real VoiceDesign weights before the swap.

## Divergences

| # | What | Where | Why | Seam |
|---|------|-------|-----|------|
| 1 | `seed` — deterministic generation (property + capture in stream task + seeding before loop; protocol surface) | `Sources/MLXAudioTTS/Generation.swift`, `Models/Qwen3TTS/Qwen3TTS.swift` | v1 parity; reproducibility knob at the v2 boundary (ADR-0038: never voice identity). Same seed → bit-identical samples. **Upstream candidate.** | spike `0945e21` |
| 2 | Two-phase first-chunk cadence (emit after 5 tokens with small window until 48 frames, then 8) | `Models/Qwen3TTS/Qwen3TTS.swift` | Upstream default `streamingInterval: 2.0` ≈ 25 tokens — worse first-audio. ~8 LOC cadence switch; the vendor's crossfade/segmentation machinery is NOT ported (obsoleted by upstream's stateful streaming decoder). **Upstream candidate.** | spike `0945e21` |
| 3 | `tokenizeForAlignment` — per-token char offsets for the word-highlight timeline | `Models/Qwen3TTS/Qwen3TTS.swift`, protocol in `Generation.swift` | v1-unique K1 path; one-token-per-frame invariant verified to hold upstream. Verbatim 13 LOC. **Upstream candidate.** (Perf note: quadratic; v2 perf pass may replace with incremental equivalent under an equality pinning test — update this row if so.) | spike `0945e21` |
| 4 | `cancelGeneration()` 4-line API-parity shim | `Models/Qwen3TTS/Qwen3TTS.swift` | Upstream #157 cancels natively (CancellationError per step); shim keeps v1 call sites compiling. Delete when engine v2 wiring removes the last call site — update this row. | spike `0945e21` |
| 5 | Voice anchor — 48-step codec-KV prefix capture/restore (fused-prompt split, `[any KVCache]` state save/restore, anchored generation) | `Models/Qwen3TTS/Qwen3TTS.swift` (~150 LOC, 3 seams) | v1-unique K3 machinery: long-form + companion voice consistency (ADR-0037/0038). | spike `0945e21` |
| 6 | **Offset-aware causal mask** for warm-cache multi-token forwards — `createCausalMask(n:offset:)` | `Models/Qwen3TTS/Qwen3TTSTalker.swift` | **Genuine upstream bug fix, load-bearing for #5**: upstream's mask ignores cache offset, breaking any warm-cache multi-token forward (broadcast error). **Upstream first in priority** — silently regressable otherwise. Pinning test required. | spike `d1d5eab` |
| 7 | Dependency block: mlx-swift pinned `dc43e62d` (= tag 0.31.4), `mlx-swift-lm` → `.package(path: "../mlx-swift-lm")` (Tesseract's vendored fork), `swift-huggingface` floor 0.8.1 | `Package.swift`, `Package.resolved` | SwiftPM can't mix revision pins and version ranges for one identity; the app's graph pins by revision. Three-way pin coupling documented in ADR-0036 accepted costs. | spike `0af3270` + relative-path fix |
| 8 | `spike-smoke` tool — runtime smoke harness (seed determinism, alignment, anchor) | `Sources/Tools/spike-smoke/` | Tesseract-only verification tooling; runs the #337 smoke suite against real weights. Not linked by the app. | spike `f20e50c` |
| 9 | `bench-339` tool + simcheck modes — TTFA/RTF/memory matrix, machine sanity gates, speaker-similarity tripwire | `Sources/Tools/bench-339/` (per bench patches) | The permanent #339/#338 measurement harness; feeds the ADR-0037 precision gate and every future re-vendor's budget scorecard. Not linked by the app. | bench `0001`+`0002` |
| 10 | `lastGeneratedCodeFrames` getter + `buildVoiceAnchor(fromCodeFrames:...)` overload | `Models/Qwen3TTS/Qwen3TTS.swift` | Anchor code frames as plain values: enables `PinnedVoice` (ADR-0038) — voice identity serialized across relaunch, anchors rebuilt without regenerating source audio. ~25 LOC additive. | v2 engine (2026-07-13) |
| 11 | Model store location: `Application Support/models/<repo>` + public `storageDirectoryName` | `Sources/MLXAudioCore/ModelUtils.swift` | Upstream stores snapshots under the Hub cache root — a purgeable Caches dir in a sandboxed app — which would orphan users' existing multi-GB downloads and split storage from the app's Models page (`ModelDownloadManager.modelStorageURL`). Restores the pre-v0.1.3 port layout. NOT an upstream candidate (app-specific policy). | v2 wiring (2026-07-13) |
| 12 | In-loop `Memory.clearCache()` (every 50 steps) removed from the generation loop | `Models/Qwen3TTS/Qwen3TTS.swift` | Perf pass (spec §6): per-frame cost degraded monotonically over a 4-segment long-form run (24.0→29.8 ms/frame) — allocator churn from dropping the buffer pool mid-generation. Removal: RTF 0.320→0.308, peak RSS flat (2.85→2.87 GB), output bit-identical at fixed seed (cmp). The v2 engine trims once per utterance end instead (ADR-0039). **Upstream candidate** (as a `clearCacheInterval: nil` default or removal). | perf pass (2026-07-13) |

## Upstreaming queue (ADR-0036 §4, post-v2, opportunistic)

1. #6 mask fix (bug), 2. #1 seed, 3. #3 alignment, 4. #2 cadence (as opt-in preset).
Success shrinks the permanent delta toward ~#5 + #7 only.
