# Capture handoff baseline — 2026-09-08

Evidence for [#478](https://github.com/spokvulcan/tesseract/issues/478) and [PR #497](https://github.com/spokvulcan/tesseract/pull/497), carried forward to #479/#480.

## Evidence locations

- [Comparison](comparison.md): timing, memory, request IDs and cancellation.
- [Compact baseline](baseline.json): exact summary values and verification outcomes.
- [Environment](environment.json): build hashes, model fingerprint, hardware and settings.
- [Download detailed evidence](https://github.com/spokvulcan/tesseract/releases/download/evidence-478-2026-09-08/capture-handoff-2026-09-08.zip) ([asset page](https://github.com/spokvulcan/tesseract/releases/tag/evidence-478-2026-09-08)): all 32 original evidence files, preserved byte for byte, including 21 diagnostic extracts, full run metadata, seed verification, loaded-model reports, and the original detailed report.

Archive SHA-256: `2944ddec10464fb1ad2d367ff6a2a7dd9079237180293b4a5152164414b00c23` (124,734 bytes). After extraction, start with `capture-handoff-2026-09-08/README.md`. The root `manifest.json` records every source-file hash and the original evidence commit `519e55e91fd14d4b2b8c7560166e4f6b9dc195b0`. Private replay inputs, continuation fixtures, model weights and tensor payloads are excluded.

## What this establishes

Capture already landed in `fc66f599`; the common measurement revision is `cb3c591f`. A frozen control changes only `LeafStorePhase.captureLiveLeaf`'s default to `move: false`. The final production fix uses the selected iterator so loaded but inactive MTP does not force a copy; active MTP keeps copying. MTP was unloaded in both measurement arms.

| Recorded prompt | Capture ms, copy → handoff | Net capture allocation, copy → handoff |
| --- | --- | --- |
| 46,154 tokens | 156.685 → 1.805 | +3.16 GB → slight decrease |
| 76,714 tokens | 242.508 → 2.449 | +5.16 GB → slight decrease |
| 92,777 tokens, matched seed | 1,118.323 → 0.204 | +6.24 GB → no increase |

Every successful handoff reports zero request-owned cache layers. Ownership/address tests establish that this is removal of a copy. Whole-process footprint does not improve uniformly: 46k warm active MLX falls from 29.948 to 27.558 GB while footprint rises from 33.867 to 36.272 GB.

## Reproduction and limits

The workload uses Qwen3.8-27B 4-bit weights, unquantized KV, DFlash2, temperature zero, medium reasoning, preserved thinking and a 128-token output ceiling on an M3 Max with 48 GiB RAM. It is one bounded-output trial per case, not the full historical session. See [the replay workflow](../../../docs/testing.md#controlled-capture-comparison-478) and [collector](../../../scripts/capture_memory_replay.py).

Matching pairs share normalized request hashes, restore offsets, shared-prefix lengths, checkpoint plans and SSD hydration status. The matched 93k pair uses separate clones of the same SSD seed. Original unequal-plan 93k trials and an unpaired warm turn remain in the archive and are excluded from paired claims. The short control overlapped compilation; initial idle gaps and allocator/OS/writer states also differ.

Peaks are sampled lower bounds. `afterRelease` is a scalar observation about one second after request return, not idle memory or an SSD-drain guarantee. Tree/pending fields were measured at `releasingRequest`; snapshot, checkpoint and payload counts overlap and must not be summed. Full payloads alias immutable leaves; extension payloads detach their retained arrays.

Matched 93k handoff diagnostics were recovered across log rotation with 392 continuous memory samples. Response hash, client wall times and a matched warm result are unavailable; token counts are diagnostic-derived. The 77k warm response hashes differ and include generated tool-call IDs. Neither case establishes response parity. Temporary result bundles and raw `/tmp` originals were unavailable after restart; the archive preserves surviving evidence only.

## Follow-up baseline and memory constraint

Checkout still restores by copy: a 46k warm restore adds about 3.19 GB. Boundary checkpoints retain another roughly 3.19 GB there, and 11.61 GB in the matched 93k pair. Their owners are `HTTPPrefixCacheGeneration.transientLastMessageBoundarySnapshot` and `transientLastUserBoundarySnapshot`. #479 should protect existing bodies without copying or retaining stale leases; #480 must remove eligible restore copies and audit checkpoint lifetimes. Full payload eligibility and recurrent rewind-state copies remain explicit constraints.

**Do not automatically repeat the long-context or repeated model-reload experiments on the user's Mac.** A later baseline HTTP rerun stalled during its final reload; the user reported a memory-pressure shutdown/crash. The exact cause was not established from a crash report. Start with the saved evidence and small-cache tests. Any further large-model run requires an explicit owner-approved resource plan and a suitable environment; run one process at a time with a predefined memory stop threshold.

## Verification

- Focused groups: 794 passed, two existing skips, zero failures, 59 suites; inactive-MTP cold/warm regression demonstrated red then green.
- Real recorded prompt: 46,154 tokens; copied and moved leaves independently restored by copy produce identical 496,640 logit bytes, with zero request cache layers after move. Default loaded-model matrix: 12/12 passed.
- Corpus: 85 recordings, 66 clean echo boundaries registered and next-resolved, zero fidelity rejections. Longest path: 92,759 tokens. Eighteen adjacent pairs lacked clean echo extensions.
- Live 2,749-token handoff tail: 38.069 ms. The extra final 18k smoke and final `dev-release` rerun were not run after the crash. The Release build and preceding loaded gates passed; the user subsequently reported successful testing.
- HTTP gate: 31/32. The unchanged baseline reproduced the different-image assertion failure: this checkpoint loads as a text instance, while the harness selects its image scenario from configuration. Real VLM coverage is not established.
- Formatting, SwiftLint, documentation checks and two independent code reviews passed. Detailed results and qualifications are in the archive; no fully green HTTP gate or resolved memory-pressure crash is claimed.
