# Warm Body parity gate — owner run manifest (2026-09-20)

Owner run of the [2026-09-19 pre-registration](../2026-09-19/README.md)
(commit `d770ba4a`) on the owner's 48 GiB M3 Max. The manifest is
[owner-plan.json](owner-plan.json); it is committed before the run and the
runner (`--warm-parity-bench`, commits `2df00ecd` + `57cd12a9`) refuses any manifest that
is not `APPROVED` or whose binary/model checksums differ.

## Thresholds re-derived for 48 GiB

The pre-registration's proposed bounds were written for a ≥96 GiB host.
This host has 48 GiB, the 4-bit Qwen3.8-27B weights are 14.09 GB, and the
fp16 attention body is ~66 KB/token, so the 32k case's body is ~2.2 GB and
the worst arm holds the resident body, its copy or dequantized form, and the
live cache at once (≈3 bodies ≈ 6.6 GB plus generation scratch).

| Bound | Value | Reason |
| --- | --- | --- |
| Minimum initial available | 20 GiB | weights + 3 bodies + headroom |
| Expected model + cache peak | 24 GiB | 14 GB weights + ~7 GB bodies + scratch |
| Footprint stop | 30 GiB | 6 GiB above the expected peak, 18 GiB below RAM |
| Minimum available stop | 6 GiB | free + inactive + speculative + purgeable pages (`host_statistics64`; the free list alone is a few hundred MB on macOS, so the first wrapper revision could not start) |
| Swap growth stop | 1 GiB | as proposed |
| Pressure stop level | 4 (critical) or unknown | warning (2) is recorded, not a stop |
| Request / campaign deadline | 300 s / 14400 s | 84 observations, the 32k prefill ≈ 20 s |

Sampling every 250 ms by `scripts/warm_body_parity.py`; a breach terminates
the harness after a 5 s grace and the campaign is not retried.

## Cases

Four cases from the loaded-model verification workload in the fixed order
`direct-4k`, `view-16k`, `boundary-8k`, `direct-32k`; one warmup block and
the six pre-registered arm orders per case. Tool continuation and
cancel/resend are declared unsupported by this runner and restrict the
claimed coverage. The filler corpus is `CONTEXT.md` at the recorded checksum.

## Results

See the results section below.

## Attempts

- Attempt 1 (manifest commit `3557e960`, runner `2df00ecd`): stopped by the
  owner in the warmup block of the first case. Two instrumentation defects:
  the arm overrides were applied before the cache manager existed (it is
  created on the first request), and the measured leaf was the noise turn's.
  No measured block was reached; the partial output is kept privately as
  `warm-body-parity-2026-09-20-run1-aborted-instrumentation`.
- Attempt 2 (runner `57cd12a9`): the campaign below.

## Results (attempt 2, run 2026-09-20 23:33 → 2026-09-21 01:48 UTC)

Runner commits `2df00ecd` + `57cd12a9`, manifest commit `51c670d4`, pre-registration `d770ba4a`; the Release binary's SHA-256 is the one in [owner-plan.json](owner-plan.json) and was verified by the wrapper before launch. All 4 cases × 6 blocks × 3 arms = 72 observations completed; 1 warmup block per case discarded as pre-registered. No resource stop; no invalid observation.

Campaign-wide sampler (250 ms, 31687 samples, no gap over 0.5 s): peak process footprint 29.11 GB (stop 32.21 GB), minimum available 7.09 GB (stop 6.44 GB), pressure level never above 1 (normal), swap growth 0 B.

The runner's final report step failed after every record had been saved (JSON encoding of a NaN placeholder for the control arm's undefined paired excess, fixed in the runner afterwards); [verdicts.py](verdicts.py) applies the pre-registered rules to the same immediately-saved records and produced [verdicts.json](verdicts.json). Raw records stay private; their SHA-256 are in [raw-record-checksums.txt](raw-record-checksums.txt).

**Verdict: the 8-bit gate FAILS in all four cases. Warm-4 passes only the planned-view case. `warmCompressionEnabled` stays off; #531 does not proceed.**

Greedy-parity failures are deterministic: within each arm the 32-token continuation is byte-identical across all six blocks, and each warm arm diverges from its matched fp16 control at the same token position in every block (direct-4k: third generated token). The quantized-restored KV changes the logits enough to flip a greedy choice on these prompts. Fidelity (Canonical-Echo) had zero mismatches everywhere; it checks token paths, not generated text, and so cannot see this.


### direct-4k

Restore offset 4330 of 4355 prompt tokens (cached in every arm and block). Body bytes at rest: fp16 0.438 GB, warm-8 0.305 GB, warm-4 0.234 GB.

| Arm | Fidelity mismatches / boundaries | Kind coverage | Greedy token mismatches (blocks) | TTFT median / p95 (ms) | Restore median (ms) | Dequantize median (ms) | Paired excess median (ms) | Timing rule | Peak MLX / footprint (GB) | Verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| fp16 copy | 0 / 12 | same | reference | 390.1 / 468.9 | 13.8 | n/a (copy 1.95) | reference | — | 18.59 / 19.54 | REFERENCE |
| warm-8 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 435.9 / 469.9 | 26.5 | 1.80 | 32.4 | fail | 18.43 / 19.40 | FAIL |
| warm-4 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 394.0 / 478.7 | 13.6 | 1.60 | -6.0 | pass | 18.36 / 19.36 | FAIL |

All six samples per arm (ms):

- fp16 copy: TTFT [380.6, 384.4, 468.9, 392.7, 387.6, 398.5]
- warm-8: TTFT [448.2, 409.0, 469.9, 433.1, 388.0, 438.7]; paired excess [67.6, 24.6, 1.0, 40.4, 0.4, 40.2]; dequantize [2.01, 1.92, 1.97, 1.95, 1.84, 1.87, 1.84, 1.82, 1.83, 1.81, 1.82, 1.83, 1.89, 1.75, 1.95, 1.71, 1.73, 1.75, 1.8, 1.77, 1.77, 1.78, 1.78, 1.77, 1.81, 1.8, 1.78, 1.81, 1.8, 1.76, 1.77, 1.77, 1.8, 1.77, 1.77, 1.72] (observer overhead median 0.242 ms)
- warm-4: TTFT [478.7, 403.6, 391.7, 372.0, 396.3, 375.9]; paired excess [98.1, 19.2, -77.2, -20.7, 8.8, -22.5]; dequantize [1.79, 1.66, 1.99, 1.65, 1.73, 1.65, 1.75, 1.62, 1.65, 1.53, 1.73, 1.55, 1.6, 1.64, 1.56, 1.81, 1.6, 1.56, 1.67, 1.63, 1.81, 1.63, 1.61, 1.58, 1.57, 1.68, 1.57, 1.54, 1.54, 1.55, 1.58, 1.54, 1.54, 1.55, 1.57, 1.52] (observer overhead median 0.245 ms)

### view-16k

Restore offset 16443 of 16474 prompt tokens (cached in every arm and block). Body bytes at rest: fp16 1.243 GB, warm-8 0.732 GB, warm-4 0.460 GB.

| Arm | Fidelity mismatches / boundaries | Kind coverage | Greedy token mismatches (blocks) | TTFT median / p95 (ms) | Restore median (ms) | Dequantize median (ms) | Paired excess median (ms) | Timing rule | Peak MLX / footprint (GB) | Verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| fp16 copy | 0 / 6 | same | reference | 522.3 / 545.1 | 27.8 | n/a (copy 6.59) | reference | — | 23.23 / 24.82 | REFERENCE |
| warm-8 | 0 / 6 | same | 0/6 | 524.7 / 544.8 | 33.3 | 8.94 | 9.9 | fail | 22.18 / 23.80 | FAIL |
| warm-4 | 0 / 6 | same | 0/6 | 518.4 / 560.6 | 30.9 | 6.43 | -2.2 | pass | 21.64 / 23.26 | PASS |

All six samples per arm (ms):

- fp16 copy: TTFT [516.6, 508.0, 511.2, 545.1, 528.1, 542.7]
- warm-8: TTFT [520.4, 529.5, 527.3, 522.1, 544.8, 516.0]; paired excess [3.8, 21.5, 16.0, -23.1, 16.7, -26.6]; dequantize [8.89, 8.93, 8.89, 8.86, 8.91, 8.89, 8.95, 8.83, 8.88, 8.82, 8.82, 8.8, 8.83, 8.91, 8.8, 8.76, 8.79, 8.76, 9.05, 9.08, 9.1, 9.01, 8.98, 9.17, 9.01, 8.97, 9.08, 8.98, 9.05, 9.35, 9.01, 9.07, 8.99, 9.0, 8.96, 8.82] (observer overhead median 0.244 ms)
- warm-4: TTFT [510.7, 520.4, 512.6, 560.6, 516.5, 530.9]; paired excess [-5.9, 12.3, 1.4, 15.5, -11.6, -11.8]; dequantize [6.31, 6.4, 6.39, 6.33, 6.37, 6.33, 6.31, 6.32, 6.23, 6.25, 6.47, 6.31, 6.44, 6.41, 6.33, 6.33, 6.31, 6.3, 6.71, 7.05, 6.96, 6.46, 6.67, 7.09, 6.8, 6.48, 6.62, 6.44, 6.61, 6.68, 6.3, 6.39, 6.57, 6.59, 6.55, 6.47] (observer overhead median 0.243 ms)

### boundary-8k

Restore offset 8248 of 8595 prompt tokens (cached in every arm and block). Body bytes at rest: fp16 0.694 GB, warm-8 0.441 GB, warm-4 0.306 GB.

| Arm | Fidelity mismatches / boundaries | Kind coverage | Greedy token mismatches (blocks) | TTFT median / p95 (ms) | Restore median (ms) | Dequantize median (ms) | Paired excess median (ms) | Timing rule | Peak MLX / footprint (GB) | Verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| fp16 copy | 0 / 12 | same | reference | 2060.9 / 2158.0 | 32.1 | n/a (copy 3.52) | reference | — | 19.43 / 20.54 | REFERENCE |
| warm-8 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 2054.9 / 2145.6 | 20.6 | 3.14 | 1.1 | pass | 19.14 / 20.27 | FAIL |
| warm-4 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 2041.5 / 2071.2 | 19.3 | 2.65 | 6.5 | fail | 19.00 / 20.18 | FAIL |

All six samples per arm (ms):

- fp16 copy: TTFT [2064.1, 2014.6, 1999.4, 2070.2, 2158.0, 2057.7]
- warm-8: TTFT [2010.9, 2032.8, 2055.6, 2054.2, 2057.7, 2145.6]; paired excess [-53.2, 18.2, 56.1, -16.0, -100.3, 87.8]; dequantize [3.11, 3.08, 3.35, 2.99, 3.08, 3.01, 3.29, 3.05, 3.16, 3.17, 3.04, 3.5, 3.08, 3.02, 3.12, 3.02, 3.21, 3.04, 3.2, 3.02, 3.25, 3.08, 3.02, 3.15, 3.23, 3.03, 3.19, 3.13, 3.23, 3.17, 3.19, 3.05, 3.24, 3.15, 3.31, 3.29] (observer overhead median 0.234 ms)
- warm-4: TTFT [2010.7, 2051.9, 2024.7, 2071.2, 2031.2, 2069.7]; paired excess [-53.4, 37.3, 25.3, 1.0, -126.8, 12.0]; dequantize [3.0, 2.56, 3.16, 2.55, 2.63, 2.59, 2.67, 2.56, 2.55, 2.67, 2.55, 2.84, 2.6, 2.75, 2.59, 2.58, 2.85, 2.58, 2.58, 3.04, 2.6, 2.58, 2.7, 2.56, 2.62, 2.74, 2.64, 2.59, 2.76, 2.71, 2.96, 2.82, 2.68, 2.87, 2.66, 2.87] (observer overhead median 0.232 ms)

### direct-32k

Restore offset 33001 of 33026 prompt tokens (cached in every arm and block). Body bytes at rest: fp16 2.317 GB, warm-8 1.303 GB, warm-4 0.762 GB.

| Arm | Fidelity mismatches / boundaries | Kind coverage | Greedy token mismatches (blocks) | TTFT median / p95 (ms) | Restore median (ms) | Dequantize median (ms) | Paired excess median (ms) | Timing rule | Peak MLX / footprint (GB) | Verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| fp16 copy | 0 / 12 | same | reference | 696.9 / 714.7 | 86.7 | n/a (copy 15.53) | reference | — | 26.35 / 28.64 | REFERENCE |
| warm-8 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 667.8 / 676.5 | 65.3 | 12.67 | -29.3 | pass | 25.31 / 27.36 | FAIL |
| warm-4 | 0 / 12 | same | 6/6 [0, 1, 2, 3, 4, 5] | 662.4 / 681.5 | 60.9 | 11.50 | -36.8 | pass | 24.77 / 26.88 | FAIL |

All six samples per arm (ms):

- fp16 copy: TTFT [645.1, 679.8, 714.7, 667.6, 714.5, 714.1]
- warm-8: TTFT [670.9, 664.8, 660.7, 676.5, 670.8, 648.3]; paired excess [25.8, -14.9, -54.0, 8.9, -43.7, -65.8]; dequantize [16.3, 14.57, 13.17, 11.78, 10.83, 10.82, 17.05, 15.14, 13.18, 11.96, 10.94, 10.97, 17.29, 14.67, 13.48, 12.07, 10.91, 10.97, 16.58, 14.46, 13.77, 12.28, 10.94, 11.19, 16.57, 14.44, 13.31, 12.18, 10.94, 10.97, 15.99, 14.24, 13.07, 11.85, 10.97, 11.25] (observer overhead median 0.233 ms)
- warm-4: TTFT [681.5, 649.6, 664.2, 660.7, 671.0, 654.8]; paired excess [36.3, -30.1, -50.5, -6.9, -43.4, -59.2]; dequantize [16.08, 14.09, 12.15, 11.2, 9.85, 9.52, 15.82, 13.96, 12.46, 10.95, 9.63, 9.97, 15.43, 13.97, 12.98, 11.03, 9.47, 10.25, 16.36, 13.89, 12.83, 10.84, 9.35, 9.67, 15.39, 13.45, 13.27, 10.83, 9.38, 10.18, 15.9, 13.43, 11.8, 11.09, 9.84, 9.41] (observer overhead median 0.23 ms)

## Notes on the record

- The fp16 control's Leaf Checkout was refused with `copyReason=checkoutDisabled` on every timed hit (restore by copy), warm hits restored with `source=warm`/`copyReason=warmBody`, and the view case restored `source=view` with `backingLeafForm=warm|ownedBody` per arm; the runner invalidates any other path and none occurred.
- Every case on this template (Qwen3.8, thinking stripped from history) takes the boundary leaf-store path, so the direct cases also exercise the transient boundary view at the end of the timed turn; the boundary-8k case additionally renders with `reasoning_effort=low`, which keeps ~347 tokens of visible answer in history and makes its TTFT prefill-dominated (~2 s) in all arms alike.
- In the planned-view case the timed request carries the fork's user message and the timed user message back to back; the restore is the view at the shared prefix (offset 16443) through its Backing Leaf. Its dequantization allowance was timed on the freshest fork leaf sliced to the view offset (same layers and offset as the Backing Leaf's prefix).
- Tool continuation and cancel/resend were declared unsupported in the manifest; the result claims no coverage of them.
- Warm-4 is experimental (option consideration only). Its single pass (view-16k) does not offset its three failures and validates nothing for warm-8.
