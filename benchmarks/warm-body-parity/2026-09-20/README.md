# Warm Body parity gate — owner run manifest (2026-09-20)

Owner run of the [2026-09-19 pre-registration](../2026-09-19/README.md)
(commit `d770ba4a`) on the owner's 48 GiB M3 Max. The manifest is
[owner-plan.json](owner-plan.json); it is committed before the run and the
runner (`--warm-parity-bench`, commit `2df00ecd`) refuses any manifest that
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
| Minimum available stop | 6 GiB | free + speculative + purgeable pages |
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

Filled in after the run; see the results section appended below.
