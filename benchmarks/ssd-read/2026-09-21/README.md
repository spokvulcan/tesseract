# SSD read-path experiment — owner run (#532), 2026-09-21

Executed under the owner's standing approval (session 2026-09-20) with the
protocol in [`../2026-09-19/README.md`](../2026-09-19/README.md), unchanged.
Plan: `plan.json` (committed as `440acfca` before the run; app commit
`5c08817d` code, Release binary SHA-256 `6947d30f…`, vendor `a3c1776c`).
Host: Apple M3 Max, 48 GiB, macOS 27.0. Model: Qwen3.8-27B-4bit
(`qwen3.8-27b`). Raw records: `~/bench-results/ssd-read-2026-09-21-run1`
(`raw-record-checksums.txt`); `records.json` and the harness's own
`harness-README.md` are copied here verbatim.

## The chain

One exact Segment Chain exported read-only from the app's SSD tier after the
#534 verification run: head `A5F80440…`, token offset 60,906, 5 files,
4,761,977,844 file bytes, 4,145,479,680 materialized bytes (superseded
whole-state blobs count in file bytes only). Per-file SHA-256 in the plan;
the wrapper verified them before launch. Model fingerprint matched the
partition.

## Bounds (48 GiB host) and resource record

| Bound | Value | Observed |
| --- | --- | --- |
| `maxSegmentBytes` / `maxMLXBytes` (harness) | 5 GiB / 32 GiB | peak MLX 22.03 GB every block |
| Footprint stop | 34 GiB | max 26.89 GB |
| Minimum available stop | 6 GiB | min 7.48 GiB |
| Swap growth stop | 1 GiB | 139 MB |
| Pressure stop | level 4 | max 2 |
| Campaign deadline | 1800 s | 70.8 s |

No resource stop; exit 0; 18 measured hydrations plus 3 warmups; all 18
snapshot digests identical across arms.

## Results

Throughput = materialized bytes / whole-hydration seconds / 1e9, OS page
cache warm (as the protocol declares), six balanced blocks ABC, BCA, CAB,
CBA, BAC, ACB.

| Arm | Block 0 | 1 | 2 | 3 | 4 | 5 | **Median GB/s** | Ratio to mapped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mapped | 3.088 | 4.343 | 4.845 | 5.712 | 5.875 | 5.412 | **5.128** | 1.000 |
| sequentialMap | 5.531 | 4.342 | 5.665 | 5.659 | 5.875 | 5.497 | **5.595** | 1.091 |
| positional | 3.446 | 3.613 | 3.595 | 3.914 | 4.093 | 3.863 | **3.738** | 0.729 |

Hydration seconds: mapped 1.342 / 0.955 / 0.856 / 0.726 / 0.706 / 0.766;
sequentialMap 0.749 / 0.955 / 0.732 / 0.733 / 0.706 / 0.754; positional
1.203 / 1.147 / 1.153 / 1.059 / 1.013 / 1.073. Mapped's first block is its
slowest (page-cache state after the positional warmup), and by block 3 the
two mmap arms are indistinguishable.

## Decision

Pre-registered gate: adopt only an arm with median throughput at least 2.0x
mapped. **sequentialMap is 1.09x and positional is 0.73x: neither
qualifies. Nothing ships; production keeps `mappedIfSafe`.** The
`readArmForBenchmark` selector stays a harness-only construction argument.
This is a warm-page-cache measurement of loaded-model hydration, per the
protocol; it says nothing about genuinely cold file pages.
