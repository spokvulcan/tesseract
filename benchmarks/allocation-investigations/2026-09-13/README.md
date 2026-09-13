# Cancellation, loading and SSD evidence — September 13, 2026

Two bounded production captures on the maintainer's 48 GiB Apple M3 Max,
following the [allocation inventory](../../allocation-inventory/2026-09-12/README.md).
The [investigation report](../../../docs/research/2026-09-13-allocation-investigations.md)
explains the root causes, fixes, hypotheses and limits.

Both arms use the same experimental Release binary, baseline source
`b1f0dca32ec9baead778221a8001fe4aa59e1850` plus each capture's preserved patch,
Qwen3.8-27B affine 4-bit target and 4-bit DFlash2, unquantized KV and temperature
zero. Each completed six responses and one deliberate disconnect, with no
resource stop. All seven request hashes, assembled assistant-message hashes
and usage records match between arms. Maximum prompt: 12,204 tokens.

| Finding | Evidence |
| --- | --- |
| Startup cancellation now reaches prefill promptly | Disconnect at about 3.008 s; signal at about 3.000 s on the slightly later server clock; terminal at 4.85–4.93 s. Both one-second signal-delay checks passed. |
| Clearing free MLX buffers before stacking reduced the observed loading transient | Largest 250 ms loading sample: 29.145 → 23.160 GiB, a 5.985 GiB difference. Active bytes before stacking were identical; cached bytes fell from 7,939,118,252 to zero. |
| The production SSD writer borrows payload buffers | Full payload 387,121,152 B; encoded file 387,153,971 B; separate encoding staging 32,819 B. Address regression and old-format goldens establish removal of the second full container allocation. |

The loading experiment has one observation per arm. Sampling can miss short
peaks, VM/page-cache and starting system swap differed, and active-MLX lifetime
peak remained unchanged. Persisted SSD metadata also changed the later restore
routes: the preserve arm exercised copy and handoff/rewind, while the clear arm
used copy restore. Consequently inference footprint and latency are not a
controlled comparison. These captures do not establish full bitwise cache/logit
parity or long-context behavior. SSD staging counts are logical bytes, not a
matched measurement of physical-memory savings.

## Contents and verification

- `preserve-cache/` and `clear-cache/`: environment/plan, model-file and binary
  hashes, exact experiment source patch and runner, synthetic request records,
  diagnostics, OS samples, outcomes and derived summaries. The latter arm enabled
  the temporary cache-clear switch. `appExitCode: -15` is intentional cleanup
  after the finite capture, not a test failure.
- `validation/`: complete compressed red/green test logs, readable summaries,
  commands and raw-log hashes, 107 distinct passing test functions in ten suites,
  experimental/final Release build summaries, final source patch and binary
  provenance. Golden fixtures are included in the final patch and live under
  [the test fixtures](../../../tesseractTests/Fixtures/PlaceholderContainer/README.md).
- `summarize.py` and `compare.py`: derive the summaries and comparison without
  loading a model. `comparison.json` retains the loading measurements and hash
  comparisons. Raw response-content hashes use assembled assistant messages;
  wire SSE chunk hashes can differ with batching.
- `SHA256SUMS`: all files in this archive except the checksum manifest itself.

From the repository root:

```bash
(cd benchmarks/allocation-investigations/2026-09-13 && shasum -a 256 -c SHA256SUMS)
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/allocation-investigations/2026-09-13/summarize.py
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/allocation-investigations/2026-09-13/compare.py
```

The final source enables the measured cache-clear branch unconditionally,
removes the temporary environment switch and explicitly flattens the startup
task group's nested optional. The 12 delivery/lifecycle tests passed again
after that expression-only cleanup. The final Release build passed; there was
no third production run. SwiftLint completed with warnings recorded in
`validation/swiftlint.log`. The ordinary app was
restored after testing. Production request records contain the fixed synthetic
recipe, hashes and scalar results; private conversation content was not used.

## Bounded rerun

The current probe prints its plan without `--run`. Close the ordinary app before
a capture and restore it afterward. A rerun with the final source uses:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/allocation_inventory_probe.py \
  --app '/absolute/path/to/Tesseract Agent.app/Contents/MacOS/Tesseract Agent' \
  --output /absolute/path/to/a/new/capture-directory \
  --comparison-label clear-before-stacking \
  --allow-pressure-warning --footprint-stop-gib 32 --swap-growth-stop-gib 2 \
  --max-cancel-signal-delay-seconds 1 --run
```

The runner uses isolated port 18321, normal starting OS pressure, 250 ms samples,
180-second response deadlines, a separate 60-second release wait, a five-second
settle interval and a 900-second campaign deadline. It stops on critical/unknown
pressure, sampled 32 GiB app footprint or 2 GiB additional system swap. These
are sampled abort conditions, not guaranteed peak ceilings. The five-second
settle is not an SSD-drain guarantee. It does not start a 45k/75k/93k campaign.

To reproduce the historical two-arm experiment, use the archived source patch
and matching archived runner: omit `--clear-load-cache` for preserve, add it for
clear, keeping all other settings equal. That option is intentionally absent
from the current runner. Preserve and document cache topology and VM state;
labels alone do not make a run a controlled comparison.
