# Cache Claim memory gate

Refs [#554](https://github.com/spokvulcan/tesseract/issues/554). The
real-model A/B the PRD sets as the merge gate, run as the owner approved it on
2026-09-22. The plans and this page are committed before the first campaign.
The baseline to compare against is #553's
[2026-09-21 profile](../2026-09-21/README.md).

## What runs

Two Release builds on this host (Mac15,9, 48 GiB, macOS 27.0), with the app
closed and one app process at a time:

| Arm | Commit | Release binary SHA-256 | Plan |
| --- | --- | --- | --- |
| main | `52ccbf8f` | `b55c24b7…6c3b59` | [plan-main.json](plan-main.json) |
| PR | `5995a0d1` | `a1b225e8…fbef74bcf` | [plan-pr.json](plan-pr.json) |

In the order main, PR, main, PR, each arm runs:

1. the cancelled-generation profile (`scripts/cancelled_generation_profile.py`)
   at its p8k and p32k steps: grow, cancel, rewind, then a short turn, on
   the think-stripping path. The 66k step is not part of this approval;
2. the allocation probe (`scripts/allocation_inventory_probe.py`): seven
   requests up to 12.2k tokens, with its default stops.

On the PR build only, after the four arms: bounded-cache parity
(`scripts/bounded_cache_parity.py`) and `prefix-cache-e2e`, the latter under
the same stops as the profile.

The model is `qwen3.8-27b` with its DFlash2 draft, KV unquantized. The profile
filler is `CONTEXT.md` at `52ccbf8f`, the same bytes as #553's compaction run.

Each campaign runs against its own new SSD cache directory, set through the
app's `prefixCacheSSDDirectoryOverride` setting as a launch argument, so it
starts cold and the owner's cache is neither read nor written.

## Attempt 1, stopped (2026-09-22)

The first attempt shared the owner's SSD cache, which still held #553's run of
the same prompts. The first campaign's first request hit 7,018 of its 7,020
tokens from SSD, at a node #553's conversation continues past, so no build can
lease it. The cancelled turn restored by copy and had no lease to rewind, and
the profile driver stopped on "Cancelled request produced no Leaf Rewind"
after 45 seconds. No resource stop was near (lowest available memory
10.5 GiB). The gate was stopped after the next campaign's first request.
Every later campaign would also have warm-started from the one before it, so
the owner approved rerunning the whole gate with a new SSD directory per
campaign. The records stay in the private results directory.

## Stops, fixed now

Footprint over 34 GiB, available memory under 6 GiB, swap up by more than
1 GiB, or memory pressure at level 4. The allocation probe's defaults are
stricter (28 GiB, 512 MiB, warning pressure). A campaign that stops is
recorded, not retried.

## Pass rule, fixed now

The PR passes when all of these hold:

- every step's peak MLX, peak footprint and settled footprint is below
  main's, or within the spread between the session's two main runs;
- restore modes, copy reasons and leaf sources match main step for step,
  except SSD-hit restores, which now hand off (`copyRefusal` is new and has
  nothing to compare against);
- the reserve reports one lane for back-to-back requests;
- the lease count is 0 after every request;
- parity and the e2e run pass.

Raw records stay in the private results directory; only scalars are
committed here.

## Results, attempt 2 (2026-09-23)

[summary.json](summary.json) holds the scalars, written by
[summarize.py](summarize.py); the raw records stay in the private results
directory. The campaigns ran from 00:07 to 00:37 UTC in the committed order,
and [finish_gate.sh](finish_gate.sh) ran what was left from 00:42 to 00:47.

- Seven of the eight arm campaigns completed: main's two profiles and two
  probes, and the PR's two profiles and first probe.
- The PR's second probe did not complete. The gate was stopped from outside
  at 00:37, after three of its seven requests. The owner approved one rerun,
  which stopped after four requests when the diagnostics sink reached its
  8 MiB cap and rotated (the probe driver stops rather than lose which request
  wrote which event, and unlike the profile driver it does not follow a
  rotation). Its four completed requests are in the summary; it was not run a
  third time.
- Parity and the e2e run completed on the PR build.
- No resource stop fired. The profiles' lowest available memory was 9.8 GiB
  (stop 6 GiB) and their largest sampled footprint 23.7 GiB (stop 34 GiB),
  swap never grew, and pressure stayed normal. The probes' stricter default
  stops never fired either.

Peak MLX, peak footprint and settled footprint per step, in GiB, main's two
runs against the PR's. A bold PR reading is above both main runs.

| Profile step | Peak MLX, main | Peak MLX, PR | Peak footprint, main | Peak footprint, PR | Settled footprint, main | Settled footprint, PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p8k-grow | 17.27 / 17.32 | **17.33** / **17.33** | 19.36 / 19.33 | 19.32 / 19.33 | 18.12 / 18.09 | 18.08 / 18.09 |
| p8k-cancel | 17.28 / 17.29 | 17.19 / **17.34** | 19.05 / 19.03 | 19.01 / **19.28** | 17.55 / 17.53 | 17.50 / 17.52 |
| p8k-after | 17.31 / 17.31 | **17.31** / **17.36** | 18.79 / 18.77 | 18.75 / **19.03** | 18.28 / 18.26 | 18.24 / 18.26 |
| p32k-grow | 21.61 / 21.61 | 21.61 / 21.61 | 23.75 / 23.03 | 23.71 / 23.29 | 20.86 / 20.83 | 20.81 / 20.83 |
| p32k-cancel | 18.81 / 18.88 | 18.73 / 18.84 | 22.32 / 22.29 | 21.72 / 21.99 | 18.98 / 18.96 | 18.93 / 18.97 |
| p32k-after | 20.18 / 20.18 | 20.18 / 20.18 | 23.08 / 23.04 | 23.03 / **23.29** | 20.88 / 20.85 | 20.83 / 20.85 |

| Probe step | Peak MLX, main | Peak MLX, PR | Peak footprint, main | Peak footprint, PR | Settled footprint, main | Settled footprint, PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 16.75 / 16.76 | **16.76** / **16.81** | 18.30 / 18.25 | **18.30** / **18.30** | 17.12 / 17.08 | **17.13** / **17.13** |
| warm | 16.92 / 16.92 | **16.93** / **16.96** | 18.23 / 18.44 | 18.23 / **18.50** | 17.29 / 17.50 | 17.30 / **17.56** |
| grow | 17.62 / 17.47 | 17.62 / 17.21 | 19.24 / 19.19 | **19.24** / **19.25** | 17.71 / 17.92 | 17.72 / **17.98** |
| cancel-growth | 17.84 / 17.79 | 17.84 / 17.73 | 19.71 / 19.67 | **19.72** / 19.68 | 18.54 / 18.73 | 17.37 / 17.60 |
| resend-growth | 18.02 / 18.34 | 18.19 / not run | 20.24 / 20.47 | 20.23 / not run | 18.11 / 18.30 | 18.12 / not run |
| warm-repeat-1 | 17.33 / 17.42 | 17.33 / not run | 18.45 / 18.67 | 18.46 / not run | 17.82 / 18.03 | 17.82 / not run |
| warm-repeat-2 | 17.33 / 17.46 | 17.33 / not run | 18.46 / 18.64 | 18.46 / not run | 17.81 / 18.00 | 17.82 / not run |

### The pass rule

1. **Memory: not met as written.** 22 of the PR's 69 readings sit above both
   main runs. Nine are within 10 MB. The largest are the PR's second profile
   run at p8k-cancel, p8k-after and p32k-after, 0.21 to 0.23 GiB of peak
   footprint above main, where the PR's first run was at or below main at the
   same steps, and up to 0.05 GiB of peak MLX. Main's own two runs differ by
   up to 0.73 GiB (p32k-grow's peak footprint) and 0.32 GiB (resend-growth's
   peak MLX), so its spread on the steps above happened to be narrow.
2. **Restore modes, copy reasons and leaf sources: met.** Identical step for
   step in every completed campaign. Every campaign started cold, so none had
   an SSD hit: the gate does not exercise the SSD-loaded handoff (item 5),
   which the model-free evidence measures instead.
3. **One reserve lane for back-to-back requests: met.** Every `budgetMeasure`
   sample in every campaign reports one lane, on main as on the PR, so these
   samples did not catch main's overlap either.
4. **Lease count 0 after every request: met.**
5. **Parity and the e2e run: met.** Parity passed its 18 checks, the
   check-out and rewind ownership and exact logits after the handoff and after
   the rewind among them, which now run through the claim's own code. The e2e
   run passed its 22 checks; its image scenario is skipped on this text-only
   model.

Beyond the rule, the PR's clearest gain is after a cancelled generation. The
probe's cancel-growth settles at 17.37 and 17.60 GiB on the PR against 18.54
and 18.73 on main, because the claim's conclusion clears MLX's buffer cache
after the rewind has compacted the leaf, and p32k-cancel's peak footprint is
21.72 and 21.99 GiB against 22.32 and 22.29.
