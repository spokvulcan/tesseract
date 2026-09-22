# Cache Claim memory gate: plan

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
