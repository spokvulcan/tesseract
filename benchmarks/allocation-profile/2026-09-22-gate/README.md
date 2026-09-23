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
