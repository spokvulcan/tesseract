# No-copy SSD writer — owner comparison (#469), 2026-09-21

Pre-registered before the arms ran; results are appended below only after
both arms completed. Plan: `plan.json` (committed first). Driver:
`scripts/ssd_writer_comparison.py`. Parent ledger:
[`../2026-09-19/README.md`](../2026-09-19/README.md).

## Design (fixed before running)

- **Builds.** `baseline` = main `e146ba46` (the last commit before the
  no-copy writer), built Release from a detached worktree with its pinned
  vendor `51542c4e`. `current` = this tree's Release binary (app code
  `3d1c0d35`, vendor `a3c1776c`). SHA-256 of both binaries is in the plan.
- **Order.** baseline first, then current. One run per arm, no retry on a
  resource stop; a stopped arm leaves the comparison without a result.
- **Workload per arm.** Fresh, empty SSD root; RAM budget cap 6 GiB;
  Qwen3.8-27B-4bit with the app's default DFlash2 draft; float16 KV;
  reasoning effort low; 64 output tokens per turn. Three single-turn
  conversations built from `CONTEXT.md`: A (165,000 chars, ~2.7 GB leaf),
  B (165,000 chars from a different offset, second clean sample; resident
  ~5.4 GB, under the cap), C (60,000 chars, ~1 GB) which pushes the resident
  total past the cap so the manager evicts an older leaf.
- **Leaf case (the acceptance measurement).** A and B are end-of-turn leaf
  writes (guarantee class) of ~2.7 GB each, on both builds. The host was
  re-derived for 48 GiB, so the payload is ~2.7 GB rather than 3 GB; the
  plan records the exact bytes.
- **Eviction stage.** Production always writes an end-of-turn leaf at
  capture, so a ~3 GB body is SSD-backed by the time eviction can reach it.
  The C step records what the eviction actually does on each build
  (`eviction` events, `ssdAdmit` write classes). If no demotion write
  occurs because the victim is already backed, the demotion case is recorded
  as **not inducible through production paths**, with those events as the
  evidence, and the comparison rests on the leaf case. No harness-only
  demotion is substituted.
- **Metrics, identical instrumentation on both builds.**
  1. *Enqueue → commit wall time* = arrival of the `storageRefCommit` line in
     the CacheDiagnostics sink (polled every 20 ms) minus the request's
     `admittingLeaf` instant (driver request start + the sample's
     `elapsedMs`). The new build's in-process `enqueueToCommitMs` and
     `writeMs` are reported alongside as a cross-check, not as the
     comparison metric.
  2. *Segment file write duration* = first sighting of the segment file in
     the SSD root (polled every 50 ms) to the sample at which it reached its
     final size.
  3. *Peak process footprint during the write* = max 250 ms sample from the
     `admittingLeaf` instant to the commit arrival, and its delta over the
     last sample before `admittingLeaf`. Sampled peaks are lower bounds.
- **Decision.** The acceptance criterion holds if, on both clean leaf
  samples, the current build's enqueue → commit time is not longer than the
  baseline's and its footprint delta during the write is smaller by about
  the payload size (the host copy the change removes). Anything else is
  recorded as-is.
- **Bounds (48 GiB host).** Minimum initial available 20 GiB; expected peak
  26 GiB; footprint stop 34 GiB; minimum available stop 6 GiB; swap growth
  stop 1 GiB; pressure stop level 4; free-disk stop 15 GiB; request 400 s;
  write settle 120 s; campaign 1800 s per arm; 250 ms sampling.

## Result: the baseline arm stopped on the available-memory bound; no comparison

Run 2026-09-21 09:10–09:14 UTC, baseline arm first as pre-registered
(`outcome-baseline.json`, raw records in
`~/bench-results/no-copy-writer-2026-09-21-baseline`,
`raw-record-checksums-baseline.txt`).

The A-doc turn (38.6k prompt tokens, leaf 2,681,143,296 bytes) prefilled and
answered; the request's `capturingLeaf`, `preparingPayload` and
`admittingLeaf` phases and the `capture` event were recorded, i.e. the
baseline's host-copy write had begun. The first sample after the response
read footprint 22.99 GB and **available memory 5.91 GiB, under the 6 GiB
stop**, so the driver terminated the app 222 s into the arm. No segment
file appeared in the SSD root (295 bytes: the partition metadata only), no
`ssdAdmit` or `storageRefCommit` was emitted, and the plan's B and C steps
did not run. Peak sampled footprint 26.37 GB (during prefill), pressure
level 1 throughout, swap growth 0.

The bound was committed before the run and is not moved; the leg is not
retried, and the current arm was not run because the design gives a
stopped arm no comparison. **#469's owner-run acceptance criterion (the
matched ~3 GB leaf and demotion timing and peak-footprint comparison) is
therefore still unmet on this host.** The unit evidence in
[`../2026-09-19/README.md`](../2026-09-19/README.md) stands as it was.

What the stop says about the host, for whoever plans the next attempt: with
the 27B weights resident (~15.1 GB) and a 2.7 GB body captured, this 48 GiB
machine sits within about 1 GiB of the floor once a multi-gigabyte file
write starts; the write's dirty pages are not counted as available. A plan
that fits would need either a smaller leaf than the ticket's ~3 GB, a
smaller model, or a host with more headroom, and would be a new
pre-registration rather than a re-run of this one.
