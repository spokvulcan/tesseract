# SSD read-path experiment — #532

**Pre-registered; owner run pending. No performance result or adoption claim.**
The production reader remains `Data(contentsOf:options:.mappedIfSafe)`.
This branch builds the three arms and a loaded-model harness; it does not
satisfy the measurement and adoption acceptance criteria until the owner
executes and reviews the approved plan. Parent: #520; no parent edits.

## Arms and decision

| Arm | Mechanism | Loaded-model median GB/s | Ratio to mapped |
| --- | --- | ---: | ---: |
| `mapped` | Current Foundation mapping, then each contributing array copied into MLX | Pending owner | 1.0 |
| `sequentialMap` | Read-only private mmap; `MADV_SEQUENTIAL` before parsing/copying | Pending owner | Pending owner |
| `positional` | 8 MiB `pread` chunks into one page-aligned host allocation per segment; the same single copy per contributing array into MLX | Pending owner | Pending owner |

On **one exact Segment Chain**, adopt only an arm whose median throughput is
at least **2.0 times** mapped, with identical bytes/metadata and acceptable
memory use. If both qualify, choose the faster median; break an exact tie in
favor of sequentialMap's lower host-buffer footprint. If neither qualifies,
keep mapped and record the result. An advice failure fails that arm rather
than silently substituting mapped. The experiment selector is only a harness
construction argument; there is no product setting. A winning production
change removes that selection and ships the measured arm without a flag.

## Owner approval before execution

This Mac must not run loaded-model, long-context, or model-reload workloads
without the owner taking over. **Do not execute these commands as an agent.**
The template is deliberately not runnable: approval, identity and limits are
empty/zero. The owner fills them and approves this protocol before execution.

1. Select an existing, immutable cache export with one current Segment Chain;
   no new long-context generation is part of the harness. Record the target
   model/directory, its fingerprint, machine/RAM/macOS, source revision, and
   the exact chain paths, SHA-256 hashes and total bytes. Stop other inference
   and cache writers before exporting. Use the same files for all arms.
2. Choose `maxSegmentBytes` and `maxMLXBytes` for this machine, plus external
   process-RSS, memory-pressure, swap-growth, and wall-time stop limits. Record
   those external limits and the monitoring/termination method alongside the
   plan. They must cover model loading as well as every hydration. No automatic
   retry/reload on failure. The internal limit is MLX memory, **not process RSS**;
   positional host buffers and file pages are not fully represented in it.
3. Record approval identity/date in `ownerApproval`. Save the completed JSON
   before running; retain it unchanged with the report. There is one model
   load (speculation off, no live SSD tier). Normal `LLMActor.loadModel`
   verification prepares "Hello" and generates one token before any timed
   hydration; there is no additional benchmark prompt/generation. Application
   background services do not start for `--ssd-read-bench`.

```json
{
  "ownerApproval": "",
  "cacheRoot": "/absolute/path/to/immutable-cache-export",
  "snapshotID": "SELECTED_CHAIN_HEAD",
  "segmentSHA256": {
    "partitions/DIGEST/snapshots/SHARD/SEGMENT.safetensors": "SHA256"
  },
  "maxSegmentBytes": 0,
  "maxMLXBytes": 0
}
```

Owner command after building the app, replacing all placeholders:

```bash
'/path/to/Tesseract Agent.app/Contents/MacOS/Tesseract Agent' \
  --ssd-read-bench --ssd-read-plan /absolute/path/approved-plan.json \
  --bench-model /absolute/path/to/model --bench-model-id MODEL_ID \
  --bench-source-revision COMMIT --bench-output /absolute/path/to/results
```

The harness bounds and hashes a scratch copy of only the selected chain,
checks the model fingerprint, and uses the existing SSD store's warm start
and hydration on the Model Session's `container.perform` boundary. Ordinary
read/decode failure cleanup can affect only scratch backing, never the source.

## Measurement protocol (fixed before observing throughput)

- One unmeasured warmup per arm, then six blocks in this order:
  ABC, BCA, CAB, CBA, BAC, ACB (A=mapped, B=sequentialMap, C=positional).
  Every position and predecessor are balanced; six observations per arm.
- The target remains loaded throughout. There is no RAM snapshot reuse;
  each hydration owns fresh MLX arrays and the previous body is released
  before the next observation. Clear only the MLX free-buffer pool between
  observations. Never purge the owner's OS page cache.
- **OS file cache is warmed** by staging/hash verification/warmup. "Cold hit"
  here means an SSD-only Snapshot Ref, not a cold OS page cache. This measures
  loaded-model hydration under that declared condition; do not present it as
  sustained drive bandwidth or generalize to genuinely cold file pages. The
  owner must confirm that this condition represents the original measurement;
  if it does not, pre-register a separate OS-cache protocol before running
  and do not use these warm-cache numbers to adopt a cold-read change.
- Whole-hydration timing begins before `loadSync` and ends after evaluation
  and GPU synchronization. It includes read setup, page faults, header parse,
  MLX copies, chain composition, and segment telemetry. Byte digest calculation
  and output writes are outside the timed region. All arms have identical
  instrumentation. Primary throughput: materialized payload bytes / seconds /
  1e9. Report all six samples, median GB/s, baseline ratio, MLX peak, external
  RSS/swap observations, failures and elapsed wall time. No trimming outliers.
- Before every hydration, require three times the chain-file bytes of approved
  MLX headroom (a conservative guard, not an RSS guarantee); fail on the
  post-hydration MLX peak limit or any mismatched snapshot bytes/metadata.
  Each completed observation is saved immediately. An interrupted/failed run
  has **no valid adoption result**, even if some samples look favorable.
- Generated `README.md`, `records.json`, `block-*.json` and the original plan
  live under `--bench-output/ssd-read-UUID/`. Preserve these and the owner's
  external resource log. Update this README's table with those exact results,
  explicitly record the decision, and only then ship a qualifying arm or close
  with unchanged mapped behavior.

## Per-segment telemetry and tests

`ssdHydrateSegment` carries owner snapshot ID, segment filename, arm,
`fileBytes`, `materializedBytes`, `readMs`, `copyMs`, `durationMs`, and
`completed`. Duration is read/setup/header work plus actual MLX copy work,
including lazy-map page faults. Cross-segment concatenation/evaluation is
accounted only in whole-hydration timing; it is not arbitrarily assigned to
one segment. Superseded whole-state blobs count in fileBytes but not
materializedBytes. A decode failure emits incomplete measurements; a read
failure that yields no owned segment bytes emits the existing `ssdMiss`.

- `SSDSnapshotStoreTests.readArmsHydrateIdenticalBytesAndReportSegmentWork`:
  all arms, byte equality, separate MLX buffer addresses, correct per-segment
  file/payload byte counts, duration fields, repeated hydration, and an 8 MiB
  positional-read chunk boundary with a non-aligned tail.
- `SSDSnapshotStoreTests.readArmsPreserveInterruptedBackingAndCondemnDamagedFiles`:
  interruption preserves backing; missing, empty and truncated files miss
  and trigger the existing hydration-failure cleanup for each arm.
- `ChainPrefixHydrationTests.prefixHydrationComposesOnlyTheLeadingSegments`:
  all arms at a historical boundary and at the full chain head; sliced KV
  composition and the correct historical/non-historical whole recurrent state.
- `PrefixCacheDiagnosticsTests.ssdHydrateSegmentReportsActualCopiedBytesAndDuration`:
  stable event units/fields and summed per-segment duration.

These are temporary-file/small-tensor checks at the existing SSD store seam.
They do not substitute for the owner's loaded-model throughput or memory run.
