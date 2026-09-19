# No-copy SSD writer — verification and owner measurement

Ticket #469, part of #520. The implementation removes the host copy for
contiguous evaluated arrays and keeps the existing bounded header/blob writer.
The default copying encoder remains a test oracle; full and suffix goldens
continue to pin the disk format.

## Automated evidence

- `deferredBytesBorrowEvaluatedStorageAndOutliveTheSnapshot`: host-view address
  equals the evaluated MLX backing address, contents match copied bytes, and the
  Data retains its array after the snapshot goes out of scope.
- `deferredEmptyArraysProduceEmptyBytes`: zero-sized state avoids the vendor's
  force-unwrapped no-copy pointer and preserves shape metadata.
- `streamingReleasesBorrowedArraysAfterEachLayer`: the first layer's weak array
  reference expires before second-layer bytes are written; all borrowed owners
  release by the end of the consuming write.
- `borrowedLeafAndDemotionMatchTheCopyEncoder`: small attention/recurrent/empty
  fixtures produce byte-identical files to the copying encoder for both normal
  leaf writes and demotion recency behavior.
- Existing store, Leaf Lease, handoff, extension, container, and hydration suites
  cover admission, exclusion, failure cleanup, chunk bounds and persisted bytes.
  The existing INT_MAX regression writes a synthetic host-byte file over 2 GiB;
  it loads no model and creates no long-context KV cache.

## Outstanding acceptance: owner-run 3 GB comparison

No 3 GB loaded-leaf measurement was run here. The ticket's final acceptance
criterion is pending. Do not infer performance or peak-footprint improvements
from address equality, logical byte counts, or the small fixtures.

Use the owner's approved environment and resource plan, including the predefined
memory/swap stop thresholds from the
[capture-handoff constraint](../../capture-handoff/2026-09-08/README.md).
Run one process at a time. No long-context or model-reload campaign is authorized
on this Mac by this implementation task.

Compare main commit `e146ba46ecbcf6e060e4fd8e59d85ebae0734bb7` with the PR head,
using the same model, request fixture, KV dtype, approximately 3 GB payload,
cache topology, SSD directory state, disk and OS pressure. Record build hashes,
model fingerprint, request hash, payload bytes, snapshot ID and whether the write
is an ordinary leaf or Snapshot Demotion. Measure both cases. A normal capture's
short settle interval is not proof the SSD writer drained.

Capture scalar diagnostics continuously from enqueue through `storageRefCommit`,
and process physical footprint at a documented sampling interval throughout the
write. The existing allocation probe retains the new events; its fixed workload
is smaller than this acceptance case and is not a substitute for the matched
3 GB leaf. On the new build, accepted `ssdAdmit` reports `enqueueToCommitMs` and
`writeMs`; `ssdPayloadPrepare` reports view preparation, not memcpy time. On the
baseline, correlate enqueue/admission allocation samples and `storageRefCommit`
by snapshot ID. Save the sampling cadence and raw timestamps; sampled peaks are
lower bounds. Report each pair's wall time and peak process footprint separately,
with disk state and any resource stop, before claiming the final criterion passed.
