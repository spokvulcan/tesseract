# Tree-side Leaf Lease evidence — 2026-09-12

Small-cache ownership and release evidence for [#479](https://github.com/spokvulcan/tesseract/issues/479), handed to [#480](https://github.com/spokvulcan/tesseract/issues/480). Production checkout remains absent: this change protects existing bodies without removing the production restore copy.

## Preserved evidence

- [Measurements](measurements.json): scalar observations from the isolated `LeafLeaseMemoryEvidenceTests` process, request IDs, token stride and access-object allocation.
- [Request timelines](memory-diagnostics.jsonl): request-correlated lease and `requestMemory` events through terminal and delayed `afterRelease`.
- [Release/writer events](lifecycle-diagnostics.jsonl): the focused suite's lease refusals, full-writer deferral and growth returns.
- [Validation](validation.json): actual Xcode test-result counts, suites, build environment and source hashes. [Manifest](manifest.json) checksums the preserved files.

Private prompts, model weights, array contents and machine/device identifiers are not included. The result bundles and full build logs were temporary verification outputs; the evidence above is the durable location.

## Workload and ownership result

One real `KVCacheSimple` attention layer (`[1, 1, 8, 64]`, Float32 keys and values) plus two 8-element Float32 recurrent arrays: **4,160 logical bytes**. The caches are captured by move. No inference model is loaded by the fixture. Twenty-four cycles alternate success, cancellation and error; return runs in `defer`, using `checkIn` for success and `rewind` for the two simulated aborts. Actual recurrent-state rewind and production abort integration remain #480 work.

The within-process comparison holds the same body and buffer addresses across acquisition/return. It measures the incremental lease overhead, not a long-context before/after benchmark.

| Boundary | Tree bytes | Leased bytes / count | Budget Floor | Observed active MLX change from previous boundary |
| --- | ---: | ---: | ---: | ---: |
| Before lease | 4,160 | 0 / 0 | 4,160 | — |
| Lease acquired | 4,160 | 4,160 / 1 | 4,160 | −48 to 0 bytes |
| Explicit quiescent return after pressure/clear refusal | 4,160 | 0 / 0 | 4,160 | −64 to +278,528 bytes |
| RAM clear after return | 0 | 0 / 0 | 0 | −4,172 to −4,160 bytes |

All 24 cycles release the cache objects and arrays, proven with weak references while the manager, telemetry recorder and every retired lease token remain alive. Acquisition never raises active MLX in the recorded observations. The returned body's clear releases at least its 4,160 bytes; allocator buffers can remain reusable. Other hosted-app activity changes the process-wide allocator baseline during this run, so the phase deltas are observations, not exclusive attribution. Buffer addresses and MLX-array identities remain unchanged during acquisition. Establishing/protecting a lease allocates no second cache. The scalar token has an 88-byte stride; the access object's measured allocation is 160 bytes. The latter excludes its separate Foundation lock and other runtime/logging allocations, so these figures are not a complete heap-overhead estimate.

The newest leaf legitimately remains the floor after return. Zero budget is not zero live cache memory. Only a subsequent explicit clear drops this last surviving leaf.

Recorded requests: success `A215ED14-DD03-4E2C-9658-E15D2D61EFF7`, cancellation `AFF53598-ECA8-4C66-91A2-028D56431F7A`, error `2F31FF86-49CF-4862-AA58-B64F7822DC5F`, and final repeated error `DD92E515-9ABC-450B-ADF7-52FB3DB350B6`. The last request includes the delayed `afterRelease` observation with zero tree, floor and lease counters.

Final validation: **886 passed, two existing skips, zero failures** in 69 focused suites, followed by **one passed** isolated memory-evidence test. Strict formatting and SwiftLint passed; the latter reported structural-size warnings listed in `validation.json`. Standards review: zero remaining findings. Spec review: zero remaining findings. Both review defects were reproduced with failing tests before their fixes.

## Writer, growth and refusal boundaries

- Restore Pins age out after the existing bounded request count; a lease survives twelve newer pin sets and `completeRequest`.
- Eviction, demotion, RAM clear, direct body drop, replacement, ancestor supersession and both queued-promotion checks refuse a leased body. Repeating the operation after return succeeds. Promotion deferral does not consume its one-shot attempt.
- A pending full payload remains unmaterialized and charged while leased, including during forced flush. It materializes after return, releases retained arrays, and drains its pending charge to zero. A reader that wins the race blocks acquisition; budget-drop and I/O-error paths release the read claim.
- A suffix cannot overtake a leased queued base; unrelated writes still complete. Detached suffix payloads need no body-read claim.
- Growth from offset 8 to 12 changes the body from 4,160 to 6,208 bytes. With a separate 4,160-byte leaf, the tree reconciles to 10,368 bytes and two bodies. The return itself preserves the supplied arrays' addresses. Invalid paths, stale tokens and returns through another tree leave the owner untouched.
- When a grown body reuses backing referenced by a pending full payload, its writer-exclusion state follows it to the new node. A new lease there still blocks that old payload. The regression uses actual shared prefix views and verifies this combined boundary.

## Interpretation and handoff

The hosted test app has substantial pre-existing MLX/OS residency. Whole-process footprint can rise while cache objects are released: startup work, asynchronous diagnostics, retained test observations and allocator/OS behavior are included in those samples. Exact process samples are preserved; this experiment does **not** establish a whole-process footprint reduction or attribute all host residency. The observations are short and are not exact phase peaks. `afterRelease` is a delayed scalar sample, not an SSD-drain guarantee; the writer tests use explicit flush and callback settlement separately. SSD pending bytes include an in-flight writer charge, whereas pending count includes waiting items only.

In this final run, process footprint rose from 1,131,300,736 to 1,225,639,832 bytes across the recorded loop boundaries (about 94.3 MB); the delayed `afterRelease` sample was 1,330,727,000 bytes. That change is not isolated to the lease implementation and its exact host-side attribution remains unresolved. The weak-reference checks rule out retained fixture caches or arrays; they do not explain unrelated process residency. Do not use these process samples as a lease heap-overhead estimate or a successful long-context memory result.

The [September 8 capture-only baseline](../../capture-handoff/2026-09-08/README.md) remains the production comparison: capture removed its full copy, but checkout still copied about 3.19 GB at the recorded 46k warm boundary. This ticket adds protection without another tensor allocation. #480 must integrate object checkout and recurrent-state rewind, reject pending full payloads at checkout, audit checkpoint lifetimes, and perform the approved long-context footprint/correctness gate. No ~45k/~75k/~93k replay or repeated model reload was performed here. The owner's explicit large-model resource-plan approval requirement still applies.

Reproduce the bounded experiment via [docs/testing.md](../../../docs/testing.md#tree-side-leaf-lease-evidence-479). Quit the running app before Xcode tests and relaunch it after validation.
