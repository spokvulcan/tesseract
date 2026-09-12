# Pi coding-session memory incident — 2026-09-12

The evidence strongly identifies a pre-existing **AlphaTuner replay allocation
bug**. A cache-policy simulation creates and deep-copies real MLX arrays at the
recorded multi-gigabyte snapshot sizes. The grid search runs synchronously on
the MainActor during post-request bookkeeping. This explains the sudden memory
surge and unresponsive app after generation had already finished.

This is a read-only diagnosis of the user's captured run. No model was loaded,
no inference was initiated, and the high-memory workload was not replayed.
The live-reproduction/fix phases of the diagnosis workflow were deliberately
skipped: the request was log analysis, and the existing owner memory constraint
prohibits automatically repeating the crash workload. The captured samples and
source allocation formula provide the evidence below; they are not a live
allocation-stack capture or a validated fix.

Subsequent mitigation: production AlphaTuner is disconnected in PR #503;
its code is retained, and the existing static `alpha = 0` is used instead.
[#504](https://github.com/spokvulcan/tesseract/issues/504) tracks reconsidering
static weights, removal, or bounded metadata-only tuning. The corrective-work
options below are proposals from the diagnosis, not an implemented redesign.

## Session statistics

| Item | Observed |
| --- | ---: |
| Client | Pi coding agent, session `01a09754-2a80-77eb-98c2-f30aa0fe1b44` |
| Model | `qwen3.8-27b`, DFlash2, unquantized KV |
| Successful completions | 33 / 33 |
| Largest input context | 91,103 tokens |
| Largest context including output | 91,294 tokens |
| Generated tokens summed across requests | 67,283 |
| Newly submitted input tokens summed across requests | 21,260 |
| Reused input tokens summed across requests | 2,134,529 |
| Restore modes | 32 handoffs; 1 initial checkpoint copy |
| Terminal tree leases | Zero for every request |
| Median post-generation tail excluding incident | 29.553 ms |
| Incident post-generation tail | 64,636.499 ms |

Summed cached input counts repeated history on each request; it is not the
unique conversation length. The Pi usage records and server input counts agree
for all 33 requests. No generated content, tool arguments or user prompts are
included in this evidence bundle.

## Exact incident

Request **#20**, `9BF54598-956E-4925-977E-96C1AC30239F`, started at
21:05:21 UTC with **82,461 prompt tokens**. It reused 82,164 tokens and
prefilled only 297. Pi counted 473 output tokens; the server's fed-token/leaf
account counted 474 (these are different counters).

All memory below is **GiB = 2^30 bytes**. Raw byte counts remain in the CSV/JSON.

| UTC | State | Active MLX | Process footprint | System swap |
| --- | --- | ---: | ---: | ---: |
| 21:05:41 | Entering request bookkeeping, after generation and leaf return | 20.96 | 24.17 | 2.985 |
| 21:05:48 | Bookkeeping spike | 36.35 | **39.63** | **13.00** |
| 21:06:26 | Another replay allocation cycle | 36.34 | 39.43 | 11.96 |
| 21:06:46 | Request complete; active arrays released, OS footprint still elevated | 20.77 | 34.59 | 12.05 |
| 21:06:47 | Following request has begun | 22.87 | 23.75 | 12.01 |

The peak footprint was **42,547,637,240 bytes** (42.55 decimal GB / 39.63 GiB),
higher than either screenshot. The screenshot at 21:06:36 was taken after the
highest sample. Its selected Activity detail shows request #11 at 69,212 tokens;
the current request in the recent list was #20, which explains the apparent
70k versus 82k discrepancy. The 21:08 screenshot shows recovery during #23 at
84,400 prompt tokens. Later requests continued through 91k without another
comparable spike.

Generation quiesced at approximately 21:05:41. The leaf was captured in
0.203 ms; the entire Leaf Store took 18.662 ms. The explicit lease-end event
returned the real cache at 21:05:41 with zero leases. The following
`recordingRequest` phase lasted 64.770 seconds and contains the memory spike.
The separate post-generation wall-clock trace reports 64.636 seconds.

The `treeLeasedBytes` field carried into periodic request-memory samples is
stale at this point: it was last observed before the return. The explicit
`leafLeaseEnd` and final terminal counters establish that the real leaf was
already returned; the carried field must not be interpreted as an orphaned
lease.

The `afterRelease` sample overlaps request #21. It shows recovery but is not a
controlled idle-memory measurement. System swap is machine-wide and was already
about 3 GiB before the incident; the run added roughly 10 GiB around the spike.

## Source and allocation match

The path in the inspected checkout is:

1. `ServerCompletion.swift:1095–1128`: marks `recordingRequest`, then calls
   `prefixCache.recordRequest` inside `MainActor.run`.
2. `PrefixCacheManager.swift:1844–1891`: forwards scalar/path metadata into the
   tuner; the call synchronously returns the grid-search winner.
3. `AlphaTuner.swift:139–149, 197–214`: the completed bootstrap window triggers
   21 candidate-alpha replay passes, all synchronously on MainActor.
4. `AlphaTuner.swift:333–349`: `makeReplaySnapshot` creates two real float32
   zero arrays sized from the recorded snapshot bytes, then calls
   `HybridCacheSnapshot.capture`.
5. `HybridCacheSnapshot.swift:215–258`: capture creates independent copies and
   calls `eval(copiedArrays)`, materializing GPU allocations for data the
   eviction simulation never reads.

For the last replay step, the previous synthetic leaf, the new zero arrays and
the new deep-copied synthetic leaf can coexist before admission replaces the
previous leaf. Using the captured leaf sizes:

| Component | Bytes |
| --- | ---: |
| Post-tuning active baseline | 22,301,257,411 |
| Previous synthetic leaf | 5,538,643,968 |
| Source zero arrays for the new synthetic leaf | 5,589,172,224 |
| Deep-copied new synthetic leaf | 5,589,172,224 |
| **Predicted active total** | **39,018,245,827** |
| **Observed active sample at 21:06:26** | **39,018,245,843** |
| Difference | **16** |

The simulation explains **16,716,988,416 additional bytes**, approximately
**15.57 GiB**, in this sample. The close match, phase timing, and synchronous
call path strongly support the diagnosis. It is not a tensor-by-tensor profiler
trace; the tuner start/end info messages were not retained in the queried
unified-log interval.

The tuner fires once after its post-eviction observation window fills; it is
not a 60k/70k context threshold or an every-20-requests timer. Once it finishes,
its temporary simulation caches are destroyed and its state becomes `tuned`.
That accounts for recovery and the absence of a second spike during the rest
of this process's session. A fresh model/cache lifecycle can start it again.

`git blame` traces the real-array simulation helper to commit `638a7af5c`
(2026-04-12), predating the leaf-handoff PR. The current checkout is `13ea4462`.
The executable used by PID 92396 was rebuilt at 20:29 and its hash differs from
the earlier review binary; no source revision is embedded, so identical source
provenance is not asserted. Its telemetry verifies checkout-by-move was active.
The implicated tuner code is unchanged across the recent PR commits.

## Other hypotheses checked

- **Restore/capture duplication of the real conversation cache:** excluded as
  the spike's timing source. Both operations finished before the spike, used
  handoff, and the explicit lease ended. One real resident leaf remained.
- **SSD payload:** the outstanding extension was only 204,472,320 bytes. Its
  materialization took 7.508 seconds and completed at 21:05:49, while the large
  allocations continued until 21:06:46. It may have suffered from pressure but
  cannot explain the recurring ~16.7 GB active allocation pattern.
- **DFlash2 decode or a huge prefill:** preparation ended after 2.697 seconds;
  decode finished before the spike. The affected request added only 297 prompt
  tokens and ~474 output/fed tokens. A prior 29,545-output-token request did not
  produce this spike.
- **Persistent lease/cache leak:** all 33 requests completed and ended with
  zero leases; the single real leaf grew normally to ~6.14 decimal GB by the
  last request. This incident is a temporary simulation allocation surge.

## Corrective work indicated

Replace real replay cache bodies with **metadata-only simulation snapshots**:
retain logical byte charges, paths, offsets, types and access times, without
creating MLX arrays, invoking cache capture, or evaluating a GPU graph. Keep
that representation isolated from real restore/admission, where an empty cache
must remain invalid. Existing pure topology fixtures demonstrate the needed
separation; simply weakening production admission is not the fix.

Add a bounded regression at the real tuner replay seam: multi-gigabyte logical
snapshot sizes must not cause proportional MLX allocations, and the grid's
scores/winner must remain correct. Measure CPU replay time separately; if it
can still block interaction, move the scalar simulation off MainActor or bound
its scheduling. Merely moving the existing array-allocating code off MainActor
would leave the memory hazard intact.

Add durable scalar tuning begin/end events, including window size, candidate
count, elapsed time and allocation deltas. Then validate the fix with small
fixtures before any owner-approved long-model confirmation. The earlier
working-memory estimate describes ordinary one-cache operation and does not
cover this avoidable extra ~15.6 GiB simulation allocation.

## Preserved evidence

- `requests.csv`: all 33 joined request summaries. Peaks exclude `afterRelease`.
- `memory-timeline.csv`: all 3,374 scalar memory samples, including separately
  labelled `afterRelease` observations.
- `incident-events.jsonl`: incident request plus system events in its interval.
- `summary.json`: exact totals and allocation comparison.
- `provenance.json`: source identities/hashes and limitations.
- `manifest.json`: SHA-256 checksums for the preserved files.

The original daily diagnostic capture and completion-trace capture were copied
locally before rotation. The bundle preserves the relevant scalar facts instead
of publishing private requests or full Pi session content.

## Disable mitigation validation

The production construction regression drives a tiny toy model through Server
Completion and reads the published cache configuration/tuner telemetry, both
before Model Identity is available and after it is installed. With the original
tuner attached, both cases failed the unavailable-tuner expectation; with the
attachment removed, both passed. No large logical snapshot or loaded Qwen
workload was replayed.

- Focused AlphaTuner suite: 19 passed, zero failed.
- Full unit target: 2,879 passed, 15 skipped, zero failed (2,894 declarations).
- Strict Swift formatting, SwiftLint (seven warnings, zero serious violations),
  documentation references and diff whitespace checks passed.
- Release build passed; the updated app was restarted and its local server
  returned `{"status":"ok"}`. The binary hash is recorded in `validation.json`.
- Independent spec and standards reviews found no code issues; two stale
  comments were corrected. Those final Swift changes were comments only.

`validation.json` preserves the exact test totals, test-result/log locations,
review disposition and limitations. These checks validate the disconnected
production wiring; they do not measure a new long-session memory peak.
