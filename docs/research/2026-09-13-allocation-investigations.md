# Cancellation, loading and SSD allocation investigations

Follow-up to the [September 12 inventory](2026-09-12-local-inference-allocation-inventory.md),
authorized by the maintainer to investigate all three findings. Baseline source:
`b1f0dca32ec9baead778221a8001fe4aa59e1850`, plus the preserved #506 diagnostics.
Production artifacts and exact experiment patches live under
[the evidence directory](../../benchmarks/allocation-investigations/2026-09-13/).

## Cancellation during generation startup

The previous production capture disconnected at 3.008 seconds, but request
telemetry recorded cancellation at 25.958 seconds: an approximately 22.950-second
gap. This was the initial failing signal, not a guarantee that network delivery
itself took that long.

Ranked hypotheses were: (1) cancellation watching begins after startup/prefill;
(2) a watcher exists but cannot observe the connection state; (3) the signal
arrives on time but prefill fails to cooperate. Source and the minimized test
support (1). `HTTPServer.monitorPeerDisconnect` already watches the socket and
marks `HTTPConnectionLifecycle` while the route runs. However,
`CompletionHandler.runCompletion` awaited `startGeneration` before calling
`CompletionDelivery.deliver`. Only delivery installed the `StreamLifecycleDriver`
that cancelled generation. The model-aware startup includes restore and prefill.
Existing driver tests began after this missed interval and therefore passed.

The minimized regression uses the real connection lifecycle and the same
startup wrapper called by the handler, with an injected slow start. Before the
fix it ran for 0.537 seconds and failed both assertions: startup did not observe
cancellation and returned its fallback failure instead of `CancellationError`.
No production model is needed for this transport/startup ordering test.

`StreamLifecycleDriver.startGeneration` now watches disconnects concurrently
with startup and cancels the startup task immediately when the watcher wins.
Its structured group awaits the losing task. If a start returns a handle despite
cancellation, the wrapper cancels and drains that handle before returning, so
it cannot release the GPU lease with an abandoned generation. Normal connected
startup transfers its handle without cancelling/draining it. The existing
delivery driver continues to own disconnects after startup and SSE keepalives.
The handler records cancelled startup as cancellation rather than a failed
response. `ServerCompletion.start` now records its task-cancellation signal
while the handle is being built; existing prefill checks, GPU quiescence and
Leaf Rewind remain responsible for safe cleanup.

Validation: 27 tests in delivery, lifecycle, completion drain and memory
telemetry suites passed, including the failing-start regression, late-handle
drain and normal connected-start ownership. This does not add cancellation
coverage while the request is still acquiring the GPU lease or loading weights.

## SSD container staging

The original encoder appended all materialized payload blobs into another
contiguous `Data` before the writer emitted its 1 GiB chunks. The prior full
production payload therefore coexisted with a 387,153,971-byte encoded container.
Chunking the already-built container did not bound its allocation.

The minimized regression sent an existing 9,009-byte `Data` through the encoding
boundary and checked the addresses of chunks handed to its consumer. The
contiguous implementation failed: payload chunks pointed into the extra
container allocation. The fix holds only the encoded header/length prefix plus
references to existing payload blobs, then borrows bounded pointers directly
from each buffer. The file writer consumes those pointers synchronously with
Foundation's throwing `write(contentsOf:)` API. The 1 GiB syscall limit,
file synchronization, temporary-file replacement sequence, error classification,
retry/admission rules and snapshot lease ordering remain in place.

Golden full and suffix container bytes were captured from the old encoder
before replacing it. New chunk output must match those bytes exactly. A
separate callback-error test requires stopping at the first failed write;
existing real-store and extension tests cover commit, recovery and byte-exact
hybrid attention/recurrent restoration. Diagnostics now distinguish total
`encodedBytes` from `encodedStagingBytes` (length prefix + JSON header), identify
`encodingMode=borrowedChunks`, and sample successful write completion.

This removes the separate payload-sized encoded buffer. It does not stream
materialization itself: the existing payload still owns all host blobs until
its final consumer releases them. Metadata, allocation capacity, VM effects and
other owners remain separate from these logical buffer counts.

Validation: 61 tests in the codec, SSD store and snapshot-ledger suites passed.
A further 50 tests in the SSD store plus the actual extraction, extension-ledger
and extension-store suites passed after the final file-write API adjustment;
the SSD store tests overlap between those runs. The real writer's >2 GiB
regression and byte-exact extension restore both ran successfully.

## Loading transient: controlled experiment

The original capture observed 29.683 GiB of footprint during projection stacking,
then 16.733 GiB at completed load. Before stacking, the draft-load boundary
reported 7.391 GiB of cached MLX buffers. The ordinary 2 GiB allocator cache limit
is configured at generation entry, after model loading. Stacking also evaluates
new weight arrays while old projection modules can still be retained by the
module traversal, so live overlap must be distinguished from reusable buffers.

Ranked hypotheses: (1) cached free buffers from target/draft loading contribute
substantially to the transient; (2) old/new live weight overlap dominates;
(3) mapped loading pages, compression or runtime memory outside MLX dominate.
The first experiment changes only a one-time `Memory.clearCache()` after draft
load and before target stacking. It preserves models, quantization, active
speculation, stacking algorithm, context and output settings. Added boundaries
separate target stacking from draft stacking and verify active versus cached
bytes before/after the clear. The temporary experiment switch was removed after
validation; the one-time cache clear is now the ordinary DFlash2 loading path.

The comparison used one Release binary and two finite seven-scenario
captures on this 48 GiB Mac, with no concurrent model process or build. Both
allow warning pressure but abort on critical/unknown pressure, 32 GiB app
footprint or 2 GiB additional system swap, with 180-second HTTP-response and
900-second campaign deadlines. A separate 60-second wait observes request
release. No 45k/75k/93k campaign is authorized or attempted here.

## Production results and decision

Both captures completed all seven scenarios (six responses and one intentional
disconnect), with no resource stop. Qwen3.8-27B affine 4-bit + DFlash2 at 4-bit,
unquantized KV, temperature zero and the 128-token output ceiling were unchanged.
All request hashes, assembled assistant-message hashes and usage matched between
arms. Largest prompt: 12,204 tokens. The comparison uses the same binary hash,
model-file hashes and fixed synthetic recipe; it is not a long-context or complete
bitwise cache/logit parity gate.

| Loading observation | Preserve cache | Clear before stacking |
| --- | ---: | ---: |
| Largest 250 ms footprint sample during loading | 31,294,652,904 B (29.145 GiB) | 24,868,323,816 B (23.160 GiB) |
| Cached MLX bytes just before target stacking | 7,939,118,252 B | 0 B |
| Active MLX bytes just before target stacking | 16,641,358,756 B | 16,641,358,756 B |
| Lifetime active-MLX peak at completed load | 23,719,247,460 B | 23,719,247,460 B |
| Reported model loading time | 9.411 s | 5.136 s |
| Target stacking interval | 3.418 s | 0.348 s |
| Maximum sampled system swap growth across capture | 1,397,358,592 B | 0 B |
| OS pressure samples | 565 normal / 6 warning | 542 normal |

The observed loading footprint difference is **6,426,329,088 bytes (5.985 GiB)**.
Clearing reusable buffers reduced the cached-byte count without reducing active
arrays. The unchanged active-MLX high-water mark shows that old/new live-weight
overlap remains; the fix does not eliminate every loading allocation. Keep the
one-time clear because it directly removes that unnecessary cached-buffer overlap
and the bounded replay retained output behavior. It neither unloads a model nor
changes quantization, stacking operations or inference parameters.

This is one paired observation, not a universal peak or speedup guarantee.
The 250 ms sampler can miss short transients, especially the now-shorter stacking
interval. System swap is machine-wide; starting swap and VM/page-cache state
differed, and those differences affect timings and footprint. Exact scalar
allocator observations and the source-established lifetime change support the
conclusion more strongly than the timing difference alone.

### Cancellation and ownership in production

| Observation | Preserve-cache arm | Clear-cache arm |
| --- | ---: | ---: |
| Client disconnect after request start | 3.008 s | 3.007 s |
| Cancellation signal, on the server request clock | 3.000 s, during prefill | 3.000 s, during prefill |
| Server request terminal | 4.927 s | 4.851 s |
| After-release observation | 5.986 s | 5.851 s |
| Cancellation restore path | Leaf Handoff + exact rewind | Copy restore |

Server elapsed time starts shortly after the HTTP request, so subtracting these
client/server durations gives a small negative value (about 7 ms), not evidence
that cancellation preceded disconnect. The signal is prompt within that clock
alignment, and both captures passed the one-second delay check. Cleanup still
waits roughly 1.9 seconds for the existing GPU/copy cleanup boundaries. The prior
trace reached terminal at 26.042 seconds; its generated history differed slightly,
so it is a symptom reference rather than an identical-input performance A/B.

In the handoff arm, the lease rewound to offset 7,135, recurrent backup bytes and
request cache ownership became zero, and resend succeeded. The restored leaf
retained 64 MiB unused array extent, compared with 306 MiB in the earlier delayed
cancellation trace at offset 7,136. No compaction policy was added: earlier
cancellation avoided most of that growth while preserving useful capacity reuse.
The copy arm has no leaf lease to rewind. Its after-release cache-byte facts
remain tagged with their earlier measurement phase and must not be mistaken
for a fresh ownership count or a leak.

Persisted SSD metadata was deliberately not deleted between runs. It changed
which prefixes were structural: the preserve arm used copy restore for its first
two warm turns and handoff for the remaining four; the clear arm used copy restore
throughout. Both had no SSD hydration. Loading observations precede request-cache
restore, but the differing cache topology prevents treating subsequent inference
footprint/latency as a controlled memory comparison. Matching outputs across
these two observed routes is useful evidence, not a replacement for #480's full
loaded-model bitwise parity gate.

### SSD encoding in production

Each arm materialized and committed one full payload and five extensions. The
full payload was 387,121,152 bytes; encoded file bytes remained 387,153,971 while
new `encodedStagingBytes` was **32,819 bytes**. The largest extension payload was
478,085,120 bytes with 478,158,624 encoded file bytes and **73,504 bytes** of
encoding staging. Across these payloads the measured header counters ranged
from 32,819 to 74,469 bytes. These are header/prefix content counts, excluding
metadata objects and allocator capacity; they are not the whole writer footprint.

The source and address regression establish removal of the separate payload-sized
encoded buffer. The production counters confirm the new path ran on full and
extension payloads. A matched old/new-encoder physical-memory A/B was not performed,
so the removed logical duplicate is not presented as an equal measured RSS or
footprint reduction. Materialization still holds the required original host blobs.

## Final validation and remaining boundaries

**107 distinct focused test functions in ten suites passed** across the main
green invocations (27 + 61 + 50, with the 31-test SSD store suite repeated).
The 12 delivery/lifecycle tests also passed after the final expression-only
cleanup that explicitly flattens the startup task group's nested optional.
The archive includes complete compressed logs, readable summaries, failed
regressions, golden fixtures, both production captures and comparison code.
Strict Swift format lint, Python syntax, docs/link checks and the final Release
build passed. SwiftLint completed with warnings, preserved in the archive.
The final source enables the measured cache-clear branch directly and includes
the equivalent optional-expression cleanup; no third production run was needed.

In the two captures, 39 allocation observations each cost 0.015/0.016 ms median,
0.100/0.073 ms maximum, and 0.938/0.911 ms total. External OS reads cost
0.060/0.062 ms median. These exclude event dispatch, encoding, I/O, bookkeeping
and other observer effects; no diagnostics-off overhead comparison was made.

The ordinary app was restored after testing. The changes preserve required
checkpoint copies, active speculation, precision, history and leaf-home policy.
Still open: cancellation during lease acquisition/model loading, live projection
weight overlap, explicit SSD hydration/unload physical peaks, long contexts,
alternate layouts/models, and complete loaded-model cache/logit parity. These
are stated limits of these three investigations, not assumed performance wins.
