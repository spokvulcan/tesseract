# Prefix cache, KV and memory: tests and evidence

The suites, opt-in gates and evidence runs behind the prefix cache, the KV
cache's capacity and schemes, and request memory. Running tests, and the rules
every test follows in a parallel run, are in [testing.md](testing.md).

## Request memory timeline (#471)

`event=requestMemory` records the HTTP completion's memory timeline in the
existing durable `Application Support/CacheDiagnostics/<yyyy-MM-dd>.jsonl`
sink. Phase transitions, cancellation signals, and terminal samples also
use notice-level unified logging. Periodic samples run once per second
while the request is preparing, generating, or cleaning up; they use info
level and remain available in the JSONL sink. The existing retention and
rotation limits apply. No prompt text, token IDs, tensor data, or file paths
are recorded.

Filter by `requestID`, then order by `sequence`. `elapsedMs` and
`phaseElapsedMs` use a monotonic clock. A phase-end sample belongs to the
operation that just ran; a phase-begin sample includes the new phase's
component facts. A periodic sample during a stall preserves its phase.

| Fields / phases | What they distinguish |
| --- | --- |
| `activeMlxBytes`, `cachedMlxBytes` | Live MLX allocations versus reusable allocator buffers. |
| `processFootprintBytes`, `processResidentBytes`, `processCompressedBytes`, `systemSwapUsedBytes` | Process footprint versus residency/compression and system-wide swap. Failed OS queries omit the affected fields. |
| `processLifetimePeakMlxBytes`, `sampledRequestPeakActiveMlxBytes`, `sampledRequestPeakFootprintBytes` | The allocator's historical high-water mark versus maxima actually observed during this request. No process-global peak reset is performed. |
| `restoring` → `restored` | Snapshot size, current restore mode (`cold`, `copy`, `failedCopy`), and the resulting cache's attention/recurrent array sizes. `restoreFallback=cold` marks a planned restore that yielded no cache, after which the turn ran cold. |
| `prefilling` → `dflashPreparing` → `prefilled` | Ordinary suffix prefill versus DFlash2's iterator preparation; loaded draft weight bytes, engagement, prompt length, and checkpoint array bytes. |
| `capturingLeaf` → `preparingPayload` → `admittingLeaf` | Capture copy versus handoff, request cache count after capture, actual SSD payload mode/bytes, and admission overhead. |
| `recordingRequest` → `finishingStream` → `releasingRequest` → `finished` | Post-generation bookkeeping, stream delivery boundary, and registry/pin release. `outcome` includes successful, cancelled, failed, and failed/cancelled-start exits. |
| `sampleKind=cancelSignal` | The first cancellation signal and its origin. `streamFinished` is the driver's normal completion cleanup, not a user abort; `caller` / `streamCancelled` distinguish abort signals. |
| `phase=settled sampleKind=afterRelease` | One scalar sample one second after the drive returns. It can overlap a new request and is not an idle-memory claim or part of this request's sampled maximum. |

Component byte counts are observations, **not additive physical ownership
accounting**. Cache `innerState` includes backing capacity; snapshot/payload
views can overlap it; full SSD payloads can share the tree's arrays.
`requestCacheMeasuredAtPhase` and `treeMeasuredAtPhase` identify where the
carried-forward component facts were last read. Tree facts include the
budget, protected floor, and pending SSD payload bytes/count. The sampler
holds only scalars and never reads mutable cache objects off-session,
evaluates a graph, clears memory, or waits for SSD work.

SSD counters preserve the writer's existing accounting:
`ssdPendingPayloadBytes` includes the active writer item's outstanding
budget charge, while `ssdPendingPayloadCount` counts waiting queue entries
only. Nonzero bytes with a zero count can therefore mean a write is already
in progress; neither counter measures exclusively retained physical arrays.

Process counters include co-resident work. One-second samples can miss
short spikes; the process lifetime peak can expose a new spike but cannot
attribute an old one to this request. DFlash2's internal round/capture
buffers are opaque to the app: its preparation interval and process
samples expose their impact, not an exact tensor-by-tensor breakdown.
Unkeyed and MTP preparation retain coarse `preparing` coverage.

Flatten a day's events for inspection (substitute the sandbox's Application
Support path when running the sandboxed distribution):

```bash
jq -c 'select(.eventName == "requestMemory") | {timestamp, requestID, modelID} + (.fields | map({(.key): .value}) | add)' \
  "$HOME/Library/Application Support/CacheDiagnostics/$(date +%F).jsonl"
```

Join `requestID` with the existing `lookup`, `leafStore`, and SSD admission
events to explain restore offsets, fallbacks, and payload completion.

Focused regression suites: `RequestMemoryTelemetryTests`,
`ManagedGenerationDriverTests`, `ServerCompletionKeyedSequencingTests`,
`ServerCompletionDrainTests`, and `PromptCacheDiagnosticsFileSinkTests`.

For the allocation inventory (#506), `requestFullAttentionLogicalBytes`,
`requestFullAttentionArrayBytes` and `requestFullAttentionUnusedArrayBytes`
separate valid rows from unused array extent for plain `KVCacheSimple` layers.
`requestFullAttentionLayerCount` states coverage. These do not measure allocator
padding or larger backings retained by views; `markCacheReleased` resets all carried
cache-byte facts. Other cache layouts retain the existing coarse byte counters.

`TESSERACT_ALLOCATION_DIAGNOSTICS=1` enables scalar `allocationMemory` events at
target/drafter loading and SSD materialization/container-encoding boundaries.
SSD events identify the snapshot and payload/encoded byte counts; they do not
claim exclusive request attribution or physical release. `observedUnixSeconds`
allows external sample alignment. `observationMilliseconds` measures OS sampling
and field assembly, excluding event dispatch, serialization and disk I/O.
The switch is off by default and never evaluates/retains model arrays.

`scripts/allocation_inventory_probe.py` prints its bounded plan by default and
requires `--run` to launch an isolated Release process. It samples process
footprint and OS pressure/swap every 250 ms, enforces response/campaign
deadlines and a bounded release wait, and stops without retry on its resource triggers. Those triggers are
sampled abort conditions, not guaranteed peak ceilings. Defaults stop at warning
pressure, 28 GiB footprint and 512 MiB additional swap. Resource overrides are
`--allow-pressure-warning`, `--footprint-stop-gib` and `--swap-growth-stop-gib`;
record them with every capture and preserve stopped attempts. Do not treat a
five-second quiet interval as SSD drain. The process log is saved as `app.log`
inside the capture directory alongside the runner and scalar evidence.

Allocation events separate target/draft projection stacking and report
`encodedStagingBytes` independently from total `encodedBytes`. The loading path
clears reusable MLX buffers before DFlash2 projection stacking, borrowed payload
chunks avoid a second full encoded buffer, and startup watches disconnects while
the generation handle is being built.

Unload emits `modelUnloadBegin`, `modelUnloadContainerReleased`,
`modelUnloadMTPReleased`, `modelUnloadDFlash2Released`,
`modelUnloadServerCompletionReleased` and `modelUnloadEnd`; the last one
carries `containerRetained`, `mtpDrafterRetained` and `dflash2DrafterRetained`
(weak probes on the released objects, `true` means something still holds
them) and follows the `Memory.clearCache()` that returns the model's
buffers, so its `activeMemory` is what survived the unload. The load path
bounds the MLX buffer cache from the first shard (generation's 2 GB limit),
reads the MTP head from its own file (by the safetensors index, or by the
files' headers when a single-file checkpoint has none: on Qwen3.8-27B PARO with
a grafted head the whole-file read peaked 32.5 GB, and the head now adds
nothing to the load's 19.0 GB peak), clears the cache after the head loads,
and packs the DFlash2 draft leaf by leaf
from its unread bfloat16 checkpoint instead of reading the whole file first
(`modelDFlash2LoadBegin` to `modelDFlash2Loaded` peaks 0.3 GB over the
resident target on the 27B pairing, where the whole-file read peaked 4.5 GB
over it). A reload-only run
(`TESSERACT_E2E_RELOAD_ONLY=3`) on 2026-09-20 with `qwen3.8-27b` held
`modelUnloadEnd` at 0.76 GB active across four loads (the proofread model),
where the previous vendor pin grew 2.3 GB per load (the fused GDN projection
read as a compile constant; see `docs/mlx-swift-lm-fork.md`).

Focused coverage includes `CompletionDeliveryTests` for startup cancellation and
handle ownership, `PlaceholderContainerEncodingTests` for borrowed addresses,
golden full/suffix bytes and write failure, and the SSD store, snapshot-ledger
and leaf-extension suites for real commits/restores. Capture results and their
limits belong in the [allocation inventory](research/2026-09-12-local-inference-allocation-inventory.md)
and [follow-up investigations](research/2026-09-13-allocation-investigations.md).
These captures do not replace loaded-model bitwise cache/logit parity or
long-context gates.

The probe accepts `--comparison-label` to label a capture and
`--max-cancel-signal-delay-seconds 1` to check prompt cancellation signaling.
The delay subtracts client and server elapsed clocks with slightly different
start points, so small negative values reflect clock alignment. The temporary
loading experiment switch exists only in archived sources and runners.

## Bounded production parity and projection lifetime

`--hybrid-cache-correctness --bench-bounded-cache-parity` selects a fixed
2,048-token, unquantized-KV gate instead of the full matrix. It loads the target
and DFlash2 with an explicit `.dflash2` policy; a bare benchmark `AgentEngine`
otherwise defaults to `.automatic` and also loads MTP. This matches the measured
server configuration without changing preferences. It checks raw cache bytes,
metadata, logits, checkout/rewind ownership and real full/extension SSD restores.
The gate does not run speculative decoding; use the separate HTTP replay for
that behavior. It cannot be combined with `--bench-replay-request`.

`scripts/bounded_cache_parity.py` prints the fixed plan without `--run` and
wraps this gate in fixed resource stops (32 GiB sampled footprint, 6 GiB minimum
available memory, 1 GiB additional system swap, critical/unknown pressure,
ten-minute deadline). Use a Release binary,
a new output directory and one validation process. Quit the app first and
restore it afterward. The scratch SSD store is flushed and removed on success,
thrown failure and cooperative cancellation. `BoundedCacheParityTests` exercises
these exits with a queued write. A forced process kill cannot run Swift cleanup;
inspect the output for `scratch-*` directories before archiving it. Captured
results are in the [evidence archive](../benchmarks/allocation-parity/2026-09-13/README.md).

`CacheStateBytesTests` checks that the exact-byte observer detects mutations and
structural differences. `ProjectionStackingLifetimeTests` is a separate small,
model-free traversal experiment. Its peak counter is process-global, so it is
disabled by default and must run alone:

```bash
TEST_RUNNER_TESSERACT_PROJECTION_LIFETIME_EVIDENCE=1 xcodebuild test \
  -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation -parallel-testing-enabled NO \
  -only-testing:tesseractTests/ProjectionStackingLifetimeTests
```

The incremental visitor exists only in that test. See the
[parity and lifetime report](research/2026-09-13-cache-parity-and-projection-lifetime.md)
for measurements and the remaining production validation boundary.

## Controlled capture comparison (#478)

`scripts/capture_memory_replay.py` sends one private HTTPRequestLogger
recording to an isolated loopback server and saves scalar diagnostics, usage,
request/response hashes, and the request ID. It uses greedy decoding with a
128-token output ceiling by default; use **the same settings and request
bytes on both builds**. This is a bounded capture experiment, not a replay
of every historical generated token or the full #480 long-session gate.

```bash
python3 scripts/capture_memory_replay.py \
  --request /private/path/to/recording-request.json \
  --output /private/path/to/handoff.json \
  --label handoff --source-revision BUILD_REVISION \
  --expect-capture-mode handoff \
  --next-request /private/path/to/warm-request.json
```

The default endpoint is `127.0.0.1:18321`. Launch the app with
`-serverPort 18321 -prefixCacheSSDDirectoryOverride /private/path/to/cache`
to isolate the experiment from ordinary clients and their disk cache. These
launch arguments override preferences for that process. Use fresh processes
and equivalent SSD/RAM/index state for the comparison; record any restarts.
Quit the app before running Xcode tests and relaunch it afterwards.

The optional continuation file contains **private request and response
content** and is created with mode `0600`; keep it outside the repository.
Tool calls receive a synthetic tool result and are never executed. Reuse
the same continuation fixture on the other build. For cancel/resend, send
that fixture once with `--cancel-after-first-delta`, then again without it.
Response hashes include tool-call IDs, so generated IDs can differ even if
function names and arguments match.

The script fails on concurrent request timelines, diagnostics rotation,
missing release telemetry, or an unexpected capture mode. Its observation
window ends at `afterRelease`; that is **not** an SSD-drain or idle-memory
guarantee. System-scoped SSD materialization events observed in that window
are retained separately in `systemEventsObserved`, with snapshot IDs, and
must not automatically be attributed to the current request. Later writer
events remain in the durable diagnostics sink. Honor the component facts'
measurement phases when reading the carried-forward fields.

The [2026-09-08 capture comparison](../benchmarks/capture-handoff/2026-09-08/README.md)
includes paired request IDs, a compact machine-readable baseline, and a
checksum-verified download of the detailed diagnostic extracts and generated
reports. It documents an explicit recovery from diagnostics rotation. Recovery requires the retained old and current
files to contain a continuous sequence from the first request sample through
terminal and after-release observations. Missing response metadata must stay
unavailable; a complete memory timeline does not recover response parity.

For a bitwise correctness check on a real recording, the loaded-model
runner also accepts `--bench-replay-request <recording>`, together with
`--hybrid-cache-correctness` and the usual `--bench-model`, `--bench-model-id`,
and `--bench-output` arguments. This selects one check instead of the default
correctness matrix: prefill the recorded prompt through all but its final
16 tokens, capture it both by copy and by move, restore each by copy, and
compare the continuation's final logit **bytes**. It uses the production
normalization and template context with preserved thinking, rejects image
requests, and logs only the input hash and counts. It tests unquantized
capture/restore correctness; DFlash2 and HTTP timing are covered by the
separate live replay.

The Qwen3.8 community checkpoint used by this comparison loads the vision
class when vision is requested (ADR-0089), so the HTTP E2E runner's image
scenario runs on it; this correctness check itself rejects image requests.

## Tree-side Leaf Lease evidence (#479)

`LeafLeaseTests` exercises body-drop refusal, pressure and a Cache Claim's
release of its pins and lane, RAM clear, demotion and queued promotion,
same-path replacement, ancestor supersession, check-in growth, both
writer/acquisition race orders, writer failures, and base/suffix ordering. All
caches are small, real MLX caches; the production check-out is covered by the
Cache Claim suites below.

Review regressions also cover pending-only return destinations, a tombstoned
writer still reading an empty structural destination, request/lease identity
on refusal, and mandatory SSD admission after explicit return. An admission
attempted during a lease reports `StoreDiagnostics.leaseRefusals`; its retry
after return still bypasses the pending-byte cap. Run
`StorageActivityGateSchedulingTests` alongside this suite when changing the
writer drain so ordinary forced flush remains covered too.

Run `LeafLeaseMemoryEvidenceTests` **alone** for process-memory observations:

```bash
xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
  -destination 'platform=macOS' -skipPackagePluginValidation \
  -parallel-testing-enabled NO \
  -only-testing:tesseractTests/LeafLeaseMemoryEvidenceTests
```

Quit the app before testing and relaunch it afterward. This test loads no
model weights itself. It performs 24 success/cancellation/error simulations
with a 4,160-byte hybrid body, checks cache-object and array release while
retired lease tokens remain alive, and prints a `LEAF_LEASE_EVIDENCE=` JSON
record. Correlate its request IDs with the `leafLease*` and `requestMemory`
events in the test log. Lease acquisition should add no active MLX bytes;
after return the freshest leaf still belongs to the Budget Floor until an
explicit RAM clear. Allocator cache bytes can remain after live arrays die.
Hosted-app process footprint includes startup work, logging and other
components, so these small-cache checks do not establish the #480
long-context footprint reduction.

`treeLeasedBytes` and `treeLeaseCount` accompany the existing tree/floor and
SSD pending counters in `requestMemory`. Lease events carry the request ID,
lease ID, original offset and bytes; end events add returned offset/bytes,
growth, and `checkIn` or `rewind`. Writer deferral includes the snapshot ID.
Only explicit quiescent check-in/rewind ends a lease; a Cache Claim's release
of its pins and lane (the tripwire's included), forced SSD flush and
write-eagerness timeout cannot do so.

See the [preserved small-cache evidence](../benchmarks/leaf-lease/2026-09-12/README.md)
for the before/after ownership table, request IDs, diagnostic extracts and
limits, and the [review follow-up](../benchmarks/leaf-lease/2026-09-12-review/README.md)
for the additional return, admission and flush regressions. The large-model
approval requirement in the capture baseline still applies to #480.

## Production Leaf Checkout and Rewind evidence (#480)

The check-out now belongs to the Cache Claim (#554), and its attempt cases
moved from the deleted `LeafCheckoutTests` to `CacheClaimTests`, described
in the next section. They check object identity and physical array independence,
body removal/accounting, exact recurrent state and metadata after growth,
every intentional fallback, and pending-full-payload materialization. They also
cover the bounded pending-payload wait (#523): a payload that materializes
inside the bound becomes a handoff, one that outlasts it copies and reports the
waited time, and a payload still queued behind other writes copies at once
without waiting. `SSDSnapshotStoreTests` covers the writer's own answer —
queued versus in the writer's hands — that the wait turns on; the shared
`BlockingMaterializer` in `tesseractTests/PrefixCacheTestFixtures.swift` parks
the writer inside one payload's materialize step so both are deterministic.
`EmittedPathSynthesizedReplayTests` covers cancellation during decode and warm
prefill, including the unload drain, followed by a resend that hits the original
leaf, and the vision-container text-only session (Bonsai 2 27B, the PARO
Qwen3.5 pack: 2D prepared tokens) registering and serving the Emitted Path
like a flat-token instance; `RequestKeyingPhaseInstanceTruthTests` pins that
such a request tokenizes through the Render+Token Cache at the processor's
rank with no `prepare` verb. `ServerCompletionKeyedSequencingTests` also covers a zero-output direct
turn that returns its original leaf and reports rewind without capturing an
empty cache. `HybridCacheSnapshotTests` covers copied recurrent metadata,
including lengths and nil/present padding. Run these alongside
`LeafLeaseTests`, `ServerCompletionDrainTests`, and
`ServerCompletionKeyedSequencingTests`; the latter includes real quantized
cache replacement so the copy path cannot accidentally retain stale objects.

Run `LeafCheckoutMemoryEvidenceTests` alone with the Xcode flags above and
`-only-testing:tesseractTests/LeafCheckoutMemoryEvidenceTests`, prefixed with
`TEST_RUNNER_TESSERACT_ISOLATED_LEAF_CHECKOUT_EVIDENCE=1`. The flag enables
process-global allocator thresholds; leave it unset for the full target or any
run with other MLX suites. Suite serialization cannot exclude allocations from
other suites. Identity/state/release assertions always run. It uses a 2 MiB
attention body and a 64-byte recurrent state, retains 24 retired request
owners, and verifies that check-in/rewind releases their cache references.
`LEAF_CHECKOUT_EVIDENCE=` contains active/cached MLX, process footprint, system
swap, and tree/lease facts for each request. These are small-cache mechanism
measurements, with no model weights loaded by the test. The full unit target
may run this test among others with allocator thresholds disabled; use only
the isolated run for process-memory comparisons. The JSON records whether
allocation assertions were enabled.

[Preserved #480 evidence](../benchmarks/leaf-checkout/2026-09-12/README.md)
contains the scalar records, baseline table, 85-recording tokenizer replay
summary and pending large-model validation plan. The tokenizer-only corpus
test does not measure live HTTP tail or Qwen3.8/DFlash2 memory. Do not repeat
45k/75k/93k workloads or model reload loops on the 48 GiB Mac without explicit
owner approval of a bounded resource plan and a suitable environment.


[PR #503 review follow-up evidence](../benchmarks/leaf-checkout/2026-09-12-review/README.md)
records each external finding's disposition, the final clean full-target run,
and the explicitly isolated allocation run after these hardening changes.

## Cache Claim (#554)

`CacheClaimTests` goes through the claim's interface on the real manager and
tree: a miss holds only its lane and a hit also pins its path; a start that
throws and a cancelled drive each conclude once; the check-out's typed outcome,
with its copy reason, precise refusal and waited time, for every refusal and
wait case; check-in committed, refused and cancelled; the leaf back before the
pins and the lane; a copy-only claim; the tripwire's violations in reporting
mode, including a lease it cannot return; and a leaf loaded from SSD handed off
on its loaded arrays with its SSD ref kept, while a loaded checkpoint still
copies and says why.

`ServerCompletionExitMatrixTests` runs every way a keyed request can end
through the Server Completion fixture on the toy Model Session: a completed
handoff, copy and cold turn, a startup cancel by the caller and by the drain, a
decode cancel, a suffix prefill that fails after the handoff (decode has no
failure exit), a refused check-in, a think-stripping turn and the direct-turn
guard. Each case waits for the drive, then checks that the leaf is back (no
lease, and a resend hits at its offset), that no pins or lane remain, and that
the request was released exactly once. `CompletionDeliveryTests` and
`ServerInferenceServiceTests` check that a completed request's delivery waits
for its drive on the HTTP and agent-chat paths. The compaction retune's toy
decodes are in `ServerCompletionKeyedSequencingTests`, and the SSD restore
harness expects the loaded leaf to be handed off with no restore call.

`ServerCompletionRestoreFallbackTests` covers a planned restore that yields no
cache (ADR-0069's 2026-10-03 amendment). An armed `ToyRestoreFault` makes the
toy session's next `restore` throw, `Injected` by default or a given error
such as `HybridCacheSnapshot.RestoreError`, and records the body it failed on.
A text-only turn restored by copy and a restore planned below a new image must
then both run cold: the whole prompt fed from position zero into a new cache,
each captured checkpoint labelled with the offset its cache held
(`ModelVerbRecorder.captures` records both), and every body left in the cache
reading back the path it is stored under
(`PrefixCacheAdmin.residentSnapshotsForTesting`; the toy writes each fed id
into its K/V row). A resident MTP drafter that traps if engaged
(`Speculation.inactiveMTP`) checks that the fallback keeps its Speculation
Plan. The failed snapshot is dropped: a system checkpoint that failed is gone
after the turn, recaptured on the same turn, and restored by the next request;
over an SSD tier a `RestoreError` also removes its old copy from the manifest,
and any other error keeps it.

`CacheClaimMemoryEvidenceTests` measures the MLX peak around one step at a time
on synthetic caches: check-in before extraction, a refused check-in, the
check-out's only allocation, per-layer compaction, and an SSD hit handed off.
The peak counter is process-global, so the byte assertions need the suite to
run alone:

```bash
TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests \
TEST_RUNNER_TESSERACT_CACHE_CLAIM_MEMORY_EVIDENCE=1 xcodebuild test \
  -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation -parallel-testing-enabled NO \
  -only-testing:tesseractTests/CacheClaimMemoryEvidenceTests
```

Without the flag the steps still run and their functional assertions hold, and
the measured bytes are printed either way (`CACHE_CLAIM_EVIDENCE=`). ADR-0069's
as-built notes record the numbers.

## Opt-in Warm Bodies (#527, #529)

`WarmBodyModelSessionTests` uses microscopic fp16/fp32 toy-model caches to check
compression and restore token parity, backing-address isolation, and whole-state
byte preservation. `WarmBodyDrainTests` checks compression before demotion,
exemptions, quantized byte accounting, copy-only checkout, default-off behavior,
and full-form SSD demotion/hydration with a temporary directory.
Its opportunistic cases cover RAM above, at and below the ceiling fraction,
default-off behavior, the default two-path Hot Leaf Set, a configured one-path
limit, successful Lease check-ins, leased paths outside that set, and waiting
for the occupied toy Model Session to quiesce. These tests observe the tree
without refreshing the cold leaf's Budget Floor recency.
`PrefixCacheDiagnosticsTests` pins `warmCompress` fields, including
`source=opportunistic` versus `source=drain`, and lookup `source=warm`;
the manager telemetry test checks the hot/warm byte totals used by the cache panel.

Use `TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests` for these app-host
unit tests. `DependencyContainer.setup` skips service bootstrap in the test host,
so tests cannot trigger model prewarms. Loaded-model parity/TTFT measurements and
the #528 enablement gate remain owner work; the default flag is off.

## Boundary leaf capture by move (ADR-0064 amendment, #501)

`EmittedPathSynthesizedReplayTests.thinkStrippingTemplateKeepsTheBoundaryPathAtAUserBoundary`
is the behaviour, alongside its pre-existing boundary-path assertions: a
think-stripping boundary turn reports `source=handoff` and
`leafCaptureMode=handoff` with no `copy` sample, and the next turn restores by
handoff instead of being refused for `immutableBody`.
`HybridCacheSnapshotTests.canCaptureMovingAgreesWithWhatAMoveActuallyTakes`
pins the pre-check against `captureMoving` itself, including the quantized
refusal that keeps the deep copy — the toy Model Session cannot build a real
`QuantizedKVCache`, so that guard is covered as a pure predicate, not through
the replay harness. Both now ask `movableClassName`, so the predicate and the
move cannot disagree; the test guards the pair rather than holding it together.
`ServerCompletionKeyedSequencingTests.canonicalFallbackRestoresAPlannedBranchView`
and `thinkStrippingTemplateKeepsTheBoundaryPathAtAUserBoundary` keep asserting
`path=boundary`; only their `source` moved from `boundary` to `handoff`.
The loaded-model evidence and the session audit behind the change are in
`benchmarks/boundary-leaf-move/2026-09-20/`.

## Backing Leaf credit (ADR-0068 amendment)

`EvictionPolicyTests.aViewHitCreditsItsBackingLeaf` checks that a lookup served
through a stored view refreshes the Backing Leaf's recency and hit count;
`SnapshotResolutionTests.storedAndTransientViewsChooseAndPinThePreferredWarmOrFullBacker`
checks the same for the transient boundary path.
`EvictionPolicyTests.aSoleBackingLeafRecoversFromItsViewsParent` pins the
terminal recovery span on a pure tree: through the view while the leaf is its
only backer, bounded at the view once a second backer or a committed ref exists.
`aSoleBackerOutranksAnEqualLeafUnderAFullBodyParent` checks the score ordering.
`ServerCompletionKeyedSequencingTests.thinkStrippingTurnRetainsOnlyWholeStateBoundaryBytes`
also checks that the live leaf checked in as the transient views' backer is
released once the canonical leaf is admitted: one resident leaf per boundary
turn, reported as a `leafSupersession` with mode `released`.
`EvictionPolicyTests.releasingTheBoundaryBackingLeafDropsOnlyAnExactUnleasedLiveLeaf`
pins the release's guards on the manager: the canonical path, a shallower
prefix, a foreign path and a leased body release nothing.
The loaded-model `prefix-cache-e2e` branch-point survival check is the
end-to-end evidence. Since ADR-0068 a planned branch point is a view with no
bytes of its own, so the check no longer counts a branch-point body outliving
interleaved noise requests: it cuts the budget by three leaves at alpha=2
without new requests, then requires the branch view to keep a Backing Leaf and
a request on the branch prefix to hit past the stable prefix.

## Warm-backed Prefix-View Checkpoints (#530)

`PrefixViewModelSessionTests.warmBackerMaterializesPrivateViewStateAndFullPayloadWithTokenParity`
uses fp16/fp32 hybrid toy caches to check prefix-only attention values, shapes and offsets,
the view's own recurrent state, deterministic token parity with an uncompressed
backer, and physical-address isolation. It also checks full-form SSD payload
pricing, shape and detached ownership; quantized Stored Form remains #531 work.
`SnapshotResolutionLadderTests.viewPrefersNearestBackerThenFullFormBeforeRecency`
checks nearest warm selection and the uncompressed tie-break.
`SnapshotResolutionTests.storedAndTransientViewsChooseAndPinThePreferredWarmOrFullBacker`
checks the manager's composition for stored and transient views, Restore Pins,
unchanged Leaf Checkout refusal, and separate hot/warm/view-only byte totals.
`PrefixCacheDiagnosticsTests.viewLookupReportsTheBackingLeafForm` checks both
backer forms while keeping `source=view` and the `checkpoint` copy reason;
the existing keyed Server Completion sequence verifies the emitted field.
Use the same app-host guard as above, and the prefix suite allowlist in
[testing.md](testing.md#unit--integration-suites). No loaded-model
work is part of this unit-test evidence.

## Warm Body parity pre-registration (#528)

The [pre-registered owner gate](../benchmarks/warm-body-parity/2026-09-19/README.md)
defines fp16 restore-by-copy, warm-8 and experimental warm-4 arms, fidelity and
paired TTFT thresholds, memory observations and a mandatory owner resource
manifest. The [2026-09-20 owner run](../benchmarks/warm-body-parity/2026-09-20/README.md)
executed it: **the 8-bit gate failed in all four cases** (deterministic greedy
divergence from the fp16 control; warm-8's paired TTFT excess above its
dequantize allowance in two cases). Warm Bodies remain default-off.

The runner is `--warm-parity-bench --warm-parity-plan <owner-plan.json>`
(`WarmBodyParityBenchRunner`), launched only through
`scripts/warm_body_parity.py --app <Release binary> --plan <manifest> --output <new dir>`,
which refuses a manifest that is not `APPROVED` or committed, verifies the
binary and model checksums, samples footprint/available/pressure/swap every
250 ms and terminates the harness on a breach without retry. Per case the
runner runs one warmup block and the six pre-registered arm orders; per
observation it clears the RAM tier, arms the form through `PrefixCacheAdmin`
(`setWarmCompression`, `setLeafCheckoutDisabled` for the control), runs the
setup turn(s) plus a short unrelated turn so the Budget Floor lets the case
leaf compress, verifies the resident form, times the dequantization
allowance on that body at the Model Session seam, then times the hit (TTFT
from submission to first delta) and walks Canonical-Echo fidelity over the
arm's own recordings. Records are appended as they exist
(`observations.jsonl`, `dequantize.jsonl`, `copy.jsonl`, `fidelity.jsonl`);
`benchmarks/warm-body-parity/2026-09-20/verdicts.py` applies the rules to
them.

`CanonicalEchoFidelityCorpusTests` reads tokenizer files and checks token
paths; it did not see the greedy divergence and is not warm-restore evidence.
The prefix suites in [testing.md](testing.md#unit--integration-suites) remain the small-cache
regression evidence; their
success does not flip the flag or unblock #531.

## Attention capacity compaction (#534)

`AttentionCapacityCompactionTests` drives real `KVCacheSimple` layers: a long
trimmed generation compacts to the offset's rows plus one step with fresh
backing addresses and identical logical rows; below the threshold nothing
changes; the threshold is the smaller of a quarter of the body and 64 MB;
quantized and recurrent layers are untouched. The `leafRewind` event reports
`fullAttentionArrayBytes`, `fullAttentionLogicalBytes`,
`fullAttentionUnusedArrayBytes` and `compactedBytes`; the `leafStore` event
and the `capturingLeaf` memory sample report `compactedBytes` /
`leafCompactedBytes`. The threshold's source measurement is the
[2026-09-21 cancelled-generation profile](../benchmarks/allocation-profile/2026-09-21/README.md)
(`scripts/cancelled_generation_profile.py`, under a committed plan with the
48 GiB stops), which found 50–117 MB retained after a cancelled generation
and 5–17 MB at ordinary check-in, inherited by the next turn's live leaf.

Since #554 compaction builds, evaluates and swaps in one layer at a time, so
a later layer may reuse an earlier layer's freed buffer: the test checks each
replacement against the arrays it replaced. The
[2026-09-22 retune measurement](../benchmarks/allocation-profile/2026-09-22/README.md)
kept the threshold and the one-step target by a rule registered before its
numbers were read.

## No-copy SSD writer (#469)

`SnapshotPayloadTests` checks borrowed backing addresses,
Data/array lifetime, empty arrays and release after each streamed layer.
`PlaceholderContainerEncodingTests` pins bounded borrowed chunks and the existing
full/suffix golden files. `SSDSnapshotStoreTests` compares normal-leaf and demotion
files with the copying encoder and checks `ssdPayloadPrepare`, `writeMs` and
`enqueueToCommitMs`. Also run `SnapshotManifestTests`, `TieredSnapshotStoreTests`,
`LeafLeaseTests`, `LeafCaptureHandoffTests`, the four `LeafExtension*Tests` suites
and the four `ChainPrefix*Tests` suites with the prefix-cache group.

Use `TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests` for the app-host
unit runs; test bootstrap skips model prewarms. The existing store INT_MAX test
writes a synthetic host-byte file over 2 GiB, without model weights or KV prefill.
The [no-copy writer measurement plan](../benchmarks/no-copy-writer/2026-09-19/README.md)
records the outstanding owner-run 3 GB leaf/demotion comparison. It was not run
as part of the implementation.


## Attention cache capacity (#533)

`ServerCompletionKeyedSequencingTests.creationAndRestoreReservePromptRowsWithoutReservingOutput`
drives cold creation, Leaf Checkout and copied quantized restoration through the
Model Session toy peer. A chunked prompt reserves its total rows before prefill;
a large output ceiling does not become a reservation. The recorders observe
backing capacity through the existing Model Session toy peer. `toyDecodeUsesGeometricCapacityGrowth`
decodes 2,048 toy tokens and observes four capacity allocations (256, 768, 1792,
3840 rows), with no model weights. Growth is geometric until the 4096-row
increment cap, then bounded linear increments; this is not a production timing
measurement. `HybridCacheSnapshotTests` covers snapshot restore and buffer
isolation under the new vendor allocation policy.

`RawGenerationStartTests.rawCreationReservesTheWholePrompt` covers both raw
Prefill Strategy routes. `canonicalLeafRestoreReservesTheStoredPath` exercises
the canonical Leaf Store restore with a think-stripping template;
`SpeculativePrefillPreemptionTests` checks that both extension chunks retain one
reservation for the entire admit path even when cancellation ends the pass.
These observations remain scalars in the toy model; no new production seam was
introduced. Run the raw-generation, Prefill Strategy, Speculative Canonical
Prefill and Generation Logit Processor suites with the prefix-cache suites.

The vendor's `CacheCapacityTests` covers simple and quantized reservation,
growth to the cap, unchanged trim/state/metaState/copy and prompt-cache
serialization, preservation across dynamic quantization, and nested CacheList
forwarding. Run it with the existing vendor cache serialization/copy tests:

```bash
scripts/vendor-test.sh CacheCapacityTests 'testCacheSerialization(creator:)' \
  'testCacheCopyIsIndependent(creator:)' 'testCacheCopyOnEmptyCache(creator:)'
```

The vendor's load-memory regressions measure MLX active memory around a
model or a stacking pass: `testCompiledDecodeReleasesFusedProjectionWithTheModel`
(`Qwen35FusedGDNProjectionTests`, a dropped fused model leaves under 64 bytes
resident), `testSameInputStackingReleasesEachBlockBeforeTheNext`
(`DFlash2Tests`, the stacking transient stays within two blocks) and the
`SiblingCycleTests` probes (which sibling-graph drop paths release their
inputs; the two open upstream cases are expected failures). Run them after
any change to the compiled traces, the projection fusion or stacking, or the
loader:

```bash
scripts/vendor-test.sh Qwen35FusedGDNProjectionTests SiblingCycleTests \
  LoadWeightsTests 'testSameInputStackingReleasesEachBlockBeforeTheNext()'
```

App test runs set `TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests`;
the test host returns before DependencyContainer starts model prewarms or
background services. Loaded-model and long-context measurements are owner work.

## SSD read experiment (#532)

The three read arms use the existing `SSDSnapshotStoreTests` temporary-file
seam and `ChainPrefixHydrationTests`, with rendered fields covered by
`PrefixCacheDiagnosticsTests`. These tests load no model. Run them along with
the prefix cache suites in [testing.md](testing.md#unit--integration-suites). The owner-only `--ssd-read-bench` harness and
pre-registered throughput/adoption protocol are documented in
[`benchmarks/ssd-read/2026-09-19/README.md`](../benchmarks/ssd-read/2026-09-19/README.md).
The owner run of 2026-09-21
([`benchmarks/ssd-read/2026-09-21/README.md`](../benchmarks/ssd-read/2026-09-21/README.md))
was a null result: sequentialMap 1.09x and positional 0.73x of mapped on a
61k-token chain, under the 2.0x gate, so production keeps `mappedIfSafe` and
the arm selector stays harness-only.
Do not run the loaded-model harness as part of automated verification.

## TurboQuant KV measurement (#603)

`--turboquant-bench` (`TurboQuantBenchRunner`) measures a live KV cache scheme
against the unquantized cache on a loaded model, with the scheme the only
change between arms. Per context it prefills once, snapshots the bf16 cache,
and restores a copy for each pass. Quality is teacher-forced, one token per
forward, against the unquantized arm's greedy stream: KL divergence and top-1
agreement, with a re-chunked prefill as the noise floor. Speed runs the
chunked Prefill Strategy route (`PrefillExecutor.makeIterator`) in reversed
rounds. Memory reads the realized cache's bytes
per token and the prefill and decode-phase peaks. The harness fails if a scheme
leaves any attention layer unconverted, if step 0 (scored before the scheme
engages) has nonzero KL, or if the unquantized speed rounds stop reproducing
the reference stream. Run it in Release through `scripts/bench.sh` and
summarize with `scripts/turboquant_summary.py`:

```bash
scripts/bench.sh quick --model qwen3.8-27b --turboquant-bench \
  --bench-corpus docs/adr --bench-contexts 8192,32768,65536 \
  --bench-schemes fp16,turbo8v4,turbo0v4
```

The 2026-10-02 run on the 48 GB M3 Max
([`benchmarks/turboquant/2026-10-02/README.md`](../benchmarks/turboquant/2026-10-02/README.md))
passed the quality bar and failed the speed bar with the vendor as shipped:
turbo8v4 decoded 51% slower than bf16 at 32K and 64% slower at 64K, with an
unchanged run peak, and TurboQuant decode was not reproducible run to run (a
data race in the vendor's value encoder). With the GQA decode kernels the
vendor pin now carries (`docs/mlx-swift-lm-fork.md`, "TurboQuant GQA decode";
the same README) both bars pass: decode within about 2% of bf16 at 8K–64K, and
two decodes from one cache give the same tokens. The app harness takes about an
hour; do not run it as part of automated verification.

Run the kernel tests after any change to the vendor's TurboQuant kernels or
cache. `TurboQuantDecodeMicrobench` in the same package is an opt-in per-layer
timing (`TEST_RUNNER_TURBOQUANT_DECODE_BENCH=1`, with `-configuration Release
ENABLE_TESTABILITY=YES`):

```bash
scripts/vendor-test.sh TurboQuantGQAFlashTests TurboQuantIntegrationTests
```

### KV Scheme in production (ADR-0083)

The **KV Scheme** (`turbo8v4`, `turbo0v4`) is a request fact, part of the
cache partition key and the partition's Stored Form. Run the vendor verify
tests after any change to positioned rows, the multi-query verify kernel or
the DFlash2 cache protocol. The script runs them serially (Swift Testing
otherwise runs the parameterized iterator test's cases at once, and two
threads tracing MLX compiles deadlock):

```bash
scripts/vendor-test.sh TurboQuantVerifyTests \
  'testDFlash2IteratorOverTurboQuantCache(keyBits:)'
```

`TurboQuantVerifyTests` checks the kernel against dequantize + SDPA (raw and
affine keys, lengths up to 4,100, ragged blocks, a lazy position), scratch
rows past the offset, growth that keeps them, causal chunks over a compressed
cache, the state round trip and `copy()`, and Qwen 3.5's verify over
TurboQuant against the plain cache. The opt-in
`TurboQuantDecodeMicrobench/testVerifyAttention` times one verify pass's
attention against bf16 SDPA, and `testWarmChunk` a prefill chunk over a
compressed cache (`TEST_RUNNER_TURBOQUANT_DECODE_BENCH_CHUNKS=64,512,1024`).

App side: `TurboQuantSnapshotTests` (capture, restore, move, prefix views,
the Stored Form, the SSD round trip), `SpeculationPlanTests` (a scheme keeps
DFlash2 and refuses MTP), `RequestFactsTests` and `SnapshotManifestTests`
(the scheme in the partition key, digest and meta),
`ServerCompletionUnkeyedSequencingTests` (an Unkeyed Completion decodes over
the scheme's layers after both the text and the anchored vision prefill).

Loaded-model checks: `scripts/dflash2-bench.sh --bench-kv-scheme turbo8v4`
runs both arms in the scheme (prompts from `--bench-prompt-file`), and
`TESSERACT_E2E_KV_SCHEME=turbo8v4 scripts/dev.sh prefix-cache-e2e
--bench-model-id qwen3.8-27b` runs the e2e's requests in it.

## A Day Thread picture seen in its turn (ADR-0090)

The e2e's Step P loads the model for vision on an SSD-backed engine and sends a
Day Thread rendered through `DayThreadPictures`: a text turn, the picture's
turn, the request after it (the picture now a line) and the one after that.
The picture's turn must restore the thread before it; the request after must
restore as far (the boundary the picture turn's leaf extended, from SSD by
Chain-Prefix Restore) and nothing past the picture; the one after must restore
the whole request before it. It runs last in a full e2e, or alone:

```bash
TESSERACT_E2E_ONLY=day-thread-pictures scripts/dev.sh prefix-cache-e2e --bench-model-id qwen3.5-4b-paro
```

