# Testing

Tests use the Swift `Testing` framework (not XCTest), in `tesseractTests/`. Run
before committing changes to server, caching, or agent engine code.

## Unit / integration suites

The suite lists below are recommended *focused* runs for the hottest areas;
there are ~100 suites in total — discover the rest with
`grep -r "@Suite" tesseractTests/`.

```bash
# Server + agent suites (recommended for fast, focused runs):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/HTTPPrefixCacheSpikeTests \
  -only-testing:tesseractTests/HTTPPrefixCacheSessionReplayTests \
  -only-testing:tesseractTests/CompletionHandlerTests \
  -only-testing:tesseractTests/CompletionDeliveryTests \
  -only-testing:tesseractTests/SSEDeliverySinkTests \
  -only-testing:tesseractTests/SSEDeliveryPumpTests \
  -only-testing:tesseractTests/StreamLifecycleDriverTests \
  -only-testing:tesseractTests/CompletionRouteTests \
  -only-testing:tesseractTests/ServerInferenceServiceTests \
  -only-testing:tesseractTests/ServerCompletionDrainTests \
  -only-testing:tesseractTests/ServerCompletionLeafStoreModeTests \
  -only-testing:tesseractTests/ServerCompletionLeafSkipLogTests \
  -only-testing:tesseractTests/LeafStoreFastPathTests \
  -only-testing:tesseractTests/CompletionProjectionTests \
  -only-testing:tesseractTests/MessageConverterTests \
  -only-testing:tesseractTests/OpenAITypesTests \
  -only-testing:tesseractTests/AgentEngineToolSpecTests \
  -only-testing:tesseractTests/GenerationStreamLoopTests \
  -only-testing:tesseractTests/ManagedGenerationDriverTests \
  -only-testing:tesseractTests/ReasoningEffortTests \
  -only-testing:tesseractTests/RawGenerationStartTests \
  -only-testing:tesseractTests/EditToolTests

# Prefix cache suites (radix tree + hybrid snapshot + stable prefix detector):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/HybridCacheSnapshotTests \
  -only-testing:tesseractTests/LeafCaptureHandoffTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/CacheKeySpaceTests \
  -only-testing:tesseractTests/PrefillPlannerTests \
  -only-testing:tesseractTests/LeafAdmissionBuilderTests \
  -only-testing:tesseractTests/ConversationRenderSourceShapeTests \
  -only-testing:tesseractTests/ConversationRenderProbeParityTests \
  -only-testing:tesseractTests/ConversationRenderProbeParityRealTests \
  -only-testing:tesseractTests/SnapshotResolutionTests \
  -only-testing:tesseractTests/SnapshotLedgerTests \
  -only-testing:tesseractTests/SnapshotStateTests \
  -only-testing:tesseractTests/LeafHomeGuaranteeTests \
  -only-testing:tesseractTests/StablePrefixDetectorNonDeterminismTests \
  -only-testing:tesseractTests/JinjaNonDeterminismReproTests \
  -only-testing:tesseractTests/EmittedPathIndexTests \
  -only-testing:tesseractTests/EmittedPathFidelityTests \
  -only-testing:tesseractTests/EmittedPathRegistrationTests \
  -only-testing:tesseractTests/ConversationRenderEmittedPathTests \
  -only-testing:tesseractTests/EmittedPathResolveRealTests \
  -only-testing:tesseractTests/EmittedPathReplayGateTests \
  -only-testing:tesseractTests/EmittedPathSynthesizedReplayTests \
  -only-testing:tesseractTests/LinearStreamingDetokenizerTests \
  -only-testing:tesseractTests/LinearStreamingDetokenizerRealTests

# Voice session + barge detector (quit the app first — its capture engine
# starves test hosts; VoiceBargeReplayTests replays real-hardware traces from
# tools/voice-hold-lab, see its RUNBOOK for regenerating fixtures):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/VoiceEndpointerTests \
  -only-testing:tesseractTests/EchoResidualFloorTests \
  -only-testing:tesseractTests/VoiceBargeReplayTests \
  -only-testing:tesseractTests/CompanionVoiceSoftBargeTests \
  -only-testing:tesseractTests/VoiceCaptureSessionTests \
  -only-testing:tesseractTests/VoiceProcessingDuckPolicyTests \
  -only-testing:tesseractTests/CaptureEngineLifecycleTests \
  -only-testing:tesseractTests/SpeechCoordinatorTests \
  -only-testing:tesseractTests/AudioPlaybackTests \
  -only-testing:tesseractTests/PlaybackEnvelopeTests

# App bindings, image input, integrations, and model-selection seams:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/AppBindingsTests \
  -only-testing:tesseractTests/SettingsManagerModelSelectionTests \
  -only-testing:tesseractTests/ImageInputAvailabilityTests \
  -only-testing:tesseractTests/ImageIngestTests \
  -only-testing:tesseractTests/ImagePreviewSetTests \
  -only-testing:tesseractTests/ImagePreviewFileCacheTests \
  -only-testing:tesseractTests/QuickLookPreviewItemTests \
  -only-testing:tesseractTests/OpenCodeSetupScriptTests \
  -only-testing:tesseractTests/OpenCodeConfigMergeTests \
  -only-testing:tesseractTests/OpenCodeIntegrationEndpointTests \
  -only-testing:tesseractTests/IntegrationSnapshotBuilderTests \
  -only-testing:tesseractTests/PreserveThinkingRenderTests \
  -only-testing:tesseractTests/VisionPrefixMemoryGuardTests \
  -only-testing:tesseractTests/Qwen3VLProcessorCapTests

# Run all tests:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests
```

## Canonical-echo fidelity gate (corpus mode)

`CanonicalEchoFidelityTests` runs with the suites above (fake tokenizer, no
extra setup). The corpus gate — `CanonicalEchoFidelityCorpusTests` — replays a
recorded session corpus (the `HTTPRequestLogger` request JSONs) through the
real normalization + reasoning-repair + probe machinery with a real model
tokenizer, and fails on any boundary whose derived leaf/speculation path is
not a token-identical prefix of the next request's render (PRD #94). It is
opt-in via environment because the corpus contains user project content and
lives outside the repo:

```bash
TEST_RUNNER_TESSERACT_FIDELITY_CORPUS="$HOME/projects/tesseract-traces/<corpus>" \
TEST_RUNNER_TESSERACT_FIDELITY_MODEL="$HOME/Library/Containers/app.tesseract.agent/Data/Library/Application Support/models/<model-dir>" \
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/CanonicalEchoFidelityCorpusTests \
  -parallel-testing-enabled NO
```

The corpus directory must contain `http-completions/*-request.json`
recordings; the model directory must hold the tokenizer files (tokenizer
+ tokenizer config + chat template JSONs, as shipped on disk). Without
both variables the test is skipped (`.enabled(if:)`), so it is safe in CI.
Note the `TEST_RUNNER_` prefix — plain environment variables do not reach the
test host process. Per-boundary verdicts print to the test log; mismatches
include decoded windows around the fork.

## Emitted Path Index replay gate (corpus mode)

`EmittedPathReplayCorpusTests` (ADR-0063, tickets #475/#476/#477) walks
the same recorded sessions through the canonical-echo harness with a
private **Emitted Path Index** learning every echoed turn — the Leaf
Store's registration simulated on the canonical encode of the stored
render past request N's prompt, the leaf source decided exactly as the
live fast path decides it (`LiveLeafCapture.decide`) — and every next
request resolving at its edge through the Conversation Render, which
serves the composition it resolves to. Every recording renders under the
context the server resolved for it (its `reasoning_effort` against the
template's declared default, the preserve-thinking render on), so the walk
feeds the bytes the build fed. `EmittedPathReplayGate` judges each turn;
the suite asserts the failure list is empty and every failure names its
turn with the whole account (kind, mode, leaf source and boundary reason,
registration, path length, next indexed prefix, prefilled count, new
message tokens, glue, tail):

- in a tool stretch the leaf source is `live`; a stop turn is `live` or
  the explained `thinkStrippingUserBoundary`;
- every live turn registers (the one tolerated skip is
  `promptNotTokenPrefix`: request N's prompt is not a token prefix of the
  stored render, so the harness cannot simulate the fed ids the live fast
  path registers directly — such a turn is exempt from the prefill, glue
  and tail rules below, and the totals count it as `exempt=`; the
  2026-09-06 corpus has none);
- the fidelity gate rejected nothing and no key was registered twice —
  asserted on the index's own counters, not only logged;
- the next request's indexed prefix is the whole registered path, and it
  prefills its new messages plus at most six glue tokens (the newline
  closing the stored turn's marker line and the five-token Qwen3.8
  thinking generation prompt; the ticket's three assumed a bare
  `<|im_start|>assistant\n` prompt) — a shallow hit lands above it;
- below 20k path tokens the simulated post-EOS CPU tail (the stored render
  to bytes plus the registration, reported as `renderMs` and
  `registerMs`) stays under 150 ms. When the corpus directory also holds
  the build's `trace-*.jsonl` completion traces, the recorded live
  `tailSeconds` of every registered turn is gated the same way; the
  2026-09-06 corpus holds none.

Every turn of the 2026-09-06 corpus passes every rule. Two fixes were
needed to get the tail there, both of work that grew quadratically with a
turn's longest newline-free run, and both paid by the live loop as well as
by the replay: the streaming detokenizer re-decoded its whole segment on
every token (the replay reads the tokens through
`LinearStreamingDetokenizer` instead, which reconstructs a byte-level
vocabulary's text from the tokens' own bytes and verifies every segment
against one full decode), and the vendor's `ToolCallProcessor` searched
the whole collected call for its end tag on every chunk (fixed in the
fork, see `docs/mlx-swift-lm-fork.md`). Together they took the corpus's
slowest tail from 14 s to 90 ms: the worst turn is now request#18, 90 ms
over 17.7k path tokens, and the corpus's longest path (91.4k tokens) costs
44 ms, of which 10 ms is the render. The `GATE … tail:` lines name any
turn that goes back over the budget.

Same variables as the fidelity gate; the reference corpus is
`~/projects/tesseract-traces/2026-09-06-emitted-path` (85 recordings from
the two 2026-09-06 Pi sessions — the ticket counted 45 — none carrying a
session header, so the walk treats them as one session):

```bash
TEST_RUNNER_TESSERACT_FIDELITY_CORPUS="$HOME/projects/tesseract-traces/2026-09-06-emitted-path" \
TEST_RUNNER_TESSERACT_FIDELITY_MODEL="$HOME/Library/Application Support/models/mlx-community_Qwen3.8-27B-4bit" \
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/EmittedPathReplayCorpusTests \
  -parallel-testing-enabled NO
```

Per-session totals print to the test log (`emitted-path boundaries=…
registered=… sources=… nextResolved=… nextSuffixTokens=…`), with one
line per boundary that did not register or resolve and one `GATE …` line
per failed rule. The walk is CPU-bound on one core: every boundary
renders and BPE-encodes the whole conversation through the Debug-build
tokenizer (the hot frames are the byte-pair merge and the regex
pretokenizer), about 4–5 s per 30k-token boundary — budget ~2 min for the
2026-09-06 corpus and prefix the command with `nice -n 20` when the
machine is in use. `EmittedPathResolveRealTests` (in the prefix-cache
group above) covers the same claims on the local PARO tokenizer without a
corpus: marker derivation, suffix-encode equality at every end-of-turn
marker, and the request-edge invariant across a tool-call boundary.

### Synthesized cases (hermetic)

`LeafCaptureHandoffTests` covers the capture-side ownership change (#478):
the tree retains the original hybrid cache objects and physical arrays, every
request reference is emptied, a system checkpoint still copies, and clearing
RAM releases the objects. Copy restore matches the previous capture's bytes
and owns independent buffers. Eviction demotes a moved leaf to SSD; a delayed
extension writer retains only detached arrays and hydrates correctly after
RAM is cleared. The synthesized replay below now expects `source=handoff`
for eligible text turns and `source=live copyReason=quantized` for quantized
partitions. Check-out by move and leases remain a later ticket.

`EmittedPathSynthesizedReplayTests` (prefix-cache group) runs the history
shapes the recordings cannot show through the real Server Completion
module — real prefix cache, Leaf Store fast path, Emitted Path Index, SSD
tier — over the content-relative toy Model Session
(`ToyLanguageModel(completions:)`, whose queue answers the generation
prompt wherever the restore put it and keeps the tape of every id fed)
and the Qwen3.8-shaped `EmittedPathToyTokenizer` (thinking template,
effort sentence in the system block, single-token `<|im_end|>`). Each
case reads the request's telemetry events, the handle's restored offset
and the tape: the served composition on a hit, the canonical encode on a
miss, never a wrong prompt. One case each for: the live baseline (the
next request restores the whole path and prefills six glue tokens plus
its new messages); an earlier user message edited; an assistant message
edited; a compacted history; a reasoning-effort change (re-prefill from
token 0, ADR-0060); an `enable_thinking` flip (partition miss, the
closed-think prompt fed canonically); two generations from one parent
with identical text and different splits (last writer wins, the later
leaf hit while resident, the later split fed from token 0 on an empty
cache); a response-conversion fault between model and client
(`FaultyStreamTokenizer`: fidelity rejected, nothing registered, the
warning event, re-prefill below the divergence next turn); an
image-bearing request on a vision-container instance (neither registered
nor resolved, the placeholder run fed, no pseudo-token); index eviction
past the byte bound; a restart with a surviving SSD leaf; and a
think-stripping template at a user boundary (the unchanged boundary
path). The edit and fault cases restore at the deepest checkpoint below
the divergence (**Chain-Prefix Restore**, ADR-0012), not at the token
itself. The cancelled-partial-turn case belongs to ticket #480.
`EmittedPathReplayGateTests` pins the gate's rules on hand-built
accounts.

## Interrupt-readiness acceptance (corpus + live drill)

`IncidentReplayAcceptanceTests` (PRD #94) is the regression net for the
Think-Strip Rewind cliff. It reuses `TEST_RUNNER_TESSERACT_FIDELITY_CORPUS`
and reads the archived `trace-2026-06-12.jsonl` completion-trace log from the
same corpus directory; without it the suite is skipped. It asserts the restore
floor never overshoots the divergence and that the replay is deterministic, so
steady-state hit rate and token reuse move only when behaviour does. The
replay report (`TraceReplayHarness`) and the live prompt-cache dashboard both
surface the rewind roll-up — event count and re-prefill size — so a future
regression shows up in telemetry without reproducing an incident.

The live drill is `scripts/interrupt-drill.sh`: it reproduces the incident
shape against a running server (tool stretch → abort → idle past the
abandonment window → steering message) and measures post-interrupt TTFT
against the 5 s bar (the incident recorded 92.8 s). The `--double` variant
also aborts the recovery prefill and re-sends, asserting the retry resumes
from the salvage rather than restarting from the floor. The server must be
running with the prefix cache enabled and the incident model loaded; the
drill's request bodies live in the incident corpus, outside the repo.

## Loaded-model verification

Not unit tests — these run against a real model.

```bash
scripts/dev.sh prefix-cache-e2e          # PrefixCacheE2ERunner — TTFT/output equivalence proxy
scripts/dev.sh hybrid-cache-correctness  # HybridCacheCorrectnessRunner — bitwise logit + state equivalence
```

Both exit non-zero on any failed check. Run before releases and after any change
to `LLMActor`, `ServerCompletion`, `PrefixCacheManager`, `HybridCacheSnapshot`,
or `StablePrefixDetector`. The correctness runner is the stronger gate (bitwise
tensor comparison via raw `ModelContainer.perform` access); the e2e runner
exercises the full HTTP path and is the right shape for catching pipeline
regressions the correctness runner can't see.
The correctness runner also compares a moved leaf restored by copy against
cold-prefill logits bitwise (`movedLeafRestoredByCopyMatchesBitwise`).

Benchmark-shaped siblings (informational, not gates):
`scripts/dev.sh prefill-step-benchmark` and `scripts/dev.sh paroquant-vlm-smoke`.

`scripts/dev.sh trace-replay` is the odd one out: it needs **no loaded
model**. It replays the Completion Trace Log corpus through the offline
LRU-baseline harness (`TraceReplayHarness`, PRD #82 slice #85) and writes
the report to `benchmark/trace-replay/latest.log`.

## Gotchas

### Request memory timeline (#471)

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
| `restoring` → `restored` | Snapshot size, current restore mode (`cold`, `copy`, `failedCopy`), and the resulting cache's attention/recurrent array sizes. |
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

### Controlled capture comparison (#478)

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

The Qwen3.8 community checkpoint used by this comparison loads as a text
instance even when vision is requested (ADR-0056). The current HTTP E2E
runner's config-based image scenario is therefore not real VLM coverage;
its different-image assertion also fails on the unchanged baseline. The
comparison report retains that failure instead of presenting it as a green
image gate. Use an actual vision-loaded model for image-specific validation.

### Test-runner caveats

- `-only-testing` filters must target **suite** granularity. A method-granularity
  filter (`-only-testing:tesseractTests/<Suite>/<testName>`) runs zero Swift Testing
  tests and still reports `** TEST SUCCEEDED **`. The suite is the `struct` name, not
  the file name: `tesseractTests/DynamicBudgetCeilingTests.swift` holds nine suites and no suite of
  that name, so a filter on the file name also runs nothing and still succeeds. Check
  the `.xcresult` for the suites that actually ran.
- `xcodebuild test` hides `#expect` failure details from stdout. Read them from
  the `.xcresult` bundle:
  `xcrun xcresulttool get test-results tests --path <bundle>.xcresult`.
- Known flake (not a regression):
  `WarmStartTests/warmStartRebuildsFromDirectoryWalkAfterCorruption` can fail in
  any run (solo included). The window: `SnapshotLedger.persistNow` clears
  `manifestDirty` under the lock but writes the manifest file after unlocking,
  so the test's `flushManifestForTesting` can no-op while the debounce task's
  write is still in flight and the `fileExists` check lands first.
- Vendor DFlash2 tests (`swift test --filter DFlash2` in `Vendor/mlx-swift-lm`):
  run with `--no-parallel`. Two of the parity tests load the 27B target each;
  in parallel they contend the single GPU until a Metal command buffer hits
  the watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`). Serial: 18/18 green.
- Heavyweight model-loading tests (27B-class) on a 48 GB machine: run them
  **one test per process** (or at most the proven pairs). Packing several into
  one `swift test` process accumulates fixtures across tests —
  `swiftpm-testing-helper` peaked at 64.5 GB physical footprint on 2026-08-20
  and had to be killed to avoid repeating the 2026-08-19 crash. Two metric
  traps: `memory_pressure` free-% lags the helper's real footprint, and `ps`
  RSS misses IOSurface/shared GPU memory. If you must guard, watch
  `vmmap -summary <pid>` "Physical footprint" of the testing helper itself.
