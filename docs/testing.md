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
  -only-testing:tesseractTests/HTTPPrefixCacheSpikeTests \
  -only-testing:tesseractTests/HTTPPrefixCacheSessionReplayTests \
  -only-testing:tesseractTests/CompletionHandlerTests \
  -only-testing:tesseractTests/CompletionRouteTests \
  -only-testing:tesseractTests/ServerInferenceServiceTests \
  -only-testing:tesseractTests/ServerCompletionDrainTests \
  -only-testing:tesseractTests/ServerCompletionLeafStoreModeTests \
  -only-testing:tesseractTests/ServerCompletionLeafSkipLogTests \
  -only-testing:tesseractTests/CompletionProjectionTests \
  -only-testing:tesseractTests/MessageConverterTests \
  -only-testing:tesseractTests/OpenAITypesTests \
  -only-testing:tesseractTests/AgentEngineToolSpecTests \
  -only-testing:tesseractTests/EditToolTests

# Prefix cache suites (radix tree + hybrid snapshot + stable prefix detector):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -only-testing:tesseractTests/HybridCacheSnapshotTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/CacheKeySpaceTests \
  -only-testing:tesseractTests/PrefillPlannerTests \
  -only-testing:tesseractTests/LeafAdmissionBuilderTests \
  -only-testing:tesseractTests/SnapshotResolutionTests \
  -only-testing:tesseractTests/SnapshotLedgerTests \
  -only-testing:tesseractTests/SnapshotStateTests \
  -only-testing:tesseractTests/StablePrefixDetectorNonDeterminismTests \
  -only-testing:tesseractTests/JinjaNonDeterminismReproTests

# App bindings, image input, integrations, and model-selection seams:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
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

Benchmark-shaped siblings (informational, not gates):
`scripts/dev.sh prefill-step-benchmark` and `scripts/dev.sh paroquant-vlm-smoke`.

`scripts/dev.sh trace-replay` is the odd one out: it needs **no loaded
model**. It replays the Completion Trace Log corpus through the offline
LRU-baseline harness (`TraceReplayHarness`, PRD #82 slice #85) and writes
the report to `benchmark/trace-replay/latest.log`.

## Gotchas

- `-only-testing` filters must target **suite** granularity. A method-granularity
  filter (`-only-testing:tesseractTests/<Suite>/<testName>`) runs zero Swift Testing
  tests and still reports `** TEST SUCCEEDED **`.
- `xcodebuild test` hides `#expect` failure details from stdout. Read them from
  the `.xcresult` bundle:
  `xcrun xcresulttool get test-results tests --path <bundle>.xcresult`.
- Known flake (not a regression):
  `WarmStartTests/warmStartRebuildsFromDirectoryWalkAfterCorruption` can fail in
  any run (solo included). The window: `SnapshotLedger.persistNow` clears
  `manifestDirty` under the lock but writes the manifest file after unlocking,
  so the test's `flushManifestForTesting` can no-op while the debounce task's
  write is still in flight and the `fileExists` check lands first.
