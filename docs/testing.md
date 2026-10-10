# Testing

Tests use the Swift `Testing` framework (not XCTest), in `tesseractTests/`. Run
before committing changes to server, caching, or agent engine code.

## Running them: `scripts/test.sh`

```bash
scripts/test.sh                                   # build-for-testing, then the whole target (~25 s)
scripts/test.sh ChatSessionTests LLMGateTests     # build, then just these suites (~3 s)
scripts/test.sh --no-build ChatSessionTests       # reuse the last build while iterating
scripts/test.sh --no-build 'ChatSessionTests/sendMessageRaisesPendingRowUntilTheUserCommit()'
scripts/test.sh --no-build --slowest 10           # also list the ten slowest tests
scripts/test-stress.sh LLMGateTests               # 3 rounds: 4 whole-target + 3 suite runs at once
```

The script runs the built `.xctestrun` directly instead of `-scheme`, which
skips the 7–10 s xcodebuild spends loading the project on every invocation.
It prints each failure with its `#expect` details (read from the result
bundle; xcodebuild's own output leaves them out), then the totals. A filter
that matches no test (a file name, or a test without its parentheses) exits 3
instead of passing. Logs and the result bundle stay under
`DerivedData/<project>/test-runs/`, one directory per run, so several runs can
share a checkout. `TEST_RUNNER_*` variables reach the tests as usual.

Several agents can run it at once. In stress rounds of four whole-target runs
and three suite runs started together (`scripts/test-stress.sh`), all 56 runs
passed, including rounds where all four whole-target runs overlapped. A
whole-target run keeps every core busy, though, so at most
`TESSERACT_TEST_SLOTS` (default 2) of them proceed at once on the machine,
across checkouts, and the rest wait for a slot: two at once take about 25 s
each, four about 30–40 s. Suite runs never wait. A build after editing one
test file takes about 25 s, half of it xcodebuild re-emitting the test module.

## Parallel runs share one test host

A parallel run starts every suite at once in one test host, so all of them
share its Swift cooperative thread pool (one thread per core), its main actor,
and its process-wide state. Before these rules, a full run failed 50–57 tests
in a dozen suites, and every one of them passed serially.

- **A suite whose cases compute for seconds without suspending takes
  `@Suite(.cpuBound)`** (`tesseractTests/CPUBoundTrait.swift`): loading a real
  tokenizer (the `*RealTests` suites), scanning the app sources. At most a
  quarter of the cores run such cases at once. Without it, the real-tokenizer
  suites held every pool thread for about 40 s. Everything else waited for a
  thread, including `Task.sleep` wake-ups, which resume through the global
  executor even in main-actor code.
- **Wait for the state change, not the clock.** Every `@MainActor` suite
  queues on the one main actor, and one hop onto it can take seconds. A poll
  against a 3 s deadline then fails a test whose code is fine. The waits live
  in `tesseractTests/Waiting.swift`:
  - `await observe(until: { !session.isGenerating })` re-checks whenever the
    `@Observable` state it reads changes. It has no deadline, so its suite sets
    `@Suite(.timeLimit(.minutes(1)))` and a real hang still fails.
  - `waitUntil { … }` polls state that isn't observable: a writer's callback,
    a file on disk.
  - Every wall-clock budget is `waitBackstop` (60 s): a backstop for a hang,
    never a latency budget.
  - A count of `Task.yield()`s counts main-actor turns, not the work. An actor
    hop, a `Task.sleep`, or a detached task crosses the thread pool, and the
    count can run out first: `Agent.prompt` posts the user message back from
    its off-main loop.
  - A negative check ("nothing more happens") keeps its fixed window of yields
    or milliseconds: `for _ in 0..<100 { await Task.yield() }`.
  - SwiftLint enforces this in `tesseractTests/`, in the pre-commit hook and
    in CI: `test_wait_budget` rejects a wall-clock budget, and
    `test_counted_wait` rejects a `for … where … { await Task.yield() }` wait.
- **The scheme lets at most 64 test cases run at once**: the Test action sets
  `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH`. Without it Swift Testing
  starts all ~3,600 cases together, and the main actor spends about 15 s
  getting through their first steps. A check that has to stay on the clock
  pays for that: `BrowserTabNavigationTimeoutTests`'s `elapsed < 5 s`
  measured 4.5 s. With the cap it measures 0.3 s, and a run takes about 3 s
  longer. The variable is experimental. If a toolchain drops it, runs start
  everything at once again, which still passed but with that thin margin.
- **A real tokenizer is loaded once per process.** Ask for it through
  `RealTokenizers.huggingFace(from:)` or `RealTokenizers.app(from:)`
  (`tesseractTests/RealTokenizers.swift`), not the loader directly. A load
  takes about 2 s in a Debug build, and every test used to pay it: over 40 loads
  a run.
- **A test server binds port 0** and reads the port back with
  `ScriptedMCPServer.waitForPort(_:)`. Another suite can take a probed "free"
  port before the listener binds it.
- **Gigabyte-scale work is opt-in.** `SSDSnapshotStoreTests.writerCommitsPayloadPastIntMaxBytes`
  writes a 2.1 GB payload (held in memory too) only with
  `TEST_RUNNER_TESSERACT_LARGE_WRITE_TEST=1`. Every run carries a cheap test
  that pins the writer's chunk bound below `INT_MAX`.
- **A test process starts on empty storage.** `StorageEnvironment.scratchRoot`
  is `$TMPDIR/TesseractTestStorage-<pid>`. When it is first resolved, it
  removes a folder left under its own pid (pids are reused) and the folders of
  test processes that have exited.
- **Process-wide state a test counts is injected, not reset in `init`.**
  Swift Testing runs a suite's `init` as a separate step from its test body,
  so another suite can run in between. `StablePrefixDetectorMemoTests` passes
  its own `StablePrefixDetector.Memo`, as the render-cache suites pass their
  own `RenderTokenCache`.

## Test-runner caveats

- `-only-testing` filters name a **suite**, or one test as `<Suite>/<testName>()`
  with the parentheses. Without them (`<Suite>/<testName>`) the filter runs zero
  Swift Testing tests and still reports `** TEST SUCCEEDED **`. The suite is the
  `struct` name, not the file name: `tesseractTests/DynamicBudgetCeilingTests.swift`
  holds nine suites and no suite of that name, so a filter on the file name also
  runs nothing and still succeeds. `scripts/test.sh` fails such a run (exit 3).
- `xcodebuild test` hides `#expect` failure details from stdout (`scripts/test.sh`
  prints them). Read them from the `.xcresult` bundle:
  `xcrun xcresulttool get test-results tests --path <bundle>.xcresult`.
- Speech package tests (`Vendor/tesseract-speech`, needs the
  `Vendor/mlx-swift-lm` submodule checked out). Two suites, no weights:
  - `TesseractSpeechTests`: scripted adapters, no GPU. `EngineContractTests`
    (the ADR-0038 contracts, the readiness updates the app's presenter
    follows, the ADR-0072 Reference Take rules: the lead
    segment becomes the take, later segments and utterances continue it,
    a pinned voice round-trips, a cancelled retake keeps the old take,
    schema-1 voices are rejected; a Preset Voice takes no Reference Take and
    an unknown one is refused when its session opens (ADR-0084);
    `SegmenterTests` for the short lead
    segment; `ModelAvailabilityTests`: a missing checkpoint fails before any
    load; word starts shifted to the utterance's frames) and
    `Qwen3CheckpointTests` (the Voice Engine completeness rule, and
    `Qwen3Synthesizer` refusing to fetch or delete anything). `WordTimerTests`
    (ADR-0077) runs the word timer on synthetic attention rows and levels:
    steps, pauses, a take's text ahead of the first word, frames the silence
    cap dropped, and when starts are sent.
    `swift test --package-path Vendor/tesseract-speech --filter TesseractSpeechTests`
    runs them.
  - `Qwen3TTSTests`: the model itself on tiny random-weight checkpoints and
    fixed logits.
    - Sampling: EOS filtered like every other token, temperature before
      top-p, top-k keeping ties, EOS held back for two frames, the windowed
      repetition penalty.
    - Prompts: the reference-take prompt, both text layouts, the dialect rule,
      the text table read from disk, where the text track sits.
    - Word timing (ADR-0077): the alignment probe reads one head's logits over
      its span through both attention paths; a render with a head streams
      the text track and one row per frame, each ahead of its audio, with
      the same samples as a render without one.
    - Loading: the codec encoder dropped, Base checkpoints refused, stacked
      projections equal to separate ones.
    - The decoder (ADR-0074): the streamed audio equals the one-pass decode
      at every chunk size, and the sliding window and its mask hold.
    - The fused Metal kernels equal the MLX ops they replace, bit for bit:
      the sampler's draw, q/k norm + RoPE over some 4 million values, add +
      norm, an attention step, a layer stack. Also bit for bit: a prompt
      evaluated a layer at a time equals the one graph.
    - The fork's causal SDPA matches attention written out in float32, and
      overlapping generations render as if alone.
    - The Core ML conv stack (ADR-0075) matches MLX's. It is compiled for
      the CPU, so no Neural Engine is needed.
    - The Neural Engine voice (ADR-0088, `Qwen3TTSNeuralTests`, also
      compiled for the CPU): the talker's step gives MLX's logits, hidden
      state and Alignment Head scores position by position, with its cache
      in Core ML state and another session's steps in between changing
      nothing; it refuses a position past its cache. The code predictor's
      frame gives MLX's greedy codes and embedding sum, draws the best of the
      top k plus the same Gumbel noise, never draws from outside the top k,
      and a layer whose MLP product outgrows fp16 is found by measuring and
      still matches. The host sampler keeps the talker's rules (no control
      codes, EOS held back, the penalty, ties at the cut, the nucleus), and a
      prepared voice renders, the same again for the same seed.

    MLX needs Metal, so run both suites through xcodebuild, from
    `Vendor/tesseract-speech`:

    ```bash
    xcodebuild test -scheme tesseract-speech-Package -destination 'platform=macOS' \
      -skipPackagePluginValidation -skipMacroValidation -parallel-testing-enabled NO \
      CODE_SIGNING_ALLOWED=NO
    ```
- Voice Engine listening and timing (`v2-listen`, real weights, the
  checkpoint the app downloaded): build the `v2-listen` scheme with
  xcodebuild (it needs MLX's metallib next to the binary), then
  `v2-listen --mode longform --text-file <passage> --seed <n>` writes the
  whole reading plus one WAV per segment and prints per-segment time to
  first audio, RTF and peak RSS. It also writes `<stem>_words.json`: each
  segment's text and its words' start frames (ADR-0077). To score them,
  transcribe the segment WAVs with Whisper word timestamps (WhisperKit's
  CLI with the app's `whisperkit-coreml` model) and compare starts; the method
  and the 2026-09-27 numbers are in
  `docs/research/2026-09-27-word-timing-from-attention.md`. `--reference none` renders every segment
  from the description alone, the control for voice-consistency listens.
  `--neural-engine off` keeps the codec's conv stack on MLX. On is the
  default, with the Core ML model cached in
  `~/Library/Caches/tesseract-speech/neural-codec`; the app keeps its own under
  its storage root.
- Qwen3-TTS model measurements and numerical checks (`qwen3-tts-bench`, real
  weights, below the engine): build the `qwen3-tts-bench` scheme with
  xcodebuild, as for `v2-listen`. The usage comment in
  `Vendor/tesseract-speech/Sources/Tools/qwen3-tts-bench/main.swift` lists
  every mode. Useful modes:
  - `--mode profile`: times the talker step, the code-predictor frame and the
    decoder.
  - `--mode generate --out FILE.json`: records golden frames at a seed.
  - `--mode trace --golden FILE.json --out DIR`: writes teacher-forced logits
    on those frames, for comparing against Qwen's PyTorch model.
  - `--mode kernels` and `--mode audit`: check each fused Metal kernel against
    the MLX ops on real data. A kernel change must still show identical codes
    and 0 mismatching calls.
  - `--mode neural`: builds and times the Neural Engine codec.
  - `--mode neural-voice` (the 0.6B CustomVoice checkpoint, `--voice` a
    speaker): prepares the talker and the code predictor on the Neural
    Engine (placement, the precision MLX measured, the check against MLX),
    then teacher-forced parity on `--golden` frames, the stages before the
    first audio, time per call, and `--repeat` renders at consecutive seeds,
    with `--mlx-renders` the same renders on MLX first. The 2026-10-03
    numbers are in ADR-0088.
- Neural Engine op placement (`ane-lab`, no weights): builds tiny ML
  programs with the package's `MLProgramBuilder` and prints where Core ML's
  compute plan puts each op on this machine, or with `--time` the latency of
  weight-heavy stacks in each weight format. Build the `ane-lab` scheme with
  xcodebuild, as for `v2-listen`; `ane-lab conv` runs the probes whose name
  contains `conv`. ADR-0088's format choices come from it.
- Vendor fork tests (`Vendor/mlx-swift-lm`, where `swift test` does not run):
  `scripts/vendor-test.sh [--no-build] [suite…]` builds for testing once, then
  runs `MLXLMTests` serialized with a two-minute per-test allowance, and
  prints only failures and totals. Xcode's parallel runner hangs on this
  GPU-heavy target, and two DFlash2 parity tests that each load the 27B target
  contend the single GPU until a Metal command buffer hits the watchdog
  (`kIOGPUCommandBufferCallbackErrorTimeout`). A Swift Testing free function
  needs its parentheses: `'testName()'`. `--docs` adds the DocC check.
- Heavyweight model-loading tests (27B-class) on a 48 GB machine: run them
  **one test per process** (or at most the proven pairs). Packing several into
  one `swift test` process accumulates fixtures across tests —
  `swiftpm-testing-helper` peaked at 64.5 GB physical footprint on 2026-08-20
  and had to be killed to avoid repeating the 2026-08-19 crash. Two metric
  traps: `memory_pressure` free-% lags the helper's real footprint, and `ps`
  RSS misses IOSurface/shared GPU memory. If you must guard, watch
  `vmmap -summary <pid>` "Physical footprint" of the testing helper itself.

## Unit / integration suites

`AlphaTunerTests.productionCacheKeepsAlphaTunerDisabled` drives toy-backed
Server Completion through production cache construction, both with and without
Model Identity, and checks the published tuner state is unavailable with static
`alpha = 0`. The other tuner tests exercise the retained implementation only;
they do not enable it in the app. See [#504](https://github.com/spokvulcan/tesseract/issues/504)
and the [captured incident](../benchmarks/incidents/2026-09-12-alpha-tuner/README.md).

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
  -only-testing:tesseractTests/ServerCompletionGenerationPromptTests \
  -only-testing:tesseractTests/RequestFactsTests \
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
  -only-testing:tesseractTests/SpeculationPlanTests \
  -only-testing:tesseractTests/SpeculationResidencyTests \
  -only-testing:tesseractTests/SpeculativeDecodeToyTests \
  -only-testing:tesseractTests/DFlash2SupportTests \
  -only-testing:tesseractTests/MTPDrafterSupportTests \
  -only-testing:tesseractTests/EditToolTests

# Prefix cache suites (radix tree + hybrid snapshot + stable prefix detector):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/HybridCacheSnapshotTests \
  -only-testing:tesseractTests/WarmBodyModelSessionTests \
  -only-testing:tesseractTests/WarmBodyDrainTests \
  -only-testing:tesseractTests/SnapshotLayerKindTests \
  -only-testing:tesseractTests/LeafCaptureHandoffTests \
  -only-testing:tesseractTests/LeafAdmissionTests \
  -only-testing:tesseractTests/LeafAdmissionSourceShapeTests \
  -only-testing:tesseractTests/SnapshotPayloadTests \
  -only-testing:tesseractTests/SnapshotAdmissionStorageTests \
  -only-testing:tesseractTests/SalvageOnCancelTests \
  -only-testing:tesseractTests/CompletionTraceAccumulatorTests \
  -only-testing:tesseractTests/SnapshotDemotionTests \
  -only-testing:tesseractTests/SurvivalGateTests \
  -only-testing:tesseractTests/LeafLeaseTests \
  -only-testing:tesseractTests/CacheClaimTests \
  -only-testing:tesseractTests/ServerCompletionExitMatrixTests \
  -only-testing:tesseractTests/ServerCompletionRestoreFallbackTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/BudgetDrainDiagnosticsTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/PrefixViewModelSessionTests \
  -only-testing:tesseractTests/CacheKeySpaceTests \
  -only-testing:tesseractTests/PrefillPlannerTests \
  -only-testing:tesseractTests/LeafAdmissionBuilderTests \
  -only-testing:tesseractTests/ConversationRenderSourceShapeTests \
  -only-testing:tesseractTests/GenerationPromptProbeTests \
  -only-testing:tesseractTests/GenerationPromptCatalogRealTests \
  -only-testing:tesseractTests/ConversationRenderProbeParityTests \
  -only-testing:tesseractTests/ConversationRenderProbeParityRealTests \
  -only-testing:tesseractTests/SnapshotResolutionTests \
  -only-testing:tesseractTests/SnapshotResolutionLadderTests \
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

# Voice session + capture engine (quit the app first — its capture engine
# starves test hosts). VoiceSessionMachineTests pins the half-duplex loop
# (ADR-0082): the mic closed under the reply, barge-in by key or click, the
# dead-capture recovery; CaptureEngineLifecycleTests pins the live-input
# check's verdicts:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/VoiceSessionMachineTests \
  -only-testing:tesseractTests/VoiceEndpointerTests \
  -only-testing:tesseractTests/VoiceCaptureSessionTests \
  -only-testing:tesseractTests/VoiceProcessingDuckPolicyTests \
  -only-testing:tesseractTests/CaptureEngineLifecycleTests \
  -only-testing:tesseractTests/SpeechCoordinatorTests \
  -only-testing:tesseractTests/AudioPlaybackTests \
  -only-testing:tesseractTests/StreamingSchedulerTests

# Dictation and Learned Words (ADR-0085/0086; quit the app first, its capture
# engine starves test hosts): the Voice Capture Session (Learned Words after the
# regex cleanup, the silent-capture skip, every caller), the regex cleanup, the
# Correction Pair store and the gold mark a fix gives, the sounds-alike key, take
# tokens, matching and the Learned Word store (exceptions per app, ordinary words
# not learned, Forget and Undo), the fix (target by sound on real mishearings,
# what a fix teaches, several fixes in one take), the Live Preview (the
# confirmation rule, decoding from the last confirmed segment over a scripted
# recognizer, release cancelling the preview), ⇧ holding a take and the Check
# Before Pasting setting, the Lens state (listening, landing, a held take pasted
# or kept), putting a fix back in the app, the hotkeys, and paste. The repo has
# no snapshot suites: the Lens views are covered by LensViewRenderTests, which
# hosts each state the way the panel does (as MainWindowPageTests hosts the
# pages) and checks it fits the card, with no pixel comparison;
# TEST_RUNNER_LENS_RENDER_DIR=<dir> also writes a PNG of each state to look at.
# The Catch Record (the Dictation page, PRD #612): CatchRecordTests pins the
# page's model (the week's catches and fixes per day, a tile per Learned Word
# still known, today's takes and how each opens in the Lens, the sentence and
# the marked words); DictationPageTests opens the page (empty, and with Learned
# Words, fixes and today's takes) and its History sheet with the app's wiring,
# as MainWindowPageTests does, so a missing dependency crashes on that case;
# TranscriptionHistoryTests pins entries keeping their catches and app across
# a reload (older entries load without them) and a fix rewriting an entry's
# text and catches (its app kept); and LensControllerTests also opens a take
# from the page (refused while a take is recorded; a held or after-paste take
# open in the Lens kept with its fixes; a page fix to the last take carrying
# into the ⌃⌥Space reopen; cancelling one never making it the last take).
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/DictationCoordinatorTests \
  -only-testing:tesseractTests/VoiceCaptureSessionTests \
  -only-testing:tesseractTests/CorrectionPairStoreTests \
  -only-testing:tesseractTests/TranscriptionPostProcessorTests \
  -only-testing:tesseractTests/TranscriptionEngineTests \
  -only-testing:tesseractTests/DictationFeedTests \
  -only-testing:tesseractTests/LivePreviewAssemblerTests \
  -only-testing:tesseractTests/LensSettleTests \
  -only-testing:tesseractTests/HotkeyMatcherTests \
  -only-testing:tesseractTests/TextInjectorTests \
  -only-testing:tesseractTests/AgentVoiceInputControllerTests \
  -only-testing:tesseractTests/SoundAlikeTests \
  -only-testing:tesseractTests/TakeTextTests \
  -only-testing:tesseractTests/LearnedWordMatcherTests \
  -only-testing:tesseractTests/LearnedWordStoreTests \
  -only-testing:tesseractTests/LensFixTests \
  -only-testing:tesseractTests/LensModelTests \
  -only-testing:tesseractTests/LensControllerTests \
  -only-testing:tesseractTests/LensViewRenderTests \
  -only-testing:tesseractTests/CatchRecordTests \
  -only-testing:tesseractTests/DictationPageTests \
  -only-testing:tesseractTests/TranscriptionHistoryTests \
  -only-testing:tesseractTests/CaptureLevelTests \
  -only-testing:tesseractTests/InAppReplacerTests \
  -only-testing:tesseractTests/InAppEditTests \
  -only-testing:tesseractTests/ModifierTapDetectorTests \
  -only-testing:tesseractTests/SettingsCatalogueTests \
  -only-testing:tesseractTests/SettingsManagerTests

# Speech page (ADR-0076/0077): the Reader over the real coordinator and
# engine, the Read-Along clock and its word timing, text geometry, the
# overlay's feed, and voice design:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/SpeechReaderTests \
  -only-testing:tesseractTests/ReadAlongTimelineTests \
  -only-testing:tesseractTests/SpeechReadAlongTests \
  -only-testing:tesseractTests/WordTimelineTests \
  -only-testing:tesseractTests/ReaderTextTests \
  -only-testing:tesseractTests/ReaderDocumentStoreTests \
  -only-testing:tesseractTests/CaptionLayoutTests \
  -only-testing:tesseractTests/CaptionFeedTests \
  -only-testing:tesseractTests/VoiceDesignTests \
  -only-testing:tesseractTests/VoiceLibraryTests \
  -only-testing:tesseractTests/SpeechCoordinatorTests \
  -only-testing:tesseractTests/MainWindowPageTests

# Dictation overlay freeze (no unit-test seam: the hang lives in SwiftUI's
# key-view loop on macOS 27.0; tools/overlay-focus-hang-lab is the regression
# loop — no flags must exit 2 while the OS still hangs, --unfocusable must exit 0):
swift run --package-path tools/overlay-focus-hang-lab overlay-focus-hang-lab --unfocusable

# App bindings, image input, integrations, and model-selection seams (and the
# Models page's in-memory mark):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/AppBindingsTests \
  -only-testing:tesseractTests/SettingsManagerModelSelectionTests \
  -only-testing:tesseractTests/LoadedModelsTests \
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

# The Companion (ADR-0080), with no model and no EventKit: the Day Engine's
# decision tables (nudges, moments, the Morning Plan's code card and its
# refinement, the plan made before the sit-down, Breakpoints, Triage — only
# people, never during a game —, agents, the Night Reflection, card actions,
# delivered nudges), banner sources and game detection, the Agenda tools over
# the in-memory store (delete_event included) and its snapshot past midnight
# (still the day that is ending until 04:00), the completion the EventKit
# store hands its Reminders fetch (nonisolated, delivered off the main thread;
# no store is touched), capture and the one-key hotkey,
# the Timeline (tomorrow after today, without today's must-do star, and the
# small hours past midnight), the Now Card (its decision table, the Inbox's
# slot offers, how long a card's line stays fresh, the time left of the step
# under way), a title's links read short (LinkedTitleTests), a calendar named
# by its address reads as its domain (AgendaCalendarLabelTests), Today
# rendered with the app's wiring over fixture days (one deep in a slot, its
# time left draining) at a wide, a regular and a phone width
# (TEST_RUNNER_TODAY_GALLERY_DIR=<dir> also writes each render there as a PNG,
# dark and light; glass doesn't survive the offscreen render, so secondary-styled
# content on .glassEffect comes out black or vanishes: judge layout in the PNGs
# and colors in the app), the Step Cue (a planned slot put on the panel at its start,
# once, and never while away, quiet, in a call, a game or a meeting, or over a
# panel that is up — what came due meanwhile is cued late once the owner can
# see it; a started step's check-in at its end, which no other start
# interrupts; and what each choice does), the wind-down (one banner as quiet
# hours begin with the owner at the Mac), the Jarvis Panel's content over
# fixture cards at the height it fits them (the plan's steps, the wrap-up's
# leftovers; TEST_RUNNER_PANEL_GALLERY_DIR=<dir> writes the PNGs), the menu
# bar's clock beside the glyph (a started step's time left, the next event or
# time to leave in its last half hour; the same directory gets its PNGs), one
# day walked through the engine signal by signal where those rules meet (the
# plan, a cue started, an absence and the late check-in, the must-do put off
# and started small, the clock, the evening),
# cards and prompts, the Day Thread's store, Shown Pictures (ADR-0090:
# DayThreadPicturesTests — the model sees a picture in the turn it's shown and
# a line in its place after, a tool's images too, an earlier turn reading the
# same as the thread grows; a real Day Thread over the in-memory arbiter whose
# owner turn hands the model the picture and whose next turn and moment read
# the line; the panel's picture rules; the Today and panel galleries render
# pictures waiting and asked), the seen ledger, the
# Delivery Ladder and governor, the Claude Code merge, the Profile and recall
# (fixture conversation files), the trace, the voice overlay's placements, and
# the prefix-cache contract (a byte-identical system prompt; the Now Tag on
# every user message):
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/DayEngineNudgeTests \
  -only-testing:tesseractTests/DayEngineMomentTests \
  -only-testing:tesseractTests/DayEngineBreakpointTests \
  -only-testing:tesseractTests/DayEngineMorningPlanTests \
  -only-testing:tesseractTests/DayEngineStepCueTests \
  -only-testing:tesseractTests/DayEngineWindDownTests \
  -only-testing:tesseractTests/JarvisPanelGalleryTests \
  -only-testing:tesseractTests/MenuBarClockGalleryTests \
  -only-testing:tesseractTests/DayWalkTests \
  -only-testing:tesseractTests/NotificationSourceTests \
  -only-testing:tesseractTests/ModifierKeyDetectorTests \
  -only-testing:tesseractTests/CardItemActionTests \
  -only-testing:tesseractTests/NightReflectionTests \
  -only-testing:tesseractTests/DayStateStoreTests \
  -only-testing:tesseractTests/AgendaToolsTests \
  -only-testing:tesseractTests/AgendaPlaceTests \
  -only-testing:tesseractTests/AgendaCalendarLabelTests \
  -only-testing:tesseractTests/EventKitAgendaStoreTests \
  -only-testing:tesseractTests/AgendaTimeTests \
  -only-testing:tesseractTests/CaptureParserTests \
  -only-testing:tesseractTests/NudgePlannerTests \
  -only-testing:tesseractTests/TimelineBuilderTests \
  -only-testing:tesseractTests/NowCardTests \
  -only-testing:tesseractTests/LinkedTitleTests \
  -only-testing:tesseractTests/TodayGalleryTests \
  -only-testing:tesseractTests/CardParserTests \
  -only-testing:tesseractTests/MomentPromptsTests \
  -only-testing:tesseractTests/DayThreadTests \
  -only-testing:tesseractTests/DayThreadPicturesTests \
  -only-testing:tesseractTests/SeenLedgerTests \
  -only-testing:tesseractTests/DeliveryLadderTests \
  -only-testing:tesseractTests/ClaudeCodeHooksTests \
  -only-testing:tesseractTests/ProfileStoreTests \
  -only-testing:tesseractTests/RecallIndexTests \
  -only-testing:tesseractTests/CompanionTraceTests \
  -only-testing:tesseractTests/DayKeyTests \
  -only-testing:tesseractTests/NowTagTests \
  -only-testing:tesseractTests/SystemPromptAssemblerTests \
  -only-testing:tesseractTests/RetiredCompanionDataTests \
  -only-testing:tesseractTests/OverlayPlacementTests

# The LLM Gate (ADR-0081) and the menu's Models section: one LLM generation at
# a time (FIFO, handoff, cancellation), the runs and the proofreader that read
# it, and what the Models section says:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/LLMGateTests \
  -only-testing:tesseractTests/AgentRunControllerTests \
  -only-testing:tesseractTests/ProofreadPassTests \
  -only-testing:tesseractTests/MenuModelsSectionTests

# Every main-window page opens with the app's own wiring, and test runs stay
# off the owner's data (ADR-0073). A page missing an environment dependency
# crashes the host: the result shows "Crash" on that page's case.
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/MainWindowPageTests \
  -only-testing:tesseractTests/StorageEnvironmentTests \
  -only-testing:tesseractTests/TelemetryEnvironmentTests

# Run all tests:
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests
```

A test run never reaches the owner's data (ADR-0073). The app is the test
host, and under a test runner it keeps everything it stores in
`$TMPDIR/TesseractTestStorage-<pid>`, keeps its settings in memory, and opens
no windows. The model folder stays in place, so suites that load installed
models still find them. A suite that goes through an app default such as
`PathSandbox.defaultRoot` gets the empty scratch copy and skips. The
Companion's Agenda is the in-memory store under a test runner, so no test asks
for or touches the owner's Reminders or Calendar, not even a scratch list. `StorageEnvironmentTests` fails when app code finds Application Support
or Caches without going through `StorageEnvironment`.

A scratch launch (`scripts/dev.sh dev --scratch`) gives the running app the
same data choices, with its windows open, so a change to a page or a panel can
be tried by hand, or driven by an agent, without the owner's data: scratch
storage, settings at their defaults with onboarding done, the in-memory
Agenda, and no OS notifications (`CompanionNotifierTests`). Each launch starts
empty.

For a validation run that must not load models, prefix the command with
`TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests`. The existing
test-host detector makes `DependencyContainer.setup()` return before app
bootstrap, including Whisper and proofreader prewarms. Use an explicit suite
allowlist (the prefix-cache block above plus touched suites) after checking its
fixtures, rather than the broad target. The prefix block's `Real` suites load
tokenizer files, not model weights. Leave corpus and allocation opt-ins unset.

**Speculative decoding** (ADR-0079) is covered on the toy.
`SpeculationPlanTests` pins the **Speculation Plan**'s table from the request's
facts alone: which arm engages, the advance allowance, and where the DFlash2
prefill splits. `SpeculationResidencyTests` runs the real drafter loader over the
toy's container for the loads that attach nothing: the setting off, no MTP head,
no pairing class, no draft downloaded, a Rotated Ternary Checkpoint.
`SpeculativeDecodeToyTests` runs the vendor DFlash2 iterator over the toy target
with a scripted drafter (`tesseractTests/ToyDFlash2Drafter.swift`), through the
Raw Generation Start and the Server Completion. The text equals the toy's script
with and without drafter misses, a thinking turn keeps its boundary snapshots, a
warm turn speculates over the stored leaf, and a quantized-KV turn keeps ordinary
decoding. The MTP iterator can't run on the toy (its head reads the target's
hidden states), so a presence-only drafter that traps if engaged
(`Speculation.inactiveMTP`) pins where MTP must stay off. `DFlash2SupportTests`
and `MTPDrafterSupportTests` keep the per-family detection, geometry and pairing.

The **Generation Prompt** (ADR-0070) is covered at three levels.
`GenerationPromptProbeTests` measures a fake template of every shape the probe
tells apart (`TemplateShapeTokenizer`): thinking by default, thinking only when
asked, a template that is not ChatML-shaped, an empty append, a merge across the
append point, a prompt that depends on the conversation, a throwing template and
a history rewrite. It also pins the memo's cost: two renders for a new render
context, none after. `ServerCompletionGenerationPromptTests` runs every consumer
through the toy Model Session: a thinking-off stop turn whose leaf the next user
turn hits, a stop-turn table after a tool result, the unknown state's answers and
an unkeyed thinking stream; `RequestFactsTests` checks what Request Keying
derives. `GenerationPromptCatalogRealTests` is the catalog gate. It loads the
tokenizer of every catalog chat model downloaded under
`~/Library/Application Support/models` (override with
`TEST_RUNNER_TESSERACT_MODELS_ROOT`), and requires each to measure by default and
with thinking off. It is skipped where no catalog chat model is downloaded, which
includes CI: the app creates that directory at launch, so a runner has it empty.
A skip must be reported as one.

The planned Prefix-View Checkpoint slice (#524, ADR-0068) is covered at the
existing seams. `CheckpointCaptureTests` checks synchronized whole-state-only
capture. `PrefixViewModelSessionTests` compares exact cache bytes and generated
tokens against an owned checkpoint for plain and quantized attention, with
disjoint backing addresses; it also covers the prepared image-prefix capture
entry. `SnapshotResolutionLadderTests` checks nearest/unleased selection,
recency and all fall-through rungs. `TokenRadixTreeTests` checks byte accounting,
eviction exclusion, lease/check-in and last-backer self-heal.
`SnapshotResolutionTests` checks both Restore Pins, view-only panel bytes, and
retirement after the final active request skips leaf storage. `CacheClaimTests`
checks that a view copies with the `checkpoint` copy reason and the `prefixView`
refusal in image and quantized partitions.
`SnapshotAdmissionStorageTests` gives a view SSD intent without extracting it;
`ServerCompletionKeyedSequencingTests` checks capture/lookup telemetry and
canonical reconstruction from a planned view. `SpeculativePrefillPreemptionTests`
checks planned-view restore and pin cleanup through the toy Model Session.
The transient-boundary slice (#525) extends those same suites.
`thinkStrippingTurnRetainsOnlyWholeStateBoundaryBytes` checks request-memory
telemetry for both an attention-only toy (0 bytes) and a hybrid toy with three
float32 recurrent values (12 bytes), and requires canonical admission from the
checked-in leaf with no older checkpoint available.
`speculativeViewRestoresOrReprefillsAfterBackingLeafDeparture` compares the exact
admitted path and KV rows for planned/transient views, a leased Backing Leaf,
and a removed backer in both ordinary and RAM-only abandonment passes; it pins
the fallback diagnostic's offsets and releases Restore Pins and the test lease.
`imageBearingThinkStripUsesTheCheckedInBackingLeaf` covers image-run expansion,
canonical admission, and the next turn's exact residual through the same toy
Model Session. Run `RequestMemoryTelemetryTests`, `SpeculativeCanonicalPrefillTests`,
`ServerCompletionKeyedSequencingTests`, `SpeculativePrefillPreemptionTests`,
`ServerCompletionDrainTests`, `PreserveThinkingRenderTests`,
`CanonicalEchoFidelityTests`, `CanonicalEchoFidelityCorpusTests`, and
`LeafStoreFastPathTests` alongside the prefix suites above.
These tests do not establish loaded-model parity or large-cache memory savings.

The view SSD slice (#526) uses those same seams. Extraction tests fix the byte
total and compare every retained array's physical address with both source
snapshots; `PrefixViewModelSessionTests` also compares plain and quantized view
payloads against owned checkpoints. `LeafLeaseTests` runs a pending view write
while its Backing Leaf is checked out. `SSDWriteEagernessTests` checks delayed
extraction, cold deferral, reuse promotion, type protection, and full-body SSD
hydration after backer loss. `TokenRadixTreeTests` checks immediate self-heal
after pending, committed, or explicitly deleted backing loss. The keyed toy
sequencing test verifies a reused planned view reaches the durable manifest
through the production successful-turn tail. Run `SSDWriteEagernessTests`,
`SSDWriteEagernessPolicyTests`, the extension-admission suites, `SnapshotLedgerTests`,
and the SSD store/manifest suites with the prefix-cache block.
The eagerness suite also holds the Model Session at a toy forward to verify
cancellation and replacement before enqueue preserve the view's SSD intent;
a busy Storage Activity Gate must not delay a pressure-triggered write-through.

## The iPhone app

`tesseract-ios` has no test target: the code it shares with the Mac is tested
in `tesseractTests`, and no test runs on a device. The phone's shared rules have
their own suites: `ReaderLibraryTests` (the **Library**: add, order, delete,
each text's Bookmark and language, a relaunch), `PhoneSettingsTests` (the phone's
Settings Facade over the shared Catalogue keys), `TTSLanguageTests` (a text's
language among the voice's ten), and `SpeechReaderTests`' tap and language
cases, and `TextIntakeTests` (a saved news page gives its article and not its
menus, ads or comments; a PDF's lines join back into paragraphs; Markdown loses
its marks; Safari's script results, plain text and a PDF from the share sheet;
the **Library Inbox** handing texts to the Library), `PocketControlsTests` (a
call stops the reading and it goes on from the heard sentence; lost headphones
stop it; the lock screen's buttons play, pause and skip by sentence; a pause
becomes a stop when the app leaves the screen), `ThermalPolicyTests` and
`ReadingMeterTests` (each segment's real-time factor from the engine's
diagnostics marks, and the report a TestFlight tester copies),
`SpeedCheckTests` (a measured real-time factor in; the neural voice or the
System Voice, and the speed menu's rates, out), `TrimmingModelFetchingTests`
(the codec's decoder kept from a file that also holds its encoder: the header
rewritten in place, MLX loading what is left, and the download manager
fetching only those bytes over the in-memory peer) and `ModelCatalogTests`'
phone entry. The package's
`SystemVoiceSynthesizerTests` drive the **System Voice**
adapter with a scripted renderer: resampling to 24 kHz, whole frames, and word
marks turned into word starts that never run ahead of their audio, and its
`VoiceHandoverTests` the switch between the neural voice and the System Voice,
which lands on a segment boundary with the frames gapless across it, and a
segment the neural voice fails: read by the System Voice when it failed before
its audio, ending the utterance when it failed partway. The
phone's own adapters (the background download, `PhoneVoice`) are checked on
the device; on this Mac, `v2-listen --mode phone` runs the phone's speech stack
(MLX on the CPU, the voice on the Neural Engine, its Voice Preparation twice and
the Speed Check's render) on the 0.6B checkpoint, and prints the memory
footprint iOS counts after each preparation and the reading. CI's
`build-ios` job only builds it. Build it the same way before pushing a change
to a shared folder (ARCHITECTURE.md → The iPhone app), so a Mac-only file that
slipped into one fails here rather than in CI:

```bash
xcodebuild build -project tesseract.xcodeproj -scheme tesseract-ios \
  -configuration Debug -destination 'generic/platform=iOS' \
  -derivedDataPath DerivedData -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
```

## Live detokenization and stream parity

`LiveStreamingDetokenizerTests` loads a tiny real BPE tokenizer through
`AppTokenizerLoader`. It pins exact chunk UTF-8 bytes and release-token steps,
decoder eligibility, cleanup and unknown-tokenizer fallback, added-token
boundaries (including empty tokens and incomplete UTF-8), template forwarding,
and newline-free work counts. The long malformed-byte test also catches
rebuilding a growing withheld chunk on every token.
`ConversationRenderSourceShapeTests` keeps server template calls at the
Conversation Render boundary, with an explicit exception for the tokenizer
bridge's forwarding methods.
`LiveTokenGenerationLoopTests` drives the production loop one token at a time:
the producer waits for text or an Argument Fragment's source delta before
advancing. A complete tagged call through a recognized byte tokenizer must emit
its parsed tool call before EOS, after its source deltas. It also checks split
Unicode and an incomplete final scalar, and uses explicit barriers to verify
upstream cleanup before mapper completion after consumer abandonment and before
natural stream completion. The one-minute
test timeout is a deadlock guard, not a delivery-latency allowance.

`LinearStreamingDetokenizerTests` retains the verified replay's window and
recomputation coverage and checks naive live fallback for those same decoders.
`LinearStreamingDetokenizerRealTests` pins live release steps with the local
Qwen MLX and PARO tokenizers, including a long newline-free tool call, and retains
the replay parity and tail-budget checks. These tokenizer-only tests load no
weights. The optional model directories are `TESSERACT_TOKENIZE_CACHE_MODEL`
(default Qwen3.8-27B-4bit) and `TESSERACT_PARO_TOKENIZE_MODEL` (default
Qwen3.6-27B-PARO); missing directories skip the corresponding real-tokenizer
checks and must be reported as skips.

Quit the running app before this focused group and relaunch it afterward:

```bash
xcodebuild test -project tesseract.xcodeproj -scheme tesseract -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:tesseractTests/LiveStreamingDetokenizerTests \
  -only-testing:tesseractTests/ConversationRenderSourceShapeTests \
  -only-testing:tesseractTests/LinearStreamingDetokenizerTests \
  -only-testing:tesseractTests/LinearStreamingDetokenizerRealTests \
  -only-testing:tesseractTests/LiveTokenGenerationLoopTests \
  -only-testing:tesseractTests/TokenGenerationLoopTests \
  -only-testing:tesseractTests/ToolCallDeltaTrackerTests \
  -only-testing:tesseractTests/GenerationStreamLoopTests \
  -only-testing:tesseractTests/ManagedGenerationDriverTests \
  -only-testing:tesseractTests/GenStreamLoopMalformedToolCallBufferTests \
  -only-testing:tesseractTests/ToolCallParserDeltaTests \
  -only-testing:tesseractTests/ArgumentTranscoderCorpusTests \
  -only-testing:tesseractTests/ArgumentTranscoderWireShapeTests \
  -only-testing:tesseractTests/ArgumentTranscoderAtomicFallbackTests \
  -only-testing:tesseractTests/ArgumentTranscoderJSONWrapperTests \
  -only-testing:tesseractTests/ArgumentTranscoderEquivalenceTests \
  -only-testing:tesseractTests/EmittedPathFidelityTests \
  -only-testing:tesseractTests/EmittedPathRegistrationTests \
  -only-testing:tesseractTests/ServerCompletionUnkeyedSequencingTests
```

The CPU benchmark (`--agent-cpu-bench`) uses the production loader and live
delivery mode for `p5 detok`, including terminal handling. Its log names the
selected path and measures increasing newline-free lengths. Linear cost is
required of the recognized byte path; naive fallback retains its current cost.

## Canonical-echo fidelity gate (corpus mode)

`CanonicalEchoFidelityTests` runs with the suites above (fake tokenizer, no
extra setup). The corpus gate — `CanonicalEchoFidelityCorpusTests` — replays a
recorded session corpus (the `HTTPRequestLogger` request JSONs) through the
real normalization + reasoning-repair + probe machinery with a real model
tokenizer loaded through `AppTokenizerLoader`, and fails on any boundary whose
derived leaf/speculation path is not a token-identical prefix of the next
request's render (PRD #94). It is opt-in via environment because the corpus
contains user project content and
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

`EmittedPathReplayCorpusTests` (ADR-0063, tickets #475/#476/#477) uses the same
production tokenizer loader and walks the same recorded sessions through the
canonical-echo harness with a private **Emitted Path Index** learning every echoed
turn — the Leaf
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

`SnapshotLayerKindTests` pins the **Layer Kind** (#521): capture of every
vendor cache class the snapshot supports (simple, quantized, rotating,
chunked, arrays and Mamba) derives sliceable attention or whole-state; the
shape guard keeps a mis-shaped attention layer whole-state (behind the
snapshot's offset, a short token axis, flat arrays, empty state); moved,
deserialized and chain-hydrated layers carry the kind; and the extraction
edge and check-out eligibility read it rather than the class.

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
image-bearing session on a vision-container instance (the text turn before
the image resolved to its emitted path, only the glue with the pad expanded
into the processor's run fed, the image-bearing turn registered in render
space, the request after it restoring that whole leaf); index eviction
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
or `StablePrefixDetector`. Every loaded-model command forwards extra arguments
to the harness, so `--bench-model-id <catalog id>` picks the model (default:
`ModelDefinition.defaultAgentModelID`). The correctness runner is the stronger gate (bitwise
tensor comparison via raw `ModelContainer.perform` access); the e2e runner
exercises the full HTTP path and is the right shape for catching pipeline
regressions the correctness runner can't see.
The correctness runner also compares a moved leaf restored by copy against
cold-prefill logits bitwise (`movedLeafRestoredByCopyMatchesBitwise`).

The e2e image scenario (Step Z) measures against a **text-prefix
baseline**: after the cold image-add turn, a text-only probe runs twice
under the same system prompt (the first pass captures the system checkpoint
a cold image plan cannot capture inside its image prefix; the second pass
restores it). The follow-up turn and the agent-shaped history must restore
*more* than that baseline (a restore past the image run), and the
different-image turn — same text, same pixel size, different bytes — must
restore *no more* than it (ADR-0007 phase 2 lets it reuse the text prefix;
anything beyond is the first image's digest-keyed run serving the second).
The agent-shaped history extends a second HTTP-stored image turn of its own:
a follow-up supersedes the leaf it extends with its own leaf past the
follow-up's prompt (Leaf Handoff, ADR-0064), so the leaf the HTTP follow-up
consumed cannot serve a second reader. All warm turns and the
different-image turn run before the cache-clearing reload that produces the
cold references for the output-equivalence checks: after that reload the
only stored state is a cold turn's own leaf, past its prompt, so nothing
below the image could be restored. The runner mirrors
the server's accumulator for a `<think>` block the token cap cut open (the
buffered thinking is folded into the visible text), so the history it
replays renders exactly like the stored turn. The scenario skips (passing)
when the loaded *instance* is a text class (issue #439): a text-only
checkpoint, or one whose layout the vision factory refuses. Both Qwen3.8-27B
entries load the vision class since ADR-0089, so on them the scenario runs
with the DFlash2 draft resident and every image turn speculates. A
`bonsai-2-27b` run on 2026-09-19 passed every check, 33 of 33 (follow-up 298
and agent history 297 vs baseline 177, different image 177, warm and agent
outputs byte-equal to cold).

The e2e runner reads three switches for memory bisects. `TESSERACT_E2E_SPECULATION=off|mtp|dflash2|automatic`
pins the drafter policy (unset = Automatic, the catalogue default).
`TESSERACT_E2E_RELOAD_ONLY=<n>` reloads the engine n times with no requests
in between and exits, isolating load/unload memory from request memory;
`TESSERACT_E2E_HOLD_SECONDS=<s>` then keeps the process alive, model
unloaded, for a heap tool. `TESSERACT_SKIP_WARMUP_GENERATION=1` (read by
`LLMActor`, any harness) skips the one-token warmup after a load. Pair them
with `TESSERACT_ALLOCATION_DIAGNOSTICS=1` and read the `allocationMemory`
events from the cache diagnostics log.

Benchmark-shaped siblings (informational, not gates):
`scripts/dev.sh prefill-step-benchmark` and `scripts/dev.sh paroquant-vlm-smoke`.
The VLM smoke currently traps after its load check on every vision model tried
(`qwen3.5-4b-paro`, `bonsai-2-27b`, 2026-09-18) at the vendor precondition
`Qwen35 cannot continue a warm prompt cache without qwen35.ropeDeltas`: its
warm-continuation step predates that precondition (2026-08-10) and needs
updating before it says anything again. The load check before the trap is
still informative.

`scripts/dev.sh rotated-checkpoint-parity` is the **Rotated Ternary
Checkpoint** gate (ADR-0067; `MODEL_ID` defaults to `bonsai-2-27b`, and extra
arguments reach the binary as for the other loaded-model subcommands). A
loader that skips the Hadamard rotation decodes plausible garbage, not an
error, so the only proof is an independent implementation. The Swift half
(`RotatedCheckpointParityRunner`, `--rotated-checkpoint-parity`) loads the
pack through `AgentEngine`, asserts the manifest modules were substituted
with rotated layers (the load's stacking pass folds q|k|v, gate|up and the
GDN qkv|z, so the count reads 257 rotated linear leaves — 129 standalone and
128 stacked — not the manifest's 401), greedy-decodes a fixed prompt and
writes the prompt and generated token ids to the JSON report (latest.json)
in `benchmark/rotated-checkpoint-parity/`.
The reference half (`scripts/rotated_checkpoint_reference.py`, run from
`research/bonsai-venv` with mlx-vlm installed) decodes the same prompt ids
through mlx-vlm's `prism_hadamard_qwen35` and scores two things: the greedy
common prefix (weak — two engines' float noise eventually forks a greedy
trajectory) and the teacher-forced agreement (the Swift continuation fed back
through the reference in one pass; a missing rotation scores near zero, float
noise costs a token or two). PASS needs a prefix of 16 and agreement of 0.9.
mlx-vlm loads the pack's float32 norms as stored and MLX promotes its residual
stream to float32; the app casts them to the manifest's float16 at load, so
`PARITY_REFERENCE_ARGS=--match-app-dtypes` runs the reference with the same
cast and isolates the rotation logic from that difference. Run it for any
change to the vendor's `HadamardQuantized` layers (MLXLMCommon), the
`PrismHadamardQwen35` classes, the same-input projection stacking pass
(`SameInputProjectionStacking`, which folds rotated siblings that share a sign
vector and runs on every load), `ModelIdentity.baseArchitecture`, or a new
rotated pack in the catalog.

`scripts/dev.sh trace-replay` is the odd one out: it needs **no loaded
model**. It replays the Completion Trace Log corpus through the offline
LRU-baseline harness (`TraceReplayHarness`, PRD #82 slice #85) and writes
the report to `benchmark/trace-replay/latest.log`.

## Prefix cache, KV and memory

Testing the prefix cache, the KV cache or request memory: their suites,
opt-in gates and evidence runs (Leaf Lease and Checkout, Cache Claim, Warm
Bodies, the SSD writer and reads, attention capacity, TurboQuant KV) are in
[prefix-cache-testing.md](prefix-cache-testing.md).
