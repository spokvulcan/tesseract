# Testing

Tests use the Swift `Testing` framework (not XCTest), in `tesseractTests/`. Run
before committing changes to server, caching, or agent engine code.

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
# the Timeline (tomorrow after today, and the small hours past midnight), the
# Now Card (its decision table, the Inbox's slot offers, how
# long a card's line stays fresh), Today rendered with the app's wiring over
# fixture days at a wide, a regular and a phone width
# (TEST_RUNNER_TODAY_GALLERY_DIR=<dir> also writes each render there as a PNG,
# dark and light), the Step Cue (a planned slot put on the panel at its start,
# once, and never while away, quiet, in a call, a game or a meeting, or over a
# panel that is up; a started step's check-in at its end, which no other start
# interrupts; and what each choice does), the Jarvis Panel's content over
# fixture cards at the height it fits them (the plan's steps, the wrap-up's
# leftovers; TEST_RUNNER_PANEL_GALLERY_DIR=<dir> writes the PNGs),
# cards and prompts, the Day Thread's store, the seen ledger, the
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
  -only-testing:tesseractTests/JarvisPanelGalleryTests \
  -only-testing:tesseractTests/NotificationSourceTests \
  -only-testing:tesseractTests/ModifierKeyDetectorTests \
  -only-testing:tesseractTests/CardItemActionTests \
  -only-testing:tesseractTests/NightReflectionTests \
  -only-testing:tesseractTests/DayStateStoreTests \
  -only-testing:tesseractTests/AgendaToolsTests \
  -only-testing:tesseractTests/EventKitAgendaStoreTests \
  -only-testing:tesseractTests/AgendaTimeTests \
  -only-testing:tesseractTests/CaptureParserTests \
  -only-testing:tesseractTests/NudgePlannerTests \
  -only-testing:tesseractTests/TimelineBuilderTests \
  -only-testing:tesseractTests/NowCardTests \
  -only-testing:tesseractTests/TodayGalleryTests \
  -only-testing:tesseractTests/CardParserTests \
  -only-testing:tesseractTests/MomentPromptsTests \
  -only-testing:tesseractTests/DayThreadTests \
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

### Bounded production parity and projection lifetime

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

The Qwen3.8 community checkpoint used by this comparison loads the vision
class when vision is requested (ADR-0089), so the HTTP E2E runner's image
scenario runs on it; this correctness check itself rejects image requests.

### Tree-side Leaf Lease evidence (#479)

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


### Production Leaf Checkout and Rewind evidence (#480)

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

### Cache Claim (#554)

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

### Opt-in Warm Bodies (#527, #529)

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

### Boundary leaf capture by move (ADR-0064 amendment, #501)

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

### Backing Leaf credit (ADR-0068 amendment)

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

### Warm-backed Prefix-View Checkpoints (#530)

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
Use the same app-host guard and prefix suite allowlist above. No loaded-model
work is part of this unit-test evidence.

### Warm Body parity pre-registration (#528)

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
The prefix suites above remain the small-cache regression evidence; their
success does not flip the flag or unblock #531.

### Attention capacity compaction (#534)

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

### No-copy SSD writer (#469)

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
the prefix cache suites above. The owner-only `--ssd-read-bench` harness and
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
