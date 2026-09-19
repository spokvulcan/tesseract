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
  -only-testing:tesseractTests/WarmBodyModelSessionTests \
  -only-testing:tesseractTests/WarmBodyDrainTests \
  -only-testing:tesseractTests/SnapshotLayerKindTests \
  -only-testing:tesseractTests/LeafCaptureHandoffTests \
  -only-testing:tesseractTests/LeafLeaseTests \
  -only-testing:tesseractTests/TokenRadixTreeTests \
  -only-testing:tesseractTests/StablePrefixDetectorTests \
  -only-testing:tesseractTests/PrefixCacheManagerTests \
  -only-testing:tesseractTests/PrefixCacheIntegrationTests \
  -only-testing:tesseractTests/CheckpointCaptureTests \
  -only-testing:tesseractTests/PrefixViewModelSessionTests \
  -only-testing:tesseractTests/CacheKeySpaceTests \
  -only-testing:tesseractTests/PrefillPlannerTests \
  -only-testing:tesseractTests/LeafAdmissionBuilderTests \
  -only-testing:tesseractTests/ConversationRenderSourceShapeTests \
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

# Dictation overlay freeze (no unit-test seam: the hang lives in SwiftUI's
# key-view loop on macOS 27.0; tools/overlay-focus-hang-lab is the regression
# loop — no flags must exit 2 while the OS still hangs, --unfocusable must exit 0):
swift run --package-path tools/overlay-focus-hang-lab overlay-focus-hang-lab --unfocusable

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

For a validation run that must not load models, prefix the command with
`TEST_RUNNER_XCTestSessionIdentifier=prefix-cache-unit-tests`. The existing
test-host detector makes `DependencyContainer.setup()` return before app
bootstrap, including Whisper, proofreader and memory-model prewarms. Use an
explicit suite allowlist (the prefix-cache block above plus touched suites)
after checking its fixtures, rather than the broad target. In particular,
`MemoryBaselineTests`, `MemoryEvalTests`, `MemoryRecallEvalTests`,
`MemoryEmbedderQualityTests`, and `RecallToolSmokeTests` intentionally load the
installed embedder. The prefix block's `Real` suites load tokenizer files,
not model weights. Leave corpus and allocation opt-ins unset.

The planned Prefix-View Checkpoint slice (#524, ADR-0068) is covered at the
existing seams. `CheckpointCaptureTests` checks synchronized whole-state-only
capture. `PrefixViewModelSessionTests` compares exact cache bytes and generated
tokens against an owned checkpoint for plain and quantized attention, with
disjoint backing addresses; it also covers the prepared image-prefix capture
entry. `SnapshotResolutionLadderTests` checks nearest/unleased selection,
recency and all fall-through rungs. `TokenRadixTreeTests` checks byte accounting,
eviction exclusion, lease/check-in and last-backer self-heal.
`SnapshotResolutionTests` checks both Restore Pins, view-only panel bytes, and
retirement after the final active request skips leaf storage. `LeafCheckoutTests`
checks that views retain the `checkpoint` copy reason in image and quantized
partitions.
`ServerCompletionExtractSnapshotPayloadsTests` keeps views RAM-only;
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

Known miss in the e2e image scenario (2026-09-18): with a Qwen3.8-template
model loaded in vision mode (`qwen3.8-27b-paro`, `bonsai-2-27b`),
`requestZ2_followup_restores_past_image` and
`agent_image_history_lands_cache_aware` report `cachedTokens=0`. The runner
caps every reply at 32 tokens and these models are still inside `<think>`
at the cap; the canonical leaf stored after Z1 (298 tokens) and the
follow-up's render of that same reply then part four tokens into the
assistant turn (shared prefix 261), so Z2 prefills cold. The image path is
intact — warm and cold outputs are identical, Z5 never hits, Z6b reuses the
text prefix through the image — and `qwen3.5-2b` (a template without
`preserve_thinking`) passes both checks with the same truncated reply. Read
the two checks as a runner limitation until the cap or the truncated-think
history render is settled; the rest of the report reads as usual.

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
wraps this gate in fixed resource stops (32 GiB sampled footprint, 2 GiB additional
system swap, critical/unknown pressure, ten-minute deadline). Use a Release binary,
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

The Qwen3.8 community checkpoint used by this comparison loads as a text
instance even when vision is requested (ADR-0056). The current HTTP E2E
runner's config-based image scenario is therefore not real VLM coverage;
its different-image assertion also fails on the unchanged baseline. The
comparison report retains that failure instead of presenting it as a green
image gate. Use an actual vision-loaded model for image-specific validation.

### Tree-side Leaf Lease evidence (#479)

`LeafLeaseTests` exercises body-drop refusal, pressure and Restore Pin
age-out, RAM clear, demotion and queued promotion, same-path replacement,
ancestor supersession, check-in growth, both writer/acquisition race orders,
writer failures, and base/suffix ordering. All caches are small, real MLX
caches; production checkout is covered by the #480 suites below.

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
Only explicit quiescent check-in/rewind ends a lease; `completeRequest`, the
pin age-out limit, forced SSD flush and write-eagerness timeout cannot do so.

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


### Production Leaf Checkout and Rewind evidence (#480)

`LeafCheckoutTests` checks object identity and physical array independence,
body removal/accounting, exact recurrent state and metadata after growth,
every intentional fallback, and pending-full-payload materialization. It also
covers the bounded pending-payload wait (#523): a payload that materializes
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
manifest. Results are **not run** and Warm Bodies remain default-off. No loaded
workload is authorized on the preparation Mac.

The existing `CanonicalEchoFidelityCorpusTests` reads tokenizer files and
checks token paths. It does not restore a Warm Body or establish generated
token parity. Existing loaded cache runners also do not implement the #528
three-arm timing protocol. The pre-registration records this execution gap;
owner-reviewed instrumentation and a frozen manifest are required before the
loaded campaign. The prefix suites above remain the small-cache regression
evidence; their success does not flip the flag or unblock #531.
