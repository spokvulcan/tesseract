//
//  AgentVoiceInputControllerTests.swift
//  tesseractTests
//
//  Tests the **Voice Input** module as a thin presenter over a **Voice Capture
//  Session** from **Voice Capture** — no `Agent`, no arbiter, no conversation
//  store. Asserts the surface-specific behavior: emit-on-commit via
//  `onVoiceTranscription`, the `AgentVoiceState` mapping, that errors stay local
//  to `voiceState`, that its takes run every stage, and the one rule for spoken
//  words joining typed ones. The deep staleness/supersede race (a late success
//  after cancel committing nothing) lives once in `VoiceCaptureSessionTests`.
//  Uses the engine-facing `ControllableTranscribing` double so a success is
//  delivered deterministically.
//

import Foundation
import Observation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct AgentVoiceInputControllerTests {

    @MainActor @Observable
    fileprivate final class CallbackRecorder {
        private(set) var values: [String] = []
        func record(_ value: String) { values.append(value) }
    }

    private func makeController(
        capture: FakeAudioCapture,
        engine: ControllableTranscribing,
        proofread: @escaping @MainActor (String) async -> ProofreadVerdict? = { _ in nil }
    ) -> AgentVoiceInputController {
        AgentVoiceInputController(
            session: makeVoiceCapture(
                microphone: capture, recognizer: engine, proofread: proofread
            ).makeSession())
    }

    // MARK: - Emit, not send

    /// `finishCapture()` transcribes and emits the processed text via the
    /// callback, then returns to `.idle`. It does not send anywhere else.
    @Test func finishCaptureEmitsTranscribedTextViaCallback() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let controller = makeController(capture: capture, engine: engine)

        let recorder = CallbackRecorder()
        controller.onVoiceTranscription = { recorder.record($0) }

        controller.start()
        #expect(controller.voiceState == .recording)

        controller.finishCapture()
        #expect(controller.voiceState == .transcribing)
        while !engine.isAwaiting { await Task.yield() }

        engine.completeWithSuccess()
        await observe(until: { !recorder.values.isEmpty })

        // Emits the post-processed transcription exactly once — the controller's
        // job is to emit, not to re-implement the post-processor's transform.
        #expect(recorder.values == [TranscriptionPostProcessor().process("hello world")])
        #expect(controller.voiceState == .idle)
    }

    /// A take the Proofread Pass rejects goes to `onVoiceRejected` when a
    /// surface set it — the capture panel shows it to check instead of
    /// saving it as heard — and not to `onVoiceTranscription`.
    @Test func rejectedTakeGoesToTheRejectedCallbackWhenSet() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let controller = makeController(
            capture: capture, engine: engine,
            proofread: { _ in .rejected(reason: "garbled noise") })
        let emitted = CallbackRecorder()
        let rejected = CallbackRecorder()
        controller.onVoiceTranscription = { emitted.record($0) }
        controller.onVoiceRejected = { rejected.record($0) }

        controller.start()
        controller.finishCapture()
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        await observe(until: { !rejected.values.isEmpty })

        #expect(rejected.values == [TranscriptionPostProcessor().process("hello world")])
        #expect(emitted.values.isEmpty)
        #expect(controller.voiceState == .idle)
    }

    // MARK: - Minimum duration (StopResult → local error)

    /// A recording shorter than the minimum duration never reaches the engine and
    /// surfaces an error in `voiceState` (local, not a shared banner).
    @Test func recordingTooShortSurfacesErrorAndSkipsTranscription() async throws {
        let engine = ControllableTranscribing()
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 0.2))
        let controller = makeController(capture: capture, engine: engine)

        controller.start()
        #expect(controller.voiceState == .recording)

        controller.finishCapture()

        #expect(controller.voiceState == .error("Recording too short"))
        #expect(engine.isAwaiting == false)
    }

    // MARK: - Every stage, on every surface (ADR-0085 §2, ADR-0096)

    /// A Voice Input's take runs every stage of the module it came from — the
    /// Capture Dump, the Learned Words, the Proofread Pass — whichever surface
    /// holds it. The Onboarding Tour's included: there is no other way to
    /// build one.
    @Test func aVoiceInputsTakeRunsEveryStage() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "ask cloud", segments: [], language: "en", processingTime: 0))
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(
                samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0,
                raw: RawCapture(samples: [0.3], sampleRate: 48_000, voiceProcessed: true)))
        let dump = FakeCaptureDump()
        let words = LearnedWordStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("voice-input-learned-\(UUID().uuidString)"))
        let id = try #require(words.learn(heard: "cloud", meant: "Claude")?.wordID)
        let proofread = CallbackRecorder()
        let controller = AgentVoiceInputController(
            session: makeVoiceCapture(
                microphone: capture, recognizer: engine, captureDump: dump, learnedWords: words,
                proofread: { text in
                    proofread.record(text)
                    return nil
                }
            ).makeSession())
        let emitted = CallbackRecorder()
        controller.onVoiceTranscription = { emitted.record($0) }

        controller.start()
        controller.finishCapture()
        while !engine.isAwaiting { await Task.yield() }
        engine.completeWithSuccess()
        await observe(until: { !emitted.values.isEmpty })

        #expect(dump.saved.count == 1)
        #expect(proofread.values == ["Ask Claude"])
        #expect(emitted.values == ["Ask Claude"])
        #expect(words.word(withID: id)?.totalCatches == 1)
    }

    // MARK: - Spoken words join typed words

    /// One rule for every field: spoken words follow the words already typed
    /// with one space between, or stand alone in an empty field.
    @Test func spokenWordsJoinTypedWords() {
        #expect(AgentVoiceInputController.joining("call Anna", to: "") == "call Anna")
        #expect(AgentVoiceInputController.joining("call Anna", to: "  \n") == "call Anna")
        #expect(
            AgentVoiceInputController.joining("call Anna", to: "Tomorrow:")
                == "Tomorrow: call Anna")
        #expect(
            AgentVoiceInputController.joining("call Anna", to: " Tomorrow: \n")
                == "Tomorrow: call Anna")
    }
}
