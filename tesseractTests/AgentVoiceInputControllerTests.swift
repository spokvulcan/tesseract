//
//  AgentVoiceInputControllerTests.swift
//  tesseractTests
//
//  Tests the **Voice Input** module as a thin composer over the **Voice Capture
//  Session** — no `Agent`, no arbiter, no conversation store. Asserts the
//  caller-specific behavior: emit-on-commit via `onVoiceTranscription`, the
//  `AgentVoiceState` mapping, and that errors stay local to `voiceState`. The deep
//  staleness/supersede race (a late success after cancel committing nothing) now
//  lives once in `VoiceCaptureSessionTests`. Uses the engine-facing
//  `ControllableTranscribing` double so a success is delivered deterministically.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentVoiceInputControllerTests {

    @MainActor
    private final class CallbackRecorder {
        private(set) var values: [String] = []
        func record(_ value: String) { values.append(value) }
    }

    private func makeController(
        capture: FakeAudioCapture,
        engine: ControllableTranscribing
    ) -> AgentVoiceInputController {
        AgentVoiceInputController(
            audioCapture: capture,
            transcriptionEngine: engine,
            settings: SettingsManager(store: InMemorySettingsStore())
        )
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
        for _ in 0..<500 where recorder.values.isEmpty { await Task.yield() }

        // Emits the post-processed transcription exactly once — the controller's
        // job is to emit, not to re-implement the post-processor's transform.
        #expect(recorder.values == [TranscriptionPostProcessor().process("hello world")])
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

    // MARK: - Fail safe when dependencies are missing

    /// With no capture/transcription dependencies the controller has no session, so
    /// `start()` fails safe with a local "not available" error rather than
    /// half-working.
    @Test func startWithoutDependenciesSurfacesUnavailableError() {
        let controller = AgentVoiceInputController()

        controller.start()

        #expect(controller.voiceState == .error("Voice input not available"))
    }
}
