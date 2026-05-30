//
//  AgentCoordinatorVoiceCancellationTests.swift
//  tesseractTests
//
//  Guards the voice-input stale-task contract: a transcription that completes
//  successfully *after* `cancelVoiceInput()` must not invoke
//  `onVoiceTranscription` or overwrite `voiceState`. Uses the engine-facing
//  `ControllableTranscribing` double so the late success is delivered
//  deterministically.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentCoordinatorVoiceCancellationTests {

    @MainActor
    private final class CallbackRecorder {
        private(set) var values: [String] = []
        func record(_ value: String) { values.append(value) }
    }

    private func makeAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "voice-cancel-test-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in AsyncThrowingStream { $0.finish() } }
        )
    }

    @Test
    func lateVoiceSuccessAfterCancelDoesNotInvokeCallback() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(text: "late voice", segments: [], language: "en", processingTime: 0)
        )
        let capture = FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let coordinator = AgentCoordinator(
            agent: makeAgent(),
            conversationStore: AgentConversationStore(),
            audioCapture: capture,
            transcriptionEngine: engine,
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        let recorder = CallbackRecorder()
        coordinator.onVoiceTranscription = { recorder.record($0) }

        coordinator.startVoiceInput()
        #expect(coordinator.voiceState == .recording)

        coordinator.stopVoiceInputAndSend()
        #expect(coordinator.voiceState == .transcribing)
        while !engine.isAwaiting { await Task.yield() }

        coordinator.cancelVoiceInput()
        #expect(coordinator.voiceState == .idle)
        #expect(engine.cancelCount == 1)

        // The engine returns a successful transcription *after* the cancel.
        engine.completeWithSuccess()
        for _ in 0..<500 { await Task.yield() }

        #expect(recorder.values.isEmpty)
        #expect(coordinator.voiceState == .idle)
    }
}
