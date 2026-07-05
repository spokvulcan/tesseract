//
//  AgentGenerationErrorSurfacingTests.swift
//  tesseractTests
//
//  A generation failure on the in-app agent path must reach the user. The HTTP
//  server already returns the error to its client; the in-app loop historically
//  only logged it (`Log.agent.error`) and emitted the normal turn/agent-end, so
//  the chat just stopped with no banner. These pin the fix: the loop emits a
//  `.generationError` carrying the message, and the coordinator surfaces it in
//  the shared `error` banner — driven through the PUBLIC `sendMessage` path.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentGenerationErrorSurfacingTests {

    /// An `Agent` whose generate stream throws immediately with the given engine
    /// error — the shape a vision-tower rejection takes (thrown at prefill, no
    /// tokens produced).
    private func makeThrowingAgent(_ error: AgentEngineError) -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "generation-error-test-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in
                AsyncThrowingStream { continuation in
                    continuation.finish(throwing: error)
                }
            }
        )
    }

    private func settle(_ coordinator: AgentCoordinator) async {
        let deadline = ContinuousClock.now + .seconds(3)
        while coordinator.isGenerating {
            try? await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("Coordinator did not settle within timeout")
                break
            }
        }
    }

    @Test func generationFailureSurfacesItsMessageInTheErrorBanner() async {
        let message =
            "this image set is too large to process: 35028 combined image patches. "
            + "Reduce the number or size of the attached images."
        let coordinator = AgentCoordinator(
            agent: makeThrowingAgent(.generationFailed(message)),
            conversationStore: InMemoryAgentConversationStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            batchEngine: InMemoryInferenceArbiter().makeBatchEngine()
        )

        coordinator.sendMessage("describe these")
        await settle(coordinator)

        // The rejection text the user needs to see is in the banner, and the
        // composer is no longer stuck in the generating state.
        #expect(coordinator.error == message)
        #expect(coordinator.isGenerating == false)
    }
}
