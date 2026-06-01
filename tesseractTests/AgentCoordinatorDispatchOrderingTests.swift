//
//  AgentCoordinatorDispatchOrderingTests.swift
//  tesseractTests
//
//  The one residual test that still needs the coordinator: it pins the
//  `isGenerating`-before-rebuild ordering invariant in the sequencing dispatcher.
//
//  It drives the PUBLIC `sendMessage` path against an `Agent` whose injected
//  `generate` scripts a stream (yield a chunk → finish), exercising the real
//  `.agentStart`/`.messageUpdate`/`.agentEnd` sequence through the production
//  subscribe + dispatch path. If the dispatcher ever rebuilt the transcript
//  *before* flipping the busy flag on `.agentEnd`, a streaming indicator would
//  persist in committed state and this test would fail. Far thinner than the
//  whole-coordinator characterizations it replaces.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentCoordinatorDispatchOrderingTests {

    /// An `Agent` whose generate yields one text chunk then finishes — enough to
    /// drive `.agentStart` → `.messageUpdate` → `.agentEnd` through the real loop.
    private func makeScriptedAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "dispatch-ordering-test-model"),
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
                    continuation.yield(.text("hello"))
                    continuation.finish()
                }
            }
        )
    }

    private func isStreamingRow(_ row: ChatRow) -> Bool {
        if row.id.hasPrefix("streaming-") { return true }
        switch row.kind {
        case .streamingIndicator, .streamingText:
            return true
        case .turnHeader(let header):
            return header.isGenerating
        default:
            return false
        }
    }

    @Test func committedRowsCarryNoStreamingIndicatorAfterCompletion() async throws {
        let coordinator = AgentCoordinator(
            agent: makeScriptedAgent(),
            conversationStore: InMemoryAgentConversationStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.sendMessage("hi")
        #expect(coordinator.isGenerating == true)   // eager, before the loop starts

        // Let the real .agentStart/.messageUpdate/.agentEnd dispatch run to settle.
        let deadline = ContinuousClock.now + .seconds(3)
        while coordinator.isGenerating {
            try await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("Coordinator did not settle within timeout")
                break
            }
        }

        #expect(coordinator.isGenerating == false)
        // The committed transcript carries no streaming row — proving the
        // dispatcher flipped the busy flag BEFORE rebuilding on `.agentEnd`.
        #expect(coordinator.rows.contains(where: isStreamingRow) == false)
        // And the committed answer is present.
        #expect(coordinator.rows.contains { row in
            if case .assistantText(let answer) = row.kind { return answer.content == "hello" }
            return false
        })
    }
}
