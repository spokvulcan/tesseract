import Foundation
import Testing
@testable import Tesseract_Agent

/// Regression coverage for the `finishRun` actor-isolation crash.
///
/// `Agent` is `@MainActor`, so `finishRun` — which writes `state.messages` and
/// `context` and drains buffered events — must run on the MainActor. The run
/// loop body is `@Sendable` and executes off-MainActor, so the `Task`
/// continuation in `beginRun` that calls `finishRun` also resumes off-MainActor
/// unless it explicitly hops back.
///
/// Before the fix, `beginRun` called `self?.finishRun(result)` directly on that
/// off-MainActor continuation. `finishRun`'s
/// `state.messages = finalContext.messages.map { $0 as any AgentMessageProtocol }`
/// then trips Swift's dynamic actor-isolation check (`_swift_task_checkIsolated`)
/// and traps with SIGTRAP — and, on the way, races MainActor readers of `state`
/// (the original crash log caught it running concurrently with
/// `AgentCoordinator.handleAgentEvent` on the main thread).
///
/// The generate stream suspends before finishing so the loop genuinely yields
/// the executor; that suspension is what lets the continuation resume on a
/// background cooperative thread, reproducing the real crash rather than a
/// synchronous fast path that never leaves the MainActor.
@MainActor
struct AgentFinishRunIsolationTests {

    private func makeAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "finishrun-isolation-test-model"),
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
                    Task {
                        // A real suspension point so `await body(...)` in
                        // beginRun yields and its continuation resumes
                        // off-MainActor — the condition under which the
                        // pre-fix `finishRun` trapped.
                        try? await Task.sleep(for: .milliseconds(5))
                        continuation.finish()
                    }
                }
            }
        )
    }

    @Test func promptDrivesFinishRunOnMainActorAndCommitsMessages() async throws {
        let agent = makeAgent()

        agent.prompt(UserMessage(content: "hello"))
        await agent.waitForIdle()

        // Reaching here at all means `finishRun` completed without tripping the
        // isolation trap. The committed projection must reflect the loop result.
        #expect(agent.state.isBusy == false)
        #expect(agent.state.messages.contains { ($0 as? UserMessage)?.content == "hello" })
    }
}
