import Foundation
import Testing
@testable import Tesseract_Agent

/// End-to-end regression at the one call site that carried the reclassify
/// ordering bug. Proves the fix is wired into the path users exercise — not
/// merely encoded in the accumulator. Drives `agentLoop` with a scripted
/// generation (text, then a `<think>` block that never closes) and asserts the
/// committed assistant message appends the reclassified thinking AFTER the text.
@MainActor
struct AgentLoopReclassifyTests {

    /// Collects the assistant messages carried by `.messageEnd` events.
    /// `@unchecked Sendable` + a lock because `emit` is `@Sendable` and may be
    /// invoked off the test's actor.
    private nonisolated final class AssistantMessageRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [AssistantMessage] = []

        func record(_ event: AgentEvent) {
            guard case .messageEnd(let message) = event,
                let assistant = message as? AssistantMessage
            else { return }
            lock.lock(); messages.append(assistant); lock.unlock()
        }

        var last: AssistantMessage? {
            lock.lock(); defer { lock.unlock() }
            return messages.last
        }
    }

    @Test func emittedMessageAppendsReclassifiedThinkingAfterText() async throws {
        // text emitted first, then a <think> block that never closes.
        let generate: LLMGenerateFunction = { _, _, _, _ in
            AsyncThrowingStream { continuation in
                continuation.yield(.text("pre"))
                continuation.yield(.thinkStart)
                continuation.yield(.thinking("R"))
                continuation.yield(.thinkReclassify)
                continuation.finish()
            }
        }

        let recorder = AssistantMessageRecorder()
        let emit: @Sendable (AgentEvent) -> Void = { recorder.record($0) }

        var context = AgentContext(systemPrompt: "", messages: [], tools: nil)
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "test"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        await agentLoop(
            prompts: [UserMessage(content: "hi")],
            context: &context,
            config: config,
            generate: generate,
            signal: nil,
            emit: emit
        )

        let assistant = try #require(recorder.last)
        #expect(assistant.text == "preR")
        #expect(assistant.thinking == nil)
    }
}
