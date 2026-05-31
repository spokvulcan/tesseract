import Foundation
import Testing
@testable import Tesseract_Agent

/// Regression for the agent-path malformed-tool-call surfacing (PRD #30, User
/// Story 4). When a turn produces only an interrupted `<tool_call>` at EOS, the
/// raw buffer must become the committed message content so the turn is persisted
/// and the user can see a tool call was attempted — mirroring the server path's
/// `CompletionHandler.malformedFallbackText`. Before the fix the
/// `.malformedToolCall` event reached a dead accumulator field and the turn was
/// dropped as contentless (empty text + no tool calls).
@MainActor
struct AgentLoopMalformedToolCallTests {

    /// Collects the assistant messages carried by `.messageEnd` events.
    private nonisolated final class AssistantMessageRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [AssistantMessage] = []

        func record(_ event: AgentEvent) {
            guard case .messageEnd(let message) = event,
                  let assistant = message as? AssistantMessage else { return }
            lock.lock(); messages.append(assistant); lock.unlock()
        }

        var last: AssistantMessage? {
            lock.lock(); defer { lock.unlock() }
            return messages.last
        }
    }

    @Test func malformedToolCallAtEosSurfacesAsContentAndPersistsTurn() async throws {
        // An interrupted tool call: `<tool_call>` body with no close tag, nothing
        // else — no text, no parsed `.toolCall`.
        let raw = "<tool_call>\n{\"name\":\"read\",\"arguments\":{\"file_path\":\"/x\"}\n</tool_call>"
        let generate: LLMGenerateFunction = { _, _, _, _ in
            AsyncThrowingStream { continuation in
                continuation.yield(.malformedToolCall(raw))
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

        // The committed message surfaces the dropped buffer as content...
        let assistant = try #require(recorder.last)
        #expect(assistant.content == raw)
        #expect(assistant.toolCalls.isEmpty)

        // ...and the turn is persisted into context (hasContent == true), not
        // dropped as a blank turn.
        let persistedAssistants = context.messages.compactMap { $0 as? AssistantMessage }
        #expect(persistedAssistants.contains { $0.content == raw })
    }

    @Test func plainEmptyTurnWithoutMalformedBufferStaysDropped() async throws {
        // No content of any kind — the fallback must NOT fabricate content, so the
        // contentless-turn drop still holds.
        let generate: LLMGenerateFunction = { _, _, _, _ in
            AsyncThrowingStream { $0.finish() }
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
        #expect(assistant.content.isEmpty)
        let persistedAssistants = context.messages.compactMap { $0 as? AssistantMessage }
        #expect(!persistedAssistants.contains { _ in true })
    }
}
