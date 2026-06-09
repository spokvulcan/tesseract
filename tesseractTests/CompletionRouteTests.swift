import MLXLMCommon
import Testing
@testable import Tesseract_Agent

/// The **Completion Route** is the dispatcher's pure request-shape decision
/// (CONTEXT.md → Server completion): cache-aware versus standard-with-reason,
/// computed from the conversation shape only — never from model state. Every
/// bypass case is assertable here without model files or actors, mirroring
/// the Prefill Planner and Leaf Admission Builder suites.
struct CompletionRouteTests {

    @Test func missingPrefixCacheConversationRoutesStandard() {
        let route = CompletionRoute.decide(conversation: nil)
        #expect(route == .standard(reason: .noUsablePrefixCacheConversation))
    }

    @Test func emptyConversationRoutesStandard() {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: []
        )
        let route = CompletionRoute.decide(conversation: conversation)
        #expect(route == .standard(reason: .emptyConversation))
    }

    /// A conversation ending on an assistant turn has nothing to complete —
    /// the prefix-cache prompt render would re-open the closed assistant turn.
    @Test func lastMessageFromAssistantRoutesStandard() {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [
                .init(role: .user, content: "Hello"),
                .assistant(content: "Hi there"),
            ]
        )
        let route = CompletionRoute.decide(conversation: conversation)
        #expect(route == .standard(reason: .lastMessageAssistant))
    }

    @Test func userLastConversationRoutesCacheAwareWithSameConversation() {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [
                .init(role: .user, content: "Hello"),
                .assistant(content: "Hi there"),
                .init(role: .user, content: "Tell me more"),
            ]
        )
        let route = CompletionRoute.decide(conversation: conversation)
        #expect(route == .cacheAware(conversation))
    }

    /// Tool-result turns are completable — only an assistant-last shape
    /// bypasses, exactly as the in-actor guards behaved before the carve.
    @Test func toolResultLastConversationRoutesCacheAware() {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [
                .init(role: .user, content: "Read the file"),
                .assistant(content: "", toolCalls: [
                    .init(name: "read", arguments: ["path": .string("/tmp/x")]),
                ]),
                .init(role: .tool, content: "file contents"),
            ]
        )
        let route = CompletionRoute.decide(conversation: conversation)
        #expect(route == .cacheAware(conversation))
    }
}
