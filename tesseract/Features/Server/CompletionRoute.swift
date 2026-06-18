import Foundation
import MLXLMCommon

/// The **Completion Route** — the pure request-shape decision for one HTTP
/// completion: serve it on the cache-aware **Server Completion** path, or fall
/// back to the standard managed path, with the reason named.
///
/// Owned by the dispatcher (`ServerInferenceService`). Computed from the
/// conversation shape only — never from model state — so every bypass case is
/// unit-testable without a loaded model. The **Server Completion** module never
/// sees a request it cannot serve (ADR-0015).
nonisolated enum CompletionRoute: Equatable, Sendable {
    /// The request canonicalized into the prefix-cache conversation shape and
    /// ends on a non-assistant message — serve it cache-aware.
    case cacheAware(HTTPPrefixCacheConversation)
    /// The request cannot ride the prefix cache; fall back to the standard
    /// managed path. The reason's raw value is the wire string for logs.
    case standard(reason: StandardReason)

    nonisolated enum StandardReason: String, Equatable, Sendable {
        /// The request could not be canonicalized into the prefix-cache
        /// conversation shape (non-text content, unsupported roles, …).
        case noUsablePrefixCacheConversation = "no-prefix-cache-conversation"
        /// The canonicalized conversation has no messages to complete from.
        case emptyConversation = "empty-conversation"
        /// The conversation ends on an assistant turn — there is nothing to
        /// complete on the prefix-cache shape.
        case lastMessageAssistant = "last-message-assistant"
    }

    static func decide(conversation: HTTPPrefixCacheConversation?) -> CompletionRoute {
        guard let conversation else {
            return .standard(reason: .noUsablePrefixCacheConversation)
        }
        guard let lastMessage = conversation.lastMessage else {
            return .standard(reason: .emptyConversation)
        }
        guard lastMessage.role != .assistant else {
            return .standard(reason: .lastMessageAssistant)
        }
        return .cacheAware(conversation)
    }
}
