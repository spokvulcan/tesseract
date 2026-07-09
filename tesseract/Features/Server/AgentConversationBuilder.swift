import Foundation
import MLXLMCommon
import os

/// The agent-side adapter into the prefix-cache conversation shape — the
/// second of the two adapters producing the one `HTTPPrefixCacheConversation`
/// value (the HTTP edge's `MessageConverter` is the first; PRD #72). It
/// canonicalizes agent history — system prompt, user turns with image
/// attachments, assistant turns with tool calls, tool results — so internal
/// agent generation routes server-compatible and the **Completion Route**
/// decides cache-aware vs standard from the conversation shape, exactly as it
/// does for HTTP. The canonicalization rules live in the shared shape
/// (`HTTPPrefixCacheMessage` / `HTTPPrefixCacheConversation` normalization),
/// not here — this adapter only maps message forms.
nonisolated enum AgentConversationBuilder {

    /// Canonicalize one agent request into the conversation value, or `nil`
    /// when the history carries content the shape cannot (an attachment that
    /// no longer decodes) — the request then rides the standard route,
    /// uncached but correct, mirroring the HTTP edge's eligibility bail.
    static func conversation(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?
    ) -> HTTPPrefixCacheConversation? {
        var converted: [HTTPPrefixCacheMessage] = []
        converted.reserveCapacity(messages.count)
        for message in messages {
            switch message {
            case .system(let content):
                converted.append(HTTPPrefixCacheMessage(role: .system, content: content))

            case .user(let content, let images):
                var cacheImages: [HTTPPrefixCacheImage] = []
                cacheImages.reserveCapacity(images.count)
                for attachment in images {
                    guard let digest = cachedDigest(for: attachment) else { return nil }
                    cacheImages.append(HTTPPrefixCacheImage(data: attachment.data, digest: digest))
                }
                converted.append(
                    HTTPPrefixCacheMessage(
                        role: .user, content: content, images: cacheImages
                    ))

            case .assistant(let content, let reasoning, let toolCalls):
                converted.append(
                    .assistant(
                        content: content,
                        reasoning: reasoning,
                        toolCalls: (toolCalls ?? []).map {
                            HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                        }
                    ))

            case .toolResult(_, let content, let images):
                // The prefix-cache shape carries images on user messages only;
                // an image-bearing tool result (browser screenshot) makes the
                // request ineligible — it rides the standard route, uncached
                // but with the pixels intact — exactly as the HTTP edge bails
                // via `.nonTextToolMessage`.
                guard images.isEmpty else { return nil }
                // The chat template matches tool results to calls positionally;
                // the agent loop already appends them in call order, so the id
                // is dropped here exactly as the HTTP edge drops it.
                converted.append(HTTPPrefixCacheMessage(role: .tool, content: content))
            }
        }

        return HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: converted,
            toolDefinitionsDigest: toolSpecsDigest(toolSpecs)
        )
    }

    /// Per-attachment digest + decodability verdict, memoized by the
    /// attachment's stable identity (`nil` = undecodable). Agent histories
    /// re-canonicalize on every generate call — each turn and each tool
    /// round-trip — and `ImageAttachment.data` is immutable, so hashing the
    /// same megabytes of pixels per step is pure waste on the TTFT path.
    /// Entries hold only the 32-byte digest (never the pixel bytes), and the
    /// cache resets at a generous bound rather than tracking session
    /// lifetimes.
    private static let digestCache = OSAllocatedUnfairLock<[UUID: ImageDigest?]>(initialState: [:])

    private static func cachedDigest(for attachment: ImageAttachment) -> ImageDigest? {
        if let cached = digestCache.withLock({ $0[attachment.id] }) {
            return cached
        }
        let verdict: ImageDigest? =
            attachment.ciImage != nil
            ? ImageDigest(imageBytes: attachment.data)
            : nil
        digestCache.withLock { cache in
            if cache.count >= 512 { cache.removeAll(keepingCapacity: true) }
            cache[attachment.id] = verdict
        }
        return verdict
    }

    /// Stable digest of the agent's tool definitions. Encoded from the
    /// `ToolSpec` dictionaries, so it differs from the HTTP edge's digest of
    /// the same tools (different wire types) — which is fine: the digest only
    /// gates session-replay continuity *within* one adapter's stream of
    /// turns, and the radix key path is digest-independent. Per-adapter
    /// stability is the requirement, not cross-adapter equality.
    private static func toolSpecsDigest(_ toolSpecs: [ToolSpec]?) -> String {
        guard let toolSpecs, !toolSpecs.isEmpty,
            JSONSerialization.isValidJSONObject(toolSpecs),
            let data = try? JSONSerialization.data(
                withJSONObject: toolSpecs, options: [.sortedKeys]
            )
        else {
            return HTTPPrefixCacheConversation.emptyToolDefinitionsDigest
        }
        return httpPrefixCacheDigest(for: data)
    }
}
