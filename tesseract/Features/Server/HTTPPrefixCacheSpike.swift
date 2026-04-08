import CryptoKit
import Foundation
import MLXLMCommon

nonisolated struct HTTPServerGenerationStart: Sendable {
    let stream: AsyncThrowingStream<AgentGeneration, Error>
    let cachedTokenCount: Int
}

nonisolated struct HTTPPrefixCacheKey: Hashable, Sendable {
    let modelID: String
    let kvBits: Int?
    let kvGroupSize: Int
    let toolDefinitionsDigest: String
    let templateContextDigest: String
}

nonisolated struct HTTPPrefixCacheToolCall: Hashable, Sendable {
    let name: String
    let argumentsJSON: String

    init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = canonicalizeHTTPPrefixCacheToolArgumentsJSON(argumentsJSON)
    }

    init(name: String, arguments: [String: JSONValue]) {
        self.name = name
        self.argumentsJSON = encodeCanonicalHTTPPrefixCacheJSONObject(arguments)
    }
}

nonisolated struct HTTPPrefixCacheAssistantSignature: Hashable, Sendable {
    let content: String
    let toolCalls: [HTTPPrefixCacheToolCall]
}

nonisolated struct HTTPPrefixCacheMessage: Hashable, Sendable {
    let role: Chat.Message.Role
    let content: String
    let reasoning: String?
    let toolCalls: [HTTPPrefixCacheToolCall]

    init(
        role: Chat.Message.Role,
        content: String,
        reasoning: String? = nil,
        toolCalls: [HTTPPrefixCacheToolCall] = []
    ) {
        self.role = role
        // Normalize assistant content by trimming surrounding whitespace.
        // OpenCode (and likely other OpenAI clients) strips trailing whitespace
        // from assistant messages when echoing them back as history — this
        // applies to BOTH whitespace-only content (turning into "") AND
        // non-empty content with trailing whitespace (e.g. "Now let me read
        // the source files…\n\n\n\n" → "Now let me read the source files…").
        // Without this, the stored cache entry has 4 extra trailing whitespace
        // chars vs the echoed-back version and `isPrefix` fails. Only assistant
        // role — user/tool content is real data and must be preserved verbatim.
        if role == .assistant {
            self.content = content.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            self.content = content
        }
        let trimmedReasoning = reasoning?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.reasoning = (trimmedReasoning?.isEmpty ?? true) ? nil : trimmedReasoning
        self.toolCalls = role == .assistant ? toolCalls : []
    }

    static func assistant(
        content: String,
        reasoning: String? = nil,
        toolCalls: [HTTPPrefixCacheToolCall] = []
    ) -> HTTPPrefixCacheMessage {
        HTTPPrefixCacheMessage(
            role: .assistant,
            content: content,
            reasoning: reasoning,
            toolCalls: toolCalls
        )
    }

    var promptContent: String {
        guard role == .assistant else { return content }
        return reconstructAssistantPromptContent(
            content,
            reasoning: reasoning,
            toolCalls: toolCalls
        )
    }

    var assistantSignature: HTTPPrefixCacheAssistantSignature? {
        guard role == .assistant else { return nil }
        return HTTPPrefixCacheAssistantSignature(content: content, toolCalls: toolCalls)
    }
}

nonisolated struct HTTPPrefixCacheConversation: Hashable, Sendable {
    static let emptyToolDefinitionsDigest = httpPrefixCacheDigest(for: Data("[]".utf8))
    static let defaultTemplateContextDigest = httpPrefixCacheDigest(for: Data("{}".utf8))

    let systemPrompt: String?
    let messages: [HTTPPrefixCacheMessage]
    let toolDefinitionsDigest: String
    let templateContextDigest: String

    init(
        systemPrompt: String?,
        messages: [HTTPPrefixCacheMessage],
        toolDefinitionsDigest: String = Self.emptyToolDefinitionsDigest,
        templateContextDigest: String = Self.defaultTemplateContextDigest
    ) {
        let trimmedSystem = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.systemPrompt = trimmedSystem?.isEmpty == true ? nil : trimmedSystem
        self.messages = messages
        self.toolDefinitionsDigest = toolDefinitionsDigest
        self.templateContextDigest = templateContextDigest
    }

    var lastMessage: HTTPPrefixCacheMessage? {
        messages.last
    }

    var prefixWithoutLastMessage: HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: Array(messages.dropLast()),
            toolDefinitionsDigest: toolDefinitionsDigest,
            templateContextDigest: templateContextDigest
        )
    }

    func appendingAssistant(_ assistantMessage: HTTPPrefixCacheMessage) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: messages + [assistantMessage],
            toolDefinitionsDigest: toolDefinitionsDigest,
            templateContextDigest: templateContextDigest
        )
    }

    func isPrefix(of other: HTTPPrefixCacheConversation) -> Bool {
        guard systemPrompt == other.systemPrompt else { return false }
        guard toolDefinitionsDigest == other.toolDefinitionsDigest else { return false }
        guard templateContextDigest == other.templateContextDigest else { return false }
        guard messages.count <= other.messages.count else { return false }
        return zip(messages, other.messages).allSatisfy(==)
    }

    /// Returns the first divergence preventing `self` from being a prefix of `request`,
    /// or `nil` if `self` IS a prefix. Used for diagnostic logging on cache miss.
    func diagnosePrefixMismatch(against request: HTTPPrefixCacheConversation) -> HTTPPrefixCacheMismatchReport? {
        if systemPrompt != request.systemPrompt {
            return .systemPromptMismatch(
                storedLength: systemPrompt?.count ?? 0,
                requestLength: request.systemPrompt?.count ?? 0,
                storedHash: shortDigest(of: systemPrompt),
                requestHash: shortDigest(of: request.systemPrompt)
            )
        }
        if toolDefinitionsDigest != request.toolDefinitionsDigest {
            return .toolDefinitionsDigestMismatch(
                storedDigest: shortPrefix(toolDefinitionsDigest),
                requestDigest: shortPrefix(request.toolDefinitionsDigest)
            )
        }
        if templateContextDigest != request.templateContextDigest {
            return .templateContextDigestMismatch(
                storedDigest: shortPrefix(templateContextDigest),
                requestDigest: shortPrefix(request.templateContextDigest)
            )
        }
        if messages.count > request.messages.count {
            return .messageCountTooLarge(
                storedCount: messages.count,
                requestCount: request.messages.count
            )
        }
        for (index, pair) in zip(messages, request.messages).enumerated() {
            let (stored, requested) = pair
            if stored == requested { continue }
            if stored.role != requested.role {
                return .messageRoleMismatch(
                    index: index,
                    storedRole: "\(stored.role)",
                    requestRole: "\(requested.role)"
                )
            }
            if stored.content != requested.content {
                return .messageFieldMismatch(
                    index: index,
                    role: "\(stored.role)",
                    field: "content",
                    storedLength: stored.content.count,
                    requestLength: requested.content.count,
                    storedHash: shortDigest(of: stored.content),
                    requestHash: shortDigest(of: requested.content),
                    storedPreview: escapedPreview(of: stored.content),
                    requestPreview: escapedPreview(of: requested.content)
                )
            }
            if stored.reasoning != requested.reasoning {
                return .messageFieldMismatch(
                    index: index,
                    role: "\(stored.role)",
                    field: "reasoning",
                    storedLength: stored.reasoning?.count ?? 0,
                    requestLength: requested.reasoning?.count ?? 0,
                    storedHash: shortDigest(of: stored.reasoning),
                    requestHash: shortDigest(of: requested.reasoning),
                    storedPreview: escapedPreview(of: stored.reasoning),
                    requestPreview: escapedPreview(of: requested.reasoning)
                )
            }
            if stored.toolCalls != requested.toolCalls {
                if stored.toolCalls.count != requested.toolCalls.count {
                    return .messageToolCallCountMismatch(
                        index: index,
                        storedCount: stored.toolCalls.count,
                        requestCount: requested.toolCalls.count
                    )
                }
                for (callIndex, callPair) in zip(stored.toolCalls, requested.toolCalls).enumerated() {
                    let (storedCall, requestCall) = callPair
                    if storedCall == requestCall { continue }
                    if storedCall.name != requestCall.name {
                        return .toolCallNameMismatch(
                            messageIndex: index,
                            toolCallIndex: callIndex,
                            storedName: storedCall.name,
                            requestName: requestCall.name
                        )
                    }
                    return .toolCallArgumentsMismatch(
                        messageIndex: index,
                        toolCallIndex: callIndex,
                        toolName: storedCall.name,
                        storedLength: storedCall.argumentsJSON.count,
                        requestLength: requestCall.argumentsJSON.count,
                        storedHash: shortDigest(of: storedCall.argumentsJSON),
                        requestHash: shortDigest(of: requestCall.argumentsJSON),
                        storedPreview: escapedPreview(of: storedCall.argumentsJSON),
                        requestPreview: escapedPreview(of: requestCall.argumentsJSON)
                    )
                }
            }
            // All known fields equal but Hashable says different — fall through to a generic report.
            return .messageFieldMismatch(
                index: index,
                role: "\(stored.role)",
                field: "unknown",
                storedLength: 0,
                requestLength: 0,
                storedHash: "?",
                requestHash: "?",
                storedPreview: "",
                requestPreview: ""
            )
        }
        return nil
    }

    var historyMessages: [Chat.Message] {
        var history: [Chat.Message] = []
        if let systemPrompt {
            history.append(.system(systemPrompt))
        }

        for message in messages {
            switch message.role {
            case .system:
                history.append(.system(message.content))
            case .user:
                history.append(.user(message.content))
            case .assistant:
                history.append(.assistant(message.promptContent))
            case .tool:
                history.append(.tool(message.content))
            }
        }

        return history
    }
}

nonisolated struct HTTPPrefixCacheMatch: @unchecked Sendable {
    let conversation: HTTPPrefixCacheConversation
    let cachedTokenCount: Int
    let cache: [KVCache]
}

nonisolated enum HTTPPrefixCacheLookupReason: String, Sendable, Equatable, CustomStringConvertible {
    case hit
    case noEntriesForKey
    case completedDescendantReplay
    case noPrefixMatch

    var description: String { rawValue }
}

/// First-divergence report explaining why a stored entry failed to prefix-match a request.
/// Populated only on `noPrefixMatch` (and only when at least one keyed entry exists).
nonisolated enum HTTPPrefixCacheMismatchReport: Sendable, Equatable, CustomStringConvertible {
    case systemPromptMismatch(storedLength: Int, requestLength: Int, storedHash: String, requestHash: String)
    case toolDefinitionsDigestMismatch(storedDigest: String, requestDigest: String)
    case templateContextDigestMismatch(storedDigest: String, requestDigest: String)
    case messageCountTooLarge(storedCount: Int, requestCount: Int)
    case messageRoleMismatch(index: Int, storedRole: String, requestRole: String)
    case messageFieldMismatch(
        index: Int,
        role: String,
        field: String,
        storedLength: Int,
        requestLength: Int,
        storedHash: String,
        requestHash: String,
        storedPreview: String,
        requestPreview: String
    )
    case messageToolCallCountMismatch(index: Int, storedCount: Int, requestCount: Int)
    case toolCallNameMismatch(messageIndex: Int, toolCallIndex: Int, storedName: String, requestName: String)
    case toolCallArgumentsMismatch(
        messageIndex: Int,
        toolCallIndex: Int,
        toolName: String,
        storedLength: Int,
        requestLength: Int,
        storedHash: String,
        requestHash: String,
        storedPreview: String,
        requestPreview: String
    )

    var description: String {
        switch self {
        case .systemPromptMismatch(let sl, let rl, let sh, let rh):
            return "systemPrompt(storedLen=\(sl) reqLen=\(rl) storedHash=\(sh) reqHash=\(rh))"
        case .toolDefinitionsDigestMismatch(let sd, let rd):
            return "toolDefinitionsDigest(stored=\(sd) req=\(rd))"
        case .templateContextDigestMismatch(let sd, let rd):
            return "templateContextDigest(stored=\(sd) req=\(rd))"
        case .messageCountTooLarge(let sc, let rc):
            return "messageCountTooLarge(storedCount=\(sc) requestCount=\(rc))"
        case .messageRoleMismatch(let i, let sr, let rr):
            return "message[\(i)].role(stored=\(sr) req=\(rr))"
        case .messageFieldMismatch(let i, let role, let field, let sl, let rl, let sh, let rh, let sp, let rp):
            return "message[\(i)](\(role)).\(field)(storedLen=\(sl) reqLen=\(rl) storedHash=\(sh) reqHash=\(rh) storedPreview=\"\(sp)\" reqPreview=\"\(rp)\")"
        case .messageToolCallCountMismatch(let i, let sc, let rc):
            return "message[\(i)].toolCalls.count(stored=\(sc) req=\(rc))"
        case .toolCallNameMismatch(let mi, let ti, let sn, let rn):
            return "message[\(mi)].toolCalls[\(ti)].name(stored=\(sn) req=\(rn))"
        case .toolCallArgumentsMismatch(let mi, let ti, let toolName, let sl, let rl, let sh, let rh, let sp, let rp):
            return "message[\(mi)].toolCalls[\(ti)](\(toolName)).arguments(storedLen=\(sl) reqLen=\(rl) storedHash=\(sh) reqHash=\(rh) storedPreview=\"\(sp)\" reqPreview=\"\(rp)\")"
        }
    }
}

/// First 8 hex chars of SHA-256 over the UTF-8 bytes of `value`. Returns "nil" for nil input.
nonisolated func shortDigest(of value: String?) -> String {
    guard let value else { return "nil" }
    let full = httpPrefixCacheDigest(for: Data(value.utf8))
    return String(full.prefix(8))
}

/// First 8 chars of an existing hex digest, used so log lines stay readable.
nonisolated func shortPrefix(_ digest: String) -> String {
    String(digest.prefix(8))
}

/// Returns an escaped, length-bounded preview of `value` for log diagnostics.
/// Newlines/tabs become `\n`/`\t`, and the result is truncated to 60 characters
/// with a `…` suffix when the original was longer.
nonisolated func escapedPreview(of value: String?) -> String {
    guard let value else { return "nil" }
    var escaped = ""
    escaped.reserveCapacity(value.count)
    for ch in value.prefix(60) {
        switch ch {
        case "\n": escaped.append("\\n")
        case "\r": escaped.append("\\r")
        case "\t": escaped.append("\\t")
        case "\"": escaped.append("\\\"")
        case "\\": escaped.append("\\\\")
        default: escaped.append(ch)
        }
    }
    if value.count > 60 {
        escaped.append("…")
    }
    return escaped
}

nonisolated struct HTTPPrefixCacheLookup: @unchecked Sendable {
    let reason: HTTPPrefixCacheLookupReason
    let keyedEntryCount: Int
    let match: HTTPPrefixCacheMatch?
    let mismatchReport: HTTPPrefixCacheMismatchReport?
}

actor HTTPPrefixCacheSpikeStore {
    private struct Entry {
        let key: HTTPPrefixCacheKey
        let conversation: HTTPPrefixCacheConversation
        let cachedTokenCount: Int
        let cache: [KVCache]
    }

    private let capacity: Int
    private var entries: [Entry] = []

    init(capacity: Int = 8) {
        self.capacity = max(1, capacity)
    }

    func clear() {
        entries.removeAll()
    }

    /// Snapshot of stored entries for memory diagnostics.
    /// Returns total entry count and the cumulative `cachedTokenCount` (sum across
    /// all entries) which is a rough proxy for KV-cache memory occupancy.
    func snapshot() -> (entryCount: Int, totalCachedTokens: Int, capacity: Int) {
        let totalTokens = entries.reduce(0) { $0 + $1.cachedTokenCount }
        return (entries.count, totalTokens, capacity)
    }

    func lookup(
        conversation request: HTTPPrefixCacheConversation,
        key: HTTPPrefixCacheKey
    ) -> HTTPPrefixCacheLookup {
        let keyedEntries = entries.filter { $0.key == key }
        guard !keyedEntries.isEmpty else {
            return HTTPPrefixCacheLookup(
                reason: .noEntriesForKey,
                keyedEntryCount: 0,
                match: nil,
                mismatchReport: nil
            )
        }

        let hasCompletedDescendant = keyedEntries.contains { entry in
            guard request.messages.count < entry.conversation.messages.count else { return false }
            return request.isPrefix(of: entry.conversation)
        }
        if hasCompletedDescendant {
            return HTTPPrefixCacheLookup(
                reason: .completedDescendantReplay,
                keyedEntryCount: keyedEntries.count,
                match: nil,
                mismatchReport: nil
            )
        }

        var bestIndex: Int?
        var bestLength = -1

        for (index, entry) in entries.enumerated() {
            guard entry.key == key else { continue }
            guard entry.conversation.messages.count < request.messages.count else { continue }
            guard entry.conversation.isPrefix(of: request) else { continue }

            if entry.conversation.messages.count > bestLength {
                bestIndex = index
                bestLength = entry.conversation.messages.count
            }
        }

        guard let bestIndex else {
            // Compute the mismatch report against the most-promising candidate so the
            // caller can log which field actually diverged. Pick the entry with the
            // largest message count that is still <= request count (the candidate
            // most likely to have been the intended prefix).
            let candidate = keyedEntries
                .filter { $0.conversation.messages.count <= request.messages.count }
                .max(by: { $0.conversation.messages.count < $1.conversation.messages.count })
                ?? keyedEntries.first!
            return HTTPPrefixCacheLookup(
                reason: .noPrefixMatch,
                keyedEntryCount: keyedEntries.count,
                match: nil,
                mismatchReport: candidate.conversation.diagnosePrefixMismatch(against: request)
            )
        }
        let entry = entries.remove(at: bestIndex)
        entries.append(entry)

        // If a strictly longer entry exists for this key but didn't match the request,
        // diagnose why so we can see whether the cache is "stuck" matching short prefixes.
        let hitLength = entry.conversation.messages.count
        let longerCandidate = keyedEntries.filter {
            $0.conversation.messages.count > hitLength
                && $0.conversation.messages.count <= request.messages.count
        }.max(by: { $0.conversation.messages.count < $1.conversation.messages.count })
        let longerMismatch = longerCandidate?.conversation.diagnosePrefixMismatch(against: request)

        return HTTPPrefixCacheLookup(
            reason: .hit,
            keyedEntryCount: keyedEntries.count,
            match: HTTPPrefixCacheMatch(
                conversation: entry.conversation,
                cachedTokenCount: entry.cachedTokenCount,
                cache: entry.cache.map { $0.copy() }
            ),
            mismatchReport: longerMismatch
        )
    }

    func match(
        conversation request: HTTPPrefixCacheConversation,
        key: HTTPPrefixCacheKey
    ) -> HTTPPrefixCacheMatch? {
        lookup(conversation: request, key: key).match
    }

    func store(
        conversation: HTTPPrefixCacheConversation,
        key: HTTPPrefixCacheKey,
        cachedTokenCount: Int,
        cache: consuming [KVCache]
    ) {
        let ownedCache = cache.map { $0.copy() }

        // Keep at most ONE entry per cache key — the latest. Each cache key
        // represents a distinct conversation chain (main agent vs subagent vs
        // title-gen all have different toolDefinitions/templateContext digests
        // → different keys). Without this, a subagent's many tool-loop turns
        // would LRU-evict the main agent's entry, forcing the main agent to
        // re-prefill its entire context (~30 K tokens, ~3 min) when control
        // returns to it. Per-key replacement keeps each chain's latest cached
        // state available regardless of how many turns the other chain runs.
        entries.removeAll { $0.key == key }

        entries.append(Entry(
            key: key,
            conversation: conversation,
            cachedTokenCount: max(0, cachedTokenCount),
            cache: ownedCache
        ))

        // Cap total entries (= number of distinct keys retained). LRU across
        // KEYS now, not across individual turns. When the limit is exceeded,
        // we drop the oldest key wholesale.
        if entries.count > capacity {
            entries.removeFirst(entries.count - capacity)
        }
    }
}

nonisolated func httpPrefixCacheOffsets(_ cache: [KVCache]) -> [Int] {
    cache.map(\.offset)
}

nonisolated func httpPrefixCacheReportedTokenCount(_ cache: [KVCache]) -> Int {
    httpPrefixCacheOffsets(cache).max() ?? 0
}

nonisolated func httpPrefixCacheHasReusableState(_ cache: [KVCache]) -> Bool {
    cache.contains { $0.offset > 0 || !$0.state.isEmpty }
}

nonisolated func reconstructAssistantPromptContent(
    _ content: String,
    reasoning: String?,
    toolCalls: [HTTPPrefixCacheToolCall]
) -> String {
    var promptContent = content

    if let reasoning, !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        let thinkBlock = "<think>\n\(reasoning)\n</think>"
        promptContent = promptContent.isEmpty
            ? thinkBlock
            : thinkBlock + "\n" + promptContent
    }

    guard !toolCalls.isEmpty else { return promptContent }

    var result = promptContent
    for call in toolCalls {
        result += "\n<tool_call>\n<function=\(call.name)>\n"

        if let arguments = ToolArgumentNormalizer.decode(call.argumentsJSON) {
            for key in arguments.keys.sorted() {
                guard let value = arguments[key] else { continue }
                result += "<parameter=\(key)>\n"
                result += formatHTTPPrefixCacheToolCallParameterValue(value)
                result += "\n</parameter>\n"
            }
        }

        result += "</function>\n</tool_call>"
    }

    return result
}

nonisolated func encodeCanonicalHTTPPrefixCacheJSONObject(_ object: [String: JSONValue]) -> String {
    encodeCanonicalHTTPPrefixCacheJSONValue(.object(object))
}

nonisolated func encodeCanonicalHTTPPrefixCacheJSONValue(_ value: JSONValue) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(value),
          let json = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return json
}

nonisolated func canonicalizeHTTPPrefixCacheToolArgumentsJSON(_ argumentsJSON: String) -> String {
    let trimmed = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return "{}" }

    if let data = trimmed.data(using: .utf8) {
        if let decodedObject = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
            return encodeCanonicalHTTPPrefixCacheJSONObject(decodedObject)
        }
        if let decodedValue = try? JSONDecoder().decode(JSONValue.self, from: data) {
            return encodeCanonicalHTTPPrefixCacheJSONValue(decodedValue)
        }
    }

    return trimmed
}

nonisolated func httpPrefixCacheDigest(for data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private nonisolated func formatHTTPPrefixCacheToolCallParameterValue(_ value: JSONValue) -> String {
    switch value {
    case .string(let string):
        return string
    case .int(let int):
        return String(int)
    case .double(let double):
        return String(double)
    case .bool(let bool):
        return bool ? "True" : "False"
    case .null:
        return "None"
    case .array, .object:
        return encodeCanonicalHTTPPrefixCacheJSONValue(value)
    }
}
