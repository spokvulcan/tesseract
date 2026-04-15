import CryptoKit
import Foundation
import MLXLMCommon

nonisolated struct HTTPServerGenerationStart: Sendable {
    let stream: AsyncThrowingStream<AgentGeneration, Error>
    let cachedTokenCount: Int
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

    var promptToolCall: [String: any Sendable] {
        let function: [String: any Sendable] = [
            "name": name,
            "arguments": decodedArguments.mapValues(httpPrefixCachePromptValue),
        ]
        return [
            "type": "function",
            "function": function,
        ]
    }

    private var decodedArguments: [String: JSONValue] {
        ToolArgumentNormalizer.decode(argumentsJSON) ?? [:]
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

    var promptMessage: [String: any Sendable] {
        var message: [String: any Sendable] = [
            "role": role.rawValue,
            "content": content,
        ]

        guard role == .assistant else { return message }

        if let reasoning {
            message["reasoning_content"] = reasoning
        }
        if !toolCalls.isEmpty {
            message["tool_calls"] = toolCalls.map(\.promptToolCall)
        }

        return message
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

    var promptMessages: [[String: any Sendable]] {
        var promptMessages: [[String: any Sendable]] = []
        if let systemPrompt {
            promptMessages.append(httpPrefixCachePromptMessage(
                role: .system,
                content: systemPrompt
            ))
        }

        promptMessages.append(contentsOf: messages.map(\.promptMessage))
        return promptMessages
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

private nonisolated func httpPrefixCachePromptMessage(
    role: Chat.Message.Role,
    content: String
) -> [String: any Sendable] {
    [
        "role": role.rawValue,
        "content": content,
    ]
}

private nonisolated func httpPrefixCachePromptValue(_ value: JSONValue) -> any Sendable {
    switch value {
    case .null:
        return "None"
    case .bool(let bool):
        return bool
    case .int(let int):
        return int
    case .double(let double):
        return double
    case .string(let string):
        return string
    case .array(let array):
        return array.map(httpPrefixCachePromptValue)
    case .object(let object):
        return object.mapValues(httpPrefixCachePromptValue)
    }
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
