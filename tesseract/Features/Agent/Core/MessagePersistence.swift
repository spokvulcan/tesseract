import Foundation

// MARK: - PersistableMessage

/// A message type that can be persisted as tagged JSON.
/// Each conformer declares a unique `persistenceTag` used as the type discriminator.
protocol PersistableMessage: AgentMessageProtocol, Codable {
    nonisolated static var persistenceTag: String { get }
}

// MARK: - Core Type Conformances

extension UserMessage: PersistableMessage {
    static let persistenceTag = "user"
}

extension AssistantMessage: PersistableMessage {
    static let persistenceTag = "assistant"
}

extension ToolResultMessage: PersistableMessage {
    static let persistenceTag = "tool_result"
}

// MARK: - TaggedMessage

/// Codable envelope for heterogeneous message persistence.
/// Stores the type discriminator alongside the serialized payload.
nonisolated struct TaggedMessage: Codable, Sendable {
    let type: String
    let payload: [String: AnyCodableValue]
}

// MARK: - OpaqueMessage

/// Fallback for messages with an unrecognized persistence tag.
/// Preserves the raw payload so re-encoding produces identical JSON.
nonisolated struct OpaqueMessage: AgentMessageProtocol, Sendable {
    let tag: String
    let rawPayload: [String: AnyCodableValue]

    func toLLMMessage() -> LLMMessage? { nil }
}

// MARK: - MessageCodecRegistry

/// Central registry for encoding/decoding protocol-typed messages to/from tagged JSON.
///
/// Each `PersistableMessage` type is registered with its tag. On encode, the registry
/// finds the matching tag and encodes the concrete type. On decode, it looks up the tag
/// and decodes to the concrete type. Unknown tags produce `OpaqueMessage`.
actor MessageCodecRegistry {
    static let shared = MessageCodecRegistry()

    /// Type-erased codec. Stored only within the actor — no Sendable requirement.
    private struct Codec {
        let tag: String
        let encode: (any AgentMessageProtocol) throws -> TaggedMessage
        let decode: (TaggedMessage) throws -> any AgentMessageProtocol
    }

    /// Tag → codec.
    private var codecsByTag: [String: Codec] = [:]

    /// Concrete type ObjectIdentifier → tag (for encoding lookup).
    private var tagsByType: [ObjectIdentifier: String] = [:]

    // MARK: - Registration

    func register<M: PersistableMessage>(_ type: M.Type) {
        let tag = M.persistenceTag
        let codec = Codec(
            tag: tag,
            encode: { message in
                guard let concrete = message as? M else {
                    throw MessageCodecError.typeMismatch(
                        expected: String(describing: M.self),
                        got: String(describing: Swift.type(of: message))
                    )
                }
                let data = try JSONEncoder().encode(concrete)
                let any = try JSONSerialization.jsonObject(with: data)
                guard let dict = any as? [String: Any] else {
                    throw MessageCodecError.encodingFailed(tag)
                }
                let payload = dict.mapValues { AnyCodableValue($0) }
                return TaggedMessage(type: tag, payload: payload)
            },
            decode: { tagged in
                let payloadAny = tagged.payload.mapValues { $0.toAny() }
                let data = try JSONSerialization.data(withJSONObject: payloadAny)
                return try JSONDecoder().decode(M.self, from: data)
            }
        )
        codecsByTag[tag] = codec
        tagsByType[ObjectIdentifier(M.self)] = tag
    }

    // MARK: - Encode

    func encode(_ message: any AgentMessageProtocol) throws -> TaggedMessage {
        // OpaqueMessage round-trips directly
        if let opaque = message as? OpaqueMessage {
            return TaggedMessage(type: opaque.tag, payload: opaque.rawPayload)
        }

        let typeId = ObjectIdentifier(type(of: message))
        guard let tag = tagsByType[typeId],
              let codec = codecsByTag[tag]
        else {
            throw MessageCodecError.unregisteredType(String(describing: type(of: message)))
        }
        return try codec.encode(message)
    }

    // MARK: - Decode

    func decode(_ tagged: TaggedMessage) throws -> any AgentMessageProtocol {
        guard let codec = codecsByTag[tagged.type] else {
            return OpaqueMessage(tag: tagged.type, rawPayload: tagged.payload)
        }
        return try codec.decode(tagged)
    }

    // MARK: - Batch

    func encodeAll(_ messages: [any AgentMessageProtocol]) throws -> [TaggedMessage] {
        try messages.map { try encode($0) }
    }

    func decodeAll(_ tagged: [TaggedMessage]) throws -> [any AgentMessageProtocol] {
        try tagged.map { try decode($0) }
    }
}

// MARK: - Errors

nonisolated enum MessageCodecError: LocalizedError {
    case typeMismatch(expected: String, got: String)
    case encodingFailed(String)
    case unregisteredType(String)

    var errorDescription: String? {
        switch self {
        case .typeMismatch(let expected, let got):
            "Message codec type mismatch: expected \(expected), got \(got)"
        case .encodingFailed(let tag):
            "Failed to encode message with tag '\(tag)'"
        case .unregisteredType(let typeName):
            "No codec registered for type '\(typeName)'"
        }
    }
}

// MARK: - AnyCodableValue → Any

extension AnyCodableValue {
    /// Convert to untyped `Any` for `JSONSerialization.data(withJSONObject:)`.
    nonisolated func toAny() -> Any {
        switch self {
        case .null: return NSNull()
        case .bool(let v): return v
        case .int(let v): return v
        case .double(let v): return v
        case .string(let v): return v
        case .array(let v): return v.map { $0.toAny() }
        case .object(let v): return v.mapValues { $0.toAny() }
        }
    }
}

// MARK: - SyncMessageCodec

/// Synchronous, stateless message codec for use by `@MainActor` code that cannot `await`.
/// Encodes/decodes using the same tagged-JSON format as `MessageCodecRegistry`.
/// Supports the four core message types + OpaqueMessage round-tripping.
nonisolated enum SyncMessageCodec {

    static func encode(_ message: any AgentMessageProtocol) throws -> TaggedMessage {
        if let opaque = message as? OpaqueMessage {
            return TaggedMessage(type: opaque.tag, payload: opaque.rawPayload)
        }
        if let msg = message as? UserMessage { return try encodeTyped(msg, tag: UserMessage.persistenceTag) }
        if let msg = message as? AssistantMessage { return try encodeTyped(msg, tag: AssistantMessage.persistenceTag) }
        if let msg = message as? ToolResultMessage { return try encodeTyped(msg, tag: ToolResultMessage.persistenceTag) }
        if let msg = message as? CompactionSummaryMessage { return try encodeTyped(msg, tag: CompactionSummaryMessage.persistenceTag) }

        // CoreMessage wrapper — unwrap and encode the inner message
        if let core = message as? CoreMessage {
            switch core {
            case .user(let u): return try encodeTyped(u, tag: UserMessage.persistenceTag)
            case .assistant(let a): return try encodeTyped(a, tag: AssistantMessage.persistenceTag)
            case .toolResult(let t): return try encodeTyped(t, tag: ToolResultMessage.persistenceTag)
            }
        }

        // AgentChatMessage (legacy UI type) — convert to appropriate core type
        if let chat = message as? AgentChatMessage {
            switch chat.role {
            case .user: return try encodeTyped(UserMessage.create(chat.content), tag: UserMessage.persistenceTag)
            case .assistant: return try encodeTyped(
                AssistantMessage(content: chat.content, thinking: chat.thinking),
                tag: AssistantMessage.persistenceTag)
            case .tool: return try encodeTyped(
                ToolResultMessage(toolCallId: "", toolName: "", content: [.text(chat.content)]),
                tag: ToolResultMessage.persistenceTag)
            case .system: return try encodeTyped(
                CompactionSummaryMessage(summary: chat.content, tokensBefore: 0),
                tag: CompactionSummaryMessage.persistenceTag)
            }
        }

        throw MessageCodecError.unregisteredType(String(describing: type(of: message)))
    }

    static func decode(_ tagged: TaggedMessage) throws -> any AgentMessageProtocol & Sendable {
        switch tagged.type {
        case UserMessage.persistenceTag:
            return try decodeTyped(UserMessage.self, from: tagged)
        case AssistantMessage.persistenceTag:
            return try decodeTyped(AssistantMessage.self, from: tagged)
        case ToolResultMessage.persistenceTag:
            return try decodeTyped(ToolResultMessage.self, from: tagged)
        case CompactionSummaryMessage.persistenceTag:
            return try decodeTyped(CompactionSummaryMessage.self, from: tagged)
        default:
            return OpaqueMessage(tag: tagged.type, rawPayload: tagged.payload)
        }
    }

    static func encodeAll(_ messages: [any AgentMessageProtocol]) throws -> [TaggedMessage] {
        try messages.map { try encode($0) }
    }

    static func decodeAll(_ tagged: [TaggedMessage]) throws -> [any AgentMessageProtocol & Sendable] {
        try tagged.map { try decode($0) }
    }

    // MARK: - Private Helpers

    private static func encodeTyped<M: Codable>(_ value: M, tag: String) throws -> TaggedMessage {
        let data = try JSONEncoder().encode(value)
        let any = try JSONSerialization.jsonObject(with: data)
        guard let dict = any as? [String: Any] else {
            throw MessageCodecError.encodingFailed(tag)
        }
        let payload = dict.mapValues { AnyCodableValue($0) }
        return TaggedMessage(type: tag, payload: payload)
    }

    private static func decodeTyped<M: Codable>(_ type: M.Type, from tagged: TaggedMessage) throws -> M {
        let payloadAny = tagged.payload.mapValues { $0.toAny() }
        let data = try JSONSerialization.data(withJSONObject: payloadAny)
        return try JSONDecoder().decode(M.self, from: data)
    }
}

// MARK: - Registration at Startup

/// Register the three core message codecs. Call once during app initialization.
func registerCoreMessageCodecs() async {
    let registry = MessageCodecRegistry.shared
    await registry.register(UserMessage.self)
    await registry.register(AssistantMessage.self)
    await registry.register(ToolResultMessage.self)
    await registry.register(CompactionSummaryMessage.self)
}
