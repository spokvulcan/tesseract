import CoreImage
import Foundation

// MARK: - ImageAttachment

/// An image attached to a user message.
///
/// Equatable/Hashable compare by `id` only — avoids byte-by-byte Data comparison
/// during SwiftUI diffing and row cache equality checks.
nonisolated struct ImageAttachment: Sendable, Codable, Equatable, Hashable, Identifiable {
    let id: UUID
    let data: Data
    let mimeType: String
    let filename: String?

    init(id: UUID = UUID(), data: Data, mimeType: String, filename: String? = nil) {
        self.id = id
        self.data = data
        self.mimeType = mimeType
        self.filename = filename
    }

    var ciImage: CIImage? {
        CIImage(data: data)
    }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

// MARK: - ToolCallInfo

/// Parsed tool call extracted from an assistant response.
nonisolated struct ToolCallInfo: Sendable, Codable, Hashable, Identifiable {
    let id: String
    let name: String
    let argumentsJSON: String
}

// MARK: - LLMMessage

/// Low-level message representation for the LLM context window.
/// Transient — not persisted. Built from higher-level message types via `toLLMMessage()`.
nonisolated enum LLMMessage: Sendable, Equatable {
    case system(content: String)
    case user(content: String, images: [ImageAttachment] = [])
    case assistant(content: String, reasoning: String? = nil, toolCalls: [ToolCallInfo]?)
    /// `images` carries tool-returned images (browser screenshots) into the
    /// context window — see `toLLMCommonMessages` for how they reach the
    /// model or degrade to a text note when the session is text-only.
    case toolResult(toolCallId: String, content: String, images: [ImageAttachment])
}

extension LLMMessage {
    /// Text-only tool result — the overwhelmingly common shape; keeps the
    /// many image-less construction sites at two arguments.
    nonisolated static func toolResult(toolCallId: String, content: String) -> LLMMessage {
        .toolResult(toolCallId: toolCallId, content: content, images: [])
    }
}

// MARK: - AgentMessageProtocol

/// Common interface for all messages flowing through the agent pipeline.
protocol AgentMessageProtocol: Sendable {
    /// Convert to LLM context representation. Returns nil for messages
    /// that should not appear in the LLM context (e.g. UI-only messages).
    nonisolated func toLLMMessage() -> LLMMessage?
}

// MARK: - UserMessage

/// A message from the user (text input or transcribed voice), optionally with image attachments.
nonisolated struct UserMessage: AgentMessageProtocol, Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let content: String
    let images: [ImageAttachment]
    let timestamp: Date

    /// Context the app rides along with this message into the model, but which
    /// the *user* never wrote and must never see in their own bubble — today,
    /// the memory system's `<memory>` block (ADR-0035 §5).
    ///
    /// Stored on the message rather than recomputed at load, for two reasons.
    /// The context a turn was actually answered with is the only honest record
    /// of it. And the radix prefix cache requires that reopening a conversation
    /// reproduce byte-identical context — a fresh retrieval at load time would
    /// silently rewrite history and miss the cache on every turn of the thread.
    let injectedContext: String?

    /// Which turn class this message opened (ADR-0046). Loop turns fold into
    /// the one Mission Control conversation, so the per-turn origin the
    /// conversation tag used to carry now rides the turn's opening message —
    /// nil for everything the owner typed. Metadata only: never rendered into
    /// the LLM context.
    let turnOrigin: TurnOrigin?

    init(
        id: UUID = UUID(), content: String, images: [ImageAttachment] = [],
        timestamp: Date = Date(), injectedContext: String? = nil,
        turnOrigin: TurnOrigin? = nil
    ) {
        self.id = id
        self.content = content
        self.images = images
        self.timestamp = timestamp
        self.injectedContext = injectedContext
        self.turnOrigin = turnOrigin
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        content = try container.decode(String.self, forKey: .content)
        images = try container.decodeIfPresent([ImageAttachment].self, forKey: .images) ?? []
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        injectedContext = try container.decodeIfPresent(String.self, forKey: .injectedContext)
        // Via the raw string: an unknown future tag reads as untagged instead
        // of failing the whole conversation file's decode (the summary's rule).
        turnOrigin = (try container.decodeIfPresent(String.self, forKey: .turnOrigin))
            .flatMap(TurnOrigin.init(rawValue:))
    }

    private enum CodingKeys: String, CodingKey {
        case id, content, images, timestamp, injectedContext, turnOrigin
    }

    /// The wrapper goes *before* the user's words, mirroring `<skill>`: what I
    /// know, then what I was asked.
    func toLLMMessage() -> LLMMessage? {
        guard let injectedContext, !injectedContext.isEmpty else {
            return .user(content: content, images: images)
        }
        return .user(content: "\(injectedContext)\n\n\(content)", images: images)
    }
}

// MARK: - AssistantMessage

/// A response from the LLM — ordered **Content Part**s plus the pi-ai message
/// envelope (api / provider / model, usage, stop reason), mirrored verbatim
/// from `packages/ai/src/types.ts` (ADR-0024). SwiftUI row identity derives
/// from (`id`, part index).
nonisolated struct AssistantMessage: AgentMessageProtocol, Codable, Equatable, Identifiable,
    Sendable
{
    let id: UUID
    var content: [ContentPart]
    var api: String
    var provider: String
    var model: String
    var usage: Usage
    var stopReason: StopReason
    var errorMessage: String?
    let timestamp: Date

    init(
        id: UUID = UUID(),
        content: [ContentPart],
        api: String = "mlx",
        provider: String = "tesseract",
        model: String = "",
        usage: Usage = Usage(),
        stopReason: StopReason = .stop,
        errorMessage: String? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.content = content
        self.api = api
        self.provider = provider
        self.model = model
        self.usage = usage
        self.stopReason = stopReason
        self.errorMessage = errorMessage
        self.timestamp = timestamp
    }

    /// Flat-string convenience — fixtures, compaction summaries, and the
    /// LLM-facing layers below the parts model construct through this shape.
    /// Part order: thinking, then text, then tool calls (generation order).
    init(
        id: UUID = UUID(),
        content: String,
        thinking: String? = nil,
        toolCalls: [ToolCallInfo] = [],
        stopReason: StopReason = .stop,
        timestamp: Date = Date()
    ) {
        var parts: [ContentPart] = []
        if let thinking, !thinking.isEmpty {
            parts.append(.thinking(ThinkingPart(thinking: thinking)))
        }
        if !content.isEmpty {
            parts.append(.text(TextPart(text: content)))
        }
        for call in toolCalls {
            parts.append(
                .toolCall(
                    ToolCallPart(id: call.id, name: call.name, argumentsJSON: call.argumentsJSON)))
        }
        self.init(id: id, content: parts, stopReason: stopReason, timestamp: timestamp)
    }

    // MARK: Flat projections (the LLM context, compaction, and logging layers
    // below the parts model keep reading the turn as flat strings)

    /// All visible text, in part order.
    var text: String {
        content.compactMap { part in
            if case .text(let t) = part { return t.text }
            return nil
        }.joined()
    }

    /// All thinking, in part order. `nil` when no thinking part exists —
    /// preserves the "never opened a `<think>` block" distinction.
    var thinking: String? {
        let parts = content.compactMap { part -> String? in
            if case .thinking(let t) = part { return t.thinking }
            return nil
        }
        return parts.isEmpty ? nil : parts.joined()
    }

    /// Tool calls in part order, bridged to the execution layer's identity type.
    var toolCalls: [ToolCallInfo] {
        content.compactMap { part in
            if case .toolCall(let call) = part {
                return ToolCallInfo(
                    id: call.id, name: call.name, argumentsJSON: call.argumentsJSON)
            }
            return nil
        }
    }

    func toLLMMessage() -> LLMMessage? {
        let calls = toolCalls
        return .assistant(
            content: text,
            reasoning: thinking,
            toolCalls: calls.isEmpty ? nil : calls
        )
    }

    /// Carries something worth committing: any non-empty part. Empty turns
    /// (model errors before any tokens, cancel paths) are dropped rather than
    /// polluting history with blank assistant messages. The single definition
    /// both `runLoop` (on persist) and the Chat Session fold commit against.
    var hasContent: Bool {
        content.contains { part in
            switch part {
            case .text(let t): return !t.text.isEmpty
            case .thinking(let t): return !t.thinking.isEmpty
            case .toolCall: return true
            }
        }
    }
}

// MARK: - ToolResultMessage

/// The result of executing a tool call.
nonisolated struct ToolResultMessage: AgentMessageProtocol, Codable, Equatable, Identifiable,
    Sendable
{
    let id: UUID
    let toolCallId: String
    let toolName: String
    let content: [ContentBlock]
    let isError: Bool
    /// Typed execution details (PRD #200) — the Tool Panels' data source.
    /// `nil` for legacy conversations, detail-less tools, and any persisted
    /// shape this build can't decode.
    let details: ToolResultDetails?
    let timestamp: Date

    init(
        id: UUID = UUID(),
        toolCallId: String,
        toolName: String,
        content: [ContentBlock],
        isError: Bool = false,
        details: ToolResultDetails? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.content = content
        self.isError = isError
        self.details = details
        self.timestamp = timestamp
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        toolCallId = try container.decode(String.self, forKey: .toolCallId)
        toolName = try container.decode(String.self, forKey: .toolName)
        content = try container.decode([ContentBlock].self, forKey: .content)
        isError = try container.decode(Bool.self, forKey: .isError)
        // Lenient on purpose: an unknown details shape (from a newer build)
        // costs only the details, never the message.
        details = try? container.decodeIfPresent(ToolResultDetails.self, forKey: .details)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
    }

    private enum CodingKeys: String, CodingKey {
        case id, toolCallId, toolName, content, isError, details, timestamp
    }

    func toLLMMessage() -> LLMMessage? {
        let textContent = content.compactMap { block -> String? in
            if case .text(let text) = block { return text }
            return nil
        }.joined(separator: "\n")
        // Image blocks (browser screenshots) ride along — dropping them here
        // fed the model only "Screenshot of <url>" and invited it to
        // hallucinate the pixels (the 2026-07-09 HN incident).
        return .toolResult(
            toolCallId: toolCallId, content: textContent,
            images: content.imageAttachments(namespace: id))
    }
}

// MARK: - CoreMessage

/// Unified wrapper for the three core message types in a conversation.
nonisolated enum CoreMessage: AgentMessageProtocol, Sendable, Equatable, Identifiable {
    case user(UserMessage)
    case assistant(AssistantMessage)
    case toolResult(ToolResultMessage)

    var id: UUID {
        switch self {
        case .user(let msg): return msg.id
        case .assistant(let msg): return msg.id
        case .toolResult(let msg): return msg.id
        }
    }

    func toLLMMessage() -> LLMMessage? {
        switch self {
        case .user(let msg): return msg.toLLMMessage()
        case .assistant(let msg): return msg.toLLMMessage()
        case .toolResult(let msg): return msg.toLLMMessage()
        }
    }
}

// MARK: - CustomAgentMessage

/// Protocol for plugin/extension messages that carry a type discriminator.
protocol CustomAgentMessage: AgentMessageProtocol {
    var customType: String { get }
}

// MARK: - AgentMessage typealias

/// Existential type for any message in the agent pipeline.
typealias AgentMessage = any AgentMessageProtocol

// MARK: - Message Unwrapping Helpers

extension AgentMessageProtocol {
    /// Extracts UUID from any Identifiable message. All core types conform.
    nonisolated var messageUUID: UUID {
        if let identifiable = self as? any Identifiable,
            let uuid = identifiable.id as? UUID
        {
            return uuid
        }
        return UUID()
    }

    /// Unwraps to `UserMessage`, handling both bare struct and `CoreMessage.user`.
    nonisolated var asUser: UserMessage? {
        if let u = self as? UserMessage { return u }
        if let core = self as? CoreMessage, case .user(let u) = core { return u }
        return nil
    }

    /// Unwraps to `AssistantMessage`, handling both bare struct and `CoreMessage.assistant`.
    nonisolated var asAssistant: AssistantMessage? {
        if let a = self as? AssistantMessage { return a }
        if let core = self as? CoreMessage, case .assistant(let a) = core { return a }
        return nil
    }

    /// Unwraps to `ToolResultMessage`, handling both bare struct and `CoreMessage.toolResult`.
    nonisolated var asToolResult: ToolResultMessage? {
        if let t = self as? ToolResultMessage { return t }
        if let core = self as? CoreMessage, case .toolResult(let t) = core { return t }
        return nil
    }
}
