import Foundation

// MARK: - ContentPart
//
// The canonical assistant-content model (ADR-0024): a verbatim Swift mirror of
// pi-ai's `AssistantMessage["content"]` union (`packages/ai/src/types.ts`) —
// an ordered list of typed **Content Part**s. Row identity in the UI derives
// from (message id, part index); the stream protocol addresses parts by
// `contentIndex` (see `AssistantMessageEvent`).
//
// One deliberate adaptation: pi-ai's `ToolCall.arguments` is a parsed JSON
// record; here the arguments stay the normalized JSON string the whole local
// toolchain speaks (`ToolCallParser` → `ToolArgumentNormalizer` → tool
// execution → display). `parsedArguments` recovers the record on demand.

/// One typed unit of assistant content, in generation order.
nonisolated enum ContentPart: Sendable, Codable, Equatable {
    case text(TextPart)
    case thinking(ThinkingPart)
    case toolCall(ToolCallPart)

    private enum CodingKeys: String, CodingKey { case type }

    private enum Kind: String, Codable {
        case text, thinking, toolCall
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .type) {
        case .text: self = .text(try TextPart(from: decoder))
        case .thinking: self = .thinking(try ThinkingPart(from: decoder))
        case .toolCall: self = .toolCall(try ToolCallPart(from: decoder))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let part):
            try container.encode(Kind.text, forKey: .type)
            try part.encode(to: encoder)
        case .thinking(let part):
            try container.encode(Kind.thinking, forKey: .type)
            try part.encode(to: encoder)
        case .toolCall(let part):
            try container.encode(Kind.toolCall, forKey: .type)
            try part.encode(to: encoder)
        }
    }
}

/// Visible assistant text.
nonisolated struct TextPart: Sendable, Codable, Equatable {
    var text: String
}

/// Reasoning inside a `<think>` block.
nonisolated struct ThinkingPart: Sendable, Codable, Equatable {
    var thinking: String
}

/// A tool call with stable identity. In a *committed* message
/// `argumentsJSON` is the normalized JSON string; while the call is still
/// streaming (the Open Tool Call in a partial message) it holds the raw
/// accumulated body fragments, replaced wholesale at `toolcallEnd`. The id
/// is minted at name-lock and survives the partial → committed transition.
nonisolated struct ToolCallPart: Sendable, Codable, Equatable, Identifiable {
    let id: String
    let name: String
    var argumentsJSON: String
}

// MARK: - StopReason

/// Why an assistant message stopped — pi-ai's `StopReason`, verbatim.
nonisolated enum StopReason: String, Sendable, Codable, Equatable {
    case stop
    case length
    case toolUse
    case error
    case aborted
}

// MARK: - Usage

/// Token accounting for one assistant message — pi-ai's `Usage`, verbatim.
/// Local inference has no billing, so `cost` stays zero; the shape is kept so
/// future protocol work ports directly.
nonisolated struct Usage: Sendable, Codable, Equatable {
    var input: Int = 0
    var output: Int = 0
    var cacheRead: Int = 0
    var cacheWrite: Int = 0
    /// Reasoning-token subset of `output`, when known.
    var reasoning: Int?
    var totalTokens: Int = 0

    nonisolated struct Cost: Sendable, Codable, Equatable {
        var input: Double = 0
        var output: Double = 0
        var cacheRead: Double = 0
        var cacheWrite: Double = 0
        var total: Double = 0
    }

    var cost = Cost()
}
