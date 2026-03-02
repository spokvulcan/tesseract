import Foundation

// MARK: - ContentBlock

/// A block of content within a tool result or message.
enum ContentBlock: Sendable, Codable, Hashable {
    case text(String)
    case image(data: Data, mimeType: String)

    // MARK: Codable

    private enum CodingKeys: String, CodingKey {
        case type, text, data, mimeType
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "image":
            let data = try container.decode(Data.self, forKey: .data)
            let mimeType = try container.decode(String.self, forKey: .mimeType)
            self = .image(data: data, mimeType: mimeType)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "Unknown ContentBlock type: \(type)")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .image(let data, let mimeType):
            try container.encode("image", forKey: .type)
            try container.encode(data, forKey: .data)
            try container.encode(mimeType, forKey: .mimeType)
        }
    }
}

// MARK: - AgentToolResult

/// Structured result returned by a tool execution.
struct AgentToolResult: Sendable {
    let content: [ContentBlock]
    let details: (any Sendable & Hashable)?

    init(content: [ContentBlock], details: (any Sendable & Hashable)? = nil) {
        self.content = content
        self.details = details
    }

    /// Convenience: single text result.
    static func text(_ string: String) -> AgentToolResult {
        AgentToolResult(content: [.text(string)])
    }

    /// Convenience: error result (same structure, semantic marker for callers).
    static func error(_ message: String) -> AgentToolResult {
        AgentToolResult(content: [.text(message)])
    }
}

// MARK: - ToolProgressCallback

/// Callback for streaming incremental tool progress during execution.
typealias ToolProgressCallback = @Sendable (AgentToolResult) -> Void
