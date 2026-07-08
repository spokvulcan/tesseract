import Foundation

// MARK: - ContentBlock

/// A block of content within a tool result or message.
nonisolated enum ContentBlock: Sendable, Codable, Hashable {
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

extension [ContentBlock] {
    /// Extracts and joins all `.text` blocks into a single string.
    nonisolated var textContent: String {
        compactMap { if case .text(let t) = $0 { t } else { nil } }.joined(separator: "\n")
    }

    /// Extracts all `.image` blocks as `ImageAttachment`s (slice #116), so
    /// tool-result images survive the transcript projection and join the Quick
    /// Look navigable set. `namespace` is the owning `ToolResultMessage.id`.
    ///
    /// The attachment `id` identifies the image **occurrence** — its position in
    /// a specific tool result — not its content: `namespace` (the stable
    /// tool-result id) folded with the block index. This is **load-bearing**:
    /// `ToolResultMessage` stores raw `ContentBlock`s with no per-image id, and
    /// this projection is recomputed both when the transcript builds the row
    /// *and* again inside `conversationImages()`. Quick Look matches the clicked
    /// id against that re-derived set, so the id must be (a) stable across calls
    /// — a fresh `UUID()` per call would never match, and would churn row
    /// identity on every streaming patch (defeating the `ToolCallRow` Equatable
    /// short-circuit and re-running the `.task(id:)` image decode) — and (b)
    /// unique per occurrence, so clicking the second of two byte-identical
    /// screenshots (same bytes, *different* tool results) opens that one, not
    /// the first. Occurrence identity gives both; deriving from the content
    /// digest gave only (a) and collided across results.
    nonisolated func imageAttachments(namespace: UUID) -> [ImageAttachment] {
        enumerated().compactMap { index, block in
            guard case .image(let data, let mimeType) = block else { return nil }
            return ImageAttachment(
                id: Self.occurrenceID(namespace: namespace, index: index),
                data: data, mimeType: mimeType
            )
        }
    }

    /// A reproducible UUID for an image occurrence: the tool-result `namespace`
    /// id with the block index XORed into its trailing bytes. Stable (both inputs
    /// are stable) and unique per (tool result, position), so re-projections
    /// match and distinct occurrences — even byte-identical ones — stay distinct.
    nonisolated private static func occurrenceID(namespace: UUID, index: Int) -> UUID {
        let idx = UInt32(truncatingIfNeeded: index)
        var u = namespace.uuid  // uuid_t — XOR the index into its trailing bytes
        u.12 ^= UInt8(truncatingIfNeeded: idx)
        u.13 ^= UInt8(truncatingIfNeeded: idx >> 8)
        u.14 ^= UInt8(truncatingIfNeeded: idx >> 16)
        u.15 ^= UInt8(truncatingIfNeeded: idx >> 24)
        return UUID(uuid: u)
    }
}

// MARK: - AgentToolResult

/// Structured result returned by a tool execution.
nonisolated struct AgentToolResult: Sendable {
    let content: [ContentBlock]
    /// Typed facts about the execution (PRD #200) — threaded onto the
    /// persisted `ToolResultMessage` by the agent loop's commit step.
    let details: ToolResultDetails?

    init(content: [ContentBlock], details: ToolResultDetails? = nil) {
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
