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
    /// Look navigable set.
    ///
    /// The attachment `id` is derived deterministically from the image bytes (the
    /// same `ImageDigest` that keys the prefix and preview-file caches) folded
    /// with the block index. This is **load-bearing**: `ToolResultMessage` stores
    /// raw `ContentBlock`s with no id, and this projection is recomputed both when
    /// the transcript builds the row *and* again inside `conversationImages()`.
    /// Quick Look matches the clicked id against that re-derived set — a fresh
    /// `UUID()` per call would never match (the viewer would never open) and would
    /// churn row identity on every streaming patch (defeating the `ToolCallRow`
    /// Equatable short-circuit and re-running the `.task(id:)` image decode). Same
    /// bytes at the same position ⇒ same id across every call.
    nonisolated var imageAttachments: [ImageAttachment] {
        enumerated().compactMap { index, block in
            guard case .image(let data, let mimeType) = block else { return nil }
            return ImageAttachment(
                id: Self.stableImageID(for: data, index: index),
                data: data, mimeType: mimeType
            )
        }
    }

    /// A reproducible UUID for a tool-result image: the leading 16 bytes of the
    /// image's SHA-256 `ImageDigest`, with the block index XORed into the trailing
    /// bytes so two byte-identical images in one result still get distinct — but
    /// stable — ids. Cross-launch stable, matching the digest used for caching.
    nonisolated private static func stableImageID(for data: Data, index: Int) -> UUID {
        let digest = ImageDigest(imageBytes: data).rawBytes   // 32 SHA-256 bytes
        var bytes = [UInt8](repeating: 0, count: 16)
        digest.copyBytes(to: &bytes, count: 16)
        let idx = UInt32(truncatingIfNeeded: index)
        bytes[12] ^= UInt8(truncatingIfNeeded: idx)
        bytes[13] ^= UInt8(truncatingIfNeeded: idx >> 8)
        bytes[14] ^= UInt8(truncatingIfNeeded: idx >> 16)
        bytes[15] ^= UInt8(truncatingIfNeeded: idx >> 24)
        let t: uuid_t = (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        )
        return UUID(uuid: t)
    }
}

// MARK: - AgentToolResult

/// Structured result returned by a tool execution.
nonisolated struct AgentToolResult: Sendable {
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
