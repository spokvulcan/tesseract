import Foundation

// MARK: - BrowserToolResult

/// The MCP-shaped outcome of a browser tool call: content blocks plus the
/// `isError` flag the wire needs. Reuses ``ContentBlock`` so text and images
/// (screenshots) flow through unchanged, and so a later adapter can wrap these
/// as `AgentToolDefinition` results when Tesseract's own agent consumes the
/// server (PRD #190).
nonisolated struct BrowserToolResult: Sendable {
    let content: [ContentBlock]
    let isError: Bool

    static func text(_ string: String) -> BrowserToolResult {
        BrowserToolResult(content: [.text(string)], isError: false)
    }

    static func error(_ message: String) -> BrowserToolResult {
        BrowserToolResult(content: [.text(message)], isError: true)
    }

    static func blocks(_ blocks: [ContentBlock], isError: Bool = false) -> BrowserToolResult {
        BrowserToolResult(content: blocks, isError: isError)
    }
}
