import Foundation
import MLXLMCommon

/// Streaming parser that detects `<tool_call>` tags in generated text and extracts `ToolCall` objects.
///
/// Handles edge cases:
/// - Partial tags split across chunks (buffers until complete)
/// - Malformed JSON (reported as `.malformedToolCall` for model retry)
/// - Multiple tool calls in a single response
///
/// Usage:
/// ```
/// let parser = ToolCallParser()
/// for chunk in textStream {
///     for event in parser.processChunk(chunk) { ... }
/// }
/// for event in parser.finalize() { ... }
/// ```
final class ToolCallParser {

    /// Events emitted as chunks are processed.
    enum Event: Sendable {
        /// Regular text content (not part of a tool call tag).
        case text(String)
        /// A successfully parsed tool call.
        case toolCall(ToolCall)
        /// A `<tool_call>` tag was found but the JSON content was malformed.
        case malformedToolCall(String)
    }

    private static let startTag = "<tool_call>"
    private static let endTag = "</tool_call>"

    private var buffer = ""

    /// Process a chunk of streaming text and return any events.
    func processChunk(_ chunk: String) -> [Event] {
        buffer += chunk
        return drain()
    }

    /// Flush any remaining buffered text as events.
    /// Call this when generation is complete.
    func finalize() -> [Event] {
        guard !buffer.isEmpty else { return [] }
        let remaining = buffer
        buffer = ""
        return [.text(remaining)]
    }

    // MARK: - Private

    private func drain() -> [Event] {
        var events: [Event] = []

        while !buffer.isEmpty {
            guard let startRange = buffer.range(of: Self.startTag) else {
                // No start tag — check for a partial match at the tail
                if let splitIndex = partialTagSplitIndex() {
                    let text = String(buffer[..<splitIndex])
                    if !text.isEmpty { events.append(.text(text)) }
                    buffer = String(buffer[splitIndex...])
                } else {
                    events.append(.text(buffer))
                    buffer = ""
                }
                break
            }

            // Emit text before the start tag
            let before = String(buffer[..<startRange.lowerBound])
            if !before.isEmpty { events.append(.text(before)) }

            // Look for end tag
            let afterStart = String(buffer[startRange.upperBound...])
            guard let endRange = afterStart.range(of: Self.endTag) else {
                // No end tag yet — keep buffering from start tag onward
                buffer = String(buffer[startRange.lowerBound...])
                break
            }

            // Extract and parse JSON content between tags
            let jsonContent = String(afterStart[..<endRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if let data = jsonContent.data(using: .utf8),
               let function = try? JSONDecoder().decode(ToolCall.Function.self, from: data)
            {
                events.append(.toolCall(ToolCall(function: function)))
            } else {
                events.append(.malformedToolCall(jsonContent))
            }

            // Continue processing after end tag
            buffer = String(afterStart[endRange.upperBound...])
        }

        return events
    }

    /// Returns the index where a partial `<tool_call>` prefix begins at the tail of the buffer,
    /// or nil if no partial match exists.
    private func partialTagSplitIndex() -> String.Index? {
        let tag = Self.startTag
        for length in stride(from: tag.count - 1, through: 1, by: -1) {
            let prefix = String(tag.prefix(length))
            if buffer.hasSuffix(prefix) {
                return buffer.index(buffer.endIndex, offsetBy: -length)
            }
        }
        return nil
    }
}
