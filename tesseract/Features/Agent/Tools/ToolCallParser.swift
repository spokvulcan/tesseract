import Foundation
import MLXLMCommon

/// Streaming parser that detects `<tool_call>` and `<think>` tags in generated text.
///
/// Handles edge cases:
/// - Partial tags split across chunks (buffers until complete)
/// - Malformed JSON (reported as `.malformedToolCall` for model retry)
/// - Multiple tool calls in a single response
/// - Think blocks emitted as streaming events for UI display
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
        /// Regular text content (not part of a tag).
        case text(String)
        /// A successfully parsed tool call.
        case toolCall(ToolCall)
        /// A `<tool_call>` tag was found but the JSON content was malformed.
        case malformedToolCall(String)
        /// The model started a `<think>` block.
        case thinkStart
        /// A streaming chunk of thinking content.
        case thinking(String)
        /// The model finished its `<think>` block.
        case thinkEnd
    }

    private static let toolStartTag = "<tool_call>"
    private static let toolEndTag = "</tool_call>"
    private static let thinkStartTag = "<think>"
    private static let thinkEndTag = "</think>"

    private var buffer = ""
    private var insideThinkBlock = false

    /// Process a chunk of streaming text and return any events.
    func processChunk(_ chunk: String) -> [Event] {
        buffer += chunk
        return drain()
    }

    /// Flush any remaining buffered text as events.
    /// Call this when generation is complete.
    func finalize() -> [Event] {
        guard !buffer.isEmpty else {
            if insideThinkBlock {
                insideThinkBlock = false
                return [.thinkEnd]
            }
            return []
        }

        let remaining = buffer
        buffer = ""

        if insideThinkBlock {
            insideThinkBlock = false
            var events: [Event] = []
            if !remaining.isEmpty { events.append(.thinking(remaining)) }
            events.append(.thinkEnd)
            return events
        }

        return [.text(remaining)]
    }

    // MARK: - Private

    private func drain() -> [Event] {
        var events: [Event] = []

        while !buffer.isEmpty {
            if insideThinkBlock {
                // Inside a think block — look for </think>
                if let endRange = buffer.range(of: Self.thinkEndTag) {
                    let content = String(buffer[..<endRange.lowerBound])
                    if !content.isEmpty { events.append(.thinking(content)) }
                    events.append(.thinkEnd)
                    insideThinkBlock = false
                    buffer = String(buffer[endRange.upperBound...])
                    continue
                }

                // No end tag yet — check for partial </think> suffix
                if let splitIndex = partialSuffixIndex(for: Self.thinkEndTag) {
                    let content = String(buffer[..<splitIndex])
                    if !content.isEmpty { events.append(.thinking(content)) }
                    buffer = String(buffer[splitIndex...])
                } else {
                    // Emit all buffered content as thinking
                    events.append(.thinking(buffer))
                    buffer = ""
                }
                break
            }

            // Not inside a think block — find whichever tag comes first
            let thinkStartRange = buffer.range(of: Self.thinkStartTag)
            let thinkEndRange = buffer.range(of: Self.thinkEndTag)
            let toolRange = buffer.range(of: Self.toolStartTag)

            // Find the earliest tag
            var earliest: (kind: Int, lower: String.Index)? // kind: 0=thinkStart, 1=thinkEnd, 2=tool
            for (kind, range) in [(0, thinkStartRange), (1, thinkEndRange), (2, toolRange)] {
                guard let r = range else { continue }
                if earliest == nil || r.lowerBound < earliest!.lower {
                    earliest = (kind, r.lowerBound)
                }
            }

            if let earliest, earliest.kind == 0, let thinkStartRange {
                // <think> — enter think block
                let before = String(buffer[..<thinkStartRange.lowerBound])
                if !before.isEmpty { events.append(.text(before)) }

                events.append(.thinkStart)
                insideThinkBlock = true
                buffer = String(buffer[thinkStartRange.upperBound...])
                continue
            }

            if let earliest, earliest.kind == 1, let thinkEndRange {
                // Stray </think> without opening <think> — model omitted the start tag.
                // Emit text before it (this is leaked thinking content), signal transition.
                let before = String(buffer[..<thinkEndRange.lowerBound])
                if !before.isEmpty { events.append(.text(before)) }
                events.append(.thinkEnd)
                buffer = String(buffer[thinkEndRange.upperBound...])
                continue
            }

            // Fall through to existing <tool_call> logic
            guard let startRange = toolRange else {
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
            guard let endRange = afterStart.range(of: Self.toolEndTag) else {
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
            } else if let toolCall = Self.parseXMLFunction(jsonContent) {
                // Fallback: parse XML function format (<function=name><parameter=key>value</parameter></function>)
                // This catches cases where the library's XMLFunctionParser intermittently fails
                events.append(.toolCall(toolCall))
            } else {
                events.append(.malformedToolCall(jsonContent))
            }

            // Continue processing after end tag
            buffer = String(afterStart[endRange.upperBound...])
        }

        return events
    }

    /// Returns the earliest index where a partial prefix of any relevant tag
    /// begins at the tail of the buffer, or nil if no partial match exists.
    private func partialTagSplitIndex() -> String.Index? {
        let tags = [Self.toolStartTag, Self.toolEndTag, Self.thinkStartTag, Self.thinkEndTag]
        var earliest: String.Index?

        for tag in tags {
            if let idx = partialSuffixIndex(for: tag) {
                if let current = earliest {
                    if idx < current { earliest = idx }
                } else {
                    earliest = idx
                }
            }
        }

        return earliest
    }

    /// Regex for extracting `<parameter=KEY>VALUE</parameter>` pairs from XML function format.
    private static let paramRegex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: #"<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>"#)
    }()

    /// Parse XML function format: `<function=name><parameter=key>value</parameter></function>`
    /// Used as fallback when the library's XMLFunctionParser fails.
    private static func parseXMLFunction(_ content: String) -> ToolCall? {
        // Match <function=NAME>...</function> (closing tag may lack ">")
        guard let funcMatch = content.range(of: #"<function=([^>]+)>"#, options: .regularExpression) else {
            return nil
        }

        let funcName = String(content[funcMatch].dropFirst("<function=".count).dropLast(">".count))

        // Extract all <parameter=KEY>VALUE</parameter> pairs
        var arguments: [String: JSONValue] = [:]
        let nsContent = content as NSString
        let matches = paramRegex.matches(in: content, range: NSRange(location: 0, length: nsContent.length))

        for match in matches {
            guard match.numberOfRanges >= 3 else { continue }
            let key = nsContent.substring(with: match.range(at: 1))
            let value = nsContent.substring(with: match.range(at: 2))
            arguments[key] = .string(value)
        }

        return ToolCall(function: ToolCall.Function(name: funcName, arguments: arguments))
    }

    /// Returns the index where a partial prefix of the given tag begins at the tail of the buffer.
    private func partialSuffixIndex(for tag: String) -> String.Index? {
        for length in stride(from: tag.count - 1, through: 1, by: -1) {
            let prefix = String(tag.prefix(length))
            if buffer.hasSuffix(prefix) {
                return buffer.index(buffer.endIndex, offsetBy: -length)
            }
        }
        return nil
    }
}
