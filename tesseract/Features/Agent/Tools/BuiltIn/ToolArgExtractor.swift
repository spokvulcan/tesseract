import Foundation
import MLXLMCommon

/// Shared argument extraction from `[String: JSONValue]` tool arguments.
/// Used by all built-in tools to avoid duplicating the same switch logic.
nonisolated enum ToolArgExtractor: Sendable {

    static func string(_ args: [String: JSONValue], key: String) -> String? {
        guard let value = args[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        default: return nil
        }
    }

    static func int(_ args: [String: JSONValue], key: String) -> Int? {
        guard let value = args[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .string(let s): return Int(s)
        default: return nil
        }
    }

    static func bool(_ args: [String: JSONValue], key: String) -> Bool? {
        guard let value = args[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .string(let s):
            let normalized = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if ["true", "1", "yes", "y", "on"].contains(normalized) { return true }
            if ["false", "0", "no", "n", "off"].contains(normalized) { return false }
            return nil
        default: return nil
        }
    }
}
