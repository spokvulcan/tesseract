import Foundation
import MLXLMCommon

/// A present-but-wrong-shape argument — the loud half of the extractor
/// (#354's summons-as-string class): the message names the expected shape so
/// the model can correct the call instead of trusting a silent coercion.
nonisolated struct ToolArgTypeError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

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

    static func stringArray(_ args: [String: JSONValue], key: String) -> [String]? {
        guard let value = args[key] else { return nil }
        guard case .array(let items) = ToolArgumentNormalizer.normalize(value) else { return nil }
        let strings = items.compactMap { item -> String? in
            switch ToolArgumentNormalizer.normalize(item) {
            case .string(let s): return s
            case .int(let i): return String(i)
            case .double(let d): return String(d)
            default: return nil
            }
        }
        return strings
    }

    /// Strict boolean: absent is nil, but present-and-not-a-JSON-boolean
    /// fails loudly — never coerced from "true"/"False"/1 (the tolerance
    /// #354 caught in the field). The companion palette reads bools here.
    static func strictBool(_ args: [String: JSONValue], key: String) throws -> Bool? {
        guard let value = args[key] else { return nil }
        guard case .bool(let flag) = ToolArgumentNormalizer.normalize(value) else {
            throw ToolArgTypeError(
                message: "'\(key)' must be a JSON boolean (true or false), "
                    + "not a string or number.")
        }
        return flag
    }

    /// A required object-shaped argument (a payload); anything else — absent,
    /// a string of JSON, an array — fails loudly with the shape named.
    static func object(_ args: [String: JSONValue], key: String) throws -> [String: JSONValue] {
        guard let value = args[key],
            case .object(let object) = ToolArgumentNormalizer.normalize(value)
        else {
            throw ToolArgTypeError(message: "'\(key)' must be a JSON object.")
        }
        return object
    }

    /// A required array of objects; any non-object element fails loudly.
    static func objectArray(_ args: [String: JSONValue], key: String) throws
        -> [[String: JSONValue]]
    {
        guard let value = args[key],
            case .array(let items) = ToolArgumentNormalizer.normalize(value)
        else {
            throw ToolArgTypeError(message: "'\(key)' must be a JSON array of objects.")
        }
        return try items.map { item in
            guard case .object(let object) = item else {
                throw ToolArgTypeError(
                    message: "Every element of '\(key)' must be a JSON object.")
            }
            return object
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
