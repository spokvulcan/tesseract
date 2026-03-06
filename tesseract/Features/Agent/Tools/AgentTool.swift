import Foundation
import MLXLMCommon

// MARK: - JSONValue Extraction Helpers

nonisolated enum ToolArgumentNormalizer: Sendable {

    static func normalize(_ arguments: [String: JSONValue]) -> [String: JSONValue] {
        arguments.mapValues { normalize($0) }
    }

    static func decode(_ json: String) -> [String: JSONValue]? {
        guard let data = json.data(using: .utf8),
              let parsed = try? JSONDecoder().decode([String: JSONValue].self, from: data) else
        {
            return nil
        }
        return normalize(parsed)
    }

    static func encode(_ arguments: [String: JSONValue]) -> String {
        let normalized = normalize(arguments)
        guard let data = try? JSONEncoder().encode(normalized),
              let json = String(data: data, encoding: .utf8) else
        {
            return "{}"
        }
        return json
    }

    static func normalize(_ value: JSONValue) -> JSONValue {
        switch value {
        case .string(let string):
            return normalizeWrappedString(string)
        case .array(let values):
            return .array(values.map { normalize($0) })
        case .object(let object):
            return .object(object.mapValues { normalize($0) })
        default:
            return value
        }
    }

    private static func normalizeWrappedString(_ string: String) -> JSONValue {
        var current: JSONValue = .string(string)

        for _ in 0..<8 {
            guard case .string(let wrapped) = current,
                  let next = parseWrappedScalar(wrapped) else {
                break
            }

            if case .string(let nextString) = next, nextString == wrapped {
                break
            }

            current = next
        }

        return current
    }

    private static func parseWrappedScalar(_ string: String) -> JSONValue? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)

        if ["null", ".null", "JSONValue.null"].contains(trimmed) {
            return .null
        }

        if let payload = wrappedPayload(
            from: trimmed,
            prefixes: ["string(", ".string(", "JSONValue.string("]
        ) {
            return .string(decodeWrappedString(payload))
        }

        if let payload = wrappedPayload(
            from: trimmed,
            prefixes: ["int(", ".int(", "JSONValue.int("]
        ),
        let value = Int(payload.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return .int(value)
        }

        if let payload = wrappedPayload(
            from: trimmed,
            prefixes: ["double(", ".double(", "JSONValue.double("]
        ),
        let value = Double(payload.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return .double(value)
        }

        if let payload = wrappedPayload(
            from: trimmed,
            prefixes: ["bool(", ".bool(", "JSONValue.bool("]
        ) {
            let value = payload.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if value == "true" { return .bool(true) }
            if value == "false" { return .bool(false) }
        }

        return nil
    }

    private static func wrappedPayload(
        from string: String,
        prefixes: [String]
    ) -> String? {
        guard string.hasSuffix(")") else { return nil }

        for prefix in prefixes where string.hasPrefix(prefix) {
            return String(string.dropFirst(prefix.count).dropLast())
        }

        return nil
    }

    private static func decodeWrappedString(_ payload: String) -> String {
        let trimmed = payload.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmed.hasPrefix("\""), trimmed.hasSuffix("\""),
           let data = trimmed.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }

        if trimmed.hasPrefix("'"), trimmed.hasSuffix("'"), trimmed.count >= 2 {
            return String(trimmed.dropFirst().dropLast())
        }

        return trimmed
    }
}

extension [String: JSONValue] {
    func string(for key: String) -> String? {
        guard let value = self[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return nil
        default: return nil
        }
    }

    func int(for key: String) -> Int? {
        guard let value = self[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .string(let s): return Int(s)
        case .null: return nil
        default: return nil
        }
    }

    func intArray(for key: String) -> [Int]? {
        guard let value = self[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .array(let arr):
            return arr.compactMap { element -> Int? in
                switch element {
                case .int(let i): return i
                case .double(let d): return Int(d)
                case .string(let s): return Int(s)
                default: return nil
                }
            }
        case .int(let i): return [i]  // Single int → treat as one-element array
        case .string(let s):
            // Accept comma-separated string like "3, 4, 5"
            let parts = s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            return parts.isEmpty ? nil : parts
        default: return nil
        }
    }

    func bool(for key: String) -> Bool? {
        guard let value = self[key] else { return nil }
        switch ToolArgumentNormalizer.normalize(value) {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .string(let s):
            let normalized = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if ["true", "1", "yes", "y", "on"].contains(normalized) { return true }
            if ["false", "0", "no", "n", "off"].contains(normalized) { return false }
            return nil
        case .null: return nil
        default: return nil
        }
    }
}
