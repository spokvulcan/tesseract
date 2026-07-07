import MLXLMCommon

// MARK: - JSONValue accessors

/// Read-side conveniences for walking a decoded `JSONValue` tree, shared across
/// the codebase: the Browser MCP *server* builds and parses its own JSON-RPC,
/// and the agent's MCP *client* parses arbitrary servers' responses. Kept
/// `nonisolated` so transports can parse off the MainActor. One home so a new
/// accessor can't clash with a duplicate declaration in another feature folder.
extension JSONValue {
    nonisolated var asObject: [String: JSONValue]? {
        if case .object(let object) = self { return object }
        return nil
    }
    nonisolated var asString: String? {
        if case .string(let string) = self { return string }
        return nil
    }
    nonisolated var asArray: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
    nonisolated var asInt: Int? {
        switch self {
        case .int(let value): return value
        case .double(let value): return Int(value)
        default: return nil
        }
    }
    nonisolated var asBool: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }
    nonisolated var asDouble: Double? {
        switch self {
        case .double(let value): return value
        case .int(let value): return Double(value)
        default: return nil
        }
    }
}
