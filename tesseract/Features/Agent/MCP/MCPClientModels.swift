import Foundation
import MLXLMCommon

// MARK: - JSONValue client accessors

/// Read-side conveniences the MCP **client** needs to walk decoded JSON-RPC
/// trees. Complements the `asObject`/`asString` pair the Browser MCP *server*
/// already defines (`MCPModels.swift`); kept `nonisolated` so the transport can
/// parse responses off the MainActor.
extension JSONValue {
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

// MARK: - MCPToolDescriptor

/// One tool as a server advertises it in `tools/list`. `inputSchema` is carried
/// as the raw JSON Schema object and passed through faithfully (never
/// re-authored) — the decision in the PRD's Implementation notes.
nonisolated struct MCPToolDescriptor: Sendable, Hashable {
    let name: String
    let description: String
    let inputSchema: JSONValue
}

// MARK: - MCPCallResult

/// The decoded outcome of a `tools/call`. `content` reuses the app's
/// ``ContentBlock`` so text and images (e.g. a browser screenshot, US #12) flow
/// straight into an `AgentToolResult`. `isError` mirrors the MCP result flag.
nonisolated struct MCPCallResult: Sendable {
    let content: [ContentBlock]
    let isError: Bool
}

// MARK: - MCPConnectionState

/// A configured server's live connection state, surfaced to the settings UI so
/// the user can tell at a glance what the agent can currently do (US #5/#6).
nonisolated enum MCPConnectionState: Sendable, Equatable {
    case idle
    case connecting
    case connected
    case failed(String)
}

// MARK: - MCPClientError

/// Failures the client raises. Tool-call failures never reach here as throws in
/// normal operation — an MCP `isError` result is delivered as content (US #11);
/// these cover handshake/transport/protocol faults.
nonisolated enum MCPClientError: LocalizedError, Sendable {
    case transport(String)
    case http(status: Int, body: String)
    case rpc(code: Int, message: String)
    case malformedResponse(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .transport(let detail):
            return "MCP transport error: \(detail)"
        case .http(let status, let body):
            let trimmed = body.prefix(200)
            return "MCP server returned HTTP \(status)\(trimmed.isEmpty ? "" : ": \(trimmed)")"
        case .rpc(let code, let message):
            return "MCP error \(code): \(message)"
        case .malformedResponse(let detail):
            return "Malformed MCP response: \(detail)"
        case .cancelled:
            return "MCP request cancelled"
        }
    }
}

// MARK: - MCPWire

/// Builds and parses the client half of the JSON-RPC / MCP wire as `JSONValue`
/// trees — the counterpart to the server's `MCPEncoding`. `nonisolated` so the
/// transport can encode requests and decode responses off the MainActor.
nonisolated enum MCPWire {

    // MARK: Envelopes

    static func request(id: Int, method: String, params: JSONValue?) -> JSONValue {
        var object: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "id": .int(id),
            "method": .string(method),
        ]
        if let params { object["params"] = params }
        return .object(object)
    }

    static func notification(method: String, params: JSONValue?) -> JSONValue {
        var object: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "method": .string(method),
        ]
        if let params { object["params"] = params }
        return .object(object)
    }

    static func encode(_ value: JSONValue) -> Data {
        (try? JSONEncoder().encode(value)) ?? Data("{}".utf8)
    }

    static func decode(_ data: Data) -> JSONValue? {
        try? JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// True when `value` is a JSON-RPC *response* (carries an `id` and either a
    /// `result` or an `error`) rather than a server-initiated notification.
    static func isResponse(_ value: JSONValue) -> Bool {
        guard let object = value.asObject else { return false }
        return object["id"] != nil && (object["result"] != nil || object["error"] != nil)
    }

    // MARK: Result parsing

    static func parseToolsList(_ result: JSONValue) -> [MCPToolDescriptor] {
        guard let tools = result.asObject?["tools"]?.asArray else { return [] }
        return tools.compactMap { entry in
            guard let object = entry.asObject, let name = object["name"]?.asString else {
                return nil
            }
            let description = object["description"]?.asString ?? ""
            let schema = object["inputSchema"] ?? .object(["type": .string("object")])
            return MCPToolDescriptor(name: name, description: description, inputSchema: schema)
        }
    }

    static func parseCallResult(_ result: JSONValue) -> MCPCallResult {
        let object = result.asObject
        let isError = object?["isError"]?.asBool ?? false
        let blocks = (object?["content"]?.asArray ?? []).compactMap(parseContentBlock)
        // A result with no decodable blocks still round-trips as empty text so
        // the model always sees *something* rather than a silent hole.
        return MCPCallResult(
            content: blocks.isEmpty ? [.text("")] : blocks, isError: isError)
    }

    /// Decode one MCP content block. Text and image are supported (US #12);
    /// audio and embedded resources are out of scope for v1 and dropped.
    static func parseContentBlock(_ value: JSONValue) -> ContentBlock? {
        guard let object = value.asObject, let type = object["type"]?.asString else { return nil }
        switch type {
        case "text":
            return .text(object["text"]?.asString ?? "")
        case "image":
            guard let base64 = object["data"]?.asString,
                let data = Data(base64Encoded: base64)
            else { return nil }
            let mimeType = object["mimeType"]?.asString ?? "image/png"
            return .image(data: data, mimeType: mimeType)
        default:
            return nil
        }
    }
}
