import Foundation
import MLXLMCommon

// MARK: - MCP constants

nonisolated enum MCPProtocol {
    /// Protocol revision we implement; echoed on `initialize` unless the client
    /// asks for a specific one we recognize.
    static let version = "2025-06-18"
    static let serverName = "tesseract-agent-browser"
    static let serverVersion = "1.0.0"

    /// JSON-RPC error codes used on the wire.
    enum ErrorCode {
        static let parseError = -32_700
        static let invalidRequest = -32_600
        static let methodNotFound = -32_601
        static let invalidParams = -32_602
        static let internalError = -32_603
        /// Server-defined: no session (missing/expired `Mcp-Session-Id`).
        static let noSession = -32_001
    }
}

// MARK: - JSONValue conveniences

extension JSONValue {
    nonisolated var asObject: [String: JSONValue]? {
        if case .object(let object) = self { return object }
        return nil
    }
    nonisolated var asString: String? {
        if case .string(let string) = self { return string }
        return nil
    }
}

// MARK: - MCPEncoding

/// Builds JSON-RPC / MCP payloads as `JSONValue` trees (encoded with a plain
/// `JSONEncoder`), so the whole transport speaks one value type end to end.
nonisolated enum MCPEncoding {

    // MARK: Envelopes

    static func result(id: JSONValue, _ result: JSONValue) -> JSONValue {
        .object([
            "jsonrpc": .string("2.0"),
            "id": id,
            "result": result,
        ])
    }

    static func error(id: JSONValue, code: Int, message: String) -> JSONValue {
        .object([
            "jsonrpc": .string("2.0"),
            "id": id,
            "error": .object([
                "code": .int(code),
                "message": .string(message),
            ]),
        ])
    }

    // MARK: Method results

    static func initializeResult(protocolVersion: String) -> JSONValue {
        .object([
            "protocolVersion": .string(protocolVersion),
            "capabilities": .object([
                "tools": .object(["listChanged": .bool(false)])
            ]),
            "serverInfo": .object([
                "name": .string(MCPProtocol.serverName),
                "version": .string(MCPProtocol.serverVersion),
            ]),
        ])
    }

    static func toolsListResult(_ specs: [BrowserToolSpec]) -> JSONValue {
        .object([
            "tools": .array(
                specs.map { spec in
                    .object([
                        "name": .string(spec.name),
                        "description": .string(spec.description),
                        "inputSchema": inputSchema(spec.inputSchema),
                    ])
                })
        ])
    }

    static func toolCallResult(_ result: BrowserToolResult) -> JSONValue {
        .object([
            "content": .array(result.content.map(content(_:))),
            "isError": .bool(result.isError),
        ])
    }

    // MARK: Pieces

    static func content(_ block: ContentBlock) -> JSONValue {
        switch block {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])
        case .image(let data, let mimeType):
            return .object([
                "type": .string("image"),
                "data": .string(data.base64EncodedString()),
                "mimeType": .string(mimeType),
            ])
        }
    }

    static func inputSchema(_ schema: JSONSchema) -> JSONValue {
        var properties: [String: JSONValue] = [:]
        for (key, prop) in schema.properties {
            properties[key] = property(prop)
        }
        return .object([
            "type": .string(schema.type),
            "properties": .object(properties),
            "required": .array(schema.required.map { .string($0) }),
        ])
    }

    private static func property(_ prop: PropertySchema) -> JSONValue {
        var fields: [String: JSONValue] = [
            "type": .string(prop.type),
            "description": .string(prop.description),
        ]
        if let enumValues = prop.enumValues {
            fields["enum"] = .array(enumValues.map { .string($0) })
        }
        if let items = prop.items {
            fields["items"] = property(items)
        }
        return .object(fields)
    }
}
