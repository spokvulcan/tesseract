import Foundation
import MLXLMCommon

// MARK: - MCPSchemaDecoder

/// Decodes a server's raw JSON-Schema `inputSchema` into the app's
/// ``JSONSchema``. The `required` list is carried faithfully because the agent
/// loop validates required parameters against it *before* dispatch
/// (`AgentLoop.executeToolCalls`); top-level property types/descriptions/enums
/// feed the model's tool spec. Nested object schemas flatten to their declared
/// `type` — the same shape the built-in tools use.
nonisolated enum MCPSchemaDecoder {
    static func decode(_ value: JSONValue) -> JSONSchema {
        let object = value.asObject ?? [:]
        let type = object["type"]?.asString ?? "object"
        let required = (object["required"]?.asArray ?? []).compactMap { $0.asString }
        var properties: [String: PropertySchema] = [:]
        for (key, propertyValue) in object["properties"]?.asObject ?? [:] {
            properties[key] = decodeProperty(propertyValue)
        }
        return JSONSchema(type: type, properties: properties, required: required)
    }

    private static func decodeProperty(_ value: JSONValue) -> PropertySchema {
        let object = value.asObject ?? [:]
        // `type` may be a string or, in some schemas, an array (["string","null"]);
        // take the first non-null string, else default to "string".
        let type =
            object["type"]?.asString
            ?? object["type"]?.asArray?.compactMap({ $0.asString }).first(where: { $0 != "null" })
            ?? "string"
        let description = object["description"]?.asString ?? ""
        let enumValues = object["enum"]?.asArray?.compactMap { $0.asString }
        let items = object["items"].map { decodeProperty($0) }
        return PropertySchema(
            type: type, description: description, enumValues: enumValues, items: items)
    }
}

// MARK: - MCPToolAdapter

/// Materializes an ``MCPToolDescriptor`` as an ``AgentToolDefinition`` the agent
/// selects and calls exactly like a built-in (US #7). The tool name is
/// namespaced by server so two servers can never collide with each other or with
/// a built-in (US #8); the un-namespaced name is kept as the display `label`.
nonisolated enum MCPToolAdapter {

    /// The closure a connection supplies to actually invoke the tool over its
    /// client. Separated from the descriptor so the adapter stays pure.
    typealias Invoke =
        @Sendable (
            _ toolName: String,
            _ arguments: [String: JSONValue],
            _ signal: CancellationToken?,
            _ onProgress: @escaping @Sendable (String) -> Void
        ) async throws -> MCPCallResult

    static func toolDefinition(
        descriptor: MCPToolDescriptor,
        namespace: String,
        invoke: @escaping Invoke
    ) -> AgentToolDefinition {
        let namespacedName = MCPServerConfig.namespacedToolName(
            namespace: namespace, tool: descriptor.name)
        let originalName = descriptor.name
        return AgentToolDefinition(
            name: namespacedName,
            label: descriptor.name,
            description: descriptor.description,
            parameterSchema: MCPSchemaDecoder.decode(descriptor.inputSchema),
            execute: { _, arguments, signal, onUpdate in
                let result = try await invoke(originalName, arguments, signal) { progressText in
                    onUpdate?(.text(progressText))
                }
                // MCP `isError` results are ordinary tool-result content, so the
                // agent reads the error and recovers in-conversation (US #11).
                return AgentToolResult(content: result.content)
            }
        )
    }
}
