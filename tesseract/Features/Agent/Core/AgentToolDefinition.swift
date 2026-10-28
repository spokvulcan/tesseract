import Foundation
import MLXLMCommon
import os

// MARK: - CancellationToken

/// Cooperative cancellation token for long-running tool executions.
/// Thread-safe — can be checked from any isolation domain.
nonisolated final class CancellationToken: Sendable {
    private let _state = OSAllocatedUnfairLock(initialState: false)

    var isCancelled: Bool { _state.withLock { $0 } }
    func cancel() { _state.withLock { $0 = true } }
}

// MARK: - JSONSchema

/// Simple JSON Schema representation for tool parameter validation.
nonisolated struct JSONSchema: Sendable {
    let type: String  // "object"
    /// The parameters in the order the tool declares them. The model reads
    /// them in this order and tends to fill them in it, so the rendered tool
    /// keeps it (``ChatTemplateTools``).
    let properties: [(name: String, schema: PropertySchema)]
    let required: [String]

    init(type: String, properties: KeyValuePairs<String, PropertySchema>, required: [String]) {
        self.init(
            type: type, orderedProperties: properties.map { (name: $0.key, schema: $0.value) },
            required: required)
    }

    init(
        type: String, orderedProperties: [(name: String, schema: PropertySchema)],
        required: [String]
    ) {
        self.type = type
        self.properties = orderedProperties
        self.required = required
    }
}

/// Schema for a single tool parameter.
/// Uses a `Box` wrapper for `items` to break the recursive value-type cycle.
nonisolated struct PropertySchema: Sendable {
    let type: String  // "string", "integer", "boolean", "array"
    let description: String
    let enumValues: [String]?
    private let _items: Box<PropertySchema>?

    var items: PropertySchema? { _items?.value }

    init(
        type: String,
        description: String,
        enumValues: [String]? = nil,
        items: PropertySchema? = nil
    ) {
        self.type = type
        self.description = description
        self.enumValues = enumValues
        self._items = items.map { Box($0) }
    }
}

/// Heap-allocated box to break recursive value-type layouts.
private nonisolated final class Box<T: Sendable>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

// MARK: - AgentToolDefinition

/// Concrete tool definition using closures. Replaces the `AgentTool` protocol.
/// Named `AgentToolDefinition` to avoid collision with the existing `AgentTool` protocol
/// until Epic 6 removes it.
nonisolated struct AgentToolDefinition: Sendable {
    let name: String
    let label: String
    let description: String
    let parameterSchema: JSONSchema

    let execute:
        @Sendable (
            _ toolCallId: String,
            _ argsJSON: [String: JSONValue],
            _ signal: CancellationToken?,
            _ onUpdate: ToolProgressCallback?
        ) async throws -> AgentToolResult

    /// Generates an OpenAI-compatible function tool spec matching the format
    /// used by `AgentTool.toolSpec` and MLXLMCommon's `ToolSpec`. The
    /// parameters' declared order rides along under
    /// ``ChatTemplateTools/propertyOrderKey``, which the render reads and drops.
    var toolSpec: [String: any Sendable] {
        var properties: [String: any Sendable] = [:]
        for (key, prop) in parameterSchema.properties {
            properties[key] = prop.toSchemaDictionary()
        }
        return [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": parameterSchema.type,
                    "properties": properties,
                    "required": parameterSchema.required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            ChatTemplateTools.propertyOrderKey: parameterSchema.properties.map(\.name),
        ]
    }
}

// MARK: - PropertySchema → Dictionary

extension PropertySchema {
    /// Converts to the `[String: any Sendable]` format expected by MLXLMCommon's tokenizer.
    nonisolated func toSchemaDictionary() -> [String: any Sendable] {
        var dict: [String: any Sendable] = [
            "type": type,
            "description": description,
        ]
        if let enumValues {
            dict["enum"] = enumValues
        }
        if let items {
            dict["items"] = items.toSchemaDictionary()
        }
        return dict
    }
}
