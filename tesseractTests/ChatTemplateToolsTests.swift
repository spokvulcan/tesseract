import Foundation
import Jinja
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The key order `ChatTemplateTools` gives the tools a chat template renders,
/// checked through swift-jinja's `tojson` (transformers' `json.dumps` format).
/// `ToolPromptParityTests` checks whole prompts against transformers on real
/// templates; these run without a model.
struct ChatTemplateToolsTests {

    private func rendered(_ tools: [ToolSpec]) throws -> String {
        try Template("{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}")
            .render(["tools": ChatTemplateTools.value(tools)])
    }

    private func definition(_ properties: KeyValuePairs<String, PropertySchema>) -> ToolSpec {
        AgentToolDefinition(
            name: "add_event", label: "add_event", description: "Add an event.",
            parameterSchema: JSONSchema(
                type: "object", properties: properties, required: ["title"]),
            execute: { _, _, _, _ in .text("") }
        ).toolSpec
    }

    /// An app tool renders as transformers renders the same tool written in
    /// Python: the builder's key order, the declared parameter order, and no
    /// trace of the order it carried.
    @Test func appToolRendersInDeclaredOrder() throws {
        let tool = definition([
            "title": PropertySchema(type: "string", description: "The title."),
            "start": PropertySchema(type: "string", description: "Local start."),
            "duration_minutes": PropertySchema(type: "integer", description: "Instead of end."),
            "tags": PropertySchema(
                type: "array", description: "Tags.",
                items: PropertySchema(
                    type: "string", description: "A tag.", enumValues: ["b", "a"])),
        ])
        #expect(
            try rendered([tool])
                == #"{"type": "function", "function": {"name": "add_event", "description": "Add an event.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "The title."}, "start": {"type": "string", "description": "Local start."}, "duration_minutes": {"type": "integer", "description": "Instead of end."}, "tags": {"type": "array", "description": "Tags.", "items": {"type": "string", "description": "A tag.", "enum": ["b", "a"]}}}, "required": ["title"]}}}"#
                + "\n")
    }

    /// The order survives the canonicalization every production render does
    /// (a JSON round trip into new dictionaries).
    @Test func declaredOrderSurvivesCanonicalization() throws {
        let tool = definition([
            "title": PropertySchema(type: "string", description: "T"),
            "after_event": PropertySchema(type: "string", description: "A"),
        ])
        let canonical = try #require(LLMActor.canonicalizeToolSpecs([tool]))
        #expect(try rendered(canonical) == rendered([tool]))
        #expect(try rendered(canonical).contains(#""properties": {"title": "#))
    }

    /// A tool without a declared order (a client's or an MCP server's) gets
    /// the conventional keys first and everything else sorted, parameter names
    /// included; data values keep their sorted keys and their array order.
    @Test func toolWithoutDeclaredOrderSortsTheRest() throws {
        let tool: ToolSpec = [
            "function": [
                "strict": true,
                "parameters": [
                    "additionalProperties": false,
                    "required": ["q"],
                    "properties": [
                        "q": [
                            "type": "string",
                            "default": ["z": 1, "a": 2] as [String: any Sendable],
                        ]
                            as [String: any Sendable],
                        "limit": ["type": "integer", "enum": [3, 1, 2] as [any Sendable]]
                            as [String: any Sendable],
                    ] as [String: any Sendable],
                    "type": "object",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                ] as [String: any Sendable],
                "description": "Search.",
                "name": "search",
            ] as [String: any Sendable],
            "type": "function",
        ]
        #expect(
            try rendered([tool])
                == #"{"type": "function", "function": {"name": "search", "description": "Search.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "enum": [3, 1, 2]}, "q": {"type": "string", "default": {"a": 2, "z": 1}}}, "required": ["q"], "$schema": "https://json-schema.org/draft/2020-12/schema", "additionalProperties": false}, "strict": true}}"#
                + "\n")
    }

    /// A declared order reorders only the tool's own parameter names: a
    /// nested `properties` object, and a name the order leaves out, sort.
    @Test func declaredOrderAppliesToTheParametersOnly() throws {
        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "f",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "b": [
                            "type": "object",
                            "properties": ["y": ["type": "string"], "x": ["type": "string"]]
                                as [String: any Sendable],
                        ] as [String: any Sendable],
                        "a": ["type": "string"] as [String: any Sendable],
                        "c": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            ChatTemplateTools.propertyOrderKey: ["c", "b"],
        ]
        #expect(
            try rendered([tool])
                == #"{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"c": {"type": "string"}, "b": {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}}, "a": {"type": "string"}}}}}"#
                + "\n")
    }

    /// A JSON `null` in a schema renders as `null`, as `json.dumps(None)` does.
    @Test func nullRendersAsNull() throws {
        let tool: ToolSpec = [
            "type": "function",
            "function": ["name": "f", "default": NSNull()] as [String: any Sendable],
        ]
        #expect(
            try rendered([tool])
                == #"{"type": "function", "function": {"name": "f", "default": null}}"# + "\n")
    }
}
