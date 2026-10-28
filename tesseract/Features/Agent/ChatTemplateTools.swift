import Foundation
import Jinja
import MLXLMCommon

/// The tools as a chat template should see them: Jinja objects whose keys
/// come in the order a Python caller writes them, so `tool | tojson` renders
/// the text the models were trained on.
///
/// A `ToolSpec` is a Swift dictionary and carries no key order, and the
/// tokenizer's own conversion sorts every key: `{"function": {"description":
/// …, "name": …}, "type": "function"}`. transformers renders the dictionary it
/// was given, and the builders of OpenAI-style tools (transformers'
/// `get_json_schema`, the OpenAI SDK, `AgentToolDefinition.toolSpec`) write
/// `type` before `function`, `name` before `description` before
/// `parameters`, and `type` before `properties` before `required`. Those keys
/// come first here, in that order; any other key follows, sorted. Parameter
/// names keep the order the tool declared them in when the tool carries it
/// under ``propertyOrderKey``, and are sorted otherwise, as data values
/// (`enum`, `default`, …) are. The order is a function of the tool alone, so a
/// tool renders the same text on every launch.
///
/// On Qwen3.8-27B with the app's file, profile and agenda tools, a sorted,
/// compact render cost 0.17 nats on the called function's name against
/// transformers' render, and sorted parameters made the model fill them
/// alphabetically: `add_event(duration_minutes:notes:start:title:)` for a
/// block given a title, a start and an end (ADR-0098).
nonisolated enum ChatTemplateTools {
    /// The key a `ToolSpec` carries its parameters' declared order under
    /// (`AgentToolDefinition.toolSpec`), since its `properties` dictionary has
    /// none. The render reads and drops it, so a template never sees it; it is
    /// plain JSON, so every digest of the tools covers it.
    static let propertyOrderKey = "x-tesseract-property-order"

    /// The keys OpenAI-style builders write, in their order: the tool, its
    /// function, and the JSON Schema keywords of its parameters.
    private static let conventionalKeyOrder = [
        "type", "function", "name", "description", "parameters", "properties", "required",
        "enum", "items",
    ]

    /// Schema keywords whose values are data, not schemas.
    private static let dataKeywords: Set<String> = ["const", "default", "enum", "examples"]

    /// Where ``propertyOrderKey`` applies: the tool's parameter names.
    private static let parameterNamesPath = ["function", "parameters", "properties"]

    static func value(_ tools: [ToolSpec]) throws -> Jinja.Value {
        try .array(
            tools.map { tool in
                var tool = tool
                let order = tool.removeValue(forKey: propertyOrderKey) as? [String] ?? []
                return try value(tool, as: .schema, at: [], parameterOrder: order)
            })
    }

    private enum Role {
        /// A tool, a function, or a JSON Schema: conventional keys first.
        case schema
        /// A `properties` object, keyed by parameter name.
        case propertyNames
        /// A value the schema holds as data.
        case data
    }

    private static func value(
        _ any: Any, as role: Role, at path: [String], parameterOrder: [String]
    ) throws -> Jinja.Value {
        if any is NSNull { return .null }
        if let dictionary = any as? [String: Any] {
            let order = path == parameterNamesPath ? parameterOrder : []
            var object = OrderedDictionary<String, Jinja.Value>()
            for key in orderedKeys(of: dictionary, as: role, declared: order) {
                guard let child = dictionary[key] else { continue }
                object[key] = try value(
                    child, as: childRole(of: key, in: role), at: path + [key],
                    parameterOrder: parameterOrder)
            }
            return .object(object)
        }
        if let array = any as? [Any] {
            return try .array(array.map { try value($0, as: role, at: path, parameterOrder: []) })
        }
        return try Jinja.Value(any: any)
    }

    private static func orderedKeys(
        of dictionary: [String: Any], as role: Role, declared: [String]
    ) -> [String] {
        let sorted = dictionary.keys.sorted()
        switch role {
        case .schema:
            let conventional = conventionalKeyOrder.filter { dictionary[$0] != nil }
            return conventional + sorted.filter { !conventionalKeyOrder.contains($0) }
        case .propertyNames:
            let declared = declared.filter { dictionary[$0] != nil }
            return declared + sorted.filter { !declared.contains($0) }
        case .data:
            return sorted
        }
    }

    private static func childRole(of key: String, in role: Role) -> Role {
        switch role {
        case .propertyNames: .schema
        case .data: .data
        case .schema:
            if key == "properties" {
                .propertyNames
            } else if dataKeywords.contains(key) {
                .data
            } else {
                .schema
            }
        }
    }
}
