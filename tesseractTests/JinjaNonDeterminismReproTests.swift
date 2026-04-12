import Foundation
import Jinja
import Testing

@testable import Tesseract_Agent

/// Directly exercises swift-jinja's `Value(any:)` + `tojson` filter path with
/// realistic tool specs to pin down whether the non-determinism observed in
/// production is reproducible in isolation.
///
/// If these tests fail → canonicalization is insufficient and the bug is
/// inside swift-jinja. If they pass → the bug is elsewhere in the pipeline.
@MainActor
struct JinjaNonDeterminismReproTests {

    // MARK: - Realistic tool fixture

    /// Builds a realistic Qwen3.5-style tool with deeply nested parameters.
    /// Mirrors the "question" tool from the production OpenCode workload
    /// that was observed producing non-deterministic JSON output.
    private func makeRealisticTool() -> [String: any Sendable] {
        return [
            "type": "function" as any Sendable,
            "function": [
                "name": "question" as any Sendable,
                "description": "Use this tool when you need to ask the user a question.",
                "parameters": [
                    "$schema": "https://json-schema.org/draft/2020-12/schema" as any Sendable,
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["questions"],
                    "properties": [
                        "questions": [
                            "type": "array" as any Sendable,
                            "description": "Questions to ask",
                            "items": [
                                "type": "object" as any Sendable,
                                "additionalProperties": false,
                                "required": ["question", "header", "options"],
                                "properties": [
                                    "question": [
                                        "type": "string",
                                        "description": "Complete question",
                                    ] as [String: any Sendable],
                                    "header": [
                                        "type": "string",
                                        "description": "Very short label (max 30 chars)",
                                    ] as [String: any Sendable],
                                    "multiple": [
                                        "type": "boolean",
                                        "description": "Allow selecting multiple choices",
                                    ] as [String: any Sendable],
                                    "options": [
                                        "type": "array" as any Sendable,
                                        "description": "Available choices",
                                        "items": [
                                            "type": "object" as any Sendable,
                                            "additionalProperties": false,
                                            "required": ["label", "description"],
                                            "properties": [
                                                "label": [
                                                    "type": "string",
                                                    "description": "Display text (1-5 words, concise)",
                                                ] as [String: any Sendable],
                                                "description": [
                                                    "type": "string",
                                                    "description": "Explanation of choice",
                                                ] as [String: any Sendable],
                                            ] as [String: any Sendable],
                                        ] as [String: any Sendable],
                                    ] as [String: any Sendable],
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    // MARK: - 1. rawValueAnyProducesDeterministicJSON

    /// Baseline: without canonicalization, does `Value(any:)` + tojson-style
    /// JSON encoding produce deterministic output across multiple calls?
    @Test func rawValueAnyProducesDeterministicJSON() throws {
        let tool = makeRealisticTool()

        var outputs: Set<String> = []
        for _ in 0..<20 {
            let value = try Jinja.Value(any: tool)
            let encoder = JSONEncoder()
            encoder.outputFormatting.insert(.sortedKeys)
            let data = try encoder.encode(value)
            let json = String(data: data, encoding: .utf8) ?? ""
            outputs.insert(json)
        }

        #expect(
            outputs.count == 1,
            "Value(any:) + JSONEncoder produced \(outputs.count) distinct outputs for the same tool — swift-jinja is non-deterministic"
        )
    }

    // MARK: - 2. canonicalizedValueAnyProducesDeterministicJSON

    /// Does running `LLMActor.canonicalizeToolSpecs` first fix the determinism?
    /// If yes, the canonicalization workaround is sufficient. If no, there's
    /// non-determinism deeper in the stack.
    @Test func canonicalizedValueAnyProducesDeterministicJSON() throws {
        let tool = makeRealisticTool()

        var outputs: Set<String> = []
        for _ in 0..<20 {
            let canonical = LLMActor.canonicalizeToolSpecs([tool])
            guard let first = canonical?.first else {
                Issue.record("Canonicalization returned nil")
                return
            }
            let value = try Jinja.Value(any: first)
            let encoder = JSONEncoder()
            encoder.outputFormatting.insert(.sortedKeys)
            let data = try encoder.encode(value)
            let json = String(data: data, encoding: .utf8) ?? ""
            outputs.insert(json)
        }

        #expect(
            outputs.count == 1,
            "Canonicalization + Value(any:) + JSONEncoder produced \(outputs.count) distinct outputs — canonicalization is insufficient to stabilize swift-jinja"
        )
    }

    // MARK: - 3. toolsArrayMappedAsValueIsDeterministic

    /// The tokenizer's actual call path is `tools.map { Value(any: $0) }`
    /// followed by `.array(values)`. Test this exact path.
    @Test func toolsArrayMappedAsValueIsDeterministic() throws {
        let tools = (1...5).map { idx -> [String: any Sendable] in
            var t = makeRealisticTool()
            if var fn = t["function"] as? [String: any Sendable] {
                fn["name"] = "tool\(idx)"
                t["function"] = fn
            }
            return t
        }

        var outputs: Set<String> = []
        for _ in 0..<10 {
            let canonical = LLMActor.canonicalizeToolSpecs(tools) ?? []
            let values = try canonical.map { try Jinja.Value(any: $0) }
            let arrayValue = Jinja.Value.array(values)
            let encoder = JSONEncoder()
            encoder.outputFormatting.insert(.sortedKeys)
            let data = try encoder.encode(arrayValue)
            outputs.insert(String(data: data, encoding: .utf8) ?? "")
        }

        #expect(
            outputs.count == 1,
            "Tools array → Value conversion produced \(outputs.count) distinct JSON outputs"
        )
    }

    // MARK: - 4. valueAnyKeyOrderingIsSorted

    /// Verify that `Value(any:)` actually produces sorted keys in its output.
    /// If this fails, swift-jinja isn't doing what the comment in Value.swift:58
    /// claims it does.
    @Test func valueAnyKeyOrderingIsSorted() throws {
        let dict: [String: any Sendable] = [
            "zebra": "z" as any Sendable,
            "alpha": "a",
            "mango": "m",
            "kilo": "k",
        ]

        let value = try Jinja.Value(any: dict)

        let encoder = JSONEncoder()
        encoder.outputFormatting.insert(.sortedKeys)
        let data = try encoder.encode(value)
        let json = String(data: data, encoding: .utf8) ?? ""

        // Expect alphabetical order in the JSON output.
        let alphaIdx = json.range(of: "alpha")?.lowerBound
        let kiloIdx = json.range(of: "kilo")?.lowerBound
        let mangoIdx = json.range(of: "mango")?.lowerBound
        let zebraIdx = json.range(of: "zebra")?.lowerBound

        #expect(alphaIdx != nil && kiloIdx != nil && mangoIdx != nil && zebraIdx != nil)
        if let a = alphaIdx, let k = kiloIdx, let m = mangoIdx, let z = zebraIdx {
            #expect(a < k, "alpha should come before kilo")
            #expect(k < m, "kilo should come before mango")
            #expect(m < z, "mango should come before zebra")
        }
    }

    // MARK: - 5. productionFlowFromJSONThroughJinja

    /// Mirrors the ACTUAL production code path:
    /// 1. Parse JSON request body → OpenAI.ToolDefinition
    /// 2. MessageConverter.convertToolDefinitions → ToolSpec via AnyCodableValue.toSendable()
    /// 3. LLMActor.canonicalizeToolSpecs
    /// 4. Jinja.Value(any:) → JSONEncoder
    ///
    /// If this test fails, the non-determinism is in one of steps 2-4.
    @Test func productionFlowFromJSONThroughJinja() throws {
        // A realistic tool definition as JSON (what OpenCode sends).
        let toolJSON = #"""
        {
            "type": "function",
            "function": {
                "name": "question",
                "description": "Use this tool when you need to ask the user a question.",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["questions"],
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "Questions to ask",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["question", "header", "options"],
                                "properties": {
                                    "question": {"type": "string", "description": "Complete question"},
                                    "header": {"type": "string", "description": "Very short label"},
                                    "multiple": {"type": "boolean", "description": "Allow multiple choices"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["label", "description"],
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """#

        var rawJSONOutputs: Set<String> = []
        var canonicalJSONOutputs: Set<String> = []

        for iteration in 0..<20 {
            // Step 1: decode JSON → OpenAI.ToolDefinition (fresh parse each time,
            // simulating a new HTTP request)
            let data = toolJSON.data(using: .utf8)!
            let decoded = try JSONDecoder().decode(OpenAI.ToolDefinition.self, from: data)

            // Step 2: convertToolDefinitions produces [ToolSpec]
            let toolSpecs = MessageConverter.convertToolDefinitions([decoded])
            #expect(toolSpecs?.count == 1, "iteration \(iteration): conversion failed")

            // Step 3a: raw path (no canonicalization) through Jinja
            let rawValue = try Jinja.Value(any: toolSpecs![0])
            let encoder = JSONEncoder()
            encoder.outputFormatting.insert(.sortedKeys)
            let rawData = try encoder.encode(rawValue)
            rawJSONOutputs.insert(String(data: rawData, encoding: .utf8) ?? "")

            // Step 3b: canonicalized path through Jinja
            let canonical = LLMActor.canonicalizeToolSpecs(toolSpecs)!
            let canonicalValue = try Jinja.Value(any: canonical[0])
            let canonicalData = try encoder.encode(canonicalValue)
            canonicalJSONOutputs.insert(String(data: canonicalData, encoding: .utf8) ?? "")
        }

        // If the raw path is non-deterministic, rawJSONOutputs.count > 1.
        // If canonicalization fixes it, canonicalJSONOutputs.count == 1.
        #expect(
            rawJSONOutputs.count == 1,
            "Raw path produced \(rawJSONOutputs.count) distinct JSON outputs — pipeline is non-deterministic"
        )
        #expect(
            canonicalJSONOutputs.count == 1,
            "Canonicalized path produced \(canonicalJSONOutputs.count) distinct JSON outputs — canonicalization insufficient"
        )
    }

    // MARK: - 7. templateRenderWithTojsonIsDeterministic

    /// The CLOSEST reproduction of production: actually renders a Jinja template
    /// that uses `{{ tool | tojson }}` (matching the Qwen3.5 chat template).
    /// If this fails, the bug is in the Jinja interpreter or tojson filter.
    /// If this passes but production still shows non-determinism, the bug is in
    /// message rendering or some state outside the tools pipeline.
    @Test func templateRenderWithTojsonIsDeterministic() throws {
        // A minimal template that mirrors the Qwen3.5 tools block render.
        let templateString = """
        {% for tool in tools %}{{ tool | tojson }}
        {% endfor %}
        """

        let toolJSON = #"""
        {
            "type": "function",
            "function": {
                "name": "question",
                "description": "Use this tool when you need to ask the user a question.",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["questions"],
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "Questions to ask",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["question", "header", "options"],
                                "properties": {
                                    "question": {"type": "string"},
                                    "header": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """#

        let template = try Jinja.Template(templateString)
        var renderedOutputs: Set<String> = []

        for _ in 0..<20 {
            // Parse JSON fresh each time to mimic production per-request parse.
            let data = toolJSON.data(using: .utf8)!
            let decoded = try JSONDecoder().decode(OpenAI.ToolDefinition.self, from: data)
            let toolSpecs = MessageConverter.convertToolDefinitions([decoded])!

            // Convert to Jinja context.
            let context: [String: Jinja.Value] = try [
                "tools": .array(toolSpecs.map { try Jinja.Value(any: $0) }),
            ]

            // Render through the actual Jinja template.
            let rendered = try template.render(context)
            renderedOutputs.insert(rendered)
        }

        #expect(
            renderedOutputs.count == 1,
            "Template render with tojson produced \(renderedOutputs.count) distinct outputs. Bug is in the Jinja interpreter / filter pipeline."
        )
    }

    // MARK: - 8. convertToolDefinitionsIsDeterministic

    /// Narrower test: does `MessageConverter.convertToolDefinitions` itself
    /// produce deterministic output across invocations with the same input JSON?
    @Test func convertToolDefinitionsIsDeterministic() throws {
        let toolJSON = #"""
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alpha": {"type": "string"},
                        "zebra": {"type": "integer"},
                        "mango": {"type": "boolean"}
                    }
                }
            }
        }
        """#

        var outputs: Set<String> = []
        for _ in 0..<20 {
            let data = toolJSON.data(using: .utf8)!
            let decoded = try JSONDecoder().decode(OpenAI.ToolDefinition.self, from: data)
            let toolSpecs = MessageConverter.convertToolDefinitions([decoded])!

            // Encode via JSONSerialization with sortedKeys — any non-determinism
            // in the Swift dict storage layout won't affect sorted output.
            let jsonData = try JSONSerialization.data(
                withJSONObject: toolSpecs[0],
                options: [.sortedKeys]
            )
            outputs.insert(String(data: jsonData, encoding: .utf8) ?? "")
        }

        #expect(
            outputs.count == 1,
            "convertToolDefinitions produced \(outputs.count) distinct serializations (sortedKeys): the AnyCodableValue → Sendable conversion is non-deterministic"
        )
    }
}
