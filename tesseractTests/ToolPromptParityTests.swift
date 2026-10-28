import CryptoKit
import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Tool prompts render token for token as Hugging Face `transformers` renders
/// them, on each chat-template family the app ships: the tools block (the
/// models were trained on `json.dumps` with its `", "` / `": "` separators
/// and the tool's own key order) and tool calls whose arguments hold arrays
/// and objects.
///
/// The tools are built through `AgentToolDefinition.toolSpec` and rendered
/// through `AppTokenizerLoader`, the production path; the reference is
/// `Fixtures/ToolPromptParity/transformers.json`, written by `generate.py`
/// beside it from the same tools as Python dicts. Each family runs against a
/// local model directory and is skipped when none is present.
@Suite(.cpuBound)
struct ToolPromptParityTests {

    // MARK: - Model directories

    private nonisolated static let modelsRoot = URL(
        fileURLWithPath: NSString(string: "~/Library/Application Support/models")
            .expandingTildeInPath)

    /// Directories that hold each family's template, in order of preference:
    /// the app's models folder, then the Hugging Face hub cache.
    private nonisolated static let candidates: [String: [String]] = [
        "qwen3.5": ["z-lab_Qwen3.5-4B-PARO", "z-lab_Qwen3.5-9B-PARO"],
        "qwen3.8": ["z-lab_Qwen3.8-27B-PARO", "mlx-community_Qwen3.8-27B-4bit"],
        "lfm3": ["LiquidAI_LFM3-1.7B-MLX-4bit", "hub:LiquidAI/LFM3-1.7B-MLX-4bit"],
        "lfm3-audio": ["LiquidAI_LFM3-Audio-2B-MLX-4bit", "hub:LiquidAI/LFM3-Audio-2B-MLX-4bit"],
    ]

    private nonisolated static func directory(_ family: String) -> URL? {
        for candidate in candidates[family] ?? [] {
            let url: URL?
            if candidate.hasPrefix("hub:") {
                url = hubSnapshot(String(candidate.dropFirst(4)))
            } else {
                url = modelsRoot.appendingPathComponent(candidate)
            }
            if let url,
                FileManager.default.fileExists(
                    atPath: url.appendingPathComponent("chat_template.jinja").path),
                FileManager.default.fileExists(
                    atPath: url.appendingPathComponent("tokenizer.json").path)
            {
                return url
            }
        }
        return nil
    }

    private nonisolated static func hubSnapshot(_ repo: String) -> URL? {
        let snapshots = URL(
            fileURLWithPath: NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        )
        .appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
        .appendingPathComponent("snapshots")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: snapshots.path)) ?? []
        return names.sorted().last.map { snapshots.appendingPathComponent($0) }
    }

    // MARK: - Reference

    private struct Reference: Decodable {
        struct Render: Decodable {
            let text: String
            let ids: [Int]
        }
        struct Family: Decodable {
            // swiftlint:disable:next identifier_name
            let template_sha256: String
            let cases: [String: Render]
        }
        let renders: [String: Family]
    }

    private nonisolated static func reference() throws -> Reference {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/ToolPromptParity/transformers.json")
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    // MARK: - The tools and conversations `generate.py` renders

    private nonisolated static func definition(
        _ name: String, _ description: String,
        _ properties: KeyValuePairs<String, PropertySchema>, required: [String]
    ) -> AgentToolDefinition {
        AgentToolDefinition(
            name: name, label: name, description: description,
            parameterSchema: JSONSchema(type: "object", properties: properties, required: required),
            execute: { _, _, _, _ in .text("") })
    }

    private nonisolated static var tools: [ToolSpec] {
        [
            definition(
                "get_weather", "Weather for a city.",
                ["city": PropertySchema(type: "string", description: "The city's name.")],
                required: ["city"]),
            definition(
                "read", "Read the contents of a file. Returns numbered lines (cat -n style).",
                [
                    "path": PropertySchema(
                        type: "string", description: "File path relative to working directory"),
                    "offset": PropertySchema(
                        type: "integer", description: "Line number to start from (1-indexed)"),
                    "limit": PropertySchema(
                        type: "integer", description: "Maximum number of lines to return"),
                ],
                required: ["path"]),
            definition(
                "edit", "Edit a file by replacing an exact text match.",
                [
                    "path": PropertySchema(type: "string", description: "File path to edit"),
                    "old_text": PropertySchema(
                        type: "string", description: "Exact text to find and replace"),
                    "new_text": PropertySchema(type: "string", description: "Replacement text"),
                ],
                required: ["path", "old_text", "new_text"]),
            definition(
                "search_notes",
                "Search notes — by title, tag or text such as </style> or \"quoted words\".",
                [
                    "query": PropertySchema(type: "string", description: "Text to find, e.g. café"),
                    "tags": PropertySchema(
                        type: "array", description: "Tags to match",
                        items: PropertySchema(
                            type: "string", description: "A tag", enumValues: ["work", "home"])),
                    "sort": PropertySchema(
                        type: "string", description: "Result order",
                        enumValues: ["newest", "oldest"]),
                    "limit": PropertySchema(
                        type: "integer", description: "At most this many notes"),
                    "archived": PropertySchema(
                        type: "boolean", description: "Include archived notes"),
                ],
                required: ["query"]),
        ].map(\.toolSpec)
    }

    private nonisolated static func toolCall(
        _ name: String, _ arguments: [String: any Sendable]
    ) -> [String: any Sendable] {
        [
            "role": "assistant", "content": "",
            "tool_calls": [
                [
                    "type": "function",
                    "function": ["name": name, "arguments": arguments] as [String: any Sendable],
                ] as [String: any Sendable]
            ] as [any Sendable],
        ]
    }

    private nonisolated static let system = "You are a careful assistant."

    private nonisolated static var cases: [(name: String, messages: [[String: any Sendable]])] {
        [
            (
                "tools",
                [
                    ["role": "system", "content": system],
                    ["role": "user", "content": "What's the weather in Paris?"],
                ]
            ),
            (
                "tool_call_history",
                [
                    ["role": "system", "content": system],
                    [
                        "role": "user",
                        "content": "Find my newest café notes tagged work or home, five at most.",
                    ],
                    toolCall(
                        "search_notes",
                        [
                            "limit": 5, "query": "café", "sort": "newest",
                            "tags": ["work", "home"] as [any Sendable],
                        ]),
                    ["role": "tool", "content": "2 notes: Lyon café, Café crawl"],
                ]
            ),
            (
                "nested_argument",
                [
                    ["role": "user", "content": "Note the café filter."],
                    toolCall(
                        "search_notes",
                        [
                            "query": ["after": "2026-01-01", "text": "café"]
                                as [String: any Sendable]
                        ]),
                    ["role": "tool", "content": "ok"],
                ]
            ),
        ]
    }

    // MARK: - Tests

    @Test(.enabled(if: directory("qwen3.5") != nil)) func qwen35() async throws {
        try await expectParity("qwen3.5")
    }

    @Test(.enabled(if: directory("qwen3.8") != nil)) func qwen38() async throws {
        try await expectParity("qwen3.8")
    }

    @Test(.enabled(if: directory("lfm3") != nil)) func lfm3() async throws {
        try await expectParity("lfm3")
    }

    @Test(.enabled(if: directory("lfm3-audio") != nil)) func lfm3Audio() async throws {
        try await expectParity("lfm3-audio")
    }

    /// The Swift tools carry the content `generate.py` rendered, so a render
    /// mismatch above is about rendering, not about a mistyped fixture.
    @Test func toolsMatchTheReferenceContent() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/ToolPromptParity/transformers.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let referenceTools = try #require((object as? [String: Any])?["tools"])
        func canonical(_ value: Any) throws -> Data {
            try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        }
        let content = Self.tools.map { $0.filter { $0.key != ChatTemplateTools.propertyOrderKey } }
        #expect(try canonical(content) == canonical(referenceTools))
    }

    private func expectParity(_ family: String) async throws {
        let directory = try #require(Self.directory(family))
        let reference = try #require(try Self.reference().renders[family])
        let template = try String(
            contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let sha = SHA256.hash(data: Data(template.utf8)).map { String(format: "%02x", $0) }
            .joined()
        try #require(
            sha == reference.template_sha256,
            "\(directory.lastPathComponent) has another chat template than the reference: regenerate Fixtures/ToolPromptParity/transformers.json"
        )

        let tokenizer = try await RealTokenizers.app(from: directory)
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        // The tools as every production render receives them.
        let tools = LLMActor.canonicalizeToolSpecs(Self.tools)
        for (name, messages) in Self.cases {
            let expected = try #require(reference.cases[name])
            let text = try rendering.renderChatTemplate(
                messages: messages, tools: tools, additionalContext: nil)
            let sameText = text == expected.text
            #expect(
                sameText,
                "\(family)/\(name) differs from transformers at \(Self.firstDifference(text, expected.text))"
            )
            let ids = try tokenizer.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: nil)
            let sameIDs = ids == expected.ids
            let divergence =
                zip(ids, expected.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? min(ids.count, expected.ids.count)
            #expect(
                sameIDs,
                "\(family)/\(name): \(ids.count) tokens, transformers \(expected.ids.count); first different token at \(divergence)"
            )
        }
    }

    /// The line where two renders part, with what each side has there.
    private static func firstDifference(_ actual: String, _ expected: String) -> String {
        let actualLines = actual.components(separatedBy: "\n")
        let expectedLines = expected.components(separatedBy: "\n")
        for (index, (a, e)) in zip(actualLines, expectedLines).enumerated() where a != e {
            return
                "line \(index + 1)\n  app:          \(a.prefix(300))\n  transformers: \(e.prefix(300))"
        }
        return "line \(min(actualLines.count, expectedLines.count) + 1) (one render is longer)"
    }
}
