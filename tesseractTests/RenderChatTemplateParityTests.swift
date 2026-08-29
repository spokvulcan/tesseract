import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// Gate 1 for experiment C25: the Layer 1/2 split is behavior-preserving —
/// `renderChatTemplate` + `encode(rendered, addSpecialTokens: false)` must
/// reproduce `applyChatTemplate` exactly on a battery of message shapes.
///
/// Asserted twice: through the HuggingFace adaptor (the app's
/// `MLXLMCommon.Tokenizer`, exercising the macro bridge's forwarding) and
/// directly on the upstream `Tokenizers.PreTrainedTokenizer` (the
/// DerivedData swift-transformers patch in isolation).
///
/// Uses the real PARO tokenizer/template from the local models directory
/// (default `~/Library/Application Support/models/z-lab_Qwen3.5-4B-PARO`,
/// overridable via `TESSERACT_TOKENIZE_CACHE_MODEL`); skipped when absent.
struct RenderChatTemplateParityTests {

    private nonisolated static var modelDirectory: URL {
        let path =
            ProcessInfo.processInfo.environment["TESSERACT_TOKENIZE_CACHE_MODEL"]
            ?? "~/Library/Application Support/models/z-lab_Qwen3.5-4B-PARO"
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private nonisolated static var modelAvailable: Bool {
        FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("tokenizer_config.json").path)
    }

    // MARK: - Battery

    private struct ParityCase {
        let name: String
        let messages: [[String: any Sendable]]
        var tools: [MLXLMCommon.ToolSpec]?
        var additionalContext: [String: any Sendable]?
    }

    private static let batteryTools: [MLXLMCommon.ToolSpec] = [
        [
            "type": "function",
            "function": [
                "name": "read_file",
                "description": "Read a file from disk.",
                "parameters": [
                    "type": "object",
                    "required": ["path"],
                    "properties": [
                        "path": ["type": "string", "description": "Absolute path."]
                            as [String: any Sendable]
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
        [
            "type": "function",
            "function": [
                "name": "search",
                "description": "Search the web.",
                "parameters": [
                    "type": "object",
                    "properties": [:] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    private static func battery() -> [ParityCase] {
        [
            ParityCase(
                name: "system+tools+multi-turn",
                messages: [
                    ["role": "system", "content": "You are a careful assistant."],
                    ["role": "user", "content": "List three facts about MLX."],
                    [
                        "role": "assistant",
                        "content":
                            "Here are three facts:\n1. It is a framework.\n2. It runs on Apple Silicon.\n3. It is open source.",
                    ],
                    ["role": "user", "content": "Expand on the second one."],
                ],
                tools: batteryTools
            ),
            ParityCase(
                name: "tool_calls + tool response",
                messages: [
                    ["role": "system", "content": "You are a careful assistant."],
                    ["role": "user", "content": "Read /tmp/notes.txt"],
                    [
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            [
                                "type": "function",
                                "function": [
                                    "name": "read_file",
                                    "arguments": ["path": "/tmp/notes.txt"]
                                        as [String: any Sendable],
                                ] as [String: any Sendable],
                            ] as [String: any Sendable]
                        ] as [any Sendable],
                    ],
                    ["role": "tool", "content": "file contents here"],
                    ["role": "assistant", "content": "The file says: file contents here."],
                    ["role": "user", "content": "Thanks. Now delete it."],
                ],
                tools: batteryTools
            ),
            ParityCase(
                name: "unicode (emoji/CJK/RTL/combining)",
                messages: [
                    ["role": "system", "content": " multilingual system 🙂🌍"],
                    [
                        "role": "user",
                        "content":
                            "Emoji: 👨‍👩‍👧‍👦🏳️‍🌈, CJK: 漢字仮名交じり文, RTL: مرحبا بالعالم, combining: e\u{0301} a\u{030A}",
                    ],
                    ["role": "assistant", "content": "回音：すべての Unicode を保持します。"],
                    [
                        "role": "user",
                        "content": "\u{200D}zero-width joiner and \u{FE0F}variation selector",
                    ],
                ]
            ),
            ParityCase(
                name: "CRLF / tabs / leading spaces",
                messages: [
                    ["role": "user", "content": "line one\r\nline two\r\n\tindented\r\n  spaced"],
                    ["role": "assistant", "content": "\r\nreply starts with CRLF"],
                    ["role": "user", "content": " trailing and leading "],
                ]
            ),
            ParityCase(
                name: "empty content strings",
                messages: [
                    ["role": "system", "content": ""],
                    ["role": "user", "content": ""],
                ]
            ),
            ParityCase(
                name: "additionalContext preserve_thinking=true",
                messages: [
                    ["role": "system", "content": "You are a careful assistant."],
                    ["role": "user", "content": "Think step by step."],
                    [
                        "role": "assistant",
                        "content": "<think>\nreasoning here\n</think>\n\nThe answer is 4.",
                    ],
                    ["role": "user", "content": "And plus two?"],
                ],
                additionalContext: ["preserve_thinking": true]
            ),
            ParityCase(
                name: "additionalContext preserve_thinking=false",
                messages: [
                    ["role": "system", "content": "You are a careful assistant."],
                    ["role": "user", "content": "Think step by step."],
                ],
                additionalContext: ["preserve_thinking": false]
            ),
            ParityCase(
                name: "single user message, no tools",
                messages: [
                    ["role": "user", "content": "Hello"]
                ]
            ),
            // Issue #439: the dropped-image shape — a text-only instance keys
            // and serves an image-bearing conversation through the C25 cache,
            // so the content-array form (`["type": "image"]` parts before the
            // text part) must hold split parity too. Built from the production
            // `promptMessages` so the shape under test cannot drift from what
            // Request Keying actually renders.
            ParityCase(
                name: "dropped-image content-array turn",
                messages: HTTPPrefixCacheConversation(
                    systemPrompt: "You are a careful assistant.",
                    messages: [
                        HTTPPrefixCacheMessage(
                            role: .user, content: "What's on this screenshot?",
                            images: [HTTPPrefixCacheImage(data: Data("opaque bytes".utf8))]
                        ),
                        .assistant(content: "A code editor."),
                        HTTPPrefixCacheMessage(role: .user, content: "Zoom into the sidebar."),
                    ]
                ).promptMessages
            ),
        ]
    }

    // MARK: - Through the HuggingFace adaptor (app tokenizer surface)

    @Test(.enabled(if: modelAvailable))
    func splitEqualsFusedThroughAdaptor() async throws {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: Self.modelDirectory)
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)

        for testCase in Self.battery() {
            let fused = try tokenizer.applyChatTemplate(
                messages: testCase.messages,
                tools: testCase.tools,
                additionalContext: testCase.additionalContext
            )
            let rendered = try rendering.renderChatTemplate(
                messages: testCase.messages,
                tools: testCase.tools,
                additionalContext: testCase.additionalContext
            )
            let split = tokenizer.encode(text: rendered, addSpecialTokens: false)
            #expect(split == fused, "adaptor split != fused on \(testCase.name)")
        }
    }

    // MARK: - Directly on the upstream swift-transformers tokenizer

    @Test(.enabled(if: modelAvailable))
    func splitEqualsFusedUpstream() async throws {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: Self.modelDirectory)

        for testCase in Self.battery() {
            let fused = try upstream.applyChatTemplate(
                messages: testCase.messages,
                tools: testCase.tools,
                additionalContext: testCase.additionalContext
            )
            let rendered = try upstream.renderChatTemplate(
                messages: testCase.messages,
                tools: testCase.tools,
                additionalContext: testCase.additionalContext
            )
            let split = upstream.encode(text: rendered, addSpecialTokens: false)
            #expect(split == fused, "upstream split != fused on \(testCase.name)")
        }
    }
}
