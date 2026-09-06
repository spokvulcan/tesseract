import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The fidelity gate on Emitted Path registration (ADR-0063 decision 9),
/// pinned without a model: emitted ids are detokenized and replayed through
/// the server's own stream pipeline, and the reconstruction must equal the
/// stored assistant message under the stored message's normalization —
/// trimmed content and reasoning, tool-call arguments as parsed JSON.
struct EmittedPathFidelityTests {

    /// The think and tool-call tags are whole pieces, as they are special
    /// tokens on the real tokenizers: the stream pipeline sees each tag in
    /// one chunk, never scalar by scalar.
    private let tokenizer = GreedyTokenizer(
        pieces: chatMLGreedyPieces + ["<think>", "</think>", "<tool_call>", "</tool_call>"])

    private func ids(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    private func check(
        _ emitted: String,
        format: ToolCallFormat = .xmlFunction,
        tools: [ToolSpec]? = nil,
        startsInsideThinkBlock: Bool = true,
        stored: HTTPPrefixCacheMessage
    ) -> EmittedPathFidelity.Verdict {
        EmittedPathFidelity.check(
            contentIDs: ids(emitted), tokenizer: tokenizer, toolCallFormat: format,
            tools: tools, startsInsideThinkBlock: startsInsideThinkBlock, stored: stored)
    }

    /// A `read` tool whose `limit` is an integer, so the XML parser converts
    /// the parameter by schema exactly as it does on the live stream.
    private static let readTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "read",
            "parameters": [
                "type": "object",
                "properties": [
                    "path": ["type": "string"],
                    "limit": ["type": "integer"],
                    "filters": ["type": "object"],
                ],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    // MARK: - Thinking and content

    @Test func thinkingThenContentReconstructsTheStoredMessage() {
        let verdict = check(
            "Let me think.\n</think>\n\nHello world\n",
            stored: .assistant(content: "Hello world", reasoning: "Let me think."))
        #expect(verdict == .match)
    }

    @Test func surroundingWhitespaceIsTemplateNormalization() {
        let verdict = check(
            "  Let me think. \n\n</think>\n\n\n Hello world  \n\n",
            stored: .assistant(content: "Hello world", reasoning: "Let me think."))
        #expect(verdict == .match)
    }

    @Test func differentContentIsAContentMismatchWithTheFirstDifference() {
        let verdict = check(
            "Let me think.\n</think>\n\nHello there",
            stored: .assistant(content: "Hello world", reasoning: "Let me think."))
        #expect(
            verdict
                == .mismatch(
                    EmittedPathFidelity.Mismatch(
                        field: .content, emittedLength: 11, storedLength: 11,
                        firstDifference: 6)))
    }

    @Test func differentThinkingIsAThinkingMismatch() {
        let verdict = check(
            "Let me ponder.\n</think>\n\nHello world",
            stored: .assistant(content: "Hello world", reasoning: "Let me think."))
        guard case .mismatch(let mismatch) = verdict else {
            Issue.record("expected a mismatch")
            return
        }
        #expect(mismatch.field == .thinking)
        #expect(mismatch.firstDifference == 7)
    }

    @Test func anUnclosedThinkBlockReclassifiesAsContentOnBothSides() {
        // The accumulator's rule: a `<think>` that never closed is text.
        let verdict = check(
            "only thinking, never closed",
            stored: .assistant(content: "only thinking, never closed"))
        #expect(verdict == .match)
    }

    @Test func aRequestOutsideAThinkBlockIsPlainContent() {
        let verdict = check(
            "Hello", startsInsideThinkBlock: false,
            stored: .assistant(content: "Hello"))
        #expect(verdict == .match)
    }

    @Test func aStrayThinkCloseWithoutOpenerLeavesTheStreamedPrefixAsContent() {
        // enable_thinking=false prompts close the block in the prompt; a
        // model that still emits `</think>` finds its prefix already
        // streamed as text (chunk by chunk, nothing is buffered behind the
        // stray tag), so the server stored it as content — and so does the
        // replay, which runs the same chunks.
        let verdict = check(
            "hmm</think>answer", startsInsideThinkBlock: false,
            stored: .assistant(content: "hmmanswer"))
        #expect(verdict == .match)
    }

    // MARK: - Tool calls

    private let toolCallTurn = """
        I'll read it.
        </think>

        Reading.
        <tool_call>
        <function=read>
        <parameter=path>
        /tmp/a
        </parameter>
        <parameter=limit>
        10
        </parameter>
        </function>
        </tool_call>
        """

    @Test func anXMLToolCallMatchesUnderParsedJSONEquality() {
        // The stored call spells its arguments spaced and in another
        // order; equality is on the parsed values.
        let verdict = check(
            toolCallTurn, tools: [Self.readTool],
            stored: .assistant(
                content: "Reading.", reasoning: "I'll read it.",
                toolCalls: [
                    HTTPPrefixCacheToolCall(
                        name: "read", argumentsJSON: "{\"path\": \"/tmp/a\", \"limit\": 10}")
                ]))
        #expect(verdict == .match)
    }

    @Test func escapedStoredArgumentsMatchTheUnescapedEmittedSpelling() {
        // The escaping classes: the template's `tojson` spells a slash as
        // `\/` and a non-ASCII scalar as `\uXXXX`; the model emits the
        // characters themselves. Equality is on the parsed values.
        let emitted = """
            Filtering.
            </think>

            <tool_call>
            <function=read>
            <parameter=path>
            /tmp/a
            </parameter>
            <parameter=filters>
            {"glob": "src/*.swift", "note": "ä"}
            </parameter>
            </function>
            </tool_call>
            """
        let verdict = check(
            emitted, tools: [Self.readTool],
            stored: .assistant(
                content: "", reasoning: "Filtering.",
                toolCalls: [
                    HTTPPrefixCacheToolCall(
                        name: "read",
                        argumentsJSON:
                            "{\"path\": \"\\/tmp\\/a\", \"filters\": {\"glob\": \"src\\/*.swift\", \"note\": \"\\u00e4\"}}"
                    )
                ]))
        #expect(verdict == .match)
    }

    @Test func aDifferentArgumentIsAToolCallMismatch() {
        let verdict = check(
            toolCallTurn, tools: [Self.readTool],
            stored: .assistant(
                content: "Reading.", reasoning: "I'll read it.",
                toolCalls: [
                    HTTPPrefixCacheToolCall(
                        name: "read", argumentsJSON: "{\"path\": \"/tmp/b\", \"limit\": 10}")
                ]))
        #expect(
            verdict
                == .mismatch(
                    EmittedPathFidelity.Mismatch(
                        field: .toolCalls, emittedLength: 1, storedLength: 1, firstDifference: 0))
        )
    }

    @Test func aMissingToolCallIsAToolCallMismatch() {
        let verdict = check(
            toolCallTurn, tools: [Self.readTool],
            stored: .assistant(content: "Reading.", reasoning: "I'll read it."))
        #expect(
            verdict
                == .mismatch(
                    EmittedPathFidelity.Mismatch(
                        field: .toolCalls, emittedLength: 1, storedLength: 0, firstDifference: 0))
        )
    }

    @Test func aStringParameterKeepsItsSpacingAndTheIntegerConvertsBySchema() {
        let replayed = EmittedPathFidelity.replay(
            contentIDs: ids(toolCallTurn), tokenizer: tokenizer, toolCallFormat: .xmlFunction,
            tools: [Self.readTool], startsInsideThinkBlock: true)
        #expect(
            replayed.toolCalls == [
                HTTPPrefixCacheToolCall(
                    name: "read", argumentsJSON: "{\"limit\": 10, \"path\": \"/tmp/a\"}")
            ])
        // The integer converted by schema, not kept as the string "10".
        #expect(replayed.toolCalls.first?.argumentsJSON.contains("\"limit\":10") == true)
        #expect(replayed.content == "Reading.")
        #expect(replayed.reasoning == "I'll read it.")
    }
}
