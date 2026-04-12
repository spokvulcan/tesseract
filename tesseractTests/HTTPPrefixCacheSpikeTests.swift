import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

/// Regression tests for the normalization layer and the free helper functions
/// that `LLMActor` still uses for cache-state inspection.
///
/// **Historical note:** this file used to exercise the legacy
/// `HTTPPrefixCacheSpikeStore` (an in-memory conversation-keyed prefix cache
/// that was the precursor to the radix tree). Task 1.7 of the Marconi plan
/// retired the store itself; these tests now cover only the pieces that are
/// still in production use:
///
/// - Normalization of `HTTPPrefixCacheMessage` content, reasoning, tool call
///   arguments, and `HTTPPrefixCacheConversation.systemPrompt` (trimming,
///   whitespace-only → "", reasoning `nil`-ification, etc.).
/// - `HTTPPrefixCacheConversation.isPrefix(of:)` as the canonical
///   conversation-level prefix check (used by `HTTPPrefixCacheSessionReplayStore`
///   and by round-trip regression tests against `MessageConverter`).
/// - Free functions `httpPrefixCacheReportedTokenCount` /
///   `httpPrefixCacheHasReusableState`, which `LLMActor` uses to decide
///   whether the finalized KV cache is worth storing.
///
/// File name is kept for git history continuity. Contents are no longer
/// "spike"-specific — the spike is gone.
struct HTTPPrefixCacheSpikeTests {

    // MARK: - Free helper functions (used by LLMActor)

    @Test func mixedCacheOffsetsUseMaximumReportedTokenCount() {
        let mamba = MambaCache()
        mamba.state = [MLXArray([1])]

        let attention = KVCacheSimple()
        attention.offset = 42

        #expect(httpPrefixCacheReportedTokenCount([mamba, attention]) == 42)
        #expect(httpPrefixCacheHasReusableState([mamba, attention]))
    }

    @Test func zeroOffsetCachesWithStateRemainReusable() {
        let mamba = MambaCache()
        mamba.state = [MLXArray([1])]

        #expect(httpPrefixCacheReportedTokenCount([mamba]) == 0)
        #expect(httpPrefixCacheHasReusableState([mamba]))
    }

    // MARK: - HTTPPrefixCacheMessage normalization

    @Test func assistantContentIsTrimmed() {
        let twoNewlines = HTTPPrefixCacheMessage.assistant(content: "\n\n")
        #expect(twoNewlines.content == "")

        let threeNewlines = HTTPPrefixCacheMessage.assistant(content: "\n\n\n")
        #expect(threeNewlines.content == "")

        let mixedWhitespace = HTTPPrefixCacheMessage.assistant(content: "  \n  \t ")
        #expect(mixedWhitespace.content == "")

        let empty = HTTPPrefixCacheMessage.assistant(content: "")
        #expect(empty.content == "")

        let realText = HTTPPrefixCacheMessage.assistant(content: "Hello!")
        #expect(realText.content == "Hello!")

        // Trailing whitespace on real content is also trimmed — OpenCode strips
        // it when echoing back, so we must too or the prefix cache won't match.
        let textWithSurroundingWhitespace = HTTPPrefixCacheMessage.assistant(content: "  Hi  ")
        #expect(textWithSurroundingWhitespace.content == "Hi")

        let textWithTrailingNewlines = HTTPPrefixCacheMessage.assistant(
            content: "\n\nNow let me read the source files\n\n\n\n"
        )
        #expect(textWithTrailingNewlines.content == "Now let me read the source files")
    }

    /// User and tool messages must NOT be normalized — their content is real data.
    @Test func userAndToolContentIsPreservedVerbatim() {
        let userWhitespace = HTTPPrefixCacheMessage(role: .user, content: "\n\n")
        #expect(userWhitespace.content == "\n\n")

        let toolWhitespace = HTTPPrefixCacheMessage(role: .tool, content: "  ")
        #expect(toolWhitespace.content == "  ")
    }

    @Test func reasoningIsTrimmedAtConstruction() {
        let trailingNewline = HTTPPrefixCacheMessage.assistant(
            content: "Hello",
            reasoning: "Some thought.\n"
        )
        #expect(trailingNewline.reasoning == "Some thought.")

        let surroundingWhitespace = HTTPPrefixCacheMessage.assistant(
            content: "Hello",
            reasoning: "  Some thought.  "
        )
        #expect(surroundingWhitespace.reasoning == "Some thought.")

        let whitespaceOnly = HTTPPrefixCacheMessage.assistant(
            content: "Hello",
            reasoning: "  \n\n  "
        )
        #expect(whitespaceOnly.reasoning == nil)

        let empty = HTTPPrefixCacheMessage.assistant(content: "Hello", reasoning: "")
        #expect(empty.reasoning == nil)

        let nilReasoning = HTTPPrefixCacheMessage.assistant(content: "Hello", reasoning: nil)
        #expect(nilReasoning.reasoning == nil)
    }

    // MARK: - HTTPPrefixCacheConversation.isPrefix() contracts

    /// Phase 8b regression: subagent flow stored assistant content
    /// `"\n\nNow let me read the source files…\n\n\n\n"` (82 chars) but OpenCode
    /// echoed it back as `"\n\nNow let me read the source files…"` (78 chars,
    /// 4 trailing newlines stripped). Trimming both sides at construction makes
    /// them equal and `isPrefix` succeeds.
    @Test func subagentTrailingWhitespaceContentMatchesEcho() {
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(content: "Now let me read the source files\n\n\n\n"),
            ]
        )
        let echoed = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(content: "Now let me read the source files"),
                .init(role: .user, content: "next"),
            ]
        )

        #expect(stored.isPrefix(of: echoed))
    }

    /// Reproduces the failure mode the user observed: server stores assistant turn
    /// with whitespace content (model emitted `"\n\n"` between `</think>` and
    /// `<tool_call>`), client (OpenCode) echoes it back with empty content. The
    /// stored entry should still prefix-match the next request because both sides
    /// normalize whitespace-only content to "".
    @Test func storedTurnWithWhitespaceContentMatchesEmptyEcho() {
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(
                    content: "\n\n",  // model emitted whitespace
                    reasoning: "Need to inspect.",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"a.swift"}"#)]
                ),
            ]
        )
        let echoedRequest = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(
                    content: "",  // OpenCode stripped the whitespace
                    reasoning: "Need to inspect.",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"a.swift"}"#)]
                ),
                .init(role: .tool, content: "file contents"),
                .init(role: .user, content: "What's next?"),
            ]
        )

        #expect(stored.isPrefix(of: echoedRequest))
    }

    /// Assistant turns with tool calls must still prefix-match when the request
    /// extends the conversation with a tool response. Before normalization was
    /// pushed into `HTTPPrefixCacheMessage.init`, the tool call arguments would
    /// hash differently between stored and echoed forms.
    @Test func assistantToolCallTurnsCanBeRestoredAsPrefixes() {
        let assistantTurn = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(
                    content: "Calling tool",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "glob", argumentsJSON: #"{"pattern":"*.swift"}"#)]
                ),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: assistantTurn.messages + [
                .init(role: .tool, content: "Main.swift"),
            ]
        )

        #expect(assistantTurn.isPrefix(of: request))
    }

    /// Round-trip regression: build the assistant message the way `LLMActor`
    /// stores it after generation, then build the message OpenCode echoes via
    /// JSON, run it through `MessageConverter.analyzePrefixCacheEligibility`,
    /// and assert that the stored conversation is a prefix of the new request.
    @MainActor
    @Test func storedAssistantTurnMatchesOpenCodeEcho() throws {
        let systemPrompt = "You are a helpful assistant."
        let toolName = "read"
        let argumentsDict: [String: any Sendable] = ["path": "main.swift", "limit": 100]

        // ── STORE side: build what LLMActor stores after generation ─────────────
        let storedAssistant = HTTPPrefixCacheMessage.assistant(
            content: "",  // Tool-only turn — no visible text
            reasoning: "Need to inspect the file first.",
            toolCalls: [
                HTTPPrefixCacheToolCall(
                    name: toolName,
                    arguments: ["path": .string("main.swift"), "limit": .int(100)]
                )
            ]
        )
        let storedConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: [
                .init(role: .user, content: "Please inspect main.swift"),
                storedAssistant,
            ],
            toolDefinitionsDigest: HTTPPrefixCacheConversation.emptyToolDefinitionsDigest,
            templateContextDigest: HTTPPrefixCacheConversation.defaultTemplateContextDigest
        )

        // ── STREAM side: build the OpenAI tool call OpenCode receives via SSE ──
        let mlxToolCall = ToolCall(
            function: ToolCall.Function(name: toolName, arguments: argumentsDict)
        )
        let oaiCalls = ToolCallConverter.convertToOpenAI([mlxToolCall])
        let oaiCall = try #require(oaiCalls.first)

        // Round-trip the tool call through JSON to mimic the SSE wire transit.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let oaiCallData = try encoder.encode(oaiCall)
        let echoedOaiCall = try JSONDecoder().decode(OpenAI.ToolCall.self, from: oaiCallData)

        // ── LOOKUP side: build the OpenAI request OpenCode would send next turn ─
        let echoedRequestMessages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text(systemPrompt)),
            .init(role: .user, content: .text("Please inspect main.swift")),
            .init(
                role: .assistant,
                content: .text(""),
                tool_calls: [echoedOaiCall],
                reasoning_content: "Need to inspect the file first."
            ),
            .init(role: .tool, content: .text("file contents"), tool_call_id: echoedOaiCall.id),
            .init(role: .user, content: .text("What's next?")),
        ]
        let eligibility = MessageConverter.analyzePrefixCacheEligibility(echoedRequestMessages)
        let lookupConversation = try #require(eligibility.conversation)

        // The stored 2-message conversation must be a prefix of the 4-message lookup conversation
        // (the one user system message is folded into systemPrompt by analyzePrefixCacheEligibility).
        #expect(storedConversation.isPrefix(of: lookupConversation))
    }

    /// Stress the exact log pattern observed with OpenCode: empty/whitespace-only
    /// content, long reasoning with trailing newline, single tool call.
    @MainActor
    @Test func storedAssistantRoundTripWithEmptyContentAndWhitespaceReasoning() throws {
        let systemPrompt = "You are a helpful assistant."
        let longReasoning = String(repeating: "Thinking. ", count: 38) + "\n"  // 380 + 1 chars
        let toolName = "Glob"
        let argumentsDict: [String: any Sendable] = ["pattern": "**/*.swift"]

        let storedAssistant = HTTPPrefixCacheMessage.assistant(
            content: "\n\n",  // Mirrors the `responseCharacters=3`-ish pattern
            reasoning: longReasoning,
            toolCalls: [
                HTTPPrefixCacheToolCall(
                    name: toolName,
                    arguments: ["pattern": .string("**/*.swift")]
                )
            ]
        )
        let storedConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: [
                .init(role: .user, content: "Find all swift files"),
                storedAssistant,
            ]
        )

        // OpenCode echoes the assistant — try both: with the trailing newline preserved,
        // and stripped (a permissive client may strip whitespace).
        let mlxToolCall = ToolCall(
            function: ToolCall.Function(name: toolName, arguments: argumentsDict)
        )
        let oaiCall = try #require(ToolCallConverter.convertToOpenAI([mlxToolCall]).first)

        let echoedKept: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text(systemPrompt)),
            .init(role: .user, content: .text("Find all swift files")),
            .init(
                role: .assistant,
                content: .text("\n\n"),
                tool_calls: [oaiCall],
                reasoning_content: longReasoning
            ),
            .init(role: .tool, content: .text("Main.swift"), tool_call_id: oaiCall.id),
        ]
        let echoedStripped: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text(systemPrompt)),
            .init(role: .user, content: .text("Find all swift files")),
            .init(
                role: .assistant,
                content: .text("\n\n"),
                tool_calls: [oaiCall],
                // OpenCode strips trailing newline:
                reasoning_content: longReasoning.trimmingCharacters(in: .whitespacesAndNewlines)
            ),
            .init(role: .tool, content: .text("Main.swift"), tool_call_id: oaiCall.id),
        ]

        let keptConversation = try #require(
            MessageConverter.analyzePrefixCacheEligibility(echoedKept).conversation
        )
        let strippedConversation = try #require(
            MessageConverter.analyzePrefixCacheEligibility(echoedStripped).conversation
        )

        #expect(storedConversation.isPrefix(of: keptConversation))
        #expect(storedConversation.isPrefix(of: strippedConversation))
    }

    @Test func reasoningTrimMakesStoreAndEchoMatch() {
        // Stored from generation: model emits a trailing newline.
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello", reasoning: "Thinking about it.\n"),
            ]
        )
        // OpenCode echoes back the same reasoning (possibly without the trailing newline);
        // either form should produce an equal HTTPPrefixCacheMessage after trimming.
        let echoedKept = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello", reasoning: "Thinking about it.\n"),
                .init(role: .user, content: "next"),
            ]
        )
        let echoedStripped = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello", reasoning: "Thinking about it."),
                .init(role: .user, content: "next"),
            ]
        )

        #expect(stored.isPrefix(of: echoedKept))
        #expect(stored.isPrefix(of: echoedStripped))
    }
}

// MARK: - Session replay store (still used in production)

struct HTTPPrefixCacheSessionReplayTests {

    @Test func sessionReplayRecoversMissingReasoningWhenSignatureMatches() async {
        let store = HTTPPrefixCacheSessionReplayStore()
        await store.record(
            sessionAffinity: "session-1",
            assistantMessage: .assistant(
                content: "Calling tool",
                reasoning: "Need to inspect the file first.",
                toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"main.swift"}"#)]
            )
        )

        let repair = await store.repair(
            messages: [
                .init(
                    role: .assistant,
                    content: .text("Calling tool"),
                    tool_calls: [
                        .init(
                            id: "call_1",
                            type: "function",
                            function: .init(name: "read", arguments: #"{"path":"main.swift"}"#)
                        ),
                    ]
                ),
            ],
            sessionAffinity: "session-1"
        )

        #expect(repair.clientCount == 0)
        #expect(repair.sessionRecoveredCount == 1)
        #expect(repair.missingCount == 0)
        #expect(repair.messages[0].reasoning_content == "Need to inspect the file first.")
    }

    @Test func sessionReplayPrefersClientReasoningOverStoredReasoning() async {
        let store = HTTPPrefixCacheSessionReplayStore()
        await store.record(
            sessionAffinity: "session-1",
            assistantMessage: .assistant(content: "Answer", reasoning: "Stored reasoning")
        )

        let repair = await store.repair(
            messages: [
                .init(
                    role: .assistant,
                    content: .text("Answer"),
                    reasoning_content: "Client reasoning"
                ),
            ],
            sessionAffinity: "session-1"
        )

        #expect(repair.clientCount == 1)
        #expect(repair.sessionRecoveredCount == 0)
        #expect(repair.missingCount == 0)
        #expect(repair.messages[0].reasoning_content == "Client reasoning")
    }

    @Test func sessionReplaySkipsRecoveryWithoutSessionAffinity() async {
        let store = HTTPPrefixCacheSessionReplayStore()
        await store.record(
            sessionAffinity: "session-1",
            assistantMessage: .assistant(content: "Answer", reasoning: "Stored reasoning")
        )

        let repair = await store.repair(
            messages: [
                .init(role: .assistant, content: .text("Answer")),
            ],
            sessionAffinity: nil
        )

        #expect(repair.clientCount == 0)
        #expect(repair.sessionRecoveredCount == 0)
        #expect(repair.missingCount == 1)
        #expect(repair.messages[0].reasoning_content == nil)
    }

    @Test func sessionReplayDoesNotRecoverMismatchedToolCalls() async {
        let store = HTTPPrefixCacheSessionReplayStore()
        await store.record(
            sessionAffinity: "session-1",
            assistantMessage: .assistant(
                content: "Calling tool",
                reasoning: "Stored reasoning",
                toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"a.swift"}"#)]
            )
        )

        let repair = await store.repair(
            messages: [
                .init(
                    role: .assistant,
                    content: .text("Calling tool"),
                    tool_calls: [
                        .init(
                            id: "call_1",
                            type: "function",
                            function: .init(name: "read", arguments: #"{"path":"b.swift"}"#)
                        ),
                    ]
                ),
            ],
            sessionAffinity: "session-1"
        )

        #expect(repair.clientCount == 0)
        #expect(repair.sessionRecoveredCount == 0)
        #expect(repair.missingCount == 1)
        #expect(repair.messages[0].reasoning_content == nil)
    }
}
