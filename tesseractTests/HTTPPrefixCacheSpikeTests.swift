import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

struct HTTPPrefixCacheSpikeTests {

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

    @Test func storeReturnsLongestPrefixMatchAcrossToolLoop() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()

        let shorter = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "A"),
                .assistant(
                    content: "Need a tool.",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"a.txt"}"#)]
                ),
            ]
        )
        let longer = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: shorter.messages + [
                .init(role: .tool, content: "contents of a.txt"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: longer.messages + [.init(role: .user, content: "What next?")]
        )

        await store.store(conversation: shorter, key: key, cachedTokenCount: 11, cache: [KVCacheSimple()])
        await store.store(conversation: longer, key: key, cachedTokenCount: 22, cache: [KVCacheSimple()])

        let match = await store.match(conversation: request, key: key)

        #expect(match?.conversation == longer)
        #expect(match?.cachedTokenCount == 22)
    }

    @Test func assistantToolCallTurnsCanBeRestoredAsPrefixes() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
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

        await store.store(
            conversation: assistantTurn,
            key: key,
            cachedTokenCount: 13,
            cache: [KVCacheSimple()]
        )

        let match = await store.match(conversation: request, key: key)

        #expect(match?.conversation == assistantTurn)
        #expect(match?.cachedTokenCount == 13)
    }

    @Test func storeRestoresDeepCopiedCaches() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "A"),
                .assistant(content: "B"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: conversation.messages + [.init(role: .user, content: "C")]
        )
        let originalCache = KVCacheSimple()

        await store.store(
            conversation: conversation,
            key: key,
            cachedTokenCount: 9,
            cache: [originalCache]
        )

        let firstMatch = await store.match(conversation: request, key: key)
        let secondMatch = await store.match(conversation: request, key: key)

        let firstCache = firstMatch?.cache.first as? KVCacheSimple
        let secondCache = secondMatch?.cache.first as? KVCacheSimple

        guard let firstCache, let secondCache else {
            Issue.record("Expected both cache restores to return KVCacheSimple copies")
            return
        }

        #expect(firstCache !== originalCache)
        #expect(secondCache !== originalCache)
        #expect(firstCache !== secondCache)
    }

    @Test func storeSeparatesKeysByToolDefinitionDigestAndTemplateContext() async {
        let store = HTTPPrefixCacheSpikeStore()
        let baseConversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "A"),
                .assistant(content: "B"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: baseConversation.messages + [.init(role: .user, content: "C")]
        )

        let keyA = makeKey(toolDefinitionsDigest: "tool-a", templateContextDigest: "ctx-a")
        let keyB = makeKey(toolDefinitionsDigest: "tool-b", templateContextDigest: "ctx-a")
        let keyC = makeKey(toolDefinitionsDigest: "tool-a", templateContextDigest: "ctx-b")

        await store.store(conversation: baseConversation, key: keyA, cachedTokenCount: 7, cache: [KVCacheSimple()])

        #expect(await store.match(conversation: request, key: keyA) != nil)
        #expect(await store.match(conversation: request, key: keyB) == nil)
        #expect(await store.match(conversation: request, key: keyC) == nil)
    }

    @Test func replayFallsBackWhenRequestAlreadyHasCompletedDescendant() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()

        let prefix = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "A"),
                .assistant(content: "B"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: prefix.messages + [.init(role: .user, content: "C")]
        )
        let completedDescendant = request.appendingAssistant(.assistant(content: "D"))

        await store.store(conversation: prefix, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])
        await store.store(conversation: completedDescendant, key: key, cachedTokenCount: 9, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .completedDescendantReplay)
        #expect(lookup.match == nil)
    }

    @Test func lookupReportsNoEntriesForKey() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [.init(role: .user, content: "Hello")]
        )

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .noEntriesForKey)
        #expect(lookup.keyedEntryCount == 0)
        #expect(lookup.match == nil)
        #expect(lookup.mismatchReport == nil)
    }

    @Test func mismatchReportPointsAtSystemPromptDrift() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: "You are helpful. Today is 2026-04-08T14:48.",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello!"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: "You are helpful. Today is 2026-04-08T14:49.",
            messages: stored.messages + [.init(role: .user, content: "How are you?")]
        )

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .noPrefixMatch)
        #expect(lookup.keyedEntryCount == 1)
        guard case .systemPromptMismatch = lookup.mismatchReport else {
            Issue.record("Expected systemPromptMismatch, got \(String(describing: lookup.mismatchReport))")
            return
        }
    }

    @Test func mismatchReportPointsAtAssistantReasoningDrift() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello", reasoning: "Long reasoning string A."),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello", reasoning: "Long reasoning string B."),
                .init(role: .user, content: "next"),
            ]
        )

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .noPrefixMatch)
        guard case .messageFieldMismatch(let i, _, let field, _, _, _, _, _, _) = lookup.mismatchReport else {
            Issue.record("Expected messageFieldMismatch, got \(String(describing: lookup.mismatchReport))")
            return
        }
        #expect(i == 1)
        #expect(field == "reasoning")
    }

    @Test func mismatchReportPointsAtAssistantContentDrift() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello there"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello"),
                .init(role: .user, content: "next"),
            ]
        )

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .noPrefixMatch)
        guard case .messageFieldMismatch(let i, _, let field, _, _, _, _, _, _) = lookup.mismatchReport else {
            Issue.record("Expected messageFieldMismatch, got \(String(describing: lookup.mismatchReport))")
            return
        }
        #expect(i == 1)
        #expect(field == "content")
    }

    @Test func mismatchReportPointsAtToolCallArgumentsDrift() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(
                    content: "Calling",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"a.swift"}"#)]
                ),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Inspect"),
                .assistant(
                    content: "Calling",
                    toolCalls: [HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path":"b.swift"}"#)]
                ),
                .init(role: .tool, content: "result"),
            ]
        )

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .noPrefixMatch)
        guard case .toolCallArgumentsMismatch(let messageIndex, let toolCallIndex, let toolName, _, _, _, _, _, _) = lookup.mismatchReport else {
            Issue.record("Expected toolCallArgumentsMismatch, got \(String(describing: lookup.mismatchReport))")
            return
        }
        #expect(messageIndex == 1)
        #expect(toolCallIndex == 0)
        #expect(toolName == "read")
    }

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

    /// Phase 8b regression: subagent flow had stored content
    /// `"\n\nNow let me read the source files…\n\n\n\n"` (82 chars) but OpenCode
    /// echoed it back as `"\n\nNow let me read the source files…"` (78 chars,
    /// 4 trailing newlines stripped). The longer cache entry then never matched
    /// even though all other fields agreed. Trimming both sides at construction
    /// makes them equal.
    @Test func subagentTrailingWhitespaceContentMatchesEcho() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()

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

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: echoed, key: key)

        #expect(lookup.reason == .hit)
        if lookup.reason != .hit {
            let report = stored.diagnosePrefixMismatch(against: echoed)
            Issue.record("Expected HIT, got \(lookup.reason); diff=\(String(describing: report))")
        }
    }

    /// Reproduces the failure mode the user observed: server stores assistant turn
    /// with whitespace content (model emitted `"\n\n"` between `</think>` and
    /// `<tool_call>`), client (OpenCode) echoes it back with empty content. The
    /// stored entry should still prefix-match the next request because both sides
    /// normalize whitespace-only content to "".
    @Test func storedTurnWithWhitespaceContentMatchesEmptyEcho() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()

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

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: echoedRequest, key: key)

        #expect(lookup.reason == .hit)
        if lookup.reason != .hit {
            let report = stored.diagnosePrefixMismatch(against: echoedRequest)
            Issue.record("Expected HIT, got \(lookup.reason); diff=\(String(describing: report))")
        }
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

    /// Round-trip test: build the assistant message the way `LLMActor` would after generation,
    /// then build the message OpenCode would echo back via JSON, run it through
    /// `MessageConverter.analyzePrefixCacheEligibility`, and assert that the stored conversation
    /// is recognised as a prefix of the new request.
    ///
    /// If this fails, the Phase 1 mismatch report identifies which field drifted.
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
        if !storedConversation.isPrefix(of: lookupConversation) {
            // Phase 1 diagnostic gives us a precise failure message.
            let report = storedConversation.diagnosePrefixMismatch(against: lookupConversation)
            Issue.record("storedConversation should prefix lookupConversation; firstDivergence=\(String(describing: report))")
            return
        }
        #expect(storedConversation.isPrefix(of: lookupConversation))
    }

    /// Stress the exact log pattern observed with OpenCode: empty/whitespace-only content,
    /// long reasoning with trailing newline, single tool call.
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

        if !storedConversation.isPrefix(of: keptConversation) {
            let report = storedConversation.diagnosePrefixMismatch(against: keptConversation)
            Issue.record("kept-newline echo should prefix-match; firstDivergence=\(String(describing: report))")
        }
        if !storedConversation.isPrefix(of: strippedConversation) {
            let report = storedConversation.diagnosePrefixMismatch(against: strippedConversation)
            Issue.record("stripped-newline echo should prefix-match; firstDivergence=\(String(describing: report))")
        }
    }

    @Test func reasoningTrimMakesStoreAndEchoMatch() async {
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

    @Test func mismatchReportIsNilWhenLookupHits() async {
        let store = HTTPPrefixCacheSpikeStore()
        let key = makeKey()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Hi"),
                .assistant(content: "Hello"),
            ]
        )
        let request = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: stored.messages + [.init(role: .user, content: "More")]
        )

        await store.store(conversation: stored, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])

        let lookup = await store.lookup(conversation: request, key: key)

        #expect(lookup.reason == .hit)
        #expect(lookup.mismatchReport == nil)
    }

    /// Per-key cache replacement: storing a new entry under the same key evicts
    /// any prior entry for that key, regardless of how many other keys exist.
    /// This is the eviction policy that prevents subagent tool loops from
    /// kicking the main agent's cache out — each cache key has exactly one slot.
    @Test func storeKeepsOnlyLatestEntryPerKey() async {
        let store = HTTPPrefixCacheSpikeStore(capacity: 8)
        let key = makeKey()

        let first = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: [.init(role: .user, content: "A"), .assistant(content: "B")]
        )
        let second = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: first.messages + [.init(role: .user, content: "C"), .assistant(content: "D")]
        )
        let third = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: second.messages + [.init(role: .user, content: "E"), .assistant(content: "F")]
        )

        await store.store(conversation: first, key: key, cachedTokenCount: 5, cache: [KVCacheSimple()])
        await store.store(conversation: second, key: key, cachedTokenCount: 11, cache: [KVCacheSimple()])
        await store.store(conversation: third, key: key, cachedTokenCount: 17, cache: [KVCacheSimple()])

        let snapshot = await store.snapshot()
        // Three stores under the same key → only the latest survives.
        #expect(snapshot.entryCount == 1)

        // Lookup with a request that extends `third` should match `third`.
        let request = HTTPPrefixCacheConversation(
            systemPrompt: "system",
            messages: third.messages + [.init(role: .user, content: "G")]
        )
        let lookup = await store.lookup(conversation: request, key: key)
        #expect(lookup.reason == .hit)
        #expect(lookup.match?.cachedTokenCount == 17)
    }

    /// Multi-key isolation: storing many entries under key B does not evict
    /// the single entry under key A. Reproduces the "subagent tool loop evicts
    /// main agent" scenario at the unit-test level.
    @Test func storeIsolatesKeysAndDoesNotEvictAcrossKeys() async {
        let store = HTTPPrefixCacheSpikeStore(capacity: 3)
        let mainAgentKey = makeKey(toolDefinitionsDigest: "main-agent")
        let subagentKey = makeKey(toolDefinitionsDigest: "subagent")

        let mainAgentConversation = HTTPPrefixCacheConversation(
            systemPrompt: "main",
            messages: [.init(role: .user, content: "main user"), .assistant(content: "main asst")]
        )
        await store.store(
            conversation: mainAgentConversation,
            key: mainAgentKey,
            cachedTokenCount: 100,
            cache: [KVCacheSimple()]
        )

        // Subagent does many turns under its own key.
        for turn in 0..<10 {
            let conv = HTTPPrefixCacheConversation(
                systemPrompt: "sub",
                messages: [
                    .init(role: .user, content: "sub user \(turn)"),
                    .assistant(content: "sub asst \(turn)"),
                ]
            )
            await store.store(
                conversation: conv,
                key: subagentKey,
                cachedTokenCount: turn + 1,
                cache: [KVCacheSimple()]
            )
        }

        // Main agent should still be able to find its entry.
        let mainAgentRequest = HTTPPrefixCacheConversation(
            systemPrompt: "main",
            messages: mainAgentConversation.messages + [.init(role: .user, content: "next")]
        )
        let lookup = await store.lookup(conversation: mainAgentRequest, key: mainAgentKey)
        #expect(lookup.reason == .hit)
        #expect(lookup.match?.cachedTokenCount == 100)
    }

    private func makeKey(
        toolDefinitionsDigest: String = HTTPPrefixCacheConversation.emptyToolDefinitionsDigest,
        templateContextDigest: String = HTTPPrefixCacheConversation.defaultTemplateContextDigest
    ) -> HTTPPrefixCacheKey {
        HTTPPrefixCacheKey(
            modelID: "qwen3.5-test",
            kvBits: 8,
            kvGroupSize: 64,
            toolDefinitionsDigest: toolDefinitionsDigest,
            templateContextDigest: templateContextDigest
        )
    }
}

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
