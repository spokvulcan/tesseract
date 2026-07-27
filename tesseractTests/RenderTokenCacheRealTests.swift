import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

//
//  RenderTokenCacheRealTests.swift
//  tesseractTests
//
//  The real-tokenizer half of the RenderTokenCache suites (split for the
//  1000-line file-length lint): the loaded PARO tokenizer/template plus the
//  C28 trim+extend suites. Fake-tokenizer mechanics and the shared
//  GreedyTokenizer live in RenderTokenCacheTests.swift /
//  RenderTokenCacheTestSupport.swift.
//

// MARK: - Real-tokenizer scenarios

struct RenderTokenCacheRealTests {

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

    private static let fingerprint = "test-fingerprint"

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static func tools() -> [MLXLMCommon.ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
    }

    /// Hit-path exactness over a growing conversation: every turn's cached+
    /// suffix token list must equal the full encode, turn over turn.
    @Test(.enabled(if: modelAvailable))
    func growingConversationHitsExactly() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Start the investigation."],
        ]

        var sawHit = false
        for turn in 0..<4 {
            let truth = try tokenizer.applyChatTemplate(
                messages: messages, tools: Self.tools(), additionalContext: nil)
            let resolution = try #require(
                try cache.resolve(
                    tokenizer: tokenizer, messages: messages, tools: Self.tools(),
                    additionalContext: nil, modelFingerprint: Self.fingerprint))
            #expect(resolution.tokens == truth, "turn \(turn) token mismatch")
            if case .hit = resolution.path { sawHit = true }

            messages.append([
                "role": "assistant",
                "content": "Turn \(turn) findings: everything checks out so far.",
            ])
            messages.append(["role": "user", "content": "Continue to step \(turn + 1)."])
        }
        #expect(sawHit, "growing conversation never hit the cache")
    }

    /// The adversarial junction classes: the previous render ends with the
    /// generation prompt (`<|im_start|>assistant\n<think>\n`) and the new
    /// assistant content starts with a letter / space / emoji / CRLF / digit.
    /// Each must verify via trim-back (k ≥ 1 — the k=0 cut fails on the
    /// `<think>\n` tail by construction) and reproduce the full encode.
    @Test(
        .enabled(if: modelAvailable),
        arguments: [
            "The results are in. ",  // letter
            " 42 files matched. ",  // leading space + digits
            "🙂 Completed the step. ",  // emoji
            "\r\nMoving on now. ",  // CRLF
            "7 checks passed. ",  // digit
            "\tTabbed answer with  interior spaces. ",  // tab + interior spaces
        ])
    func dirtyJunctionTrimBack(starter: String) async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let turn1: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: turn1, tools: Self.tools(), additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let turn2 =
            turn1 + [
                ["role": "assistant", "content": starter + "Details follow."],
                ["role": "user", "content": "Next step, please."],
            ]
        let truth = try tokenizer.applyChatTemplate(
            messages: turn2, tools: Self.tools(), additionalContext: nil)
        let resolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: turn2, tools: Self.tools(),
                additionalContext: nil, modelFingerprint: Self.fingerprint))

        #expect(resolution.tokens == truth, "junction class \(starter.debugDescription) mismatch")
        guard case .hit(let trimmedBy) = resolution.path else {
            Issue.record("expected trim-back hit, got \(resolution.path)")
            return
        }
        #expect(trimmedBy >= 1, "expected trim-back (k ≥ 1) on the generation-prompt tail")
    }

    /// Unrelated prompt → miss; different tools → miss keyed on the tools
    /// digest. Both still exact.
    @Test(.enabled(if: modelAvailable))
    func unrelatedAndContextDriftMiss() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let base: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: base, tools: Self.tools(), additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let unrelated: [[String: any Sendable]] = [
            ["role": "system", "content": "A totally different system prompt."],
            ["role": "user", "content": "Something else entirely."],
        ]
        let unrelatedResolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: unrelated, tools: Self.tools(),
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        guard case .miss = unrelatedResolution.path else {
            Issue.record("expected unrelated-prompt miss, got \(unrelatedResolution.path)")
            return
        }
        #expect(
            try unrelatedResolution.tokens
                == tokenizer.applyChatTemplate(
                    messages: unrelated, tools: Self.tools(), additionalContext: nil))

        let driftedTools: [MLXLMCommon.ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": ["type": "object"] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
        let driftResolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: base, tools: driftedTools, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        guard case .miss(let reason) = driftResolution.path else {
            Issue.record("expected context-drift miss, got \(driftResolution.path)")
            return
        }
        #expect(reason == .toolsOrContextMismatch)
        #expect(
            try driftResolution.tokens
                == tokenizer.applyChatTemplate(
                    messages: base, tools: driftedTools, additionalContext: nil))
    }

    /// Identical repeat request: no crash on the empty suffix, cached tokens
    /// returned, still exact.
    @Test(.enabled(if: modelAvailable))
    func identicalRepeatNoCrash() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        let first = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: Self.tools(),
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        let second = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: Self.tools(),
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        #expect(second.path == .hitRepeat)
        #expect(second.tokens == first.tokens)
        #expect(
            try second.tokens
                == tokenizer.applyChatTemplate(
                    messages: messages, tools: Self.tools(), additionalContext: nil))
    }

    /// CRLF / emoji / whitespace at message boundaries inside a longer
    /// conversation — every turn asserted against the full encode.
    @Test(.enabled(if: modelAvailable))
    func crlfEmojiWhitespaceJunctions() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let contents = [
            "First line.\r\nSecond line after CRLF.",
            "Emoji run 🙂🌍👨‍👩‍👧‍👦 mid-message.",
            "   Leading spaces and\ttabs\tthroughout.",
            "Trailing whitespace below.\n\n\n",
        ]
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Start."],
        ]
        for (index, content) in contents.enumerated() {
            let truth = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: nil)
            let resolution = try #require(
                try cache.resolve(
                    tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                    modelFingerprint: Self.fingerprint))
            #expect(resolution.tokens == truth, "junction case \(index) mismatch")
            messages.append(["role": "assistant", "content": content])
            messages.append(["role": "user", "content": "Acknowledged \(index). Go on."])
        }
    }
}

// MARK: - C27 truncated resolve (real tokenizer)

/// C27 on the real PARO tokenizer/template: the truncated-at-last-user
/// render's tokens must be a verified trim of the stored full entry — exact
/// against `applyChatTemplate(..., add_generation_prompt: false)` — or the
/// resolve falls back. Covers the end-of-text right-context classes at the
/// cut (whitespace runs, letter, CRLF, emoji, digits, `<|im_end|>\n`).
struct RenderTokenCacheTruncatedRealTests {

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

    private static let fingerprint = "test-fingerprint"
    private static let noGenPrompt: [String: any Sendable] = ["add_generation_prompt": false]

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static func resolveTruncated(
        _ cache: RenderTokenCache,
        tokenizer: any MLXLMCommon.Tokenizer,
        messages: [[String: any Sendable]],
        baseContext: [String: any Sendable]? = nil
    ) throws -> [Int]? {
        try cache.resolveTruncated(
            tokenizer: tokenizer, messages: messages, tools: nil,
            baseAdditionalContext: baseContext,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: Self.fingerprint)
    }

    /// The production shape: a growing conversation whose last message is its
    /// last user message. Every turn's truncated resolve must reproduce the
    /// standalone truncated encode exactly — and hit, never fall back.
    @Test(.enabled(if: modelAvailable))
    func growingConversationTruncatedHitsExactly() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Start the investigation."],
        ]

        for turn in 0..<4 {
            _ = try #require(
                try cache.resolve(
                    tokenizer: tokenizer, messages: messages, tools: nil,
                    additionalContext: nil, modelFingerprint: Self.fingerprint))
            let truncated = try #require(
                try Self.resolveTruncated(cache, tokenizer: tokenizer, messages: messages),
                "turn \(turn) truncated resolve fell back")
            let truth = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: Self.noGenPrompt)
            #expect(truncated == truth, "turn \(turn) truncated token mismatch")

            messages.append([
                "role": "assistant",
                "content": "Turn \(turn) findings: everything checks out so far.",
            ])
            messages.append(["role": "user", "content": "Continue to step \(turn + 1)."])
        }
        #expect(cache.statsSnapshot().truncatedHits == 4)
    }

    /// C31: the entry-prefix assertion (the production PrefillPlanner shape)
    /// reuses the stored chain head — byte-identical to the chain-computing
    /// resolve and to the standalone truncated encode.
    @Test(.enabled(if: modelAvailable))
    func entryPrefixAssertionHitsExactly() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Start the investigation."],
            ["role": "assistant", "content": "First findings: everything checks out."],
            ["role": "user", "content": "Continue to step 2."],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil,
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        let asserted = try #require(
            try cache.resolveTruncated(
                tokenizer: tokenizer, messages: messages, tools: nil,
                baseAdditionalContext: nil,
                mergedAdditionalContext: Self.noGenPrompt,
                modelFingerprint: Self.fingerprint,
                messagesAreEntryPrefix: true),
            "entry-prefix assertion fell back")
        let computed = try #require(
            try Self.resolveTruncated(cache, tokenizer: tokenizer, messages: messages))
        #expect(asserted == computed)
        let truth = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(asserted == truth)
    }

    /// End-of-text right-context classes: the truncated render ends in
    /// `<|im_end|>\n`, but the last user content's ending shapes the pretokens
    /// inside the cut window — whitespace runs, a letter, CRLF, emoji,
    /// digits, and the plain template trailer itself. Each truncated resolve
    /// must verify and reproduce the standalone encode.
    @Test(
        .enabled(if: modelAvailable),
        arguments: [
            "Done with trailing whitespace   ",
            "Done with a letter",
            "Done with CRLF\r\n",
            "Done with emoji 🙂",
            "Done with digits 42",
            "Done plain.",
        ])
    func endOfTextRightContextClasses(content: String) async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": content],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let truncated = try #require(
            try Self.resolveTruncated(cache, tokenizer: tokenizer, messages: messages),
            "class \(content.debugDescription) truncated resolve fell back")
        let truth = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(truncated == truth, "class \(content.debugDescription) token mismatch")
    }

    /// The context digest compares on the UNMERGED base context: a drifted
    /// base cannot borrow an entry stored under another context — fallback,
    /// exact by construction.
    @Test(.enabled(if: modelAvailable))
    func wrongContextDigestFallsBack() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let truncated = try Self.resolveTruncated(
            cache, tokenizer: tokenizer, messages: messages, baseContext: ["x": "y"])
        #expect(truncated == nil)
    }

    /// Assistant tail: the tail past the last user message is dropped
    /// content, not a generation prompt — fallback. System-only against a
    /// stored user conversation: digest-chain mismatch — fallback.
    @Test(.enabled(if: modelAvailable))
    func assistantAndSystemTailFallBack() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let assistantTail: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
            [
                "role": "assistant",
                "content": String(repeating: "A full answer with detail. ", count: 30),
            ],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: assistantTail, tools: nil,
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        let truncatedToLastUser = Array(assistantTail.dropLast())
        let assistantFallback = try Self.resolveTruncated(
            cache, tokenizer: tokenizer, messages: truncatedToLastUser)
        #expect(assistantFallback == nil)

        let systemOnly: [[String: any Sendable]] = [
            ["role": "system", "content": "An entirely different system prompt."]
        ]
        let systemFallback = try Self.resolveTruncated(
            cache, tokenizer: tokenizer, messages: systemOnly)
        #expect(systemFallback == nil)
    }

    /// Cold cache: no stored entry — fallback.
    @Test(.enabled(if: modelAvailable))
    func noStoredEntryFallsBack() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let truncated = try Self.resolveTruncated(
            cache,
            tokenizer: tokenizer,
            messages: [["role": "user", "content": "Begin."]])
        #expect(truncated == nil)
    }
}

// MARK: - C28 tail-replacement resolve (fake tokenizer)

/// Gate for experiment C28's `resolveReplacingTail` on the fake greedy
/// tokenizer: the stored conversation (request + appended assistant turn, or
/// that plus a probe continuation) is recovered as a verified trim of the
/// entry's generation-prompt tail plus a junction-verified suffix encode —
/// or the call falls back, never returning an inexact list. The stored entry
/// must never be mutated by the resolve.
struct RenderTokenCacheReplacingFakeTests {

    private static let fingerprint = "fake-model"
    private static let noGenPrompt: [String: any Sendable] = ["add_generation_prompt": false]

    private static func tokenizer(extraPieces: [String] = []) -> GreedyTokenizer {
        GreedyTokenizer(
            pieces: [
                "<|im_start|>", "<|im_end|>", "assistant", "user", "system",
                "\nThe", "\n", "The", " ", ".", "end", "Again", "start",
            ] + extraPieces)
    }

    private func user(_ content: String) -> [String: any Sendable] {
        ["role": "user", "content": content]
    }

    private func assistant(_ content: String) -> [String: any Sendable] {
        ["role": "assistant", "content": content]
    }

    private func resolveReplacingTail(
        _ cache: RenderTokenCache,
        tokenizer: GreedyTokenizer,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        baseContext: [String: any Sendable]? = nil,
        fingerprint: String = RenderTokenCacheReplacingFakeTests.fingerprint
    ) throws -> [Int]? {
        try cache.resolveReplacingTail(
            tokenizer: tokenizer, messages: messages, tools: tools,
            baseAdditionalContext: baseContext,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: fingerprint)
    }

    /// The production shape: the request's entry is cached, then the stored
    /// conversation (request + assistant turn, generation prompt OFF) is
    /// recovered as trim+extend — exact against the standalone
    /// `applyChatTemplate`.
    @Test func replacedHitTrimsAndExtendsExactly() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let stored = request + [assistant("The end.")]
        let replaced = try #require(
            try resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored))
        let truth = try tokenizer.applyChatTemplate(
            messages: stored, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(replaced == truth)
        #expect(cache.statsSnapshot().replacedHits == 1)
    }

    /// The reply-starter classes at the extension junction: letter, leading
    /// whitespace, emoji, CRLF, digits. Each must verify (the trim-back walk
    /// absorbs any merge spanning the cut) and reproduce the standalone
    /// encode exactly.
    @Test(
        arguments: [
            "Again, the results are in. ",  // letter
            " 42 files matched. ",  // leading space + digits
            "🙂 Completed the step. ",  // emoji
            "\r\nMoving on now. ",  // CRLF
            "7 checks passed. ",  // digit
            "The end. ",  // merge spanning the cut ("\nThe" piece)
        ])
    func replyStarterClasses(starter: String) throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let stored = request + [assistant(starter + "Details follow.")]
        let replaced = try #require(
            try resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored),
            "class \(starter.debugDescription) fell back")
        let truth = try tokenizer.applyChatTemplate(
            messages: stored, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(replaced == truth, "class \(starter.debugDescription) token mismatch")
    }

    /// The admission continuation render: stored conversation plus a
    /// synthetic probe message — the trim crosses the same tail and the
    /// extension spans two appended messages.
    @Test func probeContinuationExtendsExactly() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let probe: [String: any Sendable] = ["role": "user", "content": "Aqkz_strip_probe"]
        let extended = request + [assistant("The end."), probe]
        let replaced = try #require(
            try resolveReplacingTail(cache, tokenizer: tokenizer, messages: extended))
        let truth = try tokenizer.applyChatTemplate(
            messages: extended, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(replaced == truth)
    }

    /// The context digest compares on the UNMERGED base context: a drifted
    /// base cannot borrow an entry stored under another context.
    @Test func wrongBaseContextFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let replaced = try resolveReplacingTail(
            cache, tokenizer: tokenizer, messages: request + [assistant("The end.")],
            baseContext: ["x": "y"])
        #expect(replaced == nil)
        #expect(cache.statsSnapshot().replacedFallbacks == 1)
    }

    /// A merge consuming more tail tokens than the trim budget (k ≤ 4) must
    /// fall back — never a wrong token. The piece
    /// `"<|im_end|>\n<|im_start|>assistant\nThe"` spans six tail tokens'
    /// worth of the extended render, so no k in 0...4 can reproduce the true
    /// tokenization.
    @Test func spanningTokenPastTrimBudgetFallsBack() throws {
        let tokenizer = Self.tokenizer(
            extraPieces: ["<|im_end|>\n<|im_start|>assistant\nThe"])
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let replaced = try resolveReplacingTail(
            cache, tokenizer: tokenizer, messages: request + [assistant("The end.")])
        #expect(replaced == nil)
        #expect(cache.statsSnapshot().replacedFallbacks == 1)
    }

    /// The stored entry is never mutated: after a replaced hit, resolving the
    /// original request again must return the cached tokens outright
    /// (`hitRepeat` — only possible when the entry's render and tokens are
    /// byte-for-byte the ones the first resolve stored).
    @Test func entryNeverMutatedByAReplacedHit() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        let first = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))

        _ = try #require(
            try resolveReplacingTail(
                cache, tokenizer: tokenizer, messages: request + [assistant("The end.")]))

        let again = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        #expect(again.path == .hitRepeat)
        #expect(again.tokens == first.tokens)
    }

    /// Cold cache, fingerprint drift, tools drift, an edited base message
    /// (digest-chain head mismatch), and an equal-length message list (the
    /// strict-extension guard — that shape is `resolveTruncated`'s trim
    /// case): all fall back.
    @Test func mismatchShapesFallBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let request = [user("The start.")]
        let stored = request + [assistant("The end.")]

        // Cold cache.
        #expect(try resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored) == nil)

        _ = try cache.resolve(
            tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        // Fingerprint drift.
        #expect(
            try resolveReplacingTail(
                cache, tokenizer: tokenizer, messages: stored, fingerprint: "other-model") == nil)

        // Tools drift.
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": ["name": "f", "parameters": [:] as [String: any Sendable]]
                as [String: any Sendable],
        ]
        #expect(
            try resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored, tools: [tool])
                == nil)

        // Edited base message.
        let edited = [user("The EDITED start.")] + [assistant("The end.")]
        #expect(try resolveReplacingTail(cache, tokenizer: tokenizer, messages: edited) == nil)

        // Equal-length chain: the trim case, not this variant's.
        #expect(try resolveReplacingTail(cache, tokenizer: tokenizer, messages: request) == nil)

        let stats = cache.statsSnapshot()
        #expect(stats.replacedHits == 0)
        #expect(stats.replacedFallbacks == 5)
    }
}

// MARK: - C28 LeafAdmissionBuilder cache path (fake tokenizer)

/// The **Leaf Admission Builder**'s reusable-prefix probe routed through the
/// C28 tail-replacement resolve: with a fingerprint and an identity key
/// space, both probe renders recover from the entry the request cached — and
/// the probe result is identical to the uncached path. Serialized: the suite
/// drives the process-global `RenderTokenCache.shared`.
@Suite(.serialized)
struct LeafAdmissionCachePathTests {

    private static let fingerprint = "fake-model"

    private static func tokenizer() -> GreedyTokenizer {
        GreedyTokenizer(
            pieces: [
                "<|im_start|>", "<|im_end|>", "assistant", "user", "system",
                "\nThe", "\n", "The", " ", ".", "end", "Again", "start",
            ])
    }

    @Test func reusablePrefixServedByTheCacheMatchesTheUncachedProbe() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache.shared
        cache.reset()
        defer { cache.reset() }

        // The request's entry: the conversation WITHOUT its assistant turn,
        // rendered with the generation prompt — exactly what the Request
        // Keying phase caches.
        let requestConversation = HTTPPrefixCacheConversation(
            systemPrompt: "You are helpful.",
            messages: [HTTPPrefixCacheMessage(role: .user, content: "The start.")]
        )
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: requestConversation.promptMessages, tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)

        let storedConversation = requestConversation.appendingAssistant(
            HTTPPrefixCacheMessage(role: .assistant, content: "The end."))
        let cached = try #require(
            try LeafAdmissionBuilder.reusablePrefix(
                continuation: .userTurn,
                storedConversation: storedConversation,
                toolSpecs: nil,
                tokenizer: tokenizer,
                keySpace: .identity(),
                modelFingerprint: Self.fingerprint
            )?.get())
        let uncached = try #require(
            try LeafAdmissionBuilder.reusablePrefix(
                continuation: .userTurn,
                storedConversation: storedConversation,
                toolSpecs: nil,
                tokenizer: tokenizer,
                keySpace: .identity()
            )?.get())
        #expect(cached == uncached)
        // Both probe renders (stored + continuation) came off the cache.
        #expect(cache.statsSnapshot().replacedHits == 2)
    }

    /// A nil fingerprint keeps the probe on today's uncached path; a
    /// fingerprint with no cached entry falls back to it — both exact by
    /// construction.
    @Test func reusablePrefixFallsBackWithoutACachedEntry() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache.shared
        cache.reset()
        defer { cache.reset() }

        let storedConversation = HTTPPrefixCacheConversation(
            systemPrompt: "You are helpful.",
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: "The start."),
                HTTPPrefixCacheMessage(role: .assistant, content: "The end."),
            ])
        _ = try LeafAdmissionBuilder.reusablePrefix(
            continuation: .userTurn,
            storedConversation: storedConversation,
            toolSpecs: nil,
            tokenizer: tokenizer,
            keySpace: .identity(),
            modelFingerprint: "never-stored"
        )?.get()
        #expect(cache.statsSnapshot().replacedHits == 0)
        #expect(cache.statsSnapshot().replacedFallbacks == 2)
    }
}

// MARK: - C28 tail-replacement resolve (real tokenizer)

/// C28 on the real PARO tokenizer/template: the stored conversation's tokens
/// must be a verified trim+extension of the stored full entry — exact against
/// `applyChatTemplate(..., add_generation_prompt: false)` — or the resolve
/// falls back. Covers the reply-starter classes at the extension junction
/// (letter / whitespace / emoji / CRLF / digits), the admission probe
/// continuation, and the entry's non-mutation.
struct RenderTokenCacheReplacingRealTests {

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

    private static let fingerprint = "test-fingerprint"
    private static let noGenPrompt: [String: any Sendable] = ["add_generation_prompt": false]

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static func resolveReplacingTail(
        _ cache: RenderTokenCache,
        tokenizer: any MLXLMCommon.Tokenizer,
        messages: [[String: any Sendable]],
        baseContext: [String: any Sendable]? = nil
    ) throws -> [Int]? {
        try cache.resolveReplacingTail(
            tokenizer: tokenizer, messages: messages, tools: nil,
            baseAdditionalContext: baseContext,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: Self.fingerprint)
    }

    /// The production shape: a growing conversation; every turn's stored
    /// conversation (request + assistant turn) must resolve exact against the
    /// standalone no-generation-prompt encode — and hit, never fall back.
    @Test(.enabled(if: modelAvailable))
    func growingConversationReplacementsHitExactly() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Start the investigation."],
        ]

        for turn in 0..<4 {
            _ = try #require(
                try cache.resolve(
                    tokenizer: tokenizer, messages: messages, tools: nil,
                    additionalContext: nil, modelFingerprint: Self.fingerprint))
            let reply: [String: any Sendable] = [
                "role": "assistant",
                "content": "Turn \(turn) findings: everything checks out so far.",
            ]
            let stored = messages + [reply]
            let replaced = try #require(
                try Self.resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored),
                "turn \(turn) tail-replacement fell back")
            let truth = try tokenizer.applyChatTemplate(
                messages: stored, tools: nil, additionalContext: Self.noGenPrompt)
            #expect(replaced == truth, "turn \(turn) token mismatch")

            messages.append(reply)
            messages.append(["role": "user", "content": "Continue to step \(turn + 1)."])
        }
        #expect(cache.statsSnapshot().replacedHits == 4)
    }

    /// The reply-starter classes at the extension junction: the previous
    /// render ends with the generation prompt (`<|im_start|>assistant
    /// \n<think>\n`) and the appended reply starts with a letter / space /
    /// emoji / CRLF / digit. Each must verify (the trim-back absorbs any
    /// merge spanning the cut) and reproduce the standalone encode.
    @Test(
        .enabled(if: modelAvailable),
        arguments: [
            "The results are in. ",  // letter
            " 42 files matched. ",  // leading space + digits
            "🙂 Completed the step. ",  // emoji
            "\r\nMoving on now. ",  // CRLF
            "7 checks passed. ",  // digit
            "\tTabbed answer with  interior spaces. ",  // tab + interior spaces
        ])
    func replyStarterClasses(starter: String) async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let request: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let stored =
            request + [
                ["role": "assistant", "content": starter + "Details follow."]
            ]
        let replaced = try #require(
            try Self.resolveReplacingTail(cache, tokenizer: tokenizer, messages: stored),
            "class \(starter.debugDescription) fell back")
        let truth = try tokenizer.applyChatTemplate(
            messages: stored, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(replaced == truth, "class \(starter.debugDescription) token mismatch")
    }

    /// The admission continuation render: the stored conversation plus a
    /// synthetic user probe. The probe turn makes the appended assistant turn
    /// non-latest — a thinking template re-renders it (think strip) — and the
    /// resolve must still be exact: the divergence sits inside the freshly
    /// encoded extension, arbitrated by the byte-prefix check.
    @Test(.enabled(if: modelAvailable))
    func probeContinuationExtendsExactly() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let request: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let probe: [String: any Sendable] = ["role": "user", "content": "Aqkz_strip_probe"]
        let extended =
            request
            + [
                ["role": "assistant", "content": "The audit is complete."],
                probe,
            ]
        let replaced = try #require(
            try Self.resolveReplacingTail(cache, tokenizer: tokenizer, messages: extended),
            "probe continuation fell back")
        let truth = try tokenizer.applyChatTemplate(
            messages: extended, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(replaced == truth)
    }

    /// The context digest compares on the UNMERGED base context — fallback.
    @Test(.enabled(if: modelAvailable))
    func wrongContextDigestFallsBack() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let request: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        _ = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let replaced = try Self.resolveReplacingTail(
            cache, tokenizer: tokenizer,
            messages: request + [["role": "assistant", "content": "Done."]],
            baseContext: ["x": "y"])
        #expect(replaced == nil)
    }

    /// Cold cache: no stored entry — fallback.
    @Test(.enabled(if: modelAvailable))
    func noStoredEntryFallsBack() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let replaced = try Self.resolveReplacingTail(
            cache,
            tokenizer: tokenizer,
            messages: [
                ["role": "user", "content": "Begin."],
                ["role": "assistant", "content": "Done."],
            ])
        #expect(replaced == nil)
    }

    /// The stored entry is never mutated: after a replaced hit, the original
    /// request resolves as an identical repeat — only possible when the
    /// entry's render and tokens are untouched.
    @Test(.enabled(if: modelAvailable))
    func entryUntouchedByReplacedHit() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cache = RenderTokenCache()
        let request: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a careful assistant."],
            ["role": "user", "content": "Begin."],
        ]
        let first = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        _ = try #require(
            try Self.resolveReplacingTail(
                cache, tokenizer: tokenizer,
                messages: request + [["role": "assistant", "content": "Done."]]))
        let again = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: request, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        #expect(again.path == .hitRepeat)
        #expect(again.tokens == first.tokens)
    }
}
