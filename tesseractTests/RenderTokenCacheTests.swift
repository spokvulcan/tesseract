import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

//
//  RenderTokenCacheTests.swift
//  tesseractTests
//
//  Gate 2 for experiment C25: `RenderTokenCache` always reproduces the exact
//  token list `applyChatTemplate` would return — on hits (cached prefix +
//  verified suffix), repeats, and every miss shape.
//
//  Two halves:
//  - `RenderTokenCacheFakeTests`: a deterministic greedy tokenizer with a
//    controllable vocab, proving the trim-back and junction-verification
//    mechanics (including a guaranteed dirty junction and k-budget
//    exhaustion).
//  - `RenderTokenCacheRealTests`: the real PARO tokenizer/template from the
//    local models directory, covering the adversarial junction classes
//    (letter/space/emoji/CRLF/digit reply starters), context drift, edited
//    history, and identical repeats. Skipped when the model is absent.
//
//  A note on hit shapes: the loaded Qwen3.5 template's generation prompt ends
//  `<|im_start|>assistant\n<think>\n`, and the next request's render diverges
//  from the stored one right at that tail — so every growing-history hit on
//  the real model exercises token trim-back (k ≥ 1) by construction.
//

// MARK: - Fake greedy tokenizer

/// Deterministic tokenizer with a fixed piece vocab: encode is greedy
/// longest-match (unknown characters become per-character tokens), decode is
/// concatenation — so decode(encode(x)) == x always. Pieces like `"\nThe"`
/// create BPE-style merges spanning any cut the tests choose.
private struct GreedyTokenizer: ChatTemplateRendering {
    /// Longest first at encode time.
    let pieces: [String]

    init(pieces: [String]) {
        // Longest-match first; stable order otherwise.
        self.pieces = pieces.sorted { $0.count > $1.count }
    }

    /// ChatML-ish template with a generation prompt tail, mirroring the shape
    /// of the production template.
    private static let generationPrompt = "<|im_start|>assistant\n"

    let bosToken: String? = nil
    let eosToken: String? = "<|im_end|>"
    let unknownToken: String? = nil

    private var idByPiece: [String: Int] {
        Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($1, $0) })
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let piece = pieces.first(where: { rest.hasPrefix($0) }) {
                ids.append(idByPiece[piece]!)
                rest = rest.dropFirst(piece.count)
            } else {
                let scalar = rest.unicodeScalars.first!
                ids.append(10_000 + Int(scalar.value))
                rest = rest.dropFirst()
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if id >= 10_000, let scalar = UnicodeScalar(id - 10_000) {
                return String(scalar)
            }
            return pieces[id]
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        idByPiece[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id >= 10_000 {
            return UnicodeScalar(id - 10_000).map { String($0) }
        }
        return pieces[safe: id]
    }

    func renderChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        var rendered = ""
        for message in messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            rendered += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        rendered += Self.generationPrompt
        return rendered
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: try renderChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext),
            addSpecialTokens: false)
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Fake-tokenizer mechanics

struct RenderTokenCacheFakeTests {

    private static let fingerprint = "fake-model"

    /// Base vocab: ChatML scaffolding + the junction-merge pieces.
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

    /// Guaranteed dirty junction: the stored render ends `...assistant\n`
    /// (tokens `assistant`, `"\n"`), the extended render merges `"\nThe"`
    /// across the cut. k=0 must fail the window check and k=1 must verify —
    /// exactly the trim-back path, asserted token-for-token.
    @Test func dirtyJunctionVerifiedByTrimBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()

        let turn1 = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: turn1, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let turn2 = turn1 + [assistant("The end."), user("Again.")]
        let truth = try tokenizer.applyChatTemplate(messages: turn2)
        let resolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: turn2, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))

        #expect(resolution.tokens == truth)
        #expect(resolution.path == .hit(trimmedBy: 1))
    }

    /// A merge consuming more prefix tokens than the trim budget (k ≤ 4)
    /// must degrade to the miss path — still exact, never a wrong token.
    /// The single piece `"<|im_end|>\n<|im_start|>assistant\nThe"` spans six
    /// tail tokens of the stored render, so no k in 0...4 can reproduce the
    /// true tokenization of the extended render.
    @Test func junctionBeyondTrimBudgetMissesExactly() throws {
        let tokenizer = Self.tokenizer(
            extraPieces: ["<|im_end|>\n<|im_start|>assistant\nThe"])
        let cache = RenderTokenCache()

        let turn1 = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: turn1, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let turn2 = turn1 + [assistant("The end."), user("Again.")]
        let truth = try tokenizer.applyChatTemplate(messages: turn2)
        let resolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: turn2, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))

        #expect(resolution.tokens == truth)
        #expect(resolution.path == .miss(.junctionUnverified))
    }

    /// Identical repeat request: the empty-suffix case must not crash and
    /// must return the cached tokens outright.
    @Test func repeatRequestReturnsCachedTokens() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start.")]

        let first = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        let second = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))

        #expect(second.path == .hitRepeat)
        #expect(second.tokens == first.tokens)
    }

    /// An edit to an earlier message breaks the digest chain — miss — while
    /// a pure extension hits.
    @Test func editedMiddleMissesButExtensionHits() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()

        let turn1 = [user("The start.")]
        let turn2 = turn1 + [assistant("The end."), user("Again.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: turn1, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)
        let extended = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: turn2, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        guard case .hit = extended.path else {
            Issue.record("expected extension hit, got \(extended.path)")
            return
        }

        var edited = turn2
        edited[0] = user("The EDITED start.")
        let editedResolution = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: edited, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        guard case .miss(let reason) = editedResolution.path else {
            Issue.record("expected edited-middle miss, got \(editedResolution.path)")
            return
        }
        #expect(reason == .digestMismatch)
        #expect(try editedResolution.tokens == tokenizer.applyChatTemplate(messages: edited))
    }

    /// Digest sanity: nil and empty containers must key differently (they can
    /// render differently), and key order must not change a digest.
    @Test func canonicalFormDistinguishesNilFromEmpty() {
        let nilDigest = RenderTokenCache.sha256Hex(RenderTokenCache.canonicalForm(optional: nil))
        let emptyDigest = RenderTokenCache.sha256Hex(
            RenderTokenCache.canonicalForm(optional: [MLXLMCommon.ToolSpec]()))
        #expect(nilDigest != emptyDigest)

        let reordered: [String: Any] = ["b": 1, "a": "x"]
        let ordered: [String: Any] = ["a": "x", "b": 1]
        #expect(
            RenderTokenCache.canonicalForm(reordered) == RenderTokenCache.canonicalForm(ordered))
    }
}

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
