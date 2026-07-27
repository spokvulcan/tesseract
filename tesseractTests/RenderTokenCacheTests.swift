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
//  This file holds the fake-tokenizer half: `RenderTokenCacheFakeTests` and
//  `RenderTokenCacheTruncatedFakeTests`, driven by a deterministic greedy
//  tokenizer with a controllable vocab (`RenderTokenCacheTestSupport.swift`),
//  proving the trim-back and junction-verification mechanics (including a
//  guaranteed dirty junction and k-budget exhaustion). The real-tokenizer
//  suites live in `RenderTokenCacheRealTests.swift`.
//
//  A note on hit shapes: the loaded Qwen3.5 template's generation prompt ends
//  `<|im_start|>assistant\n<think>\n`, and the next request's render diverges
//  from the stored one right at that tail — so every growing-history hit on
//  the real model exercises token trim-back (k ≥ 1) by construction.
//

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

// MARK: - C27 truncated resolve (fake tokenizer)

/// Gate for experiment C27's `resolveTruncated` on the fake greedy
/// tokenizer: the truncated-at-last-user render's tokens are recovered as a
/// verified trim of the stored full entry — or the call falls back, never
/// returning an inexact list.
struct RenderTokenCacheTruncatedFakeTests {

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

    private func system(_ content: String) -> [String: any Sendable] {
        ["role": "system", "content": content]
    }

    private func resolveTruncated(
        _ cache: RenderTokenCache,
        tokenizer: GreedyTokenizer,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        baseContext: [String: any Sendable]? = nil,
        fingerprint: String = RenderTokenCacheTruncatedFakeTests.fingerprint
    ) throws -> [Int]? {
        try cache.resolveTruncated(
            tokenizer: tokenizer, messages: messages, tools: tools,
            baseAdditionalContext: baseContext,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: fingerprint)
    }

    /// The common case: the conversation's last message is its last user
    /// message, so the truncated render is the stored render minus the
    /// generation prompt. The trim must reproduce the standalone
    /// `applyChatTemplate` exactly.
    @Test func truncatedHitTrimsGenerationPromptExactly() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start."), assistant("The end."), user("Again.")]

        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)
        let truncated = try #require(
            try resolveTruncated(cache, tokenizer: tokenizer, messages: messages))

        let truth = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: Self.noGenPrompt)
        #expect(truncated == truth)
        #expect(cache.statsSnapshot().truncatedHits == 1)
    }

    /// Turn over turn of a growing conversation: every truncated resolve is
    /// exact against the standalone encode.
    @Test func truncatedHitsExactAcrossTurns() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        var messages = [user("The start.")]
        for turn in 0..<3 {
            _ = try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint)
            let truncated = try #require(
                try resolveTruncated(cache, tokenizer: tokenizer, messages: messages))
            #expect(
                truncated
                    == (try tokenizer.applyChatTemplate(
                        messages: messages, tools: nil, additionalContext: Self.noGenPrompt)),
                "turn \(turn) truncated token mismatch")
            messages.append(assistant("The end."))
            messages.append(user("Again."))
        }
        #expect(cache.statsSnapshot().truncatedHits == 3)
    }

    /// The tail must be a generation prompt, not dropped content: with an
    /// assistant message after the last user message the tail is long, and
    /// the resolve falls back (the caller's `applyChatTemplate` stays exact
    /// by construction).
    @Test func assistantTailFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let longAssistant = assistant(String(repeating: "The end. ", count: 40))
        let messages = [user("The start."), longAssistant]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let truncated = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: [user("The start.")])
        #expect(truncated == nil)
        #expect(cache.statsSnapshot().truncatedFallbacks == 1)
    }

    /// A conversation with no user message at all keys nothing like the
    /// stored entry — digest-chain mismatch, fallback.
    @Test func systemTailFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)

        let truncated = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: [system("A different head.")])
        #expect(truncated == nil)
    }

    /// No stored entry: cold cache falls back.
    @Test func coldCacheFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let truncated = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: [user("The start.")])
        #expect(truncated == nil)
        #expect(cache.statsSnapshot().truncatedFallbacks == 1)
    }

    /// The context digest compares on the UNMERGED base context: a drifted
    /// base cannot borrow an entry stored under another context.
    @Test func wrongBaseContextFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let truncated = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: messages, baseContext: ["x": "y"])
        #expect(truncated == nil)
    }

    /// Tools digest mismatch falls back, fingerprint mismatch falls back.
    @Test func toolsAndFingerprintMismatchFallBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let tool: [String: any Sendable] = [
            "type": "function",
            "function": ["name": "f", "parameters": [:] as [String: any Sendable]]
                as [String: any Sendable],
        ]
        let toolsDrift = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: messages, tools: [tool])
        #expect(toolsDrift == nil)
        let fingerprintDrift = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: messages, fingerprint: "other-model")
        #expect(fingerprintDrift == nil)
    }

    /// A token spanning the cut cannot be trimmed: the piece
    /// `<|im_end|>\n<|im_start|>` merges across the boundary, so the tail
    /// probe overshoots and the resolve falls back.
    @Test func spanningTokenAtCutFallsBack() throws {
        let tokenizer = Self.tokenizer(extraPieces: ["<|im_end|>\n<|im_start|>"])
        let cache = RenderTokenCache()
        let messages = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let truncated = try resolveTruncated(cache, tokenizer: tokenizer, messages: messages)
        #expect(truncated == nil)
    }
}
