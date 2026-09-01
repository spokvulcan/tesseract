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
        // C29: the reused chain head vacuously passes the digest guard, so
        // edited history is now rejected by the render arbiters — the miss
        // reason is `.renderNotExtended`, and the produced tokens stay exactly
        // the standalone encode.
        #expect(reason == .renderNotExtended)
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

    /// JSON-decoded numbers arrive as `NSNumber`, and `NSNumber(1) as? Bool`
    /// succeeds — so a naive `case let bool as Bool` first would serialize the
    /// integer `1` and the boolean `true` identically. Both spellings of a
    /// boolean must agree with each other and differ from the integers.
    @Test func canonicalFormSeparatesBooleansFromNumbers() {
        let boolTrue = RenderTokenCache.canonicalForm(true)
        let boolFalse = RenderTokenCache.canonicalForm(false)
        #expect(boolTrue == RenderTokenCache.canonicalForm(NSNumber(value: true)))
        #expect(boolFalse == RenderTokenCache.canonicalForm(NSNumber(value: false)))
        #expect(boolTrue != RenderTokenCache.canonicalForm(NSNumber(value: 1)))
        #expect(boolFalse != RenderTokenCache.canonicalForm(NSNumber(value: 0)))
        #expect(RenderTokenCache.canonicalForm(1) != RenderTokenCache.canonicalForm(true))

        // The collision this ordering closes, at dictionary level.
        let flagged: [String: Any] = ["stream": true]
        let counted: [String: Any] = ["stream": NSNumber(value: 1)]
        #expect(RenderTokenCache.canonicalForm(flagged) != RenderTokenCache.canonicalForm(counted))
    }

    /// Swift `String` `==` / `hasPrefix` compare under Unicode canonical
    /// equivalence, so an NFC render and an NFD render of the same text are
    /// EQUAL as `String`s while their bytes — and therefore their token lists —
    /// differ. The junction/cut arbiters cannot catch that (they re-encode
    /// their own decode, which is self-consistent), so the cache must compare
    /// bytes. Both requests must produce exactly `applyChatTemplate`'s list.
    @Test func normalizationShiftedRepeatDoesNotServeCachedTokens() throws {
        // "é" precomposed (U+00E9) vs decomposed (U+0065 U+0301): `==` true,
        // bytes different.
        let composed = "caf\u{00E9}"
        let decomposed = "cafe\u{0301}"
        #expect(composed == decomposed)
        #expect(Array(composed.utf8) != Array(decomposed.utf8))

        let tokenizer = Self.tokenizer(extraPieces: [composed, decomposed, "caf", "e"])
        let cache = RenderTokenCache()

        let first = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: [user(composed)], tools: nil,
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        #expect(first.path == .miss(.cold))
        #expect(
            first.tokens
                == (try tokenizer.applyChatTemplate(
                    messages: [user(composed)], tools: nil, additionalContext: nil)))

        // The canonically-equal-but-byte-different render must NOT be served
        // the first render's tokens.
        let second = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: [user(decomposed)], tools: nil,
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        #expect(second.path != .hitRepeat)
        #expect(
            second.tokens
                == (try tokenizer.applyChatTemplate(
                    messages: [user(decomposed)], tools: nil, additionalContext: nil)))
        #expect(second.tokens != first.tokens)
    }

    /// A byte-identical repeat still returns the cached tokens outright — the
    /// byte comparison must not have cost the repeat path.
    @Test func byteIdenticalRepeatStillHitsOutright() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)
        let repeated = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
                modelFingerprint: Self.fingerprint))
        #expect(repeated.path == .hitRepeat)
    }

    /// `reset()` is the model-unload hook, not just a test affordance: it must
    /// drop the entry (so the next resolve is `.cold`, never a cross-model
    /// hit), the memoized template hashes, and the counters.
    @Test func resetDropsEntryAndStats() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)
        #expect(cache.statsSnapshot() != RenderTokenCache.Stats())

        cache.reset()
        #expect(cache.statsSnapshot() == RenderTokenCache.Stats())

        let afterReset = try #require(
            try cache.resolve(
                tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
                additionalContext: nil, modelFingerprint: Self.fingerprint))
        #expect(afterReset.path == .miss(.cold))
    }

    /// Every miss carries a typed reason into the counters — the only
    /// production signal that the cache is degrading.
    @Test func missReasonsAreCounted() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("Again.")], tools: nil,
            additionalContext: nil, modelFingerprint: "a-different-model")
        let stats = cache.statsSnapshot()
        #expect(stats.missReasons[RenderTokenCache.MissReason.cold.rawValue] == 1)
        #expect(stats.missReasons[RenderTokenCache.MissReason.modelMismatch.rawValue] == 1)
    }
}

// MARK: - Cache eligibility

/// The **Conversation Render** owns cache eligibility — decided once at its
/// request-edge construction and re-checked at key-space sealing; the
/// predicate five seams used to spell five ways, two of which disagreed on
/// the unknown-fingerprint case.
struct ConversationRenderEligibilityTests {

    private let tokenizer = GreedyTokenizer(pieces: [" "])

    /// A key space carrying one image run — non-identity by construction,
    /// built through the same fixture the builder suites use.
    private func imageKeySpace() throws -> CacheKeySpace {
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: "describe",
                    images: [HTTPPrefixCacheImage(data: Data([0x01]))]
                )
            ]
        )
        return try FakeChatMLTokenizer.keySpace(for: conversation, runLengths: [4])
    }

    /// An unknown fingerprint must BYPASS, never resolve under a synthetic
    /// shared key: the repeat path trusts (bytes, fingerprint) with no
    /// empirical arbiter behind it, so two models sharing a key could return
    /// each other's tokens.
    @Test func unknownFingerprintBypasses() {
        #expect(makeRender(tokenizer, fingerprint: nil).cacheFingerprint == nil)
    }

    @Test func mediaAndNonFlatTokenModelsBypass() {
        #expect(makeRender(tokenizer, hasMedia: true, fingerprint: "m").cacheFingerprint == nil)
        #expect(
            makeRender(tokenizer, producesFlatTextTokens: false, fingerprint: "m")
                .cacheFingerprint == nil)
    }

    @Test func eligibleTextOnlyRequestCarriesTheFingerprint() {
        #expect(makeRender(tokenizer, fingerprint: "m").cacheFingerprint == "m")
    }

    /// Sealing re-checks eligibility against the settled key space: identity
    /// keeps the fingerprint, an image-bearing space clears it (placeholder
    /// runs need the real token list, so those requests always render).
    @Test func sealingKeepsTheFingerprintOnlyForAnIdentityKeySpace() throws {
        let render = makeRender(tokenizer, fingerprint: "m")
        #expect(render.sealed(for: .identity()).cacheFingerprint == "m")
        #expect(try render.sealed(for: imageKeySpace()).cacheFingerprint == nil)
    }

    /// The C31 plumbed base render travels with the eligibility decision and
    /// does not alter it.
    @Test func carriedBaseRenderRidesAlongWithoutChangingEligibility() {
        let render = makeRender(tokenizer, fingerprint: "m").carryingBaseRender([1, 2, 3])
        #expect(render.cacheFingerprint == "m")
        #expect(render.baseRenderTokens == [1, 2, 3])
        #expect(makeRender(tokenizer, fingerprint: nil).baseRenderTokens == nil)
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

    /// C31: under the caller's entry-prefix assertion the truncated chain is
    /// the entry's stored head — the result is byte-identical to the
    /// chain-computing resolve and to the standalone `applyChatTemplate`.
    @Test func entryPrefixAssertionHitsExactly() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start."), assistant("The end."), user("Again.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let asserted = try #require(
            try cache.resolveTruncated(
                tokenizer: tokenizer, messages: messages, tools: nil,
                baseAdditionalContext: nil,
                mergedAdditionalContext: Self.noGenPrompt,
                modelFingerprint: Self.fingerprint,
                messagesAreEntryPrefix: true))
        let computed = try #require(
            try resolveTruncated(cache, tokenizer: tokenizer, messages: messages))
        #expect(asserted == computed)
        #expect(
            asserted
                == (try tokenizer.applyChatTemplate(
                    messages: messages, tools: nil, additionalContext: Self.noGenPrompt)))
    }

    /// C31: the assertion with a strict prefix (fewer messages than the
    /// entry holds) reuses the chain head and stays exact.
    @Test func entryPrefixAssertionStrictPrefixHitsExactly() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        let messages = [user("The start."), assistant("The end."), user("Again.")]
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: messages, tools: nil, additionalContext: nil,
            modelFingerprint: Self.fingerprint)

        let prefixMessages = Array(messages.dropLast())
        let asserted = try #require(
            try cache.resolveTruncated(
                tokenizer: tokenizer, messages: prefixMessages, tools: nil,
                baseAdditionalContext: nil,
                mergedAdditionalContext: Self.noGenPrompt,
                modelFingerprint: Self.fingerprint,
                messagesAreEntryPrefix: true))
        #expect(
            asserted
                == (try tokenizer.applyChatTemplate(
                    messages: prefixMessages, tools: nil, additionalContext: Self.noGenPrompt)))
    }

    /// C31: a false entry-prefix assertion never produces wrong tokens — the
    /// byte-prefix render check rejects the candidate and the resolve falls
    /// back, exactly as the honest digest-mismatch path does.
    @Test func falseEntryPrefixAssertionFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)

        let asserted = try cache.resolveTruncated(
            tokenizer: tokenizer, messages: [user("The EDITED start.")], tools: nil,
            baseAdditionalContext: nil,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: Self.fingerprint,
            messagesAreEntryPrefix: true)
        #expect(asserted == nil)
        let honest = try resolveTruncated(
            cache, tokenizer: tokenizer, messages: [user("The EDITED start.")])
        #expect(honest == nil)
    }

    /// C31: the assertion with more messages than the entry holds cannot
    /// reuse the head — the chain is computed and the head-match guard
    /// fails, same as the unasserted path.
    @Test func entryPrefixAssertionLongerThanEntryFallsBack() throws {
        let tokenizer = Self.tokenizer()
        let cache = RenderTokenCache()
        _ = try cache.resolve(
            tokenizer: tokenizer, messages: [user("The start.")], tools: nil,
            additionalContext: nil, modelFingerprint: Self.fingerprint)

        let longer = [user("The start."), assistant("The end."), user("Again.")]
        let asserted = try cache.resolveTruncated(
            tokenizer: tokenizer, messages: longer, tools: nil,
            baseAdditionalContext: nil,
            mergedAdditionalContext: Self.noGenPrompt,
            modelFingerprint: Self.fingerprint,
            messagesAreEntryPrefix: true)
        #expect(asserted == nil)
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
