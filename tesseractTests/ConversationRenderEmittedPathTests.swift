import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The Emitted Path Resolve inside the Conversation Render (ADR-0063
/// decisions 4/5), as a dark launch: every byte-producing verb resolves
/// against the index and shadow-checks the composition, the verbs keep
/// returning the canonical tokens, and renders that never consult the
/// index — image-bearing (sealed non-identity) and uncached — say so.
struct ConversationRenderEmittedPathTests {

    private static let fingerprint = "fp-render"
    private let tokenizer = GreedyTokenizer(
        pieces: chatMLGreedyPieces + ["KN", "I", "K", "NI", "hi", "again", "hello"])

    private let turnOne: [[String: any Sendable]] = [
        ["role": "user", "content": "hi"],
        ["role": "assistant", "content": "KNI"],
    ]
    private var turnTwo: [[String: any Sendable]] {
        turnOne + [["role": "user", "content": "again"]]
    }

    private func makeRender(
        index: EmittedPathIndex?, cache: RenderTokenCache = RenderTokenCache(),
        hasMedia: Bool = false
    ) -> ConversationRender {
        ConversationRender.forTextOnlyRequest(
            tokenizer: tokenizer, toolSpecs: nil, renderContext: .canonical, hasMedia: hasMedia,
            producesFlatTextTokens: true, modelFingerprint: Self.fingerprint, cache: cache,
            emittedPathIndex: index, diagnostics: nil)
    }

    /// Register the first turn's path under its stored render, with the
    /// emitted ids given (defaults to the canonical split).
    private func registerTurnOne(
        into index: EmittedPathIndex, emitted: ((inout [Int]) -> Void)? = nil
    ) throws -> [Int] {
        let stored = try tokenizer.renderChatTemplate(
            messages: turnOne, tools: nil, additionalContext: ["add_generation_prompt": false])
        let bytes = Array(stored.utf8)
        let imEnd = try #require(tokenizer.convertTokenToId("<|im_end|>"))
        let canonical = tokenizer.encode(text: stored, addSpecialTokens: false)
        var path = Array(canonical[...(try #require(canonical.lastIndex(of: imEnd)))])
        emitted?(&path)
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: bytes, marker: Array("<|im_end|>".utf8))
        _ = index.register(
            fingerprint: Self.fingerprint, hash: try #require(hashes.last).hash, ids: path)
        return path
    }

    @Test func theRequestEdgeResolvesAgainstTheIndexAndServesCanonical() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        let canonical = try tokenizer.applyChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: nil)
        #expect(tokens == canonical)
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 1)
        #expect(summary.hits == 1)
        #expect(summary.requestEdgeIndexedPrefix == path.count)
        #expect(summary.requestEdgeSuffixTokens == canonical.count - path.count)
        #expect(summary.shadowDifferences == 0)
        #expect(index.statsSnapshot().shadowDifferences == 0)
    }

    @Test func aDifferentEmittedSplitIsAShadowDifferenceAndStillServesCanonical() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let kn = try #require(tokenizer.convertTokenToId("KN"))
        let i = try #require(tokenizer.convertTokenToId("I"))
        let k = try #require(tokenizer.convertTokenToId("K"))
        let ni = try #require(tokenizer.convertTokenToId("NI"))
        _ = try registerTurnOne(into: index) { path in
            let at = path.firstIndex(of: kn)!
            #expect(path[at + 1] == i)
            path.replaceSubrange(at...(at + 1), with: [k, ni])
        }
        let render = makeRender(index: index)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        let canonical = try tokenizer.applyChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: nil)
        #expect(tokens == canonical)
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.hits == 1)
        #expect(summary.shadowDifferences == 1)
        #expect(index.statsSnapshot().shadowDifferences == 1)
    }

    @Test func aColdIndexMissesWithAReason() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = makeRender(index: index)
        _ = try #require(render.fullRender(messages: turnTwo))
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 1)
        #expect(summary.hits == 0)
        #expect(summary.requestEdgeMissReason == "noEntry")
    }

    @Test func theLeafStoreSpellingResolvesAndReturnsBytes() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let stored = try render.storedRender(messages: turnTwo)
        let expected = try tokenizer.renderChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(stored.bytes == Array(expected.utf8))
        #expect(stored.tokens == tokenizer.encode(text: expected, addSpecialTokens: false))
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.hits == 1)
        // Not the request edge: no request-edge prefix recorded.
        #expect(summary.requestEdgeIndexedPrefix == nil)
        #expect(path.count < stored.tokens.count)
    }

    @Test func everySpellingLandsOnTheOneRequestAccount() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        _ = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        _ = render.fullRender(messages: turnTwo)
        _ = try render.lastUserPrefixRender(messages: turnTwo)
        _ = try render.continuationRender(messages: turnTwo)
        _ = try render.uncachedContinuationRender(messages: turnTwo)
        let copy = render.carryingBaseRender([1, 2, 3])
        // The plumbed base render is served without a resolve.
        #expect(try copy.baseRender(messages: turnOne) == [1, 2, 3])
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 4)
        #expect(summary.hits == 4)
        #expect(summary.shadowDifferences == 0)
        #expect(index.statsSnapshot().resolves == 4)
    }

    // MARK: - Renders that never consult the index

    @Test func aSealedNonIdentityRenderNeverConsultsTheIndex() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        _ = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let pad = 248_056
        let imageSpace = try CacheKeySpace.make(
            preparedTokens: [1, 2, 248_053, pad, pad, 248_054, 3],
            images: [.init(digest: ImageDigest(imageBytes: Data("a".utf8)), positionSpan: 2)],
            placeholderIdentity: ImagePlaceholderIdentity(imagePadTokenId: pad)
        ).get()
        #expect(!imageSpace.isIdentity)
        let sealed = render.sealed(for: imageSpace)
        #expect(sealed.fullRender(messages: turnTwo) == nil)
        // The leaf store's render runs in full; no resolve either.
        _ = try sealed.storedRender(messages: turnTwo)
        #expect(index.statsSnapshot().resolves == 0)
        let summary = try #require(sealed.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 0)
        #expect(summary.requestEdgeSkipReason == "nonIdentityKeySpace")
        if case .ineligible(let reason) = sealed.emittedPathEligibility() {
            #expect(reason == "nonIdentityKeySpace")
        } else {
            Issue.record("a sealed non-identity render must be ineligible")
        }
    }

    @Test func aMediaBearingRenderIsIneligibleAtConstruction() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = makeRender(index: index, hasMedia: true)
        #expect(render.fullRender(messages: turnTwo) == nil)
        #expect(render.emittedPathTelemetry?.summary.requestEdgeSkipReason == "media")
        #expect(index.statsSnapshot().resolves == 0)
    }

    @Test func anUncachedRenderConsultsNoIndexUnlessGivenOne() throws {
        let plain = ConversationRender.uncached(tokenizer: tokenizer)
        _ = try plain.continuationRender(messages: turnTwo)
        #expect(plain.emittedPathTelemetry == nil)
        if case .ineligible(let reason) = plain.emittedPathEligibility() {
            #expect(reason == "uncached")
        } else {
            Issue.record("an uncached render must be ineligible")
        }

        // The offline harness's spelling: a private index under a fingerprint.
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        _ = try registerTurnOne(into: index)
        let telemetry = EmittedPathRequestTelemetry(diagnostics: nil)
        let learning = ConversationRender.uncached(
            tokenizer: tokenizer, emittedPathIndex: index,
            emittedPathFingerprint: Self.fingerprint, emittedPathTelemetry: telemetry)
        _ = try learning.continuationRender(messages: turnTwo)
        #expect(telemetry.summary.hits == 1)
        #expect(telemetry.summary.shadowDifferences == 0)
    }

    @Test func aTemplateWithoutASingleTokenMarkerDisablesTheIndex() throws {
        // No `<|im_end|>` piece: the marker cannot be one token, so the
        // render resolves nothing and registration reports the reason.
        let splitTokenizer = GreedyTokenizer(pieces: ["<|im_start|>", "\n", "user", "assistant"])
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = ConversationRender.forTextOnlyRequest(
            tokenizer: splitTokenizer, toolSpecs: nil, renderContext: .canonical, hasMedia: false,
            producesFlatTextTokens: true, modelFingerprint: Self.fingerprint,
            cache: RenderTokenCache(), emittedPathIndex: index, diagnostics: nil)
        _ = try #require(render.fullRender(messages: turnTwo))
        #expect(index.statsSnapshot().resolves == 0)
        if case .ineligible(let reason) = render.emittedPathEligibility() {
            #expect(reason == "noEndOfTurnMarker")
        } else {
            Issue.record("a template without a marker must be ineligible")
        }
    }
}
