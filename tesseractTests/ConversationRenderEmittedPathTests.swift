import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The Emitted Path Resolve inside the Conversation Render (ADR-0063
/// decisions 4/5), serving: every byte-producing verb resolves against the
/// index and returns the deepest registered path plus the canonical encode
/// of the bytes after its marker; an unregistered history renders
/// canonically; renders that never consult the index — image-bearing
/// (sealed non-identity), uncached, and a template whose marker is not a
/// hard boundary — say so.
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

    /// The canonical request render of the second turn.
    private func canonicalTurnTwo() throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: turnTwo, tools: nil, additionalContext: nil)
    }

    private func makeRender(
        index: EmittedPathIndex?, tokenizer: (any Tokenizer)? = nil,
        hasMedia: Bool = false
    ) -> ConversationRender {
        ConversationRender.forTextOnlyRequest(
            tokenizer: tokenizer ?? self.tokenizer, toolSpecs: nil, renderContext: .canonical,
            hasMedia: hasMedia, producesFlatTextTokens: true, modelFingerprint: Self.fingerprint,
            cache: RenderTokenCache(), emittedPathIndex: index, diagnostics: nil)
    }

    /// Whose split of `KNI` the registered path carries.
    private enum Split {
        /// `KN`+`I`, what the canonical encode says.
        case canonical
        /// `K`+`NI`, what the model fed — the 2026-09-06 re-split class.
        case model
    }

    /// Register the first turn's path under its stored render.
    private func registerTurnOne(into index: EmittedPathIndex, split: Split = .model) throws
        -> [Int]
    {
        let stored = try tokenizer.renderChatTemplate(
            messages: turnOne, tools: nil, additionalContext: ["add_generation_prompt": false])
        let bytes = Array(stored.utf8)
        let imEnd = try #require(tokenizer.convertTokenToId("<|im_end|>"))
        let canonical = tokenizer.encode(text: stored, addSpecialTokens: false)
        var path = Array(canonical[...(try #require(canonical.lastIndex(of: imEnd)))])
        if split == .model {
            let kn = try #require(tokenizer.convertTokenToId("KN"))
            let i = try #require(tokenizer.convertTokenToId("I"))
            let k = try #require(tokenizer.convertTokenToId("K"))
            let ni = try #require(tokenizer.convertTokenToId("NI"))
            let at = try #require(path.firstIndex(of: kn))
            #expect(path[at + 1] == i)
            path.replaceSubrange(at...(at + 1), with: [k, ni])
        }
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: bytes, marker: Array("<|im_end|>".utf8))
        _ = index.register(
            fingerprint: Self.fingerprint, hash: try #require(hashes.last).hash, ids: path)
        return path
    }

    /// The served claim: the fed ids for the echoed turn — never the
    /// canonical re-encode of the same text — then the canonical tail.
    private func expectServes(
        _ tokens: [Int], path: [Int], canonical: [Int],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(Array(tokens.prefix(path.count)) == path, sourceLocation: sourceLocation)
        #expect(
            Array(tokens.dropFirst(path.count)) == Array(canonical.dropFirst(path.count)),
            sourceLocation: sourceLocation)
    }

    @Test func theRequestEdgeServesTheRegisteredPathPlusTheCanonicalSuffix() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        let canonical = try canonicalTurnTwo()
        expectServes(tokens, path: path, canonical: canonical)
        #expect(tokens != canonical)
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 1)
        #expect(summary.hits == 1)
        #expect(summary.requestEdgeIndexedPrefix == path.count)
        #expect(summary.requestEdgeSuffixTokens == canonical.count - path.count)
    }

    @Test func aCanonicalSplitRegisteredIsServedAsTheCanonicalEncode() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        _ = try registerTurnOne(into: index, split: .canonical)
        let render = makeRender(index: index)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        #expect(tokens == (try canonicalTurnTwo()))
    }

    @Test func aColdIndexMissesWithAReasonAndServesCanonical() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = makeRender(index: index)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        #expect(tokens == (try canonicalTurnTwo()))
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 1)
        #expect(summary.hits == 0)
        #expect(summary.requestEdgeMissReason == "noEntry")
    }

    @Test func anUnregisteredTurnAfterARegisteredOneIsEncodedCanonicallyPastTheHit() throws {
        // A fidelity-rejected turn registers nothing: the next request hits
        // the turn before it and encodes the rejected turn canonically.
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let threeTurns =
            turnTwo + [
                ["role": "assistant", "content": "hello"],
                ["role": "user", "content": "hi"],
            ]
        let tokens = try #require(render.fullRender(messages: threeTurns))
        let canonical = try tokenizer.applyChatTemplate(
            messages: threeTurns, tools: nil, additionalContext: nil)
        expectServes(tokens, path: path, canonical: canonical)
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.requestEdgeIndexedPrefix == path.count)
    }

    @Test func theBoundaryPathsStoredRenderComposesAndReturnsBytes() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let stored = try render.storedRender(messages: turnTwo)
        let expected = try tokenizer.renderChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(stored.bytes == Array(expected.utf8))
        expectServes(
            stored.tokens, path: path,
            canonical: tokenizer.encode(text: expected, addSpecialTokens: false))
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.hits == 1)
        // Not the request edge: no request-edge prefix recorded.
        #expect(summary.requestEdgeIndexedPrefix == nil)
    }

    @Test func theFastPathsStoredRenderIsBytesOnlyAndConsultsNoIndex() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        _ = try registerTurnOne(into: index, split: .canonical)
        let render = makeRender(index: index)
        let bytes = try #require(try render.storedRenderBytes(messages: turnTwo))
        let expected = try tokenizer.renderChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(bytes == Array(expected.utf8))
        #expect(index.statsSnapshot().resolves == 0)
        #expect(render.emittedPathTelemetry?.summary.resolves == 0)
    }

    @Test func everySpellingLandsOnTheOneRequestAccountAndAgrees() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let render = makeRender(index: index)
        let full = try #require(render.fullRender(messages: turnTwo))
        let lastUser = try render.lastUserPrefixRender(messages: turnTwo)
        let continuation = try render.continuationRender(messages: turnTwo)
        let uncached = try render.uncachedContinuationRender(messages: turnTwo)
        // Every spelling of the same history serves the same fed ids, so a
        // planner boundary measured on one is an offset into another.
        for tokens in [full, lastUser, continuation, uncached] {
            #expect(Array(tokens.prefix(path.count)) == path)
        }
        #expect(continuation == uncached)
        #expect(full.starts(with: lastUser))
        let copy = render.carryingBaseRender([1, 2, 3])
        // The plumbed base render is served without a resolve.
        #expect(try copy.baseRender(messages: turnOne) == [1, 2, 3])
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 4)
        #expect(summary.hits == 4)
        #expect(index.statsSnapshot().resolves == 4)
    }

    // MARK: - Renders that never consult the index

    /// The skip a render's registration would report.
    private func registrationSkip(of render: ConversationRender)
        -> (reason: EmittedPathRegistration.SkipReason, cause: String?)?
    {
        if case .ineligible(let reason, let cause) = render.emittedPathEligibility() {
            return (reason, cause)
        }
        return nil
    }

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
        // The leaf store's render runs in full and canonically; no resolve.
        let stored = try sealed.storedRender(messages: turnTwo)
        let expected = try tokenizer.renderChatTemplate(
            messages: turnTwo, tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(stored.tokens == tokenizer.encode(text: expected, addSpecialTokens: false))
        #expect(index.statsSnapshot().resolves == 0)
        let summary = try #require(sealed.emittedPathTelemetry?.summary)
        #expect(summary.resolves == 0)
        #expect(summary.requestEdgeSkipReason == "nonIdentityKeySpace")
        let skip = try #require(registrationSkip(of: sealed))
        #expect(skip.reason == .ineligibleRender)
        #expect(skip.cause == "nonIdentityKeySpace")
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
        let skip = try #require(registrationSkip(of: plain))
        #expect(skip.reason == .ineligibleRender)
        #expect(skip.cause == "uncached")

        // The offline harness's spelling: a private index under a fingerprint.
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let path = try registerTurnOne(into: index)
        let telemetry = EmittedPathRequestTelemetry(diagnostics: nil)
        let learning = ConversationRender.uncached(
            tokenizer: tokenizer, emittedPathIndex: index,
            emittedPathFingerprint: Self.fingerprint, emittedPathTelemetry: telemetry)
        let tokens = try learning.continuationRender(messages: turnTwo)
        #expect(Array(tokens.prefix(path.count)) == path)
        #expect(telemetry.summary.hits == 1)
    }

    @Test func aTemplateWithoutASingleTokenMarkerDisablesTheIndex() throws {
        // No `<|im_end|>` piece: the marker cannot be one token, so the
        // render resolves nothing and registration reports the reason.
        let splitTokenizer = GreedyTokenizer(pieces: ["<|im_start|>", "\n", "user", "assistant"])
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = makeRender(index: index, tokenizer: splitTokenizer)
        _ = try #require(render.fullRender(messages: turnTwo))
        #expect(index.statsSnapshot().resolves == 0)
        let skip = try #require(registrationSkip(of: render))
        #expect(skip.reason == .noEndOfTurnMarker)
    }

    @Test func aTokenizerWhoseSuffixEncodeIsNotInContextDisablesTheIndex() throws {
        // A Metaspace-style pretokenizer that prepends a word-boundary
        // token to the first pretoken of any standalone text (Nanbeige's
        // `prepend_scheme: first`): the bytes after a marker encode
        // differently on their own than inside the whole render, so no
        // composition could reproduce the canonical input. The fingerprint
        // is refused at marker derivation, before anything registers.
        let prepending = MetaspacePrependingTokenizer(inner: tokenizer)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = makeRender(index: index, tokenizer: prepending)
        let tokens = try #require(render.fullRender(messages: turnTwo))
        #expect(
            tokens
                == prepending.encode(
                    text: try prepending.renderChatTemplate(
                        messages: turnTwo, tools: nil, additionalContext: nil),
                    addSpecialTokens: false))
        #expect(index.statsSnapshot().resolves == 0)
        let skip = try #require(registrationSkip(of: render))
        #expect(skip.reason == .suffixEncodeUnstable)
    }
}
