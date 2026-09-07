import CryptoKit
import Foundation
import Testing

@testable import Tesseract_Agent

/// The **Emitted Path Index** (ADR-0063) pinned without a model: register
/// and resolve over byte arrays and token arrays. The rules under test are
/// the ones the design rests on — the whole-prefix key at every end-of-turn
/// marker, deepest-hit resolve, last writer wins, the byte budget with
/// least-recently-used eviction, per-fingerprint scoping, and the terminal-
/// token accounting of the **Emitted Path** itself.
struct EmittedPathIndexTests {

    private static let fingerprint = "fp-a"
    private static let marker = Array("<|im_end|>".utf8)

    /// A ChatML-shaped render with three end-of-turn markers.
    private static let render = Array(
        """
        <|im_start|>user
        hi<|im_end|>
        <|im_start|>assistant
        hello<|im_end|>
        <|im_start|>user
        again<|im_end|>
        <|im_start|>assistant

        """.utf8)

    private static func sha256(_ bytes: ArraySlice<UInt8>) -> [UInt8] {
        Array(SHA256.hash(data: Data(bytes)))
    }

    // MARK: - Marker scan and running hash

    @Test func prefixHashesSnapshotTheRunningHashAtEveryMarker() {
        let hashes = EmittedPathIndex.prefixHashes(renderedBytes: Self.render, marker: Self.marker)
        #expect(hashes.count == 3)
        for snapshot in hashes {
            // Each snapshot covers the bytes through (and including) its marker.
            #expect(
                Self.render[(snapshot.end - Self.marker.count)..<snapshot.end]
                    .elementsEqual(Self.marker))
            #expect(snapshot.hash == Self.sha256(Self.render[..<snapshot.end]))
        }
        #expect(hashes.map(\.end) == hashes.map(\.end).sorted())
    }

    @Test func aRenderWithoutMarkersHasNoPrefixHashes() {
        let bytes = Array("plain text, no template".utf8)
        #expect(EmittedPathIndex.prefixHashes(renderedBytes: bytes, marker: Self.marker).isEmpty)
    }

    // MARK: - Register and resolve

    @Test func resolveTakesTheDeepestRegisteredMarker() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hashes = EmittedPathIndex.prefixHashes(renderedBytes: Self.render, marker: Self.marker)
        // Register the FIRST assistant turn (marker #2 of 3), as the leaf
        // store would after generating "hello".
        let path = [1, 2, 3, 4, 5]
        let registration = index.register(
            fingerprint: Self.fingerprint, hash: hashes[1].hash, ids: path)
        #expect(registration == .inserted(evicted: 0))

        switch index.resolve(
            fingerprint: Self.fingerprint, renderedBytes: Self.render, marker: Self.marker)
        {
        case .hit(let resolution):
            #expect(resolution.path == path)
            #expect(resolution.prefixEnd == hashes[1].end)
            // Walking back from the last marker: the last one (the user's
            // "again") misses, the one before it hits.
            #expect(resolution.markerDepth == 1)
            #expect(resolution.markerCount == 3)
        case .miss(let reason):
            Issue.record("expected a hit, got miss(\(reason))")
        }
    }

    @Test func resolveMissesWithATypedReason() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        #expect(
            index.resolve(
                fingerprint: Self.fingerprint, renderedBytes: Array("no markers".utf8),
                marker: Self.marker) == .miss(.noMarker))
        #expect(
            index.resolve(
                fingerprint: Self.fingerprint, renderedBytes: Self.render, marker: Self.marker)
                == .miss(.noEntry))
        let stats = index.statsSnapshot()
        #expect(stats.resolves == 2)
        #expect(stats.hits == 0)
        #expect(stats.missReasons == ["noMarker": 1, "noEntry": 1])
    }

    @Test func aDifferentHistoryIsADifferentKey() {
        // Same assistant text after a different user message: the whole-
        // prefix hash differs, so the entry cannot be served.
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hashes = EmittedPathIndex.prefixHashes(renderedBytes: Self.render, marker: Self.marker)
        _ = index.register(fingerprint: Self.fingerprint, hash: hashes[1].hash, ids: [9, 9])
        // swiftlint:disable:next optional_data_string_conversion
        let otherHistory = Array(
            String(decoding: Self.render, as: UTF8.self)
                .replacingOccurrences(of: "user\nhi", with: "user\nho").utf8)
        #expect(
            index.resolve(
                fingerprint: Self.fingerprint, renderedBytes: otherHistory, marker: Self.marker)
                == .miss(.noEntry))
    }

    // MARK: - Same key: last writer wins

    @Test func sameKeyOverwritesAndReportsBothLengths() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hash = [UInt8](repeating: 7, count: 32)
        _ = index.register(fingerprint: Self.fingerprint, hash: hash, ids: [1, 2, 3])
        let second = index.register(fingerprint: Self.fingerprint, hash: hash, ids: [1, 2, 9, 9])
        #expect(second == .replaced(previousLength: 3, evicted: 0))
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: hash) == [1, 2, 9, 9])
        let stats = index.statsSnapshot()
        #expect(stats.registrations == 2)
        #expect(stats.overwrites == 1)
        #expect(stats.entryCount == 1)
    }

    // MARK: - Retention

    @Test func byteBudgetEvictsLeastRecentlyUsedFirst() {
        // Each 4-id entry costs 4 * 8 bytes of ids; a 72-byte budget holds two.
        let index = EmittedPathIndex(byteBudget: 2 * 4 * MemoryLayout<Int>.size)
        let a = [UInt8](repeating: 1, count: 32)
        let b = [UInt8](repeating: 2, count: 32)
        let c = [UInt8](repeating: 3, count: 32)
        _ = index.register(fingerprint: Self.fingerprint, hash: a, ids: [1, 1, 1, 1])
        _ = index.register(fingerprint: Self.fingerprint, hash: b, ids: [2, 2, 2, 2])
        // Touch `a` so `b` becomes the least recently used.
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: a) == [1, 1, 1, 1])
        let third = index.register(fingerprint: Self.fingerprint, hash: c, ids: [3, 3, 3, 3])
        #expect(third == .inserted(evicted: 1))
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: b) == nil)
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: a) == [1, 1, 1, 1])
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: c) == [3, 3, 3, 3])
        let stats = index.statsSnapshot()
        #expect(stats.evictions == 1)
        #expect(stats.entryCount == 2)
        #expect(stats.idBytes == 2 * 4 * MemoryLayout<Int>.size)
    }

    @Test func anEntryLargerThanTheBudgetIsRejected() {
        let index = EmittedPathIndex(byteBudget: 3 * MemoryLayout<Int>.size)
        let hash = [UInt8](repeating: 4, count: 32)
        #expect(
            index.register(fingerprint: Self.fingerprint, hash: hash, ids: [1, 2, 3, 4])
                == .rejectedTooLarge)
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: hash) == nil)
        #expect(index.statsSnapshot().entryCount == 0)
    }

    // MARK: - Fingerprint scoping and clearing

    @Test func aDifferentFingerprintClearsTheIndex() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hash = [UInt8](repeating: 5, count: 32)
        _ = index.register(fingerprint: "fp-a", hash: hash, ids: [1])
        #expect(index.lookup(fingerprint: "fp-b", hash: hash) == nil)
        // The old model's entries are gone, not merely hidden.
        #expect(index.statsSnapshot().entryCount == 0)
        #expect(index.lookup(fingerprint: "fp-a", hash: hash) == nil)
    }

    @Test func clearDropsEntriesAndCounters() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hash = [UInt8](repeating: 6, count: 32)
        _ = index.register(fingerprint: Self.fingerprint, hash: hash, ids: [1, 2])
        index.clear()
        #expect(index.lookup(fingerprint: Self.fingerprint, hash: hash) == nil)
        let stats = index.statsSnapshot()
        #expect(stats.entryCount == 0)
        #expect(stats.registrations == 0)
        // The post-clear lookup above counted from zero.
        #expect(stats.lookups == 1)
    }

    // MARK: - End-of-turn marker derivation

    @Test func markerIsDerivedFromTheProbeRenderAndIsOneToken() throws {
        let tokenizer = GreedyTokenizer(pieces: chatMLGreedyPieces)
        let probe = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "probe"],
                ["role": "assistant", "content": EndOfTurnMarker.probeContent],
            ],
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        let marker = try #require(EndOfTurnMarker.derive(probeRender: probe, tokenizer: tokenizer))
        #expect(marker.text == "<|im_end|>")
        #expect(marker.bytes == Self.marker)
        #expect(marker.tokenID == tokenizer.convertTokenToId("<|im_end|>"))
    }

    @Test func markerDerivationRefusesAMarkerThatIsNotASingleToken() {
        // A vocabulary where the marker splits into pieces: the standalone-
        // suffix argument needs one hard boundary token, so no marker.
        let tokenizer = GreedyTokenizer(pieces: ["<|im_start|>", "\n", "user", "assistant"])
        let probe =
            "<|im_start|>user\nprobe<|im_end|>\n<|im_start|>assistant\n"
            + EndOfTurnMarker.probeContent + "<|im_end|>\n"
        #expect(EndOfTurnMarker.derive(probeRender: probe, tokenizer: tokenizer) == nil)
    }

    @Test func markerIsMemoizedPerFingerprintAndClearedWithIt() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let tokenizer = GreedyTokenizer(pieces: chatMLGreedyPieces)
        var derivations = 0
        func derive() -> EndOfTurnMarkerStatus {
            derivations += 1
            let probe = try? tokenizer.renderChatTemplate(
                messages: [
                    ["role": "user", "content": "probe"],
                    ["role": "assistant", "content": EndOfTurnMarker.probeContent],
                ],
                tools: nil, additionalContext: ["add_generation_prompt": false])
            guard let probe,
                let marker = EndOfTurnMarker.derive(probeRender: probe, tokenizer: tokenizer)
            else { return .unavailable(.noEndOfTurnMarker) }
            return .available(marker)
        }
        let first = index.endOfTurnMarker(fingerprint: "fp-a", derive: derive)
        let second = index.endOfTurnMarker(fingerprint: "fp-a", derive: derive)
        #expect(first == second)
        #expect(first.marker?.text == "<|im_end|>")
        #expect(derivations == 1)
        _ = index.endOfTurnMarker(fingerprint: "fp-b", derive: derive)
        #expect(derivations == 2)
    }

    @Test func anUnavailableMarkerIsMemoizedWithItsReason() {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        var derivations = 0
        func derive() -> EndOfTurnMarkerStatus {
            derivations += 1
            return .unavailable(.suffixEncodeUnstable)
        }
        #expect(
            index.endOfTurnMarker(fingerprint: "fp-a", derive: derive)
                == .unavailable(.suffixEncodeUnstable))
        #expect(
            index.endOfTurnMarker(fingerprint: "fp-a", derive: derive)
                == .unavailable(.suffixEncodeUnstable))
        #expect(derivations == 1)
    }

    // MARK: - The marker as a hard boundary

    @Test func theSplitCheckAcceptsATokenizerThatEncodesTheSuffixInContext() throws {
        let tokenizer = GreedyTokenizer(pieces: chatMLGreedyPieces + ["hello", "again"])
        let render = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": "hello"],
                ["role": "user", "content": "again"],
            ],
            tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(
            EndOfTurnMarker.splitsEncoding(
                of: Array(render.utf8), marker: Self.marker, tokenizer: tokenizer))
    }

    @Test func theSplitCheckRefusesATokenizerWhoseSuffixEncodeDiffersStandalone() throws {
        // Every suffix encode gets an extra leading id the whole encode
        // never has (the Metaspace prepend-first shape).
        let tokenizer = MetaspacePrependingTokenizer(
            inner: GreedyTokenizer(pieces: chatMLGreedyPieces + ["hello", "again"]))
        let render = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": "hello"],
                ["role": "user", "content": "again"],
            ],
            tools: nil, additionalContext: ["add_generation_prompt": false])
        #expect(
            !EndOfTurnMarker.splitsEncoding(
                of: Array(render.utf8), marker: Self.marker, tokenizer: tokenizer))
    }

    // MARK: - Terminal-token accounting (the Emitted Path itself)

    private let prompt = [1, 2, 3]
    private let endOfTurn = 99

    @Test func aTurnStoppedOnTheEndOfTurnIdIsThePathAsFed() {
        let path = EmittedPath.make(
            promptKeyPath: prompt, generatedTokens: [10, 11, 99], stoppedOn: 99, endOfTurnID: 99)
        #expect(path.ids == [1, 2, 3, 10, 11, 99])
        #expect(path.appendedEndOfTurn == false)
        #expect(path.promptCount == 3)
        // The stop token is not content.
        #expect(path.contentIDs == [10, 11])
    }

    @Test func aForeignStopIdKeepsTheFedIdAndAppendsTheEndOfTurnId() {
        let path = EmittedPath.make(
            promptKeyPath: prompt, generatedTokens: [10, 11, 42], stoppedOn: 42, endOfTurnID: 99)
        #expect(path.ids == [1, 2, 3, 10, 11, 42, 99])
        #expect(path.appendedEndOfTurn == true)
        #expect(path.contentIDs == [10, 11])
    }

    @Test func aTokenLimitCutAppendsTheEndOfTurnIdAndKeepsEveryFedIdAsContent() {
        let path = EmittedPath.make(
            promptKeyPath: prompt, generatedTokens: [10, 11, 12], stoppedOn: nil, endOfTurnID: 99)
        #expect(path.ids == [1, 2, 3, 10, 11, 12, 99])
        #expect(path.appendedEndOfTurn == true)
        #expect(path.contentIDs == [10, 11, 12])
    }

    @Test func anUnfedBonusTokenStaysInThePath() {
        // DFlash2 can return a token the cache never fed; the path is what
        // was emitted, the leaf ends at the cache offset (the caller's job).
        let path = EmittedPath.make(
            promptKeyPath: prompt, generatedTokens: [10, 11, 99], stoppedOn: 99, endOfTurnID: 99)
        #expect(path.ids.count == 6)
        #expect(path.contentIDs == [10, 11])
    }

    // MARK: - Composition: the index serves the model's own split

    /// The re-split case from the 2026-09-06 sessions: the model emitted
    /// `K`+`NI` for `KNI`, the canonical encode is `KN`+`I`. The index maps
    /// the template's render bytes (identical text) to the emitted ids, and
    /// only the bytes after the marker are encoded canonically.
    @Test func compositionServesTheEmittedSplitAndEncodesOnlyTheSuffix() throws {
        let tokenizer = GreedyTokenizer(
            pieces: chatMLGreedyPieces + ["KN", "I", "K", "NI", "hi", "again", "hello"])
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": "KNI"],
                ["role": "user", "content": "again"],
            ],
            tools: nil, additionalContext: nil)
        let bytes = Array(render.utf8)
        let canonical = tokenizer.encode(text: render, addSpecialTokens: false)
        let kn = try #require(tokenizer.convertTokenToId("KN"))
        let i = try #require(tokenizer.convertTokenToId("I"))
        let k = try #require(tokenizer.convertTokenToId("K"))
        let ni = try #require(tokenizer.convertTokenToId("NI"))
        let imEnd = try #require(tokenizer.convertTokenToId("<|im_end|>"))
        // The canonical encode splits `KNI` as `KN`+`I` (longest match).
        let knIndex = try #require(canonical.firstIndex(of: kn))
        #expect(canonical[knIndex + 1] == i)

        // The model fed `K`+`NI`: register that path against the assistant
        // turn's marker (marker #2 of the three in the render).
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: bytes, marker: Array("<|im_end|>".utf8))
        let assistantMarkerIndex = canonical.indices.filter { canonical[$0] == imEnd }[1]
        var emitted = Array(canonical[...assistantMarkerIndex])
        emitted.replaceSubrange(knIndex...(knIndex + 1), with: [k, ni])
        _ = index.register(fingerprint: Self.fingerprint, hash: hashes[1].hash, ids: emitted)

        let marker = try #require(
            EndOfTurnMarker.derive(
                probeRender: try tokenizer.renderChatTemplate(
                    messages: [
                        ["role": "user", "content": "p"],
                        ["role": "assistant", "content": EndOfTurnMarker.probeContent],
                    ],
                    tools: nil, additionalContext: ["add_generation_prompt": false]),
                tokenizer: tokenizer))
        let composed = EmittedPathResolve.compose(
            index: index, fingerprint: Self.fingerprint, marker: marker,
            renderedBytes: bytes, tokenizer: tokenizer)
        guard case .indexed(let tokens, let indexedPrefix, let suffixTokens, _, _) = composed else {
            Issue.record("expected an indexed composition, got \(composed)")
            return
        }
        #expect(indexedPrefix == emitted.count)
        #expect(Array(tokens.prefix(emitted.count)) == emitted)
        #expect(tokens[knIndex] == k)
        #expect(tokens[knIndex + 1] == ni)
        // Everything after the marker is the canonical encode of the suffix.
        #expect(
            Array(tokens.suffix(from: emitted.count))
                == Array(canonical.suffix(from: emitted.count)))
        #expect(suffixTokens == canonical.count - emitted.count)
        #expect(tokens.count == canonical.count)
    }

    /// JSON spacing and parameter order: the template renders the tool call
    /// compactly and sorted; the model wrote it spaced and in its own order.
    /// The key is the template's render, the value is what the model fed —
    /// so the emitted spelling is served and the render's spelling is never
    /// encoded for that turn.
    @Test func compositionServesTheEmittedJSONSpellingUnderTheRenderedKey() throws {
        let tokenizer = GreedyTokenizer(
            pieces: chatMLGreedyPieces + ["{\"a\":1,\"b\":2}", "{\"b\": 2, \"a\": 1}", "hi", "ok"])
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let render = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": "{\"a\":1,\"b\":2}"],
                ["role": "user", "content": "ok"],
            ],
            tools: nil, additionalContext: nil)
        let bytes = Array(render.utf8)
        let canonical = tokenizer.encode(text: render, addSpecialTokens: false)
        let compact = try #require(tokenizer.convertTokenToId("{\"a\":1,\"b\":2}"))
        let spaced = try #require(tokenizer.convertTokenToId("{\"b\": 2, \"a\": 1}"))
        let imEnd = try #require(tokenizer.convertTokenToId("<|im_end|>"))
        let compactIndex = try #require(canonical.firstIndex(of: compact))
        let assistantMarkerIndex = canonical.indices.filter { canonical[$0] == imEnd }[1]
        var emitted = Array(canonical[...assistantMarkerIndex])
        emitted[compactIndex] = spaced
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: bytes, marker: Array("<|im_end|>".utf8))
        _ = index.register(fingerprint: Self.fingerprint, hash: hashes[1].hash, ids: emitted)
        let marker = EndOfTurnMarker(text: "<|im_end|>", tokenID: imEnd)
        let composed = EmittedPathResolve.compose(
            index: index, fingerprint: Self.fingerprint, marker: marker,
            renderedBytes: bytes, tokenizer: tokenizer)
        guard case .indexed(let tokens, _, _, _, _) = composed else {
            Issue.record("expected an indexed composition, got \(composed)")
            return
        }
        #expect(tokens[compactIndex] == spaced)
        #expect(
            Array(tokens.suffix(from: emitted.count))
                == Array(canonical.suffix(from: emitted.count)))
    }

    @Test func compositionWithoutAnEntryIsAMiss() throws {
        let tokenizer = GreedyTokenizer(pieces: chatMLGreedyPieces)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = EndOfTurnMarker(
            text: "<|im_end|>", tokenID: try #require(tokenizer.convertTokenToId("<|im_end|>")))
        let composed = EmittedPathResolve.compose(
            index: index, fingerprint: Self.fingerprint, marker: marker,
            renderedBytes: Self.render, tokenizer: tokenizer)
        #expect(composed == .miss(.noEntry))
    }
}
