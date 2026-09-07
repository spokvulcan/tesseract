import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// `LinearStreamingDetokenizer` releases exactly the chunks the vendor's
/// `NaiveStreamingDetokenizer` releases — the stream pipeline's input, so
/// the fidelity replay reconstructs what the live loop streamed — in time
/// linear in the turn rather than quadratic in its longest newline-free
/// run. The chunk sequence is what the pipeline folds; which token released
/// a chunk is not observable to it.
struct LinearStreamingDetokenizerTests {

    private static let bytes = FakeChatMLTokenizer()

    private static func naiveChunks(_ tokens: [Int], tokenizer: any Tokenizer) -> [String] {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var chunks: [String] = []
        for token in tokens {
            detokenizer.append(token: token)
            if let chunk = detokenizer.next() { chunks.append(chunk) }
        }
        return chunks
    }

    private static func linearChunks(
        _ tokens: [Int], tokenizer: any Tokenizer
    ) -> (chunks: [String], detokenizer: LinearStreamingDetokenizer) {
        var detokenizer = LinearStreamingDetokenizer(tokenizer: tokenizer)
        var chunks: [String] = []
        for token in tokens { chunks += detokenizer.append(token: token) }
        chunks += detokenizer.finish()
        return (chunks, detokenizer)
    }

    /// The linear chunks equal the naive chunks, by the path the vocabulary
    /// earns — a byte-level one needs neither guard.
    @discardableResult
    private static func expectSameChunks(
        _ tokens: [Int], tokenizer: any Tokenizer, viaBytes: Bool = true,
        guardsFire: Bool = false,
        _ comment: Comment? = nil, sourceLocation: SourceLocation = #_sourceLocation
    ) -> LinearStreamingDetokenizer {
        let naive = naiveChunks(tokens, tokenizer: tokenizer)
        let (linear, detokenizer) = linearChunks(tokens, tokenizer: tokenizer)
        #expect(linear == naive, comment, sourceLocation: sourceLocation)
        #expect(detokenizer.decodesFromBytes == viaBytes, comment, sourceLocation: sourceLocation)
        if !guardsFire {
            #expect(detokenizer.resyncs == 0, comment, sourceLocation: sourceLocation)
            #expect(detokenizer.fallbacks == 0, comment, sourceLocation: sourceLocation)
        }
        return detokenizer
    }

    private static func byteTokens(_ text: String) -> [Int] {
        bytes.encode(text: text, addSpecialTokens: false)
    }

    // MARK: - Byte-level decoding

    @Test func asciiParagraphsMatchAcrossWindowRestarts() {
        let text =
            "Hello world. This first line is longer than the token window.\n"
            + "Second line here\n\nThird, after a blank line, and no trailing newline"
        Self.expectSameChunks(Self.byteTokens(text), tokenizer: Self.bytes)
    }

    @Test func scalarsSplitAcrossTokensMatch() {
        // Two-, three- and four-byte scalars, each fed one byte at a time:
        // the incomplete steps release nothing, the completing step releases
        // the scalar, and a window restart lands inside a scalar.
        let text = "café €5 日本語テキスト — 😀 done, and more text to slide the window twice"
        Self.expectSameChunks(Self.byteTokens(text), tokenizer: Self.bytes)
    }

    @Test func graphemeJoinersMatch() {
        // A variation selector, a zero-width joiner sequence, a combining
        // accent, regional-indicator pairs and CRLF: every case where a
        // `Character` comparison would differ from the scalar comparison the
        // naive algorithm makes.
        let text = "flag 🏳️‍🌈 then e\u{301} then 🇺🇸🇫🇷 then a\r\nb\r\n and \u{FFFD}\n"
        Self.expectSameChunks(Self.byteTokens(text), tokenizer: Self.bytes)
    }

    @Test func sequencesUTF8NarrowsAreHeldBackOrNot() {
        // The four lead bytes whose second byte is restricted: an overlong
        // encoding, a surrogate, a beyond-U+10FFFF sequence — ill formed the
        // moment the second byte lands, so their replacement characters are
        // settled and a chunk carries them out — against the well-formed
        // sequence with the same lead, which is held back until it closes.
        let cases: [[Int]] = [
            [0xE0, 0x80, 0x80], [0xE0, 0xA0, 0x80],
            [0xED, 0xA0, 0x80], [0xED, 0x9F, 0xBF],
            [0xF0, 0x8F, 0xBF, 0xBF], [0xF0, 0x9F, 0x98, 0x80],
            [0xF4, 0x90, 0x80, 0x80], [0xF4, 0x8F, 0xBF, 0xBF],
        ]
        for sequence in cases {
            let tokens = Self.byteTokens("<") + sequence + Self.byteTokens(">\n")
            Self.expectSameChunks(
                tokens, tokenizer: Self.bytes, Comment(rawValue: "\(sequence)"))
        }
    }

    @Test func invalidByteSequencesMatch() {
        // Lone continuation bytes and truncated sequences decode to the
        // replacement character in both — including the one the naive
        // algorithm holds back at the end of a chunk.
        let tokens =
            Self.byteTokens("ok ") + [0x80, 0xBF] + Self.byteTokens(" mid ") + [0xE2, 0x82]
            + Self.byteTokens("x\n") + [0xF0, 0x9F] + Self.byteTokens("tail")
        Self.expectSameChunks(tokens, tokenizer: Self.bytes)
    }

    @Test func randomByteStreamsMatch() {
        let fragments: [[Int]] =
            [
                "a", "b", " ", ",", "{", "}", "\"", "é", "€", "😀", "🏳️‍🌈", "🇺🇸", "\u{301}", "\n",
                "\r\n", "日", "—",
            ].map(Self.byteTokens)
            + [
                [0x80], [0xC3], [0xE2, 0x82], [0xF0, 0x9F, 0x98], [0xFF], [0xE0], [0xED], [0xF4],
                [0x8F], [0x90], [0xA0],
            ]
        var generator = SeededGenerator(seed: 0x5EED_1234)
        for stream in 0..<150 {
            let length = Int.random(in: 1...60, using: &generator)
            let tokens = (0..<length).flatMap { _ in
                fragments[Int.random(in: 0..<fragments.count, using: &generator)]
            }
            Self.expectSameChunks(
                tokens, tokenizer: Self.bytes, Comment(rawValue: "stream \(stream): \(tokens)"))
        }
    }

    // MARK: - Piece decoding

    @Test func piecesWithEmbeddedNewlinesMatch() {
        // A newline inside a piece closes a segment mid-piece; consecutive
        // newline pieces close segments back to back.
        let tokenizer = GreedyTokenizer(
            pieces: chatMLGreedyPieces + ["ab\ncd", "x\n", "hello", " world", "!"])
        let text = "hello world!ab\ncdx\n\n\nhello\nhello world! hello world! hello world!"
        Self.expectSameChunks(
            tokenizer.encode(text: text, addSpecialTokens: false), tokenizer: tokenizer)
    }

    // MARK: - The window path

    @Test func aVocabularyThatDoesNotSpellItsIdsIsReadThroughTheWindow() {
        // No spelling to recover bytes from, so every step decodes the
        // trailing window and splices its change — exactly, on the same
        // shapes the byte path handles.
        let tokenizer = WindowPathTokenizer()
        let text =
            "A first line long enough to restart the window twice.\n"
            + "café 日本語 😀 and a tail with no newline"
        Self.expectSameChunks(Self.byteTokens(text), tokenizer: tokenizer, viaBytes: false)
    }

    // MARK: - Guards

    @Test func aDecoderThatReadsPastTheWindowFallsBackToTheNaiveChunks() {
        // `Q` renders as `p` when a `P` occurred anywhere earlier — further
        // back than the window sees. The splice is consistent (nothing is
        // rewritten), so the segment's verification is what catches it and
        // the naive recomputation restores the chunks.
        let tokenizer = WindowPathTokenizer(rule: .farLookbehind)
        let text = "P" + String(repeating: "a", count: 30) + "Q" + String(repeating: "b", count: 5)
        let detokenizer = Self.expectSameChunks(
            Self.byteTokens(text), tokenizer: tokenizer, viaBytes: false, guardsFire: true)
        #expect(detokenizer.fallbacks == 1)
    }

    @Test func aDecoderThatRewritesEarlierTextResynchronizesFromAFullDecode() {
        // The last scalar's case follows the whole token count: a restarted
        // window (four tokens) and the segment (twelve) disagree on it, so
        // the next step's splice finds a scalar the window did not predict
        // and resynchronizes.
        let tokenizer = WindowPathTokenizer(rule: .countCase)
        let text = String(repeating: "ab", count: 20) + "\n" + String(repeating: "cd", count: 9)
        let detokenizer = Self.expectSameChunks(
            Self.byteTokens(text), tokenizer: tokenizer, viaBytes: false, guardsFire: true)
        #expect(detokenizer.resyncs > 0)
    }

    @Test func aVocabularyWhoseTextIsNotItsBytesJoinedLeavesTheBytePath() {
        // Every token spells its bytes and decodes as them alone, so the
        // byte path takes it — but the decoder collapses a doubled space,
        // so the joined bytes are not the segment's text. The segment's
        // verification catches that, the naive recomputation restores the
        // chunks, and the rest of the turn goes through the window, which
        // sees the pair the rule spans.
        let tokenizer = CollapsingTokenizer()
        let text = "a  b\ncd ef\ngh"
        let detokenizer = Self.expectSameChunks(
            Self.byteTokens(text), tokenizer: tokenizer, viaBytes: false, guardsFire: true)
        #expect(detokenizer.fallbacks == 1)
    }

    // MARK: - Cost

    @Test func aLongRunWithoutNewlinesIsLinear() {
        // 8k single-byte tokens with no newline: the naive algorithm takes
        // seconds here (quadratic); the linear one a few milliseconds.
        let tokens = Self.byteTokens(String(repeating: "abcdefgh", count: 1000))
        let start = DispatchTime.now().uptimeNanoseconds
        let (chunks, detokenizer) = Self.linearChunks(tokens, tokenizer: Self.bytes)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        #expect(chunks.joined() == String(repeating: "abcdefgh", count: 1000))
        #expect(chunks.count == tokens.count)
        #expect(detokenizer.decodesFromBytes)
        #expect(detokenizer.resyncs == 0 && detokenizer.fallbacks == 0)
        #expect(seconds < 0.5, "\(seconds) s")
    }
}

/// A byte-level decoder that does not spell its ids, so the detokenizer can
/// only read it through the window — optionally with a rule that makes a
/// token's rendering depend on more text than the window shows, to exercise
/// the guards.
private struct WindowPathTokenizer: Tokenizer {
    enum Rule {
        /// `Q` decodes as `p` when a `P` occurs anywhere before it, `q` otherwise.
        case farLookbehind
        /// The last scalar is upper-cased when the token count is one past
        /// a multiple of three.
        case countCase
    }

    var rule: Rule?
    private let bytes = FakeChatMLTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        bytes.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let text = bytes.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        switch rule {
        case nil:
            return text
        case .farLookbehind:
            return text.replacingOccurrences(of: "Q", with: text.contains("P") ? "p" : "q")
        case .countCase:
            guard tokenIds.count % 3 == 1, let last = text.last else { return text }
            return String(text.dropLast()) + String(last).uppercased()
        }
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

/// A byte-level decoder that spells its ids — so the byte path takes it —
/// but whose text is not the joined bytes: it collapses a doubled space,
/// the way a `clean_up_tokenization_spaces` rule does. One token alone
/// never collapses anything, so only a segment reveals the difference.
private struct CollapsingTokenizer: Tokenizer {
    private let bytes = FakeChatMLTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        bytes.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        bytes.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
            .replacingOccurrences(of: "  ", with: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { bytes.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { bytes.convertIdToToken(id) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
