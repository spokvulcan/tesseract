//
//  LinearStreamingDetokenizer.swift
//  tesseract
//
//  The vendor's `NaiveStreamingDetokenizer` decodes its whole segment —
//  every token since the last chunk that ended in a newline — on every
//  token, then scans that decode for the common prefix with what it already
//  emitted: a turn costs time quadratic in its longest newline-free run,
//  seconds for an 8k-token tool call. The fidelity replay (ADR-0063
//  decision 9) pays that on the post-EOS tail of every registered turn.
//
//  This detokenizer releases the same chunks in linear time, two ways.
//
//  A byte-level vocabulary — GPT-2 style BPE, which Qwen and Llama use —
//  spells every token as the bytes it stands for through one fixed
//  alphabet, and decoding is that byte string read as UTF-8. Each token's
//  bytes are recovered once from its spelling, checked against the decode
//  of the token alone, and remembered; a step then appends them and re-reads
//  only the bytes a token can still change — the trailing incomplete scalar
//  — so it costs the token's own length and no `decode` call at all.
//
//  Any other vocabulary (the first token whose bytes do not check out
//  switches to this for good) decodes a short window of trailing tokens per
//  step and splices the window's change — the scalars it rewrote and the
//  scalars it added — onto the segment's decode. That is exact whenever a
//  token's rendering depends on at most the few tokens before it:
//  SentencePiece whitespace collapse and the `clean_up_tokenization_spaces`
//  rules span a pair or triple of tokens. Every splice checks that the
//  scalars it replaces are the ones the window predicts, resynchronizing
//  from one full decode when they are not.
//
//  Both ways, every segment is verified against one full decode before its
//  chunks are released, and recomputed with the naive algorithm itself when
//  the two disagree — so a decoder that reaches further back, or rewrites
//  text it already emitted, costs a recomputation and never a chunk. Chunks
//  are therefore released per segment rather than per token; the chunk
//  sequence, which is all the stream pipeline folds, is unchanged.
//

import Foundation
import MLXLMCommon

nonisolated struct LinearStreamingDetokenizer {
    /// Trailing tokens decoded per step on the window path; the window
    /// restarts on `windowKeep` tokens when it reaches this length.
    static let windowLength = 12
    /// The lookbehind a restarted window keeps: any UTF-8 scalar spans at
    /// most four single-byte tokens.
    static let windowKeep = 4

    /// How the segment's decode is kept up to date.
    private enum Mode {
        /// From each token's own bytes, with no `decode` call per step.
        case bytes
        /// From a decode of the trailing window, spliced on.
        case window
    }

    private let tokenizer: any Tokenizer
    private var mode = Mode.bytes

    // The naive algorithm's state, mirrored.

    /// The open segment's tokens.
    private var segmentTokens: [Int] = []
    /// `decode(segmentTokens)`, as scalars.
    private var full: [Unicode.Scalar] = []
    /// The naive `segment` — the last committed decode — as the count of
    /// leading scalars it shares with `full` plus its own scalars beyond
    /// them (non-empty only across the incomplete-scalar steps that commit
    /// nothing).
    private var committedShared = 0
    private var committedTail: [Unicode.Scalar] = []

    /// Each token's bytes, recovered and checked once.
    private var tokenBytes: [Int: [UInt8]] = [:]
    /// The segment's trailing bytes that form a valid but incomplete scalar
    /// — standing in `full` as the one replacement character they read as
    /// until the token that completes them arrives.
    private var incompleteTail: [UInt8] = []

    /// The trailing window and its decode.
    private var windowTokens: [Int] = []
    private var windowScalars: [Unicode.Scalar] = []

    /// The open segment's chunks, released when the segment verifies.
    private var pending: [String] = []
    /// The first segment starts from nothing; every later one opens on the
    /// previous segment's last token, as the naive restart does.
    private var firstSegment = true

    /// Window-path steps whose splice disagreed with the segment's decode
    /// and resynchronized from a full decode.
    private(set) var resyncs = 0
    /// Segments whose verification failed and were recomputed naively.
    private(set) var fallbacks = 0
    /// Whether the tokens so far were decoded from their own bytes.
    var decodesFromBytes: Bool { mode == .bytes }

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Feed one token; returns the chunks of the segment it closed, if any.
    mutating func append(token: Int) -> [String] {
        segmentTokens.append(token)
        advance(token)

        // The naive diff: the scalars beyond the common prefix with the
        // last committed decode.
        let common = committedShared + Self.commonPrefix(full[committedShared...], committedTail)
        var view = String.UnicodeScalarView()
        view.append(contentsOf: full[common...])
        let chunk = String(view)

        // An incomplete scalar at the end: nothing released, nothing committed.
        if chunk.last == "\u{fffd}" { return [] }

        pending.append(chunk)
        if chunk.hasSuffix("\n") {
            let released = closeSegment()
            restart(with: token)
            return released
        }
        commit()
        return []
    }

    /// The end of the stream: the chunks of the open segment.
    mutating func finish() -> [String] {
        closeSegment()
    }

    /// Open a segment on the token whose chunk closed the previous one, as
    /// the naive restart does.
    private mutating func restart(with token: Int) {
        segmentTokens = [token]
        committedShared = 0
        committedTail = []
        incompleteTail = []
        full = []
        if mode == .bytes, let bytes = bytes(of: token) {
            appendBytes(bytes)
        } else {
            mode = .window
            full = Self.scalars(tokenizer.decode(tokenIds: segmentTokens))
        }
        windowTokens = segmentTokens
        windowScalars = full
        commit()
    }

    private mutating func advance(_ token: Int) {
        switch mode {
        case .bytes:
            if let bytes = bytes(of: token) {
                appendBytes(bytes)
            } else {
                switchToWindow()
                advanceWindow(token)
            }
        case .window:
            advanceWindow(token)
        }
    }

    /// Leave the byte path mid-segment: the window has to be decoded from
    /// the tokens before this one.
    private mutating func switchToWindow() {
        mode = .window
        incompleteTail = []
        windowTokens = Array(segmentTokens.dropLast().suffix(Self.windowKeep))
        windowScalars =
            windowTokens.isEmpty ? [] : Self.scalars(tokenizer.decode(tokenIds: windowTokens))
    }

    // MARK: - The byte path

    /// The alphabet a byte-level vocabulary spells its bytes in (GPT-2's
    /// `bytes_to_unicode`, which every byte-level BPE tokenizer shares):
    /// the printable Latin-1 bytes stand for themselves, the other 68 take
    /// the code points from U+0100 up, in byte order.
    static let byteLevelAlphabet: [Unicode.Scalar: UInt8] = {
        var alphabet: [Unicode.Scalar: UInt8] = [:]
        var next: UInt32 = 256
        for byte in UInt8.min...UInt8.max {
            let printable = (33...126).contains(byte) || (161...172).contains(byte) || byte >= 174
            if printable {
                alphabet[Unicode.Scalar(byte)] = byte
            } else {
                alphabet[Unicode.Scalar(next)!] = byte
                next += 1
            }
        }
        return alphabet
    }()

    /// The token's bytes: its spelling read through the alphabet, or the
    /// spelling itself (an added token decodes verbatim) — whichever reads
    /// back as the decode of the token alone. `nil` when neither does,
    /// which is what takes a vocabulary off the byte path.
    private mutating func bytes(of token: Int) -> [UInt8]? {
        if let known = tokenBytes[token] { return known }
        guard let spelling = tokenizer.convertIdToToken(token) else { return nil }
        var candidates: [[UInt8]] = []
        let mapped = spelling.unicodeScalars.compactMap { Self.byteLevelAlphabet[$0] }
        if mapped.count == spelling.unicodeScalars.count { candidates.append(mapped) }
        candidates.append(Array(spelling.utf8))
        let decoded = tokenizer.decode(tokenIds: [token])
        guard let bytes = candidates.first(where: { Self.text(of: $0) == decoded })
        else { return nil }
        tokenBytes[token] = bytes
        return bytes
    }

    /// Append a token's bytes: the held-back incomplete tail and the new
    /// bytes are read together, replacing the one replacement character the
    /// tail stood for.
    private mutating func appendBytes(_ bytes: [UInt8]) {
        let region = incompleteTail + bytes
        let split = Self.incompleteScalarStart(in: region)
        var scalars = Self.scalars(Self.text(of: region[..<split]))
        if split < region.count { scalars.append("\u{fffd}") }
        splice(removing: incompleteTail.isEmpty ? 0 : 1, appending: scalars[...])
        incompleteTail = Array(region[split...])
    }

    /// Where a valid but incomplete UTF-8 scalar begins at the end of
    /// `bytes` — `bytes.count` when the last scalar is complete, or already
    /// ill-formed and so already reading as what it will stay.
    private static func incompleteScalarStart(in bytes: [UInt8]) -> Int {
        let count = bytes.count
        // A scalar is at most four bytes, so an incomplete one starts at
        // most three back.
        for back in 1...3 where back <= count {
            let start = count - back
            let lead = bytes[start]
            guard let length = Self.scalarLength(lead: lead) else {
                // A continuation byte: its lead is further back.
                if Self.isContinuation(lead) { continue }
                return count
            }
            guard back < length else { return count }
            for offset in 1..<back
            where !Self.isContinuation(bytes[start + offset], following: lead, at: offset) {
                return count
            }
            return start
        }
        return count
    }

    /// The scalar length a well-formed lead byte announces.
    private static func scalarLength(lead: UInt8) -> Int? {
        switch lead {
        case 0xC2...0xDF: 2
        case 0xE0...0xEF: 3
        case 0xF0...0xF4: 4
        default: nil
        }
    }

    private static func isContinuation(_ byte: UInt8) -> Bool { (0x80...0xBF).contains(byte) }

    /// The continuation ranges UTF-8 narrows for the byte right after four
    /// of the lead bytes — the surrogate and out-of-range sequences a plain
    /// `0x80...0xBF` test would wrongly call well formed.
    private static func isContinuation(_ byte: UInt8, following lead: UInt8, at offset: Int) -> Bool
    {
        guard offset == 1 else { return isContinuation(byte) }
        switch lead {
        case 0xE0: return (0xA0...0xBF).contains(byte)
        case 0xED: return (0x80...0x9F).contains(byte)
        case 0xF0: return (0x90...0xBF).contains(byte)
        case 0xF4: return (0x80...0x8F).contains(byte)
        default: return isContinuation(byte)
        }
    }

    // MARK: - The window path

    /// Decode the window with `token` appended and splice its change onto
    /// `full`.
    private mutating func advanceWindow(_ token: Int) {
        if windowTokens.count >= Self.windowLength {
            windowTokens = Array(windowTokens.suffix(Self.windowKeep))
            windowScalars = Self.scalars(tokenizer.decode(tokenIds: windowTokens))
        }
        windowTokens.append(token)
        let next = Self.scalars(tokenizer.decode(tokenIds: windowTokens))
        let common = Self.commonPrefix(windowScalars, next)
        let removed = windowScalars.count - common
        let predicted = windowScalars[common...]
        if removed <= full.count, full[(full.count - removed)...].elementsEqual(predicted) {
            splice(removing: removed, appending: next[common...])
        } else {
            resyncs += 1
            resynchronize(to: Self.scalars(tokenizer.decode(tokenIds: segmentTokens)))
        }
        windowScalars = next
    }

    // MARK: - The mirrored state

    private mutating func splice(removing removed: Int, appending added: ArraySlice<Unicode.Scalar>)
    {
        let position = full.count - removed
        if position < committedShared {
            committedTail = Array(full[position..<committedShared]) + committedTail
            committedShared = position
        }
        full.removeLast(removed)
        full.append(contentsOf: added)
    }

    private mutating func resynchronize(to truth: [Unicode.Scalar]) {
        let committed = Array(full[..<committedShared]) + committedTail
        full = truth
        committedShared = Self.commonPrefix(full, committed)
        committedTail = Array(committed[committedShared...])
    }

    private mutating func commit() {
        committedShared = full.count
        committedTail = []
    }

    // MARK: - Segments

    /// Verify the open segment against one full decode and release its
    /// chunks — the naive recomputation's when the two disagree.
    private mutating func closeSegment() -> [String] {
        defer {
            pending = []
            firstSegment = false
        }
        guard !pending.isEmpty else { return [] }
        let truth = Self.scalars(tokenizer.decode(tokenIds: segmentTokens))
        if truth == full { return pending }
        fallbacks += 1
        // Composing a token's bytes did not reproduce the decode, so this
        // vocabulary is not one the byte path can serve.
        mode = .window
        return recomputeSegment()
    }

    private func recomputeSegment() -> [String] {
        var reference = ReferenceDetokenizer(tokenizer: tokenizer)
        var tokens = segmentTokens[...]
        if !firstSegment, let opener = tokens.popFirst() {
            reference.restart(with: opener)
        }
        var chunks: [String] = []
        for token in tokens {
            reference.append(token: token)
            if let chunk = reference.next() { chunks.append(chunk) }
        }
        return chunks
    }

    // MARK: - Helpers

    /// Bytes read as UTF-8 the way a byte-level decoder reads them: the
    /// lossy read is the point, since ill-formed bytes have to reach the
    /// same replacement characters the tokenizer's own decoder gives them.
    private static func text(of bytes: some Collection<UInt8>) -> String {
        // swiftlint:disable:next optional_data_string_conversion
        String(decoding: bytes, as: UTF8.self)
    }

    private static func scalars(_ text: String) -> [Unicode.Scalar] {
        Array(text.unicodeScalars)
    }

    private static func commonPrefix<A: Collection, B: Collection>(_ lhs: A, _ rhs: B) -> Int
    where A.Element == Unicode.Scalar, B.Element == Unicode.Scalar {
        zip(lhs, rhs).prefix { $0 == $1 }.count
    }
}

// MARK: - Reference

extension LinearStreamingDetokenizer {
    /// The naive algorithm, as the vendor's `NaiveStreamingDetokenizer`
    /// spells it, with a settable restart so one segment can be recomputed
    /// from the token it opened on.
    nonisolated struct ReferenceDetokenizer {
        let tokenizer: any Tokenizer
        private var segmentTokens: [Int] = []
        private var segment = ""

        init(tokenizer: any Tokenizer) {
            self.tokenizer = tokenizer
        }

        /// The state after a chunk ending in a newline: the segment holds
        /// only its last token, decoded alone.
        mutating func restart(with token: Int) {
            segmentTokens = [token]
            segment = tokenizer.decode(tokenIds: segmentTokens)
        }

        mutating func append(token: Int) {
            segmentTokens.append(token)
        }

        mutating func next() -> String? {
            let newSegment = tokenizer.decode(tokenIds: segmentTokens)
            let common = zip(newSegment.unicodeScalars, segment.unicodeScalars)
                .prefix { $0 == $1 }.count
            let new = String(newSegment.unicodeScalars.dropFirst(common))
            if new.last == "\u{fffd}" {
                return nil
            }
            if new.hasSuffix("\n") {
                if let lastToken = segmentTokens.last {
                    restart(with: lastToken)
                } else {
                    segmentTokens = []
                    segment = ""
                }
            } else {
                segment = newSegment
            }
            return new
        }
    }
}
