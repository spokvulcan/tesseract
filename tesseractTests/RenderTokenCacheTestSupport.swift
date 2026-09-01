import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

//
//  RenderTokenCacheTestSupport.swift
//  tesseractTests
//
//  Shared fixtures for the RenderTokenCache suites (split for the 1000-line
//  file-length lint): the deterministic greedy tokenizer both the fake and
//  the real halves build their cases on.
//

// MARK: - Fake greedy tokenizer

/// Deterministic tokenizer with a fixed piece vocab: encode is greedy
/// longest-match (unknown scalars become per-scalar tokens), decode is
/// concatenation — so decode(encode(x)) == x always, BYTE-wise. Pieces like
/// `"\nThe"` create BPE-style merges spanning any cut the tests choose.
///
/// Matching runs over UTF-8 bytes, not `Character`s, deliberately: a
/// `String.hasPrefix`-based matcher compares under Unicode canonical
/// equivalence and would encode NFC and NFD spellings of the same text to the
/// SAME ids — which would make it impossible to test that the cache compares
/// bytes (a normalization-insensitive tokenizer has nothing to get wrong).
/// Real byte-level BPE is byte-exact; so is this.
struct GreedyTokenizer: ChatTemplateRendering {
    /// Longest first at encode time. A token's id is its index here.
    let pieces: [String]
    /// `pieces` as UTF-8, parallel by index.
    private let pieceBytes: [[UInt8]]

    init(pieces: [String]) {
        // Longest-match first, by BYTE length; lexicographic byte order breaks
        // ties so ids are stable regardless of the caller's argument order.
        let sorted = pieces.sorted { lhs, rhs in
            let (a, b) = (Array(lhs.utf8), Array(rhs.utf8))
            if a.count != b.count { return a.count > b.count }
            return a.lexicographicallyPrecedes(b)
        }
        self.pieces = sorted
        self.pieceBytes = sorted.map { Array($0.utf8) }
    }

    /// UTF-8 sequence width from a leading byte.
    private static func scalarWidth(leadingByte byte: UInt8) -> Int {
        byte < 0x80 ? 1 : (byte < 0xE0 ? 2 : (byte < 0xF0 ? 3 : 4))
    }

    /// Index of the longest vocab piece matching `bytes` at `offset`.
    private func matchIndex(in bytes: [UInt8], at offset: Int) -> Int? {
        pieceBytes.firstIndex { piece in
            !piece.isEmpty && piece.count <= bytes.count - offset
                && bytes[offset..<(offset + piece.count)].elementsEqual(piece)
        }
    }

    /// ChatML-ish template with a generation prompt tail, mirroring the shape
    /// of the production template.
    private static let generationPrompt = "<|im_start|>assistant\n"

    let bosToken: String? = nil
    let eosToken: String? = "<|im_end|>"
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        let bytes = Array(text.utf8)
        var offset = 0
        while offset < bytes.count {
            if let index = matchIndex(in: bytes, at: offset) {
                ids.append(index)
                offset += pieceBytes[index].count
                continue
            }
            // No vocab match: emit one Unicode scalar, byte-faithfully.
            let width = min(Self.scalarWidth(leadingByte: bytes[offset]), bytes.count - offset)
            let scalarBytes = bytes[offset..<(offset + width)]
            let scalar =
                String(bytes: scalarBytes, encoding: .utf8)?.unicodeScalars.first ?? "\u{FFFD}"
            ids.append(10_000 + Int(scalar.value))
            offset += width
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if id >= 10_000, let scalar = UnicodeScalar(id - 10_000) {
                return String(scalar)
            }
            return pieces[safe: id] ?? ""
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        let needle = Array(token.utf8)
        return pieceBytes.firstIndex { $0.elementsEqual(needle) }
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
        // The one context flag the fake honors, mirroring the production
        // template: `add_generation_prompt: false` suppresses the tail.
        if (additionalContext?["add_generation_prompt"] as? Bool) != false {
            rendered += Self.generationPrompt
        }
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

// MARK: - Shared Conversation Render fixture

/// The one test spelling of a **Conversation Render**: built through the
/// production request-edge constructor, uncached unless a `fingerprint` is
/// given, with the C31 base render optionally carried. The eligibility knobs
/// default to the eligible text-only shape; the eligibility suite flips them.
nonisolated func makeRender(
    _ tokenizer: any MLXLMCommon.Tokenizer,
    toolSpecs: [ToolSpec]? = nil,
    hasMedia: Bool = false,
    producesFlatTextTokens: Bool = true,
    fingerprint: String? = nil,
    base: [Int]? = nil,
    cache: RenderTokenCache = .shared
) -> ConversationRender {
    let render = ConversationRender.forTextOnlyRequest(
        tokenizer: tokenizer,
        toolSpecs: toolSpecs,
        renderContext: .canonical,
        hasMedia: hasMedia,
        producesFlatTextTokens: producesFlatTextTokens,
        modelFingerprint: fingerprint,
        cache: cache
    )
    return base.map { render.carryingBaseRender($0) } ?? render
}
