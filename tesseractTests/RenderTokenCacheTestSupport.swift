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
/// longest-match (unknown characters become per-character tokens), decode is
/// concatenation — so decode(encode(x)) == x always. Pieces like `"\nThe"`
/// create BPE-style merges spanning any cut the tests choose.
struct GreedyTokenizer: ChatTemplateRendering {
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
