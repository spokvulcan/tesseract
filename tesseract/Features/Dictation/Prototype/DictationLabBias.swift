//
//  DictationLabBias.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Leans Whisper toward the owner's vocabulary without a prompt (prompts
//  empty or truncate takes on turbo; errors note §1.6). When the tokens just
//  emitted begin one of the terms, the term's next token gets a bonus:
//  shallow fusion over a prefix trie, continuation only, the configuration
//  that fixed terms with no regressions on the owner's replayed audio
//  (errors note §1.7).
//

import CoreML
import Foundation
import WhisperKit

/// The vocabulary the recognizer reads on every take. The lab writes it; the
/// recognizer actor reads it; a lock keeps the two honest.
nonisolated final class LabBias: @unchecked Sendable {
    static let shared = LabBias()
    private let lock = NSLock()
    private var terms: [String] = []
    private var version = 0

    func set(_ newTerms: [String]) {
        lock.lock()
        defer { lock.unlock() }
        guard newTerms != terms else { return }
        terms = newTerms
        version += 1
    }

    func snapshot() -> (terms: [String], version: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (terms, version)
    }
}

nonisolated final class LabBiasFilter: LogitsFiltering, @unchecked Sendable {
    private var next: [[Int]: Set<Int>] = [:]
    private let maxLength: Int
    private let specialBegin: Int
    private let bonus: Float

    init(terms: [String], tokenizer: any WhisperTokenizer, bonus: Float = 6) {
        specialBegin = tokenizer.specialTokens.specialTokenBegin
        self.bonus = bonus
        var longest = 0
        for term in terms {
            for variant in [" " + term, term] {
                let tokens = tokenizer.encode(text: variant).filter {
                    $0 < tokenizer.specialTokens.specialTokenBegin
                }
                guard tokens.count > 1 else { continue }
                longest = max(longest, tokens.count)
                // A term that starts with an ordinary word (" And"+"rei") must
                // not turn every "And I" into "Andrei": its bonus waits for
                // two matched tokens.
                let first = tokenizer.decode(tokens: [tokens[0]])
                    .trimmingCharacters(in: .whitespaces).lowercased()
                let start = Self.commonWords.contains(first) ? 2 : 1
                guard start < tokens.count else { continue }
                for k in start..<tokens.count {
                    next[Array(tokens[0..<k]), default: []].insert(tokens[k])
                }
            }
        }
        maxLength = longest
    }

    var isEmpty: Bool { next.isEmpty }

    /// Words common enough that a term starting with one needs more
    /// evidence than its first token.
    static let commonWords: Set<String> = [
        "a", "an", "and", "the", "i", "you", "we", "they", "he", "she", "it", "to", "of", "in",
        "on", "at", "for", "with", "from", "by", "as", "is", "are", "was", "be", "do", "can", "so",
        "but", "or", "if", "that", "this", "what", "how", "why", "when", "where", "who", "my",
        "your", "our", "me", "us", "them", "not", "no", "yes", "all", "any", "some", "one", "two",
        "new", "old", "more", "just", "like", "get", "go", "make", "take", "look", "see", "use",
        "work", "time", "day", "app", "apps", "test", "tests", "voice", "design", "speech",
        "text", "file", "files", "code", "page", "view", "model", "open", "run", "set", "up",
        "out", "over", "back", "down", "now", "then", "here", "there", "will", "would", "should",
        "could", "have", "has", "had", "let", "please", "basically", "about", "into", "also",
        "flow", "store", "table", "tree", "trees", "cloud", "whisper", "swift",
    ]

    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard maxLength > 1 else { return logits }
        let pointer = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        for k in stride(from: min(maxLength - 1, tokens.count), through: 1, by: -1) {
            let suffix = Array(tokens.suffix(k))
            if suffix.contains(where: { $0 >= specialBegin }) { continue }
            if let continuations = next[suffix] {
                for token in continuations where token < logits.count {
                    pointer[token] = Float16(Float(pointer[token]) + bonus)
                }
                break
            }
        }
        return logits
    }
}
