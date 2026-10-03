//
//  TakeText.swift
//  tesseract
//
//  The words of a take, as Learned Words and the Lens see them (PRD #612):
//  whitespace-delimited tokens, each with its core (the token without the
//  punctuation around it, what is matched and replaced) and its bare form
//  (the core lowercased, what heard forms are compared against). Inner
//  punctuation stays in the core, so "cloud.md", "don't" and "D-flash" are
//  one word each.
//

import Foundation

nonisolated struct TakeToken: Equatable, Sendable {
    /// The whole token, punctuation included.
    let range: Range<String.Index>
    /// The token without its leading and trailing punctuation; empty for a
    /// token that is all punctuation.
    let core: Range<String.Index>
    let text: String
    let coreText: String
    /// Whether the punctuation after the core ends a sentence (".", "!",
    /// "?"). Whisper's "..." is a pause, not a sentence end.
    let endsSentence: Bool

    /// The core, lowercased: what heard forms are compared against.
    var bare: String { coreText.lowercased() }
    /// A word has a letter or a digit; "—" and "..." are not words.
    var isWord: Bool { coreText.contains { $0.isLetter || $0.isNumber } }
}

nonisolated enum TakeText {

    private static let leading: Set<Character> = [
        "\"", "'", "“", "‘", "(", "[", "{", "«", "¿", "¡",
    ]
    private static let trailing: Set<Character> = [
        ".", ",", "!", "?", ";", ":", "\"", "'", "”", "’", ")", "]", "}", "»", "…",
    ]

    /// Short words written with a dot that does not end a sentence.
    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "approx", "jr", "sr", "st",
    ]

    /// Single letters joined by dots: "e.g", "U.S", "a.m".
    private static func isInitialism(_ core: Substring) -> Bool {
        let parts = core.split(separator: ".", omittingEmptySubsequences: false)
        return parts.count >= 2 && parts.allSatisfy { $0.count == 1 && $0.first?.isLetter == true }
    }

    static func tokens(_ text: String) -> [TakeToken] {
        var tokens: [TakeToken] = []
        var index = text.startIndex
        while index < text.endIndex {
            guard !text[index].isWhitespace else {
                index = text.index(after: index)
                continue
            }
            var end = index
            while end < text.endIndex, !text[end].isWhitespace {
                end = text.index(after: end)
            }
            tokens.append(token(in: text, index..<end))
            index = end
        }
        return tokens
    }

    private static func token(in text: String, _ range: Range<String.Index>) -> TakeToken {
        var lower = range.lowerBound
        while lower < range.upperBound, leading.contains(text[lower]) {
            lower = text.index(after: lower)
        }
        var upper = range.upperBound
        while upper > lower {
            let before = text.index(before: upper)
            guard trailing.contains(text[before]) else { break }
            upper = before
        }
        let after = text[upper..<range.upperBound]
        let isEllipsis = after.hasPrefix("...") || after.hasPrefix("…")
        let core = text[lower..<upper]
        // "e.g." and "U.S." end in a dot without ending the sentence; a file
        // name or a version ("CLAUDE.md.", "2.0.") does end it.
        let isAbbreviation =
            after.first == "." && (isInitialism(core) || abbreviations.contains(core.lowercased()))
        let endsSentence =
            !isEllipsis && !isAbbreviation && after.contains { ".!?".contains($0) }
        return TakeToken(
            range: range, core: lower..<upper, text: String(text[range]),
            coreText: String(text[lower..<upper]), endsSentence: endsSentence)
    }

    /// A word starts a sentence when it is the first token or follows one
    /// that ends a sentence.
    static func startsSentence(_ tokens: [TakeToken], at index: Int) -> Bool {
        index == 0 || (index - 1 < tokens.count && tokens[index - 1].endsSentence)
    }

    /// The bare forms of a span, joined by single spaces: the normalized
    /// heard form ("D flash two," → "d flash two").
    static func bare(_ tokens: ArraySlice<TakeToken>) -> String {
        tokens.map(\.bare).joined(separator: " ")
    }

    /// Normalizes free text the way heard forms are stored: its words only,
    /// so a dash or a pause between them leaves no gap ("D ... flash" →
    /// "d flash").
    static func normalized(_ text: String) -> String {
        bare(tokens(text).filter(\.isWord)[...])
    }

    /// Fits a replacement to where it lands: at the start of a sentence a
    /// replacement whose first word is all lowercase ("worktree", "a PR")
    /// gets a capital; anywhere else, and for words with their own casing
    /// ("iPhone", "CLAUDE.md"), it is written as given.
    static func fitCase(_ replacement: String, atSentenceStart: Bool) -> String {
        guard atSentenceStart, let first = replacement.first, first.isLowercase else {
            return replacement
        }
        let firstWord = replacement.prefix { !$0.isWhitespace }
        guard !firstWord.contains(where: \.isUppercase) else { return replacement }
        return first.uppercased() + replacement.dropFirst()
    }

    /// The text of a token span from the first core to the last core,
    /// keeping the punctuation outside it.
    static func coreRange(_ tokens: ArraySlice<TakeToken>) -> Range<String.Index>? {
        guard let first = tokens.first, let last = tokens.last else { return nil }
        return first.core.lowerBound..<last.core.upperBound
    }
}
