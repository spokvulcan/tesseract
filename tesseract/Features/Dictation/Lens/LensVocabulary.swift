//
//  LensVocabulary.swift
//  tesseract
//
//  The owner's words the Lens completes from while they type (PRD #612):
//  the Learned Words first, most useful first, then the names in the
//  owner's recent takes (words with their own casing, digits or dots, and
//  capitalized words in the middle of a sentence), so a long name takes a
//  few keys even before it was ever fixed.
//

import Foundation

nonisolated enum LensVocabulary {

    /// - Parameters:
    ///   - learned: the Learned Words; forgotten ones are skipped.
    ///   - recentTexts: recent takes as the owner kept them, newest first.
    static func build(learned: [LearnedWord], recentTexts: [String], limit: Int = 400)
        -> [String]
    {
        var seen = Set<String>()
        var words: [String] = []
        func add(_ word: String) {
            guard !word.isEmpty, seen.insert(word).inserted else { return }
            words.append(word)
        }

        let ranked = learned.filter { !$0.isForgotten }
            .sorted { ($0.totalCatches + 2 * $0.fixes) > ($1.totalCatches + 2 * $1.fixes) }
        for word in ranked { add(word.meant) }

        for text in recentTexts {
            let tokens = TakeText.tokens(text)
            for (index, token) in tokens.enumerated() where isName(token, in: tokens, at: index) {
                add(token.coreText)
                if words.count >= limit { return words }
            }
        }
        return words
    }

    /// A word that is a name rather than an ordinary word: its own casing
    /// inside the word (TestFlight, iPhone), letters with digits (DFlash2),
    /// a dot inside (CLAUDE.md), an acronym (KV), or a capital in the middle
    /// of a sentence (Claude).
    static func isName(_ token: TakeToken, in tokens: [TakeToken], at index: Int) -> Bool {
        let core = token.coreText
        guard core.count >= 2, core.contains(where: \.isLetter),
            !LensFix.ordinaryWords.contains(token.bare), !token.bare.hasPrefix("i'")
        else { return false }
        let inner = core.dropFirst()
        if inner.contains(where: \.isUppercase) { return true }
        if core.contains(where: \.isNumber) { return true }
        if core.contains("."), core.last != "." { return true }
        if let first = core.first, first.isUppercase {
            return !TakeText.startsSentence(tokens, at: index)
        }
        return false
    }
}
