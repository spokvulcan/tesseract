//
//  WordTimeline.swift
//  tesseract
//
//  One segment's spoken text as words on a character line: where each word
//  starts and ends when the words are joined by single spaces, and which word
//  a character count falls in. The Read-Along places the heard character on
//  that line (`ReadAlongTimeline`); this turns it into a word. Built once per
//  segment, pure (CONTEXT.md → "Speech word timeline").
//

import Foundation

nonisolated struct WordTimeline: Equatable, Sendable {

    /// One whitespace-separated word and where it sits on the character line.
    struct Word: Equatable, Sendable {
        let text: String
        /// Character offset where this word starts (cumulative `word.count + 1`).
        let charOffset: Int
        /// `text.count`, precomputed.
        let charCount: Int
    }

    let words: [Word]
    /// Length of the words joined by single spaces — the character line.
    let totalCharCount: Int

    init(text: String) {
        // One shared definition of "a word" (see StringWordSplitting): the engine,
        // this line and the Reader's text ranges all split words the same way.
        var built: [Word] = []
        var offset = 0
        for word in text.splitIntoWords() {
            built.append(Word(text: String(word), charOffset: offset, charCount: word.count))
            offset += word.count + 1  // word characters plus one separating space
        }
        self.words = built
        // Σ(word.count + 1) less the last space (0 when empty).
        self.totalCharCount = max(0, offset - 1)
    }

    /// The word a highlighted character count falls in: the first whose end
    /// reaches it, clamping to the last word past the end of the text.
    func activeWordIndex(highlightedCharCount: Int) -> Int {
        guard !words.isEmpty else { return 0 }
        for (i, word) in words.enumerated()
        where highlightedCharCount <= word.charOffset + word.charCount {
            return i
        }
        return words.count - 1
    }
}
