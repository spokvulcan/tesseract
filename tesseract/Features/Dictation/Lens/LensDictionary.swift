//
//  LensDictionary.swift
//  tesseract
//
//  Whether a word is an ordinary dictionary word, for the Lens's learning
//  gates (PRD #612): two dictionary words that sound alike are a homophone
//  the sentence decides ("weather" → "whether"), and a dictionary word given
//  a capital ("go" → "Go") is a proper noun in that sentence only. Neither is
//  learned. The system spell checker knows the owner's languages.
//

import AppKit

nonisolated enum LensDictionary {

    /// Whether the system spell checker accepts `word`. Main thread only
    /// (the spell checker is AppKit's); the Lens runs there.
    static func contains(_ word: String) -> Bool {
        guard !word.isEmpty else { return false }
        return MainActor.assumeIsolated {
            NSSpellChecker.shared.checkSpelling(of: word, startingAt: 0).location == NSNotFound
        }
    }
}
