//
//  LearnedWord.swift
//  tesseract
//
//  A **Learned Word** (PRD #612): one "heard → meant" replacement learned
//  from the owner's fix, applied to every take after the regex cleanup.
//  It keeps the ways it was heard, the apps it is left alone in, how many
//  times a day it caught a mistake, and the take that taught it. A
//  **Catch** is one application of a Learned Word to a take.
//

import Foundation

nonisolated struct LearnedWord: Codable, Equatable, Identifiable, Sendable {

    /// An app the word is left alone in: the owner fixed it back there, so
    /// it no longer applies there.
    struct App: Codable, Equatable, Hashable, Sendable {
        let bundleID: String
        var name: String
    }

    /// The fix that taught the word, in context: a few words of the take
    /// before and after it, for the catch record's before-and-after strip.
    struct Example: Codable, Equatable, Sendable {
        let before: String
        let after: String
    }

    let id: UUID
    /// The owner's spelling: what every heard form becomes.
    var meant: String
    /// The ways it was heard, each normalized by `TakeText.normalized`
    /// (lowercased bare words joined by single spaces), first learned first.
    var heard: [String]
    var leftAloneIn: [App]
    /// Catches per local day, keyed "yyyy-MM-dd".
    var catchesByDay: [String: Int]
    /// The owner's fixes that taught it.
    var fixes: Int
    let learnedAt: Date
    var lastCaughtAt: Date?
    /// The Correction Pair of the take whose fix first taught it.
    var sourcePairID: UUID?
    var example: Example?
    /// Set when the owner forgets the word. A forgotten word is kept, so
    /// the Forget can be undone; it applies again only after a new fix.
    var forgottenAt: Date?

    init(
        id: UUID = UUID(), meant: String, heard: [String], leftAloneIn: [App] = [],
        catchesByDay: [String: Int] = [:], fixes: Int = 1, learnedAt: Date = Date(),
        lastCaughtAt: Date? = nil, sourcePairID: UUID? = nil, example: Example? = nil,
        forgottenAt: Date? = nil
    ) {
        self.id = id
        self.meant = meant
        self.heard = heard
        self.leftAloneIn = leftAloneIn
        self.catchesByDay = catchesByDay
        self.fixes = fixes
        self.learnedAt = learnedAt
        self.lastCaughtAt = lastCaughtAt
        self.sourcePairID = sourcePairID
        self.example = example
        self.forgottenAt = forgottenAt
    }

    var isForgotten: Bool { forgottenAt != nil }
    var totalCatches: Int { catchesByDay.values.reduce(0, +) }

    func isLeftAlone(in bundleID: String?) -> Bool {
        guard let bundleID else { return false }
        return leftAloneIn.contains { $0.bundleID == bundleID }
    }

    /// The day key catches are counted under.
    static func dayKey(for date: Date, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
    }
}

/// One application of a Learned Word to a take.
nonisolated struct LearnedWordCatch: Codable, Equatable, Sendable {
    let wordID: UUID
    /// What the recognizer wrote, as it stood in the take.
    let heard: String
    /// What replaced it, case fitted.
    let meant: String
    /// Where the replacement sits in the output text, in `TakeText` tokens.
    var tokenStart: Int
    let tokenCount: Int

    var tokenRange: Range<Int> { tokenStart..<(tokenStart + tokenCount) }
}

/// A take's text after its Learned Words, and what they caught.
nonisolated struct LearnedWordsApplication: Equatable, Sendable {
    let text: String
    let catches: [LearnedWordCatch]

    static func unchanged(_ text: String) -> LearnedWordsApplication {
        LearnedWordsApplication(text: text, catches: [])
    }
}

/// Applies Learned Words to a text: whole tokens, case-insensitive,
/// longest heard form first, never overlapping, case fitted at the start
/// of a sentence. Pure; the store builds its rules.
nonisolated enum LearnedWordMatcher {

    struct Rule: Equatable, Sendable {
        let wordID: UUID
        /// The normalized heard form's words.
        let heard: [String]
        let meant: String
    }

    /// The rules of every word that applies in `bundleID`.
    static func rules(for words: [LearnedWord], in bundleID: String?) -> [Rule] {
        words.filter { !$0.isForgotten && !$0.isLeftAlone(in: bundleID) }
            .flatMap { word in
                word.heard.compactMap { form -> Rule? in
                    let parts = form.split(separator: " ").map(String.init)
                    guard !parts.isEmpty else { return nil }
                    return Rule(wordID: word.id, heard: parts, meant: word.meant)
                }
            }
            .sorted {
                $0.heard.count != $1.heard.count
                    ? $0.heard.count > $1.heard.count
                    : $0.heard.joined().count > $1.heard.joined().count
            }
    }

    static func apply(_ rules: [Rule], to text: String) -> LearnedWordsApplication {
        guard !rules.isEmpty else { return .unchanged(text) }
        let tokens = TakeText.tokens(text)
        guard !tokens.isEmpty else { return .unchanged(text) }

        struct Hit {
            let range: Range<String.Index>
            let replacement: String
            let heard: String
            let wordID: UUID
            let tokenStart: Int
            let tokenCount: Int
        }
        var hits: [Hit] = []
        var i = 0
        while i < tokens.count {
            guard let hit = match(rules, tokens, at: i),
                let range = TakeText.coreRange(tokens[i..<(i + hit.length)])
            else {
                i += 1
                continue
            }
            let (rule, n) = (hit.rule, hit.length)
            let found = String(text[range])
            let replacement = TakeText.fitCase(
                rule.meant, atSentenceStart: TakeText.startsSentence(tokens, at: i))
            if found != replacement {
                hits.append(
                    Hit(
                        range: range, replacement: replacement, heard: found,
                        wordID: rule.wordID, tokenStart: i, tokenCount: n))
            }
            i += n
        }
        guard !hits.isEmpty else { return .unchanged(text) }

        var output = text
        for hit in hits.reversed() {
            output.replaceSubrange(hit.range, with: hit.replacement)
        }
        // Output token positions: each earlier hit shifts later ones by the
        // difference between its replacement's words and the words it took.
        var shift = 0
        var catches: [LearnedWordCatch] = []
        for hit in hits {
            let words = max(1, hit.replacement.split(whereSeparator: \.isWhitespace).count)
            catches.append(
                LearnedWordCatch(
                    wordID: hit.wordID, heard: hit.heard, meant: hit.replacement,
                    tokenStart: hit.tokenStart + shift, tokenCount: words))
            shift += words - hit.tokenCount
        }
        return LearnedWordsApplication(text: output, catches: catches)
    }

    /// Re-finds catches in a text a later stage rewrote (the Proofread Pass
    /// writes the committed text after the Learned Words ran): each catch
    /// moves to the next place its spelling still stands, in order, and a
    /// catch whose spelling is gone is dropped.
    static func relocate(_ catches: [LearnedWordCatch], in text: String) -> [LearnedWordCatch] {
        let tokens = TakeText.tokens(text)
        var next = 0
        var placed: [LearnedWordCatch] = []
        for item in catches {
            let n = item.tokenCount
            guard n > 0 else { continue }
            var start = next
            var found: Int?
            while start + n <= tokens.count {
                let words = tokens[start..<(start + n)].map(\.coreText).joined(separator: " ")
                if words == item.meant {
                    found = start
                    break
                }
                start += 1
            }
            guard let found else { continue }
            var moved = item
            moved.tokenStart = found
            placed.append(moved)
            next = found + n
        }
        return placed
    }

    /// The first rule whose heard words match the tokens at `index`. Inside
    /// a multi-word form, punctuation between the words breaks the match.
    private static func match(_ rules: [Rule], _ tokens: [TakeToken], at index: Int)
        -> (rule: Rule, length: Int)?
    {
        for rule in rules {
            let n = rule.heard.count
            guard index + n <= tokens.count else { continue }
            var matches = true
            for k in 0..<n {
                let token = tokens[index + k]
                guard token.isWord, token.bare == rule.heard[k] else {
                    matches = false
                    break
                }
                if k < n - 1, token.core.upperBound != token.range.upperBound {
                    matches = false
                    break
                }
                if k > 0, token.core.lowerBound != token.range.lowerBound {
                    matches = false
                    break
                }
            }
            if matches { return (rule: rule, length: n) }
        }
        return nil
    }
}
