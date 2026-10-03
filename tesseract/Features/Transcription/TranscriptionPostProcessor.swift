//
//  TranscriptionPostProcessor.swift
//  tesseract
//
//  The regex cleanup: the fixed pass every take goes through after Whisper and
//  before **Learned Words** (PRD #612). It tidies what Whisper wrote
//  (whitespace, known hallucinations, punctuation spacing, stutters, sentence
//  capitals) and every rule is written so it cannot damage text Whisper got
//  right: file names, domains, decimals and abbreviations keep their dots,
//  Whisper's "..." stays a pause, and words people repeat on purpose stay
//  repeated. Silence turned into "Thank you." is not handled here; the
//  silent-capture skip (`CaptureLevel`) keeps such captures from being
//  transcribed at all.
//

import Foundation

nonisolated struct TranscriptionPostProcessor: Sendable {
    func process(_ text: String) -> String {
        let text = collapsingWhitespace(removingHallucinations(collapsingWhitespace(text)))
        guard !text.isEmpty else { return "" }

        var words = text.split(separator: " ").map(Word.init)
        words = attachingDetachedPunctuation(words)
        words = words.flatMap(separatingGluedPunctuation)
        words = words.map(normalizingPunctuationRuns)
        words = collapsingStutters(words)
        words = capitalizing(words)

        // Nothing but punctuation left ("...", "?"): there is nothing to commit.
        guard words.contains(where: { !$0.core.isEmpty }) else { return "" }
        return words.map(\.text).joined(separator: " ")
    }

    // MARK: - Words

    /// One whitespace-delimited token. The core runs from its first letter or
    /// digit to its last, so inner punctuation stays in the core ("cloud.md",
    /// "3.5", "e.g", "main.swift:42", "don't") and only the punctuation around
    /// it is leading or trailing. A token that is all punctuation has an empty
    /// core and keeps its text in `leading`.
    private nonisolated struct Word {
        var leading: String
        var core: String
        var trailing: String

        var text: String { leading + core + trailing }

        init(_ token: Substring) {
            guard let first = token.firstIndex(where: Self.isWordCharacter),
                let last = token.lastIndex(where: Self.isWordCharacter)
            else {
                leading = String(token)
                core = ""
                trailing = ""
                return
            }
            leading = String(token[..<first])
            core = String(token[first...last])
            trailing = String(token[token.index(after: last)...])
        }

        static func isWordCharacter(_ character: Character) -> Bool {
            character.isLetter || character.isNumber
        }
    }

    // MARK: - Whitespace and hallucinations

    private func collapsingWhitespace(_ text: String) -> String {
        text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Phrases Whisper invents from silence or music, removed wherever they
    /// appear (case-sensitive, as Whisper writes them).
    private static let hallucinations = [
        "Thank you for watching.",
        "Thanks for watching.",
        "Please subscribe.",
        "Like and subscribe.",
        "[Music]",
        "[Applause]",
        "[Laughter]",
        "(upbeat music)",
        "(gentle music)",
    ]

    private func removingHallucinations(_ text: String) -> String {
        Self.hallucinations.reduce(text) { result, hallucination in
            result.replacingOccurrences(of: hallucination, with: "")
        }
    }

    // MARK: - Punctuation

    private static let detachablePunctuation: Set<Character> = [
        ".", ",", "!", "?", ";", ":", "…",
    ]

    /// "hello , world ." becomes "hello, world.": a token that is only
    /// punctuation joins the word before it. A token with a word in it stays
    /// where it is, so "the .env file" keeps its space.
    private func attachingDetachedPunctuation(_ words: [Word]) -> [Word] {
        var result: [Word] = []
        for word in words {
            if word.core.isEmpty, word.leading.allSatisfy(Self.detachablePunctuation.contains),
                let last = result.indices.last
            {
                // A word that is all punctuation keeps its text in `leading`.
                if result[last].core.isEmpty {
                    result[last].leading += word.leading
                } else {
                    result[last].trailing += word.leading
                }
            } else {
                result.append(word)
            }
        }
        return result
    }

    /// Punctuation between two word characters with no space after it is part
    /// of the word (cloud.md, example.com, Node.js, main.swift:42, 3.5, e.g,
    /// 10:30): it is never split and never ends a sentence. Three shapes get a
    /// space, because none of them occurs inside a file name, a domain or a
    /// number:
    ///
    /// - "," ";" "!" "?" right before a letter ("hello,world").
    /// - ":" between two letters ("Note:this"), but not before a digit
    ///   ("10:30", "main.swift:42").
    /// - "." between a lowercase word of two or more letters and a capitalized
    ///   word, the whole token being exactly that ("end.Then"). A lowercase
    ///   part after the dot ("cloud.md") or more dots ("e.g.") is a name.
    ///
    /// Tokens that look like addresses or code (a "/", "\", "@", "=", "_" or
    /// "::" anywhere: URLs, paths, emails, query strings, identifiers) are left
    /// exactly as written. Whisper's long pause glued between two words
    /// ("now......then") becomes "now... then".
    private func separatingGluedPunctuation(_ word: Word) -> [Word] {
        var core = word.core
        guard core.contains(where: Self.gluedPunctuation.contains) else { return [word] }

        core = core.replacingOccurrences(
            of: "(?<=\\p{L})\\.{4,}(?=\\p{L})", with: "... ", options: .regularExpression)

        let looksLikeAddress =
            core.contains(where: Self.addressCharacters.contains) || core.contains("::")
        if !looksLikeAddress {
            core = core.replacingOccurrences(
                of: "([,;!?])(?=\\p{L})", with: "$1 ", options: .regularExpression)
            core = core.replacingOccurrences(
                of: "(?<=\\p{L}):(?=\\p{L})", with: ": ", options: .regularExpression)
            core = core.replacingOccurrences(
                of: "^(\\p{Ll}{2,})\\.(\\p{Lu}\\p{Ll}+)$", with: "$1. $2",
                options: .regularExpression)
        }

        guard core != word.core else { return [word] }
        return (word.leading + core + word.trailing).split(separator: " ").map(Word.init)
    }

    private static let gluedPunctuation: Set<Character> = [".", ",", "!", "?", ";", ":", "…"]
    private static let addressCharacters: Set<Character> = ["/", "\\", "@", "=", "_"]

    /// Collapses a run of end punctuation after a word (or a word of its
    /// own): "really??" is "really?", "stop!!!" is "stop!", "?!" keeps its
    /// last mark, ".." is ".", and four or more dots are Whisper's pause
    /// "...". "..." and "…" stay as written. Runs inside a word ("1...5") are
    /// left alone.
    private func normalizingPunctuationRuns(_ word: Word) -> Word {
        var word = word
        if word.core.isEmpty {
            word.leading = Self.normalizingRuns(in: word.leading)
        } else {
            word.trailing = Self.normalizingRuns(in: word.trailing)
        }
        return word
    }

    private static func normalizingRuns(in punctuation: String) -> String {
        guard punctuation.count > 1 else { return punctuation }
        var result = ""
        var run = ""
        for character in punctuation {
            if runPunctuation.contains(character) {
                run.append(character)
            } else {
                result += normalizedRun(run)
                run = ""
                result.append(character)
            }
        }
        return result + normalizedRun(run)
    }

    private static let runPunctuation: Set<Character> = [".", "!", "?", "…"]

    private static func normalizedRun(_ run: String) -> String {
        guard run.count > 1 else { return run }
        if let mark = run.last(where: { $0 == "!" || $0 == "?" }) { return String(mark) }
        if run.contains("…") { return "…" }
        return run.count == 2 ? "." : "..."
    }

    // MARK: - Repetition

    /// Words people say twice (or more) on purpose: intensifiers, answers and
    /// interjections, sounds, digits read one by one, and "had had". A stutter
    /// on one of these stays, which costs less than deleting a word that was
    /// meant: a stutter is Whisper's text as heard, a deleted word is an error
    /// the cleanup made.
    private static let deliberateRepeats: Set<String> = [
        // Intensifiers and amounts.
        "very", "really", "so", "much", "many", "more", "too", "far", "long", "way",
        "big", "super",
        // Answers, interjections and calls.
        "no", "yes", "yeah", "yep", "nope", "ok", "okay", "oh", "ah", "ha", "haha",
        "hey", "hi", "bye", "wow", "yay", "well", "now", "there", "right", "sorry",
        "please", "wait", "stop", "go", "again",
        // Sounds and doubled words.
        "la", "na", "blah", "bla", "yada", "knock", "tick", "ding", "beep", "bang",
        "boom", "chop", "night",
        // Digits read one at a time.
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        // Grammatical doubles.
        "had",
    ]

    /// Collapses a word said again right away ("the the meeting" is "the
    /// meeting", "it it." is "it."). Repetition survives when:
    ///
    /// - the first copy has punctuation after it ("very, very", "Yes. Yes."),
    /// - the word is on the deliberate list ("very very", "no no no"),
    /// - it is a number ("4 4 7"),
    /// - both copies are capitalized, so it is a name ("Bora Bora"); "I I"
    ///   still collapses.
    private func collapsingStutters(_ words: [Word]) -> [Word] {
        var result: [Word] = []
        for word in words {
            if let previous = result.last, isStutter(previous, word) {
                result[result.count - 1].trailing = word.trailing
            } else {
                result.append(word)
            }
        }
        return result
    }

    private func isStutter(_ first: Word, _ second: Word) -> Bool {
        guard !first.core.isEmpty, first.trailing.isEmpty, second.leading.isEmpty else {
            return false
        }
        let bare = first.core.lowercased()
        guard bare == second.core.lowercased(),
            first.core.contains(where: \.isLetter),
            !first.core.contains(where: \.isNumber),
            !Self.deliberateRepeats.contains(bare)
        else { return false }
        let bothCapitalized =
            first.core.first?.isUppercase == true && second.core.first?.isUppercase == true
        return !bothCapitalized || bare == "i"
    }

    // MARK: - Capitalization

    /// Capitalizes the first word of the take and the first word after a
    /// sentence end, and the pronoun "i". Only a plain lowercase word is
    /// capitalized ("then", "don't"), never one with capitals inside
    /// ("iPhone", "macOS") or inner punctuation ("cloud.md", "e.g."). A
    /// sentence ends at ".", "!" or "?" followed by a space; Whisper's "..."
    /// and "…" are pauses, so the word after one keeps Whisper's case, and an
    /// abbreviation's dot ("e.g.", "a.m.", "vs.") ends nothing.
    private func capitalizing(_ words: [Word]) -> [Word] {
        var words = words
        var capitalizeNext = true
        for index in words.indices {
            if words[index].core.isEmpty {
                // A bare "..." before the first word continues a thought.
                if Self.endsWithPause(words[index].leading) {
                    capitalizeNext = false
                } else if Self.endsSentence(punctuation: words[index].leading) {
                    capitalizeNext = true
                }
                continue
            }
            if capitalizeNext, !Self.endsWithPause(words[index].leading) {
                words[index].core = Self.capitalizingPlainWord(words[index].core)
            }
            words[index].core = Self.capitalizingPronounI(words[index].core)
            capitalizeNext = Self.endsSentence(words[index])
        }
        return words
    }

    private static let closers: Set<Character> = ["\"", "'", "”", "’", ")", "]", "}", "»"]

    /// Abbreviations whose dot ends no sentence, lowercased without the dot.
    /// Dotted ones ("e.g.", "a.m.", "U.S.") are caught by their inner dot.
    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "cf", "approx",
        "incl", "eg", "ie", "al", "viz",
    ]

    private static func endsSentence(_ word: Word) -> Bool {
        guard endsSentence(punctuation: word.trailing) else { return false }
        if word.trailing.contains("!") || word.trailing.contains("?") { return true }
        return !word.core.contains(".") && !abbreviations.contains(word.core.lowercased())
    }

    private static func endsSentence(punctuation: String) -> Bool {
        guard let mark = punctuation.last(where: { !closers.contains($0) }) else {
            return false
        }
        return (mark == "!" || mark == "?" || mark == ".") && !endsWithPause(punctuation)
    }

    private static func endsWithPause(_ punctuation: String) -> Bool {
        var tail = Substring(punctuation)
        while let last = tail.last, closers.contains(last) { tail = tail.dropLast() }
        return tail.hasSuffix("...") || tail.hasSuffix("…")
    }

    private static func capitalizingPlainWord(_ core: String) -> String {
        guard let first = core.first, first.isLowercase,
            !core.contains(where: \.isUppercase),
            core.allSatisfy({ $0.isLetter || $0.isNumber || "'’-".contains($0) })
        else { return core }
        return first.uppercased() + core.dropFirst()
    }

    /// "i" and its contractions ("i'm", "i'll", "i've", "i'd"), nothing else:
    /// "i.e." and "pi" are left alone.
    private static func capitalizingPronounI(_ core: String) -> String {
        guard core.first == "i" else { return core }
        if core == "i" { return "I" }
        let rest = core.dropFirst()
        guard let apostrophe = rest.first, apostrophe == "'" || apostrophe == "’",
            ["m", "ll", "ve", "d"].contains(rest.dropFirst().lowercased())
        else { return core }
        return "I" + rest
    }
}
