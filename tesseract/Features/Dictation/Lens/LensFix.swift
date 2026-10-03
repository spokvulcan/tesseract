//
//  LensFix.swift
//  tesseract
//
//  The fix (PRD #612): type the word you meant and the Lens finds the word
//  it replaces. Pure decisions over a take's tokens:
//
//  - completion from the owner's words while typing;
//  - the target: the span of up to three words that sounds most like what
//    was typed, already-right words skipped, spelling breaking ties;
//  - what a fix teaches: a new Learned Word, a word left alone in an app
//    (a Learned Word fixed back), or nothing (an ordinary word, a rewrite);
//  - the edit itself, keeping the punctuation around the word and the
//    take's catches in place.
//

import Foundation

nonisolated enum LensFix {

    /// A run of consecutive tokens.
    struct Span: Equatable, Hashable, Sendable {
        let start: Int
        let count: Int

        var range: Range<Int> { start..<(start + count) }
    }

    /// What a fix teaches.
    enum Decision: Equatable, Sendable {
        /// The span already reads the word.
        case unchanged
        /// A mishearing: learn "heard → meant".
        case learn(heard: String, meant: String)
        /// A Learned Word fixed back to what was heard: it is left alone in
        /// the take's app from now on.
        case leaveAlone(wordID: UUID, catchIndex: Int)
        /// A Learned Word fixed to a third word: left alone in the app, and
        /// this take fixed.
        case leaveAloneAndFix(wordID: UUID, catchIndex: Int)
        /// An ordinary word or a rewrite: this take only, never learned.
        case thisTakeOnly
    }

    /// One fix applied to a take's text.
    struct Edit: Equatable, Sendable {
        let text: String
        /// The take's catches, minus any the fix replaced, positions shifted.
        let catches: [LearnedWordCatch]
        /// The words the fix replaced, as they stood.
        let replaced: String
        /// What the fix wrote, case fitted.
        let written: String
        /// Where the written words now sit.
        let span: Span
    }

    /// Function words: never learned, never completed. A fix to one of
    /// these is a typo in this take, not a word the recognizer mishears.
    static let ordinaryWords: Set<String> = Set(
        """
        a about above after again all also am an and any are as at be because been before being \
        below between both but by can could did do does doing down during each even few for from \
        further had has have having he her here hers herself him himself his how i if in into is it \
        its itself just let me more most my myself no nor not now of off on once only or other our \
        ours ourselves out over own same she should so some such than that the their theirs them \
        themselves then there these they this those through to too under until up us very was we \
        were what when where which while who whom why will with would yes yet you your yours \
        yourself yourselves ok okay oh uh um yeah \
        whether whose since though although unless whereas else every either neither per via \
        upon within without among across along around behind beside beyond toward towards onto \
        may might must shall ought need much many less least lot lots anyone someone everyone \
        something anything nothing everything somewhere anywhere everywhere \
        i'm i'll i've i'd you're you'll you've you'd he's he'll he'd she's she'll she'd it's \
        it'll we're we'll we've we'd they're they'll they've they'd that's there's here's \
        what's where's who's how's let's don't doesn't didn't can't couldn't won't wouldn't \
        shouldn't isn't aren't wasn't weren't haven't hasn't hadn't
        """.split(whereSeparator: \.isWhitespace).map(String.init))

    // MARK: - Completion

    /// The owner's word that `typed` completes to, from `vocabulary` in
    /// priority order: a case-insensitive match first (typing "claude"
    /// writes "Claude"), then the first word it starts. Typing a trailing
    /// space keeps exactly what was typed; ordinary words and one letter
    /// never complete.
    static func completion(for typed: String, vocabulary: [String]) -> String? {
        guard !typed.hasSuffix(" ") else { return nil }
        let text = typed.trimmingCharacters(in: .whitespaces)
        guard text.count >= 2 else { return nil }
        let lower = text.lowercased()
        if let exact = vocabulary.first(where: { $0.lowercased() == lower }) {
            return exact == text ? nil : exact
        }
        guard !ordinaryWords.contains(lower) else { return nil }
        let candidates = vocabulary.filter {
            $0.count > text.count && $0.lowercased().hasPrefix(lower)
        }
        return candidates.first { $0.hasPrefix(text) } ?? candidates.first
    }

    /// What a fix writes for `typed`: its completion, or the text as typed.
    static func word(for typed: String, vocabulary: [String]) -> String {
        completion(for: typed, vocabulary: vocabulary)
            ?? typed.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Target

    /// The span the owner's word most likely replaces: up to three words
    /// that sound like it (typed or completed), skipping spans that already
    /// read it and tokens already fixed. A span with more words than the
    /// owner's word pays for each extra one and may not start or end on an
    /// ordinary word ("the SRACT" is not a mishearing of "Tesseract").
    /// Ties go to the closer spelling, then the shorter span, then the
    /// earlier one. Nil when nothing sounds like it.
    static func target(
        for word: String, typed: String, in tokens: [TakeToken], skipping fixed: Set<Int> = []
    ) -> Span? {
        // The completed word, not the typed prefix: "tess" sounds more like
        // "this" than like "SRAX", but "Tesseract" does not.
        let probe = word.isEmpty ? typed.trimmingCharacters(in: .whitespaces) : word
        guard !probe.isEmpty else { return nil }
        let wordBare = TakeText.normalized(word.isEmpty ? probe : word)
        let wordParts = wordBare.split(separator: " ").map(String.init)
        let wordCount = max(1, wordParts.count)
        // The owner's word with the wrong case ("QWEN" for "Qwen") is the
        // mistake before any word that only sounds like it.
        if let cased = casingTarget(for: probe, in: tokens, skipping: fixed) { return cased }
        // Tokens that already read the word, whole or as one word of a
        // several-word word ("PR" in "A PR"): no span holding one of them
        // is the mistake.
        var reading: Set<Int> = []
        if !wordParts.isEmpty, wordParts.count <= tokens.count {
            for start in 0...(tokens.count - wordParts.count)
            where tokens[start..<(start + wordParts.count)].map(\.bare) == wordParts {
                reading.formUnion(start..<(start + wordParts.count))
            }
        }
        var best: (span: Span, score: Double)?
        for start in tokens.indices {
            for count in 1...3 {
                let span = Span(start: start, count: count)
                guard span.range.upperBound <= tokens.count,
                    isJoinable(tokens[span.range]),
                    !span.range.contains(where: fixed.contains)
                else { break }
                let slice = tokens[span.range]
                if span.range.contains(where: reading.contains) { continue }
                if count > wordCount,
                    let first = slice.first, let last = slice.last,
                    ordinaryWords.contains(first.bare) || ordinaryWords.contains(last.bare)
                {
                    continue
                }
                let text = slice.map(\.coreText).joined(separator: " ")
                let sound = SoundAlike.similarity(text, probe)
                guard sound >= SoundAlike.threshold else { continue }
                // The recognizer writes a word it does not know in capitals
                // (SRAX, APR): between two equally close words, that one.
                let unknown = slice.contains { Self.isShouted($0.coreText) } ? 0.03 : 0
                let score =
                    sound + 0.05 * SoundAlike.spelling(text, probe) + unknown
                    - 0.1 * Double(abs(count - wordCount)) - 0.001 * Double(count - 1)
                if score > (best?.score ?? 0) {
                    best = (span, score)
                }
            }
        }
        return best?.span
    }

    /// A word written the owner's way but cased wrong ("tesseract" for
    /// "Tesseract"): picked before anything that only sounds like it.
    private static func casingTarget(
        for word: String, in tokens: [TakeToken], skipping fixed: Set<Int>
    ) -> Span? {
        let count = max(1, word.split(whereSeparator: \.isWhitespace).count)
        guard count <= tokens.count else { return nil }
        for start in 0...(tokens.count - count) {
            let span = Span(start: start, count: count)
            guard !span.range.contains(where: fixed.contains), isJoinable(tokens[span.range])
            else { continue }
            let text = tokens[span.range].map(\.coreText).joined(separator: " ")
            if text != word, text.lowercased() == word.lowercased(),
                text
                    != TakeText.fitCase(
                        word, atSentenceStart: TakeText.startsSentence(tokens, at: start))
            {
                return span
            }
        }
        return nil
    }

    /// Two or more letters, all capitals: how Whisper writes a word it does
    /// not know.
    private static func isShouted(_ word: String) -> Bool {
        let letters = word.filter(\.isLetter)
        return letters.count >= 2 && letters.allSatisfy(\.isUppercase)
    }

    /// Words that can be fixed as one: all words, with no punctuation
    /// between them.
    static func isJoinable(_ slice: ArraySlice<TakeToken>) -> Bool {
        for (offset, token) in slice.enumerated() {
            guard token.isWord else { return false }
            if offset < slice.count - 1, token.core.upperBound != token.range.upperBound {
                return false
            }
            if offset > 0, token.core.lowerBound != token.range.lowerBound {
                return false
            }
        }
        return true
    }

    // MARK: - Decision

    /// What replacing `span` with `word` teaches.
    static func decide(
        _ span: Span, word: String, tokens: [TakeToken], catches: [LearnedWordCatch],
        isDictionaryWord: (String) -> Bool = { _ in false }
    ) -> Decision {
        guard span.range.upperBound <= tokens.count, !span.range.isEmpty else { return .unchanged }
        let slice = tokens[span.range]
        let current = slice.map(\.coreText).joined(separator: " ")
        // The span reads the word as the fix would write it: "A PR" at the
        // start of a sentence already reads "a PR".
        let fitted = TakeText.fitCase(
            word, atSentenceStart: TakeText.startsSentence(tokens, at: span.start))
        guard current != word, current != fitted else { return .unchanged }
        if let index = catches.firstIndex(where: { $0.tokenRange.overlaps(span.range) }) {
            let item = catches[index]
            let wordForm = TakeText.normalized(word)
            let meantForm = TakeText.normalized(item.meant)
            // Only the casing changes ("Worktree" at a sentence start): the
            // Learned Word was right.
            if wordForm == meantForm { return .thisTakeOnly }
            if wordForm == TakeText.normalized(item.heard) {
                return .leaveAlone(wordID: item.wordID, catchIndex: index)
            }
            // The fix keeps the Learned Word and adds to it ("Claude code" →
            // "Claude Code"): a new word, learned from what was really heard
            // ("cloud code"), and the caught word stays on.
            if contains(words: wordForm, within: meantForm) {
                let heard = heardForm(of: span, tokens: tokens, replacing: item)
                return shouldLearn(heard: heard, meant: word, isDictionaryWord: isDictionaryWord)
                    ? .learn(heard: heard, meant: word) : .thisTakeOnly
            }
            return .leaveAloneAndFix(wordID: item.wordID, catchIndex: index)
        }
        let heard = TakeText.bare(slice)
        return shouldLearn(heard: heard, meant: word, isDictionaryWord: isDictionaryWord)
            ? .learn(heard: heard, meant: word) : .thisTakeOnly
    }

    /// Whether `inner`'s words appear, in order and together, in `outer`.
    private static func contains(words outer: String, within inner: String) -> Bool {
        let outerWords = outer.split(separator: " ")
        let innerWords = inner.split(separator: " ")
        guard !innerWords.isEmpty, outerWords.count > innerWords.count else { return false }
        for start in 0...(outerWords.count - innerWords.count)
        where Array(outerWords[start..<(start + innerWords.count)]) == innerWords {
            return true
        }
        return false
    }

    /// The span as the recognizer heard it: a caught word written back as
    /// what it caught.
    private static func heardForm(
        of span: Span, tokens: [TakeToken], replacing item: LearnedWordCatch
    ) -> String {
        var words: [String] = []
        for index in span.range {
            if item.tokenRange.contains(index) {
                if index == max(item.tokenStart, span.start) {
                    words.append(TakeText.normalized(item.heard))
                }
            } else {
                words.append(tokens[index].bare)
            }
        }
        return words.filter { !$0.isEmpty }.joined(separator: " ")
    }

    /// Whether a fix from `heard` to `meant` is a mishearing worth learning:
    /// short on both sides, not made of ordinary words, not a change of
    /// word ending ("drops" → "drop" is grammar), and either a casing fix or
    /// two forms that sound alike. Anything else is a rewrite.
    static func shouldLearn(
        heard: String, meant: String, isDictionaryWord: (String) -> Bool = { _ in false }
    ) -> Bool {
        let heardWords = TakeText.tokens(heard).filter(\.isWord).map(\.bare)
        let meantWords = TakeText.tokens(meant).filter(\.isWord).map(\.bare)
        guard !heardWords.isEmpty, !meantWords.isEmpty, heardWords.count <= 3,
            meantWords.count <= 4,
            // A single letter heard ("m" for "em") would rewrite every lone
            // letter in every take.
            heardWords.joined().count >= 2
        else { return false }
        if heardWords.allSatisfy(ordinaryWords.contains)
            || meantWords.allSatisfy(ordinaryWords.contains)
        {
            return false
        }
        // Numbers are written either way on purpose ("two" or "2"): a rule
        // would rewrite every later one. A number inside a term ("D flash
        // two" → DFlash2) is still learned.
        if heardWords.allSatisfy(isNumber) || meantWords.allSatisfy(isNumber) { return false }
        // Contractions are grammar ("well" → "we'll").
        if heardWords.count == 1, meantWords.count == 1,
            heard.contains("'") || meant.contains("'") || meant.contains("’")
        {
            return false
        }
        // A casing fix is learned when it gives a word of the owner's its
        // capitals ("testflight" → "TestFlight"); a dictionary word with a
        // capital ("go" → "Go") is a proper noun in that sentence only, and
        // taking capitals away ("Claude" → "claude") depends on the sentence.
        if heardWords == meantWords {
            guard meant != meant.lowercased() else { return false }
            return hasOwnShape(meant) || !heardWords.allSatisfy(isDictionaryWord)
        }
        // Two dictionary words that sound alike are a homophone the sentence
        // decides ("weather" → "whether"), not a mishearing of a name.
        if heardWords.count == 1, meantWords.count == 1, meant == meant.lowercased(),
            isDictionaryWord(heardWords[0]), isDictionaryWord(meantWords[0])
        {
            return false
        }
        // Part of the owner's word heard as a word of its own ("flight" for
        // TestFlight across a comma): learning it would rewrite every
        // "flight".
        let heardFlat = heardWords.joined()
        let meantFlat = meantWords.joined()
        // A gap of a letter or two is a mishearing ("M dashes" for "em
        // dashes"); a whole missing word is a part.
        if meantFlat.count >= heardFlat.count + 3,
            meantFlat.hasPrefix(heardFlat) || meantFlat.hasSuffix(heardFlat)
        {
            return false
        }
        if heardWords.count == meantWords.count,
            zip(heardWords, meantWords).allSatisfy({ $0 == $1 || isInflection($0, $1) })
        {
            return false
        }
        return SoundAlike.soundsAlike(heard, meant)
    }

    private static let endings = ["'s", "s", "es", "ed", "d", "ing", "er", "ers", "ly"]

    private static let numberWords: Set<String> = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
        "million", "billion", "first", "second", "third",
    ]

    private static func isNumber(_ word: String) -> Bool {
        numberWords.contains(word)
            || (!word.isEmpty && word.allSatisfy { $0.isNumber || $0 == "." || $0 == "," })
    }

    /// Capitals inside the word, a digit or a dot: a spelling only a name or
    /// a term has (iPhone, DFlash2, CLAUDE.md).
    private static func hasOwnShape(_ word: String) -> Bool {
        word.dropFirst().contains(where: \.isUppercase) || word.contains(where: \.isNumber)
            || word.contains(".")
    }

    /// Two forms of one word that differ only in a common ending, including
    /// a doubled last letter ("run" → "running") and a dropped silent e
    /// ("make" → "making").
    static func isInflection(_ a: String, _ b: String) -> Bool {
        guard a != b else { return false }
        let (short, long) = a.count <= b.count ? (a, b) : (b, a)
        guard short.count >= 3 else { return false }
        func stem(_ word: String) -> String {
            for ending in endings.sorted(by: { $0.count > $1.count }) where word.hasSuffix(ending) {
                let base = String(word.dropLast(ending.count))
                if base.count >= 3 { return base }
            }
            return word
        }
        if long.hasPrefix(short) {
            let rest = long.dropFirst(short.count)
            if endings.contains(String(rest)) { return true }
            if let last = short.last, rest.first == last,
                endings.contains(String(rest.dropFirst()))
            {
                return true
            }
        }
        if short.hasSuffix("e"), long.hasPrefix(short.dropLast()),
            endings.contains(String(long.dropFirst(short.count - 1)))
        {
            return true
        }
        return stem(a) == stem(b)
    }

    // MARK: - Edit

    /// Writes `word` over `span`, case fitted at the start of a sentence,
    /// keeping the punctuation around the span. Catches the span overlaps
    /// are dropped; later ones shift by the change in word count.
    static func replace(
        _ span: Span, with word: String, in text: String, catches: [LearnedWordCatch]
    ) -> Edit? {
        let tokens = TakeText.tokens(text)
        guard !span.range.isEmpty, span.range.upperBound <= tokens.count,
            let range = TakeText.coreRange(tokens[span.range])
        else { return nil }
        let written = TakeText.fitCase(
            word.trimmingCharacters(in: .whitespaces),
            atSentenceStart: TakeText.startsSentence(tokens, at: span.start))
        guard !written.isEmpty else { return nil }
        var output = text
        output.replaceSubrange(range, with: written)
        let newCount = max(1, written.split(whereSeparator: \.isWhitespace).count)
        let delta = newCount - span.count
        var kept: [LearnedWordCatch] = []
        for var item in catches where !item.tokenRange.overlaps(span.range) {
            if item.tokenStart >= span.range.upperBound { item.tokenStart += delta }
            kept.append(item)
        }
        return Edit(
            text: output, catches: kept, replaced: String(text[range]), written: written,
            span: Span(start: span.start, count: newCount))
    }

    /// Token indices after an edit: those inside the replaced span go,
    /// later ones shift by the change in word count.
    static func shifted(_ indices: Set<Int>, by edit: Edit, replacing old: Span) -> Set<Int> {
        let delta = edit.span.count - old.count
        return Set(
            indices.compactMap { index in
                if old.range.contains(index) { return nil }
                return index >= old.range.upperBound ? index + delta : index
            })
    }
}
