//
//  LensModel.swift
//  tesseract
//
//  The **Lens**'s state while a take is fixed (PRD #612): the take as it
//  stands after each fix, what the owner is typing, the word it will
//  replace, and what each fix taught. Each fix lands in the stores as it is
//  made: the Learned Word (or the app it is now left alone in), the take's
//  gold **Correction Pair**, and its history entry. Putting the fixed text
//  back in the app is the controller's job; this model never touches a
//  window or another app, so the whole flow is testable with real stores.
//

import Foundation
import Observation

@Observable
@MainActor
final class LensModel {

    /// How the take reached the Lens.
    enum Mode: Equatable, Sendable {
        /// ⌃⌥Space reopened the last take after it was pasted.
        case afterPaste
        /// A take opened from the Dictation page: nothing is pasted back.
        case fromPage

        var how: CorrectionPair.Fix.How {
            switch self {
            case .afterPaste: .afterPaste
            case .fromPage: .page
            }
        }
    }

    enum Phase: Equatable, Sendable {
        case hidden
        /// The take is open and the Lens has the keyboard.
        case fixing
        /// The fix is done; the Lens shows what happened, then fades.
        case done
    }

    /// What the last fix taught, shown beside the take.
    enum Note: Equatable, Sendable {
        case learned(heard: String, meant: String)
        /// A Learned Word fixed back: left alone in the app from now on.
        case leftAlone(word: String, app: String)
        case thisTakeOnly
        case undone
    }

    /// The line the Lens shows once the fix is done.
    struct Result: Equatable, Sendable {
        let line: String
        let detail: String?
    }

    private(set) var phase: Phase = .hidden
    private(set) var mode: Mode = .afterPaste
    /// The take as it was opened.
    private(set) var take: DictatedTake?
    /// The take's text with every fix so far.
    private(set) var text = ""
    private(set) var catches: [LearnedWordCatch] = []
    private(set) var tokens: [TakeToken] = []
    /// What the owner is typing: the word they meant.
    var typed = "" {
        didSet { retarget() }
    }
    /// The owner's word `typed` completes to, when there is one.
    private(set) var completion: String?
    /// The words the typed word replaces.
    private(set) var target: LensFix.Span?
    /// Picked with ← → or a click: typing no longer moves it.
    private(set) var isManualTarget = false
    /// Tokens fixed in this session; never targeted again.
    private(set) var fixedTokens: Set<Int> = []
    private(set) var note: Note?
    private(set) var fixCount = 0
    /// What each learning fix can undo, newest last.
    private(set) var receipts: [LearnedWordStore.Receipt] = []
    private(set) var learnedPairs: [(heard: String, meant: String)] = []
    private(set) var result: Result?
    /// Bumped to put the keyboard in the Lens's field.
    private(set) var focusRequest = 0

    private var vocabulary: [String] = []

    @ObservationIgnored private let learnedWords: LearnedWordStore
    @ObservationIgnored private let pairs: CorrectionPairStore?
    @ObservationIgnored private let history: (any TranscriptionStoring)?
    @ObservationIgnored private let now: @MainActor () -> Date
    /// Whether a word is an ordinary dictionary word (the learning gates).
    @ObservationIgnored private let isDictionaryWord: (String) -> Bool

    init(
        learnedWords: LearnedWordStore, pairs: CorrectionPairStore?,
        history: (any TranscriptionStoring)?, now: @escaping @MainActor () -> Date = { Date() },
        isDictionaryWord: @escaping (String) -> Bool = LensDictionary.contains
    ) {
        self.learnedWords = learnedWords
        self.pairs = pairs
        self.history = history
        self.now = now
        self.isDictionaryWord = isDictionaryWord
    }

    // MARK: - Session

    func open(_ take: DictatedTake, mode: Mode, vocabulary: [String]) {
        self.take = take
        self.mode = mode
        self.vocabulary = vocabulary
        text = take.text
        catches = take.catches
        tokens = TakeText.tokens(take.text)
        fixedTokens = []
        note = nil
        fixCount = 0
        receipts = []
        learnedPairs = []
        result = nil
        isManualTarget = false
        target = nil
        typed = ""
        phase = .fixing
    }

    func requestFocus() {
        focusRequest += 1
    }

    /// The text the fixes produced differs from what was opened.
    var isChanged: Bool { take.map { $0.text != text } ?? false }

    /// The word a commit would write now.
    var word: String { LensFix.word(for: typed, vocabulary: vocabulary) }

    /// The rest of the completion after what was typed, for the ghost text.
    var completionSuffix: String {
        guard let completion else { return "" }
        let typedText = typed.trimmingCharacters(in: .whitespaces)
        guard completion.lowercased().hasPrefix(typedText.lowercased()) else { return "" }
        return String(completion.dropFirst(typedText.count))
    }

    /// The take's catches still standing, as token index sets.
    var caughtTokens: Set<Int> {
        Set(catches.flatMap { Array($0.tokenRange) })
    }

    func catchAt(_ index: Int) -> LearnedWordCatch? {
        catches.first { $0.tokenRange.contains(index) }
    }

    // MARK: - Picking

    /// A click on a word: it becomes the target and stays put while typing.
    func pick(_ index: Int) {
        guard phase == .fixing, tokens.indices.contains(index), tokens[index].isWord else { return }
        target = LensFix.Span(start: index, count: 1)
        isManualTarget = true
    }

    /// ← → walk the words; from nothing, they start at the last word.
    func moveTarget(by delta: Int) {
        guard phase == .fixing else { return }
        let words = tokens.indices.filter { tokens[$0].isWord }
        guard !words.isEmpty else { return }
        let next: Int
        if let current = target?.start, let position = words.firstIndex(of: current) {
            next = words[max(0, min(words.count - 1, position + delta))]
        } else {
            next = delta < 0 ? words[words.count - 1] : words[0]
        }
        target = LensFix.Span(start: next, count: 1)
        isManualTarget = true
    }

    /// ⇧← ⇧→ widen or narrow a picked target by a word, up to three words,
    /// so a mishearing split over several words can be fixed as one.
    func extendTarget(by delta: Int) {
        guard phase == .fixing else { return }
        guard let current = target else {
            moveTarget(by: delta)
            return
        }
        let start = current.start
        let count = current.count + delta
        guard count >= 1, count <= 3, start + count <= tokens.count,
            tokens[start..<(start + count)].allSatisfy(\.isWord)
        else { return }
        target = LensFix.Span(start: start, count: count)
        isManualTarget = true
    }

    private func retarget() {
        let trimmed = typed.trimmingCharacters(in: .whitespaces)
        completion = LensFix.completion(for: typed, vocabulary: vocabulary)
        guard !isManualTarget else { return }
        target =
            trimmed.isEmpty
            ? nil
            : LensFix.target(for: word, typed: typed, in: tokens, skipping: fixedTokens)
    }

    // MARK: - Fixing

    /// Esc: clears what was typed. False when there was nothing to clear.
    @discardableResult
    func clearTyping() -> Bool {
        guard !typed.isEmpty || isManualTarget else { return false }
        isManualTarget = false
        typed = ""
        return true
    }

    /// Commits the typed word over the target (⇥, or ↩ before finishing).
    /// Returns false when there was nothing to commit.
    @discardableResult
    func commit() -> Bool {
        guard phase == .fixing, let take, let span = target else { return false }
        let word = self.word
        guard !word.isEmpty else { return false }

        let decision = LensFix.decide(
            span, word: word, tokens: tokens, catches: catches, isDictionaryWord: isDictionaryWord)
        let caught = catches
        guard decision != .unchanged,
            let edit = LensFix.replace(span, with: word, in: text, catches: catches)
        else {
            resetTyping()
            return false
        }

        let before = text
        apply(edit, replacing: span)

        switch decision {
        case .learn(let heard, _):
            if let learned = learnedWords.activeWords.first(where: {
                TakeText.normalized($0.meant) == heard
                    && TakeText.normalized($0.meant) != TakeText.normalized(word)
            }) {
                // The words already read a Learned Word's spelling (written by
                // the recognizer itself, or untracked): fixing them is fixing
                // that word back. Learning the reverse would make the two
                // swap in every take.
                if learned.heard.contains(TakeText.normalized(word)),
                    let bundleID = take.app?.bundleID,
                    let receipt = learnedWords.leaveAlone(
                        learned.id,
                        in: LearnedWord.App(bundleID: bundleID, name: take.app?.name ?? bundleID))
                {
                    receipts.append(receipt)
                    note = .leftAlone(word: learned.meant, app: take.app?.name ?? "this app")
                } else {
                    note = .thisTakeOnly
                }
            } else if let receipt = learnedWords.learn(
                heard: heard, meant: word, pairID: take.pairID,
                example: example(before: before, after: text, old: span, new: edit.span))
            {
                receipts.append(receipt)
                learnedPairs.append((edit.replaced, word))
                fixOtherOccurrences(of: heard, with: word)
                note = .learned(heard: edit.replaced, meant: word)
            } else {
                note = .thisTakeOnly
            }
        case .leaveAlone(let wordID, let index), .leaveAloneAndFix(let wordID, let index):
            let item = caught[index]
            if let bundleID = take.app?.bundleID,
                let receipt = learnedWords.leaveAlone(
                    wordID,
                    in: LearnedWord.App(bundleID: bundleID, name: take.app?.name ?? bundleID))
            {
                receipts.append(receipt)
            }
            // The catch was a mistake of its own: it no longer counts.
            learnedWords.uncount(item, caughtAt: take.at)
            note = .leftAlone(word: item.meant, app: take.app?.name ?? "this app")
        case .thisTakeOnly, .unchanged:
            note = .thisTakeOnly
        }

        record(edit, how: mode.how)
        resetTyping()
        return true
    }

    /// The same mishearing elsewhere in this take, fixed with the word just
    /// learned (fixes, not catches: the owner made them).
    private func fixOtherOccurrences(of heard: String, with word: String) {
        let parts = heard.split(separator: " ").map(String.init)
        guard !parts.isEmpty else { return }
        var spans: [LensFix.Span] = []
        var i = 0
        while i + parts.count <= tokens.count {
            let span = LensFix.Span(start: i, count: parts.count)
            let slice = tokens[span.range]
            if zip(slice, parts).allSatisfy({ $0.bare == $1 && $0.isWord }),
                LensFix.isJoinable(slice),
                !span.range.contains(where: fixedTokens.contains),
                !span.range.contains(where: caughtTokens.contains)
            {
                spans.append(span)
                i += parts.count
            } else {
                i += 1
            }
        }
        for span in spans.reversed() {
            guard let edit = LensFix.replace(span, with: word, in: text, catches: catches) else {
                continue
            }
            apply(edit, replacing: span)
        }
    }

    private func apply(_ edit: LensFix.Edit, replacing span: LensFix.Span) {
        fixedTokens = LensFix.shifted(fixedTokens, by: edit, replacing: span)
            .union(edit.span.range)
        text = edit.text
        catches = edit.catches
        tokens = TakeText.tokens(edit.text)
        fixCount += 1
    }

    private func record(_ edit: LensFix.Edit, how: CorrectionPair.Fix.How) {
        guard let take, let pairID = take.pairID else { return }
        pairs?.recordFix(
            CorrectionPair.Fix(
                heard: edit.replaced, meant: edit.written, how: how, app: take.app?.bundleID,
                at: now()),
            correctedText: text, for: pairID)
        history?.replaceText(forPairID: pairID, with: text)
    }

    private func resetTyping() {
        isManualTarget = false
        typed = ""
        target = nil
    }

    /// A few words around the fix, before and after: the catch record's
    /// before-and-after strip.
    private func example(
        before: String, after: String, old: LensFix.Span, new: LensFix.Span
    ) -> LearnedWord.Example {
        func window(_ text: String, around span: LensFix.Span) -> String {
            let words = TakeText.tokens(text)
            let lower = max(0, span.start - 4)
            let upper = min(words.count, span.range.upperBound + 4)
            guard lower < upper else { return text }
            let body = words[lower..<upper].map(\.text).joined(separator: " ")
            return (lower > 0 ? "… " : "") + body + (upper < words.count ? " …" : "")
        }
        return LearnedWord.Example(
            before: window(before, around: old), after: window(after, around: new))
    }

    // MARK: - Undo

    /// Takes back what the fixes taught (the text stays as fixed: the owner
    /// typed it). The note's and the result line's Undo.
    func undoLearning() {
        for receipt in receipts.reversed() {
            learnedWords.undo(receipt)
        }
        receipts = []
        learnedPairs = []
        note = .undone
    }

    // MARK: - Ending

    /// The session ended; `result` is what the Lens says about it.
    func finish(_ result: Result) {
        resetTyping()
        self.result = result
        phase = .done
    }

    func close() {
        resetTyping()
        phase = .hidden
        take = nil
        result = nil
        note = nil
    }

    /// What the fixes taught, in one line: "Learned SRACT → Tesseract".
    var learnedSummary: String? {
        guard let first = learnedPairs.first else { return nil }
        let line = "Learned \(first.heard) → \(first.meant)"
        return learnedPairs.count > 1 ? "\(line) and \(learnedPairs.count - 1) more" : line
    }
}
