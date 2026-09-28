//
//  DictationLab.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  The one in-memory lab the five Dictation prototypes share, so flipping
//  between them keeps what was learned: the recent takes (with where each
//  was inserted), the lexicon (what was learned), the lessons (a journal of
//  every fix and what it taught), and the toast the overlay shows. The
//  variants differ in how a fix is made and how learning is shown; the
//  learning itself is the same loop underneath:
//
//      take → lexicon.apply (learned replacements) + bias (vocabulary)
//      fix  → put back in the target app → diff → gate → learn → toast
//
//  Nothing persists across a relaunch (the prototype skill's rule). The
//  Correction Pair store still records every take as before.
//

import AppKit
import Observation
import SwiftUI

nonisolated enum DictationPrototypeVariant: Int, CaseIterable, Sendable {
    case fixBar, wordCards, teach, justEdit, sayItAgain, current

    var label: String {
        switch self {
        case .fixBar: "A · Fix Bar"
        case .wordCards: "B · Word Cards"
        case .teach: "C · Teach"
        case .justEdit: "D · Just Edit"
        case .sayItAgain: "E · Say It Again"
        case .current: "Current page"
        }
    }
}

nonisolated struct LabTake: Identifiable, Equatable, Sendable {
    let id: UUID
    /// The words as they stand now (after any fix).
    var text: String
    /// The words as first inserted.
    let original: String
    /// What the lexicon changed before insertion.
    let applied: [LabApplied]
    let appName: String
    let bundleID: String?
    let pid: pid_t
    let insertedAt: Date
    /// What sits in the target app now (the text plus the trailing space).
    var insertedText: String
    /// Key presses seen when the words landed; a fix compares against it.
    var keyCountAtInsertion: Int
    let pairID: UUID?
    var fixes: Int = 0
}

nonisolated struct LabLesson: Identifiable, Equatable, Sendable {
    let id: UUID
    let date: Date
    let hunk: LabHunk
    let source: LabSource
    let appName: String
    /// False when the change looked like a rewrite and was not learned.
    let learned: Bool
    var undone: Bool = false
}

nonisolated struct LabToast: Identifiable, Equatable, Sendable {
    let id = UUID()
    let title: String
    var detail: String? = nil
    var lessonIDs: [UUID] = []
    var isWarning = false
}

@Observable @MainActor
final class DictationLab {
    static let shared = DictationLab()

    var variant: DictationPrototypeVariant = .fixBar {
        didSet { variantChanged(from: oldValue) }
    }

    private(set) var lexicon = LabLexicon() {
        didSet { publishBias() }
    }
    private(set) var takes: [LabTake] = []
    private(set) var lessons: [LabLesson] = []
    /// The latest learning event, for the overlay cards and the pages.
    private(set) var toast: LabToast?
    /// Suggested terms from the living memory, not yet accepted.
    private(set) var memorySuggestions: [String] = []

    /// The old Proofread Pass. Off in the lab (errors note §1.2).
    var usesProofreadPass = false
    var biasesRecognizer = true {
        didSet { publishBias() }
    }

    let panels = LabPanelHost()
    private(set) var replacer: LabTextReplacer?
    private(set) var watcher: LabEditWatcher?
    private(set) var voice: LabVoiceFix?
    private var keyDownCount: @MainActor () -> Int = { 0 }
    private var settings: SettingsManager?
    private(set) var feed: DictationFeed?
    private var beatTask: Task<Void, Never>?
    private var toastTask: Task<Void, Never>?

    // MARK: Wiring

    func attach(
        feed: DictationFeed, coordinator: DictationCoordinator, injector: any TextInjecting,
        settings: SettingsManager, keyDownCount: @escaping @MainActor () -> Int,
        voice: LabVoiceFix
    ) {
        self.feed = feed
        self.settings = settings
        self.keyDownCount = keyDownCount
        self.replacer = LabTextReplacer(injector: injector, keyDownCount: keyDownCount)
        self.watcher = LabEditWatcher(lab: self)
        self.voice = voice
        voice.lab = self
        // Start in the variant the switcher last showed, overlay included,
        // so the lab's cards never double up with the classic pill's beat.
        variant =
            DictationPrototypeVariant(
                rawValue: UserDefaults.standard.integer(forKey: "dictationPrototype.variant"))
            ?? .fixBar
        LabOverlay.apply(variant, settings: settings)
        beatTask = Task { [weak self] in
            var lastID: UInt64?
            for await beat in Observations({ feed.beat }) {
                guard let self, let beat, beat.id != lastID else { continue }
                lastID = beat.id
                if case .committed(let text, _, let edits) = beat.outcome {
                    self.didCommit(text: text, edits: edits, pairID: coordinator.lastTakePairID)
                }
            }
        }
        publishBias()
    }

    // MARK: Pipeline

    /// Applies what the lab learned to a fresh take (after the regex
    /// cleanup, before the commit). The edits ride the commit beat.
    func refine(_ text: String) -> (text: String, edits: [WordEdit]) {
        let result = lexicon.apply(to: text)
        guard !result.applied.isEmpty else { return (text, []) }
        lexicon.noteApplied(result.applied)
        return (
            result.text, result.applied.map { WordEdit(original: $0.heard, replacement: $0.term) }
        )
    }

    private func didCommit(text: String, edits: [WordEdit], pairID: UUID?) {
        let front = NSWorkspace.shared.frontmostApplication
        let applied = edits.map { LabApplied(heard: $0.original, term: $0.replacement) }
        let take = LabTake(
            id: UUID(), text: text, original: text, applied: applied,
            appName: front?.localizedName ?? "the app", bundleID: front?.bundleIdentifier,
            pid: front?.processIdentifier ?? 0, insertedAt: Date(), insertedText: text + " ",
            keyCountAtInsertion: keyDownCount(), pairID: pairID)
        takes.insert(take, at: 0)
        if takes.count > 60 { takes.removeLast() }
        // The paste's own Cmd+V reaches the key counter a moment later.
        Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            self?.resnapshotKeys(for: take.id)
        }
        if variant == .justEdit { watcher?.watch(take) }
        showCommitCard(for: take)
    }

    private func resnapshotKeys(for id: UUID) {
        guard let index = takes.firstIndex(where: { $0.id == id }) else { return }
        takes[index].keyCountAtInsertion = keyDownCount()
    }

    var lastTake: LabTake? { takes.first }

    // MARK: Fix

    /// A fix of a whole take: puts the corrected words back where they were
    /// (only for the latest take, and only safely), then learns every small
    /// change that sounds like the original. `allowedKeys` is what the fix
    /// gesture itself pressed.
    @discardableResult
    func fix(
        _ takeID: UUID, to corrected: String, source: LabSource, allowedKeys: Int = 1,
        putBack: Bool = true
    ) async -> LabReplaceOutcome? {
        guard let index = takes.firstIndex(where: { $0.id == takeID }) else { return nil }
        let take = takes[index]
        let corrected = corrected.trimmingCharacters(in: .whitespacesAndNewlines)
        guard corrected != take.text, !corrected.isEmpty else { return nil }
        var outcome: LabReplaceOutcome?
        if putBack, index == 0, let replacer {
            replacer.restoreClipboard(settings?.restoreClipboard ?? true)
            outcome = await replacer.replace(take, with: corrected, allowedKeys: allowedKeys)
        }
        takes[index].text = corrected
        takes[index].insertedText = corrected + " "
        takes[index].fixes += 1
        takes[index].keyCountAtInsertion = keyDownCount()
        Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            self?.resnapshotKeys(for: takeID)
        }
        learn(
            before: take.text, after: corrected, source: source, appName: take.appName,
            outcome: outcome)
        return outcome
    }

    /// Learns from a change without touching any app (a selection taught
    /// through Teach, an edit the watcher saw, a fix on the page).
    func learn(
        before: String, after: String, source: LabSource, appName: String,
        outcome: LabReplaceOutcome? = nil
    ) {
        let hunks = LabDiff.hunks(from: before, to: after)
        guard !hunks.isEmpty else { return }
        var newLessons: [LabLesson] = []
        for hunk in hunks {
            let isCorrection = hunk.isCorrection
            var learned = false
            if isCorrection { learned = lexicon.learn(hunk, source: source) }
            newLessons.append(
                LabLesson(
                    id: UUID(), date: Date(), hunk: hunk, source: source, appName: appName,
                    learned: learned))
        }
        lessons.insert(contentsOf: newLessons, at: 0)
        let learned = newLessons.filter(\.learned)
        var toast: LabToast
        if learned.isEmpty {
            toast = LabToast(
                title: "Fixed", detail: "Looked like a rewrite, so nothing was learned")
        } else {
            let names = learned.map { "\($0.hunk.before) → \($0.hunk.after)" }.joined(
                separator: ", ")
            toast = LabToast(title: "Learned", detail: names, lessonIDs: learned.map(\.id))
        }
        if case .copied(let reason) = outcome {
            toast.detail =
                (toast.detail.map { $0 + ". " } ?? "")
                + "Couldn't put it back (\(reason)); the fix is on the clipboard"
            toast.isWarning = true
        }
        showToast(toast)
    }

    /// Learns a fix of a selection made anywhere: the selected text becomes
    /// the typed text, in place, and the change is learned.
    func teach(selection: String, to corrected: String, appName: String) async {
        let corrected = corrected.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !corrected.isEmpty, corrected != selection else { return }
        // A terminal's selection is for copying: a paste there types at the
        // cursor instead of replacing it, so only an editable field is fixed.
        let editable = LabAX.focusedElement().map(LabAX.isEditableText) ?? false
        if editable {
            replacer?.restoreClipboard(settings?.restoreClipboard ?? true)
            _ = await replacer?.replaceSelection(with: corrected)
        }
        let hunk = LabHunk(before: selection, after: corrected)
        if LabDiff.hunks(from: selection, to: corrected).count <= 1,
            hunk.isCorrection
                || selection.split(separator: " ").count <= 3
        {
            let learned = lexicon.learn(hunk, source: .teach)
            let lesson = LabLesson(
                id: UUID(), date: Date(), hunk: hunk, source: .teach, appName: appName,
                learned: learned)
            lessons.insert(lesson, at: 0)
            showToast(
                LabToast(
                    title: learned ? "Learned" : "Fixed",
                    detail: "\(selection) → \(corrected)"
                        + (editable
                            ? ""
                            : ". \(appName) can't replace a selection, so the text stays as it was"),
                    lessonIDs: learned ? [lesson.id] : []))
        } else {
            learn(before: selection, after: corrected, source: .teach, appName: appName)
        }
    }

    func undo(_ lessonIDs: [UUID]) {
        for id in lessonIDs {
            guard let index = lessons.firstIndex(where: { $0.id == id }), !lessons[index].undone
            else { continue }
            lessons[index].undone = true
            lexicon.unlearn(lessons[index].hunk)
        }
        showToast(LabToast(title: "Undone", detail: "It won't be learned again"))
    }

    func add(term: String, source: LabSource = .typed) {
        lexicon.add(term: term, source: source)
        memorySuggestions.removeAll { $0 == term }
    }

    func dismissSuggestion(_ term: String) { memorySuggestions.removeAll { $0 == term } }
    func forget(_ id: UUID) { lexicon.forget(id) }
    func removeForm(_ heard: String, from id: UUID) { lexicon.removeForm(heard, from: id) }

    func suggestions(for word: String) -> [String] { lexicon.suggestions(for: word) }

    /// The word most likely to be wrong in a take: one that sounds like a
    /// known term, or one no dictionary knows, or nil.
    func suspect(in text: String) -> Int? {
        let words = text.split(separator: " ").map(String.init)
        if let i = words.firstIndex(where: { !lexicon.suggestions(for: $0, limit: 1).isEmpty }) {
            return i
        }
        let checker = NSSpellChecker.shared
        return words.firstIndex { word in
            let bare = word.trimmingCharacters(in: .punctuationCharacters)
            guard bare.count > 2 else { return false }
            return checker.checkSpelling(of: bare, startingAt: 0).location != NSNotFound
        }
    }

    // MARK: Toast

    func showToast(_ toast: LabToast) {
        self.toast = toast
        showToastCard()
        toastTask?.cancel()
        toastTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(6))
            guard !Task.isCancelled else { return }
            self?.toast = nil
        }
    }

    // MARK: Sample (snapshots only)

    /// Fills the lab with a few made-up takes and fixes, for the snapshot
    /// test and previews. Never called by the app.
    func seedSample() {
        let now = Date()
        for (before, after, source) in [
            ("cloud.md", "CLAUDE.md", LabSource.fixBar), ("sract", "Tesseract", .teach),
            ("whisper flow", "Wispr Flow", .watched), ("QWEN", "Qwen", .voice),
        ] {
            let hunk = LabHunk(before: before, after: after)
            lexicon.learn(hunk, source: source)
            lessons.append(
                LabLesson(
                    id: UUID(), date: now.addingTimeInterval(-Double(lessons.count) * 400),
                    hunk: hunk,
                    source: source,
                    appName: ["Terminal", "Notes", "Slack", "Mail"][lessons.count % 4],
                    learned: true))
        }
        lessons.append(
            LabLesson(
                id: UUID(), date: now.addingTimeInterval(-2400),
                hunk: LabHunk(before: "so please take a look", after: "please review"),
                source: .watched,
                appName: "Notes", learned: false))
        lexicon.add(term: "DFlash2", source: .typed)
        memorySuggestions = ["Yalantis", "KiwiCache", "MyEx", "Jira"]
        let texts: [(String, [LabApplied])] = [
            (
                "Please update the global CLAUDE.md file and open the PR in Tesseract.",
                [
                    LabApplied(heard: "cloud.md", term: "CLAUDE.md"),
                    LabApplied(heard: "sract", term: "Tesseract"),
                ]
            ),
            ("Can you run the Vitest browser mode only locally for now?", []),
            (
                "We should add some features from Wispr Flow to format lists.",
                [LabApplied(heard: "whisper flow", term: "Wispr Flow")]
            ),
        ]
        for (index, item) in texts.enumerated() {
            takes.append(
                LabTake(
                    id: UUID(), text: item.0, original: item.0, applied: item.1,
                    appName: "Terminal",
                    bundleID: nil, pid: 0, insertedAt: now.addingTimeInterval(-Double(index) * 300),
                    insertedText: item.0 + " ", keyCountAtInsertion: 0, pairID: nil))
        }
    }

    // MARK: Private

    private func publishBias() {
        LabBias.shared.set(biasesRecognizer ? lexicon.biasTerms : [])
    }

    private func variantChanged(from old: DictationPrototypeVariant) {
        panels.closeKey()
        panels.closeCard()
        watcher?.stop()
    }

    /// CamelCase, capitalized and versioned words from memory: names and
    /// projects the owner already talks about.
    static func candidateTerms(in texts: [String]) -> [String] {
        let common: Set<String> = [
            "The", "This", "That", "He", "She", "It", "I", "We", "They", "When", "What", "Where",
            "Correcting", "NOT", "Use", "Yes", "No", "AI", "UI", "OK",
        ]
        var counts: [String: Int] = [:]
        for text in texts {
            let words = text.split(whereSeparator: { $0 == " " || $0 == "\n" })
            for (index, raw) in words.enumerated() {
                let word = raw.trimmingCharacters(in: .punctuationCharacters.subtracting(["."]))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "."))
                guard word.count > 2, !common.contains(word) else { continue }
                let camel = word.dropFirst().contains(where: \.isUppercase)
                let capitalizedMidSentence = index > 0 && word.first?.isUppercase == true
                if camel || capitalizedMidSentence || word.contains(where: \.isNumber) {
                    counts[word, default: 0] += 1
                }
            }
        }
        return counts.filter { $0.value >= 2 }.sorted { $0.value > $1.value }.prefix(24).map(\.key)
    }
}
