//
//  LearnedWordStore.swift
//  tesseract
//
//  The **Learned Word** store (PRD #612): the local list of "heard → meant"
//  replacements the owner's fixes taught, applied by the Voice Capture
//  Session to every take after the regex cleanup. JSON on disk beside the
//  Correction Pairs; every disk failure is logged and swallowed, because
//  learning must never break dictation.
//
//  Every change that the Lens offers to undo returns a receipt holding the
//  words it touched as they were, so Undo restores them exactly.
//

import Foundation
import Observation

/// What the Voice Capture Session needs from the Learned Words: apply them
/// to a take, and count what they caught once the take commits.
@MainActor
protocol LearnedWordApplying: AnyObject {
    func apply(to text: String, appBundleID: String?) -> LearnedWordsApplication
    func recordCatches(_ catches: [LearnedWordCatch])
}

@MainActor
@Observable
final class LearnedWordStore: LearnedWordApplying {

    /// The words a change touched, as they were before it (`nil` for a word
    /// the change created). Undo puts them back.
    struct Receipt: Equatable, Sendable {
        fileprivate let previous: [UUID: LearnedWord?]
        /// The list's order before the change, so Undo puts a word a fix
        /// moved to the front back where it stood.
        fileprivate let order: [UUID]
        /// The word the change was about.
        let wordID: UUID
    }

    /// Newest first.
    private(set) var words: [LearnedWord] = []

    private let storageURL: URL
    private let now: @MainActor () -> Date
    private let calendar: Calendar

    /// - Parameters:
    ///   - directory: storage directory; defaults to the app-support home
    ///     the Correction Pairs use. Injectable for tests.
    ///   - now: the clock catches and learning are stamped with.
    init(
        directory: URL? = nil, now: @escaping @MainActor () -> Date = { Date() },
        calendar: Calendar = .current
    ) {
        let base =
            directory
            ?? StorageEnvironment.applicationSupport.appendingPathComponent(
                "Tesseract Agent", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.storageURL = base.appendingPathComponent("learned_words.json")
        self.now = now
        self.calendar = calendar
        loadFromDisk()
    }

    /// The words that apply anywhere: learned and not forgotten.
    var activeWords: [LearnedWord] { words.filter { !$0.isForgotten } }

    func word(withID id: UUID) -> LearnedWord? {
        words.first { $0.id == id }
    }

    /// The active word a normalized heard form belongs to.
    func word(heard form: String) -> LearnedWord? {
        let normalized = TakeText.normalized(form)
        return activeWords.first { $0.heard.contains(normalized) }
    }

    // MARK: - Applying

    func apply(to text: String, appBundleID: String?) -> LearnedWordsApplication {
        LearnedWordMatcher.apply(
            LearnedWordMatcher.rules(for: words, in: appBundleID), to: text)
    }

    /// Counts a committed take's catches under today.
    func recordCatches(_ catches: [LearnedWordCatch]) {
        guard !catches.isEmpty else { return }
        let date = now()
        let day = LearnedWord.dayKey(for: date, calendar: calendar)
        for item in catches {
            guard let index = words.firstIndex(where: { $0.id == item.wordID }) else { continue }
            words[index].catchesByDay[day, default: 0] += 1
            words[index].lastCaughtAt = date
        }
        saveToDisk()
    }

    /// Takes back one catch the owner fixed back: it was not a mistake.
    func uncount(_ item: LearnedWordCatch, caughtAt date: Date) {
        guard let index = words.firstIndex(where: { $0.id == item.wordID }) else { return }
        let day = LearnedWord.dayKey(for: date, calendar: calendar)
        guard let count = words[index].catchesByDay[day], count > 0 else { return }
        words[index].catchesByDay[day] = count > 1 ? count - 1 : nil
        saveToDisk()
    }

    // MARK: - Learning

    /// Learns "heard → meant" from a fix. A meant spelling the store already
    /// knows gains the heard form (and comes back if it was forgotten: a new
    /// fix by the owner is the one thing that relearns a word); a heard form
    /// that belonged to another word moves to this one, since the latest
    /// fix wins. Returns nil when there is nothing to learn.
    @discardableResult
    func learn(
        heard: String, meant: String, pairID: UUID? = nil,
        example: LearnedWord.Example? = nil
    ) -> Receipt? {
        let form = TakeText.normalized(heard)
        let spelling = meant.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !form.isEmpty, !spelling.isEmpty, form != spelling else { return nil }

        let order = words.map(\.id)
        var previous: [UUID: LearnedWord?] = [:]
        // The form leaves any other word it belonged to.
        for index in words.indices
        where words[index].meant != spelling && words[index].heard.contains(form) {
            previous[words[index].id] = words[index]
            words[index].heard.removeAll { $0 == form }
            if words[index].heard.isEmpty, words[index].forgottenAt == nil {
                words[index].forgottenAt = now()
            }
        }

        let wordID: UUID
        if let index = words.firstIndex(where: { $0.meant == spelling }) {
            wordID = words[index].id
            previous[wordID] = words[index]
            if !words[index].heard.contains(form) {
                words[index].heard.append(form)
            }
            words[index].fixes += 1
            words[index].forgottenAt = nil
            // A word revived by a new fix applies everywhere again only
            // where the owner hasn't left it alone.
            let word = words.remove(at: index)
            words.insert(word, at: 0)
        } else {
            let word = LearnedWord(
                meant: spelling, heard: [form], fixes: 1, learnedAt: now(),
                sourcePairID: pairID, example: example)
            wordID = word.id
            previous[wordID] = .some(nil)
            words.insert(word, at: 0)
        }
        saveToDisk()
        return Receipt(previous: previous, order: order, wordID: wordID)
    }

    /// Leaves a word alone in an app: the owner fixed it back there.
    @discardableResult
    func leaveAlone(_ id: UUID, in app: LearnedWord.App) -> Receipt? {
        guard let index = words.firstIndex(where: { $0.id == id }) else { return nil }
        let receipt = Receipt(previous: [id: words[index]], order: words.map(\.id), wordID: id)
        if !words[index].leftAloneIn.contains(where: { $0.bundleID == app.bundleID }) {
            words[index].leftAloneIn.append(app)
        }
        saveToDisk()
        return receipt
    }

    /// Lets a word apply in an app again.
    func applyAgain(_ id: UUID, in bundleID: String) {
        guard let index = words.firstIndex(where: { $0.id == id }) else { return }
        words[index].leftAloneIn.removeAll { $0.bundleID == bundleID }
        saveToDisk()
    }

    /// Forgets a word: it stops applying, and stays listed so the Forget
    /// can be undone.
    @discardableResult
    func forget(_ id: UUID) -> Receipt? {
        guard let index = words.firstIndex(where: { $0.id == id }), !words[index].isForgotten
        else { return nil }
        let receipt = Receipt(previous: [id: words[index]], order: words.map(\.id), wordID: id)
        words[index].forgottenAt = now()
        saveToDisk()
        return receipt
    }

    /// Restores every word a change touched to how it was before, where it
    /// stood in the list. A word the change created goes.
    func undo(_ receipt: Receipt) {
        words.removeAll { receipt.previous.keys.contains($0.id) }
        let position = Dictionary(
            receipt.order.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        let restored = receipt.previous.values.compactMap { $0 }
            .sorted { (position[$0.id] ?? 0) < (position[$1.id] ?? 0) }
        // Ascending old positions: each insert lands where the word stood,
        // since the words before it are back in place.
        for word in restored {
            words.insert(word, at: min(position[word.id] ?? 0, words.count))
        }
        saveToDisk()
    }

    /// Removes a word for good. Not offered in the UI (Forget keeps the
    /// word); here for tests and data hygiene.
    func delete(_ id: UUID) {
        words.removeAll { $0.id == id }
        saveToDisk()
    }

    // MARK: - Persistence

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: storageURL.path) else { return }
        do {
            let data = try Data(contentsOf: storageURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            words = try decoder.decode([LearnedWord].self, from: data)
        } catch {
            // Kept aside rather than overwritten by the next save: the words
            // are the owner's fixes, and a bad file can still be recovered.
            let aside = storageURL.deletingPathExtension()
                .appendingPathExtension("unreadable-\(Int(now().timeIntervalSince1970)).json")
            try? FileManager.default.moveItem(at: storageURL, to: aside)
            Log.transcription.error(
                "Failed to load learned words, kept the file as \(aside.lastPathComponent): \(error)"
            )
        }
    }

    private func saveToDisk() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(words)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            Log.transcription.error("Failed to save learned words: \(error)")
        }
    }
}
