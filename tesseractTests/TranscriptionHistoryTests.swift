//
//  TranscriptionHistoryTests.swift
//  tesseractTests
//
//  The transcription history on disk (PRD #612), each test in its own
//  temporary directory: entries written before the catch record (no
//  catches, no app) still load and survive the next save, a take's catches
//  and app survive a reload, and a fix in the **Lens** rewrites the entry
//  linked to its **Correction Pair** and nothing else.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct TranscriptionHistoryTests {

    /// The file the history keeps its entries in.
    private static let fileName = "transcription_history.json"

    private static let terminal = TranscriptionEntry.App(
        bundleID: "com.apple.Terminal", name: "Terminal")

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("transcription-history-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func claudeCatch(at token: Int) -> LearnedWordCatch {
        LearnedWordCatch(
            wordID: UUID(), heard: "cloud", meant: "Claude", tokenStart: token, tokenCount: 1)
    }

    /// Two entries in the shape the history wrote before PRD #612: no
    /// `catches` and no `app`. The second also predates the Correction
    /// Pairs (ticket #289), so it has no `pairID` either.
    private func writeOldEntries(to directory: URL, id: UUID, pairID: UUID) throws {
        let json = """
            [
              {
                "id": "\(id.uuidString)", "text": "Ask cloud why.",
                "timestamp": 781000000, "duration": 2.5, "model": "Whisper Turbo",
                "pairID": "\(pairID.uuidString)"
              },
              {
                "id": "\(UUID().uuidString)", "text": "Ship it.",
                "timestamp": 780000000, "duration": 1, "model": "Whisper Turbo"
              }
            ]
            """
        try Data(json.utf8).write(to: directory.appendingPathComponent(Self.fileName))
    }

    // MARK: - Entries from before the catch record

    @Test func anOldEntryLoadsWithNoCatchesAndNoApp() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let id = UUID()
        let pairID = UUID()
        try writeOldEntries(to: directory, id: id, pairID: pairID)

        let history = TranscriptionHistory(directory: directory)

        #expect(history.entries.count == 2)
        let entry = try #require(history.entries.first)
        #expect(entry.id == id)
        #expect(entry.text == "Ask cloud why.")
        #expect(entry.timestamp == Date(timeIntervalSinceReferenceDate: 781_000_000))
        #expect(entry.duration == 2.5)
        #expect(entry.model == "Whisper Turbo")
        #expect(entry.pairID == pairID)
        #expect(entry.catches.isEmpty)
        #expect(entry.app == nil)
        let oldest = try #require(history.entries.last)
        #expect(oldest.pairID == nil)
        #expect(oldest.catches.isEmpty)
        #expect(oldest.app == nil)
    }

    @Test func oldEntriesSurviveTheNextSave() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let id = UUID()
        try writeOldEntries(to: directory, id: id, pairID: UUID())

        let history = TranscriptionHistory(directory: directory)
        history.add(
            text: "Ask Claude why.", duration: 2, model: "Whisper Turbo", pairID: UUID(),
            catches: [claudeCatch(at: 1)], app: Self.terminal)

        let reloaded = TranscriptionHistory(directory: directory)
        #expect(reloaded.entries.map(\.text) == ["Ask Claude why.", "Ask cloud why.", "Ship it."])
        #expect(reloaded.entries.dropFirst().first?.id == id)
    }

    // MARK: - Catches and app

    @Test func aTakesCatchesAndAppSurviveAReload() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let pairID = UUID()
        let catches = [claudeCatch(at: 1)]
        let notes = TranscriptionEntry.App(bundleID: nil, name: "Notes")

        let history = TranscriptionHistory(directory: directory)
        history.add(
            text: "Ask Claude why.", duration: 2.5, model: "Whisper Turbo", pairID: pairID,
            catches: catches, app: Self.terminal)
        history.add(
            text: "Ship it.", duration: 1, model: "Whisper Turbo", pairID: nil, catches: [],
            app: notes)

        let reloaded = TranscriptionHistory(directory: directory)
        #expect(reloaded.entries.count == 2)
        let latest = try #require(reloaded.entries.first)
        #expect(latest.text == "Ship it.")
        #expect(latest.catches.isEmpty)
        #expect(latest.app == notes)
        let entry = try #require(reloaded.entries.last)
        #expect(entry.text == "Ask Claude why.")
        #expect(entry.pairID == pairID)
        #expect(entry.catches == catches)
        #expect(entry.app == Self.terminal)
        #expect(entry.duration == 2.5)
        #expect(entry.model == "Whisper Turbo")
    }

    // MARK: - A fix in the Lens

    @Test func aFixRewritesTheTakesTextAndCatches() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let pairID = UUID()
        let other = UUID()

        let history = TranscriptionHistory(directory: directory)
        history.add(
            text: "Ask cloud why.", duration: 2, model: "Whisper Turbo", pairID: pairID,
            catches: [], app: Self.terminal)
        history.add(
            text: "Ship it.", duration: 1, model: "Whisper Turbo", pairID: other, catches: [],
            app: Self.terminal)
        let id = try #require(history.entries.last?.id)
        let catches = [claudeCatch(at: 1)]

        history.replaceText(forPairID: pairID, with: "Ask Claude why.", catches: catches)

        let fixed = try #require(history.entries.first { $0.pairID == pairID })
        #expect(fixed.id == id)
        #expect(fixed.text == "Ask Claude why.")
        #expect(fixed.catches == catches)
        #expect(fixed.app == Self.terminal)
        #expect(history.entries.first { $0.pairID == other }?.text == "Ship it.")
        // The history list shows the fixed text.
        let listed = history.flattenedItems.compactMap { item -> String? in
            guard case .entry(let entry, _, _) = item else { return nil }
            return entry.text
        }
        #expect(listed.contains("Ask Claude why."))
        #expect(!listed.contains("Ask cloud why."))

        let reloaded = TranscriptionHistory(directory: directory)
        let persisted = try #require(reloaded.entries.first { $0.pairID == pairID })
        #expect(persisted.id == id)
        #expect(persisted.text == "Ask Claude why.")
        #expect(persisted.catches == catches)
        #expect(reloaded.entries.first { $0.pairID == other }?.text == "Ship it.")
    }

    @Test func aFixForAnUnknownPairChangesNothing() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let catches = [claudeCatch(at: 1)]

        let history = TranscriptionHistory(directory: directory)
        history.add(
            text: "Ask Claude why.", duration: 2, model: "Whisper Turbo", pairID: UUID(),
            catches: catches, app: Self.terminal)
        history.add(
            text: "Ship it.", duration: 1, model: "Whisper Turbo", pairID: nil, catches: [],
            app: nil)

        history.replaceText(forPairID: UUID(), with: "Something else.", catches: [])

        #expect(history.entries.map(\.text) == ["Ship it.", "Ask Claude why."])
        #expect(history.entries.last?.catches == catches)
        let reloaded = TranscriptionHistory(directory: directory)
        #expect(reloaded.entries.map(\.text) == ["Ship it.", "Ask Claude why."])
        #expect(reloaded.entries.last?.catches == catches)
    }
}
