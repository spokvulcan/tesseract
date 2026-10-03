//
//  CorrectionPairStoreTests.swift
//  tesseractTests
//
//  The **Correction Pair** store (ticket #289) against a temporary directory:
//  recording + persistence across reload, the gold-last bound, flag/correction
//  gold transitions, the protected-audio set the Capture Dump eviction reads,
//  and the JSONL export shape (#294's input).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CorrectionPairStoreTests {

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("correction-pairs-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makePair(
        raw: String = "helo world",
        committed: String? = "hello world",
        verdict: CorrectionPair.Verdict = .corrected,
        flagged: Bool = false,
        correction: String? = nil,
        audio: String? = nil
    ) -> CorrectionPair {
        CorrectionPair(
            rawASR: raw,
            cleaned: raw,
            proofread: verdict == .corrected ? committed : nil,
            verdict: verdict,
            committed: committed,
            correction: correction,
            flaggedWrong: flagged,
            conditions: CorrectionPair.Conditions(
                duration: 2.0, language: "en", asrModel: "Whisper Turbo"),
            audioFileName: audio
        )
    }

    @Test func recordsNewestFirstAndPersistsAcrossReload() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CorrectionPairStore(directory: directory)
        let first = makePair(raw: "first take")
        let second = makePair(raw: "second take")
        store.record(first)
        store.record(second)

        #expect(store.pairs.map(\.id) == [second.id, first.id])

        let reloaded = CorrectionPairStore(directory: directory)
        #expect(reloaded.pairs.map(\.id) == [second.id, first.id])
        #expect(reloaded.pair(withID: first.id)?.rawASR == "first take")
    }

    /// The bound evicts the oldest *non-gold* pair first — gold pairs are the
    /// collection's point and outlive candidates.
    @Test func boundEvictsOldestNonGoldFirst() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CorrectionPairStore(directory: directory, maxPairs: 2)
        let gold = makePair(raw: "gold", flagged: true)
        let candidate = makePair(raw: "candidate")
        let newest = makePair(raw: "newest")
        store.record(gold)
        store.record(candidate)
        store.record(newest)

        #expect(store.pairs.map(\.rawASR) == ["newest", "gold"])
    }

    @Test func flagWrongMakesThePairGoldAndProtectsItsAudio() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CorrectionPairStore(directory: directory)
        let pair = makePair(audio: "capture-1.wav")
        store.record(pair)
        #expect(store.protectedAudioFileNames.isEmpty)

        store.flagWrong(pair.id)

        #expect(store.pair(withID: pair.id)?.flaggedWrong == true)
        #expect(store.pair(withID: pair.id)?.isGold == true)
        #expect(store.protectedAudioFileNames == ["capture-1.wav"])
    }

    @Test func settingACorrectionMakesGoldAndClearingReverts() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CorrectionPairStore(directory: directory)
        let pair = makePair(audio: "capture-2.wav")
        store.record(pair)

        store.setCorrection("hello world, corrected by hand", for: pair.id)
        #expect(store.pair(withID: pair.id)?.correction == "hello world, corrected by hand")
        #expect(store.protectedAudioFileNames == ["capture-2.wav"])

        store.setCorrection("   ", for: pair.id)
        #expect(store.pair(withID: pair.id)?.correction == nil)
        #expect(store.protectedAudioFileNames.isEmpty)
    }

    @Test func exportIsOneDecodableJSONObjectPerLineOldestFirst() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CorrectionPairStore(directory: directory)
        let first = makePair(raw: "first")
        let second = makePair(raw: "second")
        store.record(first)
        store.record(second)

        let jsonl = try store.exportJSONL()
        let lines = String(decoding: jsonl, as: UTF8.self)
            .split(separator: "\n").map(String.init)
        #expect(lines.count == 2)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try lines.map {
            try decoder.decode(CorrectionPair.self, from: Data($0.utf8))
        }
        #expect(decoded.map(\.rawASR) == ["first", "second"])
    }

    // MARK: - Fixes in the Lens (PRD #612)

    /// A fix in the Lens turns the pair gold, records the heard and meant
    /// words with how and where, and keeps the whole corrected take.
    @Test func aFixInTheLensMakesThePairGold() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = CorrectionPairStore(directory: directory)
        let pair = makePair(raw: "ask cloud", committed: "Ask cloud", verdict: .skipped)
        store.record(pair)
        #expect(store.pair(withID: pair.id)?.isGold == false)

        let fix = CorrectionPair.Fix(
            heard: "cloud", meant: "Claude", how: .afterPaste, app: "com.apple.Terminal",
            at: Date(timeIntervalSince1970: 1_000))
        store.recordFix(fix, correctedText: "Ask Claude", for: pair.id)

        let reloaded = try #require(CorrectionPairStore(directory: directory).pair(withID: pair.id))
        #expect(reloaded.isGold)
        #expect(reloaded.fixes == [fix])
        #expect(reloaded.correction == "Ask Claude")
    }

    /// Pairs written before PRD #612 (no `learned`, no `fixes`) still load.
    @Test func pairsFromBeforeLearnedWordsStillLoad() throws {
        let json = """
            {"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","timestamp":"2026-09-01T10:00:00Z",
             "rawASR":"helo","cleaned":"Helo","verdict":"skipped","committed":"Helo",
             "flaggedWrong":false,"conditions":{"duration":1,"language":"en","asrModel":"W"}}
            """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let pair = try decoder.decode(CorrectionPair.self, from: Data(json.utf8))
        #expect(pair.learned == nil)
        #expect(pair.fixes.isEmpty)
        #expect(!pair.isGold)
    }
}
