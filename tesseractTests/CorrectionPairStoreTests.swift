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
}
