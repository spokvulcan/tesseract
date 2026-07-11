//
//  CorrectionPairStore.swift
//  tesseract
//
//  The **Correction Pair** store (map #283, ticket #289) — the local,
//  bounded training-pair collection the flywheel feeds from day one. JSON
//  on disk beside the transcription history; bounded by count with gold
//  pairs (owner-edited or flagged) evicted last; exportable as JSONL for
//  a future fine-tune (#294). Collection must never break dictation:
//  every disk failure is logged and swallowed.
//

import Foundation
import Observation

@MainActor
@Observable
final class CorrectionPairStore {

    private(set) var pairs: [CorrectionPair] = []

    private let maxPairs: Int
    private let storageURL: URL

    /// - Parameters:
    ///   - directory: storage directory; defaults to the app-support home the
    ///     transcription history also uses. Injectable for tests.
    ///   - maxPairs: the store bound. Text-only pairs are small; 1000 covers
    ///     months of daily use in well under a megabyte.
    init(directory: URL? = nil, maxPairs: Int = 1000) {
        self.maxPairs = maxPairs
        let base =
            directory
            ?? FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first?.appendingPathComponent("Tesseract Agent", isDirectory: true)
            ?? FileManager.default.temporaryDirectory
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.storageURL = base.appendingPathComponent("correction_pairs.json")
        loadFromDisk()
    }

    /// The Capture Dump files gold pairs reference — the dump's ring
    /// eviction skips these, so flagged/edited takes keep their audio.
    var protectedAudioFileNames: Set<String> {
        Set(pairs.filter(\.isGold).compactMap(\.audioFileName))
    }

    func pair(withID id: UUID) -> CorrectionPair? {
        pairs.first { $0.id == id }
    }

    /// Records one take. Newest first; over the bound, the oldest non-gold
    /// pair goes first (gold pairs are the collection's point — they outlive
    /// candidates).
    func record(_ pair: CorrectionPair) {
        pairs.insert(pair, at: 0)
        while pairs.count > maxPairs {
            if let index = pairs.lastIndex(where: { !$0.isGold }) {
                pairs.remove(at: index)
            } else {
                pairs.removeLast()
            }
        }
        saveToDisk()
    }

    /// The overlay's one-click "that was wrong". Idempotent.
    func flagWrong(_ id: UUID) {
        guard let index = pairs.firstIndex(where: { $0.id == id }) else { return }
        guard !pairs[index].flaggedWrong else { return }
        pairs[index].flaggedWrong = true
        saveToDisk()
    }

    /// Stores the owner's corrected text — the gold half of the pair. An
    /// empty or whitespace-only correction clears it (back to candidate).
    func setCorrection(_ correction: String, for id: UUID) {
        guard let index = pairs.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = correction.trimmingCharacters(in: .whitespacesAndNewlines)
        pairs[index].correction = trimmed.isEmpty ? nil : trimmed
        saveToDisk()
    }

    func delete(_ id: UUID) {
        pairs.removeAll { $0.id == id }
        saveToDisk()
    }

    // MARK: - Export

    /// One JSON object per line, oldest first — the standard fine-tune corpus
    /// shape (#294 consumes this).
    func exportJSONL() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var out = Data()
        for pair in pairs.reversed() {
            out.append(try encoder.encode(pair))
            out.append(0x0A)
        }
        return out
    }

    // MARK: - Persistence

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: storageURL.path) else { return }
        do {
            let data = try Data(contentsOf: storageURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            pairs = try decoder.decode([CorrectionPair].self, from: data)
        } catch {
            Log.transcription.error("Failed to load correction pairs: \(error)")
        }
    }

    private func saveToDisk() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(pairs)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            Log.transcription.error("Failed to save correction pairs: \(error)")
        }
    }
}
