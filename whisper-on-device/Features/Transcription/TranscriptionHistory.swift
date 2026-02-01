//
//  TranscriptionHistory.swift
//  whisper-on-device
//

import Foundation
import Combine
import SwiftUI

@MainActor
final class TranscriptionHistory: ObservableObject {
    @Published private(set) var entries: [TranscriptionEntry] = []

    private let maxEntries: Int
    private let storageURL: URL

    init(maxEntries: Int = 100) {
        self.maxEntries = maxEntries

        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = appSupport.appendingPathComponent("WhisperOnDevice", isDirectory: true)

        // Create directory if needed
        try? FileManager.default.createDirectory(at: appDirectory, withIntermediateDirectories: true)

        self.storageURL = appDirectory.appendingPathComponent("transcription_history.json")

        loadFromDisk()
    }

    func add(_ entry: TranscriptionEntry) {
        entries.insert(entry, at: 0)

        // Prune old entries
        if entries.count > maxEntries {
            entries = Array(entries.prefix(maxEntries))
        }

        saveToDisk()
    }

    func add(text: String, duration: TimeInterval, model: String) {
        let entry = TranscriptionEntry(
            text: text,
            duration: duration,
            model: model
        )
        add(entry)
    }

    func delete(_ entry: TranscriptionEntry) {
        entries.removeAll { $0.id == entry.id }
        saveToDisk()
    }

    func delete(at offsets: IndexSet) {
        entries.remove(atOffsets: offsets)
        saveToDisk()
    }

    func clear() {
        entries.removeAll()
        saveToDisk()
    }

    func entries(from startDate: Date, to endDate: Date) -> [TranscriptionEntry] {
        entries.filter { entry in
            entry.timestamp >= startDate && entry.timestamp <= endDate
        }
    }

    // MARK: - Persistence

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: storageURL.path) else { return }

        do {
            let data = try Data(contentsOf: storageURL)
            entries = try JSONDecoder().decode([TranscriptionEntry].self, from: data)
        } catch {
            print("Failed to load transcription history: \(error)")
        }
    }

    private func saveToDisk() {
        do {
            let data = try JSONEncoder().encode(entries)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            print("Failed to save transcription history: \(error)")
        }
    }
}
