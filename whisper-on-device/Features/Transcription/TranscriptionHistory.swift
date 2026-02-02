//
//  TranscriptionHistory.swift
//  whisper-on-device
//

import Foundation
import Combine
import SwiftUI

// MARK: - History Item for Flattened List

/// Represents either a section header or an entry in the flattened history list.
/// Using a flat structure enables true lazy loading in LazyVStack.
enum HistoryItem: Identifiable, Equatable {
    case header(String, Date)
    case entry(TranscriptionEntry, isFirst: Bool, isLast: Bool)

    var id: String {
        switch self {
        case .header(let label, _):
            return "header-\(label)"
        case .entry(let entry, _, _):
            return "entry-\(entry.id.uuidString)"
        }
    }

    static func == (lhs: HistoryItem, rhs: HistoryItem) -> Bool {
        switch (lhs, rhs) {
        case (.header(let l1, let d1), .header(let l2, let d2)):
            return l1 == l2 && d1 == d2
        case (.entry(let e1, let f1, let last1), .entry(let e2, let f2, let last2)):
            return e1.id == e2.id && f1 == f2 && last1 == last2
        default:
            return false
        }
    }
}

// MARK: - Cached DateFormatters

/// Static DateFormatters to avoid repeated allocation (expensive operation)
private enum DateFormatters {
    static let time: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h:mm a"
        return f
    }()

    static let date: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMM d"
        return f
    }()

    static func formatTime(_ date: Date) -> String {
        time.string(from: date)
    }

    static func formatDate(_ dateValue: Date) -> String {
        self.date.string(from: dateValue).uppercased()
    }

    static func dateGroupLabel(for date: Date, calendar: Calendar) -> String {
        if calendar.isDateInToday(date) {
            return "TODAY"
        } else if calendar.isDateInYesterday(date) {
            return "YESTERDAY"
        } else {
            return self.date.string(from: date).uppercased()
        }
    }
}

@MainActor
final class TranscriptionHistory: ObservableObject {
    @Published private(set) var entries: [TranscriptionEntry] = []

    /// Flattened list of items for efficient lazy rendering.
    /// Updated only when entries change.
    @Published private(set) var flattenedItems: [HistoryItem] = []

    private let maxEntries: Int
    private let storageURL: URL

    init(maxEntries: Int = 100) {
        self.maxEntries = maxEntries

        guard let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            // Use temp directory as fallback
            self.storageURL = FileManager.default.temporaryDirectory.appendingPathComponent("transcription_history.json")
            loadFromDisk()
            return
        }
        let appDirectory = appSupport.appendingPathComponent("WhisperOnDevice", isDirectory: true)

        // Create directory if needed
        try? FileManager.default.createDirectory(at: appDirectory, withIntermediateDirectories: true)

        self.storageURL = appDirectory.appendingPathComponent("transcription_history.json")

        loadFromDisk()
        updateFlattenedItems()
    }

    func add(_ entry: TranscriptionEntry) {
        entries.insert(entry, at: 0)

        // Prune old entries
        if entries.count > maxEntries {
            entries = Array(entries.prefix(maxEntries))
        }

        saveToDisk()
        updateFlattenedItems()
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
        updateFlattenedItems()
    }

    func delete(at offsets: IndexSet) {
        entries.remove(atOffsets: offsets)
        saveToDisk()
        updateFlattenedItems()
    }

    func clear() {
        entries.removeAll()
        saveToDisk()
        updateFlattenedItems()
    }

    func entries(from startDate: Date, to endDate: Date) -> [TranscriptionEntry] {
        entries.filter { entry in
            entry.timestamp >= startDate && entry.timestamp <= endDate
        }
    }

    // MARK: - Flattened Items for Lazy Rendering

    /// Rebuilds the flattened item list from entries.
    /// Called only when entries change, not on every view render.
    private func updateFlattenedItems() {
        guard !entries.isEmpty else {
            flattenedItems = []
            return
        }

        let calendar = Calendar.current
        var items: [HistoryItem] = []

        // Group entries by date label
        let grouped = Dictionary(grouping: entries) { entry in
            DateFormatters.dateGroupLabel(for: entry.timestamp, calendar: calendar)
        }

        // Sort groups by most recent first
        let sortedKeys = grouped.keys.sorted { key1, key2 in
            let date1 = grouped[key1]?.first?.timestamp ?? Date.distantPast
            let date2 = grouped[key2]?.first?.timestamp ?? Date.distantPast
            return date1 > date2
        }

        for key in sortedKeys {
            guard let groupEntries = grouped[key]?.sorted(by: { $0.timestamp > $1.timestamp }) else { continue }
            guard let firstEntry = groupEntries.first else { continue }

            // Add header
            items.append(.header(key, firstEntry.timestamp))

            // Add entries with position flags
            for (index, entry) in groupEntries.enumerated() {
                items.append(.entry(
                    entry,
                    isFirst: index == 0,
                    isLast: index == groupEntries.count - 1
                ))
            }
        }

        flattenedItems = items
    }

    /// Returns formatted time string for an entry using cached formatter.
    static func formattedTime(for date: Date) -> String {
        DateFormatters.formatTime(date)
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
