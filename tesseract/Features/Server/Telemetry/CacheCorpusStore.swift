//
//  CacheCorpusStore.swift
//  tesseract
//
//  Reads the durable per-completion trace corpus (`CompletionTraceLog`,
//  30 day-files of JSONL) into chart-ready points — the zero-new-plumbing
//  data source the Cache Overview charts on (map #269).
//

import Foundation
import Observation

@MainActor
@Observable
final class CacheCorpusStore {

    /// One cache-aware completion, reduced to what the Overview charts on.
    struct Point: Identifiable, Sendable, Equatable {
        let id: UUID
        let timestamp: Date
        let lookupSeconds: Double
        /// Restore + SSD hydration, one visual stage.
        let restoreSeconds: Double
        let prefillSeconds: Double
        let residualPromptSeconds: Double
        let ttftSeconds: Double
        let hitTokens: Int
        let promptTokens: Int
        let isHit: Bool
    }

    private(set) var points: [Point] = []
    private(set) var isLoading = false
    private(set) var lastLoadedAt: Date?

    /// Load (or re-load) the whole retained corpus off the main actor.
    /// ~tens of MB worst case, read once per page visit or manual reload.
    func reload() {
        guard !isLoading else { return }
        isLoading = true
        Task { @MainActor in
            let loaded = await Task.detached(priority: .userInitiated) {
                Self.readCorpus()
            }.value
            points = loaded
            lastLoadedAt = Date()
            isLoading = false
        }
    }

    func points(in window: CacheWindow, reference: Date = Date()) -> [Point] {
        let cutoff = reference.addingTimeInterval(-window.duration)
        return points.filter { $0.timestamp >= cutoff }
    }

    private nonisolated static func readCorpus() -> [Point] {
        let files = CompletionTraceLog.traceFiles(in: CompletionTraceLog.defaultDirectory)
        let records = CompletionTraceLog.readRecords(at: files)
        return records.map { record in
            Point(
                id: record.requestID,
                timestamp: Date(timeIntervalSinceReferenceDate: record.timestamp),
                lookupSeconds: record.lookupSeconds,
                restoreSeconds: record.restoreSeconds + record.hydrationSeconds,
                prefillSeconds: record.prefillSeconds,
                residualPromptSeconds: record.residualPromptSeconds,
                ttftSeconds: record.ttftSeconds,
                hitTokens: record.hitTokens,
                promptTokens: record.promptTokenCount,
                isHit: record.restoredOffset > 0 || record.hitTokens > 0
            )
        }
        .sorted { $0.timestamp < $1.timestamp }
    }
}

// MARK: - Window

/// The Overview's one time-window picker: 24 h · 7 d · 30 d (#272).
enum CacheWindow: String, CaseIterable, Identifiable {
    case day = "24h"
    case week = "7d"
    case month = "30d"

    var id: String { rawValue }

    var duration: TimeInterval {
        switch self {
        case .day: 24 * 3600
        case .week: 7 * 24 * 3600
        case .month: 30 * 24 * 3600
        }
    }

    /// Trend/SSD bucket granularity: hours inside a day, days beyond.
    var bucketsByHour: Bool { self == .day }
}
