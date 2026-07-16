//
//  TrackingRecords.swift
//  tesseract
//
//  The tracking grain's value types (#308): episodes are testimony,
//  observations are measurement, beliefs are conclusion — these are the
//  measurement side. Rows live in memory's SQLite database (one DB on
//  purpose); methods in `MemoryStore+Tracking.swift`.
//

import Foundation

// MARK: - Observations

nonisolated enum TrackingDomain: String, Codable, Sendable, CaseIterable {
    case work
    case body
    case mind
}

nonisolated enum ObservationSource: String, Codable, Sendable {
    /// Sensor code wrote it — no LLM between the sensor and the disk, ever.
    case sensed
    /// A conversation produced it, through a typed tool; `episodeRef` holds
    /// the utterance so the verbatim words stay recoverable behind the fact.
    case elicited
    /// A bridge from an external store (none ship in v1).
    case imported
}

/// One dated, typed fact. Append-only, kept forever: facts don't decay; a
/// correction is a newer row (recency wins at read), never an edit.
nonisolated struct TrackingObservation: Sendable, Identifiable {
    let id: UUID
    let ts: Date
    let domain: TrackingDomain
    let kind: String
    let value: String
    let source: ObservationSource
    let stream: String?
    let episodeRef: UUID?

    init(
        id: UUID = UUID(),
        ts: Date = Date(),
        domain: TrackingDomain,
        kind: String,
        value: String,
        source: ObservationSource,
        stream: String? = nil,
        episodeRef: UUID? = nil
    ) {
        self.id = id
        self.ts = ts
        self.domain = domain
        self.kind = kind
        self.value = value
        self.source = source
        self.stream = stream
        self.episodeRef = episodeRef
    }
}

// MARK: - The contract chain

nonisolated enum ContractStepStatus: String, Codable, Sendable {
    case pending
    case active
    case done
    case blocked
    /// The owner consciously moved off-contract (the anchor's hyperfocus day):
    /// terminal for today, never a failure.
    case switched
    case dropped
}

/// One hard step in the day's contract chain — ordered, at most one `active`;
/// finishing a step arms the next immediately, the same day. Step one is the
/// keystone: the day's win condition. Steps past it are surplus (chain depth),
/// never failure.
nonisolated struct ContractStep: Codable, Sendable, Equatable {
    var title: String
    var status: ContractStepStatus
    var workItemID: UUID?
    var startedAt: Date?
    var closedAt: Date?
    var note: String?

    init(
        title: String,
        status: ContractStepStatus = .pending,
        workItemID: UUID? = nil,
        startedAt: Date? = nil,
        closedAt: Date? = nil,
        note: String? = nil
    ) {
        self.title = title
        self.status = status
        self.workItemID = workItemID
        self.startedAt = startedAt
        self.closedAt = closedAt
        self.note = note
    }
}

/// One calendar day: the contract chain, the support items, tomorrow's seed,
/// and the close-out stamp (`closedAt == nil` is the "we didn't close
/// yesterday" flag the next morning reads).
nonisolated struct DayRecord: Sendable {
    /// Local calendar day, `yyyy-MM-dd`.
    let date: String
    var seed: String?
    var chain: [ContractStep]
    var support: [String]
    var closedAt: Date?
    let createdAt: Date
    var updatedAt: Date

    init(
        date: String,
        seed: String? = nil,
        chain: [ContractStep] = [],
        support: [String] = [],
        closedAt: Date? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.date = date
        self.seed = seed
        self.chain = chain
        self.support = support
        self.closedAt = closedAt
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    var activeStep: ContractStep? { chain.first { $0.status == .active } }
    var keystone: ContractStep? { chain.first }

    /// How far past the keystone the day went — "kept, depth 2/3".
    var chainDepth: Int { chain.filter { $0.status == .done }.count }

    /// Summary line the tools answer with — the agent should always see the
    /// state it just changed.
    var chainSummary: String {
        guard !chain.isEmpty else { return "No contract for \(date)." }
        let steps = chain.enumerated().map { index, step in
            let marker = index == 0 ? "keystone" : "step \(index + 1)"
            return "\(index + 1). [\(step.status.rawValue)] \(step.title) (\(marker))"
        }
        var lines = ["Contract for \(date):"] + steps
        if !support.isEmpty {
            lines.append("Support: \(support.joined(separator: "; "))")
        }
        if let seed { lines.append("Seed: \(seed)") }
        lines.append(closedAt == nil ? "Day not closed yet." : "Day closed.")
        return lines.joined(separator: "\n")
    }
}

// MARK: - Work items

nonisolated enum WorkItemCadence: String, Codable, Sendable {
    case once
    /// A recurring habit — re-arms each day; check-offs land as observations
    /// in the habit's domain, and the item itself stays open.
    case daily
}

nonisolated enum WorkItemStatus: String, Codable, Sendable {
    case open
    case done
    case dropped
}

/// One backlog entry the Companion may draw a chain step from. The successor
/// to the retired `tasks.md`.
nonisolated struct WorkItemRecord: Sendable, Identifiable {
    let id: UUID
    var title: String
    var stream: String?
    var domain: TrackingDomain
    var cadence: WorkItemCadence
    var status: WorkItemStatus
    var due: Date?
    var episodeRef: UUID?
    let createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        title: String,
        stream: String? = nil,
        domain: TrackingDomain = .work,
        cadence: WorkItemCadence = .once,
        status: WorkItemStatus = .open,
        due: Date? = nil,
        episodeRef: UUID? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.stream = stream
        self.domain = domain
        self.cadence = cadence
        self.status = status
        self.due = due
        self.episodeRef = episodeRef
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

// MARK: - Day keys

nonisolated enum TrackingDay {

    /// The local-calendar day key — `yyyy-MM-dd`. TEXT keys so the store stays
    /// terminal-readable (`WHERE date = '2026-07-16'`), same spirit as the
    /// Unix-seconds convention.
    static func key(for date: Date = Date(), calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
    }

    static func yesterdayKey(from date: Date = Date(), calendar: Calendar = .current) -> String {
        key(for: calendar.date(byAdding: .day, value: -1, to: date) ?? date, calendar: calendar)
    }
}
