//
//  SSDEnduranceAccumulator.swift
//  tesseract
//
//  The **endurance ledger** (PRD #150, ADR-0019): persistent
//  bytes-written / bytes-deleted counters for the SSD prefix-cache
//  tier, accumulated from the same diagnostics events the JSONL file
//  sink records — one pipeline for chart and forensics, so the
//  counters reconcile with `ssdAdmit accepted` byte sums by
//  construction.
//
//  "Measure before throttle": no write-rate limiter ships until these
//  field counters justify one (measured arithmetic: ~100–150 GB/day of
//  heavy-agent suffix writes ≈ a decade of consumer-SSD endurance).
//  The counters exist so that claim stops being arithmetic and starts
//  being data.
//
//  Written bytes are keyed by write class (`guarantee` /
//  `writeThrough` / `deferred`); deleted bytes by reason (eviction,
//  supersession, invalidation, stale-GC, ...). Deleted is freed-bytes
//  accounting, not a mirror of written: a tombstone-vetoed write
//  deletes a file whose `ssdAdmit accepted` never fired, so deleted
//  can exceed written by design in rare races.
//
//  Time series: hourly buckets (~3 days) and daily buckets (~60 days)
//  feed the cache panel's write-pressure chart; the lifetime totals
//  give it the endurance context. State persists to
//  `CacheDiagnostics/ssd-endurance.json`, debounced; a crash loses at
//  most the debounce window.
//

import Foundation

// MARK: - Snapshot (UI-facing value)

nonisolated struct SSDEnduranceSnapshot: Codable, Equatable, Sendable {
    struct DatedBucket: Codable, Equatable, Sendable, Identifiable {
        /// Bucket start (hour or day boundary, UTC).
        let id: Date
        let bytesWritten: Int
        let bytesDeleted: Int
    }

    /// When the ledger started counting — the "lifetime" anchor.
    let since: Date
    let lifetimeBytesWrittenByClass: [String: Int]
    let lifetimeBytesDeletedByReason: [String: Int]
    /// Ascending by date; at most `SSDEnduranceAccumulator.retainedHours`.
    let hourly: [DatedBucket]
    /// Ascending by date; at most `SSDEnduranceAccumulator.retainedDays`.
    let daily: [DatedBucket]

    var lifetimeBytesWritten: Int {
        lifetimeBytesWrittenByClass.values.reduce(0, +)
    }

    var lifetimeBytesDeleted: Int {
        lifetimeBytesDeletedByReason.values.reduce(0, +)
    }

    static let empty = SSDEnduranceSnapshot(
        since: Date(timeIntervalSinceReferenceDate: 0),
        lifetimeBytesWrittenByClass: [:],
        lifetimeBytesDeletedByReason: [:],
        hourly: [],
        daily: []
    )
}

// MARK: - Accumulator

nonisolated final class SSDEnduranceAccumulator: @unchecked Sendable {

    /// Hourly buckets retained (~3 days) — the panel's "per hour" view.
    static let retainedHours = 72
    /// Daily buckets retained (~2 months) — the panel's "per day" view.
    static let retainedDays = 60
    /// Debounce for the JSON persist; a crash loses at most this much.
    static let persistDebounce: TimeInterval = 5

    static var defaultFileURL: URL {
        PromptCacheDiagnosticsFileSink.defaultDirectory
            .appendingPathComponent("ssd-endurance.json")
    }

    // MARK: - Persisted state

    private struct Bucket: Codable, Equatable {
        var bytesWritten: Int = 0
        var bytesDeleted: Int = 0
    }

    private struct State: Codable {
        var since: Date
        var writtenByClass: [String: Int] = [:]
        var deletedByReason: [String: Int] = [:]
        /// Keyed by the bucket's start expressed as whole hours / days
        /// since the reference date, stringified (JSON dictionaries with
        /// `Int` keys encode as flat arrays — string keys keep the file
        /// greppable).
        var hourly: [String: Bucket] = [:]
        var daily: [String: Bucket] = [:]
    }

    private let fileURL: URL
    private let lock = NSLock()
    private var state: State
    private var persistScheduled = false
    private let persistQueue = DispatchQueue(
        label: "app.tesseract.agent.ssd-endurance",
        qos: .utility
    )
    private var sinkHandle: PrefixCacheDiagnostics.TelemetrySinkHandle?

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    init(
        fileURL: URL = SSDEnduranceAccumulator.defaultFileURL,
        registerSink: Bool = true
    ) {
        self.fileURL = fileURL
        if let data = try? Data(contentsOf: fileURL),
            let restored = try? Self.decoder.decode(State.self, from: data)
        {
            self.state = restored
        } else {
            self.state = State(since: Date())
        }
        if registerSink {
            sinkHandle = PrefixCacheDiagnostics.addTelemetrySink { [weak self] event in
                self?.record(event)
            }
        }
    }

    deinit {
        if let sinkHandle {
            PrefixCacheDiagnostics.removeTelemetrySink(sinkHandle)
        }
    }

    // MARK: - Event intake

    /// Fold one diagnostics event into the counters. Called from
    /// arbitrary threads by the telemetry sink; also the direct entry
    /// point for tests.
    func record(_ event: PromptCacheTelemetryEvent) {
        switch event.eventName {
        case "ssdAdmit":
            guard event.field("outcome") == "accepted",
                let bytes = event.intField("bytes")
            else { return }
            let writeClass =
                event.field("writeClass")
                ?? (event.field("mandatory") == "true" ? "guarantee" : "writeThrough")
            add(written: bytes, class: writeClass, at: event.timestamp)

        case "ssdDelete":
            guard let bytes = event.intField("bytes"),
                let reason = event.field("reason")
            else { return }
            add(deleted: bytes, reason: reason, at: event.timestamp)

        case "ssdPartitionInvalidated":
            guard let bytes = event.intField("bytes"), bytes > 0 else { return }
            let reason =
                event.field("reason") == "staleUnused" ? "staleGC" : "invalidated"
            add(deleted: bytes, reason: reason, at: event.timestamp)

        default:
            break
        }
    }

    /// Current counters as a UI-facing snapshot: buckets sorted
    /// ascending and converted to dates.
    func snapshot() -> SSDEnduranceSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return SSDEnduranceSnapshot(
            since: state.since,
            lifetimeBytesWrittenByClass: state.writtenByClass,
            lifetimeBytesDeletedByReason: state.deletedByReason,
            hourly: datedBuckets(state.hourly, secondsPerUnit: 3600),
            daily: datedBuckets(state.daily, secondsPerUnit: 86_400)
        )
    }

    /// Barrier for tests and shutdown: write the current state out now.
    func persistNow() {
        lock.lock()
        let snapshot = state
        persistScheduled = false
        lock.unlock()
        writeState(snapshot)
    }

    // MARK: - Private

    private func add(
        written bytes: Int = 0,
        class writeClass: String = "",
        deleted deletedBytes: Int = 0,
        reason: String = "",
        at timestamp: Date
    ) {
        lock.lock()
        if bytes > 0 {
            state.writtenByClass[writeClass, default: 0] += bytes
        }
        if deletedBytes > 0 {
            state.deletedByReason[reason, default: 0] += deletedBytes
        }
        updateBucket(
            &state.hourly,
            key: bucketKey(timestamp, secondsPerUnit: 3600),
            written: bytes,
            deleted: deletedBytes,
            retained: Self.retainedHours
        )
        updateBucket(
            &state.daily,
            key: bucketKey(timestamp, secondsPerUnit: 86_400),
            written: bytes,
            deleted: deletedBytes,
            retained: Self.retainedDays
        )
        let shouldSchedule = !persistScheduled
        persistScheduled = true
        lock.unlock()

        if shouldSchedule {
            persistQueue.asyncAfter(deadline: .now() + Self.persistDebounce) { [weak self] in
                self?.persistIfScheduled()
            }
        }
    }

    private func add(deleted bytes: Int, reason: String, at timestamp: Date) {
        add(written: 0, class: "", deleted: bytes, reason: reason, at: timestamp)
    }

    private func add(written bytes: Int, class writeClass: String, at timestamp: Date) {
        add(written: bytes, class: writeClass, deleted: 0, reason: "", at: timestamp)
    }

    private func bucketKey(_ date: Date, secondsPerUnit: Double) -> String {
        String(Int((date.timeIntervalSinceReferenceDate / secondsPerUnit).rounded(.down)))
    }

    /// Fold bytes into the keyed bucket, then prune to the newest
    /// `retained` buckets by key order — pruning follows the data's own
    /// clock (event timestamps), so replays stay deterministic.
    private func updateBucket(
        _ buckets: inout [String: Bucket],
        key: String,
        written: Int,
        deleted: Int,
        retained: Int
    ) {
        var bucket = buckets[key] ?? Bucket()
        bucket.bytesWritten += written
        bucket.bytesDeleted += deleted
        buckets[key] = bucket

        if buckets.count > retained {
            let sortedKeys = buckets.keys.sorted { (Int($0) ?? 0) < (Int($1) ?? 0) }
            for stale in sortedKeys.prefix(buckets.count - retained) {
                buckets.removeValue(forKey: stale)
            }
        }
    }

    private func datedBuckets(
        _ buckets: [String: Bucket],
        secondsPerUnit: Double
    ) -> [SSDEnduranceSnapshot.DatedBucket] {
        buckets
            .compactMap { key, bucket -> SSDEnduranceSnapshot.DatedBucket? in
                guard let unit = Int(key) else { return nil }
                return SSDEnduranceSnapshot.DatedBucket(
                    id: Date(
                        timeIntervalSinceReferenceDate: Double(unit) * secondsPerUnit
                    ),
                    bytesWritten: bucket.bytesWritten,
                    bytesDeleted: bucket.bytesDeleted
                )
            }
            .sorted { $0.id < $1.id }
    }

    private func persistIfScheduled() {
        lock.lock()
        guard persistScheduled else {
            lock.unlock()
            return
        }
        persistScheduled = false
        let snapshot = state
        lock.unlock()
        writeState(snapshot)
    }

    private func writeState(_ snapshot: State) {
        do {
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try Self.encoder.encode(snapshot)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            Log.agent.error(
                "SSDEnduranceAccumulator persist failed: \(String(describing: error))"
            )
        }
    }
}
