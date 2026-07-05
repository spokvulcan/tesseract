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

    /// One user-worthy SSD-tier event for the cache panel's sparing
    /// "notable events" line — a partition invalidation ("Cache for X
    /// was reset — model files changed") or a stale-GC reclaim. Kept
    /// here (the eager, persistent ledger) rather than in the window's
    /// event buffer because these fire at model load, typically before
    /// any telemetry window exists, and the 2026-07-04 incident was
    /// exactly such a silent reset.
    struct NotableEvent: Codable, Equatable, Sendable, Identifiable {
        let at: Date
        /// `fingerprintChanged` / `staleUnused` / `schemaStale` — the
        /// `ssdPartitionInvalidated` reason vocabulary — or
        /// `clientPrefixChange` (issue #158): a deep cache loss caused by
        /// the client mutating its prompt prefix, not by this tier.
        let kind: String
        let modelID: String
        let bytes: Int
        /// `clientPrefixChange` payload: where the prompt diverged and how
        /// many cached tokens it abandoned. Numbers, not copy — the view
        /// composes (and can reword) the sentence. Optional so pre-existing
        /// persisted ledgers and other kinds decode.
        var divergenceOffset: Int?
        var abandonedTokens: Int?

        var id: String { "\(at.timeIntervalSinceReferenceDate)-\(kind)-\(modelID)" }
    }

    /// When the ledger started counting — the "lifetime" anchor.
    let since: Date
    let lifetimeBytesWrittenByClass: [String: Int]
    let lifetimeBytesDeletedByReason: [String: Int]
    /// Ascending by date; at most `SSDEnduranceAccumulator.retainedHours`.
    let hourly: [DatedBucket]
    /// Ascending by date; at most `SSDEnduranceAccumulator.retainedDays`.
    let daily: [DatedBucket]
    /// Ascending by date; at most `SSDEnduranceAccumulator.retainedNotables`.
    let notable: [NotableEvent]

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
        daily: [],
        notable: []
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
    /// Notable events retained for the panel — sparing by design.
    static let retainedNotables = 20

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
        /// Optional so a pre-notables state file still decodes.
        var notable: [SSDEnduranceSnapshot.NotableEvent]?
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
            recordNotable(event)
            guard let bytes = event.intField("bytes"), bytes > 0 else { return }
            let reason =
                event.field("reason") == "staleUnused" ? "staleGC" : "invalidated"
            add(deleted: bytes, reason: reason, at: event.timestamp)

        case "lookup":
            recordClientPrefixChangeNotable(event)

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
            daily: datedBuckets(state.daily, secondsPerUnit: 86_400),
            notable: state.notable ?? []
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
        scheduleDebouncedPersist(shouldSchedule)
    }

    /// Debounced-persist kickoff shared by every mutation site. The
    /// caller claims the schedule flag under its own lock (so flag and
    /// state mutate atomically) and passes the claim result here.
    private func scheduleDebouncedPersist(_ claimed: Bool) {
        guard claimed else { return }
        persistQueue.asyncAfter(deadline: .now() + Self.persistDebounce) { [weak self] in
            self?.persistIfScheduled()
        }
    }

    private func add(deleted bytes: Int, reason: String, at timestamp: Date) {
        add(written: 0, class: "", deleted: bytes, reason: reason, at: timestamp)
    }

    /// Buffer an invalidation as a panel-worthy notable event. Persists
    /// with the counters: invalidations fire at model load, typically
    /// before any telemetry window exists to catch them live.
    private func recordNotable(_ event: PromptCacheTelemetryEvent) {
        guard let modelID = event.field("modelID"),
            let reason = event.field("reason")
        else { return }
        appendNotable(
            SSDEnduranceSnapshot.NotableEvent(
                at: event.timestamp,
                kind: reason,
                modelID: modelID,
                bytes: event.intField("bytes") ?? 0
            ))
    }

    /// Buffer a client-prefix-change divergence (issue #158) as a
    /// panel-worthy notable event. Not an SSD-tier event, but the panel's
    /// notable line is the one place a deep, *not-our-fault* cache loss
    /// can be explained instead of reading as a tier regression — the
    /// 2026-07-05 incident (see `docs/prompt-cache-client-divergence.md`).
    /// Only the deep `clientPrefixChange` classification lands here;
    /// routine tail rewinds fire on every tool turn and would drown the
    /// sparing notable line.
    private func recordClientPrefixChangeNotable(_ event: PromptCacheTelemetryEvent) {
        guard
            event.field("divergence")
                == PrefixDivergenceProbe.Classification.clientPrefixChange.rawValue,
            let offset = event.intField("divergenceOffset"),
            let abandoned = event.intField("abandonedCachedTokens")
        else { return }
        appendNotable(
            SSDEnduranceSnapshot.NotableEvent(
                at: event.timestamp,
                kind: "clientPrefixChange",
                modelID: event.modelID ?? "unknown",
                bytes: 0,
                divergenceOffset: offset,
                abandonedTokens: abandoned
            ),
            coalescingLast: true
        )
    }

    /// `coalescingLast`: replace the buffer's last entry instead of appending
    /// when it has the same kind and model — a session that keeps mutating
    /// its prompt prefix (the chatty case by construction) occupies one slot
    /// instead of evicting the rare invalidation notices out of the
    /// `retainedNotables` cap. Keeps the feed "sparing by design".
    private func appendNotable(
        _ notable: SSDEnduranceSnapshot.NotableEvent,
        coalescingLast: Bool = false
    ) {
        lock.lock()
        var buffer = state.notable ?? []
        if coalescingLast, let last = buffer.last,
            last.kind == notable.kind, last.modelID == notable.modelID
        {
            buffer[buffer.count - 1] = notable
        } else {
            buffer.append(notable)
        }
        if buffer.count > Self.retainedNotables {
            buffer.removeFirst(buffer.count - Self.retainedNotables)
        }
        state.notable = buffer
        let shouldSchedule = !persistScheduled
        persistScheduled = true
        lock.unlock()
        scheduleDebouncedPersist(shouldSchedule)
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
