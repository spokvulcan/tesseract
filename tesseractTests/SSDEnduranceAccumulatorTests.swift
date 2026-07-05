//
//  SSDEnduranceAccumulatorTests.swift
//  tesseractTests
//
//  The endurance ledger (PRD #150): bytes-written / bytes-deleted
//  accumulation from diagnostics events, hourly/daily bucketing with
//  retention, and restart survival via the JSON persist. Events are
//  synthesized directly — the accumulator's contract is "same pipeline
//  as the JSONL sink", so feeding it `PromptCacheTelemetryEvent`s IS
//  the production shape.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SSDEnduranceAccumulatorTests {

    // MARK: - Fixtures

    private func makeScratchFile() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("endurance-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("ssd-endurance.json")
    }

    private func cleanup(_ fileURL: URL) {
        try? FileManager.default.removeItem(at: fileURL.deletingLastPathComponent())
    }

    private func makeAccumulator(fileURL: URL) -> SSDEnduranceAccumulator {
        SSDEnduranceAccumulator(fileURL: fileURL, registerSink: false)
    }

    private func makeEvent(
        name: String,
        timestamp: Date = Date(),
        fields: [(String, String)]
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: timestamp,
            scope: .system,
            eventName: name,
            fields: fields
        )
    }

    private func acceptedAdmit(
        bytes: Int,
        mandatory: Bool = false,
        writeClass: String? = nil,
        timestamp: Date = Date()
    ) -> PromptCacheTelemetryEvent {
        var fields: [(String, String)] = [
            ("id", UUID().uuidString),
            ("bytes", "\(bytes)"),
            ("outcome", "accepted"),
        ]
        if mandatory { fields.append(("mandatory", "true")) }
        if let writeClass { fields.append(("writeClass", writeClass)) }
        return makeEvent(name: "ssdAdmit", timestamp: timestamp, fields: fields)
    }

    // MARK: - Write accounting

    /// Accepted admissions accumulate per class: `mandatory=true` is the
    /// guarantee class, everything else defaults to write-through. The
    /// lifetime total matches the `ssdAdmit accepted` byte sum by
    /// construction — the acceptance criterion.
    @Test func acceptedAdmitsAccumulateByClass() {
        let file = makeScratchFile()
        defer { cleanup(file) }
        let accumulator = makeAccumulator(fileURL: file)

        accumulator.record(acceptedAdmit(bytes: 1000, mandatory: true))
        accumulator.record(acceptedAdmit(bytes: 300))
        accumulator.record(acceptedAdmit(bytes: 200))
        accumulator.record(acceptedAdmit(bytes: 50, writeClass: "deferred"))

        let snapshot = accumulator.snapshot()
        #expect(snapshot.lifetimeBytesWrittenByClass["guarantee"] == 1000)
        #expect(snapshot.lifetimeBytesWrittenByClass["writeThrough"] == 500)
        #expect(snapshot.lifetimeBytesWrittenByClass["deferred"] == 50)
        #expect(snapshot.lifetimeBytesWritten == 1550)
    }

    /// Rejected admissions never count — only `outcome=accepted` moves
    /// the written counters.
    @Test func rejectedAdmitsDoNotCount() {
        let file = makeScratchFile()
        defer { cleanup(file) }
        let accumulator = makeAccumulator(fileURL: file)

        accumulator.record(
            makeEvent(
                name: "ssdAdmit",
                fields: [
                    ("id", "x"), ("bytes", "9999"), ("outcome", "droppedExceedsBudget"),
                ]
            ))

        #expect(accumulator.snapshot().lifetimeBytesWritten == 0)
    }

    // MARK: - Delete accounting

    @Test func deletesAccumulateByReason() {
        let file = makeScratchFile()
        defer { cleanup(file) }
        let accumulator = makeAccumulator(fileURL: file)

        accumulator.record(
            makeEvent(
                name: "ssdDelete",
                fields: [("id", "a"), ("bytes", "700"), ("reason", "evicted")]
            ))
        accumulator.record(
            makeEvent(
                name: "ssdDelete",
                fields: [("id", "b"), ("bytes", "300"), ("reason", "superseded")]
            ))
        accumulator.record(
            makeEvent(
                name: "ssdPartitionInvalidated",
                fields: [
                    ("digest", "d"), ("modelID", "m"), ("bytes", "5000"),
                    ("reason", "staleUnused"),
                ]
            ))
        accumulator.record(
            makeEvent(
                name: "ssdPartitionInvalidated",
                fields: [
                    ("digest", "e"), ("modelID", "m"), ("bytes", "2000"),
                    ("reason", "fingerprintChanged"),
                ]
            ))

        let snapshot = accumulator.snapshot()
        #expect(snapshot.lifetimeBytesDeletedByReason["evicted"] == 700)
        #expect(snapshot.lifetimeBytesDeletedByReason["superseded"] == 300)
        #expect(snapshot.lifetimeBytesDeletedByReason["staleGC"] == 5000)
        #expect(snapshot.lifetimeBytesDeletedByReason["invalidated"] == 2000)
        #expect(snapshot.lifetimeBytesDeleted == 8000)
    }

    // MARK: - Bucketing

    /// Events land in hour and day buckets keyed by their own
    /// timestamps; the snapshot returns them ascending.
    @Test func eventsBucketByHourAndDay() {
        let file = makeScratchFile()
        defer { cleanup(file) }
        let accumulator = makeAccumulator(fileURL: file)

        let anchor = Date(timeIntervalSinceReferenceDate: 800_000_000)
        accumulator.record(acceptedAdmit(bytes: 100, timestamp: anchor))
        accumulator.record(acceptedAdmit(bytes: 200, timestamp: anchor.addingTimeInterval(60)))
        accumulator.record(acceptedAdmit(bytes: 400, timestamp: anchor.addingTimeInterval(3700)))

        let snapshot = accumulator.snapshot()
        #expect(snapshot.hourly.count == 2)
        #expect(snapshot.hourly.first?.bytesWritten == 300)
        #expect(snapshot.hourly.last?.bytesWritten == 400)
        #expect(snapshot.daily.count == 1)
        #expect(snapshot.daily.first?.bytesWritten == 700)
    }

    /// Bucket retention prunes by the data's own clock: only the newest
    /// `retainedHours` hour buckets survive.
    @Test func hourlyBucketsPruneToRetention() {
        let file = makeScratchFile()
        defer { cleanup(file) }
        let accumulator = makeAccumulator(fileURL: file)

        let anchor = Date(timeIntervalSinceReferenceDate: 800_000_000)
        let hours = SSDEnduranceAccumulator.retainedHours + 10
        for hour in 0..<hours {
            accumulator.record(
                acceptedAdmit(
                    bytes: 10,
                    timestamp: anchor.addingTimeInterval(Double(hour) * 3600)
                ))
        }

        let snapshot = accumulator.snapshot()
        #expect(snapshot.hourly.count == SSDEnduranceAccumulator.retainedHours)
        // Lifetime totals are unaffected by bucket pruning.
        #expect(snapshot.lifetimeBytesWritten == hours * 10)
    }

    // MARK: - Restart survival

    /// Counters survive a restart: persist, construct a fresh
    /// accumulator over the same file, totals identical (the PRD
    /// acceptance criterion).
    @Test func countersSurviveRestart() {
        let file = makeScratchFile()
        defer { cleanup(file) }

        let first = makeAccumulator(fileURL: file)
        first.record(acceptedAdmit(bytes: 1234, mandatory: true))
        first.record(
            makeEvent(
                name: "ssdDelete",
                fields: [("id", "a"), ("bytes", "111"), ("reason", "evicted")]
            ))
        first.persistNow()

        let second = makeAccumulator(fileURL: file)
        let snapshot = second.snapshot()
        #expect(snapshot.lifetimeBytesWrittenByClass["guarantee"] == 1234)
        #expect(snapshot.lifetimeBytesDeletedByReason["evicted"] == 111)
        // ISO8601 persistence truncates sub-second precision; the
        // anchor only needs to survive to the second.
        #expect(abs(snapshot.since.timeIntervalSince(first.snapshot().since)) < 1.0)

        // And the reloaded ledger keeps counting on top.
        second.record(acceptedAdmit(bytes: 6))
        #expect(second.snapshot().lifetimeBytesWritten == 1240)
    }

    /// A corrupt state file starts fresh instead of crashing or
    /// poisoning the counters.
    @Test func corruptStateFileStartsFresh() throws {
        let file = makeScratchFile()
        defer { cleanup(file) }
        try Data("{broken".utf8).write(to: file)

        let accumulator = makeAccumulator(fileURL: file)
        #expect(accumulator.snapshot().lifetimeBytesWritten == 0)
    }
}
