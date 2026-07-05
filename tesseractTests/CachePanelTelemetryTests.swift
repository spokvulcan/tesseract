//
//  CachePanelTelemetryTests.swift
//  tesseractTests
//
//  Cache panel v1 plumbing (PRD #150): the SSD budget's free-disk /
//  floor context surfaced to the panel, the endurance ledger's notable
//  events ("Cache for X was reset — model files changed"), and the
//  telemetry store's per-request outcome pair + endurance polling.
//

import Foundation
import Testing

@testable import Tesseract_Agent

private let gib = 1024 * 1024 * 1024

// MARK: - Budget context (ledger → panel)

struct SSDBudgetPanelContextTests {

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("panel-budget-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeDescriptor(bytes: Int) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: UUID().uuidString,
            partitionDigest: "abcd1234",
            pathFromRoot: [1, 2, 3],
            tokenOffset: 3,
            checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
            bytes: bytes,
            createdAt: 0,
            lastAccessAt: 0,
            fileRelativePath: "x/y.safetensors",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    /// After the first measurement the panel context carries the
    /// measured budget, the floor, and the observed free-disk bytes.
    @Test func measuredContextExposesFreeDisk() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20),
            freeDiskBytesProvider: { _ in 400 * gib }
        )
        _ = ledger.admit(makeDescriptor(bytes: 1_000))

        let context = ledger.budgetContext()
        #expect(context.budgetBytes == 100 * gib)
        #expect(context.floorBytes == 1_000_000)
        #expect(context.freeDiskBytes == 400 * gib)
        #expect(!context.floorBound)
    }

    /// A nearly-full disk degrades the measured budget to the floor —
    /// and the context says so (the panel's "at floor, disk low").
    @Test func nearlyFullDiskDegradesToFloorVisibly() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20),
            freeDiskBytesProvider: { _ in 2_000_000 }  // 0.25 · 2 MB ≪ floor
        )
        _ = ledger.admit(makeDescriptor(bytes: 1_000))

        let context = ledger.budgetContext()
        #expect(context.budgetBytes == 1_000_000)
        #expect(context.freeDiskBytes == 2_000_000)
        #expect(context.floorBound)
    }

    /// No provider (tests, replay) → no free-disk claim, and a budget
    /// that merely equals the floor never shows the disk-low signal.
    @Test func staticBudgetNeverClaimsDiskLow() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20)
        )
        _ = ledger.admit(makeDescriptor(bytes: 1_000))

        let context = ledger.budgetContext()
        #expect(context.freeDiskBytes == nil)
        #expect(!context.floorBound)
    }

    /// A user cap below the floor drags the budget under it on a roomy
    /// disk — a settings choice, not a full disk. The panel signal must
    /// stay quiet (the code-review catch on the derived
    /// `budget <= floor` heuristic).
    @Test func capBelowFloorIsNotDiskLow() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20),
            budgetCapBytes: 500_000,  // below the floor
            freeDiskBytesProvider: { _ in 400 * gib }  // roomy disk
        )
        _ = ledger.admit(makeDescriptor(bytes: 1_000))

        let context = ledger.budgetContext()
        #expect(context.budgetBytes == 500_000)
        #expect(context.budgetBytes < context.floorBytes)
        #expect(!context.floorBound)
    }
}

// MARK: - Notable events (endurance ledger)

struct SSDEnduranceNotableEventTests {

    private func makeAccumulator(
        label: String
    ) -> (SSDEnduranceAccumulator, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("panel-notable-\(label)-\(UUID().uuidString).json")
        return (SSDEnduranceAccumulator(fileURL: url, registerSink: false), url)
    }

    private func invalidationEvent(
        modelID: String = "ornith-35b",
        bytes: Int = 5 * gib,
        reason: String = "fingerprintChanged",
        at date: Date = Date(timeIntervalSince1970: 1_750_000_000)
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: date,
            scope: .system,
            eventName: "ssdPartitionInvalidated",
            requestID: nil,
            modelID: nil,
            kvBits: nil,
            kvGroupSize: nil,
            fields: [
                ("digest", "deadbeef"),
                ("modelID", modelID),
                ("bytes", "\(bytes)"),
                ("reason", reason),
            ]
        )
    }

    @Test func invalidationBecomesANotableEvent() {
        let (accumulator, url) = makeAccumulator(label: "record")
        defer { try? FileManager.default.removeItem(at: url) }

        accumulator.record(invalidationEvent())

        let notable = accumulator.snapshot().notable
        #expect(notable.count == 1)
        #expect(notable.first?.kind == "fingerprintChanged")
        #expect(notable.first?.modelID == "ornith-35b")
        #expect(notable.first?.bytes == 5 * gib)
    }

    /// Invalidations fire at model load, typically before any telemetry
    /// window exists — so the notable buffer must survive a restart.
    @Test func notableEventsSurviveRestart() {
        let (accumulator, url) = makeAccumulator(label: "restart")
        defer { try? FileManager.default.removeItem(at: url) }

        accumulator.record(invalidationEvent(reason: "staleUnused"))
        accumulator.persistNow()

        let reloaded = SSDEnduranceAccumulator(fileURL: url, registerSink: false)
        let notable = reloaded.snapshot().notable
        #expect(notable.count == 1)
        #expect(notable.first?.kind == "staleUnused")
    }

    @Test func notableBufferStaysSparing() {
        let (accumulator, url) = makeAccumulator(label: "cap")
        defer { try? FileManager.default.removeItem(at: url) }

        for index in 0..<(SSDEnduranceAccumulator.retainedNotables + 5) {
            accumulator.record(
                invalidationEvent(
                    modelID: "model-\(index)",
                    at: Date(timeIntervalSince1970: Double(1_750_000_000 + index))
                ))
        }

        let notable = accumulator.snapshot().notable
        #expect(notable.count == SSDEnduranceAccumulator.retainedNotables)
        // Newest survive the cap.
        #expect(
            notable.last?.modelID
                == "model-\(SSDEnduranceAccumulator.retainedNotables + 4)")
    }
}

// MARK: - Store: outcome pair + endurance polling

@MainActor
struct CachePanelStoreTests {

    private func makeEvent(
        _ name: String,
        requestID: UUID?,
        fields: [(String, String)] = []
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: Date(timeIntervalSince1970: 100),
            scope: .request,
            eventName: name,
            requestID: requestID,
            modelID: "panel-test",
            kvBits: 8,
            kvGroupSize: 64,
            fields: fields
        )
    }

    @Test func outcomePairsLookupWithItsOwnTTFT() {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        let first = UUID()
        let second = UUID()
        store.recordForTesting([
            makeEvent("lookup", requestID: first, fields: [("reason", "hit")]),
            makeEvent("ttft", requestID: first, fields: [("ttftMs", "500.0")]),
            makeEvent("lookup", requestID: second, fields: [("reason", "missNoEntries")]),
            makeEvent("ttft", requestID: second, fields: [("ttftMs", "2400.0")]),
        ])

        let outcome = store.lastRequestOutcome
        #expect(outcome?.lookup.requestID == second)
        #expect(outcome?.ttft?.requestID == second)
        #expect(outcome?.ttft?.doubleField("ttftMs") == 2400.0)
    }

    @Test func outcomeToleratesMissingTTFT() {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        let request = UUID()
        store.recordForTesting([
            makeEvent("lookup", requestID: request, fields: [("reason", "hit")])
        ])

        let outcome = store.lastRequestOutcome
        #expect(outcome?.lookup.requestID == request)
        #expect(outcome?.ttft == nil)
    }

    @Test func outcomeNilBeforeAnyLookup() {
        let store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)
        #expect(store.lastRequestOutcome == nil)
    }

    @Test func enduranceRefreshesWithSamples() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("panel-store-endurance-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let accumulator = SSDEnduranceAccumulator(fileURL: url, registerSink: false)
        let store = PromptCacheTelemetryStore(
            registerDiagnosticsSink: false,
            enduranceAccumulator: accumulator
        )
        #expect(store.endurance.lifetimeBytesWritten == 0)

        accumulator.record(
            PromptCacheTelemetryEvent(
                timestamp: Date(),
                scope: .system,
                eventName: "ssdAdmit",
                requestID: nil,
                modelID: nil,
                kvBits: nil,
                kvGroupSize: nil,
                fields: [
                    ("id", "s1"), ("bytes", "4096"),
                    ("outcome", "accepted"), ("writeClass", "guarantee"),
                ]
            ))
        // Any sample tick re-pulls the ledger.
        store.recordForTesting([makeEvent("capture", requestID: UUID())])

        #expect(store.endurance.lifetimeBytesWritten == 4096)
        #expect(store.endurance.lifetimeBytesWrittenByClass["guarantee"] == 4096)
    }
}
