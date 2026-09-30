//
//  BudgetDrainDiagnosticsTests.swift
//  tesseractTests
//
//  A drain the prefix cache starts on its own (memory pressure, a budget
//  re-measure, a budget override) logs every body it drops as an `eviction`
//  event, as an admission's drain does, so the diagnostics log and the Prompt
//  Cache panel's events agree with its counters (#579). An admission's drain
//  stays the admission's to log, against its request.
//

import Foundation
import Testing

@testable import Tesseract_Agent

/// The system-scope event names a sink saw. The manager drains on the
/// MainActor and each test drains synchronously there, so no other test's
/// system events land in between.
nonisolated private final class SystemEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var names: [String] = []

    func record(_ event: PromptCacheTelemetryEvent) {
        guard event.scope == .system else { return }
        lock.withLock { names.append(event.eventName) }
    }

    func count(_ name: String) -> Int {
        lock.withLock { names.filter { $0 == name }.count }
    }
}

@MainActor
struct BudgetDrainDiagnosticsTests {
    private let key = CachePartitionKey(
        modelID: "budget-drain-diagnostics", kvBits: nil, kvGroupSize: 64)
    private let leafBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
        offset: 10, type: .leaf
    ).memoryBytes

    private func systemEvents(during body: () -> Void) -> SystemEvents {
        let events = SystemEvents()
        let handle = PrefixCacheDiagnostics.addTelemetrySink { events.record($0) }
        defer { PrefixCacheDiagnostics.removeTelemetrySink(handle) }
        body()
        return events
    }

    /// Ten-token leaves on separate paths. The last one admitted is the
    /// freshest leaf, which the Budget Floor keeps through any drain.
    private func admitLeaves(_ manager: PrefixCacheManager, starts: [Int] = [1, 20, 40]) {
        for start in starts {
            PrefixCacheTestFixtures.admitUniformLeaf(
                manager, tokens: Array(start...(start + 9)), partitionKey: key)
        }
    }

    @Test func aPressureDrainLogsEveryEviction() {
        let pressure = InMemoryMemoryPressureSource()
        let manager = PrefixCacheManager(
            memoryBudgetBytes: leafBytes * 100, pressureSource: pressure)
        admitLeaves(manager)
        manager.cumulativeCountersResetForTesting()

        let events = systemEvents { pressure.send(.critical) }

        #expect(manager.stats.snapshotCount == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 2)
        #expect(events.count("eviction") == 2)
    }

    /// A drop that keeps the body on SSD also logs its `ssdBodyDrop`, as an
    /// admission's recoverable drop does.
    @Test func aRecoverablePressureDropLogsItsBodyDrop() throws {
        let pressure = InMemoryMemoryPressureSource()
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: leafBytes * 4, tieredStore: store, pressureSource: pressure)
        admitLeaves(manager, starts: [1, 20, 40, 60])
        let tree = try #require(store.tree(for: key))
        for start in [1, 20] {
            let (node, _) = try #require(
                tree.findBestSnapshot(tokens: Array(start...(start + 9)), updateAccess: false))
            let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 10)
            tree.admit(node: node, ref: ref)
            tree.commitRef(node: node, expectedID: ref.snapshotID)
        }
        manager.cumulativeCountersResetForTesting()

        let events = systemEvents { pressure.send(.warning) }

        #expect(manager.cumulativeCounters.recoveredEvictions == 2)
        #expect(events.count("eviction") == 2)
        #expect(events.count("ssdBodyDrop") == 2)
    }

    @Test func aBudgetMeasurementLogsEveryEviction() {
        let headroom = InMemoryMemoryHeadroomSource()  // no sample until set
        let manager = PrefixCacheManager(
            memoryBudgetBytes: leafBytes * 100, headroomSource: headroom)
        admitLeaves(manager)
        headroom.next = MemoryHeadroomSample(freeBytes: 0, purgeableBytes: 0)

        var evicted: [PrefixCacheManager.EvictionEvent] = []
        let events = systemEvents { evicted = manager.reevaluateBudgetCeiling() }

        #expect(evicted.count == 2)
        #expect(events.count("eviction") == 2)
    }

    @Test func aBudgetOverrideLogsEveryEviction() {
        let manager = PrefixCacheManager(memoryBudgetBytes: leafBytes * 100)
        admitLeaves(manager)

        var evicted: [PrefixCacheManager.EvictionEvent] = []
        let events = systemEvents { evicted = manager.setMemoryBudget(0) }

        #expect(evicted.count == 2)
        #expect(events.count("eviction") == 2)
    }

    /// The admission logs its own drain's evictions against its request, so
    /// the drain inside `admit` logs nothing itself.
    @Test func anAdmissionsDrainIsNotLoggedTwice() {
        let manager = PrefixCacheManager(memoryBudgetBytes: leafBytes * 2)

        let events = systemEvents { admitLeaves(manager) }

        #expect(manager.stats.snapshotCount == 2)
        #expect(events.count("eviction") == 0)
    }
}
