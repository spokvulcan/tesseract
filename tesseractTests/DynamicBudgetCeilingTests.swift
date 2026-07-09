//
//  DynamicBudgetCeilingTests.swift
//  tesseractTests
//
//  PRD #149 (ADR-0018): dynamic budget ceilings. The pure pieces —
//  ceiling policy, Active-Inference Reserve, band rebase, SSD budget
//  policy, hydration gate — are tested as value-to-value functions; the
//  manager-level measurement flow is driven end-to-end through the
//  in-memory headroom peer, the same way the pressure tests drive the
//  pressure peer.
//

import Foundation
import Testing

@testable import Tesseract_Agent

private let gib = 1024 * 1024 * 1024

// MARK: - Ceiling policy (pure)

struct DynamicCeilingPolicyTests {

    @Test func ceilingIsResidentPlusClaimableHeadroomMinusReserve() {
        // 10 GiB measured headroom, 1 GiB resident, 4 GiB reserve:
        // 1 + 0.8·10 − 4 = 5 GiB. Resident bytes count as claimable —
        // they are our footprint, not missing headroom.
        let ceiling = DynamicCeilingPolicy.ceilingBytes(
            residentBytes: 1 * gib,
            measuredHeadroomBytes: 10 * gib,
            reserveBytes: 4 * gib,
            capBytes: nil
        )
        #expect(ceiling == 5 * gib)
    }

    @Test func reserveLargerThanHeadroomFloorsAtZero() {
        let ceiling = DynamicCeilingPolicy.ceilingBytes(
            residentBytes: 0,
            measuredHeadroomBytes: 2 * gib,
            reserveBytes: 4 * gib,
            capBytes: nil
        )
        #expect(ceiling == 0)
    }

    @Test func userCapOnlyLowersNeverRaises() {
        // Below the measured value: honored.
        #expect(
            DynamicCeilingPolicy.ceilingBytes(
                residentBytes: 0, measuredHeadroomBytes: 10 * gib,
                reserveBytes: 0, capBytes: 1 * gib
            ) == 1 * gib)
        // Above it: changes nothing (caps, never floors — ADR-0018).
        #expect(
            DynamicCeilingPolicy.ceilingBytes(
                residentBytes: 0, measuredHeadroomBytes: 10 * gib,
                reserveBytes: 0, capBytes: 100 * gib
            ) == 8 * gib)
    }
}

// MARK: - Headroom sample (pure)

struct MemoryHeadroomSampleTests {

    @Test func headroomSumsFreePurgeableAndReclaimable() {
        let sample = MemoryHeadroomSample(
            freeBytes: 2 * gib, purgeableBytes: 1 * gib, reclaimableBytes: 5 * gib
        )
        #expect(sample.headroomBytes == 8 * gib)
    }

    @Test func reclaimableDefaultsToZeroForFreeOnlySamples() {
        let sample = MemoryHeadroomSample(freeBytes: 3 * gib, purgeableBytes: 0)
        #expect(sample.reclaimableBytes == 0)
        #expect(sample.headroomBytes == 3 * gib)
    }

    /// Issue #236 regression: on a large-model machine the kernel parks
    /// most of RAM as inactive, so a free+purgeable-only sample reads
    /// below the Active-Inference Reserve and zeros the ceiling — the RAM
    /// cache turns off. Counting the reclaimable buckets keeps the
    /// ceiling truthful and positive.
    @Test func reclaimableMemoryKeepsCeilingPositiveUnderLargeModelPressure() {
        // Live-measured 35B-A3B shape: ~4.6 GiB free+purgeable, 4 GiB
        // bootstrap reserve → 0.8·4.6 − 4 < 0 → ceiling 0.
        let freeOnly = MemoryHeadroomSample(
            freeBytes: 4 * gib + 640 * 1024 * 1024, purgeableBytes: 0
        )
        let freeOnlyCeiling = DynamicCeilingPolicy.ceilingBytes(
            residentBytes: 0,
            measuredHeadroomBytes: freeOnly.headroomBytes,
            reserveBytes: ActiveInferenceReserve.bootstrapPerLaneBytes,
            capBytes: nil
        )
        #expect(freeOnlyCeiling == 0)

        // Same machine, ~11 GiB genuinely reclaimable (inactive +
        // speculative) folded in → the ceiling recovers to a usable size.
        let withReclaimable = MemoryHeadroomSample(
            freeBytes: 4 * gib + 640 * 1024 * 1024,
            purgeableBytes: 0,
            reclaimableBytes: 11 * gib
        )
        let recoveredCeiling = DynamicCeilingPolicy.ceilingBytes(
            residentBytes: 0,
            measuredHeadroomBytes: withReclaimable.headroomBytes,
            reserveBytes: ActiveInferenceReserve.bootstrapPerLaneBytes,
            capBytes: nil
        )
        #expect(recoveredCeiling > 8 * gib)
    }
}

// MARK: - Active-Inference Reserve (pure)

struct ActiveInferenceReserveTests {

    @Test func bootstrapsPerLaneBeforeAnyLeafIsObserved() {
        let reserve = ActiveInferenceReserve()
        #expect(reserve.perLaneBytes == ActiveInferenceReserve.bootstrapPerLaneBytes)
    }

    @Test func observedLeavesSizeTheLaneAtTwiceTheLargestLeaf() {
        var reserve = ActiveInferenceReserve()
        // Small leaves never shrink the lane below the bootstrap.
        reserve.observeLeaf(bytes: 1 * gib)
        #expect(reserve.perLaneBytes == ActiveInferenceReserve.bootstrapPerLaneBytes)
        // A leaf big enough that 2× exceeds the bootstrap takes over —
        // the capture deep-copy is the structural 2×.
        reserve.observeLeaf(bytes: 3 * gib)
        #expect(reserve.perLaneBytes == 6 * gib)
        // The estimate is a running max, not a last-value.
        reserve.observeLeaf(bytes: 2 * gib)
        #expect(reserve.perLaneBytes == 6 * gib)
    }

    @Test func reserveIsCountAwareAndFlooredAtOneLane() {
        let reserve = ActiveInferenceReserve()
        let perLane = reserve.perLaneBytes
        #expect(reserve.reserveBytes(lanes: 0) == perLane)
        #expect(reserve.reserveBytes(lanes: 1) == perLane)
        #expect(reserve.reserveBytes(lanes: 3) == 3 * perLane)
    }
}

// MARK: - Band rebase (pure)

struct BudgetBandRebaseTests {

    @Test func saturatedBandFollowsTheNewCeilingBothWays() {
        let band = PrefixCacheBudgetBand(ceilingBytes: 100)
        let up = band.rebasingCeiling(200, floorBytes: 0)
        #expect(up.ceilingBytes == 200)
        #expect(up.currentBytes == 200)
        let down = band.rebasingCeiling(40, floorBytes: 0)
        #expect(down.ceilingBytes == 40)
        #expect(down.currentBytes == 40)
    }

    @Test func retreatedBandCarriesCurrentAndRegrowsOneStep() {
        // Retreat to 25, then a measurement raises the ceiling to 160:
        // current carries over and regrows exactly one step (160/8) —
        // the measurement cadence is the slow-up clock, never a jump.
        let band = PrefixCacheBudgetBand(ceilingBytes: 100)
            .folding(.warning, floorBytes: 0)
            .folding(.warning, floorBytes: 0)
        #expect(band.currentBytes == 25)
        let rebased = band.rebasingCeiling(160, floorBytes: 0)
        #expect(rebased.ceilingBytes == 160)
        #expect(rebased.currentBytes == 25 + 160 / PrefixCacheBudgetBand.regrowthDenominator)
    }

    @Test func retreatedBandClampsIntoASmallerCeiling() {
        let band = PrefixCacheBudgetBand(ceilingBytes: 100)
            .folding(.warning, floorBytes: 0)  // current 50
        let rebased = band.rebasingCeiling(30, floorBytes: 0)
        #expect(rebased.ceilingBytes == 30)
        // Clamped to the new ceiling; the one regrowth step saturates.
        #expect(rebased.currentBytes == 30)
    }

    @Test func floorIsRespectedThroughARebase() {
        let band = PrefixCacheBudgetBand(ceilingBytes: 100)
            .folding(.critical, floorBytes: 10)  // current 10
        let rebased = band.rebasingCeiling(80, floorBytes: 20)
        #expect(rebased.currentBytes >= 20)
        #expect(rebased.currentBytes <= 80)
    }
}

// MARK: - SSD budget policy (pure)

struct SSDBudgetPolicyTests {

    @Test func budgetIsAFractionOfFreeDiskPlusOwnBytes() {
        // 400 GiB free + 40 GiB already ours → 0.25 · 440 = 110 GiB.
        let budget = SSDBudgetPolicy.budgetBytes(
            freeDiskBytes: 400 * gib,
            currentTierBytes: 40 * gib,
            capBytes: nil
        )
        #expect(budget == 110 * gib)
    }

    @Test func flooredAtTheOldTwentyGiBDefault() {
        let budget = SSDBudgetPolicy.budgetBytes(
            freeDiskBytes: 8 * gib,
            currentTierBytes: 0,
            capBytes: nil
        )
        #expect(budget == SSDBudgetPolicy.floorBytes)
    }

    @Test func cappedAtTheAbsoluteCeiling() {
        let budget = SSDBudgetPolicy.budgetBytes(
            freeDiskBytes: 4000 * gib,
            currentTierBytes: 0,
            capBytes: nil
        )
        #expect(budget == SSDBudgetPolicy.absoluteCapBytes)
    }

    @Test func userCapWinsEvenBelowTheFloor() {
        // A user cap set below the measured value is honored — caps
        // beat the floor, which only floors the *default* function.
        let budget = SSDBudgetPolicy.budgetBytes(
            freeDiskBytes: 400 * gib,
            currentTierBytes: 0,
            capBytes: 5 * gib
        )
        #expect(budget == 5 * gib)
    }
}

// MARK: - Hydration gate (pure)

struct HydrationGatePolicyTests {

    @Test func gateRejectsWhenTheReadCostsMoreThanTheRecompute() {
        // A terabyte-sized chain to recover a 4-token span: recompute
        // wins at any measured throughput.
        let admits = EvictionPolicy.hydrationGateAdmits(
            hydrationBytes: 1 << 40,
            hitOffset: 8,
            alternativeOffset: 4,
            config: EvictionConfiguration()
        )
        #expect(admits == false)
    }

    @Test func gateAdmitsWhenHydrationIsCheap() {
        let admits = EvictionPolicy.hydrationGateAdmits(
            hydrationBytes: 0,
            hitOffset: 1000,
            alternativeOffset: 0,
            config: EvictionConfiguration()
        )
        #expect(admits == true)
    }

    @Test func unmeasurableEstimatesAdmit() {
        // A zeroed throughput cannot price the trade — hydrate, which
        // is the pre-gate behavior.
        let config = EvictionConfiguration(
            estimates: MeasuredSecondsEstimates(
                prefillFlopsPerSecond: 0, hydrationBytesPerSecond: 0
            )
        )
        let admits = EvictionPolicy.hydrationGateAdmits(
            hydrationBytes: 1 << 40, hitOffset: 8, alternativeOffset: 4, config: config
        )
        #expect(admits == true)
    }
}

// MARK: - Manager measurement flow

@MainActor
struct DynamicBudgetCeilingManagerTests {

    private let key = CachePartitionKey(modelID: "ceiling-test", kvBits: nil, kvGroupSize: 64)

    private var diagnostics: PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "ceiling-test", kvBits: nil, kvGroupSize: 64
        )
    }

    /// An admission triggers a measurement, and a saturated band (no
    /// retreat in progress) follows the measured ceiling outright.
    /// The measurement runs *before* the entry lands, so resident bytes
    /// are 0 here: ceiling = 0.8 · 10 GiB − one bootstrap lane reserve.
    @Test func admissionMeasuresAndTheSaturatedBandFollowsTheCeiling() {
        let headroom = InMemoryMemoryHeadroomSource(
            next: MemoryHeadroomSample(freeBytes: 10 * gib, purgeableBytes: 0)
        )
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,  // bootstrap, replaced by measurement
            headroomSource: headroom
        )
        PrefixCacheTestFixtures.admitUniformLeaf(manager, tokens: Array(1...10), partitionKey: key)

        let expected = 8 * gib - ActiveInferenceReserve.bootstrapPerLaneBytes
        #expect(manager.budgetBand.ceilingBytes == expected)
        #expect(manager.memoryBudgetBytes == expected)
    }

    /// A contested measurement (no headroom) collapses the ceiling and
    /// the drain runs immediately — with the Budget Floor still honored:
    /// the freshest leaf is never a victim (the phase-1 invariant holds
    /// through a measured retreat too).
    @Test func contestedMeasurementDrainsButTheFloorHolds() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let headroom = InMemoryMemoryHeadroomSource()  // nil: no sample yet
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 100,
            headroomSource: headroom
        )
        PrefixCacheTestFixtures.admitUniformLeaf(manager, tokens: Array(1...10), partitionKey: key)
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(20...29), partitionKey: key)
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(40...49), partitionKey: key)
        #expect(manager.stats.snapshotCount == 3)

        headroom.next = MemoryHeadroomSample(freeBytes: 0, purgeableBytes: 0)
        let evictions = manager.reevaluateBudgetCeiling()

        #expect(manager.budgetBand.ceilingBytes == 0)
        #expect(evictions.count == 2)
        #expect(manager.stats.snapshotCount == 1)
        // The freshest leaf survived the zero-budget collapse.
        #expect(manager.lookup(tokens: Array(40...49), partitionKey: key).snapshot != nil)
    }

    /// The user cap binds at construction (bootstrap) and through every
    /// measurement; a cap above the measured ceiling changes nothing.
    @Test func userCapBindsAtBootstrapAndThroughMeasurement() {
        let headroom = InMemoryMemoryHeadroomSource(
            next: MemoryHeadroomSample(freeBytes: 100 * gib, purgeableBytes: 0)
        )
        let capped = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,
            headroomSource: headroom,
            ramBudgetCapBytes: 1 * gib
        )
        #expect(capped.memoryBudgetBytes == 1 * gib)
        capped.reevaluateBudgetCeiling()
        #expect(capped.budgetBand.ceilingBytes == 1 * gib)

        let uncappedValue = 80 * gib - ActiveInferenceReserve.bootstrapPerLaneBytes
        let generous = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,
            headroomSource: headroom,
            ramBudgetCapBytes: 500 * gib
        )
        generous.reevaluateBudgetCeiling()
        #expect(generous.budgetBand.ceilingBytes == uncappedValue)
    }

    /// The measurement cadence is throttled, and an explicit budget
    /// override (E2E tooling) suspends measurement entirely — a
    /// scripted scenario is never re-measured out from under.
    @Test func throttleAndOverrideSuspendMeasurement() {
        let headroom = InMemoryMemoryHeadroomSource(
            next: MemoryHeadroomSample(freeBytes: 10 * gib, purgeableBytes: 0)
        )
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,
            headroomSource: headroom
        )
        let start: ContinuousClock.Instant = .now
        manager.reevaluateBudgetCeilingIfDue(now: start)
        let measured = manager.memoryBudgetBytes
        #expect(measured == 8 * gib - ActiveInferenceReserve.bootstrapPerLaneBytes)

        // Inside the throttle window: a bigger sample changes nothing.
        headroom.next = MemoryHeadroomSample(freeBytes: 20 * gib, purgeableBytes: 0)
        manager.reevaluateBudgetCeilingIfDue(now: start + .seconds(5))
        #expect(manager.memoryBudgetBytes == measured)

        // Past the window: the new sample lands.
        manager.reevaluateBudgetCeilingIfDue(now: start + .seconds(16))
        #expect(
            manager.memoryBudgetBytes == 16 * gib - ActiveInferenceReserve.bootstrapPerLaneBytes)

        // An explicit override wins until the cache is rebuilt.
        manager.setMemoryBudget(1 * gib)
        manager.reevaluateBudgetCeilingIfDue(now: start + .seconds(3600))
        #expect(manager.memoryBudgetBytes == 1 * gib)
    }

    /// The reserve is count-aware over in-flight requests: two active
    /// lanes subtract two reserves; completing them returns the lane.
    @Test func reserveCountsInFlightRequestLanes() async {
        let headroom = InMemoryMemoryHeadroomSource(
            next: MemoryHeadroomSample(freeBytes: 20 * gib, purgeableBytes: 0)
        )
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,
            headroomSource: headroom
        )
        let oneLane = 16 * gib - ActiveInferenceReserve.bootstrapPerLaneBytes
        let twoLanes = 16 * gib - 2 * ActiveInferenceReserve.bootstrapPerLaneBytes

        manager.reevaluateBudgetCeiling()
        #expect(manager.budgetBand.ceilingBytes == oneLane)

        // Two requests register their lanes through resolve — a miss
        // still counts: the generation's working set exists either way.
        let first = UUID()
        let second = UUID()
        _ = await manager.resolve(
            tokens: [1, 2, 3], promptTokenCount: 3, partitionKey: key,
            modelFingerprint: nil, diagnostics: diagnostics,
            pinningRestorePathFor: first
        )
        _ = await manager.resolve(
            tokens: [4, 5, 6], promptTokenCount: 3, partitionKey: key,
            modelFingerprint: nil, diagnostics: diagnostics,
            pinningRestorePathFor: second
        )
        manager.reevaluateBudgetCeiling()
        #expect(manager.budgetBand.ceilingBytes == twoLanes)

        manager.completeRequest(requestID: first)
        manager.completeRequest(requestID: second)
        manager.reevaluateBudgetCeiling()
        #expect(manager.budgetBand.ceilingBytes == oneLane)
    }

    /// Every recomputation leaves a `budgetMeasure` trace with the full
    /// input set, and an effective move pairs it with
    /// `budgetChange reason=measurement` (PRD #149 item 5).
    @Test func measurementEmitsDiagnostics() {
        let headroom = InMemoryMemoryHeadroomSource(
            next: MemoryHeadroomSample(freeBytes: 10 * gib, purgeableBytes: 0)
        )
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 3 * gib,
            headroomSource: headroom
        )
        let sink = RecordingLineSink()
        let handle = PrefixCacheDiagnostics.addTestSink(sink.handler)
        defer { PrefixCacheDiagnostics.removeTestSink(handle) }

        manager.reevaluateBudgetCeiling()

        let lines = sink.drain()
        let measures = lines.filter { $0.contains("event=budgetMeasure") }
        #expect(measures.count == 1)
        #expect(measures[0].contains("headroomBytes=\(10 * gib)"))
        #expect(
            measures[0].contains(
                "reserveBytes=\(ActiveInferenceReserve.bootstrapPerLaneBytes)"))
        #expect(measures[0].contains("lanes=1"))
        #expect(measures[0].contains("capBytes=auto"))
        let changes = lines.filter { $0.contains("event=budgetChange") }
        #expect(changes.count == 1)
        #expect(changes[0].contains("reason=measurement"))
    }

    private final class RecordingLineSink: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []

        var handler: @Sendable (String) -> Void {
            { [weak self] line in
                guard let self else { return }
                self.lock.lock()
                self.lines.append(line)
                self.lock.unlock()
            }
        }

        func drain() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            let copy = lines
            lines.removeAll()
            return copy
        }
    }
}

// MARK: - Hydration gate + interruption through resolve

@MainActor
struct HydrationGateResolveTests {

    private let key = CachePartitionKey(modelID: "gate-test", kvBits: nil, kvGroupSize: 64)
    private let fingerprint = "fp"

    private var diagnostics: PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "gate-test", kvBits: nil, kvGroupSize: 64
        )
    }

    private func makeManager(hydrating: InMemorySnapshotHydrating) -> PrefixCacheManager {
        let manager = PrefixCacheManager(memoryBudgetBytes: 100 * 1024 * 1024)
        manager.setSnapshotHydratingForTesting(hydrating)
        return manager
    }

    /// A body-less SSD hit whose read is priced above the recompute
    /// span degrades to the deepest resident RAM body: no `loadSync`,
    /// backing intact, the shallow body served as an ordinary hit.
    @Test func gateServesTheResidentBodyInsteadOfAnExpensiveRead() async {
        let hydrating = InMemorySnapshotHydrating()
        let manager = makeManager(hydrating: hydrating)
        let path = Array(1...8)

        // Shallow resident RAM leaf at offset 4.
        PrefixCacheTestFixtures.admitUniformLeaf(
            manager, tokens: Array(path.prefix(4)), partitionKey: key)
        // Deep body-less state-5 node whose chain is absurdly large —
        // recomputing 4 tokens beats a terabyte read at any throughput.
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 8, bytesOnDisk: 1 << 40)
        manager.restoreSnapshotRef(
            path: path, snapshotRef: ref, partitionKey: key, lastAccessTime: .now
        )

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        #expect(hydrating.loadSyncCalls.isEmpty)
        #expect(resolved.hydratedFromSSD == false)
        #expect(resolved.lookup.snapshot?.tokenOffset == 4)
        // The skipped hit's ref is untouched — still the deepest
        // hittable node for a future request.
        if case .ssdHit = manager.lookup(tokens: path, partitionKey: key).reason {
        } else {
            Issue.record("expected the state-5 node to remain hittable")
        }
    }

    /// A cheap read passes the gate and hydrates exactly as before.
    @Test func gateAdmitsACheapReadAndHydrationProceeds() async {
        let hydrating = InMemorySnapshotHydrating()
        let manager = makeManager(hydrating: hydrating)
        let path = Array(1...8)
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 8, bytesOnDisk: 0)
        manager.restoreSnapshotRef(
            path: path, snapshotRef: ref, partitionKey: key, lastAccessTime: .now
        )
        hydrating.programSuccess(
            id: ref.snapshotID,
            body: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 8, type: .leaf)
        )

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics
        )

        #expect(hydrating.loadSyncCalls == [ref.snapshotID])
        #expect(resolved.hydratedFromSSD == true)
        #expect(resolved.lookup.snapshot?.tokenOffset == 8)
    }

    /// An interrupted hydration (a foreground request became due) is
    /// not a failure: the resolve surfaces a miss, but the node's ref
    /// survives — the next caller hydrates it normally (PRD #149 item
    /// 7: cancellable hydration never costs the cache anything).
    @Test func interruptedHydrationLeavesTheBackingIntact() async {
        let hydrating = InMemorySnapshotHydrating()
        let manager = makeManager(hydrating: hydrating)
        let path = Array(1...8)
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 8, bytesOnDisk: 1024)
        manager.restoreSnapshotRef(
            path: path, snapshotRef: ref, partitionKey: key, lastAccessTime: .now
        )
        hydrating.programSuccess(
            id: ref.snapshotID,
            body: PrefixCacheTestFixtures.makeUniformSnapshot(offset: 8, type: .leaf)
        )

        let resolved = await manager.resolve(
            tokens: path, promptTokenCount: path.count, partitionKey: key,
            modelFingerprint: fingerprint, diagnostics: diagnostics,
            interruption: { true }
        )

        #expect(resolved.lookup.snapshot == nil)
        #expect(hydrating.loadSyncCalls == [ref.snapshotID])
        #expect(manager.cumulativeCounters.hydrations == 0)
        // No forgiving clear ran: the node is still a hittable state-5.
        if case .ssdHit = manager.lookup(tokens: path, partitionKey: key).reason {
        } else {
            Issue.record("expected the state-5 node to survive the interruption")
        }
    }
}

// MARK: - Ledger dynamic budget

struct SnapshotLedgerDynamicBudgetTests {

    private func makeScratchDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ledger-dyn-\(UUID().uuidString)", isDirectory: true)
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

    /// The measured budget replaces the bootstrap at the admission cut:
    /// a payload larger than the bootstrap admits once free disk says
    /// there is room (fraction, floored, capped — `SSDBudgetPolicy`).
    @Test func measuredFreeDiskGrowsTheBudgetPastTheBootstrap() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,  // bootstrap + floor
            manifestDebounce: .milliseconds(20),
            freeDiskBytesProvider: { _ in 400 * gib }
        )
        // 0.25 · 400 GiB = 100 GiB measured budget.
        let (decision, evicted) = ledger.admit(makeDescriptor(bytes: 2_000_000))
        if case .admit = decision {
        } else {
            Issue.record("expected admit under the measured budget, got \(decision)")
        }
        #expect(evicted.isEmpty)
        #expect(ledger.currentBudgetBytes() == 100 * gib)
    }

    /// A user cap below the measured value is enforced at the same cut.
    @Test func userCapBindsTheMeasuredBudget() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20),
            budgetCapBytes: 1_500_000,
            freeDiskBytesProvider: { _ in 400 * gib }
        )
        let (decision, _) = ledger.admit(makeDescriptor(bytes: 2_000_000))
        if case .drop(.exceedsBudget) = decision {
        } else {
            Issue.record("expected the cap to reject, got \(decision)")
        }
        #expect(ledger.currentBudgetBytes() == 1_500_000)
    }

    /// No provider (tests, replay caches) = the static bootstrap
    /// forever — the pre-dynamic behavior every fixture pins.
    @Test func withoutAProviderTheBootstrapIsTheBudget() {
        let root = makeScratchDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = SnapshotLedger(
            rootURL: root,
            budgetBytes: 1_000_000,
            manifestDebounce: .milliseconds(20)
        )
        let (decision, _) = ledger.admit(makeDescriptor(bytes: 2_000_000))
        if case .drop(.exceedsBudget) = decision {
        } else {
            Issue.record("expected the static bootstrap to reject, got \(decision)")
        }
        #expect(ledger.currentBudgetBytes() == 1_000_000)
    }
}
