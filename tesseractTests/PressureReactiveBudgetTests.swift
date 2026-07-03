import Foundation
import Testing

@testable import Tesseract_Agent

/// Slice #88 (PRD #82), part 1: the pure budget-band fold under
/// scripted event sequences. No manager, no trees — the fold is a
/// value-to-value function and is tested as one.
struct PrefixCacheBudgetBandTests {

    private let ceiling = 1_000_000

    @Test func startsAtTheCeiling() {
        let band = PrefixCacheBudgetBand(ceilingBytes: ceiling)
        #expect(band.currentBytes == ceiling)
        #expect(band.ceilingBytes == ceiling)
    }

    @Test func warningHalvesAndCriticalDropsToFloor() {
        let floor = 100_000
        var band = PrefixCacheBudgetBand(ceilingBytes: ceiling)

        band = band.folding(.warning, floorBytes: floor)
        #expect(band.currentBytes == ceiling / 2)
        band = band.folding(.warning, floorBytes: floor)
        #expect(band.currentBytes == ceiling / 4)

        band = band.folding(.critical, floorBytes: floor)
        #expect(band.currentBytes == floor)
    }

    @Test func warningClampsAtTheFloor() {
        let floor = 600_000
        let band = PrefixCacheBudgetBand(ceilingBytes: ceiling)
            .folding(.warning, floorBytes: floor)
        // Half the ceiling would be 500k — below the floor — so the
        // floor wins.
        #expect(band.currentBytes == floor)
    }

    @Test func floorIsClampedIntoTheBand() {
        let band = PrefixCacheBudgetBand(ceilingBytes: ceiling)
        // A floor that momentarily exceeds the ceiling cannot invert
        // the band, and a negative floor cannot go below zero.
        #expect(band.folding(.critical, floorBytes: ceiling * 2).currentBytes == ceiling)
        #expect(band.folding(.critical, floorBytes: -5).currentBytes == 0)
    }

    @Test func regrowthIsSlowAndCapsAtCeiling() {
        var band = PrefixCacheBudgetBand(ceilingBytes: ceiling)
            .folding(.critical, floorBytes: 0)
        #expect(band.currentBytes == 0)

        let step = ceiling / PrefixCacheBudgetBand.regrowthDenominator
        band = band.folding(.normal, floorBytes: 0)
        #expect(band.currentBytes == step)

        // Full regrowth takes the documented number of clear events,
        // then saturates at the ceiling.
        for _ in 0..<(PrefixCacheBudgetBand.regrowthDenominator * 2) {
            band = band.folding(.normal, floorBytes: 0)
        }
        #expect(band.currentBytes == ceiling)
    }

    /// Flapping warning/normal must converge, not oscillate: the
    /// fast-down/slow-up asymmetry pulls the sequence to a fixpoint
    /// (down loses current/2, up regains only ceiling/8).
    @Test func flappingEventsConvergeWithoutOscillation() {
        var band = PrefixCacheBudgetBand(ceilingBytes: ceiling)
        var amplitudes: [Int] = []
        for _ in 0..<12 {
            let beforeFlap = band.currentBytes
            band = band.folding(.warning, floorBytes: 0)
            band = band.folding(.normal, floorBytes: 0)
            amplitudes.append(abs(band.currentBytes - beforeFlap))
        }
        // The per-flap movement shrinks monotonically to (near) zero —
        // the fixpoint where halving and one regrowth step cancel.
        #expect(amplitudes.first! > amplitudes.last!)
        let fixpoint = ceiling / 4  // c* = c*/2 + c/8
        #expect(
            abs(band.currentBytes - fixpoint) <= ceiling / PrefixCacheBudgetBand.regrowthDenominator
        )
        // And it never leaves the band.
        #expect(band.currentBytes >= 0 && band.currentBytes <= ceiling)
    }

    @Test func degenerateZeroCeilingStaysPinnedAtZero() {
        var band = PrefixCacheBudgetBand(ceilingBytes: 0)
        band = band.folding(.normal, floorBytes: 0)
        #expect(band.currentBytes == 0)
        band = band.folding(.warning, floorBytes: 0)
        #expect(band.currentBytes == 0)
    }
}

/// Slice #88, part 2: manager-level behavior driven end-to-end by the
/// in-memory pressure peer — shrink demotes down to the floor, floor
/// contents are never victims, regrowth raises the live budget, and
/// the telemetry snapshot exposes the band.
@MainActor
struct PressureReactiveBudgetManagerTests {

    private let key = CachePartitionKey(modelID: "pressure-test", kvBits: nil, kvGroupSize: 64)

    private func admit(
        _ manager: PrefixCacheManager,
        tokens: [Int],
        type: HybridCacheSnapshot.CheckpointType
    ) {
        switch type {
        case .leaf:
            PrefixCacheTestFixtures.admitUniformLeaf(manager, tokens: tokens, partitionKey: key)
        case .system, .branchPoint:
            manager.admit(
                SnapshotAdmission.checkpoints(
                    fullPromptTokens: tokens + [9_999],
                    candidates: [
                        .ramOnly(
                            PrefixCacheTestFixtures.makeUniformSnapshot(
                                offset: tokens.count, type: type
                            ))
                    ],
                    partitionKey: key
                )!)
        }
    }

    /// Critical pressure drains to exactly the floor: the `.system`
    /// chain and the most-recently-extended leaf survive; every other
    /// body is gone. Subsequent `normal` events regrow the live budget
    /// without touching contents.
    @Test func criticalShrinkStopsAtTheFloorAndRegrows() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let pressure = InMemoryMemoryPressureSource()
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 100,
            pressureSource: pressure
        )

        admit(manager, tokens: Array(1...10), type: .system)
        admit(manager, tokens: Array(20...29), type: .leaf)  // stale leaf
        admit(manager, tokens: Array(40...49), type: .leaf)  // freshest leaf
        admit(manager, tokens: Array(60...69), type: .branchPoint)
        #expect(manager.stats.snapshotCount == 4)

        let floor = manager.budgetFloorBytes()
        #expect(floor == snapBytes * 2, "floor = system + freshest leaf")

        pressure.send(.critical)
        #expect(manager.memoryBudgetBytes == floor)
        #expect(manager.totalSnapshotBytes == floor)
        // Floor membership: the system chain and the freshest leaf.
        #expect(manager.lookup(tokens: Array(1...10), partitionKey: key).snapshot != nil)
        #expect(manager.lookup(tokens: Array(40...49), partitionKey: key).snapshot != nil)
        #expect(manager.lookup(tokens: Array(20...29), partitionKey: key).snapshot == nil)
        #expect(manager.lookup(tokens: Array(60...69), partitionKey: key).snapshot == nil)

        // Pressure clears: the live budget regrows toward the ceiling,
        // one step per event, without evicting anything further.
        pressure.send(.normal)
        #expect(manager.memoryBudgetBytes > floor)
        #expect(manager.totalSnapshotBytes == floor)
        for _ in 0..<20 { pressure.send(.normal) }
        #expect(manager.memoryBudgetBytes == snapBytes * 100)
    }

    /// Warning halves the budget and the drain demotes (recovered, not
    /// terminal) when the victims are SSD-backed — the slice-87
    /// machinery is the shrink's mechanism.
    @Test func warningShrinkUsesRecoverableDrops() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let pressure = InMemoryMemoryPressureSource()
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 4,
            tieredStore: store,
            pressureSource: pressure
        )

        // Four leaves fill the ceiling exactly; back the two stalest
        // with committed refs so the warning drain's victims are
        // recoverable.
        for start in [1, 20, 40, 60] {
            admit(manager, tokens: Array(start...(start + 9)), type: .leaf)
        }
        let tree = store.tree(for: key)!
        for start in [1, 20] {
            let (node, _) = tree.findBestSnapshot(
                tokens: Array(start...(start + 9)), updateAccess: false
            )!
            let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 10)
            tree.admit(node: node, ref: ref)
            tree.commitRef(node: node, expectedID: ref.snapshotID)
        }
        manager.cumulativeCountersResetForTesting()

        pressure.send(.warning)
        #expect(manager.memoryBudgetBytes == snapBytes * 2)
        #expect(manager.totalSnapshotBytes <= snapBytes * 2)
        #expect(manager.cumulativeCounters.recoveredEvictions == 2)
        #expect(manager.cumulativeCounters.terminalEvictions == 0)
    }

    /// A machine that never signals pressure keeps today's behavior:
    /// the budget stays at the ceiling and nothing drains.
    @Test func unpressuredCeilingIsUnchanged() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 100,
            pressureSource: InMemoryMemoryPressureSource()
        )
        admit(manager, tokens: Array(1...10), type: .leaf)
        #expect(manager.memoryBudgetBytes == snapBytes * 100)
        #expect(manager.budgetBand.currentBytes == snapBytes * 100)
        #expect(manager.stats.snapshotCount == 1)
    }

    /// Telemetry surfaces the whole band: ceiling, current, and the
    /// content-defined floor.
    @Test func telemetryExposesTheBand() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let pressure = InMemoryMemoryPressureSource()
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 100,
            pressureSource: pressure
        )
        admit(manager, tokens: Array(1...10), type: .system)
        admit(manager, tokens: Array(20...29), type: .leaf)

        pressure.send(.warning)
        let snapshot = manager.makeTelemetrySnapshot()
        #expect(snapshot.budgetCeilingBytes == snapBytes * 100)
        #expect(snapshot.memoryBudgetBytes == snapBytes * 50)
        #expect(snapshot.budgetFloorBytes == snapBytes * 2)
    }

    /// The floor never starves an ordinary admission drain: a plain
    /// (non-pressure) `evictToFitBudget` keeps the unconditional
    /// semantics, including the zero-budget full drain tests rely on.
    @Test func ordinaryDrainsIgnoreTheFloor() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let manager = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)
        admit(manager, tokens: Array(1...10), type: .system)
        admit(manager, tokens: Array(20...29), type: .leaf)

        manager.setMemoryBudget(0)
        #expect(manager.stats.snapshotCount == 0)
    }

    // MARK: - Band-consistent overrides (PRD #137, user story 16)

    /// A budget override rebuilds the band around the new value, so the
    /// stale-band window is unrepresentable: subsequent pressure regrowth
    /// converges to the override, never back to the pre-override ceiling.
    /// (The retired direct-write path left the band at the old ceiling —
    /// the next `.normal` event silently clobbered the override.)
    @Test func budgetOverrideRebuildsTheBandSoRegrowthConvergesToTheOverride() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let pressure = InMemoryMemoryPressureSource()
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes * 100,
            pressureSource: pressure
        )
        admit(manager, tokens: Array(1...10), type: .leaf)

        // Shrink first, so the override lands mid-band — the exact state
        // the old direct write left inconsistent.
        pressure.send(.warning)
        #expect(manager.memoryBudgetBytes == snapBytes * 50)

        manager.setMemoryBudget(snapBytes * 10)
        #expect(manager.memoryBudgetBytes == snapBytes * 10)
        #expect(manager.budgetBand.ceilingBytes == snapBytes * 10)
        #expect(manager.budgetBand.currentBytes == snapBytes * 10)

        // Pressure clears: regrowth saturates at the OVERRIDE, not the
        // original 100-snapshot ceiling.
        for _ in 0..<20 { pressure.send(.normal) }
        #expect(manager.memoryBudgetBytes == snapBytes * 10)
    }

    /// The override drains immediately and hands back the evictions —
    /// the E2E runner's "tighten and observe pressure" step in one call.
    @Test func budgetOverrideEvictsImmediatelyAndReturnsTheEvictions() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let manager = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)
        admit(manager, tokens: Array(1...10), type: .leaf)
        admit(manager, tokens: Array(20...29), type: .leaf)
        admit(manager, tokens: Array(40...49), type: .leaf)

        let evictions = manager.setMemoryBudget(snapBytes)
        #expect(evictions.count == 2)
        #expect(manager.stats.snapshotCount == 1)
    }

    /// The alpha override writes the **Eviction Configuration** through
    /// the manager's one mutation entry.
    @Test func alphaOverrideWritesTheEvictionConfiguration() {
        let manager = PrefixCacheManager(memoryBudgetBytes: 1 << 20)
        #expect(manager.evictionConfig.alpha == 0.0)
        manager.setEvictionAlpha(2.0)
        #expect(manager.evictionConfig.alpha == 2.0)
    }
}
