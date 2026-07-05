//
//  SSDPrefixCacheConfig.swift
//  tesseract
//
//  Immutable snapshot of the SSD prefix-cache configuration. Captured on
//  MainActor at model load and held by `LLMActor` for the lifetime of the
//  load because the hot prefix-cache path inside `container.perform` is
//  synchronous and cannot await back to MainActor for settings reads.
//

import Foundation

/// The SSD tier's dynamic budget formula (ADR-0018): a fraction of free
/// disk space with an absolute cap, floored at the old 20 GiB constant —
/// which survives only as the bootstrap value and that floor. Re-evaluated
/// periodically by `SnapshotLedger` at admission time. A user cap
/// (nil = "Automatic") only ever lowers the result — caps, never floors.
nonisolated enum SSDBudgetPolicy {
    /// The old fixed budget (see issue #16 for the 20-vs-50 GiB history):
    /// now the bootstrap value before the first disk measurement and the
    /// floor of the measured default.
    static let floorBytes = 20 * 1024 * 1024 * 1024  // 20 GiB

    /// Fraction of the disk the tier may claim. Applied to free space
    /// *plus* the tier's own current bytes — what we hold is ours, not
    /// missing capacity — so the budget doesn't self-deflate as it fills.
    static let freeDiskFraction = 0.25

    /// Absolute ceiling regardless of disk size: past this, more cache
    /// stops buying TTFT (sessions don't span it) and starts costing GC
    /// time (phase 3's problem).
    static let absoluteCapBytes = 128 * 1024 * 1024 * 1024  // 128 GiB

    static func budgetBytes(
        freeDiskBytes: Int,
        currentTierBytes: Int,
        floorBytes: Int = SSDBudgetPolicy.floorBytes,
        capBytes: Int?
    ) -> Int {
        let claimable = Int(
            Double(max(freeDiskBytes + currentTierBytes, 0)) * freeDiskFraction
        )
        let measured = max(floorBytes, min(absoluteCapBytes, claimable))
        return applyBudgetCap(measured, cap: capBytes)
    }

    /// Production free-space probe for the volume holding `rootURL`.
    /// `nil` when the resource read fails — the caller keeps its
    /// current budget.
    static func measuredFreeDiskBytes(rootURL: URL) -> Int? {
        let values = try? rootURL.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        return values?.volumeAvailableCapacityForImportantUsage.map { Int(clamping: $0) }
    }
}

/// The **stale-partition GC** policy (PRD #150): partitions unused past
/// `maxUnusedAge` are reclaimed at warm start — their descriptors leave
/// the manifest, their directories are deleted, and their bytes return
/// to the budget. "Used" means an admission registered the partition or
/// an SSD hydration hit one of its residents; a warm start alone does
/// not refresh the stamp (otherwise every launch would reset the clock
/// and nothing would ever age out).
///
/// Staleness is *relative to the tier's most recent use*, not to the
/// wall clock: a partition is stale when it is `maxUnusedAge` older
/// than the freshest valid partition's stamp. An absolute clock would
/// reclaim the entire cache after any week the app sat unused; the
/// relative rule is idle-proof (all partitions age together, nothing
/// is reclaimed) while a variant abandoned mid-activity still goes.
/// The freshest partition never ages out by construction.
nonisolated enum SSDStalePartitionPolicy {
    /// Use-gap past which a partition is reclaimed. 7 days (owner
    /// decision 2026-07-05, revised down from the grilling's ~30):
    /// stale kv-config and template-digest variants of a still-loaded
    /// model used to accumulate forever — only a fingerprint change
    /// cleared them.
    static let maxUnusedAge: TimeInterval = 7 * 24 * 3600

    /// Minimum spacing between `lastUsedAt` re-stamps. Day-scale GC
    /// needs no finer resolution, and the throttle keeps the per-hit /
    /// per-admission bump from turning into a sidecar write storm.
    static let lastUsedRefreshInterval: TimeInterval = 6 * 3600
}

/// Immutable snapshot of the SSD prefix-cache tier configuration.
///
/// Downstream consumers gate on `self.ssdConfig?.enabled == true` as a
/// synchronous actor-isolated read. Keeping `enabled` explicit (rather
/// than using nil-vs-non-nil alone) lets those consumers branch on a
/// single value without reasoning about optional semantics.
nonisolated struct SSDPrefixCacheConfig: Sendable, Equatable {
    /// Mirror of `SettingsManager.prefixCacheSSDEnabled` at snapshot time.
    /// Always `true` when produced by `SettingsManager.makeSSDPrefixCacheConfig()`
    /// (the factory returns `nil` when disabled).
    let enabled: Bool

    /// Resolved SSD root directory.
    let rootURL: URL

    /// *Bootstrap* top-level byte budget for on-disk snapshots (ADR-0018):
    /// the value in force before the first free-disk measurement, and —
    /// in production, where it is `SSDBudgetPolicy.floorBytes` — the floor
    /// of the measured default. Static for the config's lifetime when
    /// `measuresFreeDisk` is `false` (tests, replay caches).
    let budgetBytes: Int

    /// User cap on the SSD budget (nil = "Automatic (recommended)").
    /// Caps the measured value, never raises it.
    let budgetCapBytes: Int?

    /// Whether `SnapshotLedger` periodically re-derives the budget from
    /// measured free disk space. Production passes `true`; the default
    /// keeps test fixtures on the exact static budgets they assert.
    let measuresFreeDisk: Bool

    /// Front-door pending-queue byte cap. The writer drops oldest pending
    /// entries when a burst of captures exceeds this size. Independent of
    /// `budgetBytes`: the pending queue is in-memory and bounds the
    /// transient memory pressure of the write-through pipeline.
    let maxPendingBytes: Int

    init(
        enabled: Bool,
        rootURL: URL,
        budgetBytes: Int,
        maxPendingBytes: Int,
        budgetCapBytes: Int? = nil,
        measuresFreeDisk: Bool = false
    ) {
        self.enabled = enabled
        self.rootURL = rootURL
        self.budgetBytes = budgetBytes
        self.budgetCapBytes = budgetCapBytes
        self.measuresFreeDisk = measuresFreeDisk
        self.maxPendingBytes = maxPendingBytes
    }

    /// Construct a config with `maxPendingBytes` auto-sized from physical
    /// RAM. The rule — `min(4 GiB, physicalMemory / 16)` — lives here so
    /// `SettingsManager` does not need to know SSD tuning constants.
    static func withAutoPendingCap(
        rootURL: URL,
        budgetBytes: Int = SSDBudgetPolicy.floorBytes,
        budgetCapBytes: Int? = nil,
        measuresFreeDisk: Bool = false
    ) -> SSDPrefixCacheConfig {
        let physicalMemory = Int(clamping: ProcessInfo.processInfo.physicalMemory)
        let maxPendingBytes = min(
            4 * 1024 * 1024 * 1024,  // 4 GiB hard ceiling
            physicalMemory / 16  // 1/16 of physical RAM
        )
        return SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: budgetBytes,
            maxPendingBytes: maxPendingBytes,
            budgetCapBytes: budgetCapBytes,
            measuresFreeDisk: measuresFreeDisk
        )
    }
}
