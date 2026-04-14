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

/// Immutable snapshot of the SSD prefix-cache tier configuration.
///
/// Downstream consumers gate on `self.ssdConfig?.enabled == true` as a
/// synchronous actor-isolated read. Keeping `enabled` explicit (rather
/// than using nil-vs-non-nil alone) lets those consumers branch on a
/// single value without reasoning about optional semantics.
struct SSDPrefixCacheConfig: Sendable, Equatable {
    /// Mirror of `SettingsManager.prefixCacheSSDEnabled` at snapshot time.
    /// Always `true` when produced by `SettingsManager.makeSSDPrefixCacheConfig()`
    /// (the factory returns `nil` when disabled).
    let enabled: Bool

    /// Resolved SSD root directory.
    let rootURL: URL

    /// Hard top-level byte budget for on-disk snapshots. The writer runs
    /// its admission-time LRU cut when a new payload would push
    /// `currentSSDBytes` above this value.
    let budgetBytes: Int

    /// Front-door pending-queue byte cap. The writer drops oldest pending
    /// entries when a burst of captures exceeds this size. Independent of
    /// `budgetBytes`: the pending queue is in-memory and bounds the
    /// transient memory pressure of the write-through pipeline.
    let maxPendingBytes: Int

    /// Construct a config with `maxPendingBytes` auto-sized from physical
    /// RAM. The rule — `min(4 GiB, physicalMemory / 16)` — lives here so
    /// `SettingsManager` does not need to know SSD tuning constants.
    static func withAutoPendingCap(
        rootURL: URL,
        budgetBytes: Int
    ) -> SSDPrefixCacheConfig {
        let physicalMemory = Int(clamping: ProcessInfo.processInfo.physicalMemory)
        let maxPendingBytes = min(
            4 * 1024 * 1024 * 1024,   // 4 GiB hard ceiling
            physicalMemory / 16        // 1/16 of physical RAM
        )
        return SSDPrefixCacheConfig(
            enabled: true,
            rootURL: rootURL,
            budgetBytes: budgetBytes,
            maxPendingBytes: maxPendingBytes
        )
    }
}
