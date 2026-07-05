//
//  StorageActivityGate.swift
//  tesseract
//
//  Shared busy signal between the inference path and the SSD writer
//  (PRD #150). Concurrent large-block reads and writes on one NVMe
//  device can collapse total bandwidth ~60 % (the #148 research
//  report; oMLX's unconfirmed 5.5× prefill regression is the same
//  suspicion), so deferred-class writes wait for the device to go
//  quiet: the writer holds them while a hydration read or a prefill
//  is in flight, bounded by `SSDSnapshotStore.maxDeferredHoldup` so
//  they always land eventually. Guarantee- and write-through-class
//  writes ignore the gate — durability outranks bandwidth.
//
//  A `nonisolated` lock-based class because both sides are synchronous
//  hot paths: the store's writer task polls between items, and the
//  prefill/hydration marks run inside `container.perform`.
//

import Foundation

nonisolated final class StorageActivityGate: @unchecked Sendable {
    private let lock = NSLock()
    private var activeHydrations = 0
    private var activePrefills = 0

    /// True while any hydration read or prefill run is in flight —
    /// the window deferred-class writes stay out of.
    var isBusy: Bool {
        lock.lock()
        defer { lock.unlock() }
        return activeHydrations > 0 || activePrefills > 0
    }

    func hydrationDidBegin() {
        lock.lock()
        activeHydrations += 1
        lock.unlock()
    }

    func hydrationDidEnd() {
        lock.lock()
        activeHydrations = max(0, activeHydrations - 1)
        lock.unlock()
    }

    func prefillDidBegin() {
        lock.lock()
        activePrefills += 1
        lock.unlock()
    }

    func prefillDidEnd() {
        lock.lock()
        activePrefills = max(0, activePrefills - 1)
        lock.unlock()
    }

    /// Scoped prefill mark for the call-site ergonomics of the two
    /// `session.prefill` sites and the speculative pass.
    func withPrefillMarked<T>(_ body: () throws -> T) rethrows -> T {
        prefillDidBegin()
        defer { prefillDidEnd() }
        return try body()
    }

    func withPrefillMarked<T>(_ body: () async throws -> T) async rethrows -> T {
        prefillDidBegin()
        defer { prefillDidEnd() }
        return try await body()
    }
}
