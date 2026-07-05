//
//  DynamicBudgetCeiling.swift
//  tesseract
//
//  **Dynamic Budget Ceilings** (ADR-0018, PRD #149): the RAM-tier
//  ceiling comes from *measurement*, not load-time constants. A
//  periodic headroom sample (free + purgeable memory) feeds the
//  pressure band's ceiling; the old `(physRAM − weights − 20 GiB)/2`
//  formula survives only as the bootstrap value before the first
//  measurement. The `/2` divisor's job — protecting the in-flight
//  generation's working set — moves to the named, count-aware
//  **Active-Inference Reserve**.
//
//  This file is the whole seam: the Memory Headroom Source port, the
//  thin Mach adapter, the in-memory peer tests drive, the pure ceiling
//  policy, and the reserve fold.
//

import Foundation

// MARK: - Headroom sample

/// One measurement of machine memory headroom: bytes the OS could hand
/// this process without swapping. Free pages plus purgeable pages
/// (ADR-0018's "free + purgeable"); deliberately conservative — file-backed
/// reclaimable pages are not counted, and the fast pressure retreat
/// (ADR-0011) remains the guardrail for anything the sample got wrong.
nonisolated struct MemoryHeadroomSample: Sendable, Equatable {
    let freeBytes: Int
    let purgeableBytes: Int

    var headroomBytes: Int { freeBytes + purgeableBytes }
}

// MARK: - Port

/// The port `PrefixCacheManager` pulls headroom samples through. Same
/// `@MainActor`-sibling shape as `MemoryPressureSource`: the manager
/// samples synchronously from its own isolation at re-evaluation time
/// (admission-driven, throttled). The production adapter is
/// `MachMemoryHeadroomSource`; the test peer is
/// `InMemoryMemoryHeadroomSource`.
@MainActor
protocol MemoryHeadroomSource: AnyObject, Sendable {
    /// One fresh measurement, or `nil` when the host statistics call
    /// fails — the caller then keeps its current ceiling.
    func sample() -> MemoryHeadroomSample?
}

// MARK: - Production adapter

/// Thin adapter over `host_statistics64`. All it does is convert page
/// counts to bytes — every decision lives in `DynamicCeilingPolicy`
/// and the manager.
@MainActor
final class MachMemoryHeadroomSource: MemoryHeadroomSource {
    /// Stateless — constructible from any isolation (e.g. `LLMActor`'s
    /// load path); `sample()` stays MainActor.
    nonisolated init() {}

    func sample() -> MemoryHeadroomSample? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            Log.server.error("MachMemoryHeadroomSource: host_statistics64 failed (\(result))")
            return nil
        }
        let pageSize = Int(getpagesize())
        return MemoryHeadroomSample(
            freeBytes: Int(clamping: stats.free_count) * pageSize,
            purgeableBytes: Int(clamping: stats.purgeable_count) * pageSize
        )
    }
}

// MARK: - In-memory peer

/// Test peer: the sample is whatever the test scripted, set
/// synchronously on MainActor — exactly how the manager reads the
/// production adapter.
@MainActor
final class InMemoryMemoryHeadroomSource: MemoryHeadroomSource {
    var next: MemoryHeadroomSample?

    init(next: MemoryHeadroomSample? = nil) {
        self.next = next
    }

    func sample() -> MemoryHeadroomSample? {
        next
    }
}

// MARK: - Active-Inference Reserve

/// The named replacement for the retired `/2` divisor (ADR-0018): the
/// bytes withheld from the cache ceiling for in-flight generations'
/// KV working sets. Count-aware — N lanes subtract N reserves; the
/// **Batch Engine**'s Lane Admission (PRD #173, ADR-0022) prices each
/// pool lane against this per-lane figure — and never zero: even an
/// idle server reserves one lane, so the *next* request always has
/// room (SGLang's shared-pool arbitration invariant: active generation
/// always outranks cache, the reserve is subtracted first).
///
/// Per-lane sizing is measured where possible: a lane's KV working set
/// at end of turn *is* its leaf snapshot, and the capture deep-copies
/// it, so the structural peak is twice the largest leaf this cache has
/// admitted. The bootstrap constant covers the window before the first
/// leaf; prefill activation transients beyond the KV bytes remain
/// guarded by fast pressure retreat (ADR-0018's explicit trade).
///
/// Storage posture (ADR-0023): v1 lanes deep-copy their restored
/// prefixes — the duplicate-prefix-bytes meter records the cost — so
/// the capture-copy factor stays. The paged (refcounted) KV tier is
/// the recorded follow-up (kernel gate PASSED 2026-07-05); landing it
/// drops `captureCopyFactor` — a lane's marginal RAM becomes its new
/// decode pages only.
nonisolated struct ActiveInferenceReserve: Sendable, Equatable {
    /// Per-lane bytes before any leaf has been observed: two ~2 GiB
    /// 96k-token leaves' worth (measured 2026-07-04, ornith-35b,
    /// ~21.5 KB/token, kvBits=8).
    static let bootstrapPerLaneBytes = 4 << 30  // 4 GiB

    /// `HybridCacheSnapshot.capture` deep-copies the lane's live KV —
    /// for one moment both copies are resident.
    static let captureCopyFactor = 2

    private(set) var largestObservedLeafBytes = 0

    /// Fold one admitted leaf's size into the per-lane estimate.
    mutating func observeLeaf(bytes: Int) {
        largestObservedLeafBytes = max(largestObservedLeafBytes, bytes)
    }

    var perLaneBytes: Int {
        max(Self.bootstrapPerLaneBytes, Self.captureCopyFactor * largestObservedLeafBytes)
    }

    /// The reserve for `lanes` in-flight requests, floored at one lane.
    func reserveBytes(lanes: Int) -> Int {
        max(lanes, 1) * perLaneBytes
    }
}

// MARK: - Ceiling policy

/// The pure ceiling formula (ADR-0018). The cache's own resident bytes
/// count as claimable — they are *our* footprint, not missing headroom
/// — so the ceiling is what we hold plus what the machine can still
/// give, minus the Active-Inference Reserve. Evictable cache counting
/// as available capacity for admitting work (the SGLang arbitration
/// invariant's other half) falls out of the same accounting: work is
/// never rejected for cache fullness, the ceiling just contracts and
/// the drain demotes (Recoverable Eviction, ADR-0019).
/// Apply a user budget cap that **only ever lowers** the value — caps,
/// never floors (ADR-0018, "Automatic (recommended)" default is `nil`).
/// A negative cap is treated as zero. Shared by both tiers' budget
/// policies and their bootstrap inits so the "cap, never floor" rule
/// lives in one place.
nonisolated func applyBudgetCap(_ value: Int, cap: Int?) -> Int {
    guard let cap else { return value }
    return min(value, max(cap, 0))
}

nonisolated enum DynamicCeilingPolicy {
    /// Fraction of measured headroom the cache may claim. Not a sizing
    /// constant in the old sense — it scales with the machine — but a
    /// damping term: claiming the last free byte would park the OS
    /// permanently at the pressure threshold and turn the band into an
    /// oscillator.
    static let headroomFraction = 0.8

    /// A user cap (Slice D, "Automatic (recommended)" default) only
    /// ever lowers the result — caps, never floors (ADR-0018).
    static func ceilingBytes(
        residentBytes: Int,
        measuredHeadroomBytes: Int,
        reserveBytes: Int,
        capBytes: Int?
    ) -> Int {
        let claimable = Int(Double(max(measuredHeadroomBytes, 0)) * headroomFraction)
        let uncapped = max(0, max(residentBytes, 0) + claimable - max(reserveBytes, 0))
        return applyBudgetCap(uncapped, cap: capBytes)
    }
}
