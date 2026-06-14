//
//  PressureReactiveBudget.swift
//  tesseract
//
//  The **Pressure-Reactive Budget** (ADR-0011, PRD #82 slice #88): the
//  RAM-tier byte budget as a band, not a constant. The load-time
//  auto-sized budget is the *ceiling*; OS memory-pressure events push a
//  *current* value down toward the content-defined **Budget Floor**
//  (shrinks demote via Snapshot Demotion), and hysteresis regrows it —
//  fast down, slow up — when pressure clears. The cache is greedy when
//  RAM is idle, polite when it is contested.
//
//  This file is the whole seam: the Memory Pressure Source port, the
//  thin production adapter over the OS dispatch source, the in-memory
//  peer tests drive, and the pure band fold.
//

import Foundation

// MARK: - Pressure level

/// The three OS memory-pressure states, decoupled from the Dispatch
/// event mask so the band fold and its tests never import Dispatch
/// semantics.
nonisolated enum MemoryPressureLevel: String, Sendable {
    case normal
    case warning
    case critical
}

// MARK: - Port

/// The port `PrefixCacheManager` subscribes to for memory-pressure
/// events. Same `@MainActor`-sibling shape as the speech ports
/// (ADR-0003/0004): class-bound, main-actor-isolated, handler invoked
/// on MainActor. The production adapter is
/// `DispatchMemoryPressureSource`; the test peer is
/// `InMemoryMemoryPressureSource`.
@MainActor
protocol MemoryPressureSource: AnyObject {
    /// Begin delivering events to `handler`. Implementations are
    /// single-subscriber; repeat calls are ignored. Delivery stops when
    /// the source is deallocated — the manager holds its source
    /// strongly and the handler captures the manager weakly, so a cache
    /// teardown (model unload) tears the subscription down with it.
    func start(handler: @escaping @MainActor (MemoryPressureLevel) -> Void)
}

// MARK: - Production adapter

/// Thin adapter over `DispatchSource.makeMemoryPressureSource`. All it
/// does is map the event mask to a `MemoryPressureLevel` on the main
/// queue — every decision lives in the band fold and the manager.
@MainActor
final class DispatchMemoryPressureSource: MemoryPressureSource {
    private var source: (any DispatchSourceMemoryPressure)?

    func start(handler: @escaping @MainActor (MemoryPressureLevel) -> Void) {
        guard source == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: .all,
            queue: .main
        )
        source.setEventHandler { [weak source] in
            guard let source, !source.isCancelled else { return }
            let event = source.data
            let level: MemoryPressureLevel =
                event.contains(.critical)
                ? .critical
                : event.contains(.warning)
                    ? .warning
                    : .normal
            // The handler runs on the main queue by construction.
            MainActor.assumeIsolated { handler(level) }
        }
        source.activate()
        self.source = source
    }

    deinit {
        source?.cancel()
    }
}

// MARK: - In-memory peer

/// Test peer: events are injected by the test, synchronously, on
/// MainActor — which is exactly how the production adapter delivers
/// them.
@MainActor
final class InMemoryMemoryPressureSource: MemoryPressureSource {
    private var handler: (@MainActor (MemoryPressureLevel) -> Void)?

    func start(handler: @escaping @MainActor (MemoryPressureLevel) -> Void) {
        guard self.handler == nil else { return }
        self.handler = handler
    }

    func send(_ level: MemoryPressureLevel) {
        handler?(level)
    }
}

// MARK: - Band fold

/// The band itself: an immutable ceiling and a current value the fold
/// moves between the (caller-supplied, content-defined) floor and the
/// ceiling. Pure value — `folding` returns the next band, so every
/// scripted-event test crosses the same global-free seam.
///
/// Hysteresis is fast-down / slow-up: `warning` halves the current
/// value, `critical` drops it straight to the floor, and each `normal`
/// event regrows by one `regrowthDenominator`-th of the ceiling. The
/// asymmetry is what damps flapping: a warning/normal flap loses
/// `current/2` and regains only `ceiling/8` per cycle, so the band
/// converges instead of oscillating, and a machine whose pressure has
/// genuinely cleared regrows to the ceiling in a handful of events.
nonisolated struct PrefixCacheBudgetBand: Sendable, Equatable {
    /// One regrowth step is `ceilingBytes / regrowthDenominator`.
    static let regrowthDenominator = 8

    /// The load-time auto-sized budget. Never exceeded.
    let ceilingBytes: Int

    /// The live RAM-tier budget. Starts at the ceiling — a machine that
    /// never signals pressure keeps today's behavior unchanged.
    private(set) var currentBytes: Int

    init(ceilingBytes: Int) {
        self.ceilingBytes = max(ceilingBytes, 0)
        self.currentBytes = self.ceilingBytes
    }

    /// Fold one pressure event. `floorBytes` is the **Budget Floor**
    /// computed by the caller at event time (content-defined — the
    /// `.system` chains plus the most-recently-extended leaf), clamped
    /// into `[0, ceiling]` here so a floor that momentarily exceeds the
    /// ceiling cannot invert the band.
    func folding(
        _ level: MemoryPressureLevel,
        floorBytes: Int
    ) -> PrefixCacheBudgetBand {
        let floor = min(max(floorBytes, 0), ceilingBytes)
        var next = self
        switch level {
        case .warning:
            next.currentBytes = max(floor, currentBytes / 2)
        case .critical:
            next.currentBytes = floor
        case .normal:
            let step = max(ceilingBytes / Self.regrowthDenominator, 1)
            next.currentBytes = min(ceilingBytes, currentBytes + step)
        }
        return next
    }
}
