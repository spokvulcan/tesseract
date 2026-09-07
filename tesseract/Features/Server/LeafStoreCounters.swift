//
//  LeafStoreCounters.swift
//  tesseract
//
//  Lifetime tally of the turns the **Leaf Store** sent down the boundary
//  path, by reason. Logged (notice
//  level, so it survives in `log show`) and reset at model unload, beside
//  the Emitted Path Index summary.
//

import Foundation

/// `@unchecked Sendable`: the tally is NSLock-guarded.
nonisolated final class LeafStoreCounters: @unchecked Sendable {

    nonisolated static let shared = LeafStoreCounters()

    private let lock = NSLock()
    private var boundaryTurns: [String: Int] = [:]

    func noteBoundaryTurn(reason: String) {
        lock.withLock { boundaryTurns[reason, default: 0] += 1 }
    }

    func reset() {
        lock.withLock { boundaryTurns.removeAll() }
    }

    /// The tally by boundary reason (the `liveLeafCapture` skip token).
    func logSummary(context: String) {
        let counts = lock.withLock { boundaryTurns }
        Log.server.notice(
            "leaf-store [\(context)] boundaryTurns=\(counts.values.reduce(0, +))"
                + " byReason=\(PrefixCacheDiagnostics.histogram(counts))")
    }
}
