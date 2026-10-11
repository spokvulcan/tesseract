//
//  TurnReport.swift
//  tesseract
//
//  The **Turn Report** (ADR-0069's 2026-10-11 amendment): one request's
//  account of its cache, kept by its Cache Claim, and the one place the
//  turn's outputs are made from. Plan application files the turn's **Turn
//  Restore** here once, and the `leafStore` event, its leaf's source for the
//  Active-Inference Reserve, the `recordingRequest` phase and the trace
//  record all read it from here. The report carries the trace's eviction
//  tally (the Completion Trace Accumulator it grew from) and owns the turn's
//  one `leafStore` event: the Leaf Store phase's report once the phase has
//  begun, or, for a turn that ends before the phase, the claim's Leaf
//  Rewind, which the claim reports here instead of logging a report of its
//  own.
//

import Foundation

nonisolated final class TurnReport: @unchecked Sendable {
    private let diagnostics: PrefixCacheDiagnostics.Context
    private let lock = NSLock()
    private var filedRestore = TurnRestore.cold
    private var phaseBegan = false
    private var tally = CompletionTraceAccumulator()

    init(diagnostics: PrefixCacheDiagnostics.Context) {
        self.diagnostics = diagnostics
    }

    // MARK: - The Turn Restore

    /// How the turn's cache opened: `.cold` until plan application files
    /// it, and for a turn that planned no restore or was not keyed.
    var restore: TurnRestore { lock.withLock { filedRestore } }

    /// Plan application files the turn's restore once its check-out and
    /// restore have settled.
    func file(_ restore: TurnRestore) {
        lock.withLock { filedRestore = restore }
    }

    // MARK: - The one `leafStore` event

    /// The Leaf Store phase begins: from here its report is the turn's one
    /// `leafStore` event, a rewind in the phase or at the claim's conclusion
    /// included.
    func beginLeafStorePhase() {
        lock.withLock { phaseBegan = true }
    }

    /// The claim rewound the leaf its `handoff` leased, back to `offset`.
    /// Before the phase that is all that became of the turn's leaf, so the
    /// rewind is the turn's one `leafStore` event (`path=rewind
    /// restoreMode=handoff`, with the wait the handoff cost); from the phase
    /// on, the phase's report says it.
    func leafRewound(_ handoff: TurnRestore.Handoff, offset: Int) {
        guard !lock.withLock({ phaseBegan }) else { return }
        var report = LeafStorePhase.Report()
        report.mode = "keyed"
        report.path = .rewind
        report.restore = .handoff(handoff)
        report.leafOffset = offset
        diagnostics.log(report, level: .notice)
    }

    /// The phase's report with the turn's restore filed in, which its leaf's
    /// source and copy reason read.
    func filed(_ report: LeafStorePhase.Report) -> LeafStorePhase.Report {
        var report = report
        report.restore = restore
        return report
    }

    /// The phase's report as the turn's one `leafStore` event.
    func logLeafStore(_ report: LeafStorePhase.Report) {
        diagnostics.log(report, level: .notice)
    }

    // MARK: - The trace tally

    /// Classify one batch of the turn's evictions into the trace tally and
    /// log their correlated lines (`CompletionTraceAccumulator.ingest`).
    func ingest(evictions: [PrefixCacheManager.EvictionEvent]) {
        lock.withLock { tally.ingest(evictions: evictions, diagnostics: diagnostics) }
    }

    /// Add a Leaf Admission's tally, whose lines it already logged.
    func merge(_ admission: CompletionTraceAccumulator) {
        lock.withLock { tally.merge(admission) }
    }

    /// Log one admission's superseded leaves.
    func logSupersessions(_ supersededLeaves: [PrefixCacheManager.LeafSupersession]) {
        lock.withLock { tally.logSupersessions(supersededLeaves, diagnostics: diagnostics) }
    }

    /// The tally so far, which makes the turn's trace record
    /// (`CompletionTraceAccumulator.makeRecord`) with the filed restore.
    var trace: CompletionTraceAccumulator { lock.withLock { tally } }
}
