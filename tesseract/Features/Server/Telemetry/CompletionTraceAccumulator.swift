//
//  CompletionTraceAccumulator.swift
//  tesseract
//
//  Folds one cache-aware **Server Completion**'s trace facts into the
//  `CompletionTraceRecord` appended to the **Completion Trace Log** — the
//  one home for the derivation rules previously smeared through the
//  drive: the terminal-vs-recovered eviction tally (paired with its
//  correlated diagnostics events, so the tally and the log lines cannot
//  drift apart), the restored-offset rule, and the admitted-snapshot
//  projections. The drive feeds facts; this value decides what the
//  record contains.
//
//  Every structured-leaf admission (`admitStructuredLeaf`) also routes its
//  eviction/supersession events through here: the Leaf Store phase tallies
//  them into the per-request record; the speculative pass and the
//  salvage-on-cancel path use a throwaway accumulator for the correlated
//  logging alone.
//

import Foundation

nonisolated struct CompletionTraceAccumulator {

    /// The start-time cache facts a record derives from — a plain value
    /// so derivation is testable without building a live generation
    /// start. Field names mirror the generation start they are copied
    /// from (`lookupSeconds` et al. hold seconds).
    struct StartFacts {
        let partitionDigest: String
        let unkeyedReason: CacheKeySpace.UnkeyedReason?
        let keyPath: [Int]
        let lookupReason: PrefixCacheManager.LookupReason
        let restoredFromSSD: Bool
        let hitTokens: Int
        let sharedPrefixLength: Int
        let lookupSeconds: Double
        let restoreSeconds: Double
        let hydrationSeconds: Double
        let prefillSeconds: Double
    }

    private(set) var terminalEvictionCount = 0
    private(set) var recoveredEvictionCount = 0

    /// The cache state the generation began from: the hit offset, or 0
    /// for a miss. The one home for the rule (previously derived inline
    /// at the record-build site).
    static func restoredOffset(for lookupReason: PrefixCacheManager.LookupReason) -> Int {
        if case .hit(let offset, _, _) = lookupReason {
            return offset
        }
        return 0
    }

    /// Classify one batch of eviction events into the trace tallies and
    /// emit their correlated diagnostics events (the eviction line plus
    /// the `ssdBodyDrop` correlation for a body-drop that left a live
    /// Snapshot Ref) — one call per admit site, so the tallies and the
    /// emitted lines move together.
    mutating func ingest(
        evictions: [PrefixCacheManager.EvictionEvent],
        diagnostics: PrefixCacheDiagnostics.Context
    ) {
        for event in evictions {
            if event.isTerminal {
                terminalEvictionCount += 1
            } else {
                recoveredEvictionCount += 1
            }
            diagnostics.log(PrefixCacheDiagnostics.EvictionEvent(event))
            if let id = event.bodyDroppedSnapshotRefID {
                diagnostics.log(PrefixCacheDiagnostics.SSDBodyDropEvent(id: id))
            }
        }
    }

    /// Emit the supersession diagnostics for one admit's superseded
    /// leaves. Supersessions carry no tally — they are lifecycle
    /// events, not losses — but they belong to the same one-home rule
    /// set as the eviction emission.
    func logSupersessions(
        _ supersededLeaves: [PrefixCacheManager.LeafSupersession],
        diagnostics: PrefixCacheDiagnostics.Context
    ) {
        for supersession in supersededLeaves {
            diagnostics.log(
                PrefixCacheDiagnostics.LeafSupersessionEvent(
                    offset: supersession.offset,
                    snapshotRefID: supersession.bodyDroppedSnapshotRefID,
                    mode: supersession.mode
                ))
        }
    }

    /// Derive the terminal trace record. `nil` for an **Unkeyed
    /// Completion** (never touched the radix tree — no signal for
    /// policy replay), mirroring `CompletionTraceRecord.make`.
    // swiftlint:disable:next function_parameter_count
    func makeRecord(
        timestamp: Double,
        requestID: UUID,
        modelID: String,
        start: StartFacts,
        capturedSnapshots: [HybridCacheSnapshot],
        leafStore: AlphaTuner.LeafStore?,
        ramBudgetBytes: Int,
        residualPromptSeconds: Double,
        deviceEstimates: MeasuredSecondsEstimates?
    ) -> CompletionTraceRecord? {
        let restoredOffset = Self.restoredOffset(for: start.lookupReason)
        return CompletionTraceRecord.make(
            timestamp: timestamp,
            requestID: requestID,
            modelID: modelID,
            partitionDigest: start.partitionDigest,
            unkeyedReason: start.unkeyedReason,
            keyPath: start.keyPath,
            admittedCheckpoints: capturedSnapshots.map { snap in
                TraceAdmittedSnapshot(
                    offset: snap.tokenOffset,
                    bytes: snap.memoryBytes,
                    checkpointType: snap.checkpointType.wireString
                )
            },
            admittedLeaf: leafStore.map { leaf in
                TraceAdmittedSnapshot(
                    offset: leaf.storedTokens.count,
                    bytes: leaf.bytes,
                    checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString
                )
            },
            ramBudgetBytes: ramBudgetBytes,
            restoredOffset: restoredOffset,
            restoredFromSSD: start.restoredFromSSD,
            hitTokens: start.hitTokens,
            sharedPrefixLength: start.sharedPrefixLength,
            lookupSeconds: start.lookupSeconds,
            restoreSeconds: start.restoreSeconds,
            hydrationSeconds: start.hydrationSeconds,
            prefillSeconds: start.prefillSeconds,
            residualPromptSeconds: residualPromptSeconds,
            terminalEvictionCount: terminalEvictionCount,
            recoveredEvictionCount: recoveredEvictionCount,
            deviceEstimates: deviceEstimates,
            rewind: RewindTelemetry.make(
                sharedPrefixLength: start.sharedPrefixLength,
                restoredOffset: restoredOffset
            )
        )
    }
}
