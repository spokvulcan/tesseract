//
//  TraceReplayHarness.swift
//  tesseract
//
//  Offline trace-replay harness (PRD #82, slice #85): replays the
//  **Completion Trace Log** corpus through a pure simulation of
//  today's RAM-tier baseline — plain LRU under each record's recorded
//  budget — and reports the TTFT-proxy that corpus would pay under it.
//  This is the yardstick every candidate eviction policy must beat
//  before it ships (ADR-0011's ablation gate, slice #91), and the
//  rebuilt AlphaTuner's evaluation harness (slice #92).
//
//  Pure: records in → report out. No clocks, no I/O, no globals;
//  recency is a logical clock (the record index) and every tie-break
//  is fully specified, so one corpus always produces an identical
//  report (the golden test pins it).
//
//  Granularity: prefix sharing is reconstructed from the cumulative
//  block digests (`TraceBlockDigest`), so hits, depths, and leaf
//  supersession all resolve at `blockSize`-token granularity. The sim
//  credits a hit only up to the deepest *digest-verified* block
//  boundary — a conservative under-count of sub-block token sharing —
//  and snapshots shallower than one block are never hittable in
//  replay. The same systematic bias applies to every policy replayed
//  over the corpus, so comparisons stay fair.
//
//  Units: re-prefill seconds are denominated in each record's own
//  `deviceEstimates` (slice #84) — never hardcoded constants; records
//  predating slice #84 fall back to the estimator's cold-start
//  defaults. The trace carries no FLOP profile (only `modelID`), so
//  the FLOP curve itself is `ModelFlopProfile.fallback` for all
//  corpora — a constant scale factor that cancels out of any
//  policy-vs-policy comparison.
//

import Foundation

nonisolated enum TraceReplayHarness {

    // MARK: - Report

    /// What the live system actually did, aggregated straight from the
    /// records' outcome fields — the ground truth the simulation sits
    /// next to.
    struct ObservedMetrics: Codable, Equatable {
        let ttftP50Seconds: Double
        let ttftP95Seconds: Double
        let totalHitTokens: Int
        let terminalEvictions: Int
        let recoveredEvictions: Int
        let ssdRestores: Int
        let totalHydrationSeconds: Double
        /// **Rewind telemetry** roll-up (issue #101): how often a
        /// **Think-Strip Rewind** landed in this corpus and how deep the
        /// re-prefill ran. `rewindEvents` counts records whose restore
        /// floor sat below the divergence; the percentiles are over those
        /// records' rewind sizes. The dashboard surfaces these so a
        /// regression shows up as a rising rewind count or size without
        /// reproducing an incident.
        let rewindEvents: Int
        let rewindSizeP50Tokens: Int
        let rewindSizeP95Tokens: Int
        let maxRewindSizeTokens: Int
    }

    /// What the pure-LRU RAM-only baseline would have paid on the same
    /// workload. Every simulated eviction is terminal by construction
    /// (the baseline has no SSD tier), so there is no recovered count.
    struct SimulatedLRUMetrics: Codable, Equatable {
        /// Re-prefill seconds for each request's unmatched suffix,
        /// denominated in that record's `deviceEstimates`.
        let ttftProxyP50Seconds: Double
        let ttftProxyP95Seconds: Double
        /// Digest-verified hit tokens (block-aligned).
        let totalHitTokens: Int
        let hitRequestCount: Int
        let evictionCount: Int
        let evictedBytes: Int
    }

    struct Report: Codable, Equatable {
        let recordCount: Int
        let blockSize: Int
        let observed: ObservedMetrics
        let simulatedLRU: SimulatedLRUMetrics
    }

    // MARK: - Replay

    static func replay(records: [CompletionTraceRecord]) -> Report {
        var cache = SimulatedCache()
        var proxies: [Double] = []
        var simHitTokens = 0
        var simHitRequests = 0
        var simEvictionCount = 0
        var simEvictedBytes = 0

        for (clock, record) in records.enumerated() {
            let credit = cache.lookup(record: record, clock: clock)
            if credit > 0 { simHitRequests += 1 }
            simHitTokens += credit

            let estimates = record.deviceEstimates ?? MeasuredSecondsEstimates()
            let flops = EvictionPolicy.parentRelativeFlops(
                nodeOffset: record.promptTokenCount,
                parentOffset: credit,
                profile: .fallback
            )
            proxies.append(flops / estimates.prefillFlopsPerSecond)

            for checkpoint in record.admittedCheckpoints {
                cache.admit(checkpoint, record: record, clock: clock)
            }
            if let leaf = record.admittedLeaf {
                cache.admit(leaf, record: record, clock: clock)
            }
            let evicted = cache.evictToFit(budgetBytes: record.ramBudgetBytes)
            simEvictionCount += evicted.count
            simEvictedBytes += evicted.bytes
        }

        let observedTTFTs = records.map(\.ttftSeconds).sorted()
        let sortedProxies = proxies.sorted()
        let rewindSizes = records
            .compactMap(\.rewind)
            .filter(\.isRewind)
            .map(\.rewindSize)
            .sorted()
        return Report(
            recordCount: records.count,
            blockSize: TraceBlockDigest.blockSize,
            observed: ObservedMetrics(
                ttftP50Seconds: percentile(observedTTFTs, 0.50),
                ttftP95Seconds: percentile(observedTTFTs, 0.95),
                totalHitTokens: records.reduce(0) { $0 + $1.hitTokens },
                terminalEvictions: records.reduce(0) { $0 + $1.terminalEvictionCount },
                recoveredEvictions: records.reduce(0) { $0 + $1.recoveredEvictionCount },
                ssdRestores: records.count(where: \.restoredFromSSD),
                totalHydrationSeconds: records.reduce(0) { $0 + $1.hydrationSeconds },
                rewindEvents: rewindSizes.count,
                rewindSizeP50Tokens: intPercentile(rewindSizes, 0.50),
                rewindSizeP95Tokens: intPercentile(rewindSizes, 0.95),
                maxRewindSizeTokens: rewindSizes.last ?? 0
            ),
            simulatedLRU: SimulatedLRUMetrics(
                ttftProxyP50Seconds: percentile(sortedProxies, 0.50),
                ttftProxyP95Seconds: percentile(sortedProxies, 0.95),
                totalHitTokens: simHitTokens,
                hitRequestCount: simHitRequests,
                evictionCount: simEvictionCount,
                evictedBytes: simEvictedBytes
            )
        )
    }

    /// Stable human-readable rendering — the CLI's output and the
    /// golden test's pinned form.
    static func renderText(_ report: Report) -> String {
        let o = report.observed
        let s = report.simulatedLRU
        return """
        Trace Replay — LRU baseline (PRD #82, slice #85)
        records: \(report.recordCount) · block size: \(report.blockSize) tokens
        observed      ttft p50 \(seconds(o.ttftP50Seconds)) · p95 \(seconds(o.ttftP95Seconds)) \
        · hit tokens \(o.totalHitTokens) · evictions \(o.terminalEvictions) terminal \
        / \(o.recoveredEvictions) recovered · ssd restores \(o.ssdRestores) \
        · hydration \(seconds(o.totalHydrationSeconds))
        rewinds       events \(o.rewindEvents) · size p50 \(o.rewindSizeP50Tokens) \
        · p95 \(o.rewindSizeP95Tokens) · max \(o.maxRewindSizeTokens) tokens
        simulated LRU ttft-proxy p50 \(seconds(s.ttftProxyP50Seconds)) \
        · p95 \(seconds(s.ttftProxyP95Seconds)) · hit tokens \(s.totalHitTokens) \
        · hit requests \(s.hitRequestCount)/\(report.recordCount) \
        · evictions \(s.evictionCount) (all terminal — RAM-only baseline) \
        · evicted bytes \(s.evictedBytes)
        """
    }

    private static func seconds(_ value: Double) -> String {
        String(format: "%.4fs", value)
    }

    /// Nearest-rank percentile over an ascending-sorted sample; `0` for
    /// an empty sample.
    private static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard let index = percentileIndex(count: sorted.count, q) else { return 0 }
        return sorted[index]
    }

    /// Nearest-rank percentile over an ascending-sorted integer sample.
    private static func intPercentile(_ sorted: [Int], _ q: Double) -> Int {
        guard let index = percentileIndex(count: sorted.count, q) else { return 0 }
        return sorted[index]
    }

    private static func percentileIndex(count: Int, _ q: Double) -> Int? {
        guard count > 0 else { return nil }
        let rank = Int((q * Double(count)).rounded(.up))
        return max(0, min(count - 1, rank - 1))
    }

    // MARK: - Simulated RAM tier (pure-LRU baseline)

    private struct SimSnapshot {
        let partitionDigest: String
        /// The record's cumulative digests up to this snapshot's
        /// offset — the block-granular identity of its prefix path.
        let digests: [String]
        let offset: Int
        let bytes: Int
        let isLeaf: Bool
        var lastAccess: Int
        /// Admission sequence number: the deterministic LRU tie-break.
        let order: Int
    }

    private struct SimulatedCache {
        private var snapshots: [SimSnapshot] = []
        private var totalBytes = 0
        private var nextOrder = 0

        /// Deepest digest-verified match for the request, refreshing
        /// only the winner's recency (mirroring the live tree, which
        /// touches the restored node, not its ancestors). Returns the
        /// credited hit tokens — block-aligned, `0` on a miss.
        mutating func lookup(record: CompletionTraceRecord, clock: Int) -> Int {
            var winner: Int?
            for index in snapshots.indices {
                let candidate = snapshots[index]
                guard candidate.partitionDigest == record.partitionDigest,
                      !candidate.digests.isEmpty,
                      record.prefixBlockDigests.starts(with: candidate.digests)
                else { continue }
                if let current = winner {
                    let best = snapshots[current]
                    if (candidate.digests.count, candidate.offset, candidate.order)
                        > (best.digests.count, best.offset, best.order) {
                        winner = index
                    }
                } else {
                    winner = index
                }
            }
            guard let winner else { return 0 }
            snapshots[winner].lastAccess = clock
            return snapshots[winner].digests.count * TraceBlockDigest.blockSize
        }

        /// Insert one admitted snapshot. A leaf supersedes stored
        /// leaves on its own (block-granular) path at shallower
        /// offsets, as production leaf supersession does; an exact
        /// (partition, path, offset, kind) duplicate is replaced
        /// in place.
        mutating func admit(
            _ admitted: TraceAdmittedSnapshot,
            record: CompletionTraceRecord,
            clock: Int
        ) {
            let digests = Array(
                record.prefixBlockDigests.prefix(admitted.offset / TraceBlockDigest.blockSize)
            )
            let isLeaf = admitted.checkpointType
                == HybridCacheSnapshot.CheckpointType.leaf.wireString
            remove { existing in
                existing.partitionDigest == record.partitionDigest
                    && existing.digests == digests
                    && existing.offset == admitted.offset
                    && existing.isLeaf == isLeaf
            }
            if isLeaf {
                remove { existing in
                    existing.partitionDigest == record.partitionDigest
                        && existing.isLeaf
                        && existing.offset < admitted.offset
                        && digests.starts(with: existing.digests)
                }
            }
            snapshots.append(SimSnapshot(
                partitionDigest: record.partitionDigest,
                digests: digests,
                offset: admitted.offset,
                bytes: admitted.bytes,
                isLeaf: isLeaf,
                lastAccess: clock,
                order: nextOrder
            ))
            totalBytes += admitted.bytes
            nextOrder += 1
        }

        /// Plain LRU: evict stalest-first until the recorded budget
        /// holds. A non-positive budget drains everything, matching
        /// the live manager's flush semantics.
        mutating func evictToFit(budgetBytes: Int) -> (count: Int, bytes: Int) {
            var count = 0
            var bytes = 0
            while totalBytes > max(budgetBytes, 0), !snapshots.isEmpty {
                let victim = snapshots.indices.min { a, b in
                    (snapshots[a].lastAccess, snapshots[a].order)
                        < (snapshots[b].lastAccess, snapshots[b].order)
                }!
                bytes += snapshots[victim].bytes
                count += 1
                totalBytes -= snapshots[victim].bytes
                snapshots.remove(at: victim)
            }
            return (count, bytes)
        }

        private mutating func remove(where shouldRemove: (SimSnapshot) -> Bool) {
            snapshots.removeAll { snapshot in
                guard shouldRemove(snapshot) else { return false }
                totalBytes -= snapshot.bytes
                return true
            }
        }
    }
}
