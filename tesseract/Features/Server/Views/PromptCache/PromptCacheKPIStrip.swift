import SwiftUI

struct PromptCacheKPIStrip: View {
    let snapshot: PromptCacheTelemetrySnapshot?
    let aggregate: PromptCacheTelemetryAggregate

    var body: some View {
        LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 118), spacing: Theme.Spacing.xs)],
            spacing: Theme.Spacing.xs
        ) {
            ForEach(metrics) { metric in
                KPIBlock(metric: metric)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var memoryRatio: Double {
        guard let snapshot, snapshot.memoryBudgetBytes > 0 else { return 0 }
        return Double(snapshot.residentSnapshotBytes) / Double(snapshot.memoryBudgetBytes)
    }

    /// RAM tile detail: utilization, plus the **Pressure-Reactive
    /// Budget** ceiling whenever pressure has squeezed the current
    /// budget below it — the band is invisible while the cache is
    /// unpressured.
    private var ramDetail: String {
        let percent = PromptCacheFormatting.percent(memoryRatio)
        guard let snapshot,
            snapshot.budgetCeilingBytes > 0,
            snapshot.memoryBudgetBytes < snapshot.budgetCeilingBytes
        else { return percent }
        return "\(percent) · ceiling \(PromptCacheFormatting.bytes(snapshot.budgetCeilingBytes))"
    }

    private var ssdDetail: String {
        guard let snapshot, snapshot.ssd.enabled else { return "disabled" }
        if snapshot.ssd.pendingCount > 0 {
            return "\(snapshot.ssd.pendingCount) pending"
        }
        return PromptCacheFormatting.percent(
            snapshot.ssd.budgetBytes > 0
                ? Double(snapshot.ssd.currentBytes) / Double(snapshot.ssd.budgetBytes)
                : 0)
    }

    private var restoreMix: String {
        let leaf = aggregate.restoredCheckpointCounts["leaf", default: 0]
        let system = aggregate.restoredCheckpointCounts["system", default: 0]
        let branch = aggregate.restoredCheckpointCounts["branchPoint", default: 0]
        return "\(leaf)/\(system)/\(branch)"
    }

    private func memoryValue(used: Int, budget: Int) -> String {
        guard budget > 0 else { return PromptCacheFormatting.bytes(used) }
        return "\(PromptCacheFormatting.bytes(used)) / \(PromptCacheFormatting.bytes(budget))"
    }

    private var metrics: [PromptCacheMetricTile] {
        [
            PromptCacheMetricTile(
                title: "RAM",
                value: memoryValue(
                    used: snapshot?.residentSnapshotBytes ?? 0,
                    budget: snapshot?.memoryBudgetBytes ?? 0
                ),
                detail: ramDetail,
                symbol: "memorychip",
                tint: memoryRatio > 0.9 ? .red : .blue
            ),
            PromptCacheMetricTile(
                title: "SSD",
                value: memoryValue(
                    used: snapshot?.ssd.currentBytes ?? 0,
                    budget: snapshot?.ssd.budgetBytes ?? 0
                ),
                detail: ssdDetail,
                symbol: "internaldrive",
                tint: .teal
            ),
            PromptCacheMetricTile(
                title: "Hit Rate",
                value: PromptCacheFormatting.percent(aggregate.hitRate),
                detail:
                    "\(aggregate.hitCount + aggregate.ssdHitCount)/\(aggregate.lookupCount) lookups",
                symbol: "checkmark.circle",
                tint: .green
            ),
            PromptCacheMetricTile(
                title: "Token Reuse",
                value: PromptCacheFormatting.percent(aggregate.tokenReuseRate),
                detail: "\(PromptCacheFormatting.compactNumber(aggregate.cachedTokens)) cached",
                symbol: "arrow.triangle.2.circlepath",
                tint: .mint
            ),
            PromptCacheMetricTile(
                title: "Snapshots",
                value: "\(snapshot?.snapshotCount ?? 0)",
                detail: "\(snapshot?.partitionCount ?? 0) partitions",
                symbol: "tray.2",
                tint: .indigo
            ),
            PromptCacheMetricTile(
                title: "Restore Mix",
                value: restoreMix,
                detail: "leaf/system/branch",
                symbol: "point.3.connected.trianglepath.dotted",
                tint: .purple
            ),
            PromptCacheMetricTile(
                title: "Lookup",
                value: PromptCacheFormatting.milliseconds(aggregate.averageLookupMs),
                detail: "restore \(PromptCacheFormatting.milliseconds(aggregate.averageRestoreMs))",
                symbol: "magnifyingglass",
                tint: .orange
            ),
            PromptCacheMetricTile(
                title: "Prefill",
                value: PromptCacheFormatting.milliseconds(aggregate.averagePrefillMs),
                detail: "TTFT \(PromptCacheFormatting.milliseconds(aggregate.averageTTFTMs))",
                symbol: "timer",
                tint: .cyan
            ),
            PromptCacheMetricTile(
                title: "Eviction α",
                value: tunerValue,
                detail: tunerDetail,
                symbol: "dial.medium",
                tint: .pink
            ),
            PromptCacheMetricTile(
                title: "Cache Wins",
                value: PromptCacheFormatting.compactNumber(
                    snapshot?.counters.hitTokens ?? 0
                ),
                detail: "\(snapshot?.counters.hydrations ?? 0) hydrations",
                symbol: "bolt.badge.checkmark",
                tint: .green
            ),
            PromptCacheMetricTile(
                title: "Losses",
                value: "\(snapshot?.counters.recoveredEvictions ?? 0)"
                    + "/\(snapshot?.counters.terminalEvictions ?? 0)",
                detail: "recovered/terminal",
                symbol: "arrow.down.to.line.compact",
                tint: .red
            ),
            PromptCacheMetricTile(
                title: "Rewinds",
                value: "\(aggregate.rewindEventCount)",
                detail: rewindDetail,
                symbol: "arrow.uturn.backward.circle",
                tint: aggregate.rewindEventCount > 0 ? .orange : .secondary
            ),
            PromptCacheMetricTile(
                title: "Device",
                value: deviceEstimatesValue,
                detail: deviceEstimatesDetail,
                symbol: "gauge.with.needle",
                tint: .brown
            ),
        ]
    }

    /// Rewind tile detail: the re-prefill the **Chain-Prefix Restore**
    /// floor saved versus rewinding to the strip floor (issue #101).
    private var rewindDetail: String {
        guard aggregate.rewindEventCount > 0 else { return "none" }
        return "\(PromptCacheFormatting.compactNumber(aggregate.rewindTokens)) tokens floored"
    }

    private var deviceEstimatesValue: String {
        guard let estimates = snapshot?.estimates else { return "—" }
        return String(
            format: "%.1f TFLOP/s", estimates.prefillFlopsPerSecond / 1e12
        )
    }

    private var deviceEstimatesDetail: String {
        guard let estimates = snapshot?.estimates else { return "no data" }
        let bandwidth = String(
            format: "SSD %.1f GB/s", estimates.hydrationBytesPerSecond / 1e9
        )
        guard estimates.prefillSampleCount > 0 || estimates.hydrationSampleCount > 0 else {
            return bandwidth + " · defaults"
        }
        return bandwidth
    }

    private var tunerValue: String {
        guard let tuner = snapshot?.tuner, tuner.phase != "unavailable" else { return "—" }
        return String(format: "%.2f", tuner.alpha)
    }

    private var tunerDetail: String {
        guard let tuner = snapshot?.tuner else { return "no data" }
        switch tuner.phase {
        case "waitingForFirstEviction": return "LRU · awaiting eviction"
        case "bootstrapping": return "tuning \(tuner.bootstrapProgress)/\(tuner.bootstrapTarget)"
        case "tuned": return "tuned"
        default: return "no data"
        }
    }
}

private struct PromptCacheMetricTile: Identifiable {
    let title: String
    let value: String
    let detail: String
    let symbol: String
    let tint: Color

    var id: String { title }
}

private struct KPIBlock: View {
    let metric: PromptCacheMetricTile

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Image(systemName: metric.symbol)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(metric.tint)
                    .frame(width: 14)

                Text(metric.title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
            }
            Text(metric.value)
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.68)
            Text(metric.detail)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .padding(Theme.Spacing.sm)
        .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        .glassEffect(
            .regular, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }
}
