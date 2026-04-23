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

    private var ssdDetail: String {
        guard let snapshot, snapshot.ssd.enabled else { return "disabled" }
        if snapshot.ssd.pendingCount > 0 {
            return "\(snapshot.ssd.pendingCount) pending"
        }
        return PromptCacheFormatting.percent(snapshot.ssd.budgetBytes > 0
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
                detail: PromptCacheFormatting.percent(memoryRatio),
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
                detail: "\(aggregate.hitCount + aggregate.ssdHitCount)/\(aggregate.lookupCount) lookups",
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
        ]
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
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }
}
