import SwiftUI

/// The cache panel's vitals strip (PRD #150, v1 read-only), sitting
/// between the hero band and the tree canvas in the same plain-content
/// grammar: a humanized per-request outcome line (the `lookup` + `ttft`
/// pair), a sparing non-alarming notable-events line ("Cache for X was
/// reset — model files changed"), and a collapsible write-pressure
/// chart fed by the persistent endurance ledger.
struct PromptCacheVitalsStrip: View {
    let outcome: (lookup: PromptCacheTelemetryEvent, ttft: PromptCacheTelemetryEvent?)?
    let endurance: SSDEnduranceSnapshot
    let ssdEnabled: Bool

    @AppStorage("server.promptCache.writePressure.open") private var isChartOpen = false
    @AppStorage("server.promptCache.writePressure.daily") private var showDaily = false

    var body: some View {
        if outcome != nil || latestNotable != nil || hasWriteData {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                if let outcome {
                    HStack(spacing: Theme.Spacing.md) {
                        Text(outcomeLine(outcome))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .textSelection(.enabled)

                        Spacer(minLength: 0)

                        if hasWriteData {
                            chartToggle
                        }
                    }
                } else if hasWriteData {
                    HStack {
                        Spacer(minLength: 0)
                        chartToggle
                    }
                }

                if let notable = latestNotable {
                    Label(notableLine(notable), systemImage: "info.circle")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .textSelection(.enabled)
                }

                if isChartOpen && hasWriteData {
                    writePressureChart
                }
            }
            .font(.caption.monospaced())
            .monospacedDigit()
            .padding(.horizontal, Theme.Spacing.xl)
            .padding(.bottom, Theme.Spacing.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Per-request outcome line

    private func outcomeLine(
        _ outcome: (lookup: PromptCacheTelemetryEvent, ttft: PromptCacheTelemetryEvent?)
    ) -> String {
        let lookup = outcome.lookup
        let reason = lookup.field("reason") ?? ""
        let hydrated = lookup.field("hydratedFromSSD") == "true"
        let isHit = reason == "hit" || reason == "ssdHit" || reason == "chainPrefixHit"

        var parts = ["last request"]
        if isHit {
            let tier =
                (hydrated || reason == "ssdHit" || reason == "chainPrefixHit") ? "ssd" : "ram"
            var hit = "\(tier) hit"
            if let offset = lookup.intField("snapshotOffset") {
                hit += " @ \(offset.formatted())"
                if let prompt = lookup.intField("promptTokens"), prompt > 0 {
                    hit += " / \(prompt.formatted()) tok"
                } else {
                    hit += " tok"
                }
            }
            parts.append(hit)
        } else {
            var miss = "miss"
            if let prompt = lookup.intField("promptTokens"), prompt > 0 {
                miss += " · \(prompt.formatted()) tok prompt"
            }
            parts.append(miss)
        }
        if let ttftMs = outcome.ttft?.doubleField("ttftMs"), ttftMs > 0 {
            parts.append("ttft " + PromptCacheFormatting.milliseconds(ttftMs))
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Notable events

    private var latestNotable: SSDEnduranceSnapshot.NotableEvent? {
        endurance.notable.last
    }

    /// Deliberately non-alarming copy (PRD #150): the 2026-07-04
    /// incident was a *correct* invalidation whose only defect was
    /// silence, so the line explains rather than warns.
    private func notableLine(_ event: SSDEnduranceSnapshot.NotableEvent) -> String {
        let when = Self.relativeFormatter.localizedString(
            for: event.at, relativeTo: Date()
        )
        let size = event.bytes > 0 ? " (\(PromptCacheFormatting.bytes(event.bytes)))" : ""
        switch event.kind {
        case "fingerprintChanged":
            return "Cache for \(event.modelID) was reset\(size) — model files changed · \(when)"
        case "staleUnused":
            return "Reclaimed unused cache for \(event.modelID)\(size) · \(when)"
        case "schemaStale":
            return "Cache for \(event.modelID) was reset\(size) — storage format upgraded · \(when)"
        case "clientPrefixChange":
            let detail = event.detail.map { " \($0)" } ?? ""
            return "Client changed its prompt prefix\(detail) — not a cache fault · \(when)"
        default:
            return "Cache for \(event.modelID) was reset\(size) · \(when)"
        }
    }

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter
    }()

    // MARK: - Write-pressure chart

    private var hasWriteData: Bool {
        ssdEnabled && (endurance.lifetimeBytesWritten > 0 || endurance.lifetimeBytesDeleted > 0)
    }

    private var buckets: [SSDEnduranceSnapshot.DatedBucket] {
        showDaily ? endurance.daily : endurance.hourly
    }

    private var chartToggle: some View {
        Button {
            isChartOpen.toggle()
        } label: {
            Label(
                isChartOpen ? "hide writes" : "ssd writes",
                systemImage: "chart.bar"
            )
            .foregroundStyle(.tertiary)
        }
        .buttonStyle(.plain)
        .help(isChartOpen ? "Hide the SSD write-pressure chart" : "Show SSD write pressure")
    }

    private var writePressureChart: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: Theme.Spacing.md) {
                Picker("", selection: $showDaily) {
                    Text("per hour").tag(false)
                    Text("per day").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.mini)
                .frame(width: 160)

                Spacer(minLength: 0)

                Text(peakLine)
                    .foregroundStyle(.tertiary)
            }

            WritePressureBars(buckets: buckets)
                .frame(height: 56)
                .frame(maxWidth: .infinity)

            Text(lifetimeLine)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
        }
        .padding(.top, Theme.Spacing.xs)
    }

    private var peakLine: String {
        let window = showDaily ? "last \(buckets.count)d" : "last \(buckets.count)h"
        guard let peak = buckets.map(\.bytesWritten).max(), peak > 0 else {
            return window
        }
        let unit = showDaily ? "/d" : "/h"
        return window + " · peak " + PromptCacheFormatting.bytes(peak) + unit
    }

    private var lifetimeLine: String {
        var parts = [
            "lifetime " + PromptCacheFormatting.bytes(endurance.lifetimeBytesWritten)
                + " written"
        ]
        let byClass = endurance.lifetimeBytesWrittenByClass
            .sorted { $0.value > $1.value }
            .map { "\($0.key) \(PromptCacheFormatting.bytes($0.value))" }
        if !byClass.isEmpty {
            parts.append("(" + byClass.joined(separator: " · ") + ")")
        }
        parts.append(
            PromptCacheFormatting.bytes(endurance.lifetimeBytesDeleted) + " deleted")
        parts.append(
            "since " + endurance.since.formatted(date: .abbreviated, time: .omitted))
        return parts.joined(separator: " · ")
    }
}

/// Canvas bar chart of the endurance buckets — written bytes as full
/// bars, deleted bytes as a dimmer overlay. Same Canvas grammar as
/// `RateSparkline`; no chart framework in the content layer.
private struct WritePressureBars: View {
    let buckets: [SSDEnduranceSnapshot.DatedBucket]

    var body: some View {
        Canvas { context, size in
            guard !buckets.isEmpty else { return }
            let peak = max(
                buckets.map { max($0.bytesWritten, $0.bytesDeleted) }.max() ?? 1, 1
            )
            let slot = size.width / CGFloat(buckets.count)
            let barWidth = max(slot * 0.7, 1)

            for (index, bucket) in buckets.enumerated() {
                let x = CGFloat(index) * slot + (slot - barWidth) / 2
                if bucket.bytesWritten > 0 {
                    let height = max(
                        CGFloat(Double(bucket.bytesWritten) / Double(peak)) * size.height, 1
                    )
                    context.fill(
                        Path(
                            roundedRect: CGRect(
                                x: x, y: size.height - height,
                                width: barWidth, height: height
                            ),
                            cornerRadius: 1
                        ),
                        with: .color(.blue.opacity(0.8))
                    )
                }
                if bucket.bytesDeleted > 0 {
                    let height = max(
                        CGFloat(Double(bucket.bytesDeleted) / Double(peak)) * size.height, 1
                    )
                    context.fill(
                        Path(
                            roundedRect: CGRect(
                                x: x, y: size.height - height,
                                width: barWidth, height: height
                            ),
                            cornerRadius: 1
                        ),
                        with: .color(.gray.opacity(0.35))
                    )
                }
            }
        }
        .accessibilityLabel("SSD bytes written per bucket")
    }
}
