//
//  CacheOverviewView.swift
//  tesseract
//
//  The Cache page's Overview mode: "what is the cache buying me" — the
//  saved-prefill headline, stat tiles off the cumulative counters + live
//  snapshot, and three Swift Charts on durable sources only
//  (completion-trace corpus + endurance ledger). Chart discipline per the
//  design language §5: one axis per chart, fixed categorical hue order
//  (ChartPalette), thin marks, legends present, text in text tokens.
//

import Charts
import SwiftUI
import Textual

struct CacheOverviewView: View {
    let snapshot: PromptCacheTelemetrySnapshot?
    let endurance: SSDEnduranceSnapshot
    let corpus: CacheCorpusStore
    let window: CacheWindow

    private var windowPoints: [CacheCorpusStore.Point] {
        corpus.points(in: window)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xxl) {
                headline

                statTiles

                TTFTBreakdownChart(points: windowPoints, window: window)
                HitRateTrendChart(points: windowPoints, window: window)
                SSDWriteChart(endurance: endurance, window: window)
            }
            .padding(Theme.Spacing.xl)
            .frame(maxWidth: 920, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(.background)
    }

    // MARK: - Headline

    /// The one hero figure: prefill wall-clock the cache absorbed — the
    /// page's answer in a single number (`savedPrefillSeconds`, cumulative
    /// counters; survives the event-buffer cap, resets on model reload).
    private var headline: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HeroNumber(
                value: savedValue.0,
                unit: savedValue.1,
                label: "prefill time saved",
                isLive: false
            )
            Text(savedSubline)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
        }
    }

    private var savedValue: (String, String?) {
        guard let counters = snapshot?.counters else { return ("—", nil) }
        let seconds = counters.savedPrefillSeconds
        if seconds < 1 { return ("0", "s") }
        if seconds < 60 { return (String(format: "%.0f", seconds), "s") }
        if seconds < 3600 {
            return (
                String(format: "%dm %02ds", Int(seconds) / 60, Int(seconds) % 60), nil
            )
        }
        return (
            String(format: "%dh %02dm", Int(seconds) / 3600, (Int(seconds) % 3600) / 60), nil
        )
    }

    private var savedSubline: String {
        guard let counters = snapshot?.counters else {
            return "load a model and run completions to start counting"
        }
        var parts = ["since model load"]
        parts.append("\(PromptCacheFormatting.compactNumber(counters.hitTokens)) tokens from cache")
        if counters.hydrations > 0 {
            parts.append("\(counters.hydrations) SSD hydrations")
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Tiles

    private var statTiles: some View {
        LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 168), spacing: Theme.Spacing.lg)],
            alignment: .leading,
            spacing: Theme.Spacing.lg
        ) {
            CacheStatTile(
                label: "hit rate · \(window.rawValue)",
                value: hitRateText,
                detail: "\(windowPoints.count) cache-aware requests"
            )
            CacheStatTile(
                label: "token reuse · \(window.rawValue)",
                value: tokenReuseText,
                detail: reuseDetail
            )
            CacheGaugeTile(
                label: "RAM budget",
                used: snapshot?.residentSnapshotBytes ?? 0,
                capacity: snapshot?.memoryBudgetBytes ?? 0,
                emptyHint: "no live cache"
            )
            CacheGaugeTile(
                label: "SSD tier",
                used: snapshot?.ssd.currentBytes ?? 0,
                capacity: snapshot?.ssd.budgetBytes ?? 0,
                emptyHint: snapshot?.ssd.enabled == false ? "disabled" : "no live cache"
            )
        }
    }

    private var hitRateText: String {
        guard !windowPoints.isEmpty else { return "—" }
        let hits = windowPoints.filter(\.isHit).count
        return PromptCacheFormatting.percent(Double(hits) / Double(windowPoints.count))
    }

    private var tokenReuseText: String {
        let promptTotal = windowPoints.reduce(0) { $0 + $1.promptTokens }
        guard promptTotal > 0 else { return "—" }
        let hitTotal = windowPoints.reduce(0) { $0 + $1.hitTokens }
        return PromptCacheFormatting.percent(Double(hitTotal) / Double(promptTotal))
    }

    private var reuseDetail: String {
        let hitTotal = windowPoints.reduce(0) { $0 + $1.hitTokens }
        guard hitTotal > 0 else { return "prompt tokens served from cache" }
        return "\(PromptCacheFormatting.compactNumber(hitTotal)) tokens skipped prefill"
    }
}

// MARK: - Tiles

private struct CacheStatTile: View {
    let label: String
    let value: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(value)
                .font(.system(size: 22, weight: .semibold))
                .monospacedDigit()
                .contentTransition(.numericText())
            Text(label)
                .font(.caption2.weight(.semibold))
                .textCase(.uppercase)
                .kerning(0.8)
                .foregroundStyle(.secondary)
            Text(detail)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct CacheGaugeTile: View {
    let label: String
    let used: Int
    let capacity: Int
    let emptyHint: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(capacity > 0 ? PromptCacheFormatting.bytes(used) : "—")
                .font(.system(size: 22, weight: .semibold))
                .monospacedDigit()
                .contentTransition(.numericText())
            Text(label)
                .font(.caption2.weight(.semibold))
                .textCase(.uppercase)
                .kerning(0.8)
                .foregroundStyle(.secondary)
            if capacity > 0 {
                Gauge(value: min(Double(used) / Double(capacity), 1)) {
                    EmptyView()
                }
                .gaugeStyle(.accessoryLinearCapacity)
                .controlSize(.small)
                Text("of \(PromptCacheFormatting.bytes(capacity)) budget")
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            } else {
                Text(emptyHint)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Chart section chrome

private struct ChartSection<Content: View>: View {
    let title: String
    let footnote: String?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text(title)
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
                .kerning(0.8)
                .foregroundStyle(.secondary)
            content
            if let footnote {
                Text(footnote)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
    }
}

private struct ChartEmptyHint: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, minHeight: 120)
    }
}

// MARK: - Chart 1: TTFT breakdown

/// The showpiece: each request's time-to-first-token stacked into its four
/// stages — lookup · restore · prefill · residual. A hit shows as a bar
/// with almost no orange; a miss is mostly orange (prefill).
private struct TTFTBreakdownChart: View {
    let points: [CacheCorpusStore.Point]
    let window: CacheWindow

    private static let maxBars = 80

    private struct StageSlice: Identifiable {
        let id: String
        let position: Int
        let stage: String
        let seconds: Double
    }

    @State private var hoveredPosition: Int?
    @State private var chartWidth: CGFloat = 0

    private var recentPoints: [CacheCorpusStore.Point] {
        Array(points.suffix(Self.maxBars))
    }

    /// `.ratio` width silently resolves to zero on a continuous Int x-axis
    /// (verified offscreen: marks vanish while rule/legend survive), and the
    /// automatic fallback width doesn't shrink with density, so bars merge at
    /// ~400 pt. Fixed width from measured chart width is the one that scales.
    private var barWidth: CGFloat {
        let count = max(recentPoints.count, 1)
        let axisGutter: CGFloat = 44
        guard chartWidth > axisGutter else { return 4 }
        return min(18, max(1.5, (chartWidth - axisGutter) / CGFloat(count) * 0.72))
    }

    private var slices: [StageSlice] {
        recentPoints.enumerated().flatMap { index, point in
            stages(of: point).map { stage, seconds in
                StageSlice(
                    id: "\(point.id)-\(stage)",
                    position: index,
                    stage: stage,
                    seconds: seconds
                )
            }
        }
    }

    private func stages(of point: CacheCorpusStore.Point) -> [(String, Double)] {
        [
            ("Lookup", point.lookupSeconds),
            ("Restore", point.restoreSeconds),
            ("Prefill", point.prefillSeconds),
            ("Residual", point.residualPromptSeconds),
        ]
    }

    private func stageColor(_ stage: String) -> DynamicColor {
        switch stage {
        case "Lookup": ChartPalette.slot1
        case "Restore": ChartPalette.slot2
        case "Prefill": ChartPalette.slot3
        default: ChartPalette.slot4
        }
    }

    /// Heavy-tail guard: a single cold multi-second prefill on a linear axis
    /// squashes the typical sub-second bar below one pixel. Cap the y-domain
    /// near p95 when the tail is heavy; off-scale bars clip at the top edge
    /// (reading, truthfully, as "off the chart") and keep exact values in
    /// the tooltip.
    private var yDomainInfo: (upper: Double, offScaleCount: Int) {
        let totals = recentPoints.map(\.ttftSeconds).sorted()
        guard let maxTotal = totals.last, maxTotal > 0 else { return (1, 0) }
        let fitted = maxTotal * 1.05
        guard totals.count >= 8 else { return (fitted, 0) }
        let p95 = totals[Int(Double(totals.count - 1) * 0.95)]
        guard p95 > 0, maxTotal > p95 * 1.6 else { return (fitted, 0) }
        let cap = p95 * 1.15
        return (cap, totals.filter { $0 > cap }.count)
    }

    var body: some View {
        let domain = yDomainInfo

        ChartSection(
            title: "TTFT breakdown",
            footnote: points.isEmpty
                ? nil
                : "last \(min(points.count, Self.maxBars)) of \(points.count) requests in \(window.rawValue), oldest → newest · lookup + restore + prefill + residual = time to first token"
                    + (domain.offScaleCount > 0
                        ? " · y-axis capped near p95 — \(domain.offScaleCount) slow "
                            + "\(domain.offScaleCount == 1 ? "request runs" : "requests run") "
                            + "off-scale, hover for exact"
                        : "")
        ) {
            if points.isEmpty {
                ChartEmptyHint(
                    text: "No completions in this window yet — TTFT stages chart here per request."
                )
            } else {
                Chart {
                    ForEach(slices) { slice in
                        BarMark(
                            x: .value("Request", slice.position),
                            y: .value("Seconds", slice.seconds),
                            width: .fixed(barWidth)
                        )
                        .foregroundStyle(by: .value("Stage", slice.stage))
                        .cornerRadius(1.5)
                    }

                    if let hoveredPosition,
                        recentPoints.indices.contains(hoveredPosition)
                    {
                        let point = recentPoints[hoveredPosition]
                        RuleMark(x: .value("Request", hoveredPosition))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                            .foregroundStyle(.quaternary)
                            .annotation(
                                position: .top,
                                spacing: 6,
                                overflowResolution: .init(
                                    x: .fit(to: .chart), y: .fit(to: .chart))
                            ) {
                                ChartTooltipChrome {
                                    ChartTooltipRow(
                                        label: "ttft",
                                        value: PromptCacheFormatting.milliseconds(
                                            point.ttftSeconds * 1000)
                                    )
                                    ForEach(stages(of: point), id: \.0) { stage, seconds in
                                        ChartTooltipRow(
                                            dot: stageColor(stage),
                                            label: stage.lowercased(),
                                            value: PromptCacheFormatting.milliseconds(
                                                seconds * 1000)
                                        )
                                    }
                                }
                            }
                    }
                }
                .chartForegroundStyleScale([
                    "Lookup": ChartPalette.slot1,
                    "Restore": ChartPalette.slot2,
                    "Prefill": ChartPalette.slot3,
                    "Residual": ChartPalette.slot4,
                ])
                .chartYScale(domain: 0...domain.upper)
                // Swift Charts does not clip marks to the plot area; with a
                // capped domain the off-scale bars would otherwise paint over
                // the tiles above (verified offscreen).
                .chartPlotStyle { $0.clipped() }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .trailing) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let seconds = value.as(Double.self) {
                                Text(String(format: "%.1fs", seconds))
                                    .font(.caption2.monospacedDigit())
                            }
                        }
                    }
                }
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        onMove: { location in
                            guard let x = proxy.value(atX: location.x, as: Double.self) else {
                                hoveredPosition = nil
                                return
                            }
                            let index = Int(x.rounded())
                            hoveredPosition =
                                recentPoints.indices.contains(index) ? index : nil
                        },
                        onExit: { hoveredPosition = nil }
                    )
                }
                .frame(height: 180)
                .onGeometryChange(for: CGFloat.self) { proxy in
                    proxy.size.width
                } action: { width in
                    chartWidth = width
                }
            }
        }
    }
}

// MARK: - Chart 2: hit rate & token reuse trend

private struct HitRateTrendChart: View {
    let points: [CacheCorpusStore.Point]
    let window: CacheWindow

    private struct TrendBucket: Identifiable, Equatable {
        let id: Date
        let hitRate: Double
        let tokenReuse: Double
        let requestCount: Int
    }

    @State private var hovered: TrendBucket?

    private var buckets: [TrendBucket] {
        let calendar = Calendar.current
        let component: Calendar.Component = window.bucketsByHour ? .hour : .day
        let grouped = Dictionary(grouping: points) { point in
            calendar.dateInterval(of: component, for: point.timestamp)?.start ?? point.timestamp
        }
        return grouped.keys.sorted().compactMap { bucket -> TrendBucket? in
            let bucketPoints = grouped[bucket] ?? []
            guard !bucketPoints.isEmpty else { return nil }
            let hitRate = Double(bucketPoints.filter(\.isHit).count) / Double(bucketPoints.count)
            let promptTotal = bucketPoints.reduce(0) { $0 + $1.promptTokens }
            let hitTotal = bucketPoints.reduce(0) { $0 + $1.hitTokens }
            return TrendBucket(
                id: bucket,
                hitRate: hitRate,
                tokenReuse: promptTotal > 0 ? Double(hitTotal) / Double(promptTotal) : 0,
                requestCount: bucketPoints.count
            )
        }
    }

    private var series: [(name: String, color: DynamicColor, value: (TrendBucket) -> Double)] {
        [
            (name: "Hit rate", color: ChartPalette.slot1, value: { $0.hitRate }),
            (name: "Token reuse", color: ChartPalette.slot2, value: { $0.tokenReuse }),
        ]
    }

    var body: some View {
        ChartSection(
            title: "Hit rate & token reuse",
            footnote: points.isEmpty
                ? nil
                : "per \(window.bucketsByHour ? "hour" : "day"), completion-trace corpus"
        ) {
            if points.isEmpty {
                ChartEmptyHint(
                    text: "No completions in this window yet — the reuse trend charts here."
                )
            } else {
                Chart {
                    ForEach(buckets) { bucket in
                        ForEach(series, id: \.name) { series in
                            LineMark(
                                x: .value("Time", bucket.id),
                                y: .value("Rate", series.value(bucket))
                            )
                            .foregroundStyle(by: .value("Series", series.name))
                            .lineStyle(
                                StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

                            PointMark(
                                x: .value("Time", bucket.id),
                                y: .value("Rate", series.value(bucket))
                            )
                            .foregroundStyle(by: .value("Series", series.name))
                            .symbolSize(28)
                        }
                    }

                    if let hovered {
                        RuleMark(x: .value("Time", hovered.id))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                            .foregroundStyle(.quaternary)
                            .annotation(
                                position: .top,
                                spacing: 6,
                                overflowResolution: .init(
                                    x: .fit(to: .chart), y: .fit(to: .chart))
                            ) {
                                ChartTooltipChrome {
                                    Text(bucketLabel(hovered.id))
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                    ChartTooltipRow(
                                        dot: ChartPalette.slot1,
                                        label: "hit rate",
                                        value: PromptCacheFormatting.percent(hovered.hitRate)
                                    )
                                    ChartTooltipRow(
                                        dot: ChartPalette.slot2,
                                        label: "token reuse",
                                        value: PromptCacheFormatting.percent(hovered.tokenReuse)
                                    )
                                    ChartTooltipRow(
                                        label: "requests",
                                        value: "\(hovered.requestCount)"
                                    )
                                }
                            }

                        ForEach(series, id: \.name) { series in
                            PointMark(
                                x: .value("Time", hovered.id),
                                y: .value("Rate", series.value(hovered))
                            )
                            .symbolSize(56)
                            .foregroundStyle(series.color)
                        }
                    }
                }
                .chartForegroundStyleScale([
                    "Hit rate": ChartPalette.slot1,
                    "Token reuse": ChartPalette.slot2,
                ])
                .chartYScale(domain: 0...1)
                .chartYAxis {
                    AxisMarks(position: .trailing, values: [0, 0.25, 0.5, 0.75, 1]) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let rate = value.as(Double.self) {
                                Text("\(Int(rate * 100))%")
                                    .font(.caption2.monospacedDigit())
                            }
                        }
                    }
                }
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        onMove: { location in
                            guard let date = proxy.value(atX: location.x, as: Date.self) else {
                                hovered = nil
                                return
                            }
                            hovered = buckets.min {
                                abs($0.id.timeIntervalSince(date))
                                    < abs($1.id.timeIntervalSince(date))
                            }
                        },
                        onExit: { hovered = nil }
                    )
                }
                .frame(height: 160)
            }
        }
    }

    private func bucketLabel(_ date: Date) -> String {
        if window.bucketsByHour {
            return date.formatted(.dateTime.weekday(.abbreviated).hour())
        }
        return date.formatted(.dateTime.month(.abbreviated).day())
    }
}

// MARK: - Chart 3: SSD writes & deletes

private struct SSDWriteChart: View {
    let endurance: SSDEnduranceSnapshot
    let window: CacheWindow

    private struct WritePoint: Identifiable {
        let id: String
        let bucket: Date
        let kind: String
        let bytes: Int
    }

    @State private var hovered: SSDEnduranceSnapshot.DatedBucket?

    private func bucketLabel(_ date: Date) -> String {
        if window.bucketsByHour {
            return date.formatted(.dateTime.weekday(.abbreviated).hour())
        }
        return date.formatted(.dateTime.month(.abbreviated).day())
    }

    private var buckets: [SSDEnduranceSnapshot.DatedBucket] {
        let source = window.bucketsByHour ? endurance.hourly : endurance.daily
        let cutoff = Date().addingTimeInterval(-window.duration)
        return source.filter { $0.id >= cutoff }
    }

    private var writePoints: [WritePoint] {
        buckets.flatMap { bucket in
            [
                WritePoint(
                    id: "w-\(bucket.id.timeIntervalSinceReferenceDate)",
                    bucket: bucket.id, kind: "Written", bytes: bucket.bytesWritten),
                WritePoint(
                    id: "d-\(bucket.id.timeIntervalSinceReferenceDate)",
                    bucket: bucket.id, kind: "Deleted", bytes: bucket.bytesDeleted),
            ]
        }
    }

    var body: some View {
        ChartSection(
            title: "SSD writes & deletes",
            footnote: buckets.isEmpty
                ? nil
                : "per \(window.bucketsByHour ? "hour" : "day"), endurance ledger · lifetime "
                    + "\(PromptCacheFormatting.bytes(endurance.lifetimeBytesWritten)) written · "
                    + "\(PromptCacheFormatting.bytes(endurance.lifetimeBytesDeleted)) deleted"
        ) {
            if buckets.isEmpty {
                ChartEmptyHint(
                    text: "No SSD activity in this window — writes and deletes chart here."
                )
            } else {
                Chart {
                    ForEach(writePoints) { point in
                        BarMark(
                            x: .value(
                                "Time", point.bucket, unit: window.bucketsByHour ? .hour : .day),
                            y: .value("Bytes", point.bytes)
                        )
                        .foregroundStyle(by: .value("Kind", point.kind))
                        .position(by: .value("Kind", point.kind))
                        .cornerRadius(1.5)
                    }

                    if let hovered {
                        RuleMark(x: .value("Time", hovered.id))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                            .foregroundStyle(.quaternary)
                            .annotation(
                                position: .top,
                                spacing: 6,
                                overflowResolution: .init(
                                    x: .fit(to: .chart), y: .fit(to: .chart))
                            ) {
                                ChartTooltipChrome {
                                    Text(bucketLabel(hovered.id))
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                    ChartTooltipRow(
                                        dot: ChartPalette.slot3,
                                        label: "written",
                                        value: PromptCacheFormatting.bytes(hovered.bytesWritten)
                                    )
                                    ChartTooltipRow(
                                        dot: ChartPalette.slot1,
                                        label: "deleted",
                                        value: PromptCacheFormatting.bytes(hovered.bytesDeleted)
                                    )
                                }
                            }
                    }
                }
                .chartForegroundStyleScale([
                    "Written": ChartPalette.slot3,
                    "Deleted": ChartPalette.slot1,
                ])
                .chartYAxis {
                    AxisMarks(position: .trailing) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let bytes = value.as(Int.self) {
                                Text(PromptCacheFormatting.bytes(bytes))
                                    .font(.caption2.monospacedDigit())
                            }
                        }
                    }
                }
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        onMove: { location in
                            guard let date = proxy.value(atX: location.x, as: Date.self) else {
                                hovered = nil
                                return
                            }
                            hovered = buckets.min {
                                abs($0.id.timeIntervalSince(date))
                                    < abs($1.id.timeIntervalSince(date))
                            }
                        },
                        onExit: { hovered = nil }
                    )
                }
                .frame(height: 160)
            }
        }
    }
}
