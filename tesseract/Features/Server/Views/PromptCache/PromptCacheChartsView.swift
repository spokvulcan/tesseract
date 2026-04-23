import Charts
import SwiftUI

struct PromptCacheChartsView: View {
    let samples: [PromptCacheMetricSample]
    let availableHeight: CGFloat
    let layoutWidth: CGFloat

    var body: some View {
        LazyVGrid(columns: columns, spacing: Theme.Spacing.xs) {
            ForEach(PromptCacheChartKind.allCases) { chart in
                PromptCacheChartPanel(
                    kind: chart,
                    samples: samples,
                    height: panelHeight
                )
            }
        }
    }

    private var columns: [GridItem] {
        if layoutWidth >= PromptCacheLayout.wideWidth {
            return Array(repeating: GridItem(.flexible(minimum: 180), spacing: Theme.Spacing.xs), count: 3)
        }
        return [GridItem(.flexible(minimum: 0), spacing: Theme.Spacing.xs)]
    }

    private var panelHeight: CGFloat {
        if layoutWidth >= PromptCacheLayout.wideWidth {
            return max(220, availableHeight)
        }
        return max(178, (availableHeight - Theme.Spacing.xs * 2) / 3)
    }
}

private struct PromptCacheChartPanel: View {
    let kind: PromptCacheChartKind
    let samples: [PromptCacheMetricSample]
    let height: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Label(kind.title, systemImage: kind.symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            chartContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(Theme.Spacing.sm)
        .frame(maxWidth: .infinity, minHeight: height, maxHeight: height, alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }

    @ViewBuilder
    private var chartContent: some View {
        if samples.isEmpty {
            chartPlaceholder
        } else {
            switch kind {
            case .efficiency:
                efficiencyChart
            case .memory:
                memoryChart
            case .latency:
                latencyChart
            }
        }
    }

    private var efficiencyChart: some View {
        Chart(samples) { sample in
            LineMark(
                x: .value("Time", sample.date),
                y: .value("Hit Rate", sample.hitRate * 100)
            )
            .foregroundStyle(.green)
            .interpolationMethod(.catmullRom)

            LineMark(
                x: .value("Time", sample.date),
                y: .value("Token Reuse", sample.tokenReuseRate * 100)
            )
            .foregroundStyle(.mint)
            .interpolationMethod(.catmullRom)
        }
        .chartYScale(domain: 0...100)
        .chartYAxisLabel("%")
        .accessibilityLabel("Prompt cache hit rate and token reuse rate over time")
    }

    private var memoryChart: some View {
        Chart(samples) { sample in
            LineMark(
                x: .value("Time", sample.date),
                y: .value("RAM bytes", sample.ramBytes)
            )
            .foregroundStyle(.blue)
            .interpolationMethod(.catmullRom)

            if sample.ramBudgetBytes > 0 {
                RuleMark(y: .value("RAM budget", sample.ramBudgetBytes))
                    .foregroundStyle(.blue.opacity(0.25))
            }

            if sample.ssdBudgetBytes > 0 {
                LineMark(
                    x: .value("Time", sample.date),
                    y: .value("SSD bytes", sample.ssdBytes)
                )
                .foregroundStyle(.teal)
                .interpolationMethod(.catmullRom)
            }
        }
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let bytes = value.as(Int.self) {
                        Text(PromptCacheFormatting.bytes(bytes))
                    }
                }
            }
        }
        .accessibilityLabel("Prompt cache RAM and SSD memory over time")
    }

    private var latencyChart: some View {
        Chart(samples) { sample in
            LineMark(
                x: .value("Time", sample.date),
                y: .value("Lookup", sample.averageLookupMs)
            )
            .foregroundStyle(.orange)

            LineMark(
                x: .value("Time", sample.date),
                y: .value("Restore", sample.averageRestoreMs)
            )
            .foregroundStyle(.purple)

            LineMark(
                x: .value("Time", sample.date),
                y: .value("Prefill", sample.averagePrefillMs)
            )
            .foregroundStyle(.cyan)
        }
        .chartYAxisLabel("ms")
        .accessibilityLabel("Prompt cache lookup, restore, and prefill latency over time")
    }

    private var chartPlaceholder: some View {
        VStack(spacing: Theme.Spacing.xs) {
            Image(systemName: "chart.xyaxis.line")
                .foregroundStyle(.tertiary)
            Text("Waiting for telemetry")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
