import SwiftUI

struct PromptCacheOverviewView: View {
    let snapshot: PromptCacheTelemetrySnapshot?
    let aggregate: PromptCacheTelemetryAggregate
    let samples: [PromptCacheMetricSample]

    var body: some View {
        GeometryReader { proxy in
            let compact = proxy.size.width < PromptCacheLayout.compactWidth
            let chartHeight = chartAvailableHeight(in: proxy.size)

            ScrollView {
                VStack(alignment: .leading, spacing: compact ? Theme.Spacing.xs : Theme.Spacing.sm) {
                    PromptCacheKPIStrip(snapshot: snapshot, aggregate: aggregate)

                    PromptCacheChartsView(
                        samples: samples,
                        availableHeight: chartHeight,
                        layoutWidth: proxy.size.width
                    )
                }
                .frame(maxWidth: .infinity, minHeight: proxy.size.height, alignment: .topLeading)
            }
        }
    }

    private func chartAvailableHeight(in size: CGSize) -> CGFloat {
        let compact = size.width < PromptCacheLayout.compactWidth
        let estimatedMetricRows: CGFloat = compact ? 4 : (size.width < PromptCacheLayout.wideWidth ? 3 : 2)
        let metricsHeight = estimatedMetricRows * 76
        let spacing = compact ? Theme.Spacing.xs : Theme.Spacing.sm
        return max(180, size.height - metricsHeight - spacing)
    }
}
