//
//  CatchWeekChart.swift
//  tesseract
//
//  The catch record's week (PRD #612): per day, the mistakes the Learned
//  Words caught and the words the owner fixed in the Lens, stacked. Chart
//  rules per the design language §5: fixed palette slots, one axis, a
//  legend for the two series, hover with a hairline cursor and a tooltip,
//  and a footnote naming the window.
//

import Charts
import SwiftUI
import Textual

struct CatchWeekChart: View {
    /// Seven days, oldest first, today last.
    let days: [CatchRecord.Day]
    var calendar: Calendar = .current

    static let caughtSeries = "Caught"
    static let fixedSeries = "You fixed"
    static let footnote =
        "Last 7 days, today on the right. Caught counts Learned Words applied to your takes; "
        + "You fixed counts words you fixed in the Lens."

    /// The hovered day's start.
    @State private var hoveredDate: Date?

    private struct Slice: Identifiable {
        let date: Date
        let series: String
        let count: Int

        var id: String { "\(series)-\(date.timeIntervalSinceReferenceDate)" }
    }

    /// Caught first, so it stacks at the bottom and leads the legend.
    private var slices: [Slice] {
        days.flatMap { day in
            [
                Slice(date: day.date, series: Self.caughtSeries, count: day.caught),
                Slice(date: day.date, series: Self.fixedSeries, count: day.fixed),
            ]
        }
    }

    private var hoveredDay: CatchRecord.Day? {
        guard let hoveredDate else { return nil }
        return days.first { $0.date == hoveredDate }
    }

    /// Whole-number ticks from zero: at least 0 to 4, four steps at most,
    /// the top rounded up to a whole step.
    static func yTicks(for days: [CatchRecord.Day]) -> [Int] {
        let top = max(4, days.map { $0.caught + $0.fixed }.max() ?? 0)
        let step = max(1, Int((Double(top) / 4).rounded(.up)))
        let upper = Int((Double(top) / Double(step)).rounded(.up)) * step
        return Array(stride(from: 0, through: upper, by: step))
    }

    var body: some View {
        let ticks = Self.yTicks(for: days)

        VStack(alignment: .leading, spacing: 6) {
            Chart {
                ForEach(slices) { slice in
                    // A date on a day unit is a binned axis, where `.ratio`
                    // widths are defined (design language §5).
                    BarMark(
                        x: .value("Day", slice.date, unit: .day),
                        y: .value("Words", slice.count),
                        width: .ratio(0.55)
                    )
                    .foregroundStyle(by: .value("Series", slice.series))
                    .cornerRadius(2)
                    .opacity(hoveredDate == nil || hoveredDate == slice.date ? 1 : 0.45)
                    .accessibilityLabel(dayName(slice.date))
                    .accessibilityValue(accessibilityValue(slice))
                }

                if let day = hoveredDay {
                    RuleMark(x: .value("Day", day.date, unit: .day))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                        .foregroundStyle(.quaternary)
                        .annotation(
                            position: .top,
                            spacing: 6,
                            overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                        ) {
                            ChartTooltipChrome {
                                Text(dayName(day.date))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                ChartTooltipRow(
                                    dot: ChartPalette.slot1, label: "caught",
                                    value: "\(day.caught)")
                                ChartTooltipRow(
                                    dot: ChartPalette.slot2, label: "you fixed",
                                    value: "\(day.fixed)")
                            }
                        }
                }
            }
            .chartForegroundStyleScale(
                domain: [Self.caughtSeries, Self.fixedSeries],
                range: [ChartPalette.slot1, ChartPalette.slot2]
            )
            .chartLegend(position: .bottom, alignment: .leading)
            .chartYScale(domain: 0...(ticks.last ?? 4))
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { value in
                    AxisValueLabel(centered: true) {
                        if let date = value.as(Date.self) {
                            let today = isToday(date)
                            Text(date.formatted(.dateTime.weekday(.abbreviated)))
                                .font(.caption2)
                                .fontWeight(today ? .semibold : .regular)
                                .foregroundStyle(today ? .primary : .secondary)
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(position: .trailing, values: ticks) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let count = value.as(Int.self) {
                            Text("\(count)")
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
                            hoveredDate = nil
                            return
                        }
                        hoveredDate = nearestDay(to: date)?.date
                    },
                    onExit: { hoveredDate = nil }
                )
            }
            .frame(height: 160)

            Text(Self.footnote)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func isToday(_ date: Date) -> Bool {
        guard let today = days.last?.date else { return false }
        return calendar.isDate(date, inSameDayAs: today)
    }

    /// The day whose bar holds `date`, else the one whose middle is nearest.
    private func nearestDay(to date: Date) -> CatchRecord.Day? {
        if let day = days.first(where: { calendar.isDate($0.date, inSameDayAs: date) }) {
            return day
        }
        return days.min {
            abs($0.date.addingTimeInterval(43_200).timeIntervalSince(date))
                < abs($1.date.addingTimeInterval(43_200).timeIntervalSince(date))
        }
    }

    /// "Today", "Yesterday", or "Thursday, Oct 2".
    private func dayName(_ date: Date) -> String {
        if isToday(date) { return "Today" }
        if let today = days.last?.date,
            let yesterday = calendar.date(byAdding: .day, value: -1, to: today),
            calendar.isDate(date, inSameDayAs: yesterday)
        {
            return "Yesterday"
        }
        return date.formatted(.dateTime.weekday(.wide).month(.abbreviated).day())
    }

    private func accessibilityValue(_ slice: Slice) -> String {
        slice.series == Self.caughtSeries
            ? "\(slice.count) caught" : "\(slice.count) fixed by you"
    }
}
