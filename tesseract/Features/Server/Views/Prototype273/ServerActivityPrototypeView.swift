//
//  ServerActivityPrototypeView.swift
//  tesseract
//
//  PROTOTYPE (wayfinder #273) — THROWAWAY. The Activity page of the locked
//  direction (#272): "what is it doing now" — a status strip over the
//  full-exchange transcript, with the recent-requests rail. The ⌘` console
//  drawer is retired; its jobs live in the strip, the transcript header,
//  and the rail.
//

import Charts
import SwiftUI
import Textual

struct ServerActivityPrototypeView: View {
    @Environment(ServerGenerationLog.self) private var log
    @Environment(HTTPServer.self) private var server

    /// Explicit rail selection. `nil` = Live: follow the newest request as
    /// it arrives. Local to the page so the log's own auto-selection
    /// behavior stays untouched.
    @State private var pinnedTraceID: UUID?
    @State private var contentWidth: CGFloat = 0

    private var isWide: Bool { contentWidth >= Proto273Layout.activityWideBreakpoint }

    var body: some View {
        VStack(spacing: 0) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                ServerStatusStrip(server: server, now: context.date)
            }

            Divider()

            if isWide {
                HStack(spacing: 0) {
                    transcript
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    Divider()
                    RecentRequestsRail(
                        traces: log.traces,
                        heroTraceID: heroTrace?.id,
                        isLive: pinnedTraceID == nil,
                        isCompact: false,
                        onSelect: { pinnedTraceID = $0 },
                        onGoLive: { pinnedTraceID = nil }
                    )
                    .frame(width: Proto273Layout.railWidth)
                }
            } else {
                transcript
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                RecentRequestsRail(
                    traces: log.traces,
                    heroTraceID: heroTrace?.id,
                    isLive: pinnedTraceID == nil,
                    isCompact: true,
                    onSelect: { pinnedTraceID = $0 },
                    onGoLive: { pinnedTraceID = nil }
                )
                .frame(height: Proto273Layout.stackedRailHeight)
            }
        }
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { width in
            contentWidth = width
        }
        .navigationTitle("Activity (Prototype)")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if let cancellable = cancellableTrace {
                    Button {
                        log.requestCancel(traceID: cancellable.id)
                    } label: {
                        Label {
                            Text("Stop")
                        } icon: {
                            Image(systemName: "stop.fill")
                                .foregroundStyle(.red)
                        }
                    }
                    .keyboardShortcut(".", modifiers: .command)
                    .help("Stop the current generation (⌘.)")
                }

                Button {
                    Proto273ReplayDriver.shared.play(into: log)
                } label: {
                    Label("Demo Traffic", systemImage: "testtube.2")
                }
                .disabled(Proto273ReplayDriver.shared.isPlaying)
                .help("Stream two synthetic requests through the transcript (prototype only)")

                Menu {
                    ShareLink(item: heroTrace?.concatenatedText ?? "") {
                        Label("Share Output", systemImage: "square.and.arrow.up")
                    }
                    .disabled((heroTrace?.concatenatedText ?? "").isEmpty)

                    Button {
                        NSWorkspace.shared.open(HTTPRequestLogger.shared.directoryURL)
                    } label: {
                        Label("Reveal Raw Requests", systemImage: "doc.text.magnifyingglass")
                    }

                    Divider()

                    Button(role: .destructive) {
                        pinnedTraceID = nil
                        log.clear()
                    } label: {
                        Label("Clear Requests", systemImage: "trash")
                    }
                    .disabled(log.traces.isEmpty)
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
            }
        }
    }

    // MARK: - Content

    @ViewBuilder
    private var transcript: some View {
        if let heroTrace {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                ActivityTranscriptView(trace: heroTrace, now: context.date)
            }
        } else {
            ContentUnavailableView {
                Label("No Requests Yet", systemImage: "waveform")
            } description: {
                Text(
                    "Requests to /v1/chat/completions stream in here as full exchanges."
                        + "\nUse Demo Traffic to preview one without a client."
                )
            }
        }
    }

    /// The request on stage: the pinned rail selection when valid, else
    /// live — the newest trace (the log auto-selects arrivals).
    private var heroTrace: RequestTrace? {
        if let pinnedTraceID,
            let pinned = log.traces.first(where: { $0.id == pinnedTraceID })
        {
            return pinned
        }
        return log.traces.last
    }

    private var cancellableTrace: RequestTrace? {
        log.traces.last { $0.isActive && $0.isCancellable && !$0.cancelRequested }
    }
}

// MARK: - Status strip

/// The server's process vitals in one quiet line: running state, bound
/// port, active connections, total served — `HTTPServer` is already
/// observable; no page showed it before this one.
private struct ServerStatusStrip: View {
    let server: HTTPServer
    let now: Date

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)

            Text(statusText)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer(minLength: 0)

            Text(countsText)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, 6)
        .background(.background.secondary)
    }

    private var statusColor: Color {
        if server.isRunning { return .green }
        if server.isStarting { return .orange }
        if server.lastStartError != nil { return .red }
        return Color(nsColor: .tertiaryLabelColor)
    }

    private var statusText: String {
        if server.isRunning, let port = server.boundPort {
            return "serving on 127.0.0.1:\(port)"
        }
        if server.isStarting { return "starting…" }
        if let error = server.lastStartError { return "failed to start: \(error)" }
        return "server stopped — enable it in Settings"
    }

    private var countsText: String {
        "\(server.activeConnections) active · \(server.totalRequestsServed) served"
    }
}

// MARK: - Recent rail

/// The 20 in-memory traces as metric-rich rows, newest first. Clicking a
/// row pins the transcript to that request; the Live control resumes
/// following arrivals. Icon-light rows — outcome reads from text color
/// and the metrics, actions stay in the transcript's context menu.
private struct RecentRequestsRail: View {
    let traces: [RequestTrace]
    let heroTraceID: UUID?
    let isLive: Bool
    let isCompact: Bool
    let onSelect: (UUID) -> Void
    let onGoLive: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Recent")
                    .font(.caption.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(0.8)
                    .foregroundStyle(.secondary)

                Spacer(minLength: 0)

                Button(action: onGoLive) {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(isLive ? AnyShapeStyle(.green) : AnyShapeStyle(.tertiary))
                            .frame(width: 6, height: 6)
                        Text("Live")
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isLive ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Follow the newest request as it arrives")
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)

            Divider()

            RailThroughputCharts(traces: traces, isCompact: isCompact)

            Divider()

            if traces.isEmpty {
                Spacer(minLength: 0)
                Text("No requests yet")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                Spacer(minLength: 0)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(traces.reversed()) { trace in
                            RecentRequestRow(
                                trace: trace,
                                isSelected: trace.id == heroTraceID,
                                onSelect: { onSelect(trace.id) }
                            )
                        }
                    }
                }
            }
        }
        .background(.background.secondary)
    }
}

// MARK: - Throughput charts

/// Decode tok/s and prefill tok/s per request, over the session's recent
/// traces — the two rates where drift actually shows (context growth
/// drags decode; a cooling cache drags prefill). Replaces the per-request
/// decode sparkline, which drew a near-constant rate as a flat line and
/// said nothing. One series per chart, so the caption is the legend;
/// prefill wears the palette's orange (slot 3) everywhere, decode blue.
private struct RailThroughputCharts: View {
    let traces: [RequestTrace]
    let isCompact: Bool

    private var decodePoints: [RailRatePoint] {
        traces.compactMap { trace in
            guard trace.tokensPerSecond > 0 else { return nil }
            return RailRatePoint(
                id: trace.id, sequence: trace.sequence, rate: trace.tokensPerSecond)
        }
    }

    private var prefillPoints: [RailRatePoint] {
        traces.compactMap { trace in
            guard let rate = trace.prefillTokensPerSecond else { return nil }
            return RailRatePoint(id: trace.id, sequence: trace.sequence, rate: rate)
        }
    }

    var body: some View {
        if isCompact {
            HStack(alignment: .top, spacing: Theme.Spacing.lg) {
                RailRateChart(
                    title: "decode tok/s",
                    points: decodePoints,
                    color: ChartPalette.slot1,
                    isCompact: true
                )
                RailRateChart(
                    title: "prefill tok/s",
                    points: prefillPoints,
                    color: ChartPalette.slot3,
                    isCompact: true
                )
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
        } else {
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                RailRateChart(
                    title: "decode tok/s",
                    points: decodePoints,
                    color: ChartPalette.slot1,
                    isCompact: false
                )
                RailRateChart(
                    title: "prefill tok/s",
                    points: prefillPoints,
                    color: ChartPalette.slot3,
                    isCompact: false
                )
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)
        }
    }
}

private struct RailRatePoint: Identifiable {
    let id: UUID
    let sequence: Int
    let rate: Double
}

/// One rail rate chart with the hover cursor: a hairline rule snapped to
/// the nearest request plus an emphasized point and a tooltip chip with
/// the exact number.
private struct RailRateChart: View {
    let title: String
    let points: [RailRatePoint]
    let color: DynamicColor
    let isCompact: Bool

    @State private var hovered: RailRatePoint?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.caption2.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(0.8)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
                if let last = points.last {
                    Text(String(format: "%.0f", last.rate))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            if points.count >= 2 {
                Chart {
                    ForEach(points) { point in
                        LineMark(
                            x: .value("Request", point.sequence),
                            y: .value("tok/s", point.rate)
                        )
                        .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                        .foregroundStyle(color)

                        PointMark(
                            x: .value("Request", point.sequence),
                            y: .value("tok/s", point.rate)
                        )
                        .symbolSize(20)
                        .foregroundStyle(color)
                    }

                    if let hovered {
                        RuleMark(x: .value("Request", hovered.sequence))
                            .lineStyle(StrokeStyle(lineWidth: 1))
                            .foregroundStyle(.quaternary)

                        PointMark(
                            x: .value("Request", hovered.sequence),
                            y: .value("tok/s", hovered.rate)
                        )
                        .symbolSize(56)
                        .foregroundStyle(color)
                        .annotation(
                            position: .top,
                            spacing: 6,
                            overflowResolution: .init(x: .fit(to: .chart), y: .fit(to: .chart))
                        ) {
                            ChartTooltipChrome {
                                ChartTooltipRow(
                                    label: "#\(hovered.sequence)",
                                    value: String(format: "%.1f", hovered.rate)
                                )
                            }
                        }
                    }
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .trailing, values: .automatic(desiredCount: 2)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let rate = value.as(Double.self) {
                                Text(String(format: "%.0f", rate))
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
                                hovered = nil
                                return
                            }
                            hovered = points.min {
                                abs(Double($0.sequence) - x) < abs(Double($1.sequence) - x)
                            }
                        },
                        onExit: { hovered = nil }
                    )
                }
                .frame(height: isCompact ? 44 : 52)
            } else {
                Text("charts after 2+ requests")
                    .font(.caption2)
                    .foregroundStyle(.quaternary)
                    .frame(maxWidth: .infinity, minHeight: isCompact ? 44 : 52, alignment: .center)
            }
        }
        .help("Per request, oldest → newest — this session's \(title)")
    }
}

private struct RecentRequestRow: View {
    let trace: RequestTrace
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("#\(trace.sequence)")
                        .foregroundStyle(.tertiary)
                    Text(outcomeText)
                        .foregroundStyle(outcomeStyle)
                    Spacer(minLength: 0)
                    Text(trace.startedAt, format: .dateTime.hour().minute().second())
                        .foregroundStyle(.quaternary)
                }
                Text(metricsLine)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .font(.caption.monospaced())
            .monospacedDigit()
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(
                isSelected ? AnyShapeStyle(.selection.opacity(0.5)) : AnyShapeStyle(.clear)
            )
        }
        .buttonStyle(.plain)
    }

    private var outcomeText: String {
        switch trace.phase {
        case .queued: "queued"
        case .lookingUp: "lookup"
        case .prefilling: "prefill"
        case .decoding: "decoding"
        case .completed: "done"
        case .failed: "failed"
        case .cancelled: "cancelled"
        }
    }

    private var outcomeStyle: AnyShapeStyle {
        switch trace.phase {
        case .failed: AnyShapeStyle(DynamicColor.chatError)
        case .completed: AnyShapeStyle(.secondary)
        case .cancelled: AnyShapeStyle(.tertiary)
        default: AnyShapeStyle(.green)
        }
    }

    private var metricsLine: String {
        var parts: [String] = []
        if let promptTokens = trace.promptTokens, promptTokens > 0 {
            let ratio = Double(trace.cachedTokens) / Double(promptTokens)
            parts.append(String(format: "hit %.0f%%", ratio * 100))
        }
        if let ttft = trace.ttftMs {
            parts.append(
                ttft < 1000
                    ? String(format: "%.0f ms", ttft)
                    : String(format: "%.1f s", ttft / 1000))
        }
        if trace.tokensPerSecond > 0 {
            parts.append(String(format: "%.0f tok/s", trace.tokensPerSecond))
        } else if trace.displayOutputTokens > 0 {
            parts.append("\(trace.displayOutputTokens) out")
        }
        if parts.isEmpty {
            parts.append(trace.model.isEmpty ? "—" : trace.model)
        }
        return parts.joined(separator: " · ")
    }
}
