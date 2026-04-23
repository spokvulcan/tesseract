import AppKit
import SwiftUI

/// Live generation activity panel shown on the Server API settings page.
/// Split layout: request rail (left) + selected-trace detail (right).
struct ServerActivityPanel: View {
    @Environment(ServerGenerationLog.self) private var log

    var body: some View {
        HStack(spacing: 0) {
            RequestRail()
                .frame(width: 220)

            Divider()

            if let selected = selectedTrace {
                ActiveTraceDetailView(trace: selected)
                    .frame(maxWidth: .infinity)
            } else {
                EmptyDetailPlaceholder()
            }
        }
        .frame(maxWidth: .infinity)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous))
    }

    private var selectedTrace: RequestTrace? {
        guard let id = log.selectedTraceID else { return log.traces.last }
        return log.traces.first { $0.id == id }
    }
}

// MARK: - Request Rail

private struct RequestRail: View {
    @Environment(ServerGenerationLog.self) private var log

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Requests")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !log.traces.isEmpty {
                    Button {
                        log.clear()
                    } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.glass)
                    .help("Clear all")
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)

            Divider()

            if log.traces.isEmpty {
                VStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "waveform")
                        .foregroundStyle(.tertiary)
                    Text("Waiting for requests…")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Eager `ScrollView { VStack { ForEach } }` instead of
                // `List(selection:)`: List's per-row NSHostingView prefetch
                // (LazyLayoutViewCache.signalPrefetch) races constraint
                // updates against the 33ms streamingVersion bumps + 100ms
                // TimelineView ticks inside active rows and trips an AppKit
                // re-entrant layout trap. Same workaround StreamingSpanListView
                // uses for active traces.
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(log.traces.reversed()) { trace in
                            RequestRailRow(
                                trace: trace,
                                isSelected: log.selectedTraceID == trace.id
                            ) {
                                log.selectedTraceID = trace.id
                            }
                        }
                    }
                    .padding(.vertical, Theme.Spacing.xs)
                }
                .scrollBounceBehavior(.basedOnSize)
            }
        }
    }
}

// MARK: - Request rail row

private struct RequestRailRow: View {
    let trace: RequestTrace
    let isSelected: Bool
    let onSelect: () -> Void
    @State private var isHovered = false

    var body: some View {
        RequestTraceRow(trace: trace)
            .padding(.vertical, 4)
            .padding(.horizontal, Theme.Spacing.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous)
                    .fill(fillStyle)
            )
            .onHover { isHovered = $0 }
            .onTapGesture(perform: onSelect)
            .padding(.horizontal, Theme.Spacing.xs)
    }

    private var fillStyle: AnyShapeStyle {
        if isSelected { return AnyShapeStyle(.selection) }
        if isHovered  { return AnyShapeStyle(.quaternary) }
        return AnyShapeStyle(Color.clear)
    }
}

// MARK: - Empty placeholder

private struct EmptyDetailPlaceholder: View {
    var body: some View {
        VStack(spacing: Theme.Spacing.sm) {
            Image(systemName: "terminal")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text("No request selected")
                .font(.body)
                .foregroundStyle(.secondary)
            Text("Token-by-token generation appears here once a request arrives at /v1/chat/completions.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
