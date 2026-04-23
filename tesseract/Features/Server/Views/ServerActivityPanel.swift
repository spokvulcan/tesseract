import SwiftUI

/// Live generation activity panel shown on the Server API settings page.
/// Split layout: request rail (left) + selected-trace detail (right).
struct ServerActivityPanel: View {
    @Environment(ServerGenerationLog.self) private var log
    @State private var selectedTraceID: UUID?

    var body: some View {
        HStack(spacing: 0) {
            RequestRail(
                traces: log.traces,
                selectedTraceID: effectiveSelectedTraceID,
                onSelect: { selectedTraceID = $0 },
                onClear: clearTraces
            )
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
        .background(.background, in: RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous)
                .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous))
    }

    private var effectiveSelectedTraceID: UUID? {
        if let selectedTraceID,
           log.traces.contains(where: { $0.id == selectedTraceID }) {
            return selectedTraceID
        }
        return log.traces.last?.id
    }

    private var selectedTrace: RequestTrace? {
        guard let id = effectiveSelectedTraceID else { return nil }
        return log.traces.first { $0.id == id }
    }

    private func clearTraces() {
        selectedTraceID = nil
        log.clear()
    }
}

// MARK: - Request Rail

private struct RequestRail: View {
    let traces: [RequestTrace]
    let selectedTraceID: UUID?
    let onSelect: (UUID) -> Void
    let onClear: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Requests")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !traces.isEmpty {
                    Button {
                        onClear()
                    } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Clear all")
                }
            }
            .padding(.horizontal, Theme.Spacing.md)
            .padding(.vertical, Theme.Spacing.sm)

            Divider()

            if traces.isEmpty {
                VStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "waveform")
                        .foregroundStyle(.tertiary)
                    Text("Waiting for requests…")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(traces.reversed()) { trace in
                            RequestRailRow(
                                trace: trace,
                                isSelected: selectedTraceID == trace.id
                            ) {
                                onSelect(trace.id)
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
