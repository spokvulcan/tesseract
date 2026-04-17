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
                .background(Color(nsColor: .textBackgroundColor).opacity(0.5))

            Divider()

            if let selected = selectedTrace {
                ActiveTraceDetailView(trace: selected)
                    .frame(maxWidth: .infinity)
            } else {
                EmptyDetailPlaceholder()
            }
        }
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.medium, style: .continuous)
                .fill(Color(nsColor: .underPageBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.medium, style: .continuous)
                .strokeBorder(.separator, lineWidth: 0.5)
        )
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
        @Bindable var log = log

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
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
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
                List(selection: $log.selectedTraceID) {
                    // Newest first for easy scanning.
                    ForEach(log.traces.reversed()) { trace in
                        RequestTraceRow(trace: trace)
                            .tag(trace.id as UUID?)
                            .listRowInsets(EdgeInsets(
                                top: 4, leading: 8, bottom: 4, trailing: 8
                            ))
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            }
        }
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
