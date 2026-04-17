import AppKit
import SwiftUI

/// Detail pane for the currently-selected request. Header + streaming span
/// list + footer controls, all monospace.
struct ActiveTraceDetailView: View {
    let trace: RequestTrace

    @State private var isAutoScrollEnabled: Bool = true
    @State private var showDiagnosticsDetail: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
                .background(.thinMaterial)

            Divider()

            StreamingSpanListView(
                trace: trace,
                isAutoScrollEnabled: isAutoScrollEnabled
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            footer
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.sm)
                .background(.thinMaterial)
        }
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(spacing: Theme.Spacing.sm) {
                PhaseBadge(trace: trace)

                Text(shortCompletionID)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .help(trace.completionID)

                if !trace.model.isEmpty {
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(trace.model)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Text(trace.stream ? "streaming" : "non-streaming")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            // Metrics strip — compact, monospace.
            HStack(spacing: Theme.Spacing.md) {
                MetricCell(label: "cached",
                           value: "\(trace.cachedTokens)")
                MetricCell(label: "prompt",
                           value: formatOpt(trace.promptTokens))
                MetricCell(label: "out",
                           value: "\(trace.generationTokens)")
                MetricCell(label: "tok/s",
                           value: trace.tokensPerSecond > 0
                           ? String(format: "%.1f", trace.tokensPerSecond)
                           : "—")
                MetricCell(label: "TTFT",
                           value: trace.ttftMs.map {
                               String(format: "%.0fms", $0)
                           } ?? "—")
                Spacer()
                Button {
                    showDiagnosticsDetail.toggle()
                } label: {
                    Image(systemName: showDiagnosticsDetail
                          ? "chevron.up.circle"
                          : "chevron.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Toggle diagnostics detail")
            }

            if showDiagnosticsDetail {
                HStack(spacing: Theme.Spacing.md) {
                    MetricCell(label: "lookup",
                               value: formatOptMs(trace.lookupMs))
                    MetricCell(label: "restore",
                               value: formatOptMs(trace.restoreMs))
                    MetricCell(label: "prefill",
                               value: formatOptMs(trace.prefillMs))
                    MetricCell(label: "shared",
                               value: "\(trace.sharedPrefixLength)")
                    Spacer()
                    if let reason = trace.cacheReason {
                        Text(reason)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.easeInOut(duration: 0.15), value: showDiagnosticsDetail)
    }

    // MARK: - Footer

    @ViewBuilder
    private var footer: some View {
        HStack(spacing: Theme.Spacing.md) {
            Toggle(isOn: $isAutoScrollEnabled) {
                Label("Auto-scroll", systemImage: isAutoScrollEnabled
                      ? "arrow.down.to.line"
                      : "pause.rectangle")
                    .labelStyle(.iconOnly)
            }
            .toggleStyle(.button)
            .help(isAutoScrollEnabled
                  ? "Auto-scroll on — toggle to pin scroll"
                  : "Auto-scroll paused")

            Button {
                copyOutput()
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help("Copy generated text")

            Button {
                revealRawRequest()
            } label: {
                Label("Raw JSON", systemImage: "doc.text.magnifyingglass")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help("Reveal raw request JSON in Finder")

            Spacer()

            if let error = trace.errorMessage, trace.phase == .failed {
                Text(error)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .truncationMode(.middle)
            } else {
                Text(statusSummary)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Helpers

    private var shortCompletionID: String {
        let tail = trace.completionID.suffix(12)
        return String(tail)
    }

    private var statusSummary: String {
        switch trace.phase {
        case .queued: return "queued"
        case .lookingUp: return "cache lookup"
        case .prefilling: return "prefill"
        case .decoding:
            return "\(trace.generationTokens) tokens"
        case .completed:
            return "\(trace.generationTokens) tokens in \(String(format: "%.1fs", trace.elapsedFromStart))"
        case .failed: return "failed"
        case .cancelled: return "cancelled"
        }
    }

    private func formatOpt(_ v: Int?) -> String {
        v.map { "\($0)" } ?? "—"
    }

    private func formatOptMs(_ v: Double?) -> String {
        guard let v else { return "—" }
        if v < 1 { return String(format: "%.1fms", v) }
        return String(format: "%.0fms", v)
    }

    private func copyOutput() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(trace.concatenatedText, forType: .string)
    }

    private func revealRawRequest() {
        NSWorkspace.shared.open(HTTPRequestLogger.shared.directoryURL)
    }
}

// MARK: - Phase badge

struct PhaseBadge: View {
    let trace: RequestTrace

    var body: some View {
        HStack(spacing: 4) {
            StatusLED(phase: trace.phase).frame(width: 6, height: 6)
            Text(label)
                .font(.caption2.weight(.semibold).monospaced())
                .foregroundStyle(textColor)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.pill, style: .continuous)
                .fill(bgColor.opacity(0.15))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.pill, style: .continuous)
                .strokeBorder(bgColor.opacity(0.4), lineWidth: 0.5)
        )
    }

    private var label: String {
        switch trace.phase {
        case .queued: return "QUEUED"
        case .lookingUp: return "LOOKUP"
        case .prefilling: return "PREFILL"
        case .decoding: return "DECODE"
        case .completed: return "DONE"
        case .failed: return "FAILED"
        case .cancelled: return "CANCEL"
        }
    }

    private var bgColor: Color {
        switch trace.phase {
        case .queued: return .gray
        case .lookingUp: return .indigo
        case .prefilling: return .blue
        case .decoding: return .green
        case .completed: return .green
        case .failed: return .red
        case .cancelled: return .orange
        }
    }

    private var textColor: Color {
        switch trace.phase {
        case .completed: return .secondary
        default: return bgColor
        }
    }
}

// MARK: - Metric cell

struct MetricCell: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label)
                .font(.system(size: 9, design: .monospaced).weight(.medium))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }
}
