import SwiftUI

/// Compact, one-line row for the request rail. Shows status LED, sequence
/// number, timestamp, cache indicator, and a phase-dependent right-side metric.
struct RequestTraceRow: View {
    let trace: RequestTrace

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            StatusLED(phase: trace.phase)
                .frame(width: 8, height: 8)

            Text("#\(trace.sequence)")
                .font(.system(.caption, design: .monospaced).weight(.medium))
                .foregroundStyle(.primary)

            Text(formattedStartTime)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)

            if trace.cachedTokens > 0 {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                    .help("Prefix cache hit — \(trace.cachedTokens) tokens reused")
            }

            Spacer()

            rightMetric
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    private var formattedStartTime: String {
        Self.timeFormatter.string(from: trace.startedAt)
    }

    @ViewBuilder
    private var rightMetric: some View {
        switch trace.phase {
        case .queued:
            Text("queued")
                .foregroundStyle(.secondary)
        case .lookingUp:
            Text("lookup")
                .foregroundStyle(.indigo)
        case .prefilling:
            Text("prefill")
                .foregroundStyle(.blue)
        case .decoding:
            if trace.tokensPerSecond > 0 {
                Text(String(format: "%.0f t/s", trace.tokensPerSecond))
                    .foregroundStyle(.green)
            } else {
                Text("decoding…")
                    .foregroundStyle(.green)
            }
        case .completed:
            Text(trace.finishReason ?? "done")
                .foregroundStyle(.secondary)
        case .failed:
            Text("failed")
                .foregroundStyle(.red)
        case .cancelled:
            Text("cancelled")
                .foregroundStyle(.orange)
        }
    }
}

// MARK: - Status LED

struct StatusLED: View {
    let phase: RequestTrace.Phase

    var body: some View {
        Circle()
            .fill(color)
            .overlay(
                Circle().stroke(color.opacity(0.5), lineWidth: 1)
            )
    }

    private var color: Color {
        switch phase {
        case .queued: return .gray
        case .lookingUp: return .indigo
        case .prefilling: return .blue
        case .decoding: return .green
        case .completed: return .green.opacity(0.6)
        case .failed: return .red
        case .cancelled: return .orange
        }
    }
}
