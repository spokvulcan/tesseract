import SwiftUI

/// The telemetry face of the Server dashboard: four large live numbers, a
/// decode-rate sparkline, and one monospace meta line of diagnostics.
/// Plain content — deliberately no glass, cards, or tiles (HIG: Liquid Glass
/// does not belong in the content layer).
struct ServerHeroBand: View {
    let trace: RequestTrace?
    let now: Date

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            HStack(alignment: .center, spacing: Theme.Spacing.xxl * 1.5) {
                HeroNumber(
                    value: rateValue,
                    unit: nil,
                    label: rateLabel,
                    isLive: trace?.phase == .decoding
                )
                HeroNumber(
                    value: outputValue,
                    unit: nil,
                    label: "tokens out",
                    isLive: false
                )
                HeroNumber(
                    value: ttftValue.0,
                    unit: ttftValue.1,
                    label: "first token",
                    isLive: false
                )
                HeroNumber(
                    value: cacheValue.0,
                    unit: cacheValue.1,
                    label: "cache hit",
                    isLive: false
                )

                Spacer(minLength: 0)

                RateSparkline(samples: trace?.rateSamples ?? [])
                    .frame(width: 148, height: 40)
            }

            Text(metaLine)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.top, Theme.Spacing.lg)
        .padding(.bottom, Theme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Values

    private var rateValue: String {
        guard let trace else { return "—" }
        if trace.phase.isTerminal, trace.tokensPerSecond > 0 {
            return String(format: "%.1f", trace.tokensPerSecond)
        }
        if let live = trace.liveTokensPerSecond(at: now) {
            return String(format: "%.0f", live)
        }
        return "—"
    }

    private var rateLabel: String {
        guard let trace else { return "tok / s" }
        switch trace.phase {
        case .decoding: return "live tok / s"
        case .queued: return "tok / s · queued"
        case .lookingUp: return "tok / s · lookup"
        case .prefilling: return "tok / s · prefill"
        case .completed, .failed:
            return "tok / s"
        case .cancelled:
            return trace.tokensPerSecond > 0 ? "tok / s" : "tok / s · est"
        }
    }

    private var outputValue: String {
        guard let trace else { return "—" }
        return trace.displayOutputTokens.formatted(.number.grouping(.automatic))
    }

    private var ttftValue: (String, String?) {
        guard let ttft = trace?.ttftMs else { return ("—", nil) }
        if ttft < 1000 { return (String(format: "%.0f", ttft), "ms") }
        return (String(format: "%.2f", ttft / 1000), "s")
    }

    private var cacheValue: (String, String?) {
        guard let trace, let promptTokens = trace.promptTokens, promptTokens > 0 else {
            return ("—", nil)
        }
        let ratio = Double(trace.cachedTokens) / Double(promptTokens)
        return (String(format: "%.1f", ratio * 100), "%")
    }

    private var metaLine: String {
        guard let trace else {
            return "waiting for the first request to /v1/chat/completions"
        }
        var parts: [String] = ["#\(trace.sequence)"]
        if !trace.model.isEmpty { parts.append(trace.model) }
        if let rate = trace.prefillTokensPerSecond {
            parts.append(String(format: "prefill %.0f tok/s", rate))
        } else if trace.phase == .prefilling {
            parts.append("prefilling…")
        }
        if let promptTokens = trace.promptTokens, promptTokens > 0 {
            parts.append(
                "cached \(trace.cachedTokens.formatted())/\(promptTokens.formatted())")
        }
        if let lookup = trace.lookupMs {
            parts.append(String(format: "lookup %.1f ms", lookup))
        }
        if let restore = trace.restoreMs, restore > 0 {
            parts.append(String(format: "restore %.0f ms", restore))
        }
        if trace.phase == .failed, let error = trace.errorMessage {
            parts.append("failed: \(error)")
        } else if trace.phase == .cancelled {
            parts.append("cancelled")
        } else if let reason = trace.finishReason, trace.phase == .completed {
            parts.append(reason)
        }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Hero number

private struct HeroNumber: View {
    let value: String
    let unit: String?
    let label: String
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: 32, weight: .bold))
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .foregroundStyle(.primary)
                if let unit {
                    Text(unit)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 5) {
                if isLive {
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                }
                Text(label)
                    .font(.caption2.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(0.8)
                    .foregroundStyle(isLive ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary))
            }
        }
    }
}

// MARK: - Sparkline

/// Minimal decode-rate sparkline: 2 pt line, emphasized endpoint, no axes.
struct RateSparkline: View {
    let samples: [Double]

    var body: some View {
        Canvas { context, size in
            guard samples.count >= 2 else { return }
            let maxValue = max(samples.max() ?? 1, 1)
            let stepX = size.width / CGFloat(samples.count - 1)
            let points = samples.enumerated().map { index, value in
                CGPoint(
                    x: CGFloat(index) * stepX,
                    y: size.height - (CGFloat(value / maxValue) * (size.height - 6)) - 3
                )
            }
            var path = Path()
            path.addLines(points)
            context.stroke(
                path,
                with: .color(.green),
                style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round)
            )
            if let last = points.last {
                context.fill(
                    Path(ellipseIn: CGRect(x: last.x - 3, y: last.y - 3, width: 6, height: 6)),
                    with: .color(.green)
                )
            }
        }
        .accessibilityLabel("Decode rate over time")
    }
}
