import SwiftUI

/// Slide-up console for the Server dashboard (⌘`): every request as one
/// monospace log line, diagnostics under the selected one, and a status bar
/// of server vitals. A standard-material panel — no custom glass.
struct ServerConsoleDrawer: View {
    @Environment(ServerGenerationLog.self) private var log
    @Environment(SettingsManager.self) private var settings

    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            feed
            Divider()
            statusBar
        }
        .frame(height: 300)
        .background(.regularMaterial)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Text("Console")
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
                .buttonStyle(.borderless)
                .help("Clear all requests")
            }
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("Close console (⌘`)")
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 6)
    }

    // MARK: - Feed

    @ViewBuilder
    private var feed: some View {
        if log.traces.isEmpty {
            VStack(spacing: Theme.Spacing.xs) {
                Text("No requests yet")
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(log.traces) { trace in
                        ConsoleRequestRow(
                            trace: trace,
                            isSelected: trace.id == log.selectedTraceID,
                            onSelect: { log.selectedTraceID = trace.id },
                            onCancel: { log.requestCancel(traceID: trace.id) }
                        )
                        if trace.id == log.selectedTraceID {
                            ConsoleDiagnosticsLines(trace: trace)
                        }
                    }
                }
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Follow new rows once content overflows, but let a short log
            // sit at the top like a real console.
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
        }
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack(spacing: Theme.Spacing.lg) {
            HStack(spacing: 5) {
                Circle()
                    .fill(.green)
                    .frame(width: 6, height: 6)
                Text("serving :\(String(settings.serverPort))")
            }

            if let rate = liveRate {
                Text(String(format: "%.0f tok/s", rate))
                    .foregroundStyle(.green)
            }

            if let hitRate = sessionCacheHitRate {
                Text(String(format: "cache %.1f%%", hitRate * 100))
            }

            Spacer()

            Text("\(activeCount) active · \(log.traces.count) recent · ⌘. stop")
                .foregroundStyle(.tertiary)
        }
        .font(.caption2.monospaced())
        .monospacedDigit()
        .foregroundStyle(.secondary)
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 6)
    }

    private var activeCount: Int {
        log.traces.count(where: \.isActive)
    }

    private var liveRate: Double? {
        log.traces.last(where: { $0.phase == .decoding })?.liveTokensPerSecond()
    }

    /// Session-wide token reuse across all logged requests with known prompts.
    private var sessionCacheHitRate: Double? {
        let known = log.traces.compactMap { trace -> (Int, Int)? in
            guard let prompt = trace.promptTokens, prompt > 0 else { return nil }
            return (trace.cachedTokens, prompt)
        }
        guard !known.isEmpty else { return nil }
        let cached = known.reduce(0) { $0 + $1.0 }
        let prompt = known.reduce(0) { $0 + $1.1 }
        return Double(cached) / Double(prompt)
    }
}

// MARK: - Request row

private struct ConsoleRequestRow: View {
    let trace: RequestTrace
    let isSelected: Bool
    let onSelect: () -> Void
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(isSelected ? "▾" : "▸")
                .foregroundStyle(.quaternary)
            Text(Self.timeFormatter.string(from: trace.startedAt))
                .foregroundStyle(.tertiary)
            Text("#\(trace.sequence)")
                .foregroundStyle(isSelected ? .primary : .secondary)
            if !trace.model.isEmpty {
                Text(trace.model)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            Text("·").foregroundStyle(.quaternary)
            verdict
            Text("·").foregroundStyle(.quaternary)
            Text("\(trace.displayOutputTokens.formatted()) tok")
                .foregroundStyle(.secondary)
            if let rate = displayRate {
                Text("·").foregroundStyle(.quaternary)
                Text(String(format: "%.0f t/s", rate))
                    .foregroundStyle(trace.phase == .decoding ? .green : .secondary)
            }

            Spacer(minLength: 0)

            if trace.isCancellable, trace.isActive {
                Button(action: onCancel) {
                    Image(systemName: "stop.circle.fill")
                        .foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
                .help("Stop generation")
            }
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.vertical, 2.5)
        .padding(.horizontal, 6)
        .background(
            isSelected ? AnyShapeStyle(.quaternary.opacity(0.6)) : AnyShapeStyle(.clear),
            in: RoundedRectangle(cornerRadius: 5, style: .continuous)
        )
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
    }

    @ViewBuilder
    private var verdict: some View {
        switch trace.phase {
        case .queued: Text("queued").foregroundStyle(.secondary)
        case .lookingUp: Text("lookup").foregroundStyle(.indigo)
        case .prefilling: Text("prefill").foregroundStyle(.blue)
        case .decoding: Text("decoding").foregroundStyle(.green)
        case .completed: Text(trace.finishReason ?? "done").foregroundStyle(.green)
        case .failed: Text("failed").foregroundStyle(.red)
        case .cancelled: Text("cancelled").foregroundStyle(.orange)
        }
    }

    private var displayRate: Double? {
        if trace.phase.isTerminal, trace.tokensPerSecond > 0 { return trace.tokensPerSecond }
        return trace.liveTokensPerSecond()
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}

// MARK: - Diagnostics lines

/// Phase log lines shown under the selected request, console-style.
private struct ConsoleDiagnosticsLines: View {
    let trace: RequestTrace

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            if let lookup = lookupLine {
                line(label: "lookup", text: lookup)
            }
            if let divergence = divergenceLine {
                line(label: "divergence", text: divergence, color: .orange)
            }
            if let prefill = prefillLine {
                line(label: "prefill", text: prefill)
            }
            if let decode = decodeLine {
                line(label: "decode", text: decode)
            }
            if trace.phase == .failed, let error = trace.errorMessage {
                line(label: "error", text: error, color: .red)
            }
        }
        .padding(.leading, 34)
        .padding(.bottom, 4)
    }

    private func line(label: String, text: String, color: Color = .secondary) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .foregroundStyle(color)
            Text(text)
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
        }
        .font(.caption2.monospaced())
        .monospacedDigit()
    }

    private var lookupLine: String? {
        guard trace.lookupMs != nil || trace.cacheReason != nil else { return nil }
        var parts: [String] = []
        if let reason = trace.cacheReason { parts.append(reason) }
        if let prompt = trace.promptTokens, prompt > 0 {
            parts.append("\(trace.cachedTokens.formatted())/\(prompt.formatted())")
        }
        if let lookup = trace.lookupMs { parts.append(String(format: "%.1f ms", lookup)) }
        if let restore = trace.restoreMs, restore > 0 {
            parts.append(String(format: "restore %.0f ms", restore))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Miss attribution (issue #158): shown only for a client-changed
    /// prompt *prefix* — the deep loss an operator would otherwise
    /// misread as a server-side eviction. Routine per-turn tail rewinds
    /// stay silent.
    private var divergenceLine: String? {
        guard let divergence = trace.divergence,
            divergence.indicatesClientPrefixChange
        else { return nil }
        return "prompt changed at token \(divergence.offset.formatted())"
            + " — client prefix change abandoned"
            + " \(divergence.abandonedTokens.formatted()) cached tokens"
    }

    private var prefillLine: String? {
        guard let newTokens = trace.newTokensToPrefill else { return nil }
        var parts = ["\(newTokens.formatted()) tok"]
        if let ms = trace.prefillMs, ms > 0 {
            parts.append(String(format: "%.1f s", ms / 1000))
        }
        if let rate = trace.prefillTokensPerSecond {
            parts.append(String(format: "%.0f tok/s", rate))
        }
        return parts.joined(separator: " · ")
    }

    private var decodeLine: String? {
        guard trace.firstTokenAt != nil else { return nil }
        var parts: [String] = []
        if let ttft = trace.ttftMs {
            parts.append(String(format: "first token %.0f ms", ttft))
        }
        if trace.phase == .decoding {
            parts.append("streaming")
        } else if trace.tokensPerSecond > 0 {
            parts.append(String(format: "%.1f tok/s", trace.tokensPerSecond))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}
