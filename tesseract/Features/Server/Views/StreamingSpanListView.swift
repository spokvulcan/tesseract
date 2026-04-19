import SwiftUI

/// Monospace, scrolling span list for a single `RequestTrace`. Token-by-token
/// updates flow through `ServerGenerationLog.streamingVersion`, which triggers
/// a scroll-to-bottom only when auto-scroll is enabled and we are already
/// near the bottom (mirrors `AgentConversationListView`'s pattern).
struct StreamingSpanListView: View {
    let trace: RequestTrace
    let isAutoScrollEnabled: Bool

    @Environment(ServerGenerationLog.self) private var log
    @State private var isNearBottom: Bool = true

    private var bottomAnchorID: String { "trace-bottom-\(trace.id)" }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                spanRows
                .padding(Theme.Spacing.md)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor).opacity(0.6))
            .defaultScrollAnchor(.bottom)
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.contentSize.height > 0
                && geo.visibleRect.maxY >= geo.contentSize.height - 80
            } action: { _, nearBottom in
                isNearBottom = nearBottom
            }
            .overlay {
                StreamingScrollTrigger(
                    anchorID: bottomAnchorID,
                    proxy: proxy,
                    isAutoScrollEnabled: isAutoScrollEnabled,
                    isNearBottom: isNearBottom
                )
            }
        }
    }

    /// The same AppKit trap that shows up in the agent chat can be triggered
    /// here if SwiftUI lazy-prefetch runs while spans are still streaming and
    /// auto-scroll is nudging the view. Use an eager stack only for active
    /// traces; completed traces keep lazy loading.
    @ViewBuilder
    private var spanRows: some View {
        if trace.isActive {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                spanRowContents
            }
        } else {
            LazyVStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                spanRowContents
            }
        }
    }

    @ViewBuilder
    private var spanRowContents: some View {
        if trace.spans.isEmpty {
            PhaseHint(trace: trace)
        } else {
            ForEach(trace.spans) { span in
                SpanView(span: span)
            }
        }

        if trace.phase == .decoding {
            BlinkingCursor()
        }

        Color.clear
            .frame(height: 1)
            .id(bottomAnchorID)
    }
}

// MARK: - Span view

private struct SpanView: View {
    let span: RequestTrace.Span

    var body: some View {
        switch span {
        case .text(_, let content):
            Text(content)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

        case .thinking(_, let content):
            VStack(alignment: .leading, spacing: 2) {
                Text("<think>")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                Text(content)
                    .font(.system(.body, design: .monospaced).italic())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 8)
                Text("</think>")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case .toolCall(_, let name, let argumentsJSON):
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("tool_call")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.orange)
                    Text(name)
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(.orange)
                }
                Text(argumentsJSON)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.orange.opacity(0.06))
            )

        case .toolCallBuilding(_, let name, let argumentsJSON):
            // Visually identical to `.toolCall` — the only difference is
            // that content streams in as `.toolCallDelta` events arrive.
            // On `</tool_call>` close the span transitions to `.toolCall`
            // (or `.malformedToolCall` on parse failure) with the same
            // span id, so SwiftUI preserves position and styling.
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("tool_call")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.orange)
                    Text(name.isEmpty ? "…" : name)
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(.orange)
                }
                Text(argumentsJSON)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.orange.opacity(0.06))
            )

        case .malformedToolCall(_, let raw):
            VStack(alignment: .leading, spacing: 2) {
                Text("malformed tool_call")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.red)
                Text(raw)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.red.opacity(0.06))
            )
        }
    }
}

// MARK: - Phase hint

private struct PhaseHint: View {
    let trace: RequestTrace

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            Text(hintTitle)
                .font(.system(.callout, design: .monospaced).weight(.medium))
                .foregroundStyle(.secondary)
            Text(hintSubtitle)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, Theme.Spacing.md)
    }

    private var hintTitle: String {
        switch trace.phase {
        case .queued: return "Waiting for inference lease…"
        case .lookingUp: return "Looking up prefix cache…"
        case .prefilling: return "Prefilling prompt…"
        case .decoding: return "Decoding…"
        case .completed: return "(no output produced)"
        case .failed: return "Generation failed before any tokens"
        case .cancelled: return "Cancelled before any tokens"
        }
    }

    private var hintSubtitle: String {
        switch trace.phase {
        case .queued:
            return "Another request is using the model."
        case .lookingUp:
            return "Scanning radix tree for a reusable KV snapshot."
        case .prefilling:
            return "Processing the prompt through the model. This can take a while on cold caches and long contexts."
        case .decoding:
            return "First tokens are on the way."
        default:
            return ""
        }
    }
}

// MARK: - Blinking cursor

private struct BlinkingCursor: View {
    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { context in
            let visible = Int(context.date.timeIntervalSinceReferenceDate * 2) % 2 == 0
            Rectangle()
                .fill(Color.primary)
                .frame(width: 8, height: 14)
                .opacity(visible ? 0.7 : 0.0)
        }
    }
}

// MARK: - Streaming scroll trigger

/// Reads only `streamingVersion` from the log, then scrolls to the bottom
/// anchor if auto-scroll is enabled and the user was already near the
/// bottom. Isolated so the parent view body doesn't re-diff on every bump.
private struct StreamingScrollTrigger: View {
    let anchorID: String
    let proxy: ScrollViewProxy
    let isAutoScrollEnabled: Bool
    let isNearBottom: Bool

    @Environment(ServerGenerationLog.self) private var log
    @State private var pendingScrollTask: Task<Void, Never>?

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: log.streamingVersion) { _, _ in
                guard isAutoScrollEnabled, isNearBottom else { return }
                scheduleScrollToBottom()
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
    }

    /// Defers the scroll until the current layout pass completes. Calling
    /// `scrollTo` inline while SwiftUI is reconciling a `LazyVStack` can
    /// re-enter AppKit constraint updates and trip the Release-only crash
    /// guarded elsewhere in the app's streaming views.
    private func scheduleScrollToBottom() {
        pendingScrollTask?.cancel()
        pendingScrollTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            proxy.scrollTo(anchorID, anchor: .bottom)
        }
    }
}
