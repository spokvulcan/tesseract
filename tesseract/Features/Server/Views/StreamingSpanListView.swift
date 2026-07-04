import SwiftUI

/// Passive monospace span list for a single `RequestTrace`.
/// This view intentionally avoids lazy stacks, scroll readers, geometry
/// callbacks, timeline ticks, and programmatic scrolling; it is rendered from
/// the current trace snapshot only. Follow-the-stream behavior comes from the
/// declarative `defaultScrollAnchor(.bottom, for: .sizeChanges)` — the system
/// keeps the bottom pinned only while the user is already there.
struct StreamingSpanListView: View {
    private static let maxRenderedSpans = 200

    let trace: RequestTrace

    private var hiddenSpanCount: Int {
        max(0, trace.spans.count - Self.maxRenderedSpans)
    }

    private var displayedSpans: ArraySlice<RequestTrace.Span> {
        trace.spans.suffix(Self.maxRenderedSpans)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                spanContents
            }
            .padding(Theme.Spacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .defaultScrollAnchor(trace.isActive ? .bottom : .top)
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
        .background(.background)
        // Fresh scroll state per trace so the initial anchor applies when
        // the selection changes, instead of inheriting the previous trace's
        // scroll offset.
        .id(trace.id)
    }

    @ViewBuilder
    private var spanContents: some View {
        if trace.spans.isEmpty {
            PhaseHint(trace: trace)
        } else {
            if hiddenSpanCount > 0 {
                OmittedSpanNotice(count: hiddenSpanCount)
            }

            let liveSpanID = trace.phase == .decoding ? displayedSpans.last?.id : nil
            ForEach(displayedSpans) { span in
                SpanView(span: span, isLive: span.id == liveSpanID)
            }
        }
    }
}

// MARK: - Span view

private struct SpanView: View {
    let span: RequestTrace.Span
    var isLive: Bool = false

    var body: some View {
        switch span {
        case .text(_, let content):
            if isLive {
                LiveStreamingText(content: content, isThinking: false)
            } else {
                Text(content)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

        case .thinking(_, let content):
            VStack(alignment: .leading, spacing: 2) {
                Text("<think>")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                Group {
                    if isLive {
                        LiveStreamingText(content: content, isThinking: true)
                    } else {
                        Text(content)
                            .font(.system(.body, design: .monospaced).italic())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.leading, 8)
                if !isLive {
                    Text("</think>")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case .toolCall(_, let name, let argumentsJSON):
            ToolCallBox(name: name, argumentsJSON: argumentsJSON, malformed: false)

        case .toolCallBuilding(_, let name, let argumentsJSON):
            // Visually identical to `.toolCall` — the only difference is
            // that content streams in as `.toolCallDelta` events arrive.
            // On `</tool_call>` close the span transitions to `.toolCall`
            // (or `.malformedToolCall` on parse failure) with the same
            // span id, so SwiftUI preserves position and styling.
            ToolCallBox(
                name: name.isEmpty ? "…" : name,
                argumentsJSON: argumentsJSON,
                malformed: false
            )

        case .malformedToolCall(_, let raw):
            ToolCallBox(name: nil, argumentsJSON: raw, malformed: true)
        }
    }
}

// MARK: - Live streaming text

/// Streaming text split at the last newline: the stable prefix `Text`
/// receives an unchanged string on most coalesced flushes, so SwiftUI skips
/// its (large, monospaced) re-layout — only the short live tail lays out per
/// flush. The split is at a paragraph boundary, so stacking the two `Text`s
/// at zero spacing renders identically to one combined `Text`.
private struct LiveStreamingText: View {
    let content: String
    let isThinking: Bool

    var body: some View {
        let split = Self.splitAtLastNewline(content)
        VStack(alignment: .leading, spacing: 0) {
            if !split.stable.isEmpty {
                styled(Text(split.stable))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            styled(Text(split.live) + Text("▍").foregroundColor(.green))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func styled(_ text: Text) -> Text {
        if isThinking {
            return
                text
                .font(.system(.body, design: .monospaced).italic())
                .foregroundColor(.secondary)
        }
        return
            text
            .font(.system(.body, design: .monospaced))
    }

    static func splitAtLastNewline(_ content: String) -> (stable: String, live: String) {
        guard let idx = content.lastIndex(of: "\n") else { return ("", content) }
        return (
            stable: String(content[..<idx]),
            live: String(content[content.index(after: idx)...])
        )
    }
}

// MARK: - Tool call box

private struct ToolCallBox: View {
    let name: String?
    let argumentsJSON: String
    let malformed: Bool

    private var tint: Color { malformed ? .red : .orange }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                if !malformed {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.caption2)
                        .foregroundStyle(tint)
                }
                Text(malformed ? "malformed tool_call" : "tool_call")
                    .font(.caption2.monospaced())
                    .foregroundStyle(tint)
                if let name {
                    Text(name)
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(tint)
                }
            }
            Text(argumentsJSON)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .padding(.leading, 8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(6)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(tint.opacity(0.06))
        )
    }
}

// MARK: - Omitted span notice

private struct OmittedSpanNotice: View {
    let count: Int

    var body: some View {
        Text("\(count) older spans hidden")
            .font(.caption.monospaced())
            .foregroundStyle(.tertiary)
            .padding(.vertical, Theme.Spacing.xs)
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
            if trace.cachedTokens == 0, let promptTokens = trace.promptTokens {
                return "Cache miss; processing \(promptTokens) prompt tokens."
            }
            if let newTokens = trace.newTokensToPrefill {
                return
                    "Processing \(newTokens) uncached prompt tokens after reusing \(trace.cachedTokens)."
            }
            return
                "Processing the prompt through the model. This can take a while on cold caches and long contexts."
        case .decoding:
            return "First tokens are on the way."
        default:
            return ""
        }
    }
}
