import SwiftUI

/// Passive monospace span list for a single `RequestTrace`.
/// This view intentionally avoids lazy stacks, scroll readers, geometry
/// callbacks, timeline ticks, and programmatic scrolling; it is rendered from
/// the current trace snapshot only.
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
        .background(.background)
    }

    @ViewBuilder
    private var spanContents: some View {
        if trace.spans.isEmpty {
            PhaseHint(trace: trace)
        } else {
            if hiddenSpanCount > 0 {
                OmittedSpanNotice(count: hiddenSpanCount)
            }

            ForEach(displayedSpans) { span in
                SpanView(span: span)
            }
        }

        if trace.phase == .decoding {
            DecodingMarker()
        }
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
                .frame(maxWidth: .infinity, alignment: .leading)

        case .thinking(_, let content):
            VStack(alignment: .leading, spacing: 2) {
                Text("<think>")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                Text(content)
                    .font(.system(.body, design: .monospaced).italic())
                    .foregroundStyle(.secondary)
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
            return "Processing the prompt through the model. This can take a while on cold caches and long contexts."
        case .decoding:
            return "First tokens are on the way."
        default:
            return ""
        }
    }
}

// MARK: - Decoding marker

private struct DecodingMarker: View {
    var body: some View {
        Text("decoding...")
            .font(.caption.monospaced())
            .foregroundStyle(.tertiary)
            .padding(.top, Theme.Spacing.xs)
    }
}
