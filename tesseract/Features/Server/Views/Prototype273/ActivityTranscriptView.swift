//
//  ActivityTranscriptView.swift
//  tesseract
//
//  PROTOTYPE (wayfinder #273) — THROWAWAY. The Activity page's centerpiece:
//  one request as a full exchange — per-request header, the inbound
//  message list collapsed by default, and the response span stream in the
//  chat's full grammar. Scrolling lifts the chat's 4-rule contract
//  (`ChatTranscriptView`): follow growth at the bottom, disengage the
//  moment the user scrolls up, re-arm when they return.
//

import SwiftUI
import Textual

struct ActivityTranscriptView: View {
    let trace: RequestTrace
    let now: Date

    /// Per-trace raw fallback (the PagePanel `Rendered · Raw` pattern, not
    /// the agent's markdown toggle). State resets with the transcript's
    /// `.id(trace.id)` — each request starts rendered.
    @State private var showsRaw = false

    @State private var scrollPosition = ScrollPosition()
    @State private var autoFollow = true

    var body: some View {
        VStack(spacing: 0) {
            ActivityRequestHeader(trace: trace, now: now, showsRaw: $showsRaw)

            Divider()

            if showsRaw {
                StreamingSpanListView(trace: trace)
            } else {
                renderedExchange
            }
        }
    }

    // MARK: - Rendered exchange

    private var renderedExchange: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
                InboundSection(trace: trace)

                Divider()

                if trace.spans.isEmpty {
                    ActivityPhaseHint(trace: trace)
                } else {
                    let liveSpanID = trace.phase == .decoding ? trace.spans.last?.id : nil
                    ForEach(trace.spans) { span in
                        ServerSpanRow(span: span, isLive: span.id == liveSpanID)
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .frame(maxWidth: ChatLayout.columnMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .scrollPosition($scrollPosition, anchor: .bottom)
        .defaultScrollAnchor(trace.isActive ? .bottom : .top, for: .initialOffset)
        // The auto-follow gate (chat rule 4): re-arms at the bottom, disarms
        // only on a *user* scroll away.
        .onScrollGeometryChange(for: Bool.self) { geo in
            geo.contentSize.height > 0
                && geo.visibleRect.maxY >= geo.contentSize.height - 80
        } action: { _, nearBottom in
            if nearBottom {
                autoFollow = true
            } else if scrollPosition.isPositionedByUser {
                autoFollow = false
            }
        }
        // The follow engine (chat rule 2): measured content growth pins back
        // to the bottom while the gate is armed.
        .onScrollGeometryChange(for: CGFloat.self) { geo in
            geo.contentSize.height
        } action: { oldHeight, newHeight in
            if autoFollow, newHeight > oldHeight {
                scrollPosition.scrollTo(edge: .bottom)
            }
        }
        // End of the request (chat rule 3): settle at the absolute bottom.
        .onChange(of: trace.phase.isTerminal) { _, terminal in
            if terminal, autoFollow {
                withAnimation(.smooth) {
                    scrollPosition.scrollTo(edge: .bottom)
                }
            }
        }
        // Fresh scroll state per trace (chat rule 1's analog: a new or
        // newly-selected request lands at its natural anchor).
        .id(trace.id)
    }
}

// MARK: - Request header

/// The per-request header: model + phase + finish on the first line,
/// cache outcome / TTFT / rate metrics on the monospace meta line, the
/// decode sparkline riding the right edge — the Dashboard hero grammar
/// compressed to one request's vitals.
private struct ActivityRequestHeader: View {
    let trace: RequestTrace
    let now: Date
    @Binding var showsRaw: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.sm) {
                Text("#\(trace.sequence)")
                    .font(.callout.monospaced())
                    .foregroundStyle(.tertiary)
                Text(trace.model.isEmpty ? "unknown model" : trace.model)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                phaseBadge
                Spacer(minLength: Theme.Spacing.sm)
                modeSwitch
            }

            Text(promptMetaLine)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)

            Text(decodeMetaLine)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.md)
        .contextMenu {
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(trace.concatenatedText, forType: .string)
            } label: {
                Label("Copy Output", systemImage: "doc.on.doc")
            }
            .disabled(trace.concatenatedText.isEmpty)

            Button {
                NSWorkspace.shared.open(HTTPRequestLogger.shared.directoryURL)
            } label: {
                Label("Reveal Raw Request File", systemImage: "doc.text.magnifyingglass")
            }
        }
    }

    @ViewBuilder
    private var phaseBadge: some View {
        switch trace.phase {
        case .queued, .lookingUp, .prefilling, .decoding:
            HStack(spacing: 5) {
                Circle().fill(.green).frame(width: 6, height: 6)
                Text(phaseLabel)
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.green)
        case .completed:
            Text(phaseLabel)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        case .failed:
            Text(phaseLabel)
                .font(.caption.weight(.semibold))
                .foregroundStyle(DynamicColor.chatError)
        case .cancelled:
            Text(phaseLabel)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        }
    }

    private var phaseLabel: String {
        switch trace.phase {
        case .queued: "queued"
        case .lookingUp: "lookup"
        case .prefilling: "prefilling"
        case .decoding: "decoding"
        case .completed: "completed"
        case .failed: "failed"
        case .cancelled: "cancelled"
        }
    }

    /// The quiet `Rendered · Raw` text switch (the PagePanel pattern).
    private var modeSwitch: some View {
        HStack(spacing: 4) {
            modeButton("Rendered", isActive: !showsRaw) { showsRaw = false }
            Text("·").foregroundStyle(.quaternary)
            modeButton("Raw", isActive: showsRaw) { showsRaw = true }
        }
        .font(.callout)
    }

    private func modeButton(
        _ label: String, isActive: Bool, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(label)
                .foregroundStyle(isActive ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(label == "Raw" ? "Show the exact captured spans" : "Render the response as markdown")
    }

    /// The prompt side, exact — no mental math: cached / prompt tokens, the
    /// exact prefill token count, and the measured prefill rate.
    private var promptMetaLine: String {
        var parts: [String] = []
        if let promptTokens = trace.promptTokens, promptTokens > 0 {
            let ratio = Double(trace.cachedTokens) / Double(promptTokens)
            parts.append(String(format: "cache %.0f%%", ratio * 100))
            parts.append(
                "\(trace.cachedTokens.formatted())/\(promptTokens.formatted()) tok")
        } else if let reason = trace.cacheReason {
            parts.append("cache \(reason)")
        }
        if let newTokens = trace.newTokensToPrefill {
            if let rate = trace.prefillTokensPerSecond {
                parts.append(
                    String(
                        format: "prefill %@ tok @ %.0f tok/s",
                        newTokens.formatted(), rate))
            } else if trace.phase == .prefilling {
                parts.append("prefilling \(newTokens.formatted()) tok…")
            } else {
                parts.append("prefill \(newTokens.formatted()) tok")
            }
        }
        if parts.isEmpty {
            parts.append("waiting for cache lookup")
        }
        return parts.joined(separator: " · ")
    }

    /// The decode side: first-token latency, decode rate, output size, finish.
    private var decodeMetaLine: String {
        var parts: [String] = []
        if let ttft = trace.ttftMs {
            parts.append("ttft " + PromptCacheFormatting.milliseconds(ttft))
        }
        if trace.phase.isTerminal, trace.tokensPerSecond > 0 {
            parts.append(String(format: "%.1f tok/s", trace.tokensPerSecond))
        } else if let live = trace.liveTokensPerSecond(at: now) {
            parts.append(String(format: "%.0f tok/s live", live))
        }
        if trace.displayOutputTokens > 0 {
            parts.append("\(trace.displayOutputTokens.formatted()) out")
        }
        if trace.phase == .failed, let error = trace.errorMessage {
            parts.append("failed: \(error)")
        } else if let reason = trace.finishReason, trace.phase == .completed {
            parts.append(reason)
        }
        if parts.isEmpty {
            parts.append("waiting for first token")
        }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Inbound section

/// The request's inbound message list, collapsed by default behind a
/// one-line disclosure (#272). Reads the real capture attached to the
/// trace at `startRequest` (#274); when the per-trace budget dropped
/// middle messages, the gap is shown in place, never papered over.
private struct InboundSection: View {
    let trace: RequestTrace

    @State private var isExpanded = false

    private var inbound: RequestTrace.InboundCapture { trace.inbound }

    var body: some View {
        VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    CollapseMarker(isExpanded: isExpanded)
                    Text("Inbound")
                        .font(.system(size: chatBodyFontSize, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text(summaryLine)
                        .font(.system(size: chatBodyFontSize))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isExpanded ? "Hide the request's messages" : "Show the request's messages")

            if isExpanded {
                Group {
                    if inbound.isEmpty {
                        Text("The request carried no messages.")
                            .font(.system(size: chatBodyFontSize))
                            .foregroundStyle(.tertiary)
                    } else {
                        VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
                            ForEach(Array(inbound.messages.enumerated()), id: \.element.id) {
                                index, message in
                                if index == 1, inbound.elidedMessages > 0 {
                                    Text("… \(inbound.elidedMessages) messages elided …")
                                        .font(.system(size: chatBodyFontSize))
                                        .foregroundStyle(.tertiary)
                                }
                                InboundMessageBlock(message: message)
                            }
                        }
                    }
                }
                .padding(.leading, ChatLayout.markerWidth + 8)
            }
        }
    }

    private var summaryLine: String {
        let count = inbound.messages.count + inbound.elidedMessages
        guard count > 0 else { return "no messages" }
        let tokens: String
        if let promptTokens = trace.promptTokens, promptTokens > 0 {
            tokens = "\(promptTokens.formatted()) tokens"
        } else {
            tokens = "~\(inbound.estimatedTokens.formatted()) tokens"
        }
        return "\(count) \(count == 1 ? "message" : "messages") · \(tokens)"
    }
}

/// One inbound message: role label over the content in a quiet quinary box —
/// the chat's user-block grammar, muted to telemetry weight.
private struct InboundMessageBlock: View {
    let message: RequestTrace.InboundMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(message.role)
                .font(.system(size: chatBodyFontSize, weight: .medium))
                .foregroundStyle(.tertiary)
            Text(message.content.chatDisplayTrimmed)
                .font(.system(size: chatBodyFontSize, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(12)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
        }
    }
}

// MARK: - Phase hint

/// Pre-token hint in the transcript's own type, not the raw view's
/// monospace — the exchange column stays one surface.
private struct ActivityPhaseHint: View {
    let trace: RequestTrace

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            CollapseMarker(isExpanded: false)
                .hidden()
                .overlay {
                    if trace.isActive {
                        ProgressView()
                            .controlSize(.mini)
                    }
                }
            Text(hintTitle)
                .font(.system(size: chatBodyFontSize, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
    }

    private var hintTitle: String {
        switch trace.phase {
        case .queued: "Waiting for inference lease…"
        case .lookingUp: "Looking up prefix cache…"
        case .prefilling: "Reading context…"
        case .decoding: "First tokens are on the way…"
        case .completed: "No output produced"
        case .failed: "Generation failed before any tokens"
        case .cancelled: "Cancelled before any tokens"
        }
    }
}
