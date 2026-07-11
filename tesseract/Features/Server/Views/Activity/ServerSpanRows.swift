//
//  ServerSpanRows.swift
//  tesseract
//
//  The `RequestTrace.Span` → chat grammar adapter (map #269, seams in
//  #271): server spans rendered through the chat's own markdown styles,
//  prose palette, and tool-panel atoms. Thin server-owned row views over
//  the shared atoms — the chat's session-coupled rows (`ThinkingRowView`,
//  `ToolCallRowView`) are not touched.
//
//  Streaming discipline: one subview per span — committed spans are pure
//  value views SwiftUI skips on deltas; only the live span's view
//  invalidates, re-parsing markdown at most every ~100 ms (the chat's
//  LivePart precedent) even though the store flushes at 30 Hz.
//

import SwiftUI
import Textual

// MARK: - Span dispatch

/// One transcript row per server span, in the chat grammar.
struct ServerSpanRow: View {
    let span: RequestTrace.Span
    let isLive: Bool

    var body: some View {
        switch span {
        case .text(_, let content):
            if isLive {
                ServerLiveProseView(content: content)
            } else {
                ServerProseView(text: content)
            }

        case .thinking(_, let content):
            ServerThinkingRow(text: content, isLive: isLive)

        case .toolCall(_, let name, let argumentsJSON):
            ServerToolCallRow(name: name, argumentsJSON: argumentsJSON, isBuilding: false)

        case .toolCallBuilding(_, let name, let argumentsJSON):
            ServerToolCallRow(name: name, argumentsJSON: argumentsJSON, isBuilding: true)

        case .malformedToolCall(_, let raw):
            ServerMalformedToolCallRow(raw: raw)
        }
    }
}

// MARK: - Prose

/// Server response text in the chat's exact markdown composition
/// (`ChatItemViews.swift` canonical five modifiers), minus the agent's
/// `agentUseMarkdown` gate — raw reachability is the transcript header's
/// per-trace `Rendered · Raw` switch instead (#272).
struct ServerProseView: View {
    let text: String

    var body: some View {
        StructuredText(markdown: text)
            .textual.structuredTextStyle(ChatMarkdownStyle())
            .textual.codeBlockStyle(CopyableCodeBlockStyle())
            .textual.highlighterTheme(.codeAccents)
            .textual.textSelection(.enabled)
            .font(.system(size: chatBodyFontSize))
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The live text span: full markdown, but re-published at most every
/// ~100 ms with a trailing flush — wiring `StructuredText` straight to the
/// 30 Hz span flushes would triple the chat's live parse rate (#271 §7.1).
struct ServerLiveProseView: View {
    let content: String

    @State private var displayed = ""
    @State private var lastPublish = Date.distantPast
    @State private var trailingFlush: Task<Void, Never>?

    private static let throttle: TimeInterval = 0.1

    var body: some View {
        ServerProseView(text: displayed)
            .onAppear {
                displayed = content
                lastPublish = Date()
            }
            .onChange(of: content) { _, newValue in
                publish(newValue)
            }
            .onDisappear { trailingFlush?.cancel() }
    }

    private func publish(_ newValue: String) {
        let now = Date()
        if now.timeIntervalSince(lastPublish) >= Self.throttle {
            trailingFlush?.cancel()
            displayed = newValue
            lastPublish = now
        } else if trailingFlush == nil {
            let wait = Self.throttle - now.timeIntervalSince(lastPublish)
            trailingFlush = Task { @MainActor in
                try? await Task.sleep(nanoseconds: UInt64(max(wait, 0.01) * 1_000_000_000))
                trailingFlush = nil
                displayed = content
                lastPublish = Date()
            }
        }
    }
}

// MARK: - Thinking

/// The chat's collapsible thinking row, server-owned: "+ Thought" with an
/// inline preview, expanding to the full reasoning. While live it streams
/// plain text (the chat's own precedent — markdown lands on commit), with
/// a spinner in the marker slot.
struct ServerThinkingRow: View {
    let text: String
    let isLive: Bool

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    markerSlot
                    Text(isLive ? "Thinking…" : "Thought")
                        .font(.system(size: chatBodyFontSize, weight: .medium))
                        .foregroundStyle(.secondary)
                    if !isExpanded {
                        Text(previewLine)
                            .font(.system(size: chatBodyFontSize))
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(isLive ? .head : .tail)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isExpanded ? "Hide reasoning" : "Show reasoning")

            if isExpanded {
                Text(text.chatDisplayTrimmed)
                    .font(.system(size: chatBodyFontSize))
                    .lineSpacing(chatLineSpacing)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, ChatLayout.markerWidth + 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    private var markerSlot: some View {
        if isLive {
            CollapseMarker(isExpanded: isExpanded)
                .hidden()
                .overlay {
                    ProgressView()
                        .controlSize(.mini)
                }
        } else {
            CollapseMarker(isExpanded: isExpanded)
        }
    }

    private var previewLine: String {
        text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Tool calls

/// The chat's one-line tool row, server-owned: +/− marker (spinner while
/// the call body is still streaming), verb + target title, expanding to the
/// arguments-only generic panel — a server-side call is a *request*; no
/// result exists within it (#271 §4).
struct ServerToolCallRow: View {
    let name: String
    let argumentsJSON: String
    let isBuilding: Bool

    @State private var isExpanded = false

    private var props: ToolDisplayProps {
        ToolDisplayHelpers.displayProps(
            for: ToolCallInfo(
                id: name,
                name: name.isEmpty ? "…" : name,
                argumentsJSON: argumentsJSON
            ))
    }

    var body: some View {
        let props = self.props

        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    markerSlot
                    Text(props.title)
                        .font(.system(size: chatBodyFontSize))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isExpanded ? "Hide arguments" : "Show arguments")

            if isExpanded {
                ToolPanelView(panel: .generic(resultText: nil), argsFormatted: props.argsFormatted)
                    .padding(.leading, ChatLayout.markerWidth + 8)
            }
        }
    }

    @ViewBuilder
    private var markerSlot: some View {
        if isBuilding {
            CollapseMarker(isExpanded: isExpanded)
                .hidden()
                .overlay {
                    ProgressView()
                        .controlSize(.mini)
                }
        } else {
            CollapseMarker(isExpanded: isExpanded)
        }
    }
}

/// A malformed tool call in the chat's error grammar: error-tinted title
/// row expanding to the raw payload through the chat's error panel.
struct ServerMalformedToolCallRow: View {
    let raw: String

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    CollapseMarker(isExpanded: isExpanded)
                    Text("Malformed tool call")
                        .font(.system(size: chatBodyFontSize))
                        .foregroundStyle(DynamicColor.chatError)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isExpanded ? "Hide payload" : "Show the unparseable payload")

            if isExpanded {
                ToolPanelView(panel: .error(raw), argsFormatted: "")
                    .padding(.leading, ChatLayout.markerWidth + 8)
            }
        }
    }
}
