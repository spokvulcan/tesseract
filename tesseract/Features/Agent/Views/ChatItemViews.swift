//
//  ChatItemViews.swift
//  tesseract
//
//  The transcript rows of the flat document chat (ADR-0024): assistant text
//  as first-class flat prose in the readable column, user messages as neutral
//  blocks, and thinking / tool calls as one-line collapsible rows with +/−
//  markers (OpenCode structure, macOS-native dress — no icon zoo). Committed
//  rows are pure value views over `ChatItem`; only the Live Part view
//  observes streaming state.
//
//  Design language: one type size (`chatBodyFontSize`) for every piece of
//  transcript text — hierarchy comes from color and weight, never from size.
//  Monochrome content layer — system primary/secondary/tertiary styles and
//  system materials only. No glass here, ever (HIG: glass belongs to the
//  navigation/control layer). Red appears exclusively as the semantic error
//  tint. Row actions (copy, speak, edit & resend, Show in Finder) live in
//  right-click context menus — no visible button chrome, and nothing
//  hover-revealed (unreachable the moment the row loses hover).
//

import SwiftUI
import Textual
import os

// In the `var body` ViewBuilders below, `let _ = signposter.emitEvent(...)` is a
// DEBUG-only profiling idiom: in a result builder `let _ =` is a declaration
// (skipped), whereas a bare `_ =` is a `Void` expression the builder tries to
// render ("type '()' cannot conform to 'View'"). The discardable `let` is
// required here, not redundant. These signposts are the evidence trail for the
// ADR-0024 invariant: committed rows never re-render on a token delta — only
// the Live Part's row does.
// swiftlint:disable redundant_discardable_let

// MARK: - Collapse marker

/// The +/− disclosure marker of a collapsible row: "+" collapsed, "−"
/// expanded (OpenCode's convention). Fixed width so toggling never shifts
/// the title.
struct CollapseMarker: View {
    let isExpanded: Bool

    var body: some View {
        Text(isExpanded ? "\u{2212}" : "+")
            .font(.system(size: chatBodyFontSize, weight: .medium))
            .foregroundStyle(.tertiary)
            .frame(width: ChatLayout.markerWidth)
    }
}

// MARK: - Assistant message

/// One committed assistant message: its ordered Content Parts as flat rows.
/// Copy and Speak live in the right-click context menu — no visible chrome
/// on the prose.
struct AssistantMessageView: View {
    let message: AssistantMessage
    var isSpeaking = false
    var onPlay: (() -> Void)?
    var onStop: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: ChatLayout.rowSpacing) {
            ForEach(Array(message.content.enumerated()), id: \.offset) { index, part in
                if !part.isBlankRow {
                    AssistantPartView(part: part, messageID: message.id, partIndex: index)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contextMenu {
            if !message.text.isEmpty {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(message.text, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }

                if isSpeaking {
                    Button {
                        onStop?()
                    } label: {
                        Label("Stop Speaking", systemImage: "stop.fill")
                    }
                } else if onPlay != nil {
                    Button {
                        onPlay?()
                    } label: {
                        Label("Speak", systemImage: "speaker.wave.2")
                    }
                }
            }
        }
    }
}

/// Dispatch one Content Part to its row view. `messageID` + `partIndex` are
/// the part's identity (parts carry no IDs of their own) — the thinking row
/// uses them to look up its session-scoped duration.
struct AssistantPartView: View {
    let part: ContentPart
    let messageID: UUID
    let partIndex: Int

    var body: some View {
        switch part {
        case .text(let text):
            AssistantProseView(text: text.text)
        case .thinking(let thinking):
            ThinkingRowView(text: thinking.thinking, messageID: messageID, partIndex: partIndex)
        case .toolCall(let call):
            ToolCallRowView(part: call)
        }
    }
}

/// Assistant prose — the document body. Markdown by default; the toolbar
/// toggle falls back to plain text.
struct AssistantProseView: View {
    let text: String
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("AssistantProseView.body")
        #endif
        if useMarkdown {
            StructuredText(markdown: text)
                .textual.structuredTextStyle(ChatMarkdownStyle())
                .textual.codeBlockStyle(CopyableCodeBlockStyle())
                .textual.textSelection(.enabled)
                .font(.system(size: chatBodyFontSize))
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text(text)
                .font(.system(size: chatBodyFontSize))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Thinking row

/// One-line collapsible thinking row: "+ Thought" with an inline single-line
/// preview; expands ("− Thought") to the full reasoning text, with the
/// session-measured duration taking the slot the preview vacates.
struct ThinkingRowView: View {
    let text: String
    let messageID: UUID
    let partIndex: Int

    @Environment(ChatSession.self) private var session
    @State private var isExpanded = false

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ThinkingRowView.body")
        #endif
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    CollapseMarker(isExpanded: isExpanded)
                    Text("Thought")
                        .font(.system(size: chatBodyFontSize, weight: .medium))
                        .foregroundStyle(.secondary)
                    if isExpanded {
                        if let duration = session.thinkingDuration(
                            messageID: messageID, partIndex: partIndex),
                            duration >= ChatLayout.minBadgeDuration
                        {
                            Text(duration.chatBadge)
                                .font(.system(size: chatBodyFontSize))
                                .foregroundStyle(.tertiary)
                                .monospacedDigit()
                        }
                    } else {
                        Text(previewLine)
                            .font(.system(size: chatBodyFontSize))
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(isExpanded ? "Hide reasoning" : "Show reasoning")

            if isExpanded {
                Text(text)
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, ChatLayout.markerWidth + 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var previewLine: String {
        text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Tool call row

/// One-line tool row: +/− marker (spinner while running), verb + target,
/// duration badge — expanding to arguments, result, images. File tools get
/// Show in Finder in the right-click context menu.
struct ToolCallRowView: View {
    let part: ToolCallPart

    @Environment(ChatSession.self) private var session
    @State private var isExpanded = false

    private var props: ToolDisplayProps {
        ToolDisplayHelpers.displayProps(
            for: ToolCallInfo(id: part.id, name: part.name, argumentsJSON: part.argumentsJSON))
    }

    private var result: ToolResultMessage? { session.toolResult(for: part.id) }
    private var isRunning: Bool { result == nil && session.isGenerating }
    private var isError: Bool { result?.isError == true }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ToolCallRowView.body")
        #endif
        let props = self.props
        let result = self.result

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
                } label: {
                    HStack(spacing: 8) {
                        markerSlot
                        Text(props.title)
                            .font(.system(size: chatBodyFontSize))
                            .foregroundStyle(
                                isError
                                    ? AnyShapeStyle(DynamicColor.chatError)
                                    : AnyShapeStyle(.secondary)
                            )
                            .lineLimit(1)
                            // Middle truncation keeps both the verb and the
                            // filename visible on long workspace paths.
                            .truncationMode(.middle)
                        if let duration = session.toolDuration(for: part.id),
                            duration >= ChatLayout.minBadgeDuration
                        {
                            Text(duration.chatBadge).toolDurationBadge()
                        } else if isRunning, let start = session.toolStartInstant(for: part.id) {
                            LiveToolClockBadge(start: start)
                        }
                        Spacer(minLength: 0)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isExpanded ? "Hide details" : "Show arguments and result")

            }
            .contextMenu {
                if props.filePath != nil {
                    Button(action: openContainingFolder) {
                        Label("Show in Finder", systemImage: "folder")
                    }
                }
            }

            if isExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    detailSection("Arguments", text: props.argsFormatted, isError: false)

                    if let result {
                        let text = result.content.textContent
                        if !text.isEmpty {
                            detailSection("Result", text: text, isError: result.isError)
                        }
                        let images = result.content.imageAttachments(namespace: result.id)
                        if !images.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 8) {
                                    ForEach(images) { attachment in
                                        AsyncImageAttachmentView(attachment: attachment)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding(.leading, ChatLayout.markerWidth + 8)
            }
        }
    }

    @ViewBuilder
    private var markerSlot: some View {
        if isRunning {
            ProgressView()
                .controlSize(.mini)
                .frame(width: ChatLayout.markerWidth)
        } else {
            CollapseMarker(isExpanded: isExpanded)
        }
    }

    private func detailSection(_ label: String, text: String, isError: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: chatBodyFontSize, weight: .medium))
                .foregroundStyle(.tertiary)

            Text(text)
                .font(.system(size: chatBodyFontSize, design: .monospaced))
                .foregroundStyle(
                    isError
                        ? AnyShapeStyle(DynamicColor.chatError)
                        : AnyShapeStyle(.secondary)
                )
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
        }
    }

    private func openContainingFolder() {
        guard let path = props.filePath else { return }
        let fileURL: URL
        if path.hasPrefix("/") {
            fileURL = URL(fileURLWithPath: path).standardizedFileURL
        } else {
            fileURL = PathSandbox.defaultRoot.appendingPathComponent(path).standardizedFileURL
        }
        NSWorkspace.shared.activateFileViewerSelecting([fileURL])
    }
}

/// The ticking Tool Clock readout, in the same trailing slot the frozen
/// duration badge takes over on completion. Whole seconds at 1 Hz — the tick
/// is scoped to this view, so nothing else invalidates — and it appears only
/// once elapsed reaches 1s (the live analog of `minBadgeDuration`): a call
/// that finishes fast never flashes "0s".
private struct LiveToolClockBadge: View {
    let start: ContinuousClock.Instant

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let elapsed = ContinuousClock.now - start
            if elapsed >= .seconds(1) {
                Text(elapsed.liveChatBadge).toolDurationBadge()
            }
        }
    }
}

extension Text {
    /// The one styling of the trailing duration slot — the live and frozen
    /// badges share it so the tick-to-frozen transition changes nothing but
    /// the number.
    fileprivate func toolDurationBadge() -> some View {
        font(.system(size: chatBodyFontSize))
            .foregroundStyle(.tertiary)
            .monospacedDigit()
    }
}

// MARK: - User message

/// The user's message: a trailing-aligned neutral block on the system
/// quinary fill — no brand color in the content layer. (Owner-approved as-is;
/// do not restyle.)
struct UserMessageRow: View {
    let message: UserMessage

    @Environment(ChatSession.self) private var session
    @Environment(ComposerDraftController.self) private var composerDraft

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("UserMessageRow.body")
        #endif
        VStack(alignment: .leading, spacing: 8) {
            if !message.images.isEmpty {
                HStack(spacing: 8) {
                    ForEach(message.images) { attachment in
                        AsyncImageAttachmentView(attachment: attachment)
                    }
                }
            }
            if !message.content.isEmpty {
                Text(message.content)
                    .font(.system(size: chatBodyFontSize))
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quinary, in: RoundedRectangle(cornerRadius: 12))
        .contextMenu {
            Button {
                beginEditing()
            } label: {
                Label("Edit & Resend", systemImage: "pencil")
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message.content, forType: .string)
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
        .help(message.timestamp.formatted(date: .abbreviated, time: .shortened))
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    private func beginEditing() {
        guard let draft = session.beginEditingMessage(message.id) else { return }
        composerDraft.restore(text: draft.text, images: draft.images)
    }
}

// MARK: - System note

/// Centered marker for context transforms (compaction) — quiet, out of the
/// conversation flow.
struct SystemNoteRow: View {
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "arrow.triangle.2.circlepath")
                .font(.system(size: 12))
            Text(text)
                .font(.system(size: chatBodyFontSize))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .background(.quinary, in: Capsule())
        .frame(maxWidth: .infinity, alignment: .center)
    }
}

// MARK: - Live Part

/// The one streaming row. Text renders markdown from the throttled
/// `displayText` (~10 Hz); thinking shows the live one-line tail.
struct LivePartView: View {
    let live: LivePart

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("LivePartView.body")
        #endif
        switch live.kind {
        case .text:
            AssistantProseView(text: live.displayText)
        case .thinking:
            LiveThinkingRowView(live: live)
        }
    }
}

/// Streaming thinking: "Thinking…" with the reasoning tail as a live
/// single-line preview; expandable mid-stream to the full text so far.
struct LiveThinkingRowView: View {
    let live: LivePart
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    CollapseMarker(isExpanded: isExpanded)
                    Text("Thinking…")
                        .font(.system(size: chatBodyFontSize, weight: .medium))
                        .foregroundStyle(.secondary)
                    if !isExpanded {
                        Text(tailPreview)
                            .font(.system(size: chatBodyFontSize))
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.head)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                Text(live.displayText)
                    .font(.system(size: chatBodyFontSize))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, ChatLayout.markerWidth + 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    /// The last stretch of the stream, single line — head-truncated so the
    /// newest words stay visible.
    private var tailPreview: String {
        String(live.displayText.suffix(160))
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}

// swiftlint:enable redundant_discardable_let
