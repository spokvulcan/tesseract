import SwiftUI
import Textual
import os

// In the `var body` ViewBuilders below, `let _ = signposter.emitEvent(...)` is a
// DEBUG-only profiling idiom: in a result builder `let _ =` is a declaration
// (skipped), whereas a bare `_ =` is a `Void` expression the builder tries to
// render ("type '()' cannot conform to 'View'"). The discardable `let` is
// required here, not redundant.
// swiftlint:disable redundant_discardable_let

/// Base size for chat message body text. The markdown renderer (Textual)
/// derives all its font-scaled metrics from the environment font, so applying
/// `.font(.system(size: chatBodyFontSize))` keeps both render modes identical.
let chatBodyFontSize: CGFloat = 16

// MARK: - Step Gutter (shared timeline column)

/// Shared timeline gutter: vertical line (when not last) + icon. Used by all step row views.
private struct StepGutter: View {
    let iconName: String
    var iconColor: Color = .secondary
    let isLast: Bool

    var body: some View {
        ZStack(alignment: .top) {
            if !isLast {
                Rectangle()
                    .fill(Color(white: 0.2))
                    .frame(width: 1)
                    .padding(.top, 16)
            }
            Image(systemName: iconName)
                .font(.system(size: 11))
                .foregroundStyle(iconColor)
                .frame(width: 16, height: 16)
        }
        .frame(width: 20)
    }
}

// MARK: - Copy Button

private struct CopyButton: View {
    let text: String

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        } label: {
            Image(systemName: "doc.on.doc")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - User Bubble

struct UserBubble: View, Equatable {
    let data: UserRow

    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    @State private var isHovering = false
    @Environment(AgentCoordinator.self) private var coordinator

    // Equality compares only `data` — the environment-injected coordinator is
    // read lazily on user action, never during diffing.
    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("UserBubble.body")
        #endif
        VStack(alignment: .trailing, spacing: 4) {
            VStack(alignment: .leading, spacing: 8) {
                if !data.images.isEmpty {
                    HStack(spacing: 8) {
                        ForEach(data.images) { attachment in
                            AsyncImageAttachmentView(attachment: attachment)
                        }
                    }
                }

                if !data.content.isEmpty {
                    HStack(alignment: .bottom, spacing: 8) {
                        if useMarkdown {
                            StructuredText(markdown: data.content)
                                .textual.structuredTextStyle(.default)
                                .textual.codeBlockStyle(CopyableCodeBlockStyle())
                                .textual.textSelection(.enabled)
                                .font(.system(size: chatBodyFontSize))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        } else {
                            Text(data.content)
                                .font(.system(size: chatBodyFontSize))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        Text(data.timestamp)
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(.bottom, 2)
                    }
                } else {
                    HStack {
                        Spacer()
                        Text(data.timestamp)
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(red: 0.79, green: 0.28, blue: 0.65))
            .foregroundStyle(.white)
            .clipShape(
                .rect(
                    topLeadingRadius: 18,
                    bottomLeadingRadius: 18,
                    bottomTrailingRadius: 4,
                    topTrailingRadius: 18
                )
            )
            .contextMenu {
                Button {
                    coordinator.beginEditingMessage(data.messageID)
                } label: {
                    Label("Edit & resend", systemImage: "pencil")
                }
            }

            HStack(spacing: 10) {
                Button {
                    coordinator.beginEditingMessage(data.messageID)
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Edit & resend — trim the images or text and send again")

                CopyButton(text: data.content)
            }
            .opacity(isHovering ? 1 : 0)
            .animation(.easeInOut(duration: 0.15), value: isHovering)
        }
        .onHover { isHovering = $0 }
    }
}

// MARK: - Assistant Bubble

struct AssistantBubble: View, Equatable {
    let data: AssistantTextRow
    var isSpeaking: Bool = false
    var onPlay: (() -> Void)?
    var onStop: (() -> Void)?

    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    @State private var isHovering = false

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.data == rhs.data && lhs.isSpeaking == rhs.isSpeaking
    }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("AssistantBubble.body")
        #endif
        VStack(alignment: .trailing, spacing: 4) {
            HStack(alignment: .bottom, spacing: 8) {
                VStack(alignment: .leading, spacing: 6) {
                    if useMarkdown {
                        StructuredText(markdown: data.content)
                            .textual.structuredTextStyle(.gitHub)
                            .textual.codeBlockStyle(CopyableCodeBlockStyle())
                            .textual.textSelection(.enabled)
                            .font(.system(size: chatBodyFontSize))
                    } else {
                        Text(data.content)
                            .font(.system(size: chatBodyFontSize))
                            .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Text(data.timestamp)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 2)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .padding(.top, data.hasStepsAbove ? 4 : 0)
            .background(Color(white: 0.15))
            .clipShape(
                .rect(
                    topLeadingRadius: 18,
                    bottomLeadingRadius: 4,
                    bottomTrailingRadius: 18,
                    topTrailingRadius: 18
                )
            )

            HStack(spacing: 8) {
                CopyButton(text: data.content)

                if onPlay != nil || isSpeaking {
                    Button {
                        isSpeaking ? onStop?() : onPlay?()
                    } label: {
                        Image(systemName: isSpeaking ? "stop.circle.fill" : "play.circle.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(isSpeaking ? .red : .secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .opacity(isHovering || isSpeaking ? 1 : 0)
            .animation(.easeInOut(duration: 0.15), value: isHovering)
            .animation(.easeInOut(duration: 0.15), value: isSpeaking)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onHover { isHovering = $0 }
    }
}

// MARK: - Streaming Bubble

struct StreamingBubble: View, Equatable {
    let data: StreamingTextRow

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("StreamingBubble.body")
        #endif
        Text(data.content)
            .font(.system(size: chatBodyFontSize))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(white: 0.15))
            .clipShape(
                .rect(
                    topLeadingRadius: 18,
                    bottomLeadingRadius: 4,
                    bottomTrailingRadius: 18,
                    topTrailingRadius: 18
                )
            )
    }
}

// MARK: - Thinking Row

struct ThinkingRowView: View, Equatable {
    let data: ThinkingRow
    @State private var isExpanded = false

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ThinkingRowView.body")
        #endif
        HStack(alignment: .top, spacing: 12) {
            StepGutter(iconName: "brain", isLast: data.isLast)

            VStack(alignment: .leading, spacing: 0) {
                Button(
                    action: { isExpanded.toggle() },
                    label: {
                        HStack(spacing: 8) {
                            Text("Thinking")
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.tertiary)
                                .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        }
                        .padding(.vertical, 0)
                        .contentShape(Rectangle())
                    }
                )
                .buttonStyle(.plain)

                if isExpanded {
                    Text(data.content)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(.trailing, 14)
                        .padding(.bottom, 8)
                        .padding(.top, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text(data.content)
                        .font(.system(size: 13))
                        .foregroundStyle(.tertiary)
                        .lineLimit(2)
                        .truncationMode(.tail)
                        .padding(.trailing, 14)
                        .padding(.bottom, 8)
                        .padding(.top, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .allowsHitTesting(false)
                }
            }
            .padding(.bottom, data.isLast ? 0 : 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Tool Call Row

struct ToolCallRowView: View, Equatable {
    let data: ToolCallRow
    let rowID: String

    @Environment(AgentCoordinator.self) private var coordinator
    @State private var isHovering = false

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.data == rhs.data && lhs.rowID == rhs.rowID
    }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ToolCallRowView.body")
        #endif
        HStack(alignment: .top, spacing: 12) {
            StepGutter(
                iconName: data.iconName, iconColor: data.isError ? .red : .secondary,
                isLast: data.isLast)

            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Button(
                        action: { coordinator.toggleDetailExpanded(rowID) },
                        label: {
                            HStack(spacing: 8) {
                                Text(data.displayTitle)
                                    .font(.system(size: 13))
                                    .foregroundStyle(data.isError ? .red : .secondary)
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.tertiary)
                                    .rotationEffect(.degrees(data.isDetailExpanded ? 90 : 0))
                            }
                            .contentShape(Rectangle())
                        }
                    )
                    .buttonStyle(.plain)

                    if data.filePath != nil {
                        Button(action: openContainingFolder) {
                            Image(systemName: "folder")
                                .font(.system(size: 11))
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                        .help("Show in Finder")
                        .opacity(isHovering ? 1 : 0)
                        .animation(.easeInOut(duration: 0.15), value: isHovering)
                    }
                }
                .padding(.vertical, 0)

                if data.isDetailExpanded {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Arguments")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)
                                .textCase(.uppercase)

                            Text(data.argumentsFormatted)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(white: 0.1).opacity(0.5))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                        }

                        if let result = data.resultContent {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Result")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundStyle(.tertiary)
                                    .textCase(.uppercase)

                                Text(result)
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundStyle(data.isError ? .red : .primary)
                                    .textSelection(.enabled)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .padding(8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color(white: 0.1).opacity(0.5))
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }

                        if !data.resultImages.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Images")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundStyle(.tertiary)
                                    .textCase(.uppercase)

                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 8) {
                                        ForEach(data.resultImages) { attachment in
                                            AsyncImageAttachmentView(attachment: attachment)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding(.trailing, 14)
                    .padding(.bottom, 8)
                    .padding(.top, 6)
                }
            }
            .padding(.bottom, data.isLast ? 0 : 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onHover { isHovering = $0 }
    }

    private func openContainingFolder() {
        guard let path = data.filePath else { return }
        let fileURL: URL
        if path.hasPrefix("/") {
            fileURL = URL(fileURLWithPath: path).standardizedFileURL
        } else {
            fileURL = PathSandbox.defaultRoot.appendingPathComponent(path).standardizedFileURL
        }
        NSWorkspace.shared.activateFileViewerSelecting([fileURL])
    }
}

// MARK: - Tool Text Row

struct ToolTextRowView: View, Equatable {
    let data: ToolTextRow

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ToolTextRowView.body")
        #endif
        HStack(alignment: .top, spacing: 12) {
            StepGutter(iconName: "text.bubble", isLast: data.isLast)

            Text(data.content)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.bottom, data.isLast ? 0 : 8)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Turn Header

struct TurnHeaderView: View, Equatable {
    let data: TurnHeaderRow

    @Environment(AgentCoordinator.self) private var coordinator

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("TurnHeaderView.body")
        #endif
        Button(
            action: { coordinator.toggleTurnExpanded(data.turnID) },
            label: {
                HStack(spacing: 4) {
                    Text("\(data.stepCount) step\(data.stepCount == 1 ? "" : "s")")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)

                    if data.isGenerating {
                        ProgressView()
                            .controlSize(.mini)
                    }

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(data.isExpanded ? 0 : -90))
                }
                .padding(.vertical, 4)
                .contentShape(Rectangle())
            }
        )
        .buttonStyle(.plain)
    }
}

// MARK: - Async Image Attachment

/// Decodes image data off the main thread to avoid blocking scroll. Clicking
/// opens the image full size in Quick Look (slice #114), navigable across the
/// whole conversation; the temp file is pre-warmed on decode so opening is
/// near-instant.
struct AsyncImageAttachmentView: View {
    let attachment: ImageAttachment
    @Environment(AgentCoordinator.self) private var coordinator
    @State private var nsImage: NSImage?

    var body: some View {
        Group {
            if let nsImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(white: 0.2))
                    .overlay(ProgressView().controlSize(.small))
            }
        }
        .frame(maxWidth: 200, maxHeight: 200)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .onTapGesture { coordinator.composerDraft.openQuickLook(clicked: attachment.id) }
        .help("Click to view full size")
        .task(id: attachment.id) {
            let data = attachment.data
            nsImage = await Task.detached {
                NSImage(data: data)
            }.value
            // Pre-warm the Quick Look temp file so the click opens instantly.
            coordinator.composerDraft.prewarmImagePreview(attachment)
        }
    }
}

// swiftlint:enable redundant_discardable_let
