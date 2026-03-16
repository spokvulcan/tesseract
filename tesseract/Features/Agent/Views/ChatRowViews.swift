import SwiftUI
import Textual
import os

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

// MARK: - User Bubble

struct UserBubble: View, Equatable {
    let data: UserRow

    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        let _ = ChatViewPerf.signposter.emitEvent("UserBubble.body")
        HStack(alignment: .bottom, spacing: 8) {
            if useMarkdown {
                StructuredText(markdown: data.content)
                    .textual.structuredTextStyle(.default)
                    .textual.textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(data.content)
                    .font(.system(size: 15))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Text(data.timestamp)
                .font(.system(size: 11))
                .foregroundStyle(.white.opacity(0.7))
                .padding(.bottom, 2)
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
    }
}

// MARK: - Assistant Bubble

struct AssistantBubble: View, Equatable {
    let data: AssistantTextRow
    var isSpeaking: Bool = false
    var onPlay: (() -> Void)? = nil
    var onStop: (() -> Void)? = nil

    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    @State private var isHovering = false

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.data == rhs.data && lhs.isSpeaking == rhs.isSpeaking
    }

    var body: some View {
        let _ = ChatViewPerf.signposter.emitEvent("AssistantBubble.body")
        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 6) {
                if useMarkdown {
                    StructuredText(markdown: data.content)
                        .textual.structuredTextStyle(.gitHub)
                        .textual.textSelection(.enabled)
                } else {
                    Text(data.content)
                        .font(.system(size: 15))
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 4) {
                ZStack {
                    if (isHovering && onPlay != nil) || isSpeaking {
                        Button {
                            isSpeaking ? onStop?() : onPlay?()
                        } label: {
                            Image(systemName: isSpeaking ? "stop.circle.fill" : "play.circle.fill")
                                .foregroundStyle(isSpeaking ? .red : .secondary)
                        }
                        .buttonStyle(.plain)
                        .transition(.opacity)
                    }
                }
                .frame(width: 16, alignment: .trailing)

                Text(data.timestamp)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
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
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

// MARK: - Streaming Bubble

struct StreamingBubble: View, Equatable {
    let data: StreamingTextRow

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        let _ = ChatViewPerf.signposter.emitEvent("StreamingBubble.body")
        Text(data.content)
            .font(.system(size: 15))
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
        let _ = ChatViewPerf.signposter.emitEvent("ThinkingRowView.body")
        HStack(alignment: .top, spacing: 12) {
            StepGutter(iconName: "brain", isLast: data.isLast)

            VStack(alignment: .leading, spacing: 0) {
                Button(action: { isExpanded.toggle() }) {
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

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.data == rhs.data && lhs.rowID == rhs.rowID
    }

    var body: some View {
        let _ = ChatViewPerf.signposter.emitEvent("ToolCallRowView.body")
        HStack(alignment: .top, spacing: 12) {
            StepGutter(iconName: data.iconName, iconColor: data.isError ? .red : .secondary, isLast: data.isLast)

            VStack(alignment: .leading, spacing: 0) {
                Button(action: { coordinator.toggleDetailExpanded(rowID) }) {
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
                    .padding(.vertical, 0)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

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
                    }
                    .padding(.trailing, 14)
                    .padding(.bottom, 8)
                    .padding(.top, 6)
                }
            }
            .padding(.bottom, data.isLast ? 0 : 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Tool Text Row

struct ToolTextRowView: View, Equatable {
    let data: ToolTextRow

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.data == rhs.data }

    var body: some View {
        let _ = ChatViewPerf.signposter.emitEvent("ToolTextRowView.body")
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
        let _ = ChatViewPerf.signposter.emitEvent("TurnHeaderView.body")
        Button(action: { coordinator.toggleTurnExpanded(data.turnID) }) {
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
        .buttonStyle(.plain)
    }
}
