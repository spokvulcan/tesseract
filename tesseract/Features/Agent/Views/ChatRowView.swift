import SwiftUI
import os

/// Routes a `ChatRow` to the appropriate leaf view with correct alignment.
struct ChatRowView: View {
    let row: ChatRow
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @Environment(AgentCoordinator.self) private var coordinator

    var body: some View {
        #if DEBUG
        let _ = ChatViewPerf.signposter.emitEvent("ChatRowView.body")
        #endif
        switch row.kind {
        case .user(let data):
            if coordinator.isViewingBackgroundSession {
                VStack(spacing: 12) {
                    HStack {
                        VStack { Divider() }
                        Text(data.timestamp)
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.tertiary)
                            .textCase(.uppercase)
                        VStack { Divider() }
                    }
                    .padding(.vertical, 8)
                    
                    HStack {
                        Spacer(minLength: 60)
                        UserBubble(data: data)
                            .equatable()
                    }
                }
            } else {
                HStack {
                    Spacer(minLength: 60)
                    UserBubble(data: data)
                        .equatable()
                }
            }

        case .assistantText(let data):
            HStack {
                AssistantBubble(
                    data: data,
                    isSpeaking: speakingMessageID == data.messageID && isSpeechActive,
                    onPlay: {
                        speakingMessageID = data.messageID
                        coordinator.speakMessage(data.messageID)
                    },
                    onStop: {
                        coordinator.stopSpeaking()
                        speakingMessageID = nil
                    }
                )
                .equatable()
                Spacer(minLength: 60)
            }

        case .streamingText(let data):
            HStack {
                StreamingBubble(data: data)
                    .equatable()
                Spacer(minLength: 60)
            }

        case .thinking(let data):
            ThinkingRowView(data: data)
                .equatable()
                .padding(.leading, 18)

        case .toolCall(let data):
            ToolCallRowView(data: data, rowID: row.id)
                .equatable()
                .padding(.leading, 18)

        case .toolText(let data):
            ToolTextRowView(data: data)
                .equatable()
                .padding(.leading, 18)

        case .turnHeader(let data):
            TurnHeaderView(data: data)
                .equatable()

        case .system(let data):
            HStack {
                Spacer(minLength: 60)
                Text(data.content)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
                Spacer(minLength: 60)
            }

        case .streamingIndicator:
            HStack(spacing: 6) {
                ProgressView()
                    .controlSize(.small)
                Text("Generating…")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
        }
    }
}
