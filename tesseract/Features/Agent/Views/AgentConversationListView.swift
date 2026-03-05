//
//  AgentConversationListView.swift
//  tesseract
//

import SwiftUI
import Textual

struct AgentConversationListView: View {
    @Binding var speakingMessageID: UUID?
    var isSpeechActive: Bool

    @EnvironmentObject private var coordinator: AgentCoordinator

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if coordinator.messages.isEmpty && !coordinator.isGenerating {
                        emptyState
                    }

                    ForEach(coordinator.messages) { message in
                        messageBubble(message)
                    }

                    if coordinator.isGenerating &&
                        (!coordinator.streamingText.isEmpty || !coordinator.streamingThinking.isEmpty) {
                        streamingBubble
                    }

                    if coordinator.isGenerating
                        && coordinator.streamingText.isEmpty
                        && coordinator.streamingThinking.isEmpty {
                        HStack(spacing: 6) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Generating…")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 12)
                        .id("generating")
                    }
                }
                .padding()
            }
            .onChange(of: coordinator.streamingText) {
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("streaming", anchor: .bottom)
                }
            }
            .onChange(of: coordinator.streamingThinking) {
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("streaming", anchor: .bottom)
                }
            }
            .onChange(of: coordinator.messages.count) {
                if let lastID = coordinator.messages.last?.id {
                    let target: AnyHashable = coordinator.isGenerating ? "streaming" : lastID
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(target, anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 40))
                .foregroundStyle(.quaternary)
            Text("Start a conversation")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }

    // MARK: - Message Bubbles

    @ViewBuilder
    private func messageBubble(_ message: AgentChatMessage) -> some View {
        if message.role == .assistant {
            HStack {
                AssistantMessageBubble(
                    message: message,
                    isSpeaking: speakingMessageID == message.id && isSpeechActive,
                    onPlay: {
                        speakingMessageID = message.id
                        coordinator.speakMessage(message)
                    },
                    onStop: {
                        coordinator.stopSpeaking()
                        speakingMessageID = nil
                    }
                )
                .id(message.id)
                
                Spacer(minLength: 60)
            }
        } else if message.role == .user {
            HStack {
                Spacer(minLength: 60)
                
                UserMessageBubble(message: message)
                    .id(message.id)
            }
        } else if message.role == .tool {
            HStack {
                AgentToolResultBubbleView(message: message)
                    .id(message.id)
                Spacer(minLength: 60)
            }
        } else {
            // System messages
            HStack {
                Spacer(minLength: 60)
                Text(message.content)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
                Spacer(minLength: 60)
            }
            .id(message.id)
        }
    }

    // MARK: - Streaming Bubble

    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var streamingBubble: some View {
        HStack {
            VStack(alignment: .leading, spacing: 6) {
                if !coordinator.streamingThinking.isEmpty {
                    DisclosureGroup(isExpanded: .constant(true)) {
                        Text(coordinator.streamingThinking)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .padding(.top, 4)
                    } label: {
                        Label(
                            coordinator.isThinking ? "Thinking…" : "Thinking",
                            systemImage: "brain.head.profile"
                        )
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 6)
                }

                if !coordinator.streamingText.isEmpty {
                    if useMarkdown {
                        StructuredText(markdown: coordinator.streamingText)
                            .textual.structuredTextStyle(.gitHub)
                            .textual.textSelection(.enabled)
                    } else {
                        Text(coordinator.streamingText)
                            .font(.system(size: 15))
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(white: 0.15))
            .foregroundStyle(.white)
            .clipShape(
                .rect(
                    topLeadingRadius: 18,
                    bottomLeadingRadius: 4,
                    bottomTrailingRadius: 18,
                    topTrailingRadius: 18
                )
            )

            Spacer(minLength: 60)
        }
        .id("streaming")
    }
}
