//
//  AgentConversationListView.swift
//  tesseract
//

import SwiftUI

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
        } else {
            HStack {
                if message.role == .user { Spacer(minLength: 60) }

                Text(message.content)
                    .textSelection(.enabled)
                    .bubbleBackground(
                        message.role == .user
                            ? AnyShapeStyle(.tint.opacity(0.15))
                            : AnyShapeStyle(.fill.quaternary)
                    )

                if message.role != .user { Spacer(minLength: 60) }
            }
            .id(message.id)
        }
    }

    // MARK: - Streaming Bubble

    private var streamingBubble: some View {
        HStack {
            VStack(alignment: .leading, spacing: 0) {
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
                    Text(coordinator.streamingText)
                        .textSelection(.enabled)
                }
            }
            .bubbleBackground()

            Spacer(minLength: 60)
        }
        .id("streaming")
    }
}
