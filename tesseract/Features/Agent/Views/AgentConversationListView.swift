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

    enum DisplayBlock: Identifiable {
        case user(AgentChatMessage)
        case assistant(AgentChatMessage, [AgentChatMessage])
        case system(AgentChatMessage)
        
        var id: UUID {
            switch self {
            case .user(let msg): return msg.id
            case .assistant(let msg, _): return msg.id
            case .system(let msg): return msg.id
            }
        }
    }
    
    private var displayBlocks: [DisplayBlock] {
        var blocks: [DisplayBlock] = []
        var currentAssistant: AgentChatMessage?
        var currentToolResults: [AgentChatMessage] = []
        
        for message in coordinator.messages {
            switch message.role {
            case .user, .system:
                if let asst = currentAssistant {
                    blocks.append(.assistant(asst, currentToolResults))
                    currentAssistant = nil
                    currentToolResults = []
                }
                blocks.append(message.role == .user ? .user(message) : .system(message))
            case .assistant:
                if let asst = currentAssistant {
                    blocks.append(.assistant(asst, currentToolResults))
                    currentToolResults = []
                }
                currentAssistant = message
            case .tool:
                if currentAssistant != nil {
                    currentToolResults.append(message)
                } else {
                    // Orphan tool result (fallback)
                    blocks.append(.system(message))
                }
            }
        }
        if let asst = currentAssistant {
            blocks.append(.assistant(asst, currentToolResults))
        }
        return blocks
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if !coordinator.assembledSystemPrompt.isEmpty {
                        AgentSystemPromptView()
                    }

                    if coordinator.messages.isEmpty && !coordinator.isGenerating {
                        emptyState
                    }

                    ForEach(displayBlocks) { block in
                        blockView(block)
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
    private func blockView(_ block: DisplayBlock) -> some View {
        switch block {
        case .user(let message):
            HStack {
                Spacer(minLength: 60)
                UserMessageBubble(message: message)
                    .id(message.id)
            }
        case .assistant(let message, let toolResults):
            assistantBlockView(message: message, toolResults: toolResults)
                .id(message.id)
        case .system(let message):
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

    @ViewBuilder
    private func assistantBlockView(message: AgentChatMessage, toolResults: [AgentChatMessage]) -> some View {
        let hasContent = !message.content.isEmpty
        let hasThinkingOrTools = (message.thinking?.isEmpty == false) || !message.toolCalls.isEmpty
        
        if hasContent {
            // Normal message bubble
            HStack {
                AssistantMessageBubble(
                    message: message,
                    toolResults: toolResults,
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
                
                Spacer(minLength: 60)
            }
        } else if hasThinkingOrTools {
            // "Ghost" list lane (no bubble)
            HStack {
                AssistantMessageListBlock(message: message, toolResults: toolResults)
                Spacer(minLength: 60)
            }
        }
    }

    // MARK: - Streaming Bubble

    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    private var streamingBubble: some View {
        HStack {
            let hasText = !coordinator.streamingText.isEmpty
            
            VStack(alignment: .leading, spacing: 6) {
                if !coordinator.streamingThinking.isEmpty {
                    VStack(alignment: .leading, spacing: 0) {
                        Button(action: {}) {
                            HStack(spacing: 8) {
                                Image(systemName: "brain")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.secondary)
                                    .frame(width: 16)
                                
                                Text(coordinator.isThinking ? "Thinking…" : "Thinking")
                                    .font(.system(size: 13))
                                    .foregroundStyle(.secondary)
                                
                                Spacer()
                                
                                Image(systemName: "chevron.right")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.tertiary)
                                    .rotationEffect(.degrees(90))
                            }
                            .padding(.vertical, 6)
                            .padding(.horizontal, hasText ? 0 : 14)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text(coordinator.streamingThinking)
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .padding(.leading, hasText ? 24 : 38)
                                .padding(.trailing, 14)
                                .padding(.bottom, 6)
                        }
                    }
                }

                if hasText {
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
            .padding(.horizontal, hasText ? 14 : 0)
            .padding(.vertical, hasText ? 10 : 4)
            .background(hasText ? Color(white: 0.15) : Color.clear)
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
