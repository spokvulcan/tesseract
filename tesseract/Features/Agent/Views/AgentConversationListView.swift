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

    /// Stable ID for the synthetic streaming turn — avoids UUID() churn on every render.
    private static let streamingTurnID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    enum DisplayBlock: Identifiable {
        case user(AgentChatMessage)
        case assistant(AssistantTurn)
        case system(AgentChatMessage)
        
        var id: UUID {
            switch self {
            case .user(let msg): return msg.id
            case .assistant(let turn): return turn.id
            case .system(let msg): return msg.id
            }
        }
    }
    
    private var displayBlocks: [DisplayBlock] {
        var blocks: [DisplayBlock] = []
        var currentTurnMessages: [AgentChatMessage] = []
        
        func commitTurn() {
            if !currentTurnMessages.isEmpty {
                let turn = AssistantTurn(id: currentTurnMessages[0].id, messages: currentTurnMessages)
                blocks.append(.assistant(turn))
                currentTurnMessages = []
            }
        }
        
        for message in coordinator.messages {
            switch message.role {
            case .user, .system:
                commitTurn()
                blocks.append(message.role == .user ? .user(message) : .system(message))
            case .assistant, .tool:
                currentTurnMessages.append(message)
            }
        }
        commitTurn()
        
        // If generating and we didn't just commit an assistant turn (meaning the assistant hasn't appended a final message to the DB yet but might be streaming)
        // Actually, if it's generating, we always want the LAST block to be the one showing the stream.
        if coordinator.isGenerating {
            if case .assistant = blocks.last {
                // The last block is an assistant turn, it will handle it.
            } else {
                // There is no assistant turn at the end (e.g. just user message), add an empty one for streaming
                let emptyTurn = AssistantTurn(id: Self.streamingTurnID, messages: [])
                blocks.append(.assistant(emptyTurn))
            }
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

                }
                .padding()
            }
            .onChange(of: coordinator.streamUpdateCount) {
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
        case .assistant(let turn):
            HStack {
                AssistantTurnView(
                    turn: turn,
                    isGenerating: coordinator.isGenerating && block.id == displayBlocks.last?.id,
                    speakingMessageID: $speakingMessageID,
                    isSpeechActive: isSpeechActive
                )
                .id(turn.id)
                Spacer(minLength: 60)
            }
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

    // MARK: - Streaming Bubble

    @AppStorage("agentUseMarkdown") private var useMarkdown = true
}
