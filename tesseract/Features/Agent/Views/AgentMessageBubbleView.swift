//
//  AgentMessageBubbleView.swift
//  tesseract
//

import SwiftUI

struct AssistantMessageBubble: View {
    let message: AgentChatMessage
    var isSpeaking: Bool = false
    var onPlay: (() -> Void)? = nil
    var onStop: (() -> Void)? = nil
    @State private var isThinkingExpanded = false
    @State private var isHovering = false

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 6) {
                if let thinking = message.thinking, !thinking.isEmpty {
                    thinkingSection(thinking)
                }
                if !message.content.isEmpty {
                    Text(message.content)
                        .font(.system(size: 15))
                        .textSelection(.enabled)
                }
                
                if !message.toolCalls.isEmpty {
                    ForEach(Array(message.toolCalls.enumerated()), id: \.offset) { index, toolCall in
                        AgentToolCallView(toolCall: toolCall)
                    }
                }
            }
            
            // Timestamp and playback controls
            HStack(spacing: 4) {
                // Fixed width container prevents layout shift on hover
                ZStack {
                    if isHovering || isSpeaking {
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
                
                Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, 2)
        }
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
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) { isHovering = hovering }
        }
    }

    private func thinkingSection(_ thinking: String) -> some View {
        DisclosureGroup(isExpanded: $isThinkingExpanded) {
            Text(thinking)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.top, 4)
        } label: {
            Label("Thinking", systemImage: "brain.head.profile")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 6)
    }
}

struct UserMessageBubble: View {
    let message: AgentChatMessage
    
    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Text(message.content)
                .font(.system(size: 15))
                .textSelection(.enabled)
            
            Text(message.timestamp.formatted(date: .omitted, time: .shortened))
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
