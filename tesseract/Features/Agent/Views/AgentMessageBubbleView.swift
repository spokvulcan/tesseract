//
//  AgentMessageBubbleView.swift
//  tesseract
//

import SwiftUI
import Textual

struct AssistantMessageBubble: View {
    let message: AgentChatMessage
    var toolResults: [AgentChatMessage] = []
    var isSpeaking: Bool = false
    var onPlay: (() -> Void)? = nil
    var onStop: (() -> Void)? = nil
    @State private var isThinkingExpanded = false
    @State private var isHovering = false
    
    @AppStorage("agentUseMarkdown") private var useMarkdown = true

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 6) {
                if let thinking = message.thinking, !thinking.isEmpty {
                    thinkingSection(thinking)
                }
                if !message.content.isEmpty {
                    if useMarkdown {
                        StructuredText(markdown: message.content)
                            .textual.structuredTextStyle(.gitHub)
                            .textual.textSelection(.enabled)
                    } else {
                        Text(message.content)
                            .font(.system(size: 15))
                            .textSelection(.enabled)
                    }
                }
                
                if !message.toolCalls.isEmpty {
                    ForEach(Array(message.toolCalls.enumerated()), id: \.offset) { index, toolCall in
                        let result = index < toolResults.count ? toolResults[index] : nil
                        AgentToolCallView(toolCall: toolCall, toolResult: result)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
            // Timestamp and playback controls
            HStack(spacing: 4) {
                // Fixed width container prevents layout shift on hover
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
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isThinkingExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: "brain")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .frame(width: 16)
                    
                    Text("Thinking")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isThinkingExpanded ? 90 : 0))
                }
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            if isThinkingExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Text(thinking)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(.leading, 24)
                        .padding(.bottom, 6)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}

struct UserMessageBubble: View {
    let message: AgentChatMessage
    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    
    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if useMarkdown {
                StructuredText(markdown: message.content)
                    .textual.structuredTextStyle(.default)
                    .textual.textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(message.content)
                    .font(.system(size: 15))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
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
