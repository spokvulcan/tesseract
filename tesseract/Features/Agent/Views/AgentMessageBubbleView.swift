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
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 0) {
                if let thinking = message.thinking, !thinking.isEmpty {
                    thinkingSection(thinking)
                }
                if !message.content.isEmpty {
                    Text(message.content)
                        .textSelection(.enabled)
                }
            }
            .bubbleBackground()

            if isHovering || isSpeaking {
                Button {
                    isSpeaking ? onStop?() : onPlay?()
                } label: {
                    Image(systemName: isSpeaking ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title3)
                        .foregroundStyle(isSpeaking ? .red : .secondary)
                }
                .buttonStyle(.plain)
                .help(isSpeaking ? "Stop speaking" : "Speak this message")
                .transition(.opacity)
            }

            Spacer(minLength: 60)
        }
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
