import SwiftUI
import Textual
import MLXLMCommon

struct AssistantTurn: Identifiable {
    let id: UUID
    var messages: [AgentChatMessage]
}

struct AssistantTurnView: View {
    let turn: AssistantTurn
    let isGenerating: Bool
    
    @EnvironmentObject private var coordinator: AgentCoordinator
    @Binding var speakingMessageID: UUID?
    let isSpeechActive: Bool
    
    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    @State private var isExpanded: Bool = false
    @State private var hasAutoExpandedForGeneration: Bool = false

    /// Stable ID for the streaming final answer bubble — avoids UUID() churn breaking SwiftUI identity.
    private static let streamingAnswerID = UUID(uuidString: "00000000-0000-0000-0000-000000000002")!

    /// Progressive streaming message — reads through to agent's @Observable state.
    private var streamMessage: AssistantMessage? { coordinator.streamMessage }

    struct Step: Identifiable {
        let id: String
        let type: StepType
        
        enum StepType {
            case thinking(String)
            case text(String)
            case toolCall(ToolCall, result: AgentChatMessage?)
        }
    }
    
    private var steps: [Step] {
        var result: [Step] = []
        
        // Map tool calls
        for message in turn.messages {
            if message.role == .assistant {
                if let thinking = message.thinking, !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    result.append(Step(id: "\(message.id)-thinking", type: .thinking(thinking.trimmingCharacters(in: .whitespacesAndNewlines))))
                }
                
                if !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isFinalMessage(message) {
                    let trimmedText = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedText.isEmpty {
                        result.append(Step(id: "\(message.id)-text", type: .text(trimmedText)))
                    }
                }
                
                // Map tool calls
                for (index, toolCall) in message.toolCalls.enumerated() {
                    // Try to find matching tool result
                    // The coordinator matches tool calls to results by order or toolCallId
                    // For legacy, it might just be the next .tool message
                    var resultMsg: AgentChatMessage? = nil
                    
                    // Fallback logic by order
                    var pastMessage = false
                    for msg in turn.messages {
                        if msg.id == message.id {
                            pastMessage = true
                            continue
                        }
                        if pastMessage && msg.role == .tool {
                            let isUsed = result.contains { step in
                                if case .toolCall(_, let res) = step.type, res?.id == msg.id { return true }
                                return false
                            }
                            if !isUsed {
                                resultMsg = msg
                                break
                            }
                        }
                    }
                    
                    result.append(Step(id: "\(message.id)-tool-\(index)", type: .toolCall(toolCall, result: resultMsg)))
                }
            } else if message.role == .tool {
                // If this tool message hasn't been claimed by any assistant tool call, show it directly (should be rare)
                let isUsed = result.contains { step in
                    if case .toolCall(_, let res) = step.type, res?.id == message.id { return true }
                    return false
                }
                if !isUsed {
                    // Create a dummy tool call just to show the result
                    let dummyToolCall = ToolCall(function: .init(name: "unknown_tool", arguments: [:]))
                    result.append(Step(id: "\(message.id)-orphan-tool", type: .toolCall(dummyToolCall, result: message)))
                }
            } else if message.role == .system {
                 // For example context compaction messages
                 result.append(Step(id: "\(message.id)-text", type: .text(message.content)))
            }
        }
        
        // Add streaming state from progressive streamMessage
        if isGenerating, let stream = streamMessage {
            let streamChat = AgentChatMessage(from: stream)

            if let thinking = streamChat.thinking,
               !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(Step(id: "streaming-thinking", type: .thinking(thinking.trimmingCharacters(in: .whitespacesAndNewlines))))
            }

            // Show text as a step only when tool calls are present (intermediate text).
            // When no tool calls, text is the final answer — rendered in the bubble below.
            if !streamChat.toolCalls.isEmpty {
                let trimmed = streamChat.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    result.append(Step(id: "streaming-text", type: .text(trimmed)))
                }
            }

            // Show each streaming tool call as a step (no result yet)
            for (index, toolCall) in streamChat.toolCalls.enumerated() {
                result.append(Step(id: "streaming-tool-\(index)", type: .toolCall(toolCall, result: nil)))
            }
        }
        
        return result
    }
    
    private func isFinalMessage(_ message: AgentChatMessage) -> Bool {
        // A message is final if it's the last assistant message and has no tool calls
        return message.id == turn.messages.last(where: { $0.role == .assistant })?.id && message.toolCalls.isEmpty
    }
    
    /// Streaming final answer — text with no tool calls, shown as a bubble.
    private var streamingFinalAnswer: AgentChatMessage? {
        guard isGenerating, let stream = streamMessage, stream.toolCalls.isEmpty else { return nil }
        let txt = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !txt.isEmpty else { return nil }
        // Force thinking: nil — it's already rendered as a step above.
        // Use stable ID to preserve SwiftUI view identity across re-renders.
        return AgentChatMessage(
            id: Self.streamingAnswerID, timestamp: stream.timestamp, role: .assistant,
            content: txt, thinking: nil
        )
    }

    private var finalAnswer: AgentChatMessage? {
        let lastAssistant = turn.messages.last(where: { $0.role == .assistant })
        if let msg = lastAssistant, msg.toolCalls.isEmpty {
            let trimmedContent = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedContent.isEmpty { return nil } // Do not show empty final answers
            
            // Return a new message with trimmed content
            return AgentChatMessage(
                id: msg.id,
                timestamp: msg.timestamp,
                role: msg.role,
                content: trimmedContent,
                thinking: nil,
                toolCalls: msg.toolCalls,
                toolCallId: msg.toolCallId,
                isError: msg.isError
            )
        }
        return nil
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Steps Timeline
            if !steps.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    // Header toggle
                    Button(action: {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            isExpanded.toggle()
                            hasAutoExpandedForGeneration = false // Manual override
                        }
                    }) {
                        HStack(spacing: 4) {
                            Text("\(steps.count) step\(steps.count == 1 ? "" : "s")")
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)

                            if isGenerating {
                                ProgressView()
                                    .controlSize(.mini)
                            }

                            Image(systemName: "chevron.down")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundStyle(.tertiary)
                                .rotationEffect(.degrees(isExpanded ? 0 : -90))
                        }
                        .padding(.vertical, 4)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    
                    if isExpanded {
                        // Timeline
                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                                AgentStepView(step: step, isLast: index == steps.count - 1)
                            }
                        }
                        .padding(.leading, 18)
                        .padding(.top, 16)
                        .padding(.bottom, 8)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
            }
            
            // Pre-stream loading indicator — visible before first token arrives
            if isGenerating && steps.isEmpty && finalAnswer == nil && streamingFinalAnswer == nil {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
            }

            // Final Answer or Streaming Final Answer
            if let committed = finalAnswer {
                AssistantMessageBubble(
                    message: committed,
                    toolResults: [],
                    isSpeaking: speakingMessageID == committed.id && isSpeechActive,
                    onPlay: {
                        speakingMessageID = committed.id
                        coordinator.speakMessage(committed)
                    },
                    onStop: {
                        coordinator.stopSpeaking()
                        speakingMessageID = nil
                    }
                )
                .padding(.top, steps.isEmpty ? 0 : 4)
            } else if let streaming = streamingFinalAnswer {
                // No TTS controls during streaming — text is incomplete.
                AssistantMessageBubble(
                    message: streaming,
                    toolResults: [],
                    isSpeaking: false
                )
                .padding(.top, steps.isEmpty ? 0 : 4)
            }

            // Scroll anchor for streaming auto-scroll
            if isGenerating {
                Color.clear.frame(height: 0).id("streaming")
            }
        }
        .onChange(of: isGenerating) { _, generating in
            if generating {
                if !isExpanded && !hasAutoExpandedForGeneration {
                    withAnimation {
                        isExpanded = true
                        hasAutoExpandedForGeneration = true
                    }
                }
            } else {
                if hasAutoExpandedForGeneration {
                    withAnimation {
                        isExpanded = false
                        hasAutoExpandedForGeneration = false
                    }
                }
            }
        }
        .onAppear {
            if isGenerating {
                isExpanded = true
                hasAutoExpandedForGeneration = true
            }
        }
    }
}
