import Foundation
import MLXLMCommon

/// Parameters controlling text generation behavior.
struct AgentGenerateParameters: Sendable {
    var maxTokens: Int = 2048
    var temperature: Float = 0.6
    var topP: Float = 0.95
    var repetitionPenalty: Float? = nil
    var repetitionContextSize: Int = 20

    static let `default` = AgentGenerateParameters()
}

/// Events emitted during streaming text generation.
enum AgentGeneration: Sendable {
    /// A chunk of decoded text from the model.
    case text(String)

    /// A parsed tool call extracted from `<tool_call>` tags.
    case toolCall(ToolCall)

    /// A `<tool_call>` tag was found but contained malformed JSON.
    /// The associated string is the raw content between the tags.
    case malformedToolCall(String)

    /// The model started a `<think>` block.
    case thinkStart
    /// A streaming chunk of thinking content.
    case thinking(String)
    /// The model finished its `<think>` block.
    case thinkEnd

    /// Completion metrics emitted once generation finishes.
    case info(Info)

    struct Info: Sendable {
        let promptTokenCount: Int
        let generationTokenCount: Int
        let promptTime: TimeInterval
        let generateTime: TimeInterval

        var tokensPerSecond: Double {
            guard generateTime > 0 else { return 0 }
            return Double(generationTokenCount) / generateTime
        }
    }

    /// Bridge from ``ToolCallParser/Event`` to ``AgentGeneration``.
    init(parserEvent: ToolCallParser.Event) {
        switch parserEvent {
        case .text(let text): self = .text(text)
        case .toolCall(let call): self = .toolCall(call)
        case .malformedToolCall(let raw): self = .malformedToolCall(raw)
        case .thinkStart: self = .thinkStart
        case .thinking(let text): self = .thinking(text)
        case .thinkEnd: self = .thinkEnd
        }
    }
}
