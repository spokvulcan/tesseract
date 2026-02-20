import Foundation
import MLXLMCommon

/// Parameters controlling text generation behavior.
///
/// Defaults match the official Nanbeige4.1-3B recommendations:
/// temperature=0.6, top_p=0.95, repeat_penalty=1.0 (disabled), max_tokens=131072.
/// See: https://huggingface.co/Nanbeige/Nanbeige4.1-3B
struct AgentGenerateParameters: Sendable, Codable {
    var maxTokens: Int = 131_072
    var temperature: Float = 0.6
    var topP: Float = 0.95
    var repetitionPenalty: Float? = nil
    var repetitionContextSize: Int = 20

    static let `default` = AgentGenerateParameters()

    /// Qwen3-4B-Instruct-2507 recommended parameters for non-thinking mode.
    /// See: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    static let qwen3 = AgentGenerateParameters(
        temperature: 0.7,
        topP: 0.8
    )

    /// Returns the recommended parameters for a given model ID.
    static func forModel(_ modelID: String) -> AgentGenerateParameters {
        if modelID.hasPrefix("qwen3") { return .qwen3 }
        return .default
    }
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
