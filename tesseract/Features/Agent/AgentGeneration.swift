import Foundation

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
}
