import Foundation

@MainActor
final class ServerInferenceService {
    private let engine: any ServerInferenceEngine
    private let modelStateProvider: @MainActor () -> ServerInferenceModelState?

    init(
        engine: some ServerInferenceEngine,
        arbiter: InferenceArbiter
    ) {
        self.engine = engine
        self.modelStateProvider = {
            arbiter.loadedLLMState.map {
                ServerInferenceModelState(
                    modelID: $0.modelID,
                    visionMode: $0.visionMode
                )
            }
        }
    }

    init(
        engine: some ServerInferenceEngine,
        modelStateProvider: @escaping @MainActor () -> ServerInferenceModelState?
    ) {
        self.engine = engine
        self.modelStateProvider = modelStateProvider
    }

    func currentModelState() -> ServerInferenceModelState? {
        modelStateProvider()
    }

    func start(_ request: ServerInferenceRequest) async throws -> ServerInferenceStart {
        switch request.input {
        case .prompt(let prompt):
            let start = try engine.startPromptInference(
                prompt: prompt,
                parameters: request.parameters
            )
            return ServerInferenceStart(start, modelState: currentModelState())

        case .chat(let chat):
            switch request.route {
            case .standard:
                let start = try engine.startChatInference(
                    systemPrompt: chat.systemPrompt,
                    messages: chat.messages,
                    toolSpecs: chat.toolSpecs,
                    parameters: request.parameters
                )
                return ServerInferenceStart(start, modelState: currentModelState())

            case .serverCompatible:
                let modelState = currentModelState() ?? .unavailable
                let start = try await engine.startServerChatInference(
                    modelID: modelState.modelID,
                    systemPrompt: chat.systemPrompt,
                    messages: chat.messages,
                    toolSpecs: chat.toolSpecs,
                    prefixCacheConversation: chat.prefixCacheConversation,
                    parameters: request.parameters
                )
                return ServerInferenceStart(start, modelState: modelState)
            }
        }
    }
}
