import Foundation
import MLXLMCommon
import os

/// The inference **dispatcher**: composes the two arms of server inference —
/// the cache-aware **Server Completion** arm (production adapter: `LLMActor`)
/// and the managed arm (production adapter: `AgentEngine`) — and owns the
/// **Completion Route**, the pure request-shape decision between them
/// (ADR-0006). Deleting this service would scatter that routing into the
/// completion handler; it earns its keep as the composition point of the two
/// adapters.
@MainActor
final class ServerInferenceService {
    private let completionStarter: any ServerCompletionStarting
    private let engine: any ManagedInferenceStarting
    private let modelStateProvider: @MainActor () -> ServerInferenceModelState?

    init(
        completionStarter: some ServerCompletionStarting,
        engine: some ManagedInferenceStarting,
        arbiter: InferenceArbiter
    ) {
        self.completionStarter = completionStarter
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
        completionStarter: some ServerCompletionStarting,
        engine: some ManagedInferenceStarting,
        modelStateProvider: @escaping @MainActor () -> ServerInferenceModelState?
    ) {
        self.completionStarter = completionStarter
        self.engine = engine
        self.modelStateProvider = modelStateProvider
    }

    func currentModelState() -> ServerInferenceModelState? {
        modelStateProvider()
    }

    func start(_ request: ServerInferenceRequest) async throws -> ServerInferenceStart {
        let modelState = currentModelState()
        let routeDescription: String = switch request.route {
        case .standard:
            "standard"
        case .serverCompatible:
            "serverCompatible"
        }
        let inputDescription: String = switch request.input {
        case .prompt:
            "prompt"
        case .chat:
            "chat"
        }
        Log.server.info(
            "Server inference start — route=\(routeDescription) input=\(inputDescription) "
            + "model=\(modelState?.modelID ?? "") visionMode=\(modelState?.visionMode ?? false)"
        )

        switch request.input {
        case .prompt(let prompt):
            let start = try engine.startPromptInference(
                prompt: prompt,
                parameters: request.parameters
            )
            return ServerInferenceStart(start, modelState: modelState)

        case .chat(let chat):
            switch request.route {
            case .standard:
                let start = try engine.startChatInference(
                    systemPrompt: chat.systemPrompt,
                    messages: chat.messages,
                    toolSpecs: chat.toolSpecs,
                    parameters: request.parameters,
                    progressHandler: chat.progressHandler
                )
                return ServerInferenceStart(start, modelState: modelState)

            case .serverCompatible:
                let modelState = modelState ?? .unavailable

                switch CompletionRoute.decide(conversation: chat.prefixCacheConversation) {
                case .cacheAware(let conversation):
                    let start = try await completionStarter.startServerCompletion(
                        modelID: modelState.modelID,
                        conversation: conversation,
                        toolSpecs: chat.toolSpecs,
                        parameters: request.parameters,
                        progressHandler: chat.progressHandler
                    )
                    Log.server.info(
                        "HTTP completion using prefix-cache path — model=\(modelState.modelID) "
                        + "cachedTokens=\(start.cachedTokenCount)"
                    )
                    return ServerInferenceStart(start, modelState: modelState)

                case .standard(let reason):
                    Log.server.info(
                        "HTTP completion using standard generation path — "
                        + "model=\(modelState.modelID) reason=\(reason.rawValue) "
                        + "toolDefinitions=\(chat.toolSpecs?.count ?? 0)"
                    )
                    let start = try engine.startChatInference(
                        systemPrompt: chat.systemPrompt,
                        messages: chat.messages,
                        toolSpecs: chat.toolSpecs,
                        parameters: request.parameters,
                        progressHandler: chat.progressHandler
                    )
                    return ServerInferenceStart(start, modelState: modelState)
                }
            }
        }
    }
}
