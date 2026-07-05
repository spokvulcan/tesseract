import Foundation
import MLXLMCommon
import os

/// The inference **dispatcher**: composes the two arms of server inference —
/// the cache-aware **Server Completion** arm (production adapter: `LLMActor`)
/// and the managed arm (production adapter: `AgentEngine`) — and owns the
/// **Completion Route**, the pure request-shape decision between them
/// (ADR-0015). Deleting this service would scatter that routing into the
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
                    visionMode: $0.visionMode,
                    declaredTemplateFlags: arbiter.loadedDeclaredTemplateFlags,
                    toolCallFormat: arbiter.loadedToolCallFormat
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

    /// `lane` is non-nil only for a **pool lane** (PRD #173): a submission
    /// the Batch Engine admitted non-exclusively because the handler's
    /// submit-time probe said this request rides the cache-aware arm. A pool
    /// lane's generation must run engine-stepped, so the managed arm — whose
    /// GPU work is monolithic — is forbidden for it; the probe and the route
    /// compute the same pure decision over the same conversation shape, so
    /// the mismatch guard below is an invariant check, not a reachable path.
    func start(
        _ request: ServerInferenceRequest,
        lane: BatchLane? = nil
    ) async throws -> ServerInferenceStart {
        let modelState = currentModelState()
        let routeDescription: String =
            switch request.route {
            case .standard:
                "standard"
            case .serverCompatible:
                "serverCompatible"
            }
        let inputDescription: String =
            switch request.input {
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
            try Self.assertNoPoolLane(lane, arm: "prompt")
            let start = try engine.startPromptInference(
                prompt: prompt,
                parameters: request.parameters
            )
            return ServerInferenceStart(start, modelState: modelState)

        case .chat(let chat):
            switch request.route {
            case .standard:
                try Self.assertNoPoolLane(lane, arm: "standard-route")
                return try startStandardChat(
                    chat, parameters: request.parameters, modelState: modelState
                )

            case .serverCompatible:
                let modelState = modelState ?? .unavailable

                switch CompletionRoute.decide(conversation: chat.prefixCacheConversation) {
                case .cacheAware(let conversation):
                    let start = try await completionStarter.startServerCompletion(
                        modelID: modelState.modelID,
                        conversation: conversation,
                        toolSpecs: chat.toolSpecs,
                        parameters: request.parameters,
                        renderContext: chat.templateRenderContext,
                        progressHandler: chat.progressHandler,
                        lane: lane
                    )
                    Log.server.info(
                        "HTTP completion using prefix-cache path — model=\(modelState.modelID) "
                            + "cachedTokens=\(start.cachedTokenCount) "
                            + "poolLane=\(lane != nil)"
                    )
                    return ServerInferenceStart(start, modelState: modelState)

                case .standard(let reason):
                    try Self.assertNoPoolLane(lane, arm: "standard-fallback")
                    Log.server.info(
                        "HTTP completion using standard generation path — "
                            + "model=\(modelState.modelID) reason=\(reason.rawValue) "
                            + "toolDefinitions=\(chat.toolSpecs?.count ?? 0)"
                    )
                    return try startStandardChat(
                        chat, parameters: request.parameters, modelState: modelState
                    )
                }
            }
        }
    }

    /// A pool lane reaching a monolithic arm means the submit-time probe and
    /// the **Completion Route** disagreed — supposed to be impossible (both
    /// are the same pure decision). Fail the request loudly instead of
    /// running unstepped GPU work beside live sibling lanes.
    private nonisolated static func assertNoPoolLane(_ lane: BatchLane?, arm: String) throws {
        guard lane != nil else { return }
        Log.server.error("Pool lane routed to a monolithic arm — arm=\(arm)")
        throw AgentEngineError.generationFailed(
            "pool lane routed to the \(arm) arm — submit-time probe mismatch"
        )
    }

    /// The managed arm for chat input — shared by the `.standard` route and
    /// the **Completion Route**'s standard-with-reason fallback. Threads the
    /// request's render context through so opt-in flags (`preserve_thinking`)
    /// survive the standard path, not just the cache-aware arm.
    private func startStandardChat(
        _ chat: ServerInferenceRequest.ChatInput,
        parameters: AgentGenerateParameters,
        modelState: ServerInferenceModelState?
    ) throws -> ServerInferenceStart {
        let start = try engine.startChatInference(
            systemPrompt: chat.systemPrompt,
            messages: chat.messages,
            toolSpecs: chat.toolSpecs,
            parameters: parameters,
            renderContext: chat.templateRenderContext,
            progressHandler: chat.progressHandler
        )
        return ServerInferenceStart(start, modelState: modelState)
    }
}
