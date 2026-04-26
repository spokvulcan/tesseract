import Foundation
import MLXLMCommon

nonisolated struct ServerInferenceModelState: Sendable, Equatable {
    let modelID: String
    let visionMode: Bool
    let triAttention: TriAttentionConfiguration
    let triAttentionFallbackReason: TriAttentionDenseFallbackReason?

    init(
        modelID: String,
        visionMode: Bool,
        triAttention: TriAttentionConfiguration = .v1Disabled,
        triAttentionFallbackReason: TriAttentionDenseFallbackReason? = nil
    ) {
        self.modelID = modelID
        self.visionMode = visionMode
        self.triAttention = triAttention
        self.triAttentionFallbackReason = triAttentionFallbackReason
    }

    static let unavailable = ServerInferenceModelState(
        modelID: "",
        visionMode: false,
        triAttention: .v1Disabled,
        triAttentionFallbackReason: nil
    )
}

nonisolated struct ServerInferenceStart: Sendable {
    let stream: AsyncThrowingStream<AgentGeneration, Error>
    let cachedTokenCount: Int
    let cancel: @Sendable () -> Void
    let waitForCompletion: @Sendable () async -> Void
    let hasPostCompletionWork: Bool
    let postCompletionWork: @Sendable () async -> Void
    let modelState: ServerInferenceModelState?
    let diagnostics: HTTPServerGenerationStart.Diagnostics

    init(
        stream: AsyncThrowingStream<AgentGeneration, Error>,
        cachedTokenCount: Int,
        cancel: @escaping @Sendable () -> Void = {},
        waitForCompletion: @escaping @Sendable () async -> Void = {},
        hasPostCompletionWork: Bool = false,
        postCompletionWork: @escaping @Sendable () async -> Void = {},
        modelState: ServerInferenceModelState? = nil,
        diagnostics: HTTPServerGenerationStart.Diagnostics = .unavailable
    ) {
        self.stream = stream
        self.cachedTokenCount = cachedTokenCount
        self.cancel = cancel
        self.waitForCompletion = waitForCompletion
        self.hasPostCompletionWork = hasPostCompletionWork
        self.postCompletionWork = postCompletionWork
        self.modelState = modelState
        self.diagnostics = diagnostics
    }

    init(
        _ start: HTTPServerGenerationStart,
        modelState: ServerInferenceModelState? = nil
    ) {
        self.init(
            stream: start.stream,
            cachedTokenCount: start.cachedTokenCount,
            cancel: start.cancel,
            waitForCompletion: start.waitForCompletion,
            hasPostCompletionWork: start.hasPostCompletionWork,
            postCompletionWork: start.postCompletionWork,
            modelState: modelState,
            diagnostics: start.diagnostics
        )
    }
}

nonisolated struct ServerInferenceRequest: Sendable {
    nonisolated enum Route: Sendable {
        case standard
        case serverCompatible
    }

    nonisolated enum Input: Sendable {
        case prompt(String)
        case chat(ChatInput)
    }

    nonisolated struct ChatInput: Sendable {
        let systemPrompt: String
        let messages: [LLMMessage]
        let toolSpecs: [ToolSpec]?
        let prefixCacheConversation: HTTPPrefixCacheConversation?

        init(
            systemPrompt: String,
            messages: [LLMMessage],
            toolSpecs: [ToolSpec]?,
            prefixCacheConversation: HTTPPrefixCacheConversation?
        ) {
            self.systemPrompt = systemPrompt
            self.messages = messages
            self.toolSpecs = toolSpecs
            self.prefixCacheConversation = prefixCacheConversation
        }
    }

    let input: Input
    let parameters: AgentGenerateParameters
    let route: Route

    init(
        input: Input,
        parameters: AgentGenerateParameters,
        route: Route = .standard
    ) {
        self.input = input
        self.parameters = parameters
        self.route = route
    }
}

@MainActor
protocol ServerInferenceEngine: AnyObject {
    func startPromptInference(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart

    func startChatInference(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart

    func startServerChatInference(
        modelID: String,
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        prefixCacheConversation: HTTPPrefixCacheConversation?,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart
}
