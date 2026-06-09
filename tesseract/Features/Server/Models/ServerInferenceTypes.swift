import Foundation
import MLXLMCommon

typealias ServerInferenceProgressHandler = @MainActor @Sendable (ServerInferenceProgressEvent) -> Void

nonisolated enum ServerInferenceProgressEvent: Sendable, Equatable {
    case cacheLookupStarted
    case cacheLookupFinished(CacheLookupInfo)
    case prefillStarted(PrefillInfo)
    case prefillFinished(PrefillInfo)

    nonisolated struct CacheLookupInfo: Sendable, Equatable {
        let reason: String
        let cachedTokens: Int
        let sharedPrefixLength: Int
        let promptTokens: Int
        let newTokensToPrefill: Int
        let lookupMs: Double
        let restoreMs: Double
    }

    nonisolated struct PrefillInfo: Sendable, Equatable {
        let promptTokens: Int
        let cachedTokens: Int
        let newTokensToPrefill: Int
        let prefillMs: Double?
    }
}

nonisolated struct ServerInferenceModelState: Sendable, Equatable {
    let modelID: String
    let visionMode: Bool

    static let unavailable = ServerInferenceModelState(
        modelID: "",
        visionMode: false
    )
}

nonisolated struct ServerInferenceStart: Sendable {
    let stream: AsyncThrowingStream<AgentGeneration, Error>
    let cachedTokenCount: Int
    let cancel: @Sendable () -> Void
    let waitForCompletion: @Sendable () async -> Void
    let modelState: ServerInferenceModelState?
    let diagnostics: HTTPServerGenerationStart.Diagnostics

    init(
        stream: AsyncThrowingStream<AgentGeneration, Error>,
        cachedTokenCount: Int,
        cancel: @escaping @Sendable () -> Void = {},
        waitForCompletion: @escaping @Sendable () async -> Void = {},
        modelState: ServerInferenceModelState? = nil,
        diagnostics: HTTPServerGenerationStart.Diagnostics = .unavailable
    ) {
        self.stream = stream
        self.cachedTokenCount = cachedTokenCount
        self.cancel = cancel
        self.waitForCompletion = waitForCompletion
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
        let progressHandler: ServerInferenceProgressHandler?

        init(
            systemPrompt: String,
            messages: [LLMMessage],
            toolSpecs: [ToolSpec]?,
            prefixCacheConversation: HTTPPrefixCacheConversation?,
            progressHandler: ServerInferenceProgressHandler? = nil
        ) {
            self.systemPrompt = systemPrompt
            self.messages = messages
            self.toolSpecs = toolSpecs
            self.prefixCacheConversation = prefixCacheConversation
            self.progressHandler = progressHandler
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

/// The dispatcher's cache-aware arm — one **Server Completion** on a servable
/// conversation shape (the **Completion Route** guarantees the shape before
/// this is called). Production adapter: `LLMActor`. Declared `nonisolated`
/// because the build's MainActor-default isolation would otherwise drag the
/// actor's conformance onto the main actor.
nonisolated protocol ServerCompletionStarting: AnyObject, Sendable {
    func startServerCompletion(
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPServerGenerationStart
}

/// The dispatcher's managed arm — the agent engine's lifecycle-managed
/// generation entries (busy flag, engine registry, engine-level cancel).
/// Production adapter: `AgentEngine`. Also serves the standard fallback for
/// HTTP requests the **Completion Route** bypasses, which is why the chat
/// entry carries the progress handler.
@MainActor
protocol ManagedInferenceStarting: AnyObject {
    func startPromptInference(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart

    func startChatInference(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler?
    ) throws -> HTTPServerGenerationStart
}
