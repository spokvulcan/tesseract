import Foundation
import MLXLMCommon
import os

/// Temporary adapter for the pre-Epic-0 internal inference path.
@MainActor
protocol LegacyInternalInferenceEngine: AnyObject, Sendable {
    func generate(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error>

    func generate(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error>
}

private nonisolated struct InternalInferenceSelection: Sendable {
    let stream: AsyncThrowingStream<AgentGeneration, Error>
    let cancel: @Sendable () -> Void
    let waitForCompletion: @Sendable () async -> Void

    init(
        stream: AsyncThrowingStream<AgentGeneration, Error>,
        cancel: @escaping @Sendable () -> Void = {},
        waitForCompletion: @escaping @Sendable () async -> Void = {}
    ) {
        self.stream = stream
        self.cancel = cancel
        self.waitForCompletion = waitForCompletion
    }

    init(_ start: ServerInferenceStart) {
        self.init(
            stream: start.stream,
            cancel: start.cancel,
            waitForCompletion: start.waitForCompletion
        )
    }
}

private nonisolated struct InternalInferenceCancellationState: Sendable {
    var cancel: @Sendable () -> Void = {}
}

private nonisolated func makeInternalInferenceStream(
    inferenceService: ServerInferenceService,
    fallbackEngine: any LegacyInternalInferenceEngine,
    rollbackEnabled: @escaping @MainActor @Sendable () -> Bool,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters,
    requestBuilder: @escaping @MainActor @Sendable (AgentGenerateParameters) -> ServerInferenceRequest,
    fallbackStreamBuilder: @escaping @MainActor @Sendable (AgentGenerateParameters) throws -> AsyncThrowingStream<AgentGeneration, Error>
) -> AsyncThrowingStream<AgentGeneration, Error> {
    let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
    let cancellationState = OSAllocatedUnfairLock(
        initialState: InternalInferenceCancellationState()
    )

    let task = Task { @MainActor in
        var selection: InternalInferenceSelection?
        do {
            let parameters = parametersProvider()
            if rollbackEnabled() {
                selection = InternalInferenceSelection(
                    stream: try fallbackStreamBuilder(parameters)
                )
            } else {
                selection = InternalInferenceSelection(
                    try await inferenceService.start(requestBuilder(parameters))
                )
            }

            if let cancel = selection?.cancel {
                cancellationState.withLock { $0.cancel = cancel }
            }

            try Task.checkCancellation()

            if let selectedStream = selection?.stream {
                for try await generation in selectedStream {
                    try Task.checkCancellation()
                    continuation.yield(generation)
                }
            }

            continuation.finish()
        } catch is CancellationError {
            selection?.cancel()
            await selection?.waitForCompletion()
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    continuation.onTermination = { _ in
        cancellationState.withLock { $0.cancel() }
        task.cancel()
    }
    return stream
}

// MARK: - Summarize Closure Factory

/// Creates an `LLMGenerateFunction` that routes internal agent chat generation
/// through `ServerInferenceService` while keeping the managed chat path.
nonisolated func makeServerInferenceGenerateClosure(
    inferenceService: ServerInferenceService,
    fallbackEngine: any LegacyInternalInferenceEngine,
    rollbackEnabled: @escaping @MainActor @Sendable () -> Bool,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters
) -> LLMGenerateFunction {
    return { systemPrompt, messages, tools, _ in
        makeInternalInferenceStream(
            inferenceService: inferenceService,
            fallbackEngine: fallbackEngine,
            rollbackEnabled: rollbackEnabled,
            parametersProvider: parametersProvider,
            requestBuilder: { parameters in
                ServerInferenceRequest(
                    input: .chat(.init(
                        systemPrompt: systemPrompt,
                        messages: messages,
                        toolSpecs: tools?.map(\.toolSpec),
                        prefixCacheConversation: nil
                    )),
                    parameters: parameters
                )
            },
            fallbackStreamBuilder: { parameters in
                try fallbackEngine.generate(
                    systemPrompt: systemPrompt,
                    messages: messages,
                    toolSpecs: tools?.map(\.toolSpec),
                    parameters: parameters
                )
            }
        )
    }
}

/// Creates a `@Sendable` summarize closure that collects streamed text from the
/// shared inference service. Used by compaction (both automatic and `/compact`)
/// and background agents.
nonisolated func makeSummarizeClosure(
    inferenceService: ServerInferenceService,
    fallbackEngine: any LegacyInternalInferenceEngine,
    rollbackEnabled: @escaping @MainActor @Sendable () -> Bool,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters
) -> @Sendable (String) async throws -> String {
    return { prompt in
        let stream = makeInternalInferenceStream(
            inferenceService: inferenceService,
            fallbackEngine: fallbackEngine,
            rollbackEnabled: rollbackEnabled,
            parametersProvider: parametersProvider,
            requestBuilder: { parameters in
                ServerInferenceRequest(
                    input: .prompt(prompt),
                    parameters: parameters
                )
            },
            fallbackStreamBuilder: { parameters in
                try fallbackEngine.generate(
                    prompt: prompt,
                    parameters: parameters
                )
            }
        )
        var textContent = ""
        var thinkingContent: String?
        for try await gen in stream {
            switch gen {
            case .text(let chunk):
                textContent += chunk
            case .thinkStart:
                if thinkingContent == nil { thinkingContent = "" }
            case .thinking(let chunk):
                thinkingContent = (thinkingContent ?? "") + chunk
            case .thinkEnd:
                thinkingContent = nil
            case .thinkReclassify:
                textContent += thinkingContent ?? ""
                thinkingContent = nil
            case .toolCall, .malformedToolCall, .info:
                break
            }
        }
        return textContent
    }
}
