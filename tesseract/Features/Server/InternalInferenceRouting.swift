import Foundation
import os

private nonisolated func makeInternalInferenceStream(
    inferenceService: ServerInferenceService,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters,
    requestBuilder: @escaping @MainActor @Sendable (AgentGenerateParameters) -> ServerInferenceRequest
) -> AsyncThrowingStream<AgentGeneration, Error> {
    let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
    let cancelHandle = OSAllocatedUnfairLock<@Sendable () -> Void>(initialState: {})

    let task = Task { @MainActor in
        var start: ServerInferenceStart?
        do {
            let parameters = parametersProvider()
            start = try await inferenceService.start(requestBuilder(parameters))

            if let cancel = start?.cancel {
                cancelHandle.withLock { $0 = cancel }
            }

            try Task.checkCancellation()

            if let selectedStream = start?.stream {
                for try await generation in selectedStream {
                    try Task.checkCancellation()
                    continuation.yield(generation)
                }
            }

            continuation.finish()
        } catch is CancellationError {
            start?.cancel()
            await start?.waitForCompletion()
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    continuation.onTermination = { _ in
        cancelHandle.withLock { $0() }
        task.cancel()
    }
    return stream
}

// MARK: - Generate / Summarize Factories

/// Creates an `LLMGenerateFunction` that routes internal agent chat generation
/// through `ServerInferenceService`.
///
/// The request goes out server-compatible with the agent history canonicalized
/// into the prefix-cache conversation shape (PRD #72): the **Completion
/// Route** then decides cache-aware vs standard from the conversation shape,
/// exactly as it does for HTTP — agent chat and the OpenAI edge ride one
/// Server Completion machinery. A `nil` conversation (undecodable attachment)
/// falls back to the standard managed path, today's uncached behavior.
nonisolated func makeServerInferenceGenerateClosure(
    inferenceService: ServerInferenceService,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters
) -> LLMGenerateFunction {
    return { systemPrompt, messages, tools, _ in
        makeInternalInferenceStream(
            inferenceService: inferenceService,
            parametersProvider: parametersProvider,
            requestBuilder: { parameters in
                let toolSpecs = tools?.map(\.toolSpec)
                return ServerInferenceRequest(
                    input: .chat(.init(
                        systemPrompt: systemPrompt,
                        messages: messages,
                        toolSpecs: toolSpecs,
                        prefixCacheConversation: AgentConversationBuilder.conversation(
                            systemPrompt: systemPrompt,
                            messages: messages,
                            toolSpecs: toolSpecs
                        )
                    )),
                    parameters: parameters,
                    route: .serverCompatible
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
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters
) -> @Sendable (String) async throws -> String {
    return { prompt in
        let stream = makeInternalInferenceStream(
            inferenceService: inferenceService,
            parametersProvider: parametersProvider,
            requestBuilder: { parameters in
                ServerInferenceRequest(
                    input: .prompt(prompt),
                    parameters: parameters
                )
            }
        )
        // Fold the stream in one place; the summarizer's Generation Projection
        // is just the accumulated text (thinking and tool calls are
        // intentionally ignored). This drops the old `.thinkEnd`-nulls-thinking
        // step — the summarizer was the only consumer to discard closed-block
        // reasoning; it now folds like every other consumer.
        var accumulator = GenerationAccumulator()
        for try await gen in stream {
            accumulator.ingest(gen)
        }
        return accumulator.text
    }
}
