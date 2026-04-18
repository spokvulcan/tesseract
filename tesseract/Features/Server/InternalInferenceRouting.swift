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
nonisolated func makeServerInferenceGenerateClosure(
    inferenceService: ServerInferenceService,
    parametersProvider: @escaping @MainActor @Sendable () -> AgentGenerateParameters
) -> LLMGenerateFunction {
    return { systemPrompt, messages, tools, _ in
        makeInternalInferenceStream(
            inferenceService: inferenceService,
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
            case .thinkTruncate(let safePrefix):
                thinkingContent = safePrefix
            case .toolCall, .malformedToolCall, .info:
                break
            }
        }
        return textContent
    }
}
