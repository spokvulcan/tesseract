import Foundation
import MLXLMCommon
import os

/// Handles `POST /v1/chat/completions` requests by acquiring an inference lease
/// from the `InferenceArbiter`, running generation through `AgentEngine`, and
/// writing the response.
///
/// All generation runs inside `arbiter.withExclusiveGPU(.llm)` to prevent
/// overlap with the internal Agent chat or other HTTP requests.
struct CompletionHandler: Sendable {

    /// Maximum seconds to wait for the inference lease before returning 503.
    static let leaseTimeoutSeconds: UInt64 = 60
    private static let sessionReplayStore = HTTPPrefixCacheSessionReplayStore()

    private let arbiter: InferenceArbiter
    private let engine: AgentEngine

    init(arbiter: InferenceArbiter, engine: AgentEngine) {
        self.arbiter = arbiter
        self.engine = engine
    }

    private struct StartedGeneration {
        let modelID: String
        let stream: AsyncThrowingStream<AgentGeneration, Error>
        let cachedTokenCount: Int
    }

    /// Entry point called by the HTTP server route.
    func handle(request: HTTPRequest, writer: HTTPResponseWriter) async throws {
        guard let body = request.body, !body.isEmpty else {
            try await writer.send(.badRequest("Request body is required"))
            return
        }

        let completionRequest: OpenAI.ChatCompletionRequest
        do {
            completionRequest = try JSONDecoder().decode(OpenAI.ChatCompletionRequest.self, from: body)
        } catch {
            try await writer.send(.badRequest("Invalid JSON: \(error.localizedDescription)"))
            return
        }

        guard !completionRequest.messages.isEmpty else {
            try await writer.send(.badRequest("messages array must not be empty"))
            return
        }

        let sessionAffinity = request.header("x-session-affinity")

        // File-based request logging — writes the raw request body to
        // tmp/tesseract-debug/http-completions/ for offline investigation of
        // prefix cache misses and tokenization drift.
        let logPrefix = HTTPRequestLogger.shared.logRequest(
            body: body, sessionAffinity: sessionAffinity
        )
        Log.server.info(
            "HTTP request logged — prefix=\(logPrefix) dir=\(HTTPRequestLogger.shared.directoryURL.path)"
        )

        do {
            try await withAcquisitionTimeout { signal in
                try await arbiter.withExclusiveGPU(.llm) {
                    signal.set()
                    await self.runCompletion(
                        completionRequest,
                        sessionAffinity: sessionAffinity,
                        writer: writer
                    )
                }
            }
        } catch is CancellationError {
            try await writer.send(.serviceUnavailable("Request cancelled"))
        } catch is LeaseTimeoutError {
            let base = HTTPResponse.serviceUnavailable("Model is busy, try again later")
            try await writer.send(HTTPResponse(
                statusCode: base.statusCode,
                statusText: base.statusText,
                headers: base.headers + [("Retry-After", "5")],
                body: base.body
            ))
        } catch let error as AgentEngineError {
            try await writer.send(.serviceUnavailable(error.localizedDescription))
        } catch {
            Log.server.error("Completion handler error: \(error)")
            try await writer.send(.internalError(error.localizedDescription))
        }
    }

    // MARK: - Private

    private func runCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter
    ) async {
        if request.stream == true {
            await runStreamingCompletion(
                request,
                sessionAffinity: sessionAffinity,
                writer: writer
            )
        } else {
            await runNonStreamingCompletion(
                request,
                sessionAffinity: sessionAffinity,
                writer: writer
            )
        }
    }

    /// Convert request, read model state, and start generation in one MainActor hop.
    private func startGeneration(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?
    ) async -> Result<StartedGeneration, Error> {
        let repairedRequest = await Self.sessionReplayStore.repair(
            messages: request.messages,
            sessionAffinity: sessionAffinity
        )
        let (systemPrompt, messages) = MessageConverter.convertMessages(repairedRequest.messages)
        let toolSpecs = MessageConverter.convertToolDefinitions(request.tools)
        let prefixCacheEligibility = MessageConverter.analyzePrefixCacheEligibility(
            repairedRequest.messages,
            tools: request.tools
        )
        let prefixCacheConversation = prefixCacheEligibility.conversation
        let modelID = await MainActor.run {
            arbiter.loadedLLMModelID ?? ""
        }
        var params = AgentGenerateParameters.forModel(modelID)
        if let maxTokens = request.effectiveMaxTokens { params.maxTokens = maxTokens }
        if let temp = request.temperature { params.temperature = Float(temp) }
        if let topP = request.top_p { params.topP = Float(topP) }

        Log.server.info(
            "HTTP completion reasoning sources — sessionAffinityPresent=\(sessionAffinity != nil) "
            + "client=\(repairedRequest.clientCount) "
            + "sessionRecovered=\(repairedRequest.sessionRecoveredCount) "
            + "missing=\(repairedRequest.missingCount)"
        )
        Log.server.info(
            "HTTP completion start — model=\(request.model ?? modelID) stream=\(request.stream == true) "
            + "messages=\(repairedRequest.messages.count) normalizedMessages=\(messages.count) "
            + "toolDefinitions=\(toolSpecs?.count ?? 0) prefixCache=\(prefixCacheEligibility) "
            + "maxTokens=\(params.maxTokens)"
        )

        do {
            let start = try await engine.generateServerTextCompletion(
                modelID: modelID,
                systemPrompt: systemPrompt ?? "",
                messages: messages,
                toolSpecs: toolSpecs,
                prefixCacheConversation: prefixCacheConversation,
                parameters: params
            )
            return .success(.init(
                modelID: modelID,
                stream: start.stream,
                cachedTokenCount: start.cachedTokenCount
            ))
        } catch {
            Log.server.error("HTTP completion failed to start generation: \(error)")
            return .failure(error)
        }
    }

    /// Non-streaming: accumulate all generation events, then send a single JSON response.
    private func runNonStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter
    ) async {
        let start: StartedGeneration
        switch await startGeneration(request, sessionAffinity: sessionAffinity) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Generation failed to start: \(error)")
            try? await writer.send(.serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        // Accumulate events
        var textContent = ""
        var thinkingContent = ""
        var toolCalls: [ToolCall] = []
        var info: AgentGeneration.Info?

        do {
            for try await event in start.stream {
                switch event {
                case .text(let chunk):
                    textContent += chunk
                case .thinkStart:
                    break
                case .thinking(let chunk):
                    thinkingContent += chunk
                case .thinkEnd:
                    break
                case .thinkReclassify:
                    textContent += thinkingContent
                    thinkingContent = ""
                case .toolCall(let call):
                    toolCalls.append(call)
                case .malformedToolCall(let raw):
                    Log.server.warning("Malformed tool call in HTTP response: \(raw)")
                case .info(let i):
                    info = i
                }
            }
        } catch {
            Log.server.error("Generation stream error: \(error)")
            try? await writer.send(.internalError("Generation error: \(error.localizedDescription)"))
            return
        }

        await Self.sessionReplayStore.record(
            sessionAffinity: sessionAffinity,
            assistantMessage: makeReplayAssistantMessage(
                textContent: textContent,
                thinkingContent: thinkingContent,
                toolCalls: toolCalls
            )
        )

        let finishReason: OpenAI.FinishReason
        if !toolCalls.isEmpty {
            finishReason = .tool_calls
        } else if let info, let maxTokens = request.effectiveMaxTokens,
                  info.generationTokenCount >= maxTokens {
            finishReason = .length
        } else {
            finishReason = .stop
        }

        let openAIToolCalls = toolCalls.isEmpty ? nil : ToolCallConverter.convertToOpenAI(toolCalls)

        let response = OpenAI.ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString)",
            model: request.model ?? start.modelID,
            created: Int(Date().timeIntervalSince1970),
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: finishReason,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        content: textContent.isEmpty ? nil : textContent,
                        reasoning_content: thinkingContent.isEmpty ? nil : thinkingContent,
                        tool_calls: openAIToolCalls
                    )
                ),
            ],
            usage: OpenAI.Usage(
                prompt_tokens: info?.promptTokenCount ?? 0,
                completion_tokens: info?.generationTokenCount ?? 0,
                total_tokens: (info?.promptTokenCount ?? 0) + (info?.generationTokenCount ?? 0),
                prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: start.cachedTokenCount)
            )
        )

        // Encodable conformance requires MainActor context (Swift 6.2 isolation inference)
        let data: Data = await MainActor.run {
            (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
        }
        Log.server.info(
            "HTTP completion finished — stream=false finishReason=\(finishReason.rawValue) "
            + "promptTokens=\(info?.promptTokenCount ?? 0) completionTokens=\(info?.generationTokenCount ?? 0) "
            + "cachedTokens=\(start.cachedTokenCount)"
        )
        try? await writer.send(.jsonBody(data))
    }

    // MARK: - Streaming Completion

    /// Streaming: emit SSE chunks as generation events arrive.
    private func runStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter
    ) async {
        let start: StartedGeneration
        switch await startGeneration(request, sessionAffinity: sessionAffinity) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Streaming generation failed to start: \(error)")
            try? await writer.send(.serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        let sse = SSEWriter(writer)
        do { try await sse.open() } catch {
            Log.server.error("Failed to open SSE stream: \(error)")
            return
        }

        let completionID = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        let model = request.model ?? start.modelID
        let includeUsage = request.stream_options?.include_usage == true

        // Emit initial chunk with role
        guard await sse.send(makeChunk(
            id: completionID, model: model, created: created,
            delta: OpenAI.ChunkDelta(role: .assistant)
        )) else {
            await sse.done()
            return
        }

        // Run generation and keepalive concurrently. If the client disconnects
        // during prefill, the keepalive write fails and throws ClientDisconnected,
        // cancelling the generation task and releasing the arbiter lease promptly.
        let streamResult: StreamResult
        do {
            streamResult = try await withThrowingTaskGroup(of: StreamResult?.self) { group in
                // Keepalive: SSE comments every 5s, throws on disconnect
                group.addTask {
                    while true {
                        try await Task.sleep(nanoseconds: 5_000_000_000)
                        try Task.checkCancellation()
                        guard await sse.keepalive("keepalive") else {
                            throw ClientDisconnected()
                        }
                    }
                }

                // Generation: stream events, return accumulated state
                group.addTask {
                    await self.streamGenerationEvents(
                        start.stream, sse: sse,
                        completionID: completionID, model: model, created: created
                    )
                }

                // First non-nil result is the generation task finishing
                let result = try await group.next() ?? nil
                group.cancelAll()
                return result ?? StreamResult()
            }
        } catch is ClientDisconnected {
            Log.server.debug("Client disconnected during streaming")
            return
        } catch is CancellationError {
            return
        } catch {
            Log.server.error("Streaming generation error: \(error)")
            return
        }

        // Skip final chunk if client disconnected mid-stream
        guard await !sse.isDisconnected else { return }

        await Self.sessionReplayStore.record(
            sessionAffinity: sessionAffinity,
            assistantMessage: makeReplayAssistantMessage(
                textContent: streamResult.textContent,
                thinkingContent: streamResult.thinkingContent,
                toolCalls: streamResult.toolCalls
            )
        )

        var finishReason: OpenAI.FinishReason = .stop
        if streamResult.hasToolCalls {
            finishReason = .tool_calls
        } else if let info = streamResult.info, let maxTokens = request.effectiveMaxTokens,
                  info.generationTokenCount >= maxTokens {
            finishReason = .length
        }

        var finalChunk = makeChunk(
            id: completionID, model: model, created: created,
            delta: OpenAI.ChunkDelta(),
            finishReason: finishReason
        )
        if includeUsage, let info = streamResult.info {
            finalChunk.usage = OpenAI.Usage(
                prompt_tokens: info.promptTokenCount,
                completion_tokens: info.generationTokenCount,
                total_tokens: info.promptTokenCount + info.generationTokenCount,
                prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: start.cachedTokenCount)
            )
        }
        Log.server.info(
            "HTTP completion finished — stream=true finishReason=\(finishReason.rawValue) "
            + "promptTokens=\(streamResult.info?.promptTokenCount ?? 0) "
            + "completionTokens=\(streamResult.info?.generationTokenCount ?? 0) "
            + "cachedTokens=\(start.cachedTokenCount)"
        )
        await sse.send(finalChunk)
        await sse.done()
    }

    private func makeChunk(
        id: String,
        model: String,
        created: Int,
        delta: OpenAI.ChunkDelta,
        finishReason: OpenAI.FinishReason? = nil
    ) -> OpenAI.ChatCompletionChunk {
        OpenAI.ChatCompletionChunk(
            id: id,
            model: model,
            created: created,
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: delta,
                    finish_reason: finishReason
                ),
            ]
        )
    }

    // MARK: - Stream Event Loop

    private struct StreamResult: Sendable {
        var hasToolCalls = false
        var info: AgentGeneration.Info?
        var textContent = ""
        var thinkingContent = ""
        var toolCalls: [ToolCall] = []
    }

    /// Consume generation events, emit SSE chunks, return accumulated metadata.
    private func streamGenerationEvents(
        _ stream: AsyncThrowingStream<AgentGeneration, Error>,
        sse: SSEWriter,
        completionID: String,
        model: String,
        created: Int
    ) async -> StreamResult {
        var result = StreamResult()
        var toolCallIndex = 0

        do {
            generation: for try await event in stream {
                switch event {
                case .text(let chunk):
                    result.textContent += chunk
                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(content: chunk)
                    )) else { break generation }

                case .thinkStart:
                    break

                case .thinking(let chunk):
                    result.thinkingContent += chunk
                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(reasoning_content: chunk)
                    )) else { break generation }

                case .thinkEnd:
                    break

                case .thinkReclassify:
                    result.textContent += result.thinkingContent
                    result.thinkingContent = ""
                    break

                case .toolCall(let call):
                    let index = toolCallIndex
                    toolCallIndex += 1
                    result.hasToolCalls = true
                    result.toolCalls.append(call)
                    let openAICalls = ToolCallConverter.convertToOpenAI([call])
                    guard let oaiCall = openAICalls.first else { continue }

                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(tool_calls: [
                            OpenAI.ToolCall(id: oaiCall.id, type: "function",
                                function: OpenAI.FunctionCall(name: oaiCall.function?.name, arguments: ""),
                                index: index),
                        ])
                    )) else { break generation }

                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(tool_calls: [
                            OpenAI.ToolCall(function: OpenAI.FunctionCall(arguments: oaiCall.function?.arguments),
                                index: index),
                        ])
                    )) else { break generation }

                case .malformedToolCall(let raw):
                    Log.server.warning("Malformed tool call in stream: \(raw)")

                case .info(let i):
                    result.info = i
                }
            }
        } catch {
            Log.server.error("Streaming generation error: \(error)")
        }

        return result
    }

    private func makeReplayAssistantMessage(
        textContent: String,
        thinkingContent: String,
        toolCalls: [ToolCall]
    ) -> HTTPPrefixCacheMessage {
        HTTPPrefixCacheMessage.assistant(
            content: textContent,
            reasoning: thinkingContent.isEmpty ? nil : thinkingContent,
            toolCalls: toolCalls.map {
                HTTPPrefixCacheToolCall(
                    name: $0.function.name,
                    arguments: $0.function.arguments
                )
            }
        )
    }

    /// Timeout that covers only lease acquisition + model loading, not generation.
    ///
    /// The timer task sleeps for the timeout duration, then checks whether the
    /// lease was acquired. If not, it throws `LeaseTimeoutError` which cancels
    /// the body (still waiting in the arbiter queue). If the lease WAS acquired,
    /// the timer suspends indefinitely — only the body's completion or failure
    /// will finish the group.
    private func withAcquisitionTimeout(
        body: @escaping @Sendable (LeaseAcquiredSignal) async throws -> Void
    ) async throws {
        try await Self.withAcquisitionTimeout(
            timeoutNanoseconds: Self.leaseTimeoutSeconds * 1_000_000_000,
            body: body
        )
    }

    /// Testable core: acquisition timeout with configurable duration.
    static func withAcquisitionTimeout(
        timeoutNanoseconds: UInt64,
        body: @escaping @Sendable (LeaseAcquiredSignal) async throws -> Void
    ) async throws {
        let signal = LeaseAcquiredSignal()

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await body(signal)
            }

            group.addTask {
                try await Task.sleep(nanoseconds: timeoutNanoseconds)
                if signal.isSet {
                    // Lease acquired — park until cancelled by group cleanup
                    while !Task.isCancelled {
                        try await Task.sleep(nanoseconds: 60 * 1_000_000_000)
                    }
                    return
                }
                throw LeaseTimeoutError()
            }

            // First to finish/throw wins — cancel the other
            try await group.next()
            group.cancelAll()
        }
    }
}

/// Thread-safe flag signaling that the inference lease has been acquired.
final class LeaseAcquiredSignal: Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: false)
    nonisolated var isSet: Bool { storage.withLock { $0 } }
    nonisolated func set() { storage.withLock { $0 = true } }
}

struct LeaseTimeoutError: Error {}
private struct ClientDisconnected: Error {}
