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

    private let arbiter: InferenceArbiter
    private let engine: AgentEngine

    init(arbiter: InferenceArbiter, engine: AgentEngine) {
        self.arbiter = arbiter
        self.engine = engine
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

        do {
            try await withAcquisitionTimeout { signal in
                try await arbiter.withExclusiveGPU(.llm) {
                    signal.set()
                    await self.runCompletion(completionRequest, writer: writer)
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
        writer: HTTPResponseWriter
    ) async {
        if request.stream == true {
            // T11 will implement streaming
            try? await writer.send(.serviceUnavailable("Streaming not yet implemented"))
            return
        }

        await runNonStreamingCompletion(request, writer: writer)
    }

    /// Non-streaming: accumulate all generation events, then send a single JSON response.
    private func runNonStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        writer: HTTPResponseWriter
    ) async {
        let (systemPrompt, messages) = MessageConverter.convertMessages(request.messages)
        let toolSpecs = MessageConverter.convertToolDefinitions(request.tools)

        var params = AgentGenerateParameters.forModel("")
        if let maxTokens = request.effectiveMaxTokens { params.maxTokens = maxTokens }
        if let temp = request.temperature { params.temperature = Float(temp) }
        if let topP = request.top_p { params.topP = Float(topP) }

        // Read model ID and start generation in a single MainActor hop
        let result: Result<(String, AsyncThrowingStream<AgentGeneration, Error>), Error>
        result = await MainActor.run {
            let modelID = arbiter.loadedLLMModelID ?? ""
            params = AgentGenerateParameters.forModel(modelID)
            if let maxTokens = request.effectiveMaxTokens { params.maxTokens = maxTokens }
            if let temp = request.temperature { params.temperature = Float(temp) }
            if let topP = request.top_p { params.topP = Float(topP) }
            do {
                let stream = try engine.generate(
                    systemPrompt: systemPrompt ?? "",
                    messages: messages,
                    toolSpecs: toolSpecs,
                    parameters: params
                )
                return .success((modelID, stream))
            } catch {
                return .failure(error)
            }
        }

        let modelID: String
        let generationStream: AsyncThrowingStream<AgentGeneration, Error>
        switch result {
        case .success(let (id, stream)):
            modelID = id
            generationStream = stream
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
            for try await event in generationStream {
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

        // Build response
        var content = textContent
        if !thinkingContent.isEmpty {
            content = "<think>\(thinkingContent)</think>\(textContent)"
        }

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
            model: request.model ?? modelID,
            created: Int(Date().timeIntervalSince1970),
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: finishReason,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        content: content.isEmpty ? nil : content,
                        tool_calls: openAIToolCalls
                    )
                ),
            ],
            usage: OpenAI.Usage(
                prompt_tokens: info?.promptTokenCount ?? 0,
                completion_tokens: info?.generationTokenCount ?? 0,
                total_tokens: (info?.promptTokenCount ?? 0) + (info?.generationTokenCount ?? 0),
                prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: 0)
            )
        )

        // Encodable conformance requires MainActor context (Swift 6.2 isolation inference)
        let data: Data = await MainActor.run {
            (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
        }
        try? await writer.send(.jsonBody(data))
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
