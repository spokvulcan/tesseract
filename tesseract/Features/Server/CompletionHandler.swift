import Foundation
import MLXLMCommon
import os

/// Handles `POST /v1/chat/completions` requests by acquiring an inference lease
/// from the `InferenceArbiter`, running generation through
/// `ServerInferenceService`, and writing the response.
///
/// All generation runs inside `arbiter.withExclusiveGPU(.llm)` to prevent
/// overlap with the internal Agent chat or other HTTP requests.
struct CompletionHandler: Sendable {

    /// Maximum seconds to wait for the inference lease before returning 503.
    static let leaseTimeoutSeconds: UInt64 = 60
    private static let sessionReplayStore = HTTPPrefixCacheSessionReplayStore()

    private let arbiter: InferenceArbiter
    private let inferenceService: ServerInferenceService
    private let downloads: ModelDownloadManager
    private let activityLog: ServerGenerationLog
    private let settings: SettingsManager

    init(
        arbiter: InferenceArbiter,
        inferenceService: ServerInferenceService,
        downloads: ModelDownloadManager,
        activityLog: ServerGenerationLog,
        settings: SettingsManager
    ) {
        self.arbiter = arbiter
        self.inferenceService = inferenceService
        self.downloads = downloads
        self.activityLog = activityLog
        self.settings = settings
    }

    private struct StartedGeneration {
        let modelID: String
        /// Physical vision-mode flag of the loaded container at generation
        /// start. Used alongside `modelID` to partition the session replay
        /// store so that recovered reasoning content cannot cross two
        /// different physical LLM slots with the same client session.
        let visionMode: Bool
        let completionID: String
        let stream: AsyncThrowingStream<AgentGeneration, Error>
        let cachedTokenCount: Int
        let cancel: @Sendable () -> Void
        let waitForCompletion: @Sendable () async -> Void
        let diagnostics: HTTPServerGenerationStart.Diagnostics
    }

    /// Routing decision for the request's `model` field.
    ///
    /// Consumed by `handle()` to short-circuit unknown/undownloaded requests
    /// with a 404 before queueing for the inference lease.
    ///
    /// Marked `nonisolated` so tests (and any other call site) can construct
    /// and compare values from outside the MainActor; Swift 6.2 would
    /// otherwise infer MainActor isolation from the enclosing type.
    nonisolated enum ModelSelection: Sendable, Equatable {
        /// Request.model is missing / empty / whitespace-only. Fall back to
        /// whatever Settings has selected (existing behavior).
        case useSettings
        /// Exact-match agent ID, downloaded and routable. Passed into the
        /// lease API as `llmModelIDOverride`.
        case override(String)
        /// Not in `ModelDefinition.all` filtered to `.agent`. Returns 404
        /// `model_not_found` with an "unknown" message.
        case unknown(String)
        /// In the catalog but `ModelDownloadManager.statuses[id]` reports
        /// anything other than `.downloaded`. Returns 404 with a
        /// "not downloaded — Settings → Models" message.
        case notDownloaded(String)
    }

    /// Resolve the request's `model` string into a routing decision.
    ///
    /// Exact match only on `ModelDefinition.id`. No displayName fallback, no
    /// repoID fallback, no case folding. Trimming is used **only** to detect
    /// whitespace-only strings (which normalize to `.useSettings` alongside
    /// nil and empty); non-empty values are compared verbatim so that subtle
    /// client config bugs like a trailing space surface as `.unknown` instead
    /// of silently matching.
    ///
    /// `nonisolated` because this is a pure function over value-type inputs —
    /// callable from tests without a MainActor hop. The caller (`handle()`)
    /// is responsible for reading `ModelDownloadManager.statuses` on the
    /// MainActor and passing the snapshot in.
    nonisolated static func resolveModelSelection(
        requestModel: String?,
        agentIDs: [String],
        statuses: [String: ModelStatus]
    ) -> ModelSelection {
        let raw = requestModel ?? ""
        let trimmedForEmptinessCheck = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedForEmptinessCheck.isEmpty { return .useSettings }
        guard agentIDs.contains(raw) else { return .unknown(raw) }
        guard case .downloaded = statuses[raw] else { return .notDownloaded(raw) }
        return .override(raw)
    }

    /// Decide what to put in the response's `model` field.
    ///
    /// OpenAI echoes back whatever the client sent, but we substitute the
    /// physical model ID when the client sent nothing / whitespace / empty —
    /// otherwise a request with `"model":"   "` round-trips as
    /// `"model":"   "` in the response body, which is a nonsense echo.
    ///
    /// `nonisolated` for the same reason as `resolveModelSelection` — pure
    /// function, no actor state touched.
    nonisolated static func echoModelID(
        requestModel: String?,
        physical: String
    ) -> String {
        guard let raw = requestModel else { return physical }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? physical : raw
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

        // Pre-lease validation of `request.model`. If the client asked for a
        // model we can't serve, return 404 `model_not_found` immediately
        // without touching the arbiter queue. Downloaded + in-catalog models
        // produce an `llmModelIDOverride` that flows into the lease API so
        // `ensureLoaded` targets it instead of `settingsManager.selectedAgentModelID`.
        let selection: ModelSelection = await MainActor.run {
            let agentIDs = ModelDefinition.all
                .filter { $0.category == .agent }
                .map(\.id)
            return Self.resolveModelSelection(
                requestModel: completionRequest.model,
                agentIDs: agentIDs,
                statuses: downloads.statuses
            )
        }

        let llmModelIDOverride: String?
        switch selection {
        case .useSettings:
            llmModelIDOverride = nil
        case .override(let id):
            llmModelIDOverride = id
        case .unknown(let id):
            try await writer.send(.modelNotFound(modelID: id, reason: .unknownID))
            return
        case .notDownloaded(let id):
            try await writer.send(.modelNotFound(modelID: id, reason: .notDownloaded))
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

        let completionID = "chatcmpl-\(UUID().uuidString)"
        let requestedModelName = Self.echoModelID(
            requestModel: completionRequest.model,
            physical: ""
        )
        let logHandle = await activityLog.startRequest(
            completionID: completionID,
            model: requestedModelName,
            stream: completionRequest.stream == true,
            sessionAffinity: sessionAffinity
        )

        do {
            try await withAcquisitionTimeout { signal in
                try await arbiter.withExclusiveGPU(
                    .llm,
                    llmModelIDOverride: llmModelIDOverride
                ) {
                    signal.set()
                    await self.activityLog.markLeaseAcquired(handle: logHandle)
                    await self.runCompletion(
                        completionRequest,
                        sessionAffinity: sessionAffinity,
                        writer: writer,
                        completionID: completionID,
                        logHandle: logHandle
                    )
                }
            }
        } catch is CancellationError {
            await activityLog.cancel(handle: logHandle)
            try await writer.send(.serviceUnavailable("Request cancelled"))
        } catch is LeaseTimeoutError {
            await activityLog.fail(handle: logHandle, error: "Model is busy")
            let base = HTTPResponse.serviceUnavailable("Model is busy, try again later")
            try await writer.send(HTTPResponse(
                statusCode: base.statusCode,
                statusText: base.statusText,
                headers: base.headers + [("Retry-After", "5")],
                body: base.body
            ))
        } catch AgentEngineError.modelNotDownloaded(let id) {
            await activityLog.fail(handle: logHandle, error: "Model not downloaded")
            // Post-lease race: validated pre-lease, then the model was
            // deleted from Settings → Models while we were queued. Surface
            // the same 404 `model_not_found` shape as the pre-lease path so
            // clients see one consistent error contract regardless of
            // whether the check failed before or after queueing.
            try await writer.send(.modelNotFound(modelID: id, reason: .notDownloaded))
        } catch let error as AgentEngineError {
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try await writer.send(.serviceUnavailable(error.localizedDescription))
        } catch {
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            Log.server.error("Completion handler error: \(error)")
            try await writer.send(.internalError(error.localizedDescription))
        }
    }

    // MARK: - Private

    private func runCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter,
        completionID: String,
        logHandle: TraceHandle
    ) async {
        if request.stream == true {
            await runStreamingCompletion(
                request,
                sessionAffinity: sessionAffinity,
                writer: writer,
                completionID: completionID,
                logHandle: logHandle
            )
        } else {
            await runNonStreamingCompletion(
                request,
                sessionAffinity: sessionAffinity,
                writer: writer,
                completionID: completionID,
                logHandle: logHandle
            )
        }
    }

    /// Convert request, read model state, and start generation in one MainActor hop.
    private func startGeneration(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        completionID: String
    ) async -> Result<StartedGeneration, Error> {
        let modelState = inferenceService.currentModelState() ?? .unavailable

        let repairedRequest = await Self.sessionReplayStore.repair(
            messages: request.messages,
            sessionAffinity: sessionAffinity,
            modelID: modelState.modelID,
            visionMode: modelState.visionMode
        )
        let (systemPrompt, messages) = MessageConverter.convertMessages(repairedRequest.messages)
        let toolSpecs = MessageConverter.convertToolDefinitions(request.tools)
        let prefixCacheEligibility = MessageConverter.analyzePrefixCacheEligibility(
            repairedRequest.messages,
            tools: request.tools
        )
        let prefixCacheConversation = prefixCacheEligibility.conversation
        let params = Self.makeGenerateParameters(
            from: request,
            modelState: modelState,
            userPreset: settings.samplingPreset
        )

        Log.server.info(
            "HTTP completion reasoning sources — sessionAffinityPresent=\(sessionAffinity != nil) "
            + "client=\(repairedRequest.clientCount) "
            + "sessionRecovered=\(repairedRequest.sessionRecoveredCount) "
            + "missing=\(repairedRequest.missingCount)"
        )
        Log.server.info(
            "HTTP completion start — completionID=\(completionID) "
            + "model=\(Self.echoModelID(requestModel: request.model, physical: modelState.modelID)) "
            + "stream=\(request.stream == true) "
            + "messages=\(repairedRequest.messages.count) normalizedMessages=\(messages.count) "
            + "toolDefinitions=\(toolSpecs?.count ?? 0) prefixCache=\(prefixCacheEligibility) "
            + "triAttentionEnabled=\(modelState.triAttention.enabled) "
            + "triAttentionFallbackReason=\(modelState.triAttentionFallbackReason?.rawValue ?? "none") "
            + "maxTokens=\(params.maxTokens)"
        )

        do {
            let inferenceRequest = ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: systemPrompt ?? "",
                    messages: messages,
                    toolSpecs: toolSpecs,
                    prefixCacheConversation: prefixCacheConversation
                )),
                parameters: params,
                route: .serverCompatible
            )
            let start = try await inferenceService.start(
                inferenceRequest
            )
            let startModelState = start.modelState ?? modelState
            return .success(.init(
                modelID: startModelState.modelID,
                visionMode: startModelState.visionMode,
                completionID: completionID,
                stream: start.stream,
                cachedTokenCount: start.cachedTokenCount,
                cancel: start.cancel,
                waitForCompletion: start.waitForCompletion,
                diagnostics: start.diagnostics
            ))
        } catch {
            Log.server.error("HTTP completion failed to start generation: \(error)")
            return .failure(error)
        }
    }

    @MainActor
    static func makeGenerateParameters(
        from request: OpenAI.ChatCompletionRequest,
        modelState: ServerInferenceModelState,
        userPreset: SamplingPreset = .automatic
    ) -> AgentGenerateParameters {
        var params = AgentGenerateParameters.forModel(modelState.modelID)
        params.triAttention = modelState.triAttention
        params = userPreset.apply(to: params)
        if let maxTokens = request.effectiveMaxTokens { params.maxTokens = maxTokens }
        if let temp = request.temperature { params.temperature = Float(temp) }
        if let topP = request.top_p { params.topP = Float(topP) }
        if let topK = request.top_k { params.topK = topK }
        if let minP = request.min_p { params.minP = Float(minP) }
        if let presencePenalty = request.presence_penalty {
            params.presencePenalty = Float(presencePenalty)
        }
        if let repetitionPenalty = request.repetition_penalty {
            let penalty = Float(repetitionPenalty)
            params.repetitionPenalty = penalty == 1.0 ? nil : penalty
        }
        if let frequencyPenalty = request.frequency_penalty {
            let penalty = Float(frequencyPenalty)
            params.frequencyPenalty = penalty == 0 ? nil : penalty
        }
        if let sg = request.thinking_safeguard {
            if let enabled = sg.enabled { params.thinkingSafeguard.enabled = enabled }
            if let m = sg.max_thinking_chars { params.thinkingSafeguard.maxThinkingChars = m }
            if let g = sg.min_chars_before_intervention {
                params.thinkingSafeguard.minCharsBeforeIntervention = g
            }
            if let r = sg.max_line_repeats { params.thinkingSafeguard.maxLineRepeats = r }
            if let msg = sg.injection_message { params.thinkingSafeguard.injectionMessage = msg }
        }
        return params
    }

    /// Non-streaming: accumulate all generation events, then send a single JSON response.
    private func runNonStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter,
        completionID: String,
        logHandle: TraceHandle
    ) async {
        let start: StartedGeneration
        switch await startGeneration(
            request,
            sessionAffinity: sessionAffinity,
            completionID: completionID
        ) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Generation failed to start: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(.serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        await recordCacheLookup(start: start, logHandle: logHandle)

        var textContent = ""
        var thinkingContent = ""
        var toolCalls: [ToolCall] = []
        var info: AgentGeneration.Info?
        var safeguardReport: OpenAI.ThinkingSafeguardReport?
        var malformedToolCallRaw = ""

        do {
            for try await event in start.stream {
                await activityLog.ingest(handle: logHandle, event: event)
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
                case .thinkTruncate(let safePrefix):
                    // Safeguard fired. Replace polluted reasoning with the clean
                    // prefix so the client (and the session replay store) never
                    // sees the garbage we buffered up to the trigger.
                    thinkingContent = safePrefix
                    safeguardReport = OpenAI.ThinkingSafeguardReport(
                        safePrefixChars: safePrefix.count
                    )
                case .toolCall(let call):
                    toolCalls.append(call)
                case .malformedToolCall(let raw):
                    malformedToolCallRaw += raw
                    Log.server.warning(
                        "Malformed tool call in HTTP response — "
                        + "completionID=\(start.completionID) "
                        + "rawLen=\(raw.count) "
                        + "head=\(String(raw.prefix(120)).debugDescription) "
                        + "tail=\(String(raw.suffix(80)).debugDescription)"
                    )
                case .toolCallDelta:
                    // Non-streaming path accumulates the final `.toolCall` event
                    // only; in-flight deltas are consumed by the activity log
                    // above (live Requests-log rendering) and not by this
                    // accumulator, which has no progressive output channel.
                    break
                case .info(let i):
                    info = i
                }
            }
        } catch {
            Log.server.error("Generation stream error: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(.internalError("Generation error: \(error.localizedDescription)"))
            return
        }

        // Surface a dropped tool-call buffer as text so the caller sees the
        // attempted tool call instead of an empty-stop response. See the
        // parallel block in the streaming path for the rationale.
        if toolCalls.isEmpty,
           textContent.isEmpty,
           !malformedToolCallRaw.isEmpty {
            textContent = malformedToolCallRaw
            Log.server.info(
                "Surfaced dropped tool-call buffer as text content — "
                + "completionID=\(start.completionID) rawLen=\(malformedToolCallRaw.count)"
            )
        }

        await Self.sessionReplayStore.record(
            sessionAffinity: sessionAffinity,
            modelID: start.modelID,
            visionMode: start.visionMode,
            assistantMessage: makeReplayAssistantMessage(
                textContent: textContent,
                thinkingContent: thinkingContent,
                toolCalls: toolCalls
            )
        )

        var response = Self.makeNonStreamingResponse(
            completionID: start.completionID,
            requestModel: request.model,
            physicalModelID: start.modelID,
            created: Int(Date().timeIntervalSince1970),
            textContent: textContent,
            thinkingContent: thinkingContent,
            toolCalls: toolCalls,
            info: info,
            cachedTokenCount: start.cachedTokenCount,
            maxTokens: request.effectiveMaxTokens
        )
        response.tesseract_thinking_safeguard = safeguardReport

        // Encodable conformance requires MainActor context (Swift 6.2 isolation inference)
        let data: Data = await MainActor.run {
            (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
        }
        let finishReason = response.choices[0].finish_reason ?? .stop
        Log.server.info(
            "HTTP completion finished — completionID=\(start.completionID) "
            + "stream=false finishReason=\(finishReason.rawValue) "
            + "promptTokens=\(info?.promptTokenCount ?? 0) completionTokens=\(info?.generationTokenCount ?? 0) "
            + "cachedTokens=\(start.cachedTokenCount)"
        )
        await activityLog.complete(handle: logHandle, finishReason: finishReason.rawValue)
        do {
            try await writer.send(.jsonBody(data))
        } catch {
            Log.server.error("Failed to send HTTP completion response: \(error)")
        }
    }

    // MARK: - Streaming Completion

    /// Streaming: emit SSE chunks as generation events arrive.
    private func runStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter,
        completionID: String,
        logHandle: TraceHandle
    ) async {
        let start: StartedGeneration
        switch await startGeneration(
            request,
            sessionAffinity: sessionAffinity,
            completionID: completionID
        ) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Streaming generation failed to start: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(.serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        await recordCacheLookup(start: start, logHandle: logHandle)

        let sse = SSEWriter(writer)
        do { try await sse.open() } catch {
            await cancelAndDrainGeneration(start)
            await activityLog.fail(handle: logHandle, error: "Failed to open SSE stream")
            Log.server.error("Failed to open SSE stream: \(error)")
            return
        }

        let created = Int(Date().timeIntervalSince1970)
        let model = Self.echoModelID(requestModel: request.model, physical: start.modelID)
        let includeUsage = request.stream_options?.include_usage == true
        let idleKeepaliveInterval: Duration = .milliseconds(250)

        // Emit initial chunk with role
        guard await sse.send(makeChunk(
            id: start.completionID, model: model, created: created,
            delta: OpenAI.ChunkDelta(role: .assistant)
        )) else {
            await cancelAndDrainGeneration(start)
            return
        }

        let outcome = await withTaskGroup(of: StreamingOutcome.self) { group in
            group.addTask {
                await writer.waitForDisconnect()
                guard !Task.isCancelled else { return .cancelled }
                start.cancel()
                return .disconnected(.connectionState)
            }

            group.addTask {
                // Keepalive: while the stream is idle, probe the transport
                // frequently so client aborts cancel long prefill promptly.
                while true {
                    do {
                        try await Task.sleep(for: idleKeepaliveInterval)
                        try Task.checkCancellation()
                    } catch is CancellationError {
                        return .cancelled
                    } catch {
                        return .failed(error.localizedDescription)
                    }

                    guard await sse.idleFor(atLeast: idleKeepaliveInterval) else {
                        continue
                    }

                    guard await sse.keepalive("keepalive") else {
                        start.cancel()
                        return .disconnected(.keepaliveWrite)
                    }
                }
            }

            group.addTask {
                await self.streamGenerationEvents(
                    start.stream,
                    sse: sse,
                    completionID: start.completionID,
                    model: model,
                    created: created,
                    cancel: start.cancel,
                    logHandle: logHandle
                )
            }

            let first = await group.next() ?? .cancelled
            group.cancelAll()
            return first
        }

        switch outcome {
        case .completed(let streamResult):
            var finishReason: OpenAI.FinishReason = .stop
            if streamResult.hasToolCalls {
                finishReason = .tool_calls
            } else if let info = streamResult.info, let maxTokens = request.effectiveMaxTokens,
                      info.generationTokenCount >= maxTokens {
                finishReason = .length
            }

            // Diagnostic log before any client bytes go out: correlates which
            // state inputs produced the finish_reason. Warning path catches
            // the exact shape of request #68 (stop with empty text AND empty
            // tool_calls but non-empty reasoning) — this is the
            // jundot/omlx#825 stale-recurrent-state symptom on Qwen3.6.
            let stopWithEmptyPayload = finishReason == .stop
                && streamResult.textContent.isEmpty
                && streamResult.toolCalls.isEmpty
            let finishReasonLog =
                "HTTP streaming finish_reason decision — "
                + "completionID=\(start.completionID) "
                + "finishReason=\(finishReason.rawValue) "
                + "textLen=\(streamResult.textContent.count) "
                + "toolCalls=\(streamResult.toolCalls.count) "
                + "reasoningLen=\(streamResult.thinkingContent.count) "
                + "malformedLen=\(streamResult.malformedToolCallRaw.count) "
                + "genTokens=\(streamResult.info?.generationTokenCount ?? 0) "
                + "maxTokens=\(request.effectiveMaxTokens ?? -1) "
                + "stopReason=\(streamResult.info.map { describeStopReason($0.stopReason) } ?? "nil")"
            let hadMalformed = !streamResult.malformedToolCallRaw.isEmpty
            if stopWithEmptyPayload && streamResult.thinkingContent.isEmpty == false {
                Log.server.warning("\(finishReasonLog) — EMPTY PAYLOAD WITH REASONING")
            } else if hadMalformed && streamResult.toolCalls.isEmpty {
                Log.server.warning("\(finishReasonLog) — MALFORMED TOOL CALL DROPPED")
            } else {
                Log.server.info("\(finishReasonLog)")
            }

            let safeguardReport: OpenAI.ThinkingSafeguardReport? =
                streamResult.thinkingSafeguardTriggered
                ? OpenAI.ThinkingSafeguardReport(
                    safePrefixChars: streamResult.thinkingSafeguardSafePrefixChars
                )
                : nil

            // Surface a dropped tool-call buffer as final text content when the
            // response would otherwise be empty. Without this the client sees
            // `finish_reason=stop` with empty `content` and empty `tool_calls`
            // and has no way to know the model attempted a tool call — it
            // treats the turn as "model chose to stop", so no retry happens at
            // the agent-loop layer upstream. Emitting the raw buffer lets the
            // caller detect the pattern (e.g. content contains `<tool_call>`)
            // and decide how to recover.
            var streamResult = streamResult
            if streamResult.toolCalls.isEmpty,
               streamResult.textContent.isEmpty,
               !streamResult.malformedToolCallRaw.isEmpty {
                let raw = streamResult.malformedToolCallRaw
                streamResult.textContent = raw
                _ = await sse.send(makeChunk(
                    id: start.completionID,
                    model: Self.echoModelID(
                        requestModel: request.model,
                        physical: start.modelID
                    ),
                    created: created,
                    delta: OpenAI.ChunkDelta(content: raw)
                ))
                Log.server.info(
                    "Surfaced dropped tool-call buffer as text content — "
                    + "completionID=\(start.completionID) rawLen=\(raw.count)"
                )
            }

            let finalChunk = Self.makeFinalStreamingChunk(
                completionID: start.completionID,
                requestModel: request.model,
                physicalModelID: start.modelID,
                created: created,
                hasToolCalls: streamResult.hasToolCalls,
                info: streamResult.info,
                cachedTokenCount: start.cachedTokenCount,
                maxTokens: request.effectiveMaxTokens,
                includeUsage: includeUsage,
                thinkingSafeguard: safeguardReport
            )

            guard await sse.send(finalChunk) else {
                start.cancel()
                await activityLog.cancel(handle: logHandle)
                Log.server.info(
                    "HTTP streaming disconnect — completionID=\(start.completionID) source=\(DisconnectSource.chunkWrite.rawValue)"
                )
                return
            }
            guard await sse.done() else {
                Log.server.info(
                    "HTTP streaming disconnect — completionID=\(start.completionID) source=\(DisconnectSource.chunkWrite.rawValue)"
                )
                await activityLog.cancel(handle: logHandle)
                return
            }

            await Self.sessionReplayStore.record(
                sessionAffinity: sessionAffinity,
                modelID: start.modelID,
                visionMode: start.visionMode,
                assistantMessage: makeReplayAssistantMessage(
                    textContent: streamResult.textContent,
                    thinkingContent: streamResult.thinkingContent,
                    toolCalls: streamResult.toolCalls
                )
            )

            Log.server.info(
                "HTTP completion finished — completionID=\(start.completionID) "
                + "stream=true finishReason=\(finishReason.rawValue) "
                + "promptTokens=\(streamResult.info?.promptTokenCount ?? 0) "
                + "completionTokens=\(streamResult.info?.generationTokenCount ?? 0) "
                + "cachedTokens=\(start.cachedTokenCount)"
            )
            await activityLog.complete(handle: logHandle, finishReason: finishReason.rawValue)

        case .disconnected(let source):
            Log.server.info(
                "HTTP streaming disconnect — completionID=\(start.completionID) source=\(source.rawValue)"
            )
            Log.server.debug("HTTP streaming cancel dispatched — completionID=\(start.completionID)")
            await cancelAndDrainGeneration(start)
            await activityLog.cancel(handle: logHandle)
            return

        case .failed(let message):
            await cancelAndDrainGeneration(start)
            await activityLog.fail(handle: logHandle, error: message)
            Log.server.error(
                "Streaming generation error — completionID=\(start.completionID) error=\(message)"
            )
            return

        case .cancelled:
            await cancelAndDrainGeneration(start)
            await activityLog.cancel(handle: logHandle)
            return
        }
    }

    private func recordCacheLookup(start: StartedGeneration, logHandle: TraceHandle) async {
        let diagnostics = start.diagnostics
        await activityLog.markCacheLookup(
            handle: logHandle,
            reason: diagnostics.cacheReason,
            cachedTokens: start.cachedTokenCount,
            sharedPrefixLength: diagnostics.sharedPrefixLength,
            promptTokens: diagnostics.promptTokenCount,
            lookupMs: diagnostics.lookupMs,
            restoreMs: diagnostics.restoreMs,
            prefillMs: diagnostics.prefillMs
        )
    }

    private func cancelAndDrainGeneration(_ start: StartedGeneration) async {
        start.cancel()
        await start.waitForCompletion()
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

    nonisolated static func finishReason(
        hasToolCalls: Bool,
        generationTokenCount: Int?,
        maxTokens: Int?
    ) -> OpenAI.FinishReason {
        if hasToolCalls {
            return .tool_calls
        }
        if let generationTokenCount, let maxTokens, generationTokenCount >= maxTokens {
            return .length
        }
        return .stop
    }

    nonisolated static func makeUsage(
        info: AgentGeneration.Info?,
        cachedTokenCount: Int
    ) -> OpenAI.Usage {
        OpenAI.Usage(
            prompt_tokens: info?.promptTokenCount ?? 0,
            completion_tokens: info?.generationTokenCount ?? 0,
            total_tokens: (info?.promptTokenCount ?? 0) + (info?.generationTokenCount ?? 0),
            prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: cachedTokenCount)
        )
    }

    static func makeNonStreamingResponse(
        completionID: String,
        requestModel: String?,
        physicalModelID: String,
        created: Int,
        textContent: String,
        thinkingContent: String,
        toolCalls: [ToolCall],
        info: AgentGeneration.Info?,
        cachedTokenCount: Int,
        maxTokens: Int?
    ) -> OpenAI.ChatCompletionResponse {
        let finishReason = finishReason(
            hasToolCalls: !toolCalls.isEmpty,
            generationTokenCount: info?.generationTokenCount,
            maxTokens: maxTokens
        )

        // Mirror of the streaming-path diagnostic at ~line 603: record the
        // state that produced this finish_reason so an empty-payload .stop
        // on the non-streaming path leaves the same log fingerprint.
        let stopWithEmptyPayload = finishReason == .stop
            && textContent.isEmpty
            && toolCalls.isEmpty
        let finishReasonLog =
            "HTTP non-streaming finish_reason decision — "
            + "completionID=\(completionID) "
            + "finishReason=\(finishReason.rawValue) "
            + "textLen=\(textContent.count) "
            + "toolCalls=\(toolCalls.count) "
            + "reasoningLen=\(thinkingContent.count) "
            + "genTokens=\(info?.generationTokenCount ?? 0) "
            + "maxTokens=\(maxTokens ?? -1) "
            + "stopReason=\(info.map { describeStopReason($0.stopReason) } ?? "nil")"
        if stopWithEmptyPayload && !thinkingContent.isEmpty {
            Log.server.warning("\(finishReasonLog) — EMPTY PAYLOAD WITH REASONING")
        } else {
            Log.server.info("\(finishReasonLog)")
        }

        let openAIToolCalls = toolCalls.isEmpty ? nil : ToolCallConverter.convertToOpenAI(toolCalls)

        return OpenAI.ChatCompletionResponse(
            id: completionID,
            model: echoModelID(requestModel: requestModel, physical: physicalModelID),
            created: created,
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
            usage: makeUsage(
                info: info,
                cachedTokenCount: cachedTokenCount
            )
        )
    }

    static func makeFinalStreamingChunk(
        completionID: String,
        requestModel: String?,
        physicalModelID: String,
        created: Int,
        hasToolCalls: Bool,
        info: AgentGeneration.Info?,
        cachedTokenCount: Int,
        maxTokens: Int?,
        includeUsage: Bool,
        thinkingSafeguard: OpenAI.ThinkingSafeguardReport? = nil
    ) -> OpenAI.ChatCompletionChunk {
        var chunk = OpenAI.ChatCompletionChunk(
            id: completionID,
            model: echoModelID(requestModel: requestModel, physical: physicalModelID),
            created: created,
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChunkChoice(
                    index: 0,
                    delta: OpenAI.ChunkDelta(),
                    finish_reason: finishReason(
                        hasToolCalls: hasToolCalls,
                        generationTokenCount: info?.generationTokenCount,
                        maxTokens: maxTokens
                    )
                ),
            ]
        )
        if includeUsage {
            if let info {
                chunk.usage = makeUsage(
                    info: info,
                    cachedTokenCount: cachedTokenCount
                )
            }
        }
        chunk.tesseract_thinking_safeguard = thinkingSafeguard
        return chunk
    }

    // MARK: - Stream Event Loop

    private struct StreamResult: Sendable {
        var hasToolCalls = false
        var info: AgentGeneration.Info?
        var textContent = ""
        var thinkingContent = ""
        var toolCalls: [ToolCall] = []
        /// Set when a `.thinkTruncate` event arrives mid-stream. Surfaces the
        /// safeguard firing to downstream response-emission code so it can emit
        /// a vendor sidecar / header.
        var thinkingSafeguardTriggered = false
        var thinkingSafeguardSafePrefixChars: Int?
        /// Accumulates raw content from `.malformedToolCall` events. When set,
        /// the model tried to emit a tool call that couldn't be parsed (usually
        /// vendor ToolCallProcessor's EOS-drop path for Qwen3.6 intermittent
        /// malformed output). Surfaced in the finish_reason decision log and
        /// used by the empty-content retry signal.
        var malformedToolCallRaw = ""
    }

    private enum DisconnectSource: String, Sendable {
        case connectionState = "connection_state"
        case keepaliveWrite = "keepalive_write"
        case chunkWrite = "chunk_write"
    }

    private enum StreamingOutcome: Sendable {
        case completed(StreamResult)
        case disconnected(DisconnectSource)
        case failed(String)
        case cancelled
    }

    /// Consume generation events, emit SSE chunks, return accumulated metadata.
    private func streamGenerationEvents(
        _ stream: AsyncThrowingStream<AgentGeneration, Error>,
        sse: SSEWriter,
        completionID: String,
        model: String,
        created: Int,
        cancel: @escaping @Sendable () -> Void,
        logHandle: TraceHandle
    ) async -> StreamingOutcome {
        var result = StreamResult()
        var toolCallIndex = 0

        do {
            generation: for try await event in stream {
                await activityLog.ingest(handle: logHandle, event: event)
                switch event {
                case .text(let chunk):
                    result.textContent += chunk
                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(content: chunk)
                    )) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .thinkStart:
                    break

                case .thinking(let chunk):
                    result.thinkingContent += chunk
                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(reasoning_content: chunk)
                    )) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .thinkEnd:
                    break

                case .thinkReclassify:
                    result.textContent += result.thinkingContent
                    result.thinkingContent = ""
                    break

                case .thinkTruncate(let safePrefix):
                    // Safeguard fired. Reset reasoning accumulator so the final
                    // replay / non-streaming response has a clean prefix even
                    // though streaming clients have already received the garbage
                    // chunks. The full intervention flow also emits a subsequent
                    // `.thinking(injectionMessage)` + `.thinkEnd`.
                    result.thinkingContent = safePrefix
                    result.thinkingSafeguardTriggered = true
                    result.thinkingSafeguardSafePrefixChars = safePrefix.count

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
                    )) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                    guard await sse.send(makeChunk(
                        id: completionID, model: model, created: created,
                        delta: OpenAI.ChunkDelta(tool_calls: [
                            OpenAI.ToolCall(function: OpenAI.FunctionCall(arguments: oaiCall.function?.arguments),
                                index: index),
                        ])
                    )) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .malformedToolCall(let raw):
                    result.malformedToolCallRaw += raw
                    Log.server.warning(
                        "Malformed tool call in stream — "
                        + "completionID=\(completionID) "
                        + "rawLen=\(raw.count) "
                        + "head=\(String(raw.prefix(120)).debugDescription) "
                        + "tail=\(String(raw.suffix(80)).debugDescription)"
                    )

                case .toolCallDelta:
                    // SSE forwarding of tool-call argument deltas is deferred
                    // (OpenAI protocol supports it via
                    // `choices[].delta.tool_calls[].function.arguments`, but
                    // the current ask is in-app UI streaming only — see the
                    // plan's Part B7). The activity log above already receives
                    // `.toolCallDelta` and drives live UI rendering.
                    break

                case .info(let i):
                    result.info = i
                }
            }
        } catch is CancellationError {
            return .cancelled
        } catch {
            return .failed(error.localizedDescription)
        }

        return .completed(result)
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
