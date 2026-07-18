import Foundation
import MLXLMCommon
import os

// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable type_body_length
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
        /// Tool-call format of the loaded model — what the streaming path's
        /// Argument Transcoder keys off (`nil` ⇒ the vendor JSON default,
        /// mirroring the parser's own fallback).
        let toolCallFormat: ToolCallFormat?
        /// The request's converted tool definitions, for schema-typed
        /// argument transcoding.
        let toolSpecs: [ToolSpec]?
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
        guard ModelCatalog.isDownloaded(raw, statuses: statuses) else { return .notDownloaded(raw) }
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
            completionRequest = try JSONDecoder().decode(
                OpenAI.ChatCompletionRequest.self, from: body)
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
            let agentIDs = ModelDefinition.ids(in: .agent)
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

        // Pre-lease audio-capability guard (#358 story 7): input_audio parts
        // aimed at a model that cannot hear fail loudly with a 400 — never a
        // silent drop into an unkeyed text completion.
        let requestHasAudio = completionRequest.messages.contains { message in
            if case .parts(let parts) = message.content {
                return parts.contains { $0.type == .input_audio }
            }
            return false
        }
        if requestHasAudio {
            let (targetModelID, audioCapable) = await MainActor.run {
                let id = llmModelIDOverride ?? settings.selectedAgentModelID
                return (id, downloads.isAudioCapable(id))
            }
            guard audioCapable else {
                try await writer.send(
                    .badRequest(
                        "Model '\(targetModelID)' does not support audio input "
                            + "(input_audio content parts)"))
                return
            }
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
            sessionAffinity: sessionAffinity,
            inbound: RequestTrace.captureInbound(completionRequest.messages)
        )

        do {
            try await withAcquisitionTimeout { signal in
                try await arbiter.withExclusiveGPU(
                    .llm,
                    llmModelIDOverride: llmModelIDOverride,
                    // ADR-0008: HTTP requests load the vision variant whenever
                    // the target model is capable — the chat toggle never
                    // gates what a configured client was promised.
                    llmVision: .visionIfCapable
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
            try await writer.send(
                HTTPResponse(
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
        completionID: String,
        logHandle: TraceHandle
    ) async -> Result<StartedGeneration, Error> {
        let modelState = inferenceService.currentModelState() ?? .unavailable

        let repairedRequest = await Self.sessionReplayStore.repair(
            messages: request.messages,
            sessionAffinity: sessionAffinity,
            modelID: modelState.modelID,
            visionMode: modelState.visionMode
        )
        // Resolve the render context once (issue #98): request kwargs win,
        // the per-model setting is the fallback, and only template-declared
        // flags participate. The conversation digest and the render kwargs
        // both derive from this one value, so they can never disagree.
        let appEnabledFlags: Set<TemplateRenderFlag> =
            settings.preserveThinkingRender(modelID: modelState.modelID)
            ? [.preserveThinking] : []
        let renderContext = TemplateRenderContext.resolve(
            requestKwargs: request.chat_template_kwargs?.booleanFlags,
            appEnabledFlags: appEnabledFlags,
            declaredFlags: modelState.declaredTemplateFlags
        )
        let normalized = MessageConverter.normalizeRequest(
            repairedRequest.messages,
            tools: request.tools,
            templateContextDigest: renderContext.digest
        )
        let (systemPrompt, messages) = (normalized.systemPrompt, normalized.messages)
        let toolSpecs = MessageConverter.convertToolDefinitions(request.tools)
        let prefixCacheEligibility = normalized.prefixCacheEligibility
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
                + "maxTokens=\(params.maxTokens)"
        )

        do {
            let inferenceRequest = ServerInferenceRequest(
                input: .chat(
                    .init(
                        systemPrompt: systemPrompt ?? "",
                        messages: messages,
                        toolSpecs: toolSpecs,
                        prefixCacheConversation: prefixCacheConversation,
                        templateRenderContext: renderContext,
                        progressHandler: Self.makeProgressHandler(
                            activityLog: activityLog,
                            logHandle: logHandle
                        )
                    )),
                parameters: params,
                route: .serverCompatible
            )
            let start = try await inferenceService.start(
                inferenceRequest
            )
            let startModelState = start.modelState ?? modelState
            return .success(
                .init(
                    modelID: startModelState.modelID,
                    visionMode: startModelState.visionMode,
                    toolCallFormat: startModelState.toolCallFormat,
                    toolSpecs: toolSpecs,
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
            completionID: completionID,
            logHandle: logHandle
        ) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Generation failed to start: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(
                .serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        // Expose the transport-level cancel to the dashboard so an in-flight
        // generation can be stopped from inside the app, not just by client
        // disconnect.
        await activityLog.registerCancelAction(handle: logHandle, start.cancel)

        await recordCacheLookup(start: start, logHandle: logHandle)

        var accumulator = GenerationAccumulator()
        var info: AgentGeneration.Info?

        do {
            for try await event in start.stream {
                await activityLog.ingest(handle: logHandle, event: event)
                accumulator.ingest(event)
                switch event {
                case .malformedToolCall(let raw):
                    Log.server.warning(
                        "Malformed tool call in HTTP response — "
                            + "completionID=\(start.completionID) "
                            + "rawLen=\(raw.count) "
                            + "head=\(String(raw.prefix(120)).debugDescription) "
                            + "tail=\(String(raw.suffix(80)).debugDescription)"
                    )
                case .info(let i):
                    info = i
                default:
                    // text / thinking / tool-call accumulation is folded by the
                    // accumulator above; in-flight `.toolCallDelta`s are consumed
                    // only by the activity log (live Requests-log rendering).
                    break
                }
            }
        } catch {
            Log.server.error("Generation stream error: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(
                .internalError("Generation error: \(error.localizedDescription)"))
            return
        }

        // One Generation Projection maps the terminal accumulator to this path's
        // output: finish_reason, fallback-applied text, reasoning, tool calls,
        // the safeguard sidecar, and the finish-reason diagnostic.
        let projection = CompletionProjection(
            accumulator: accumulator,
            info: info,
            maxTokens: request.effectiveMaxTokens,
            completionID: start.completionID
        )

        // The caller logs the diagnostic the projection classified on pre-fallback
        // state — identical classification to the streaming path.
        projection.diagnostic.emit(label: "non-streaming")

        // Surface a dropped tool-call buffer as text so the caller sees the
        // attempted tool call instead of an empty-stop response (shared with the
        // streaming path, which additionally emits one SSE content chunk).
        if projection.malformedFallbackSurfaced {
            logSurfacedFallback(
                completionID: start.completionID,
                rawLen: projection.diagnostic.malformedLen
            )
        }

        await Self.sessionReplayStore.record(
            sessionAffinity: sessionAffinity,
            modelID: start.modelID,
            visionMode: start.visionMode,
            assistantMessage: makeReplayAssistantMessage(
                textContent: projection.textContent,
                thinkingContent: projection.thinkingContent,
                toolCalls: projection.toolCalls
            )
        )

        var response = Self.makeNonStreamingResponse(
            projection: projection,
            completionID: start.completionID,
            requestModel: request.model,
            physicalModelID: start.modelID,
            created: Int(Date().timeIntervalSince1970),
            cachedTokenCount: start.cachedTokenCount
        )
        response.tesseract_thinking_safeguard = projection.safeguardReport

        // Encodable conformance requires MainActor context (Swift 6.2 isolation inference)
        let data: Data = await MainActor.run {
            (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
        }
        let finishReason = projection.finishReason
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

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length
    /// Streaming: emit SSE chunks as generation events arrive.
    private func runStreamingCompletion(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        writer: HTTPResponseWriter,
        completionID: String,
        logHandle: TraceHandle
    ) async {
        // swiftlint:enable function_body_length
        let start: StartedGeneration
        switch await startGeneration(
            request,
            sessionAffinity: sessionAffinity,
            completionID: completionID,
            logHandle: logHandle
        ) {
        case .success(let started):
            start = started
        case .failure(let error):
            Log.server.error("Streaming generation failed to start: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(
                .serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        // Expose the transport-level cancel to the dashboard so an in-flight
        // generation can be stopped from inside the app, not just by client
        // disconnect.
        await activityLog.registerCancelAction(handle: logHandle, start.cancel)

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
        guard
            await sse.send(
                Self.makeChunk(
                    id: start.completionID, model: model, created: created,
                    delta: OpenAI.ChunkDelta(role: .assistant)
                ))
        else {
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
                await Self.streamGenerationEvents(
                    start.stream,
                    envelope: ChunkEnvelope(
                        completionID: start.completionID, model: model, created: created
                    ),
                    // The Argument Transcoder keys off the loaded model's
                    // tool-call format — the same identity that selects the
                    // parser; `nil` mirrors the parser's vendor JSON default
                    // (ADR-0020).
                    transcoder: ArgumentTranscoder(
                        format: start.toolCallFormat ?? .json,
                        toolSpecs: start.toolSpecs
                    ),
                    activityLog: self.activityLog,
                    logHandle: logHandle,
                    cancel: start.cancel,
                    send: { await sse.send($0) }
                )
            }

            let first = await group.next() ?? .cancelled
            group.cancelAll()
            return first
        }

        switch outcome {
        case .completed(let accumulator, let info, let wireStreamedToolCalls):
            // One Generation Projection — identical construction to the
            // non-streaming path — owns finish_reason, the malformed→text
            // fallback, the safeguard sidecar, and the diagnostic.
            let projection = CompletionProjection(
                accumulator: accumulator,
                info: info,
                maxTokens: request.effectiveMaxTokens,
                completionID: start.completionID
            )

            // Diagnostic log before the terminal chunk goes out: correlates which
            // state inputs produced the finish_reason. The warning paths catch a
            // stop with empty text AND empty tool_calls but non-empty reasoning
            // (the jundot/omlx#825 stale-recurrent-state symptom on Qwen3.6) and a
            // dropped malformed tool call — classified once, on pre-fallback state.
            projection.diagnostic.emit(label: "streaming")

            // Surface a dropped tool-call buffer as final text content when the
            // response would otherwise be empty. Without this the client sees
            // `finish_reason=stop` with empty `content` and empty `tool_calls`
            // and has no way to know the model attempted a tool call — it
            // treats the turn as "model chose to stop", so no retry happens at
            // the agent-loop layer upstream. Emitting the raw buffer (as the
            // final text content and one SSE content chunk) lets the caller
            // detect the pattern (e.g. content contains `<tool_call>`) and
            // decide how to recover.
            //
            // The fallback survives only where nothing streamed (ADR-0020):
            // once Argument Fragments went out, the attempted call is already
            // on the wire wire-valid — re-sending it as text would duplicate
            // it, so the extra content chunk is suppressed.
            if projection.malformedFallbackSurfaced {
                logSurfacedFallback(
                    completionID: start.completionID,
                    rawLen: projection.diagnostic.malformedLen
                )
                if !wireStreamedToolCalls {
                    _ = await sse.send(
                        Self.makeChunk(
                            id: start.completionID,
                            model: model,
                            created: created,
                            delta: OpenAI.ChunkDelta(content: projection.textContent)
                        ))
                }
            }

            // A call that streamed but never produced a parsed `.toolCall`
            // (malformation, dashboard cancel) leaves the projection at
            // `.stop` — but the wire carries a closed tool call, so the
            // finish reason must say so. `.length` keeps priority.
            let finishReason = Self.resolvedStreamingFinishReason(
                projection: projection.finishReason,
                wireStreamedToolCalls: wireStreamedToolCalls
            )

            let finalChunk = Self.makeFinalStreamingChunk(
                projection: projection,
                completionID: start.completionID,
                requestModel: request.model,
                physicalModelID: start.modelID,
                created: created,
                cachedTokenCount: start.cachedTokenCount,
                includeUsage: includeUsage,
                finishReasonOverride: finishReason
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
                    textContent: projection.textContent,
                    thinkingContent: projection.thinkingContent,
                    toolCalls: projection.toolCalls
                )
            )

            Log.server.info(
                "HTTP completion finished — completionID=\(start.completionID) "
                    + "stream=true finishReason=\(finishReason.rawValue) "
                    + "promptTokens=\(projection.info?.promptTokenCount ?? 0) "
                    + "completionTokens=\(projection.info?.generationTokenCount ?? 0) "
                    + "cachedTokens=\(start.cachedTokenCount)"
            )
            await activityLog.complete(
                handle: logHandle, finishReason: finishReason.rawValue)

        case .disconnected(let source):
            Log.server.info(
                "HTTP streaming disconnect — completionID=\(start.completionID) source=\(source.rawValue)"
            )
            Log.server.debug(
                "HTTP streaming cancel dispatched — completionID=\(start.completionID)")
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
        await activityLog.markCacheLookupFinished(
            handle: logHandle,
            reason: diagnostics.cacheReason,
            cachedTokens: start.cachedTokenCount,
            sharedPrefixLength: diagnostics.sharedPrefixLength,
            promptTokens: diagnostics.promptTokenCount,
            lookupMs: diagnostics.lookupMs,
            restoreMs: diagnostics.restoreMs,
            newTokensToPrefill: max(0, diagnostics.promptTokenCount - start.cachedTokenCount)
        )
        await activityLog.markPrefillFinished(
            handle: logHandle,
            prefillMs: diagnostics.prefillMs
        )
    }

    static func makeProgressHandler(
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle
    ) -> ServerInferenceProgressHandler {
        { event in
            applyProgressEvent(event, activityLog: activityLog, logHandle: logHandle)
        }
    }

    @MainActor
    static func applyProgressEvent(
        _ event: ServerInferenceProgressEvent,
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle
    ) {
        switch event {
        case .cacheLookupStarted:
            activityLog.markCacheLookupStarted(handle: logHandle)
        case .cacheLookupFinished(let info):
            activityLog.markCacheLookupFinished(
                handle: logHandle,
                reason: info.reason,
                cachedTokens: info.cachedTokens,
                sharedPrefixLength: info.sharedPrefixLength,
                promptTokens: info.promptTokens,
                lookupMs: info.lookupMs,
                restoreMs: info.restoreMs,
                newTokensToPrefill: info.newTokensToPrefill,
                divergence: info.divergence
            )
        case .prefillStarted(let info):
            activityLog.markPrefillStarted(
                handle: logHandle,
                promptTokens: info.promptTokens,
                cachedTokens: info.cachedTokens,
                newTokensToPrefill: info.newTokensToPrefill
            )
        case .prefillFinished(let info):
            activityLog.markPrefillFinished(
                handle: logHandle,
                prefillMs: info.prefillMs,
                promptTokens: info.promptTokens,
                cachedTokens: info.cachedTokens,
                newTokensToPrefill: info.newTokensToPrefill
            )
        }
    }

    private func cancelAndDrainGeneration(_ start: StartedGeneration) async {
        start.cancel()
        await start.waitForCompletion()
    }

    private nonisolated static func makeChunk(
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
                )
            ]
        )
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
        projection: CompletionProjection,
        completionID: String,
        requestModel: String?,
        physicalModelID: String,
        created: Int,
        cachedTokenCount: Int
    ) -> OpenAI.ChatCompletionResponse {
        let openAIToolCalls =
            projection.toolCalls.isEmpty
            ? nil
            : ToolCallConverter.convertToOpenAI(projection.toolCalls)

        return OpenAI.ChatCompletionResponse(
            id: completionID,
            model: echoModelID(requestModel: requestModel, physical: physicalModelID),
            created: created,
            system_fingerprint: "tesseract-1.0-mlx",
            choices: [
                OpenAI.ChatCompletionChoice(
                    index: 0,
                    finish_reason: projection.finishReason,
                    message: OpenAI.ResponseMessage(
                        role: .assistant,
                        content: projection.textContent.isEmpty ? nil : projection.textContent,
                        reasoning_content: projection.thinkingContent.isEmpty
                            ? nil : projection.thinkingContent,
                        tool_calls: openAIToolCalls
                    )
                )
            ],
            usage: makeUsage(
                info: projection.info,
                cachedTokenCount: cachedTokenCount
            )
        )
    }

    /// The streaming-path finish-reason resolution: once Argument Fragments
    /// streamed on the wire, a `.stop` (no parsed call survived — the
    /// transcoder closed the call wire-valid) must still read `tool_calls`;
    /// `.length` and an already-computed `tool_calls` pass through.
    nonisolated static func resolvedStreamingFinishReason(
        projection finishReason: OpenAI.FinishReason,
        wireStreamedToolCalls: Bool
    ) -> OpenAI.FinishReason {
        if wireStreamedToolCalls && finishReason == .stop {
            return .tool_calls
        }
        return finishReason
    }

    static func makeFinalStreamingChunk(
        projection: CompletionProjection,
        completionID: String,
        requestModel: String?,
        physicalModelID: String,
        created: Int,
        cachedTokenCount: Int,
        includeUsage: Bool,
        finishReasonOverride: OpenAI.FinishReason? = nil
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
                    finish_reason: finishReasonOverride ?? projection.finishReason
                )
            ]
        )
        if includeUsage, let info = projection.info {
            chunk.usage = makeUsage(
                info: info,
                cachedTokenCount: cachedTokenCount
            )
        }
        chunk.tesseract_thinking_safeguard = projection.safeguardReport
        return chunk
    }

    // MARK: - Stream Event Loop

    enum DisconnectSource: String, Sendable {
        case connectionState = "connection_state"
        case keepaliveWrite = "keepalive_write"
        case chunkWrite = "chunk_write"
    }

    enum StreamingOutcome: Sendable {
        /// The terminal Generation Accumulator plus captured completion metrics
        /// and whether the Argument Transcoder streamed tool-call fragments on
        /// the wire. Both completion paths build one `CompletionProjection`
        /// from the first two; the flag adjusts only this path's closure.
        case completed(GenerationAccumulator, AgentGeneration.Info?, wireStreamedToolCalls: Bool)
        case disconnected(DisconnectSource)
        case failed(String)
        case cancelled
    }

    /// The per-completion identity every streamed SSE chunk repeats. Bundled
    /// so tests can drive the stream event loop directly with an injected
    /// chunk sink (the production sink is `SSEWriter.send`).
    struct ChunkEnvelope: Sendable {
        let completionID: String
        let model: String
        let created: Int
    }

    /// Consume generation events, emit SSE chunks, return accumulated metadata.
    ///
    /// The Argument Transcoder owns every tool-call wire delta on this path:
    /// in-flight `.toolCallDelta`s become Argument Fragments for transcodable
    /// formats (Qwen XML, JSON wrapper), `.toolCall` closes the streamed call
    /// — or falls back to the atomic two-delta emission when nothing streamed
    /// — and any termination after engagement gets a Wire-Valid Close.
    nonisolated static func streamGenerationEvents(
        _ stream: AsyncThrowingStream<AgentGeneration, Error>,
        envelope: ChunkEnvelope,
        transcoder: ArgumentTranscoder,
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle,
        cancel: @escaping @Sendable () -> Void,
        send: @Sendable (OpenAI.ChatCompletionChunk) async -> Bool
    ) async -> StreamingOutcome {
        let completionID = envelope.completionID
        let model = envelope.model
        let created = envelope.created
        var accumulator = GenerationAccumulator()
        var info: AgentGeneration.Info?
        var transcoder = transcoder
        var loggedCrossCheckMismatches = 0

        // Send every wire tool-call delta the transcoder produced for one
        // event, one SSE chunk each. Returns false on client disconnect.
        func sendToolCallDeltas(_ wireCalls: [OpenAI.ToolCall]) async -> Bool {
            for wireCall in wireCalls {
                guard
                    await send(
                        makeChunk(
                            id: completionID, model: model, created: created,
                            delta: OpenAI.ChunkDelta(tool_calls: [wireCall])
                        ))
                else { return false }
            }
            return true
        }

        do {
            for try await event in stream {
                await activityLog.ingest(handle: logHandle, event: event)
                // Fold accumulated turn state in one place; the switch below
                // keeps only this path's SSE side effects (per-event deltas).
                accumulator.ingest(event)
                switch event {
                case .text(let chunk):
                    guard
                        await send(
                            makeChunk(
                                id: completionID, model: model, created: created,
                                delta: OpenAI.ChunkDelta(content: chunk)
                            ))
                    else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .thinking(let chunk):
                    guard
                        await send(
                            makeChunk(
                                id: completionID, model: model, created: created,
                                delta: OpenAI.ChunkDelta(reasoning_content: chunk)
                            ))
                    else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .toolCallDelta, .toolCall:
                    guard await sendToolCallDeltas(transcoder.ingest(event)) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }
                    if transcoder.crossCheckMismatchCount > loggedCrossCheckMismatches {
                        loggedCrossCheckMismatches = transcoder.crossCheckMismatchCount
                        Log.server.warning(
                            "Argument Transcoder cross-check mismatch — streamed "
                                + "fragments disagree semantically with the parsed tool "
                                + "call (wire not corrected) — completionID=\(completionID)"
                        )
                    }

                case .malformedToolCall(let raw):
                    Log.server.warning(
                        "Malformed tool call in stream — "
                            + "completionID=\(completionID) "
                            + "rawLen=\(raw.count) "
                            + "head=\(String(raw.prefix(120)).debugDescription) "
                            + "tail=\(String(raw.suffix(80)).debugDescription)"
                    )
                    // Wire-Valid Close for an engaged call — after fragments
                    // streamed there is no retraction, so the malformed→text
                    // fallback no longer applies to this call.
                    guard await sendToolCallDeltas(transcoder.ingest(event)) else {
                        cancel()
                        return .disconnected(.chunkWrite)
                    }

                case .info(let i):
                    info = i

                case .thinkStart, .thinkEnd, .thinkReclassify, .thinkTruncate:
                    // No SSE side effect. text/thinking state is folded by the
                    // accumulator above; reclassify/truncate only adjust the
                    // final accumulated content — deltas already sent to the
                    // client stand.
                    break
                }
            }
        } catch is CancellationError {
            return .cancelled
        } catch {
            return .failed(error.localizedDescription)
        }

        // Wire-Valid Close for a stream that terminated (dashboard cancel,
        // max-tokens, intervention) while a transcoded call was engaged: the
        // accumulated Argument Fragments must parse before the final chunk.
        guard await sendToolCallDeltas(transcoder.finish()) else {
            cancel()
            return .disconnected(.chunkWrite)
        }

        // Hand the terminal accumulator (plus completion metrics) to the caller;
        // both completion paths build one CompletionProjection from it.
        return .completed(
            accumulator, info, wireStreamedToolCalls: transcoder.hasStreamedFragments)
    }

    /// Emit the shared "surfaced dropped tool-call buffer" info-log. Both
    /// completion paths call this when `CompletionProjection.malformedFallbackSurfaced`
    /// is set, so the line has one home; `rawLen` reads through the projection's
    /// diagnostic rather than re-walking the raw accumulator buffer.
    private func logSurfacedFallback(completionID: String, rawLen: Int) {
        Log.server.info(
            "Surfaced dropped tool-call buffer as text content — "
                + "completionID=\(completionID) rawLen=\(rawLen)"
        )
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
// swiftlint:enable type_body_length

/// Thread-safe flag signaling that the inference lease has been acquired.
final class LeaseAcquiredSignal: Sendable {
    private let storage = OSAllocatedUnfairLock(initialState: false)
    nonisolated var isSet: Bool { storage.withLock { $0 } }
    nonisolated func set() { storage.withLock { $0 = true } }
}

struct LeaseTimeoutError: Error {}
