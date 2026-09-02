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

        // Vocabulary check before the lease (ADR-0060): an unknown
        // `reasoning_effort` is a client error regardless of which model is
        // loaded — on *either* channel, even the one precedence would ignore.
        // Whether the loaded model honors the value is decided later,
        // post-lease, from its template.
        if let raw = Self.requestedReasoningEffortRawValues(completionRequest)
            .first(where: { OpenAI.nativeReasoningEffort(fromWire: $0) == nil })
        {
            try await writer.send(
                .badRequest(
                    "Unsupported reasoning_effort '\(raw)'. Supported values: "
                        + "\(OpenAI.supportedReasoningEffortWireValues). To disable "
                        + "thinking, send chat_template_kwargs {\"enable_thinking\": false} instead."
                ))
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
        let generation: CompletionDelivery.Generation
        switch await startGeneration(
            request,
            sessionAffinity: sessionAffinity,
            completionID: completionID,
            logHandle: logHandle
        ) {
        case .success(let started):
            generation = started
        case .failure(let error):
            Log.server.error("Generation failed to start: \(error)")
            await activityLog.fail(handle: logHandle, error: error.localizedDescription)
            try? await writer.send(
                .serviceUnavailable("Generation failed: \(error.localizedDescription)"))
            return
        }

        // Transport is the only thing that varies past this point: the
        // Completion Delivery script runs once, behind the Delivery Sink seam.
        let envelope = CompletionDelivery.Envelope(
            completionID: generation.completionID,
            requestModel: request.model,
            physicalModelID: generation.modelID,
            created: Int(Date().timeIntervalSince1970)
        )
        let sink: any CompletionDeliverySink
        if request.stream == true {
            sink = SSEDeliverySink(
                wire: .live(sse: SSEWriter(writer), writer: writer),
                envelope: envelope,
                // The Argument Transcoder keys off the loaded model's
                // tool-call format — the same identity that selects the
                // parser; `nil` mirrors the parser's vendor JSON default
                // (ADR-0020).
                transcoder: ArgumentTranscoder(
                    format: generation.toolCallFormat ?? .json,
                    toolSpecs: generation.toolSpecs
                ),
                includeUsage: request.stream_options?.include_usage == true
            )
        } else {
            sink = NonStreamingDeliverySink(
                envelope: envelope,
                waitForDisconnect: { await writer.waitForDisconnect() },
                send: { try await writer.send(.jsonBody($0)) }
            )
        }

        let modelID = generation.modelID
        let visionMode = generation.visionMode
        await CompletionDelivery.deliver(
            generation,
            maxTokens: request.effectiveMaxTokens,
            sink: sink,
            activityLog: activityLog,
            logHandle: logHandle,
            recordReplay: { message in
                await Self.sessionReplayStore.record(
                    sessionAffinity: sessionAffinity,
                    modelID: modelID,
                    visionMode: visionMode,
                    assistantMessage: message
                )
            }
        )
    }

    /// Convert request, read model state, and start generation in one MainActor hop.
    private func startGeneration(
        _ request: OpenAI.ChatCompletionRequest,
        sessionAffinity: String?,
        completionID: String,
        logHandle: TraceHandle
    ) async -> Result<CompletionDelivery.Generation, Error> {
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
        // `enable_thinking` has no app setting — absent from `appDesired`, it
        // follows the template default and is emitted only on an explicit
        // request value.
        let requestedEffort = Self.requestedReasoningEffortRaw(request)
            .flatMap(OpenAI.nativeReasoningEffort(fromWire:))
        if requestedEffort != nil, !modelState.declaresReasoningEffort {
            Log.server.info(
                "reasoning_effort ignored — model=\(modelState.modelID) "
                    + "template does not declare the kwarg"
            )
        }
        let renderContext = TemplateRenderContext.resolve(
            requestKwargs: request.chat_template_kwargs?.booleanFlags,
            appDesired: [
                .preserveThinking: settings.preserveThinkingRender(modelID: modelState.modelID)
            ],
            declaredFlags: modelState.declaredTemplateFlags,
            templateDefaults: modelState.templateFlagDefaults,
            requestedReasoningEffort: requestedEffort,
            declaresReasoningEffort: modelState.declaresReasoningEffort,
            reasoningEffortTemplateDefault: modelState.reasoningEffortTemplateDefault
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
            userPreset: settings.samplingPreset,
            thinkingCutoffEnabled: settings.thinkingBudgetCutoffEnabled,
            thinkingCutoffChars: settings.thinkingBudgetCutoffChars
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
                        ),
                        clientStreams: request.stream == true
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

    /// The wire `reasoning_effort` for one request: the native
    /// `chat_template_kwargs` channel wins over the OpenAI top-level field
    /// (the same request-kwargs-first precedence the boolean flags use).
    nonisolated static func requestedReasoningEffortRaw(
        _ request: OpenAI.ChatCompletionRequest
    ) -> String? {
        requestedReasoningEffortRawValues(request).first
    }

    /// Every `reasoning_effort` value present on the wire, kwargs channel
    /// first. Precedence uses the first; validation rejects an unknown value
    /// on either channel, including the one precedence would ignore.
    nonisolated static func requestedReasoningEffortRawValues(
        _ request: OpenAI.ChatCompletionRequest
    ) -> [String] {
        [
            request.chat_template_kwargs?
                .stringValues[TemplateRenderContext.reasoningEffortKwargName],
            request.reasoning_effort,
        ].compactMap(\.self)
    }

    @MainActor
    static func makeGenerateParameters(
        from request: OpenAI.ChatCompletionRequest,
        modelState: ServerInferenceModelState,
        userPreset: SamplingPreset = .automatic,
        thinkingCutoffEnabled: Bool = SettingsCatalogue.thinkingBudgetCutoffEnabled.default,
        thinkingCutoffChars: Int = SettingsCatalogue.thinkingBudgetCutoffChars.default
    ) -> AgentGenerateParameters {
        var params = AgentGenerateParameters.forModel(modelState.modelID)
        params = userPreset.apply(to: params)
        // ADR-0060 budget split — before the vendor extension below, so an
        // explicit per-request `thinking_safeguard` stays authoritative.
        if modelState.declaresReasoningEffort {
            params.thinkingSafeguard.applyNativeReasoningEffortCeiling()
        } else {
            params.thinkingSafeguard.applyLegacyThinkingCutoff(
                enabled: thinkingCutoffEnabled, chars: thinkingCutoffChars)
        }
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
        case .speculationEngaged(let arm):
            activityLog.markSpeculationEngaged(handle: logHandle, arm: arm)
        }
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
