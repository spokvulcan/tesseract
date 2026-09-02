import Foundation
import MLXLMCommon

/// **Completion Delivery** — the one script that carries a started HTTP
/// completion from its first generation event to the client's last byte.
///
/// Before this module the streaming and non-streaming arms of
/// `CompletionHandler` were two hand copies of the same nine steps — register
/// the dashboard cancel, record the cache lookup, open the response, drive the
/// stream, project the terminal accumulator, emit the diagnostic, surface the
/// malformed→text fallback, record the session replay, log and complete — and
/// the only thing that differed was the transport. The script now exists once;
/// the transport sits behind the **Delivery Sink** seam with two adapters
/// (`NonStreamingDeliverySink`, `SSEDeliverySink`) plus whatever recording
/// peer a test supplies.
///
/// Ordering rules the script fixes for both transports:
/// - Every drive runs under the **Stream Lifecycle Driver**, so a client that
///   drops mid-generation cancels generation on *both* transports (the
///   non-streaming arm used to run bare and only learned of the drop when the
///   final write failed). A transport with no keepalive channel simply gets no
///   prober.
/// - The session replay record and the activity log's `complete` happen only
///   after the sink reports the response fully delivered. A body the client
///   never received is not replayed as the next turn's assistant message, and
///   the Requests log never says "completed" for bytes that never left the
///   socket.
/// - `Log.server` lines and `ServerGenerationLog` transitions have one home.
///
/// `nonisolated` so it composes with the handler's off-actor delivery with no
/// isolation change; every closure it holds is `@Sendable`.
nonisolated enum CompletionDelivery {

    // MARK: - Values

    /// One started generation, as delivery sees it: the identity every wire
    /// envelope repeats, the event stream, and the cancel/drain handles.
    struct Generation: Sendable {
        let modelID: String
        /// Physical vision-mode flag of the loaded container at generation
        /// start. Used alongside `modelID` to partition the session replay
        /// store so that recovered reasoning content cannot cross two
        /// different physical LLM slots with the same client session.
        let visionMode: Bool
        /// Tool-call format of the loaded model — what the SSE sink's
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

    /// The per-completion identity every response envelope repeats — one
    /// echoed model name and one `created` stamp per delivery, taken when the
    /// script starts, so the role chunk, every delta, the final chunk and a
    /// JSON body all agree. The wire envelopes are built here, from this one
    /// value, on both transports.
    struct Envelope: Sendable {
        let completionID: String
        /// The model name echoed on the wire (request verbatim when non-empty,
        /// else the physical id), resolved once.
        let model: String
        let created: Int

        init(completionID: String, requestModel: String?, physicalModelID: String, created: Int) {
            self.completionID = completionID
            self.model = CompletionHandler.echoModelID(
                requestModel: requestModel, physical: physicalModelID)
            self.created = created
        }

        /// One SSE chunk carrying a single delta.
        func chunk(delta: OpenAI.ChunkDelta) -> OpenAI.ChatCompletionChunk {
            OpenAI.ChatCompletionChunk(
                id: completionID,
                model: model,
                created: created,
                system_fingerprint: Self.systemFingerprint,
                choices: [
                    OpenAI.ChatCompletionChunkChoice(index: 0, delta: delta, finish_reason: nil)
                ]
            )
        }

        /// The terminal SSE chunk: empty delta, the finish reason, usage when
        /// the client asked for it, and the safeguard sidecar.
        func finalChunk(
            projection: CompletionProjection,
            finishReason: OpenAI.FinishReason,
            cachedTokenCount: Int,
            includeUsage: Bool
        ) -> OpenAI.ChatCompletionChunk {
            var chunk = OpenAI.ChatCompletionChunk(
                id: completionID,
                model: model,
                created: created,
                system_fingerprint: Self.systemFingerprint,
                choices: [
                    OpenAI.ChatCompletionChunkChoice(
                        index: 0, delta: OpenAI.ChunkDelta(), finish_reason: finishReason)
                ]
            )
            if includeUsage, let info = projection.info {
                chunk.usage = Self.usage(info: info, cachedTokenCount: cachedTokenCount)
            }
            chunk.tesseract_thinking_safeguard = projection.safeguardReport
            return chunk
        }

        /// The single JSON body of a non-streaming completion, safeguard
        /// sidecar included.
        func response(
            projection: CompletionProjection,
            finishReason: OpenAI.FinishReason,
            cachedTokenCount: Int
        ) -> OpenAI.ChatCompletionResponse {
            let openAIToolCalls =
                projection.toolCalls.isEmpty
                ? nil
                : ToolCallConverter.convertToOpenAI(projection.toolCalls)

            var response = OpenAI.ChatCompletionResponse(
                id: completionID,
                model: model,
                created: created,
                system_fingerprint: Self.systemFingerprint,
                choices: [
                    OpenAI.ChatCompletionChoice(
                        index: 0,
                        finish_reason: finishReason,
                        message: OpenAI.ResponseMessage(
                            role: .assistant,
                            content: projection.textContent.isEmpty ? nil : projection.textContent,
                            reasoning_content: projection.thinkingContent.isEmpty
                                ? nil : projection.thinkingContent,
                            tool_calls: openAIToolCalls
                        )
                    )
                ],
                usage: Self.usage(info: projection.info, cachedTokenCount: cachedTokenCount)
            )
            response.tesseract_thinking_safeguard = projection.safeguardReport
            return response
        }

        static let systemFingerprint = "tesseract-1.0-mlx"

        static func usage(info: AgentGeneration.Info?, cachedTokenCount: Int) -> OpenAI.Usage {
            OpenAI.Usage(
                prompt_tokens: info?.promptTokenCount ?? 0,
                completion_tokens: info?.generationTokenCount ?? 0,
                total_tokens: (info?.promptTokenCount ?? 0) + (info?.generationTokenCount ?? 0),
                prompt_tokens_details: OpenAI.PromptTokensDetails(cached_tokens: cachedTokenCount)
            )
        }
    }

    /// What the script hands the sink to close the response: the terminal
    /// **Generation Projection** and the finish reason the script resolved
    /// from it (wire-streamed tool calls promote a `.stop`).
    struct Terminal: Sendable {
        let projection: CompletionProjection
        let finishReason: OpenAI.FinishReason
        let cachedTokenCount: Int
    }

    /// The sink's answer to "the generation stream ended normally": either the
    /// wire is closed cleanly and reports whether Argument Fragments streamed,
    /// or the client is gone.
    enum StreamClose: Sendable {
        case closed(wireStreamedToolCalls: Bool)
        case disconnected
    }

    enum DisconnectSource: String, Sendable {
        case connectionState = "connection_state"
        case keepaliveWrite = "keepalive_write"
        case chunkWrite = "chunk_write"
        case bodyWrite = "body_write"
    }

    /// The drive's terminal state — what the Stream Lifecycle Driver races.
    enum StreamingOutcome: Sendable {
        /// The terminal Generation Accumulator plus captured completion
        /// metrics and whether the sink streamed tool-call fragments on the
        /// wire. The script builds one `CompletionProjection` from the first
        /// two; the flag adjusts only the finish reason.
        case completed(GenerationAccumulator, AgentGeneration.Info?, wireStreamedToolCalls: Bool)
        case disconnected(DisconnectSource)
        case failed(String)
        case cancelled
    }

    /// The delivery-time finish-reason rule: once Argument Fragments streamed
    /// on the wire, a `.stop` (no parsed call survived — the sink closed the
    /// call wire-valid) must still read `tool_calls`; `.length` and an
    /// already-computed `tool_calls` pass through. A sink that never streams
    /// fragments reports `false` and the projection's own reason stands.
    static func resolvedFinishReason(
        projection finishReason: OpenAI.FinishReason,
        wireStreamedToolCalls: Bool
    ) -> OpenAI.FinishReason {
        if wireStreamedToolCalls && finishReason == .stop {
            return .tool_calls
        }
        return finishReason
    }

    // MARK: - The script

    /// Deliver one started generation to one sink.
    ///
    /// - Parameters:
    ///   - generation: the started generation (stream, cancel, diagnostics).
    ///   - maxTokens: the request's effective `max_tokens`, for the projection's
    ///     finish-reason rule.
    ///   - sink: the transport adapter.
    ///   - activityLog: the Requests-log recorder; every transition it sees for
    ///     this request after start happens here.
    ///   - logHandle: this request's trace.
    ///   - keepaliveCadence: the driver's prober interval — production default;
    ///     tests shorten it.
    ///   - recordReplay: the session replay store's record verb, already bound
    ///     to this request's affinity and physical slot.
    static func deliver(
        _ generation: Generation,
        maxTokens: Int?,
        sink: any CompletionDeliverySink,
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle,
        keepaliveCadence: Duration = .milliseconds(250),
        recordReplay: @escaping @Sendable (HTTPPrefixCacheMessage) async -> Void
    ) async {
        let completionID = generation.completionID

        // Expose the transport-level cancel to the dashboard so an in-flight
        // generation can be stopped from inside the app, not just by client
        // disconnect.
        await activityLog.registerCancelAction(handle: logHandle, generation.cancel)

        await recordCacheLookup(generation, activityLog: activityLog, logHandle: logHandle)

        guard await sink.open() else {
            await cancelAndDrain(generation)
            await activityLog.fail(handle: logHandle, error: "Failed to open response")
            Log.server.error(
                "Failed to open HTTP completion response — completionID=\(completionID)")
            return
        }

        // The transport-lifecycle race — disconnect watch, idle keepalive
        // prober, and the drive as first-finisher-wins — lives in the driver.
        let outcome = await StreamLifecycleDriver.run(
            transport: sink.transport,
            keepaliveCadence: keepaliveCadence,
            onTransportCancel: generation.cancel,
            drive: {
                await pump(
                    generation.stream,
                    sink: sink,
                    activityLog: activityLog,
                    logHandle: logHandle,
                    cancel: generation.cancel
                )
            }
        )

        switch outcome {
        case .completed(let accumulator, let info, let wireStreamedToolCalls):
            // One Generation Projection owns finish_reason, the malformed→text
            // fallback, the safeguard sidecar, and the diagnostic.
            let projection = CompletionProjection(
                accumulator: accumulator,
                info: info,
                maxTokens: maxTokens,
                completionID: completionID
            )

            // Diagnostic log before the terminal bytes go out: correlates which
            // state inputs produced the finish_reason. The warning paths catch a
            // stop with empty text AND empty tool_calls but non-empty reasoning
            // (the jundot/omlx#825 stale-recurrent-state symptom on Qwen3.6) and
            // a dropped malformed tool call — classified once, on pre-fallback
            // state.
            projection.diagnostic.emit(label: sink.isStreaming ? "streaming" : "non-streaming")

            // Surface a dropped tool-call buffer as text so the caller sees the
            // attempted tool call instead of an empty-stop response. The SSE
            // sink additionally emits one content chunk when nothing streamed.
            if projection.malformedFallbackSurfaced {
                Log.server.info(
                    "Surfaced dropped tool-call buffer as text content — "
                        + "completionID=\(completionID) rawLen=\(projection.diagnostic.malformedLen)"
                )
            }

            let finishReason = resolvedFinishReason(
                projection: projection.finishReason,
                wireStreamedToolCalls: wireStreamedToolCalls
            )

            let delivered = await sink.finish(
                Terminal(
                    projection: projection,
                    finishReason: finishReason,
                    cachedTokenCount: generation.cachedTokenCount
                ))
            guard delivered else {
                generation.cancel()
                await activityLog.cancel(handle: logHandle)
                let source: DisconnectSource = sink.isStreaming ? .chunkWrite : .bodyWrite
                Log.server.info(
                    "HTTP completion disconnect — completionID=\(completionID) source=\(source.rawValue)"
                )
                return
            }

            await recordReplay(
                HTTPPrefixCacheMessage.assistant(
                    content: projection.textContent,
                    reasoning: projection.thinkingContent.isEmpty
                        ? nil : projection.thinkingContent,
                    toolCalls: projection.toolCalls.map {
                        HTTPPrefixCacheToolCall(
                            name: $0.function.name,
                            arguments: $0.function.arguments
                        )
                    }
                ))

            Log.server.notice(
                "HTTP completion finished — completionID=\(completionID) "
                    + "stream=\(sink.isStreaming) finishReason=\(finishReason.rawValue) "
                    + "promptTokens=\(info?.promptTokenCount ?? 0) "
                    + "completionTokens=\(info?.generationTokenCount ?? 0) "
                    + "cachedTokens=\(generation.cachedTokenCount) "
                    + "decodeTokS=\(String(format: "%.1f", info?.tokensPerSecond ?? 0))"
            )
            await activityLog.complete(handle: logHandle, finishReason: finishReason.rawValue)

        case .disconnected(let source):
            Log.server.info(
                "HTTP completion disconnect — completionID=\(completionID) source=\(source.rawValue)"
            )
            await cancelAndDrain(generation)
            await activityLog.cancel(handle: logHandle)

        case .failed(let message):
            await cancelAndDrain(generation)
            await activityLog.fail(handle: logHandle, error: message)
            Log.server.error(
                "HTTP completion generation error — completionID=\(completionID) error=\(message)"
            )

        case .cancelled:
            await cancelAndDrain(generation)
            await activityLog.cancel(handle: logHandle)
        }
    }

    /// Consume generation events: fold the accumulator, feed the Requests log,
    /// and hand each event to the sink for its wire side effect. The sink
    /// answering `false` means the client is gone.
    ///
    /// Internal seam — tests drive it with a scripted stream and a recording
    /// or SSE sink to assert the chunk stream a client would decode.
    static func pump(
        _ stream: AsyncThrowingStream<AgentGeneration, Error>,
        sink: any CompletionDeliverySink,
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle,
        cancel: @escaping @Sendable () -> Void
    ) async -> StreamingOutcome {
        var accumulator = GenerationAccumulator()
        var info: AgentGeneration.Info?

        do {
            for try await event in stream {
                await activityLog.ingest(handle: logHandle, event: event)
                // Fold accumulated turn state in one place; the sink keeps only
                // its transport's per-event side effects.
                accumulator.ingest(event)
                switch event {
                case .malformedToolCall(let raw):
                    Log.server.warning(
                        "Malformed tool call in HTTP completion — "
                            + "rawLen=\(raw.count) "
                            + "head=\(String(raw.prefix(120)).debugDescription) "
                            + "tail=\(String(raw.suffix(80)).debugDescription)"
                    )
                case .info(let i):
                    info = i
                default:
                    break
                }
                guard await sink.ingest(event) else {
                    cancel()
                    return .disconnected(.chunkWrite)
                }
            }
        } catch is CancellationError {
            return .cancelled
        } catch {
            return .failed(error.localizedDescription)
        }

        // Wire-Valid Close for a stream that terminated (dashboard cancel,
        // max-tokens, intervention) while a transcoded call was engaged: the
        // sink must close the wire before the terminal envelope.
        switch await sink.closeStream() {
        case .disconnected:
            cancel()
            return .disconnected(.chunkWrite)
        case .closed(let wireStreamedToolCalls):
            return .completed(accumulator, info, wireStreamedToolCalls: wireStreamedToolCalls)
        }
    }

    // MARK: - Private

    private static func recordCacheLookup(
        _ generation: Generation,
        activityLog: ServerGenerationLog,
        logHandle: TraceHandle
    ) async {
        let diagnostics = generation.diagnostics
        await activityLog.markCacheLookupFinished(
            handle: logHandle,
            reason: diagnostics.cacheReason,
            cachedTokens: generation.cachedTokenCount,
            sharedPrefixLength: diagnostics.sharedPrefixLength,
            promptTokens: diagnostics.promptTokenCount,
            lookupMs: diagnostics.lookupMs,
            restoreMs: diagnostics.restoreMs,
            newTokensToPrefill: max(0, diagnostics.promptTokenCount - generation.cachedTokenCount)
        )
        await activityLog.markPrefillFinished(
            handle: logHandle,
            prefillMs: diagnostics.prefillMs
        )
    }

    private static func cancelAndDrain(_ generation: Generation) async {
        generation.cancel()
        await generation.waitForCompletion()
    }
}

// MARK: - Delivery Sink seam

/// **Delivery Sink** — the transport adapter one Completion Delivery writes to.
///
/// The script owns accumulation, the Requests log, the projection, the replay
/// record and every log line; a sink owns only what its transport needs per
/// event and at close. Two production adapters (`NonStreamingDeliverySink`,
/// `SSEDeliverySink`) make the seam real; a recording peer makes it the test
/// surface.
nonisolated protocol CompletionDeliverySink: Sendable {
    /// Whether this transport streams deltas (labels diagnostics and logs).
    var isStreaming: Bool { get }
    /// The probes the Stream Lifecycle Driver races against the drive.
    var transport: StreamLifecycleDriver.Transport { get }
    /// Open the response. `false` ⇒ the client is gone; delivery aborts.
    func open() async -> Bool
    /// One generation event during the drive. `false` ⇒ the client is gone.
    func ingest(_ event: AgentGeneration) async -> Bool
    /// The stream ended normally: close any engaged wire state and report
    /// whether tool-call fragments streamed.
    func closeStream() async -> CompletionDelivery.StreamClose
    /// Write the terminal envelope. `true` only when the response was fully
    /// delivered.
    func finish(_ terminal: CompletionDelivery.Terminal) async -> Bool
}

// MARK: - Non-streaming adapter

/// The single-JSON-body adapter: nothing goes out until the projection exists.
/// Stateless, so a plain value — no actor hop per event.
struct NonStreamingDeliverySink: CompletionDeliverySink {
    let isStreaming = false
    let transport: StreamLifecycleDriver.Transport

    private let envelope: CompletionDelivery.Envelope
    private let send: @Sendable (Data) async throws -> Void

    /// - Parameters:
    ///   - envelope: the response identity.
    ///   - waitForDisconnect: suspends until the client drops (production:
    ///     `HTTPResponseWriter.waitForDisconnect`).
    ///   - send: writes the complete JSON body (production:
    ///     `HTTPResponseWriter.send(.jsonBody(_:))`).
    init(
        envelope: CompletionDelivery.Envelope,
        waitForDisconnect: @escaping @Sendable () async -> Void,
        send: @escaping @Sendable (Data) async throws -> Void
    ) {
        self.envelope = envelope
        self.send = send
        // No bytes can precede a JSON body, so there is no keepalive channel;
        // only the disconnect watch races the drive.
        self.transport = StreamLifecycleDriver.Transport(
            waitForDisconnect: waitForDisconnect,
            keepalive: nil
        )
    }

    func open() async -> Bool { true }

    func ingest(_ event: AgentGeneration) async -> Bool { true }

    func closeStream() async -> CompletionDelivery.StreamClose {
        .closed(wireStreamedToolCalls: false)
    }

    func finish(_ terminal: CompletionDelivery.Terminal) async -> Bool {
        let response = envelope.response(
            projection: terminal.projection,
            finishReason: terminal.finishReason,
            cachedTokenCount: terminal.cachedTokenCount
        )
        let data = (try? JSONEncoder().encode(response)) ?? Data("{}".utf8)
        do {
            try await send(data)
            return true
        } catch {
            Log.server.error("Failed to send HTTP completion response: \(error)")
            return false
        }
    }
}

// MARK: - SSE adapter

/// The server-sent-events adapter: role chunk on open, one chunk per delta,
/// Argument Fragments through the **Argument Transcoder**, final chunk and
/// `[DONE]` on finish.
actor SSEDeliverySink: CompletionDeliverySink {

    /// The SSE transport verbs, so the adapter runs without a socket in tests.
    struct Wire: Sendable {
        var open: @Sendable () async throws -> Void
        var send: @Sendable (OpenAI.ChatCompletionChunk) async -> Bool
        var done: @Sendable () async -> Bool
        var transport: StreamLifecycleDriver.Transport

        /// The production wire over one request's `SSEWriter` and its
        /// underlying `HTTPResponseWriter`.
        static func live(sse: SSEWriter, writer: HTTPResponseWriter) -> Wire {
            Wire(
                open: { try await sse.open() },
                send: { await sse.send($0) },
                done: { await sse.done() },
                transport: StreamLifecycleDriver.Transport(
                    waitForDisconnect: { await writer.waitForDisconnect() },
                    keepalive: .init(
                        idleFor: { await sse.idleFor(atLeast: $0) },
                        send: { await sse.keepalive("keepalive") }
                    )
                )
            )
        }
    }

    nonisolated let isStreaming = true
    nonisolated let transport: StreamLifecycleDriver.Transport

    private let wire: Wire
    private let envelope: CompletionDelivery.Envelope
    private let includeUsage: Bool
    private var transcoder: ArgumentTranscoder
    private var loggedCrossCheckMismatches = 0

    /// - Parameters:
    ///   - wire: the transport verbs.
    ///   - envelope: the identity every chunk repeats.
    ///   - transcoder: the Argument Transcoder keyed to the loaded model's
    ///     tool-call format (ADR-0020).
    ///   - includeUsage: the request's `stream_options.include_usage`.
    init(
        wire: Wire,
        envelope: CompletionDelivery.Envelope,
        transcoder: ArgumentTranscoder,
        includeUsage: Bool
    ) {
        self.wire = wire
        self.transport = wire.transport
        self.envelope = envelope
        self.transcoder = transcoder
        self.includeUsage = includeUsage
    }

    func open() async -> Bool {
        do {
            try await wire.open()
        } catch {
            Log.server.error("Failed to open SSE stream: \(error)")
            return false
        }
        // Emit initial chunk with role.
        return await wire.send(envelope.chunk(delta: OpenAI.ChunkDelta(role: .assistant)))
    }

    /// The Argument Transcoder owns every tool-call wire delta on this path:
    /// in-flight `.toolCallDelta`s become Argument Fragments for transcodable
    /// formats (Qwen XML, JSON wrapper), `.toolCall` closes the streamed call
    /// — or falls back to the atomic two-delta emission when nothing streamed
    /// — and any termination after engagement gets a Wire-Valid Close.
    func ingest(_ event: AgentGeneration) async -> Bool {
        switch event {
        case .text(let piece):
            return await wire.send(envelope.chunk(delta: OpenAI.ChunkDelta(content: piece)))

        case .thinking(let piece):
            return await wire.send(
                envelope.chunk(delta: OpenAI.ChunkDelta(reasoning_content: piece)))

        case .toolCallDelta, .toolCall:
            guard await sendToolCallDeltas(transcoder.ingest(event)) else { return false }
            if transcoder.crossCheckMismatchCount > loggedCrossCheckMismatches {
                loggedCrossCheckMismatches = transcoder.crossCheckMismatchCount
                Log.server.warning(
                    "Argument Transcoder cross-check mismatch — streamed "
                        + "fragments disagree semantically with the parsed tool "
                        + "call (wire not corrected) — completionID=\(envelope.completionID)"
                )
            }
            return true

        case .malformedToolCall:
            // Wire-Valid Close for an engaged call — after fragments streamed
            // there is no retraction, so the malformed→text fallback no longer
            // applies to this call.
            return await sendToolCallDeltas(transcoder.ingest(event))

        case .info, .thinkStart, .thinkEnd, .thinkReclassify, .thinkTruncate:
            // No SSE side effect. text/thinking state is folded by the script's
            // accumulator; reclassify/truncate only adjust the final accumulated
            // content — deltas already sent to the client stand.
            return true
        }
    }

    func closeStream() async -> CompletionDelivery.StreamClose {
        // Wire-Valid Close for a stream that terminated while a transcoded
        // call was engaged: the accumulated Argument Fragments must parse
        // before the final chunk.
        guard await sendToolCallDeltas(transcoder.finish()) else { return .disconnected }
        return .closed(wireStreamedToolCalls: transcoder.hasStreamedFragments)
    }

    func finish(_ terminal: CompletionDelivery.Terminal) async -> Bool {
        // The malformed→text fallback goes out as one content chunk only where
        // nothing streamed (ADR-0020): once Argument Fragments went out, the
        // attempted call is already on the wire wire-valid — re-sending it as
        // text would duplicate it.
        if terminal.projection.malformedFallbackSurfaced, !transcoder.hasStreamedFragments {
            guard
                await wire.send(
                    envelope.chunk(
                        delta: OpenAI.ChunkDelta(content: terminal.projection.textContent)))
            else { return false }
        }

        let finalChunk = envelope.finalChunk(
            projection: terminal.projection,
            finishReason: terminal.finishReason,
            cachedTokenCount: terminal.cachedTokenCount,
            includeUsage: includeUsage
        )
        guard await wire.send(finalChunk) else { return false }
        return await wire.done()
    }

    // MARK: Private

    /// Send every wire tool-call delta the transcoder produced for one
    /// event, one SSE chunk each. Returns false on client disconnect.
    private func sendToolCallDeltas(_ wireCalls: [OpenAI.ToolCall]) async -> Bool {
        for wireCall in wireCalls {
            guard await wire.send(envelope.chunk(delta: OpenAI.ChunkDelta(tool_calls: [wireCall])))
            else { return false }
        }
        return true
    }
}
