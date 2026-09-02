import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Recording sink

/// A Delivery Sink that records every verb the script calls, in order, and
/// answers each with a scripted result. Its transport never disconnects
/// unless told to.
private actor RecordingSink: CompletionDeliverySink {
    enum Call: Equatable {
        case open
        case ingest(String)
        case closeStream
        case finish(text: String, finishReason: OpenAI.FinishReason, cachedTokens: Int)
    }

    nonisolated let isStreaming = false
    nonisolated let transport: StreamLifecycleDriver.Transport

    private(set) var calls: [Call] = []
    private let acceptFinish: Bool

    init(
        acceptFinish: Bool = true,
        waitForDisconnect: @escaping @Sendable () async -> Void = {
            try? await Task.sleep(for: .seconds(60))
        }
    ) {
        self.acceptFinish = acceptFinish
        self.transport = StreamLifecycleDriver.Transport(
            waitForDisconnect: waitForDisconnect, keepalive: nil)
    }

    func open() async -> Bool {
        calls.append(.open)
        return true
    }

    func ingest(_ event: AgentGeneration) async -> Bool {
        switch event {
        case .text(let piece): calls.append(.ingest(piece))
        case .info: calls.append(.ingest("<info>"))
        default: calls.append(.ingest("<other>"))
        }
        return true
    }

    func closeStream() async -> CompletionDelivery.StreamClose {
        calls.append(.closeStream)
        return .closed(wireStreamedToolCalls: false)
    }

    func finish(_ terminal: CompletionDelivery.Terminal) async -> Bool {
        calls.append(
            .finish(
                text: terminal.projection.textContent,
                finishReason: terminal.finishReason,
                cachedTokens: terminal.cachedTokenCount
            ))
        return acceptFinish
    }
}

/// Records what the session replay store was asked to keep.
private actor ReplayRecorder {
    private(set) var messages: [HTTPPrefixCacheMessage] = []
    func record(_ message: HTTPPrefixCacheMessage) { messages.append(message) }
}

// MARK: - Scripted generation

private struct ScriptedGeneration {
    let generation: CompletionDelivery.Generation
    let cancelled: LeaseAcquiredSignal
    let drained: LeaseAcquiredSignal
}

private let scriptedInfo = GenerationFixtures.info(promptTokenCount: 12, generationTokenCount: 3)

/// A started generation whose stream yields `events` and then finishes. With
/// `hangAfterEvents`, the stream stays open after the last event until
/// `cancel` runs — the shape of a client dropping mid-decode.
private func scriptedGeneration(
    _ events: [AgentGeneration],
    hangAfterEvents: Bool = false,
    failure: (any Error)? = nil,
    cachedTokenCount: Int = 5
) -> ScriptedGeneration {
    let cancelled = LeaseAcquiredSignal()
    let drained = LeaseAcquiredSignal()
    let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
    for event in events { continuation.yield(event) }
    if let failure {
        continuation.finish(throwing: failure)
    } else if !hangAfterEvents {
        continuation.finish()
    }
    let generation = CompletionDelivery.Generation(
        modelID: "toy/model",
        visionMode: false,
        toolCallFormat: nil,
        toolSpecs: nil,
        completionID: "chatcmpl-delivery",
        stream: stream,
        cachedTokenCount: cachedTokenCount,
        cancel: {
            cancelled.set()
            continuation.finish()
        },
        waitForCompletion: { drained.set() },
        diagnostics: .unavailable
    )
    return ScriptedGeneration(generation: generation, cancelled: cancelled, drained: drained)
}

private struct ScriptedFailure: Error, LocalizedError {
    var errorDescription: String? { "scripted failure" }
}

/// One Requests-log trace to deliver into.
@MainActor
private func makeTrace(stream: Bool) -> (ServerGenerationLog, TraceHandle) {
    let log = ServerGenerationLog()
    let handle = log.startRequest(
        completionID: "chatcmpl-delivery", model: "toy/model", stream: stream, sessionAffinity: nil)
    return (log, handle)
}

/// `RequestTrace.Phase`'s Equatable conformance is MainActor-isolated, so the
/// comparison happens on the actor and only the verdict crosses.
@MainActor
private func hasPhase(_ log: ServerGenerationLog, _ expected: RequestTrace.Phase) -> Bool {
    log.traces.first?.phase == expected
}

// MARK: - The script

/// Drives `CompletionDelivery.deliver` — the one script both transports share —
/// with a scripted generation and a recording sink, asserting the ordering
/// rules the two hand-written arms used to disagree on.
struct CompletionDeliveryTests {

    /// The script's fixed order: open, one ingest per event, close, then
    /// finish with the projected text and finish reason. The replay record
    /// and the log's `complete` follow a delivered finish.
    @Test func deliversInOrderThenRecordsReplayAndCompletes() async {
        let scripted = scriptedGeneration([.text("Hel"), .text("lo"), .info(scriptedInfo)])
        let sink = RecordingSink()
        let replay = ReplayRecorder()
        let (log, handle) = await makeTrace(stream: false)

        await CompletionDelivery.deliver(
            scripted.generation,
            maxTokens: 64,
            sink: sink,
            activityLog: log,
            logHandle: handle,
            recordReplay: { await replay.record($0) }
        )

        #expect(
            await sink.calls == [
                .open, .ingest("Hel"), .ingest("lo"), .ingest("<info>"), .closeStream,
                .finish(text: "Hello", finishReason: .stop, cachedTokens: 5),
            ])
        let recorded = await replay.messages
        #expect(recorded.count == 1)
        #expect(recorded.first?.content == "Hello")
        #expect(await hasPhase(log, .completed))
        #expect(!scripted.cancelled.isSet)
    }

    /// A finish the sink could not deliver is a disconnect: the log reads
    /// cancelled and nothing is replayed as the next turn's assistant message.
    @Test func undeliveredFinishCancelsLogAndSkipsReplay() async {
        let scripted = scriptedGeneration([.text("gone"), .info(scriptedInfo)])
        let sink = RecordingSink(acceptFinish: false)
        let replay = ReplayRecorder()
        let (log, handle) = await makeTrace(stream: true)

        await CompletionDelivery.deliver(
            scripted.generation,
            maxTokens: nil,
            sink: sink,
            activityLog: log,
            logHandle: handle,
            recordReplay: { await replay.record($0) }
        )

        #expect(await replay.messages.isEmpty)
        #expect(await hasPhase(log, .cancelled))
        #expect(scripted.cancelled.isSet)
    }

    /// The non-streaming transport now runs under the Stream Lifecycle Driver
    /// too: a client that drops mid-generation cancels generation, drains it,
    /// and marks the request cancelled — instead of decoding to the end and
    /// failing on the final write.
    @Test func clientDisconnectCancelsGenerationOnNonStreamingTransport() async {
        let scripted = scriptedGeneration([.text("partial")], hangAfterEvents: true)
        let sink = RecordingSink(
            waitForDisconnect: { try? await Task.sleep(for: .milliseconds(50)) })
        let replay = ReplayRecorder()
        let (log, handle) = await makeTrace(stream: false)

        await CompletionDelivery.deliver(
            scripted.generation,
            maxTokens: nil,
            sink: sink,
            activityLog: log,
            logHandle: handle,
            keepaliveCadence: .milliseconds(10),
            recordReplay: { await replay.record($0) }
        )

        #expect(scripted.cancelled.isSet)
        #expect(scripted.drained.isSet)
        #expect(await hasPhase(log, .cancelled))
        #expect(await replay.messages.isEmpty)
        let calls = await sink.calls
        #expect(!calls.contains { if case .finish = $0 { return true } else { return false } })
    }

    /// A generation stream that throws reads as a failure: drained, logged
    /// failed, nothing delivered or replayed.
    @Test func streamFailureFailsTheRequest() async {
        let scripted = scriptedGeneration([.text("x")], failure: ScriptedFailure())
        let sink = RecordingSink()
        let replay = ReplayRecorder()
        let (log, handle) = await makeTrace(stream: false)

        await CompletionDelivery.deliver(
            scripted.generation,
            maxTokens: nil,
            sink: sink,
            activityLog: log,
            logHandle: handle,
            recordReplay: { await replay.record($0) }
        )

        #expect(await hasPhase(log, .failed))
        #expect(scripted.drained.isSet)
        #expect(await replay.messages.isEmpty)
        #expect(await sink.calls == [.open, .ingest("x")])
    }
}

// MARK: - SSE adapter

/// Collects the wire an SSE client would see — chunks in order, bracketed by
/// the open and the `[DONE]` sentinel — and can refuse sends past
/// `acceptLimit` to play a client that dropped mid-stream. Shared with the
/// pump suite.
actor SSEWireCollector {
    enum Item {
        case opened
        case chunk(OpenAI.ChatCompletionChunk)
        case done

        var chunk: OpenAI.ChatCompletionChunk? {
            if case .chunk(let c) = self { return c }
            return nil
        }
        var isOpened: Bool { if case .opened = self { return true } else { return false } }
        var isDone: Bool { if case .done = self { return true } else { return false } }
    }

    private(set) var items: [Item] = []
    private let acceptLimit: Int?

    init(acceptLimit: Int? = nil) { self.acceptLimit = acceptLimit }

    var chunks: [OpenAI.ChatCompletionChunk] { items.compactMap(\.chunk) }

    func opened() { items.append(.opened) }
    func send(_ chunk: OpenAI.ChatCompletionChunk) -> Bool {
        if let acceptLimit, chunks.count >= acceptLimit { return false }
        items.append(.chunk(chunk))
        return true
    }
    func done() -> Bool {
        items.append(.done)
        return true
    }
}

/// An `SSEDeliverySink` over a recording wire with an inert transport, so the
/// adapter runs without a socket.
func makeSSEDeliverySink(
    collector: SSEWireCollector,
    format: ToolCallFormat,
    includeUsage: Bool = false
) -> SSEDeliverySink {
    SSEDeliverySink(
        wire: SSEDeliverySink.Wire(
            open: { await collector.opened() },
            send: { await collector.send($0) },
            done: { await collector.done() },
            transport: .init(
                waitForDisconnect: { try? await Task.sleep(for: .seconds(60)) }, keepalive: nil)
        ),
        envelope: .init(
            completionID: "chatcmpl-sse", requestModel: nil, physicalModelID: "toy/model",
            created: 7
        ),
        transcoder: ArgumentTranscoder(format: format, toolSpecs: GenerationFixtures.demoToolSpecs),
        includeUsage: includeUsage
    )
}

private func sseDelivery(
    _ events: [AgentGeneration],
    format: ToolCallFormat = .xmlFunction,
    includeUsage: Bool = false
) async -> [SSEWireCollector.Item] {
    let collector = SSEWireCollector()
    let sink = makeSSEDeliverySink(collector: collector, format: format, includeUsage: includeUsage)
    let scripted = scriptedGeneration(events)
    let (log, handle) = await makeTrace(stream: true)
    await CompletionDelivery.deliver(
        scripted.generation,
        maxTokens: nil,
        sink: sink,
        activityLog: log,
        logHandle: handle,
        recordReplay: { _ in }
    )
    return await collector.items
}

private func chunks(_ items: [SSEWireCollector.Item]) -> [OpenAI.ChatCompletionChunk] {
    items.compactMap(\.chunk)
}

/// The SSE Delivery Sink end to end under the script: open → role chunk,
/// deltas, the Wire-Valid Close, the final chunk, `[DONE]`.
struct SSEDeliverySinkTests {

    @Test func roleChunkFirstFinalChunkAndDoneLast() async {
        let items = await sseDelivery([.text("hi"), .info(scriptedInfo)], includeUsage: true)
        let wire = chunks(items)

        #expect(items.first?.isOpened == true)
        #expect(items.last?.isDone == true)
        #expect(wire.first?.choices.first?.delta.role == .assistant)
        #expect(wire.dropFirst().first?.choices.first?.delta.content == "hi")
        let final = wire.last
        #expect(final?.choices.first?.finish_reason == .stop)
        #expect(final?.usage?.prompt_tokens_details?.cached_tokens == 5)
        #expect(final?.model == "toy/model")
        #expect(wire.allSatisfy { $0.created == 7 })
    }

    /// Argument Fragments streamed but no parsed call closed the turn (stream
    /// ended mid-call): the final chunk must still say `tool_calls`.
    @Test func streamedFragmentsPromoteStopToToolCalls() async {
        let partial = "<tool_call>\n<function=demo>\n<parameter=text>\nhalf"
        let items = await sseDelivery(
            partial.map { .toolCallDelta(name: nil, argumentsDelta: String($0)) })

        #expect(chunks(items).last?.choices.first?.finish_reason == .tool_calls)
        #expect(items.last?.isDone == true)
    }

    /// A dropped malformed tool-call buffer, with nothing streamed, goes out as
    /// one content chunk before the final chunk (ADR-0020 fallback).
    @Test func malformedFallbackEmitsOneContentChunkWhenNothingStreamed() async {
        let raw = "<tool_call>{not json}</tool_call>"
        let items = await sseDelivery([.malformedToolCall(raw)], format: .glm4)
        let wire = chunks(items)

        let contents = wire.compactMap { $0.choices.first?.delta.content }
        #expect(contents == [raw])
        #expect(wire.last?.choices.first?.finish_reason == .stop)
    }
}
