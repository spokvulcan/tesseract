import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Streaming event pump (scripted events → chunk sink)

/// Tool schema for the streaming pump tests: one string parameter.
private let pumpToolSpecs: [ToolSpec] = [
    [
        "type": "function",
        "function": [
            "name": "demo",
            "description": "demo tool",
            "parameters": [
                "type": "object",
                "properties": [
                    "text": ["type": "string"] as [String: any Sendable]
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
]

/// Captures every chunk the pump sends; optionally simulates a client
/// disconnect by refusing sends once `acceptLimit` chunks are in.
private actor PumpChunkCollector {
    private(set) var chunks: [OpenAI.ChatCompletionChunk] = []
    private let acceptLimit: Int?

    init(acceptLimit: Int?) { self.acceptLimit = acceptLimit }

    func collect(_ chunk: OpenAI.ChatCompletionChunk) -> Bool {
        if let acceptLimit, chunks.count >= acceptLimit { return false }
        chunks.append(chunk)
        return true
    }
}

/// Drive the production stream event pump with a scripted event sequence and
/// return the chunks a client would have received, in order.
private func pumpEvents(
    _ events: [AgentGeneration],
    format: ToolCallFormat,
    acceptLimit: Int? = nil,
    cancel: @escaping @Sendable () -> Void = {}
) async -> (chunks: [OpenAI.ChatCompletionChunk], outcome: CompletionHandler.StreamingOutcome) {
    let collector = PumpChunkCollector(acceptLimit: acceptLimit)
    let stream = AsyncThrowingStream<AgentGeneration, Error> { continuation in
        for event in events { continuation.yield(event) }
        continuation.finish()
    }
    let activityLog = await MainActor.run { ServerGenerationLog() }
    let outcome = await CompletionHandler.streamGenerationEvents(
        stream,
        envelope: .init(completionID: "chatcmpl-pump", model: "toy/model", created: 1),
        transcoder: ArgumentTranscoder(format: format, toolSpecs: pumpToolSpecs),
        activityLog: activityLog,
        logHandle: TraceHandle(id: UUID()),
        cancel: cancel,
        send: { await collector.collect($0) }
    )
    return (await collector.chunks, outcome)
}

/// The single tool-call delta a chunk carries, if any.
private func toolDelta(_ chunk: OpenAI.ChatCompletionChunk) -> OpenAI.ToolCall? {
    chunk.choices.first?.delta.tool_calls?.first
}

/// One `.toolCallDelta` event per character — the adversarial fragmentation.
private func xmlDeltas(_ text: String) -> [AgentGeneration] {
    text.map { .toolCallDelta(name: nil, argumentsDelta: String($0)) }
}

/// Drives `CompletionHandler.streamGenerationEvents` — the production event
/// pump behind the SSE writer — end to end with scripted generation events,
/// asserting on the chunk stream a client would decode.
struct CompletionHandlerStreamingPumpTests {

    private let xmlBlock =
        "<tool_call>\n<function=demo>\n<parameter=text>\nhello world\n</parameter>\n</function>\n"
    private let parsedCall = ToolCall(
        function: .init(name: "demo", arguments: ["text": JSONValue.string("hello world")])
    )

    /// Fragment chunk shape: the engagement chunk carries id/type/name with
    /// empty arguments; every later fragment carries only index + an
    /// arguments piece; the parsed close re-sends nothing (the accumulated
    /// fragments alone reproduce the arguments exactly once).
    @Test func transcodedCallStreamsEngagementThenFragments() async throws {
        let events =
            [AgentGeneration.text("Let me ")] + xmlDeltas(xmlBlock)
            + [AgentGeneration.toolCall(parsedCall)]
        let (chunks, outcome) = await pumpEvents(events, format: .xmlFunction)

        #expect(chunks.first?.choices.first?.delta.content == "Let me ")
        #expect(
            chunks.allSatisfy { ($0.choices.first?.delta.tool_calls?.count ?? 1) == 1 },
            "every tool-call chunk carries exactly one delta"
        )

        let toolChunks = chunks.compactMap(toolDelta)
        let engagement = try #require(toolChunks.first)
        #expect(engagement.id?.isEmpty == false)
        #expect(engagement.type == "function")
        #expect(engagement.function?.name == "demo")
        #expect(engagement.function?.arguments?.isEmpty == true)
        #expect(engagement.index == 0)

        var accumulated = ""
        for fragment in toolChunks.dropFirst() {
            #expect(fragment.id == nil)
            #expect(fragment.function?.name == nil)
            #expect(fragment.index == 0)
            accumulated += fragment.function?.arguments ?? ""
        }
        #expect(accumulated == #"{"text":"hello world"}"#)

        guard case .completed(_, _, let wireStreamed) = outcome else {
            Issue.record("expected .completed, got \(outcome)")
            return
        }
        #expect(wireStreamed)
    }

    /// Text/tool-call interleaving: chunk order preserves event order.
    @Test func textAroundToolCallPreservesChunkOrder() async {
        let events =
            [AgentGeneration.text("before")] + xmlDeltas(xmlBlock)
            + [AgentGeneration.toolCall(parsedCall), .text("after")]
        let (chunks, _) = await pumpEvents(events, format: .xmlFunction)

        let kinds = chunks.compactMap { chunk -> String? in
            let delta = chunk.choices.first?.delta
            if delta?.content != nil { return "text" }
            if delta?.tool_calls != nil { return "tool" }
            return nil
        }
        #expect(kinds.first == "text")
        #expect(kinds.last == "text")
        #expect(Set(kinds.dropFirst().dropLast()) == ["tool"])
    }

    /// Multi-call turns: the second call streams under index 1 with a fresh
    /// engagement id.
    @Test func secondCallStreamsWithIncrementedIndex() async {
        let zeroParamBlock = "<tool_call>\n<function=demo>\n</function>\n"
        let secondCall = ToolCall(
            function: .init(name: "demo", arguments: [:] as [String: JSONValue]))
        let events =
            xmlDeltas(xmlBlock) + [AgentGeneration.toolCall(parsedCall)]
            + xmlDeltas(zeroParamBlock) + [AgentGeneration.toolCall(secondCall)]
        let (chunks, _) = await pumpEvents(events, format: .xmlFunction)

        let indices = Set(chunks.compactMap { toolDelta($0)?.index })
        #expect(indices == [0, 1])
        let engagementIDs = chunks.compactMap { toolDelta($0)?.id }
        #expect(engagementIDs.count == 2)
        #expect(Set(engagementIDs).count == 2)
    }

    /// Fallback atomic emission for a non-transcodable format: exactly two
    /// chunks — name first, then the full arguments — and the turn does not
    /// count as fragment-streamed.
    @Test func nonTranscodableFormatEmitsTwoAtomicChunks() async {
        let events: [AgentGeneration] = [
            .toolCallDelta(name: nil, argumentsDelta: "demo<arg_key>text</arg_key>"),
            .toolCall(parsedCall),
        ]
        let (chunks, outcome) = await pumpEvents(events, format: .glm4)

        let toolChunks = chunks.compactMap(toolDelta)
        #expect(toolChunks.count == 2)
        #expect(toolChunks[0].function?.name == "demo")
        #expect(toolChunks[0].function?.arguments?.isEmpty == true)
        #expect(toolChunks[1].id == nil)
        #expect(toolChunks[1].function?.arguments == #"{"text":"hello world"}"#)

        guard case .completed(_, _, let wireStreamed) = outcome else {
            Issue.record("expected .completed, got \(outcome)")
            return
        }
        #expect(!wireStreamed)
    }

    /// Wire-Valid Close: a stream that terminates mid-call (cancel,
    /// max-tokens) still leaves parseable accumulated arguments on the wire.
    @Test func streamEndMidCallClosesWireValid() async throws {
        let partial = "<tool_call>\n<function=demo>\n<parameter=text>\nhalf way"
        let (chunks, outcome) = await pumpEvents(xmlDeltas(partial), format: .xmlFunction)

        let accumulated = chunks.compactMap { toolDelta($0)?.function?.arguments }.joined()
        let data = try #require(accumulated.data(using: .utf8))
        #expect((try? JSONSerialization.jsonObject(with: data)) is [String: Any])

        guard case .completed(_, _, let wireStreamed) = outcome else {
            Issue.record("expected .completed, got \(outcome)")
            return
        }
        #expect(wireStreamed)
    }

    /// A refused send reads as a client disconnect: the pump cancels
    /// generation and reports the chunk-write source.
    @Test func refusedSendCancelsAndReportsChunkWriteDisconnect() async {
        let cancelled = LeaseAcquiredSignal()
        let (chunks, outcome) = await pumpEvents(
            [.text("a"), .text("b")],
            format: .xmlFunction,
            acceptLimit: 1,
            cancel: { cancelled.set() }
        )

        #expect(chunks.count == 1)
        #expect(cancelled.isSet)
        guard case .disconnected(let source) = outcome else {
            Issue.record("expected .disconnected, got \(outcome)")
            return
        }
        #expect(source == .chunkWrite)
    }
}
