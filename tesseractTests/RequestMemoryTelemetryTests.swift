import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct RequestMemoryTelemetryTests {
    @Test func phaseSamplesSeparateLifetimeAndObservedRequestPeaksAndStopAtFinish() throws {
        let modelID = "memory-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let probe = ScriptedMemorySample()
        let memory = RequestMemoryTelemetry(
            context: .init(requestID: UUID(), modelID: modelID, kvBits: nil, kvGroupSize: 64),
            sample: { probe.read() })
        memory.mark(.restoring)
        probe.setActive(200)
        #expect(memory.tick())
        memory.mark(.restored)
        probe.setActive(50)
        memory.finish(outcome: "cancelled")
        #expect(!memory.tick())
        memory.finish(outcome: "completed")
        let events = capture.drain()
        let terminal = try #require(events.last)
        #expect(terminal.field("outcome") == "cancelled")
        #expect(terminal.field("activeMlxBytes") == "50")
        #expect(terminal.field("sampledRequestPeakActiveMlxBytes") == "200")
        #expect(terminal.field("processLifetimePeakMlxBytes") == "1000")
        #expect(terminal.field("cachedMlxBytes") == "20")
        #expect(terminal.field("processFootprintBytes") == nil)
        #expect(events.filter { $0.field("sampleKind") == "terminal" }.count == 1)
        #expect(
            events.map { Int($0.field("sequence") ?? "") }
                == Array(1...events.count).map(Optional.some))
        #expect(
            events.contains {
                $0.field("phase") == "restoring" && $0.field("sampleKind") == "periodic"
            })
    }

    @Test func recordingCacheFactsDoesNotKeepItsObjectsAlive() {
        let memory = RequestMemoryTelemetry(
            context: .init(requestID: UUID(), modelID: "ownership", kvBits: nil, kvGroupSize: 64))
        weak var retained: KVCacheSimple?
        do {
            let cache = KVCacheSimple()
            retained = cache
            cache.state = [MLXArray.ones([1, 1, 8, 4]), MLXArray.ones([1, 1, 8, 4])]
            memory.mark(.restored, facts: RequestMemoryTelemetry.cacheFacts([cache]))
        }
        #expect(retained == nil)
    }

    @Test func coldAndWarmCompletionExposeRestoreHandoffAndRelease() async throws {
        let modelID = "memory-replay-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let tokenizer = ToySequencingTokenizer()
        let first = HTTPPrefixCacheConversation(
            systemPrompt: nil, messages: [.init(role: .user, content: "Hi")])
        let next = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                .init(role: .user, content: "Hi"), .assistant(content: "Hello!"),
                .init(role: .user, content: "More?"),
            ])
        let script =
            try tokenizer.applyChatTemplate(
                messages: next.promptMessages, tools: nil, additionalContext: nil)
            + Array("Sure.".utf8).map(Int.init)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script), tokenizer: tokenizer),
            modelID: modelID)
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        let firstHandle = try await fixture.start(conversation: first, parameters: parameters)
        #expect(try await collectServerText(firstHandle).text == "Hello!")
        let nextHandle = try await fixture.start(conversation: next, parameters: parameters)
        #expect(try await collectServerText(nextHandle).text == "Sure.")
        await fixture.drain()
        let events = capture.drain().filter { $0.eventName == "requestMemory" }
        let terminals = events.filter { $0.field("sampleKind") == "terminal" }
        #expect(terminals.count == 2)
        #expect(terminals.allSatisfy { $0.field("outcome") == "completed" })
        let warmID = try #require(terminals.last?.requestID)
        let warm = events.filter { $0.requestID == warmID }
        let restored = try #require(warm.first { $0.field("phase") == "restored" })
        #expect(restored.field("restoreMode") == "copy")
        #expect(try #require(Int(restored.field("restoreSnapshotBytes") ?? "")) > 0)
        let captured = try #require(warm.first { $0.field("phase") == "preparingPayload" })
        #expect(captured.field("leafCaptureMode") == "handoff")
        #expect(captured.field("requestCacheLayerCountAfterCapture") == "0")
        #expect(warm.contains { $0.field("phase") == "generationQuiescent" })
        #expect(warm.contains { $0.field("phase") == "releasingRequest" })
        #expect(terminals.last?.field("treeSnapshotBytes") != nil)
        #expect(terminals.last?.field("ssdPendingPayloadBytes") != nil)
    }

    @Test func cancelledDecodeStillEmitsTerminalMemory() async throws {
        let modelID = "memory-cancel-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let tokenizer = ToySequencingTokenizer()
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil, messages: [.init(role: .user, content: "Hi")])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let gate = ForwardGate(threshold: render.count + 8)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: render + Array(String(repeating: "x", count: 64).utf8).map(Int.init),
                    onForward: gate.onForward), tokenizer: tokenizer), modelID: modelID)
        var parameters = AgentGenerateParameters()
        parameters.kvBits = nil
        let handle = try await fixture.start(conversation: conversation, parameters: parameters)
        await gate.reached()
        handle.cancel()
        gate.open()
        for try await _ in handle.stream {}
        await handle.waitForCompletion()
        await fixture.drain()
        let events = capture.drain()
        let terminal = try #require(events.last { $0.field("sampleKind") == "terminal" })
        #expect(terminal.field("outcome") == "cancelled")
        #expect(terminal.field("phase") == "finished")
        #expect(terminal.field("cancelSignalOrigin") == "caller")
        #expect(terminal.field("requestCacheMeasuredAtPhase") == "generationQuiescent")
        #expect(terminal.field("treeMeasuredAtPhase") == "releasingRequest")
        #expect(events.contains { $0.field("sampleKind") == "cancelSignal" })
    }

    @Test(arguments: ["caller", "streamCancelled", "streamFinished"])
    func cancellationDuringCleanupIsDistinctFromNaturalStreamFinish(origin: String) throws {
        let modelID = "memory-late-cancel-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let memory = RequestMemoryTelemetry(
            context: .init(requestID: UUID(), modelID: modelID, kvBits: nil, kvGroupSize: 64))
        memory.mark(.admittingLeaf)
        memory.recordCancellationSignal(origin: origin)
        memory.finish(outcome: "completed")
        let terminal = try #require(capture.drain().last)
        #expect(
            terminal.field("outcome")
                == (origin == "streamFinished" ? "completed" : "cancelledDuringCleanup"))
    }

    @Test func failedStartStillEmitsTerminalMemory() async throws {
        let modelID = "memory-failed-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let actor = LLMActor()
        let module = ServerCompletion(cacheAdmin: PrefixCacheAdmin())
        do {
            _ = try await module.start(
                on: actor, sessions: UnavailableMemoryTestSession(), modelID: modelID,
                conversation: .init(
                    systemPrompt: nil, messages: [.init(role: .user, content: "Hi")]),
                toolSpecs: nil, parameters: AgentGenerateParameters())
            Issue.record("the unavailable session must fail the start")
        } catch is UnavailableMemoryTestSession.Failure {}
        let terminal = try #require(capture.drain().last { $0.field("sampleKind") == "terminal" })
        #expect(terminal.field("outcome") == "startFailed")
    }
}

private nonisolated struct UnavailableMemoryTestSession: ModelSessionProviding {
    struct Failure: Error {}
    func withSession<V, R: Sendable>(
        nonSendable payload: sending V,
        _ body: @Sendable (any ModelSession, V) async throws -> R
    ) async throws -> R { throw Failure() }
}

private nonisolated final class ScriptedMemorySample: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 40
    func setActive(_ bytes: Int) { lock.withLock { active = bytes } }
    func read() -> RequestMemoryTelemetry.Sample {
        lock.withLock {
            .init(activeBytes: active, cacheBytes: 20, lifetimePeakBytes: 1000)
        }
    }
}
