import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

@MainActor
struct ServerInferenceServiceTests {

    @Test func promptRequestsRouteToPromptInference() async throws {
        let engine = StubServerInferenceEngine()
        engine.promptStart = makeStart(textChunks: ["prompt path"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let start = try await service.start(
            ServerInferenceRequest(
                input: .prompt("Summarize this"),
                parameters: .default
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
        #expect(engine.calls[0].prompt == "Summarize this")
        #expect(start.modelState == nil)
        #expect(try await collectText(from: start.stream) == "prompt path")
    }

    @Test func standardChatRequestsRouteToManagedChatInference() async throws {
        let engine = StubServerInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["chat path"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: nil
                )),
                parameters: .default
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .chat)
        #expect(engine.calls[0].systemPrompt == "System")
        #expect(engine.calls[0].messageCount == 1)
        #expect(engine.calls[0].usedPrefixCacheConversation == false)
        #expect(try await collectText(from: start.stream) == "chat path")
    }

    @Test func serverCompatibleChatRoutesToServerPathAndPreservesCachedTokens() async throws {
        let engine = StubServerInferenceEngine()
        engine.serverStart = makeStart(
            textChunks: ["server path"],
            cachedTokenCount: 42
        )
        let expectedState = ServerInferenceModelState(
            modelID: "qwen3.5-4b-paro",
            visionMode: true
        )
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { expectedState }
        )
        let prefixConversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [.init(role: .user, content: "Hello")]
        )

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: prefixConversation
                )),
                parameters: .default,
                route: .serverCompatible
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .serverChat)
        #expect(engine.calls[0].modelID == expectedState.modelID)
        #expect(engine.calls[0].usedPrefixCacheConversation)
        #expect(start.cachedTokenCount == 42)
        #expect(start.modelState == expectedState)
        #expect(try await collectText(from: start.stream) == "server path")
    }

    @Test func serverCompatibleChatForwardsProgressBeforeStreamConsumption() async throws {
        let engine = StubServerInferenceEngine()
        engine.serverStart = makeStart(textChunks: ["server path"])
        engine.serverProgressEvents = [
            .cacheLookupStarted,
            .cacheLookupFinished(.init(
                reason: "missNoEntries",
                cachedTokens: 0,
                sharedPrefixLength: 0,
                promptTokens: 4096,
                newTokensToPrefill: 4096,
                lookupMs: 2.0,
                restoreMs: 0
            )),
            .prefillStarted(.init(
                promptTokens: 4096,
                cachedTokens: 0,
                newTokensToPrefill: 4096,
                prefillMs: nil
            )),
        ]
        let log = ServerGenerationLog()
        let handle = log.startRequest(
            completionID: "id", model: "m", stream: true, sessionAffinity: nil
        )
        log.markLeaseAcquired(handle: handle)
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: {
                ServerInferenceModelState(modelID: "qwen3.5-4b-paro", visionMode: false)
            }
        )
        let prefixConversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [.init(role: .user, content: "Hello")]
        )

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: prefixConversation,
                    progressHandler: CompletionHandler.makeProgressHandler(
                        activityLog: log,
                        logHandle: handle
                    )
                )),
                parameters: .default,
                route: .serverCompatible
            )
        )

        #expect(engine.calls[0].progressHandlerForwarded)
        #expect(log.traces[0].phase == .prefilling)
        #expect(log.traces[0].cachedTokens == 0)
        #expect(log.traces[0].newTokensToPrefill == 4096)
        #expect(try await collectText(from: start.stream) == "server path")
    }

    @Test func serverCompatibleChatWithoutPrefixConversationStillUsesServerPath() async throws {
        let engine = StubServerInferenceEngine()
        engine.serverStart = makeStart(textChunks: ["server fallback"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: {
                ServerInferenceModelState(modelID: "qwen3.5-9b-paro", visionMode: false)
            }
        )

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: nil
                )),
                parameters: .default,
                route: .serverCompatible
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .serverChat)
        #expect(engine.calls[0].usedPrefixCacheConversation == false)
        #expect(try await collectText(from: start.stream) == "server fallback")
    }

    @Test func serverCompatibleChatWithoutModelStateUsesUnavailableSentinel() async throws {
        let engine = StubServerInferenceEngine()
        engine.serverStart = makeStart(textChunks: ["server fallback"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: nil
                )),
                parameters: .default,
                route: .serverCompatible
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .serverChat)
        #expect(engine.calls[0].modelID == "")
        #expect(start.modelState == .unavailable)
        #expect(try await collectText(from: start.stream) == "server fallback")
    }

    @Test func sharedGenerateClosureRoutesManagedChatThroughService() async throws {
        let engine = StubServerInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["foreground", " background"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let stream = generate(
            "System",
            [.user(content: "Hello")],
            nil,
            nil
        )

        #expect(try await collectText(from: stream) == "foreground background")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .chat)
    }

    @Test func summarizeClosureRoutesPromptThroughService() async throws {
        let engine = StubServerInferenceEngine()
        engine.promptStart = makeStart(textChunks: ["sum", "mary"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let summarize = makeSummarizeClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let summary = try await summarize("Summarize this")

        #expect(summary == "summary")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
    }

    @Test func summarizeClosureReclassifiesThinkingIntoReturnedText() async throws {
        let engine = StubServerInferenceEngine()
        engine.promptStart = HTTPServerGenerationStart(
            stream: makeEventStream(events: [
                .thinkStart,
                .thinking("draft"),
                .thinkReclassify,
                .text(" answer"),
                .thinkEnd,
            ]),
            cachedTokenCount: 0
        )
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let summarize = makeSummarizeClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let summary = try await summarize("Summarize this")

        #expect(summary == "draft answer")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
    }

    /// The `AgentFactory`-style parametersProvider must live-read the current
    /// `SettingsManager` values on every call. Capturing them at agent-build
    /// time is what made /compact and auto-compaction miss the user switching
    /// model mid-session.
    @Test func agentFactoryStyleProviderLiveReadsModelFromSettings() async throws {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = "qwen3.5-4b-paro"

        let engine = StubServerInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["ok"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )

        // Mirrors the provider that `AgentFactory.makeAgent` wires up: each
        // call re-reads the live settings rather than freezing them at init.
        let provider: @MainActor @Sendable () -> AgentGenerateParameters = {
            [settings] in
            AgentGenerateParameters.forModel(settings.selectedAgentModelID)
        }
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            parametersProvider: provider
        )

        _ = try await collectText(from: generate("System", [.user(content: "q1")], nil, nil))
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].parameters.temperature == AgentGenerateParameters.qwen35.temperature)

        // Flip the model after the closure exists. The provider must observe
        // the new value on the very next call.
        settings.selectedAgentModelID = "qwen3-thinking-2507"

        engine.chatStart = makeStart(textChunks: ["ok"])
        _ = try await collectText(from: generate("System", [.user(content: "q2")], nil, nil))
        #expect(engine.calls.count == 2)
        #expect(engine.calls[1].parameters.temperature == AgentGenerateParameters.qwen3Thinking.temperature)
    }

    /// Internal agent sessions must reach `ServerInferenceService` whether or
    /// not the public HTTP listener is enabled — the listener is a transport
    /// concern, not a dependency of the canonical inference path.
    @Test func serverEnabledFlagDoesNotAffectInternalRouting() async throws {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.isServerEnabled = false

        let engine = StubServerInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["service path"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let stream = generate(
            "System",
            [.user(content: "Hello")],
            nil,
            nil
        )

        #expect(try await collectText(from: stream) == "service path")
        #expect(engine.calls.count == 1)
    }

    @Test func sharedGenerateClosureCancelsUnderlyingServiceStartWhenConsumerTaskIsCancelled() async {
        let engine = StubServerInferenceEngine()
        let probe = ControlledInferenceStart()
        let firstChunkSeen = AsyncFlag()
        engine.chatStart = await probe.makeStart()
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let stream = generate(
            "System",
            [.user(content: "Hello")],
            nil,
            nil
        )

        let consumer = Task {
            do {
                for try await _ in stream {
                    await firstChunkSeen.set()
                }
            } catch is CancellationError {
            } catch {
                Issue.record("Unexpected stream error: \(error)")
            }
        }

        await firstChunkSeen.waitUntilSet()
        consumer.cancel()
        _ = await consumer.result

        for _ in 0..<20 {
            if await probe.cancelCount > 0 { break }
            await Task.yield()
        }

        #expect(await probe.cancelCount > 0)
    }
}

@MainActor
private final class StubServerInferenceEngine: ServerInferenceEngine {
    enum CallKind {
        case prompt
        case chat
        case serverChat
    }

    struct Call {
        let kind: CallKind
        let prompt: String?
        let systemPrompt: String?
        let messageCount: Int
        let toolSpecCount: Int
        let modelID: String?
        let usedPrefixCacheConversation: Bool
        let progressHandlerForwarded: Bool
        let parameters: AgentGenerateParameters
    }

    var calls: [Call] = []
    var promptStart = makeStart(textChunks: [])
    var chatStart = makeStart(textChunks: [])
    var serverStart = makeStart(textChunks: [])
    var serverProgressEvents: [ServerInferenceProgressEvent] = []

    func startPromptInference(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        calls.append(.init(
            kind: .prompt,
            prompt: prompt,
            systemPrompt: nil,
            messageCount: 0,
            toolSpecCount: 0,
            modelID: nil,
            usedPrefixCacheConversation: false,
            progressHandlerForwarded: false,
            parameters: parameters
        ))
        return promptStart
    }

    func startChatInference(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        calls.append(.init(
            kind: .chat,
            prompt: nil,
            systemPrompt: systemPrompt,
            messageCount: messages.count,
            toolSpecCount: toolSpecs?.count ?? 0,
            modelID: nil,
            usedPrefixCacheConversation: false,
            progressHandlerForwarded: false,
            parameters: parameters
        ))
        return chatStart
    }

    func startServerChatInference(
        modelID: String,
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        prefixCacheConversation: HTTPPrefixCacheConversation?,
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPServerGenerationStart {
        calls.append(.init(
            kind: .serverChat,
            prompt: nil,
            systemPrompt: systemPrompt,
            messageCount: messages.count,
            toolSpecCount: toolSpecs?.count ?? 0,
            modelID: modelID,
            usedPrefixCacheConversation: prefixCacheConversation != nil,
            progressHandlerForwarded: progressHandler != nil,
            parameters: parameters
        ))
        for event in serverProgressEvents {
            progressHandler?(event)
        }
        return serverStart
    }
}

private func makeStart(
    textChunks: [String],
    cachedTokenCount: Int = 0
) -> HTTPServerGenerationStart {
    return HTTPServerGenerationStart(
        stream: makeEventStream(textChunks: textChunks),
        cachedTokenCount: cachedTokenCount
    )
}

private func makeEventStream(
    textChunks: [String]
) -> AsyncThrowingStream<AgentGeneration, Error> {
    makeEventStream(events: textChunks.map(AgentGeneration.text))
}

private func makeEventStream(
    events: [AgentGeneration]
) -> AsyncThrowingStream<AgentGeneration, Error> {
    AsyncThrowingStream<AgentGeneration, Error> { continuation in
        for event in events {
            continuation.yield(event)
        }
        continuation.finish()
    }
}

private func collectText(
    from stream: AsyncThrowingStream<AgentGeneration, Error>
) async throws -> String {
    var chunks: [String] = []
    for try await event in stream {
        if case .text(let chunk) = event {
            chunks.append(chunk)
        }
    }
    return chunks.joined()
}

private actor AsyncFlag {
    private var isSet = false

    func set() {
        isSet = true
    }

    func waitUntilSet() async {
        while isSet == false {
            await Task.yield()
        }
    }
}

private actor ControlledInferenceStart {
    private var continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation?
    private(set) var cancelCount = 0

    func makeStart() -> HTTPServerGenerationStart {
        let stream = AsyncThrowingStream<AgentGeneration, Error> { continuation in
            storeContinuation(continuation)
            continuation.yield(.text("pending"))
        }
        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: 0,
            cancel: {
                Task {
                    await self.cancel()
                }
            }
        )
    }

    private func storeContinuation(
        _ continuation: AsyncThrowingStream<AgentGeneration, Error>.Continuation
    ) {
        self.continuation = continuation
    }

    private func cancel() {
        cancelCount += 1
        continuation?.finish()
    }
}
