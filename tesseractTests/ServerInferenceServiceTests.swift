import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

@MainActor
struct ServerInferenceServiceTests {

    private static let enabledTriAttention = TriAttentionConfiguration(
        enabled: true,
        budgetTokens: TriAttentionConfiguration.v1BudgetTokens,
        calibrationArtifactIdentity: nil,
        implementationVersion: .v1
    )

    @Test func promptRequestsRouteToPromptInference() async throws {
        let engine = StubServerInferenceEngine()
        engine.promptStart = makeStart(textChunks: ["prompt path"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        var parameters = AgentGenerateParameters.default
        parameters.triAttention = Self.enabledTriAttention

        let start = try await service.start(
            ServerInferenceRequest(
                input: .prompt("Summarize this"),
                parameters: parameters
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
        #expect(engine.calls[0].prompt == "Summarize this")
        #expect(engine.calls[0].parameters.triAttention == Self.enabledTriAttention)
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
        var parameters = AgentGenerateParameters.default
        parameters.triAttention = Self.enabledTriAttention

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: nil
                )),
                parameters: parameters
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .chat)
        #expect(engine.calls[0].systemPrompt == "System")
        #expect(engine.calls[0].messageCount == 1)
        #expect(engine.calls[0].usedPrefixCacheConversation == false)
        #expect(engine.calls[0].parameters.triAttention == Self.enabledTriAttention)
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
            visionMode: true,
            triAttention: .v1Disabled,
            triAttentionFallbackReason: .visionMode
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
        #expect(engine.calls[0].parameters.triAttention == .v1Disabled)
        #expect(start.cachedTokenCount == 42)
        #expect(start.modelState == expectedState)
        #expect(start.modelState?.triAttentionFallbackReason == .visionMode)
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

    /// Epic 3 Task 3 parity gate — identical HTTP contract between dense and
    /// TriAttention modes. Runs the same chat request twice against a service
    /// backed by dense and TriAttention model states, and asserts:
    ///
    /// - **Model routing**: engine receives identical `modelID`.
    /// - **Tool formatting**: engine receives identical `toolSpecCount`.
    /// - **Response envelope inputs**: engine receives identical `systemPrompt`,
    ///   `messageCount`, and `usedPrefixCacheConversation`.
    /// - **cached_tokens accounting**: `ServerInferenceStart.cachedTokenCount`
    ///   is forwarded byte-identically from the engine layer.
    ///
    /// The one permitted difference is `parameters.triAttention`, which is the
    /// intended carrier for the vendor cache/runtime selection below this
    /// layer. Everything HTTP clients can observe must be invariant.
    @Test func serverCompatibleChatPreservesParityAcrossAttentionModes() async throws {
        let modelID = "qwen3.5-4b-paro"
        let systemPrompt = "System"
        let chatMessages: [LLMMessage] = [.user(content: "Hello")]
        let toolSpec: ToolSpec = [
            "type": "function",
            "function": [
                "name": "fetch",
                "description": "Fetch data",
                "parameters": [
                    "type": "object",
                    "properties": [:] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let prefixConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: [.init(role: .user, content: "Hello")]
        )

        func run(
            triAttention: TriAttentionConfiguration,
            fallbackReason: TriAttentionDenseFallbackReason?
        ) async throws -> (call: StubServerInferenceEngine.Call, start: ServerInferenceStart) {
            let engine = StubServerInferenceEngine()
            engine.serverStart = makeStart(
                textChunks: ["shared response"],
                cachedTokenCount: 23
            )
            let service = ServerInferenceService(
                engine: engine,
                modelStateProvider: {
                    ServerInferenceModelState(
                        modelID: modelID,
                        visionMode: false,
                        triAttention: triAttention,
                        triAttentionFallbackReason: fallbackReason
                    )
                }
            )
            var parameters = AgentGenerateParameters.default
            parameters.triAttention = triAttention

            let start = try await service.start(
                ServerInferenceRequest(
                    input: .chat(.init(
                        systemPrompt: systemPrompt,
                        messages: chatMessages,
                        toolSpecs: [toolSpec],
                        prefixCacheConversation: prefixConversation
                    )),
                    parameters: parameters,
                    route: .serverCompatible
                )
            )
            return (engine.calls[0], start)
        }

        let dense = try await run(triAttention: .v1Disabled, fallbackReason: nil)
        let tri = try await run(
            triAttention: Self.enabledTriAttention,
            fallbackReason: nil
        )

        // Model routing: identical physical model targets the engine.
        #expect(dense.call.modelID == tri.call.modelID)
        #expect(dense.call.modelID == modelID)

        // Tool formatting: the ToolSpec count reaches the engine unchanged.
        #expect(dense.call.toolSpecCount == tri.call.toolSpecCount)
        #expect(dense.call.toolSpecCount == 1)

        // Response envelope inputs that drive chat templating are invariant.
        #expect(dense.call.kind == tri.call.kind)
        #expect(dense.call.kind == .serverChat)
        #expect(dense.call.systemPrompt == tri.call.systemPrompt)
        #expect(dense.call.messageCount == tri.call.messageCount)
        #expect(dense.call.usedPrefixCacheConversation == tri.call.usedPrefixCacheConversation)

        // cached_tokens accounting contract: the same engine-reported count
        // reaches the transport unchanged by attention-mode selection.
        #expect(dense.start.cachedTokenCount == tri.start.cachedTokenCount)
        #expect(dense.start.cachedTokenCount == 23)

        // Stream content is identical (the helper engine yields the same text).
        let denseText = try await collectText(from: dense.start.stream)
        let triText = try await collectText(from: tri.start.stream)
        #expect(denseText == triText)
        #expect(denseText == "shared response")

        // The permitted difference — the vendor runtime carrier — is the only
        // thing that changed. If this becomes equal in the future, TriAttention
        // mode stopped propagating to the vendor layer.
        #expect(dense.call.parameters.triAttention != tri.call.parameters.triAttention)
        #expect(dense.call.parameters.triAttention.enabled == false)
        #expect(tri.call.parameters.triAttention.enabled == true)
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
        let fallback = StubLegacyInternalInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["foreground", " background"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { false },
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
        #expect(fallback.chatCalls.isEmpty)
    }

    @Test func summarizeClosureRoutesPromptThroughService() async throws {
        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
        engine.promptStart = makeStart(textChunks: ["sum", "mary"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let summarize = makeSummarizeClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { false },
            parametersProvider: { .default }
        )

        let summary = try await summarize("Summarize this")

        #expect(summary == "summary")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
        #expect(fallback.promptCalls.isEmpty)
    }

    @Test func summarizeClosureReclassifiesThinkingIntoReturnedText() async throws {
        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
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
            fallbackEngine: fallback,
            rollbackEnabled: { false },
            parametersProvider: { .default }
        )

        let summary = try await summarize("Summarize this")

        #expect(summary == "draft answer")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .prompt)
        #expect(fallback.promptCalls.isEmpty)
    }

    @Test func sharedGenerateClosureRoutesToLegacyEngineWhenRollbackEnabled() async throws {
        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
        fallback.chatStream = makeEventStream(textChunks: ["legacy", " chat"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { true },
            parametersProvider: { .default }
        )

        let stream = generate(
            "System",
            [.user(content: "Hello")],
            nil,
            nil
        )

        #expect(try await collectText(from: stream) == "legacy chat")
        #expect(engine.calls.isEmpty)
        #expect(fallback.chatCalls.count == 1)
        #expect(fallback.chatCalls[0].systemPrompt == "System")
    }

    @Test func summarizeClosureRoutesToLegacyEngineWhenRollbackEnabled() async throws {
        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
        fallback.promptStream = makeEventStream(textChunks: ["legacy", " summary"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let summarize = makeSummarizeClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { true },
            parametersProvider: { .default }
        )

        let summary = try await summarize("Summarize this")

        #expect(summary == "legacy summary")
        #expect(engine.calls.isEmpty)
        #expect(fallback.promptCalls == ["Summarize this"])
    }

    @Test func serverEnabledFlagDoesNotAffectInternalRouting() async throws {
        clearInferenceRoutingDefaults()
        defer { clearInferenceRoutingDefaults() }

        let settings = SettingsManager()
        settings.isServerEnabled = false

        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
        engine.chatStart = makeStart(textChunks: ["service path"])
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { settings.internalServerInferenceRollbackEnabled },
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
        #expect(fallback.chatCalls.isEmpty)
    }

    @Test func rollbackSettingDefaultsFalseAndRoundTripsAcrossInstances() {
        clearInferenceRoutingDefaults()
        defer { clearInferenceRoutingDefaults() }

        let first = SettingsManager()
        #expect(first.internalServerInferenceRollbackEnabled == false)

        first.internalServerInferenceRollbackEnabled = true

        let second = SettingsManager()
        #expect(second.internalServerInferenceRollbackEnabled == true)
    }

    @Test func sharedGenerateClosureCancelsUnderlyingServiceStartWhenConsumerTaskIsCancelled() async {
        let engine = StubServerInferenceEngine()
        let fallback = StubLegacyInternalInferenceEngine()
        let probe = ControlledInferenceStart()
        let firstChunkSeen = AsyncFlag()
        engine.chatStart = await probe.makeStart()
        let service = ServerInferenceService(
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            fallbackEngine: fallback,
            rollbackEnabled: { false },
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
        let parameters: AgentGenerateParameters
    }

    var calls: [Call] = []
    var promptStart = makeStart(textChunks: [])
    var chatStart = makeStart(textChunks: [])
    var serverStart = makeStart(textChunks: [])

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
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart {
        calls.append(.init(
            kind: .serverChat,
            prompt: nil,
            systemPrompt: systemPrompt,
            messageCount: messages.count,
            toolSpecCount: toolSpecs?.count ?? 0,
            modelID: modelID,
            usedPrefixCacheConversation: prefixCacheConversation != nil,
            parameters: parameters
        ))
        return serverStart
    }
}

@MainActor
private final class StubLegacyInternalInferenceEngine: LegacyInternalInferenceEngine {
    struct ChatCall {
        let systemPrompt: String
        let messages: [LLMMessage]
        let toolSpecCount: Int
        let parameters: AgentGenerateParameters
    }

    var promptCalls: [String] = []
    var chatCalls: [ChatCall] = []
    var promptStream = makeEventStream(textChunks: [])
    var chatStream = makeEventStream(textChunks: [])

    func generate(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        promptCalls.append(prompt)
        return promptStream
    }

    func generate(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        chatCalls.append(.init(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecCount: toolSpecs?.count ?? 0,
            parameters: parameters
        ))
        return chatStream
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

private func clearInferenceRoutingDefaults() {
    let keys = [
        "internalServerInferenceRollbackEnabled",
        "isServerEnabled",
    ]
    for key in keys {
        UserDefaults.standard.removeObject(forKey: key)
    }
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
