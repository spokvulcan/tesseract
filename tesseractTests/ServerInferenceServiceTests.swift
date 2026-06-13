import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

/// The dispatcher composes two arms — the cache-aware **Server Completion**
/// arm and the managed arm — and owns the **Completion Route** (ADR-0006).
/// Each test injects a double per arm and asserts exactly one arm serves
/// each request, model state is attached, and errors propagate.
@MainActor
struct ServerInferenceServiceTests {

    @Test func promptRequestsRouteToPromptInference() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.promptStart = makeStart(textChunks: ["prompt path"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(completion.calls.isEmpty)
        #expect(start.modelState == nil)
        #expect(try await collectText(from: start.stream) == "prompt path")
    }

    @Test func standardChatRequestsRouteToManagedChatInference() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.chatStart = makeStart(textChunks: ["chat path"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(completion.calls.isEmpty)
        #expect(try await collectText(from: start.stream) == "chat path")
    }

    @Test func serverCompatibleChatWithServableConversationRoutesToCompletionArm() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.start = makeStart(
            textChunks: ["server path"],
            cachedTokenCount: 42
        )
        let expectedState = ServerInferenceModelState(
            modelID: "qwen3.5-4b-paro",
            visionMode: true
        )
        let service = ServerInferenceService(
            completionStarter: completion,
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

        #expect(completion.calls.count == 1)
        #expect(completion.calls[0].modelID == expectedState.modelID)
        #expect(completion.calls[0].conversation == prefixConversation)
        #expect(engine.calls.isEmpty)
        #expect(start.cachedTokenCount == 42)
        #expect(start.modelState == expectedState)
        #expect(try await collectText(from: start.stream) == "server path")
    }

    @Test func serverCompatibleChatForwardsProgressBeforeStreamConsumption() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.start = makeStart(textChunks: ["server path"])
        completion.progressEvents = [
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
            completionStarter: completion,
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

        #expect(completion.calls[0].progressHandlerForwarded)
        #expect(log.traces[0].phase == .prefilling)
        #expect(log.traces[0].cachedTokens == 0)
        #expect(log.traces[0].newTokensToPrefill == 4096)
        #expect(try await collectText(from: start.stream) == "server path")
    }

    /// No usable prefix-cache conversation ⇒ the **Completion Route** decides
    /// standard, and the dispatcher falls back to the managed arm — the
    /// completion arm never sees a request it cannot serve.
    @Test func serverCompatibleChatWithoutPrefixConversationFallsBackToManagedArm() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.chatStart = makeStart(textChunks: ["managed fallback"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(engine.calls[0].kind == .chat)
        #expect(completion.calls.isEmpty)
        #expect(try await collectText(from: start.stream) == "managed fallback")
    }

    /// A conversation ending on an assistant turn routes standard — the bypass
    /// that used to be a `nil`-return inside the actor is now a route case
    /// observable at this seam.
    @Test func serverCompatibleChatEndingOnAssistantTurnFallsBackToManagedArm() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.chatStart = makeStart(textChunks: ["managed fallback"])
        let service = ServerInferenceService(
            completionStarter: completion,
            engine: engine,
            modelStateProvider: {
                ServerInferenceModelState(modelID: "qwen3.5-9b-paro", visionMode: false)
            }
        )
        let assistantLast = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [
                .init(role: .user, content: "Hello"),
                .assistant(content: "Hi there"),
            ]
        )

        let start = try await service.start(
            ServerInferenceRequest(
                input: .chat(.init(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: nil,
                    prefixCacheConversation: assistantLast
                )),
                parameters: .default,
                route: .serverCompatible
            )
        )

        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .chat)
        #expect(completion.calls.isEmpty)
        #expect(try await collectText(from: start.stream) == "managed fallback")
    }

    @Test func serverCompatibleChatWithoutModelStateUsesUnavailableSentinel() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.chatStart = makeStart(textChunks: ["managed fallback"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(start.modelState == .unavailable)
        #expect(try await collectText(from: start.stream) == "managed fallback")
    }

    @Test func completionArmErrorsPropagateToCaller() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.error = AgentEngineError.modelNotLoaded
        let service = ServerInferenceService(
            completionStarter: completion,
            engine: engine,
            modelStateProvider: { nil }
        )
        let prefixConversation = HTTPPrefixCacheConversation(
            systemPrompt: "System",
            messages: [.init(role: .user, content: "Hello")]
        )

        await #expect(throws: AgentEngineError.self) {
            _ = try await service.start(
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
        }
        #expect(engine.calls.isEmpty)
    }

    /// Agent chat goes out server-compatible with the history canonicalized by
    /// `AgentConversationBuilder` (PRD #72) — the **Completion Route** decides
    /// cache-aware and the completion arm serves, exactly as it does for HTTP.
    @Test func sharedGenerateClosureRoutesAgentChatThroughCompletionArm() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.start = makeStart(textChunks: ["foreground", " background"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(completion.calls.count == 1)
        #expect(engine.calls.isEmpty)
        #expect(completion.calls[0].conversation == AgentConversationBuilder.conversation(
            systemPrompt: "System",
            messages: [.user(content: "Hello")],
            toolSpecs: nil
        ))
    }

    /// A history the conversation shape cannot carry (an attachment that no
    /// longer decodes) builds no conversation — the request still goes out
    /// server-compatible and the **Completion Route** falls back to the
    /// managed arm: uncached but correct, never a dropped request.
    @Test func sharedGenerateClosureFallsBackToManagedArmWhenAttachmentIsUndecodable() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.chatStart = makeStart(textChunks: ["managed fallback"])
        let service = ServerInferenceService(
            completionStarter: completion,
            engine: engine,
            modelStateProvider: { nil }
        )
        let generate = makeServerInferenceGenerateClosure(
            inferenceService: service,
            parametersProvider: { .default }
        )

        let stream = generate(
            "System",
            [.user(content: "What is in this image?", images: [
                ImageAttachment(data: Data([0x00, 0x01]), mimeType: "image/png")
            ])],
            nil,
            nil
        )

        #expect(try await collectText(from: stream) == "managed fallback")
        #expect(engine.calls.count == 1)
        #expect(engine.calls[0].kind == .chat)
        #expect(completion.calls.isEmpty)
    }

    @Test func summarizeClosureRoutesPromptThroughService() async throws {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        engine.promptStart = makeStart(textChunks: ["sum", "mary"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
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
            completionStarter: completion,
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

        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.start = makeStart(textChunks: ["ok"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(completion.calls.count == 1)
        #expect(completion.calls[0].parameters.temperature == AgentGenerateParameters.qwen35.temperature)

        // Flip the model after the closure exists. The provider must observe
        // the new value on the very next call.
        settings.selectedAgentModelID = "qwen3-thinking-2507"

        completion.start = makeStart(textChunks: ["ok"])
        _ = try await collectText(from: generate("System", [.user(content: "q2")], nil, nil))
        #expect(completion.calls.count == 2)
        #expect(completion.calls[1].parameters.temperature == AgentGenerateParameters.qwen3Thinking.temperature)
    }

    /// Internal agent sessions must reach `ServerInferenceService` whether or
    /// not the public HTTP listener is enabled — the listener is a transport
    /// concern, not a dependency of the canonical inference path.
    @Test func serverEnabledFlagDoesNotAffectInternalRouting() async throws {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.isServerEnabled = false

        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        completion.start = makeStart(textChunks: ["service path"])
        let service = ServerInferenceService(
            completionStarter: completion,
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
        #expect(completion.calls.count == 1)
    }

    @Test func sharedGenerateClosureCancelsUnderlyingServiceStartWhenConsumerTaskIsCancelled() async {
        let engine = StubManagedInferenceEngine()
        let completion = StubServerCompletionStarter()
        let probe = ControlledInferenceStart()
        let firstChunkSeen = AsyncFlag()
        completion.start = await probe.makeStart()
        let service = ServerInferenceService(
            completionStarter: completion,
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

/// In-memory double for the managed arm.
@MainActor
private final class StubManagedInferenceEngine: ManagedInferenceStarting {
    enum CallKind {
        case prompt
        case chat
    }

    struct Call {
        let kind: CallKind
        let prompt: String?
        let systemPrompt: String?
        let messageCount: Int
        let toolSpecCount: Int
        let progressHandlerForwarded: Bool
        let parameters: AgentGenerateParameters
        var renderContext: TemplateRenderContext = .canonical
    }

    var calls: [Call] = []
    var promptStart = makeStart(textChunks: [])
    var chatStart = makeStart(textChunks: [])

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
            progressHandlerForwarded: false,
            parameters: parameters
        ))
        return promptStart
    }

    func startChatInference(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext,
        progressHandler: ServerInferenceProgressHandler?
    ) throws -> HTTPServerGenerationStart {
        calls.append(.init(
            kind: .chat,
            prompt: nil,
            systemPrompt: systemPrompt,
            messageCount: messages.count,
            toolSpecCount: toolSpecs?.count ?? 0,
            progressHandlerForwarded: progressHandler != nil,
            parameters: parameters,
            renderContext: renderContext
        ))
        return chatStart
    }
}

/// In-memory double for the cache-aware **Server Completion** arm.
@MainActor
private final class StubServerCompletionStarter: ServerCompletionStarting {
    struct Call {
        let modelID: String
        let conversation: HTTPPrefixCacheConversation
        let toolSpecCount: Int
        let progressHandlerForwarded: Bool
        let parameters: AgentGenerateParameters
        let renderContext: TemplateRenderContext
    }

    var calls: [Call] = []
    var start = makeStart(textChunks: [])
    var progressEvents: [ServerInferenceProgressEvent] = []
    var error: (any Error)?

    // Explicitly `@MainActor`: the `nonisolated` protocol would otherwise
    // pull the witness off the main actor and away from the stub's state.
    // An isolated method may witness a nonisolated async requirement —
    // callers hop, exactly as they do for the production actor adapter.
    @MainActor
    func startServerCompletion(
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPServerGenerationStart {
        calls.append(.init(
            modelID: modelID,
            conversation: conversation,
            toolSpecCount: toolSpecs?.count ?? 0,
            progressHandlerForwarded: progressHandler != nil,
            parameters: parameters,
            renderContext: renderContext
        ))
        if let error {
            throw error
        }
        for event in progressEvents {
            progressHandler?(event)
        }
        return start
    }
}

@MainActor
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
