import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

struct CompletionHandlerTests {

    // MARK: - LeaseAcquiredSignal

    @Test func signalStartsFalse() {
        let signal = LeaseAcquiredSignal()
        #expect(!signal.isSet)
    }

    @Test func signalBecomesTrue() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        #expect(signal.isSet)
    }

    @Test func signalSetIsIdempotent() {
        let signal = LeaseAcquiredSignal()
        signal.set()
        signal.set()
        #expect(signal.isSet)
    }

    // MARK: - withAcquisitionTimeout

    @Test func timeoutThrowsWhenBodyNeverSignals() async {
        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 50_000_000
            ) { _ in
                try await Task.sleep(nanoseconds: 5_000_000_000)
            }
            Issue.record("Expected LeaseTimeoutError")
        } catch is LeaseTimeoutError {
            // Expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func longBodyNotCancelledAfterSignal() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 100_000_000
        ) { signal in
            signal.set()
            try await Task.sleep(nanoseconds: 300_000_000)
            completed.set()
        }

        #expect(completed.isSet)
    }

    @Test func fastBodyCompletesBeforeTimeout() async throws {
        let completed = LeaseAcquiredSignal()

        try await CompletionHandler.withAcquisitionTimeout(
            timeoutNanoseconds: 1_000_000_000
        ) { signal in
            signal.set()
            completed.set()
        }

        #expect(completed.isSet)
    }

    @Test func bodyErrorPropagatesNotTimeout() async {
        struct BodyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { signal in
                signal.set()
                throw BodyError()
            }
            Issue.record("Expected BodyError")
        } catch is BodyError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test func bodyErrorBeforeSignalPropagates() async {
        struct EarlyError: Error {}

        do {
            try await CompletionHandler.withAcquisitionTimeout(
                timeoutNanoseconds: 1_000_000_000
            ) { _ in
                throw EarlyError()
            }
            Issue.record("Expected EarlyError")
        } catch is EarlyError {
            // Expected
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @MainActor
    @Test func makeGenerateParametersUsesModelDefaultsWhenRequestOmitsSampling() {
        let request = OpenAI.ChatCompletionRequest(
            model: "qwen3.5-4b-paro",
            messages: [.init(role: .user, content: .text("hi"))]
        )
        let modelState = ServerInferenceModelState(
            modelID: "qwen3.5-4b-paro",
            visionMode: false,
            triAttention: .v1Disabled
        )

        let params = CompletionHandler.makeGenerateParameters(from: request, modelState: modelState)

        #expect(params.maxTokens == AgentGenerateParameters.qwen35.maxTokens)
        #expect(params.temperature == AgentGenerateParameters.qwen35.temperature)
        #expect(params.topP == AgentGenerateParameters.qwen35.topP)
        #expect(params.topK == AgentGenerateParameters.qwen35.topK)
        #expect(params.minP == AgentGenerateParameters.qwen35.minP)
        #expect(params.presencePenalty == AgentGenerateParameters.qwen35.presencePenalty)
        #expect(params.repetitionPenalty == AgentGenerateParameters.qwen35.repetitionPenalty)
        #expect(params.triAttention == .v1Disabled)
    }

    @MainActor
    @Test func makeGenerateParametersAppliesRequestSamplingOverrides() {
        let request = OpenAI.ChatCompletionRequest(
            model: "qwen3.5-4b-paro",
            messages: [.init(role: .user, content: .text("hi"))],
            max_tokens: 4096,
            temperature: 0.4,
            top_p: 0.9,
            top_k: 48,
            min_p: 0.1,
            presence_penalty: 0.75,
            repetition_penalty: 1.1
        )
        let triAttention = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 4096,
            calibrationArtifactIdentity: .init(rawValue: "test-artifact")
        )
        let modelState = ServerInferenceModelState(
            modelID: "qwen3.5-4b-paro",
            visionMode: false,
            triAttention: triAttention
        )

        let params = CompletionHandler.makeGenerateParameters(from: request, modelState: modelState)

        #expect(params.maxTokens == 4096)
        #expect(params.temperature == 0.4)
        #expect(params.topP == 0.9)
        #expect(params.topK == 48)
        #expect(params.minP == 0.1)
        #expect(params.presencePenalty == 0.75)
        #expect(params.repetitionPenalty == 1.1)
        #expect(params.triAttention == triAttention)
    }

    @MainActor
    @Test func makeGenerateParametersTreatsNeutralRepetitionPenaltyAsDisabled() {
        let request = OpenAI.ChatCompletionRequest(
            model: "qwen3.5-4b-paro",
            messages: [.init(role: .user, content: .text("hi"))],
            repetition_penalty: 1.0
        )
        let modelState = ServerInferenceModelState(
            modelID: "qwen3.5-4b-paro",
            visionMode: false,
            triAttention: .v1Disabled
        )

        let params = CompletionHandler.makeGenerateParameters(from: request, modelState: modelState)

        #expect(params.repetitionPenalty == nil)
    }

    @MainActor @Test func nonStreamingResponseIncludesReasoningToolCallsAndUsage() throws {
        let info = AgentGeneration.Info(
            promptTokenCount: 100,
            generationTokenCount: 24,
            promptTime: 0.2,
            generateTime: 0.4,
            stopReason: .stop
        )
        let response = CompletionHandler.makeNonStreamingResponse(
            completionID: "chatcmpl-123",
            requestModel: "client-model",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            textContent: "Done",
            thinkingContent: "Need to inspect first.",
            toolCalls: [
                ToolCall(function: .init(
                    name: "bash",
                    arguments: ["command": "ls"]
                )),
            ],
            info: info,
            cachedTokenCount: 12,
            maxTokens: 256
        )

        #expect(response.model == "client-model")
        #expect(response.choices[0].finish_reason == .tool_calls)
        #expect(response.choices[0].message.content == "Done")
        #expect(response.choices[0].message.reasoning_content == "Need to inspect first.")
        let toolCall = try #require(response.choices[0].message.tool_calls?.first)
        #expect(toolCall.function?.name == "bash")
        #expect(response.usage?.prompt_tokens == 100)
        #expect(response.usage?.completion_tokens == 24)
        #expect(response.usage?.total_tokens == 124)
        #expect(response.usage?.prompt_tokens_details?.cached_tokens == 12)
    }

    @MainActor @Test func nonStreamingResponseUsesLengthFinishReasonAtMaxTokens() {
        let info = AgentGeneration.Info(
            promptTokenCount: 80,
            generationTokenCount: 32,
            promptTime: 0.1,
            generateTime: 0.3,
            stopReason: .length
        )
        let response = CompletionHandler.makeNonStreamingResponse(
            completionID: "chatcmpl-456",
            requestModel: "   ",
            physicalModelID: "qwen3.5-9b-paro",
            created: 1_712_345_678,
            textContent: "Truncated",
            thinkingContent: "",
            toolCalls: [],
            info: info,
            cachedTokenCount: 0,
            maxTokens: 32
        )

        #expect(response.model == "qwen3.5-9b-paro")
        #expect(response.choices[0].finish_reason == .length)
        #expect(response.choices[0].message.content == "Truncated")
        #expect(response.choices[0].message.reasoning_content == nil)
        #expect(response.choices[0].message.tool_calls == nil)
    }

    @MainActor @Test func finalStreamingChunkIncludesUsageWhenRequested() {
        let info = AgentGeneration.Info(
            promptTokenCount: 60,
            generationTokenCount: 15,
            promptTime: 0.1,
            generateTime: 0.2,
            stopReason: .stop
        )
        let chunk = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-789",
            requestModel: "client-model",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: false,
            info: info,
            cachedTokenCount: 7,
            maxTokens: 256,
            includeUsage: true
        )

        #expect(chunk.model == "client-model")
        #expect(chunk.choices[0].finish_reason == .stop)
        #expect(chunk.usage?.prompt_tokens == 60)
        #expect(chunk.usage?.completion_tokens == 15)
        #expect(chunk.usage?.prompt_tokens_details?.cached_tokens == 7)
    }

    @MainActor @Test func finalStreamingChunkOmitsUsageWhenNotRequested() {
        let info = AgentGeneration.Info(
            promptTokenCount: 60,
            generationTokenCount: 15,
            promptTime: 0.1,
            generateTime: 0.2,
            stopReason: .stop
        )
        let chunk = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-999",
            requestModel: nil,
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: true,
            info: info,
            cachedTokenCount: 7,
            maxTokens: 256,
            includeUsage: false
        )

        #expect(chunk.model == "qwen3.5-4b-paro")
        #expect(chunk.choices[0].finish_reason == .tool_calls)
        #expect(chunk.usage == nil)
    }

    /// Epic 3 Task 3 parity gate — the OpenAI response envelope is invariant
    /// over attention mode. `makeNonStreamingResponse` and
    /// `makeFinalStreamingChunk` do not take an attention-mode parameter: the
    /// HTTP contract sees only `modelID`, `cachedTokenCount`, and generation
    /// `info`. Identical inputs must therefore produce byte-identical envelopes
    /// regardless of whether the engine ran dense or TriAttention underneath.
    ///
    /// This regression-guards against future refactors that might thread an
    /// attention-mode flag into envelope construction and accidentally leak
    /// runtime-mode information into the OpenAI-compatible response.
    @MainActor @Test func responseEnvelopesDoNotVaryWhenInputsAreIdentical() throws {
        let info = AgentGeneration.Info(
            promptTokenCount: 120,
            generationTokenCount: 30,
            promptTime: 0.2,
            generateTime: 0.4,
            stopReason: .stop
        )

        let firstNonStreaming = CompletionHandler.makeNonStreamingResponse(
            completionID: "chatcmpl-parity",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            textContent: "Answer",
            thinkingContent: "Reasoning",
            toolCalls: [],
            info: info,
            cachedTokenCount: 19,
            maxTokens: 512
        )
        let secondNonStreaming = CompletionHandler.makeNonStreamingResponse(
            completionID: "chatcmpl-parity",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            textContent: "Answer",
            thinkingContent: "Reasoning",
            toolCalls: [],
            info: info,
            cachedTokenCount: 19,
            maxTokens: 512
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let firstData = try encoder.encode(firstNonStreaming)
        let secondData = try encoder.encode(secondNonStreaming)
        #expect(firstData == secondData)

        let firstStreaming = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-parity",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: false,
            info: info,
            cachedTokenCount: 19,
            maxTokens: 512,
            includeUsage: true
        )
        let secondStreaming = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-parity",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: false,
            info: info,
            cachedTokenCount: 19,
            maxTokens: 512,
            includeUsage: true
        )
        let firstChunkData = try encoder.encode(firstStreaming)
        let secondChunkData = try encoder.encode(secondStreaming)
        #expect(firstChunkData == secondChunkData)
    }

    /// Epic 3 Task 3 parity gate — `cached_tokens` accounting contract.
    ///
    /// The `cached_tokens` value reported by the engine must flow verbatim into
    /// both the non-streaming `usage.prompt_tokens_details.cached_tokens` and
    /// the streaming final chunk's identical field. No transform, no mode
    /// gating. This test documents the contract: the same engine-reported
    /// count reaches both envelopes unchanged.
    @MainActor @Test func cachedTokenCountFlowsUnchangedIntoBothEnvelopes() {
        let info = AgentGeneration.Info(
            promptTokenCount: 200,
            generationTokenCount: 42,
            promptTime: 0.5,
            generateTime: 0.9,
            stopReason: .stop
        )

        let nonStreaming = CompletionHandler.makeNonStreamingResponse(
            completionID: "chatcmpl-cached",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            textContent: "Ok",
            thinkingContent: "",
            toolCalls: [],
            info: info,
            cachedTokenCount: 77,
            maxTokens: 1024
        )
        let streaming = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-cached",
            requestModel: "qwen3.5-4b-paro",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: false,
            info: info,
            cachedTokenCount: 77,
            maxTokens: 1024,
            includeUsage: true
        )

        #expect(nonStreaming.usage?.prompt_tokens_details?.cached_tokens == 77)
        #expect(streaming.usage?.prompt_tokens_details?.cached_tokens == 77)
        #expect(
            nonStreaming.usage?.prompt_tokens_details?.cached_tokens
                == streaming.usage?.prompt_tokens_details?.cached_tokens
        )
        #expect(nonStreaming.usage?.prompt_tokens == streaming.usage?.prompt_tokens)
        #expect(nonStreaming.usage?.completion_tokens == streaming.usage?.completion_tokens)
        #expect(nonStreaming.usage?.total_tokens == streaming.usage?.total_tokens)
    }

    @MainActor @Test func finalStreamingChunkOmitsUsageWhenMetricsAreUnavailable() {
        let chunk = CompletionHandler.makeFinalStreamingChunk(
            completionID: "chatcmpl-1000",
            requestModel: "client-model",
            physicalModelID: "qwen3.5-4b-paro",
            created: 1_712_345_678,
            hasToolCalls: false,
            info: nil,
            cachedTokenCount: 7,
            maxTokens: 256,
            includeUsage: true
        )

        #expect(chunk.model == "client-model")
        #expect(chunk.choices[0].finish_reason == .stop)
        #expect(chunk.usage == nil)
    }
}

// MARK: - ModelSelection resolver (per-request model switching)
//
// These tests exercise the pure `CompletionHandler.resolveModelSelection`
// function, which decides how `request.model` routes. They verify the
// contract documented in docs/HTTP_SERVER_SPEC.md §4.2 Model Routing:
// - missing / empty / whitespace-only → .useSettings
// - exact id + downloaded → .override
// - exact id + not downloaded → .notDownloaded
// - anything else → .unknown (no displayName fallback, no trimming)

struct ModelSelectionTests {

    private static let qwen4 = "qwen3.5-4b-paro"
    private static let qwen9 = "qwen3.5-9b-paro"
    private static let agentIDs = [qwen4, qwen9, "qwen3.5-4b"]

    private static let downloadedStatuses: [String: ModelStatus] = [
        qwen4: .downloaded(sizeOnDisk: 1),
        qwen9: .downloaded(sizeOnDisk: 1),
        "qwen3.5-4b": .downloaded(sizeOnDisk: 1),
    ]

    @Test func resolvesNilToUseSettings() {
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: nil,
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .useSettings
        )
    }

    @Test func resolvesEmptyStringToUseSettings() {
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: "",
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .useSettings
        )
    }

    @Test func resolvesWhitespaceOnlyToUseSettings() {
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: "   ",
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .useSettings
        )
    }

    @Test func resolvesKnownAndDownloadedToOverride() {
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: Self.qwen9,
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .override(Self.qwen9)
        )
    }

    @Test func resolvesUnknownIDToUnknown() {
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: "gpt-4",
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .unknown("gpt-4")
        )
    }

    @Test func resolvesKnownButNotDownloadedToNotDownloaded() {
        let statuses: [String: ModelStatus] = [
            Self.qwen4: .downloaded(sizeOnDisk: 1),
            Self.qwen9: .notDownloaded,
            "qwen3.5-4b": .downloaded(sizeOnDisk: 1),
        ]
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: Self.qwen9,
                agentIDs: Self.agentIDs,
                statuses: statuses
            ) == .notDownloaded(Self.qwen9)
        )
    }

    @Test func resolvesKnownButDownloadingToNotDownloaded() {
        let statuses: [String: ModelStatus] = [
            Self.qwen9: .downloading(progress: 0.5),
        ]
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: Self.qwen9,
                agentIDs: Self.agentIDs,
                statuses: statuses
            ) == .notDownloaded(Self.qwen9)
        )
    }

    @Test func resolvesKnownButErroredToNotDownloaded() {
        let statuses: [String: ModelStatus] = [
            Self.qwen9: .error("network failure"),
        ]
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: Self.qwen9,
                agentIDs: Self.agentIDs,
                statuses: statuses
            ) == .notDownloaded(Self.qwen9)
        )
    }

    @Test func resolvesKnownButMissingFromStatusesToNotDownloaded() {
        // Agent IDs contain qwen9 but statuses dict is empty (not yet
        // refreshed, for instance). Should fail closed → .notDownloaded.
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: Self.qwen9,
                agentIDs: Self.agentIDs,
                statuses: [:]
            ) == .notDownloaded(Self.qwen9)
        )
    }

    @Test func resolverRejectsDisplayName() {
        // Display names ("Qwen3.5-9B PARO") must NOT match the canonical id
        // ("qwen3.5-9b-paro"). Exact match only per decision #3.
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: "Qwen3.5-9B PARO",
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .unknown("Qwen3.5-9B PARO")
        )
    }

    @Test func resolverRejectsTrailingWhitespace() {
        // A trailing space is NOT trimmed for the comparison — trimming is
        // only used for the is-empty check. Catching a silent client config
        // mistake is the whole point.
        #expect(
            CompletionHandler.resolveModelSelection(
                requestModel: "\(Self.qwen9) ",
                agentIDs: Self.agentIDs,
                statuses: Self.downloadedStatuses
            ) == .unknown("\(Self.qwen9) ")
        )
    }
}

// MARK: - echoModelID (per-request model switching)
//
// Covers the bug where `{"model":"   "}` round-tripped through the
// non-streaming and streaming echo sites — the raw whitespace string would
// appear in the response body. The helper substitutes the physical model ID
// when the client sent nothing useful.

struct EchoModelIDTests {

    @Test func echoReturnsPhysicalForNilRequest() {
        #expect(
            CompletionHandler.echoModelID(
                requestModel: nil,
                physical: "qwen3.5-4b-paro"
            ) == "qwen3.5-4b-paro"
        )
    }

    @Test func echoReturnsPhysicalForEmptyRequest() {
        #expect(
            CompletionHandler.echoModelID(
                requestModel: "",
                physical: "qwen3.5-4b-paro"
            ) == "qwen3.5-4b-paro"
        )
    }

    @Test func echoReturnsPhysicalForWhitespaceRequest() {
        #expect(
            CompletionHandler.echoModelID(
                requestModel: "   ",
                physical: "qwen3.5-4b-paro"
            ) == "qwen3.5-4b-paro"
        )
    }

    @Test func echoReturnsRequestVerbatimWhenNonEmpty() {
        // Echo preserves exactly what the client sent, regardless of
        // whether it matched the resolver. `"Qwen3.5-9B PARO"` would fail
        // the resolver (display name, not canonical id) — but if it reached
        // the echo helper, it would be echoed verbatim. That's OpenAI's
        // contract: echo what the client sent.
        #expect(
            CompletionHandler.echoModelID(
                requestModel: "Qwen3.5-9B PARO",
                physical: "qwen3.5-9b-paro"
            ) == "Qwen3.5-9B PARO"
        )
    }
}

// MARK: - HTTPServer Integration Tests

@MainActor
struct HTTPServerIntegrationTests {

    @Test func healthEndpointReturnsOK() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/health") { _, writer in
            try await writer.send(.json(["status": "ok"] as [String: String]))
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: String]
        #expect(json["status"] == "ok")
    }

    @Test func notFoundForUnknownPath() async throws {
        let server = HTTPServer(port: 0)
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/nonexistent")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 404)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let error = json["error"] as? [String: Any]
        #expect(error?["code"] as? Int == 404)
    }

    @Test func methodNotAllowedReturnsAllowHeader() async throws {
        let server = HTTPServer(port: 0)
        server.route(.POST, "/only-post") { _, writer in
            try await writer.send(.json(["ok": true] as [String: Bool]))
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/only-post")!)
        request.httpMethod = "GET"
        let (_, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 405)
        #expect(http.value(forHTTPHeaderField: "Allow")?.contains("POST") == true)
    }

    @Test func sseStreamDeliversChunksAndDone() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/test-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()
            for i in 1...3 {
                await sse.send(["n": "\(i)"] as [String: String])
            }
            await sse.done()
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (bytes, response) = try await URLSession.shared.bytes(
            from: URL(string: "http://127.0.0.1:\(port)/test-sse")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)

        var lines: [String] = []
        for try await line in bytes.lines {
            lines.append(line)
            if line == "data: [DONE]" { break }
        }

        let dataLines = lines.filter { $0.hasPrefix("data: {") }
        #expect(dataLines.count == 3)
        #expect(lines.last == "data: [DONE]")
    }

    @Test func sseStreamSupportsReasoningChunks() async throws {
        let server = HTTPServer(port: 0)
        server.route(.GET, "/test-reasoning-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            await sse.sendRaw(
                #"{"choices":[{"delta":{"reasoning_content":"Thinking...","role":"assistant"},"index":0}],"created":1712345678,"id":"chatcmpl-reason","model":"qwen3.5-4b-paro","object":"chat.completion.chunk"}"#
            )
            await sse.sendRaw(
                #"{"choices":[{"delta":{"content":"Hello"},"index":0}],"created":1712345678,"id":"chatcmpl-reason","model":"qwen3.5-4b-paro","object":"chat.completion.chunk"}"#
            )
            await sse.done()
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (bytes, response) = try await URLSession.shared.bytes(
            from: URL(string: "http://127.0.0.1:\(port)/test-reasoning-sse")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 200)

        var payloads: [OpenAI.ChatCompletionChunk] = []
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            if line == "data: [DONE]" { break }

            let json = Data(line.dropFirst(6).utf8)
            payloads.append(try JSONDecoder().decode(OpenAI.ChatCompletionChunk.self, from: json))
        }

        #expect(payloads.count == 2)
        #expect(payloads[0].choices[0].delta.reasoning_content == "Thinking...")
        #expect(payloads[0].choices[0].delta.content == nil)
        #expect(payloads[1].choices[0].delta.content == "Hello")
        #expect(payloads.allSatisfy { $0.choices[0].delta.content != "<think>" })
    }

    @Test func sseWriterDetectsDisconnect() async throws {
        // Verify that SSEWriter.send returns false when the connection fails,
        // and that the handler does not run all 200 iterations.
        let chunksSent = LeaseAcquiredSignal() // reuse as "at least some sent" flag
        let handlerDone = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/slow-sse") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()
            var sent = 0
            for i in 1...200 {
                let ok = await sse.send(["n": "\(i)"] as [String: String])
                if !ok { break }
                sent += 1
                if sent == 2 { chunksSent.set() }
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            handlerDone.set()
        }
        let port = try await startOnRandomPort(server)

        // Connect, read 1 chunk to confirm stream works, then stop server
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/slow-sse")!
            )
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { break }
            }
        }

        try? await readTask.value
        // Server stop cancels connection tasks, causing writes to fail
        server.stop()

        try await Task.sleep(nanoseconds: 500_000_000)
        // Handler must have exited (not still running all 200 iterations)
        #expect(handlerDone.isSet)
    }

    // MARK: - Prefill Disconnect

    @Test func prefillDisconnectCancelsGeneration() async throws {
        // Exercises the exact task-group pattern from runStreamingCompletion:
        // keepalive detects disconnect → throws → cancels generation child task.
        let generationCancelled = LeaseAcquiredSignal()
        let handlerExited = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/prefill-disconnect") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            // Send initial role chunk so the client can connect
            await sse.send(["role": "assistant"] as [String: String])

            // Simulate the CompletionHandler task group pattern:
            // keepalive monitors for disconnect, generation blocks on prefill.
            struct Disconnected: Error {}

            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    // Keepalive: check connection every 100ms (fast for testing)
                    group.addTask {
                        while true {
                            try await Task.sleep(nanoseconds: 100_000_000)
                            try Task.checkCancellation()
                            guard await sse.keepalive("keepalive") else {
                                throw Disconnected()
                            }
                        }
                    }

                    // Fake "generation" that blocks for 30s (simulating prefill)
                    group.addTask {
                        do {
                            try await Task.sleep(nanoseconds: 30_000_000_000)
                        } catch is CancellationError {
                            generationCancelled.set()
                        }
                    }

                    try await group.next()
                    group.cancelAll()
                }
            } catch is Disconnected {
                // Expected: keepalive detected client gone
            } catch {}

            handlerExited.set()
        }
        let port = try await startOnRandomPort(server)

        // Connect and read the initial chunk, then disconnect by stopping server
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/prefill-disconnect")!
            )
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { break }
            }
        }

        try? await readTask.value
        server.stop()

        // The keepalive should detect disconnect within ~200ms, cancel generation
        try await Task.sleep(nanoseconds: 500_000_000)
        #expect(generationCancelled.isSet)
        #expect(handlerExited.isSet)
    }

    // MARK: - OpenAI Error Shape (model_not_found)

    @Test func modelNotFoundUnknownReturnsOpenAIShape() async throws {
        let server = HTTPServer(port: 0)
        server.route(.POST, "/v1/chat/completions") { _, writer in
            try await writer.send(.modelNotFound(modelID: "gpt-4", reason: .unknownID))
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"gpt-4","messages":[]}"#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 404)

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let error = json["error"] as? [String: Any]
        #expect(error != nil)
        #expect(error?["type"] as? String == "invalid_request_error")
        // Strict shape: `code` is a string, not an integer.
        #expect(error?["code"] as? String == "model_not_found")
        #expect(error?["param"] as? String == "model")
        let message = error?["message"] as? String ?? ""
        #expect(message.contains("gpt-4"))
        #expect(message.contains("does not exist"))
    }

    @Test func modelNotFoundNotDownloadedReturnsOpenAIShape() async throws {
        let server = HTTPServer(port: 0)
        server.route(.POST, "/v1/chat/completions") { _, writer in
            try await writer.send(
                .modelNotFound(modelID: "qwen3.5-9b-paro", reason: .notDownloaded)
            )
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"qwen3.5-9b-paro","messages":[]}"#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 404)

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let error = json["error"] as? [String: Any]
        #expect(error?["type"] as? String == "invalid_request_error")
        #expect(error?["code"] as? String == "model_not_found")
        #expect(error?["param"] as? String == "model")
        let message = error?["message"] as? String ?? ""
        #expect(message.contains("qwen3.5-9b-paro"))
        #expect(message.contains("not downloaded"))
        #expect(message.contains("Settings"))
    }

    @Test func openAIErrorStrictPreservesNullParam() async throws {
        // Sanity check: when `param` is nil, the JSON body should still
        // include the key (as `null`) rather than omitting it entirely —
        // OpenAI SDK clients expect the key to be present.
        let server = HTTPServer(port: 0)
        server.route(.GET, "/custom-error") { _, writer in
            try await writer.send(
                .openAIError(
                    status: 400,
                    type: "invalid_request_error",
                    code: "generic_failure",
                    message: "something went wrong",
                    param: nil
                )
            )
        }
        let port = try await startOnRandomPort(server)
        defer { server.stop() }

        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/custom-error")!
        )
        let http = response as! HTTPURLResponse
        #expect(http.statusCode == 400)

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let error = json["error"] as! [String: Any]
        #expect(error["type"] as? String == "invalid_request_error")
        #expect(error["code"] as? String == "generic_failure")
        // `param` key must exist and be NSNull.
        #expect(error.keys.contains("param"))
        #expect(error["param"] is NSNull)
    }

    @Test func midStreamDisconnectBreaksGenerationLoop() async throws {
        // Verify that failed sse.send() breaks the labeled generation loop,
        // not just the switch statement.
        let loopExited = LeaseAcquiredSignal()

        let server = HTTPServer(port: 0)
        server.route(.GET, "/midstream-disconnect") { _, writer in
            let sse = SSEWriter(writer)
            try await sse.open()

            generation: for i in 1...500 {
                guard await sse.send(["n": "\(i)"] as [String: String]) else {
                    break generation
                }
                try? await Task.sleep(nanoseconds: 10_000_000)
                if Task.isCancelled { break generation }
            }
            loopExited.set()
        }
        let port = try await startOnRandomPort(server)

        // Read a few chunks then disconnect
        let readTask = Task {
            let (bytes, _) = try await URLSession.shared.bytes(
                from: URL(string: "http://127.0.0.1:\(port)/midstream-disconnect")!
            )
            var count = 0
            for try await line in bytes.lines {
                if line.hasPrefix("data: {") { count += 1 }
                if count >= 3 { break }
            }
        }

        try? await readTask.value
        server.stop()

        try await Task.sleep(nanoseconds: 500_000_000)
        // Loop must have exited — not still running 500 iterations
        #expect(loopExited.isSet)
    }

    // MARK: - Helpers

    /// Start server on a random available port, return the actual port.
    private func startOnRandomPort(_ server: HTTPServer) async throws -> UInt16 {
        // Port 0 isn't supported by NWListener, so find a free port
        let port = try findFreePort()
        await server.updatePort(port)
        await server.start()
        // Brief pause for listener to become ready
        try await Task.sleep(nanoseconds: 100_000_000)
        return port
    }

    private func findFreePort() throws -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw PortError() }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else { throw PortError() }

        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        guard nameResult == 0 else { throw PortError() }

        return UInt16(bigEndian: addr.sin_port)
    }
}

private struct PortError: Error {}
