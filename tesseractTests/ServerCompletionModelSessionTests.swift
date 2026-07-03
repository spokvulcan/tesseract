import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Collects `ServerInferenceProgressEvent`s across isolations: the handler
/// fires on the MainActor, assertions read after the drive settles.
private nonisolated final class ProgressEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [ServerInferenceProgressEvent] = []

    var events: [ServerInferenceProgressEvent] {
        lock.withLock { _events }
    }

    func append(_ event: ServerInferenceProgressEvent) {
        lock.withLock { _events.append(event) }
    }
}

/// First sequencing coverage at the **Model Session** seam (PRD #137, PR A;
/// ADR-0016): the **Unkeyed Completion** arm — the smallest complete
/// consumer — driven end-to-end with the toy-model peer. The real
/// `StateThreadedTokenIterator` runs its genuine prime forward, the real
/// generation loop detokenizes and stops on the scripted EOS; assertions
/// cover the emitted stream, the resulting cache state, and the verb order
/// on the session — the seam's contract.
@Suite struct ServerCompletionUnkeyedSequencingTests {

    private static let userText = "Hi"
    private static let completionText = "Hello!"

    private static func promptTokens(_ tokenizer: FakeChatMLTokenizer) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": userText]],
            tools: nil,
            additionalContext: nil
        )
    }

    private static func makeProvider() throws -> (
        ToyModelSessionProvider, prompt: [Int], completion: [Int]
    ) {
        let tokenizer = FakeChatMLTokenizer()
        let prompt = try promptTokens(tokenizer)
        let completion = Array(completionText.utf8).map(Int.init)
        let model = ToyLanguageModel(script: prompt + completion)
        return (ToyModelSessionProvider(model: model, tokenizer: tokenizer), prompt, completion)
    }

    private static func diagnostics() -> PrefixCacheDiagnostics.Context {
        PrefixCacheDiagnostics.Context(
            requestID: UUID(), modelID: "toy/model", kvBits: nil, kvGroupSize: 64
        )
    }

    @Test func unkeyedArmStreamsScriptedCompletionInVerbOrder() async throws {
        let (provider, prompt, completion) = try Self.makeProvider()
        let progress = ProgressEventLog()

        let generation = try await provider.withSession { session in
            let fullInput = try await session.prepare(
                UserInput(messages: [["role": "user", "content": Self.userText]])
            )
            let fullTokens = LLMActor.extractTokenSequence(fullInput.text.tokens)
            return try await ServerCompletion.makeUnkeyedGeneration(
                session: session,
                fullInput: fullInput,
                fullTokens: fullTokens,
                reason: .unrecognizedPlaceholderFamily,
                parameters: GenerateParameters(temperature: 0),
                toolSpecs: nil,
                partitionKey: CachePartitionKey(
                    modelID: "toy/model", kvBits: nil, kvGroupSize: 64
                ),
                fullAttentionScratchProfile: nil,
                visionAttentionScratchProfile: nil,
                ssdEnabled: false,
                diagnosticsContext: Self.diagnostics(),
                progressHandler: { event in progress.append(event) }
            )
        }

        // The toy processor's prepared tokens must match the script's prompt —
        // otherwise every downstream assertion is about the wrong sequence.
        #expect(generation.fullTokens == prompt)

        var text = ""
        var info: GenerateCompletionInfo?
        for await event in generation.stream {
            switch event {
            case .chunk(let chunk):
                text += chunk
            case .info(let completionInfo):
                info = completionInfo
            case .toolCall, .toolCallBufferDelta:
                Issue.record("unexpected tool-call event in scripted plain-text completion")
            }
        }
        await generation.completion.value

        // Externally visible stream behaviour: the scripted completion, then
        // the authoritative `.info` with a genuine stop-token finish.
        #expect(text == Self.completionText)
        let completionInfo = try #require(info)
        #expect(completionInfo.promptTokenCount == prompt.count)
        #expect(completionInfo.generationTokenCount == completion.count)
        guard case .stop = completionInfo.stopReason else {
            Issue.record("expected .stop, got \(completionInfo.stopReason)")
            return
        }

        // Cache state afterward: the whole prompt, the scripted completion,
        // and the final forward that produced the EOS.
        let finalOffset = generation.finalCache.first?.offset
        #expect(finalOffset == prompt.count + completion.count + 1)

        // Unkeyed metadata: zero cache participation, by contract.
        #expect(generation.unkeyedReason == .unrecognizedPlaceholderFamily)
        #expect(generation.skippedPrefillTokens == 0)
        #expect(generation.promptTokenCount == prompt.count)
        #expect(generation.snapshotAdmission == nil)

        // The seam's contract: verb order on the session. Text-only, so the
        // vision-continuation query is never made.
        #expect(
            provider.recorder.verbs == [.prepare, .newCache, .makePreparingDecodeIterator]
        )

        // Progress events: one started/finished pair, nothing cached.
        let events = progress.events
        #expect(
            events.first
                == .prefillStarted(
                    .init(
                        promptTokens: prompt.count,
                        cachedTokens: 0,
                        newTokensToPrefill: prompt.count,
                        prefillMs: nil
                    ))
        )
        #expect(events.count == 2)
        if case .prefillFinished(let finished)? = events.last {
            #expect(finished.promptTokens == prompt.count)
            #expect(finished.cachedTokens == 0)
            #expect(finished.prefillMs != nil)
        } else {
            Issue.record("expected prefillFinished as the second progress event")
        }
    }

    /// An image-bearing unkeyed request on a model without the windowed
    /// vision continuation: the arm must *query* the continuation (after
    /// cache creation) and fall back to the single-shot prepare.
    @Test func unkeyedArmQueriesVisionContinuationForImageInput() async throws {
        let (provider, prompt, completion) = try Self.makeProvider()

        let generation = try await provider.withSession { session in
            let prepared = try await session.prepare(
                UserInput(messages: [["role": "user", "content": Self.userText]])
            )
            // Attach a tiny processed image so the arm takes its image
            // branch; the toy model is not a `WindowedVisionContinuation`,
            // so the query returns nil and the single-shot path runs.
            let fullInput = LMInput(
                text: prepared.text,
                image: LMInput.ProcessedImage(
                    pixels: MLXArray.zeros([4, 3]),
                    frames: [THW(1, 2, 2)]
                )
            )
            return try await ServerCompletion.makeUnkeyedGeneration(
                session: session,
                fullInput: fullInput,
                fullTokens: LLMActor.extractTokenSequence(fullInput.text.tokens),
                reason: .placeholderRunCountMismatch,
                parameters: GenerateParameters(temperature: 0),
                toolSpecs: nil,
                partitionKey: CachePartitionKey(
                    modelID: "toy/model", kvBits: nil, kvGroupSize: 64
                ),
                fullAttentionScratchProfile: nil,
                visionAttentionScratchProfile: nil,
                ssdEnabled: false,
                diagnosticsContext: Self.diagnostics(),
                progressHandler: nil
            )
        }

        var text = ""
        for await event in generation.stream {
            if case .chunk(let chunk) = event { text += chunk }
        }
        await generation.completion.value

        #expect(text == Self.completionText)
        #expect(generation.fullTokens == prompt)
        #expect(generation.unkeyedReason == .placeholderRunCountMismatch)
        #expect(generation.finalCache.first?.offset == prompt.count + completion.count + 1)
        #expect(
            provider.recorder.verbs
                == [.prepare, .newCache, .visionContinuationQuery, .makePreparingDecodeIterator]
        )
    }
}
