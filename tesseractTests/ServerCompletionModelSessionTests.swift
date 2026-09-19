import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite struct PrefixViewModelSessionTests {
    @Test func captureAtPreparedImagePrefixKeepsSystemOwnedAndBranchAsView() async throws {
        let provider = ToyModelSessionProvider(model: ToyLanguageModel(script: Array(1...8)))
        try await provider.withSession { session in
            let cache = try session.newCache(parameters: GenerateParameters())
            _ = try session.prefill(
                text: .init(tokens: MLXArray(Array(1...4).map(Int32.init))[.newAxis]), cache: cache,
                checkpoints: [:], checkpointBaseOffset: 0, prefillStepSize: 4,
                consumeAll: true, initialState: nil, evalPolicy: .checkedSynchronous)
            // The image-prefix edge captures directly after checked evaluation,
            // before the chunk loop starts at this absolute key-path offset.
            let branch = try #require(
                session.captureSnapshot(cache: cache, offset: 4, type: .branchPoint))
            let system = try #require(
                session.captureSnapshot(cache: cache, offset: 4, type: .system))
            #expect(branch.memoryBytes == 0)
            #expect(system.memoryBytes == 256)
            #expect(try session.restore(system).first?.offset == 4)
        }
    }

    @Test(arguments: [nil, 8] as [Int?])
    func forkFromViewMatchesOwnedCheckpointWithoutSharingBuffers(kvBits: Int?) async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: Array(1...12), headDim: 64))
        try await provider.withSession { session in
            let parameters = GenerateParameters(kvBits: kvBits, kvGroupSize: 64, temperature: 0)
            var live = try session.newCache(parameters: parameters)
            session.quantizeKVCache(&live, parameters: parameters)
            let prefilled = try session.prefill(
                text: .init(tokens: MLXArray(Array(1...8).map(Int32.init))), cache: live,
                checkpoints: [4: .branchPoint], checkpointBaseOffset: 0,
                prefillStepSize: 4, consumeAll: true, initialState: nil, evalPolicy: .pipelined)
            let view = try #require(prefilled.snapshots.first)
            let leaf = try #require(session.captureSnapshot(cache: live, offset: 8, type: .leaf))
            var baseline = try session.newCache(parameters: parameters)
            session.quantizeKVCache(&baseline, parameters: parameters)
            _ = try session.prefill(
                text: .init(tokens: MLXArray(Array(1...4).map(Int32.init))), cache: baseline,
                checkpoints: [:], checkpointBaseOffset: 0,
                prefillStepSize: 4, consumeAll: true, initialState: nil, evalPolicy: .pipelined)
            let owned = try #require(
                session.captureSnapshot(cache: baseline, offset: 4, type: .system))
            let fromOwned = try session.restore(owned)
            let fromView = try session.restore(view, backingLeaf: leaf)
            let treeAddresses = Set(leaf.layers.flatMap(\.state).map(backingAddress))
            for (actual, expected) in zip(fromView.flatMap(\.state), fromOwned.flatMap(\.state)) {
                #expect(!treeAddresses.contains(backingAddress(actual)))
                #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
            }
            let suffix = LMInput.Text(tokens: MLXArray([Int32(5)]))
            var viewIterator = session.makeDecodeIterator(
                remainder: suffix, fullText: suffix, cache: fromView, state: nil,
                parameters: parameters)
            var ownedIterator = session.makeDecodeIterator(
                remainder: suffix, fullText: suffix, cache: fromOwned, state: nil,
                parameters: parameters)
            let viewTokens = (0..<3).compactMap { _ in viewIterator.next() }
            let ownedTokens = (0..<3).compactMap { _ in ownedIterator.next() }
            #expect(viewTokens == [6, 7, 8])
            #expect(viewTokens == ownedTokens)
        }
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
            // branch; the toy model has no anchored vision `prepare`,
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
