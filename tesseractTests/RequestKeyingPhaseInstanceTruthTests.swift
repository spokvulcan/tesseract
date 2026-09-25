import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Instance truth over family intent at the **Request Keying** phase: the
/// loaded model instance — not the family's `config.json` — decides whether
/// images participate in cache keying.
///
/// The regression this pins (2026-08-18): a vision-family checkpoint whose
/// weight layout the VLM factory rejects loads silently as a text-only
/// `LLMModel` instance (mlx-community Qwen3.8-27B-4bit). Its `prepare` can
/// never return image grids, so keying the request on its images degraded
/// EVERY turn of an image-bearing conversation to an **Unkeyed Completion**
/// (`image-grid-count-mismatch`) — 0% cache hit for a whole OpenCode session
/// over one pasted screenshot. The phase must key such requests text-only,
/// matching what the model actually sees.
@Suite struct RequestKeyingPhaseInstanceTruthTests {

    /// The family-recognized keying facts (the 27B config's claim): present
    /// even though the loaded instance is text-only — the disagreement under
    /// test.
    private static let visionFamilyKeying = ModelIdentity.ImageKeying(
        imagePadTokenId: 400, spatialMergeSize: 2
    )

    @MainActor
    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        // The toy's head dimension is far below any quantization group size;
        // KV quantization stays a recorded no-op verb in these suites.
        parameters.kvBits = nil
        return parameters
    }

    private static func imageConversation(imageData: Data) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: "look",
                    images: [HTTPPrefixCacheImage(data: imageData)]
                )
            ]
        )
    }

    /// The Sendable facts of an outcome — `Outcome` itself carries the
    /// non-Sendable prepared input that must not leave the session.
    private struct OutcomeFacts: Sendable {
        let isKeyed: Bool
        let isIdentity: Bool
        let seedsPositionAnchor: Bool
        let unkeyedReason: String?
        let fullTokens: [Int]
        /// The prepared input's token rank: 1 on the LLM-class text
        /// processor, 2 (`[batch, seq]`) on a vision container.
        let tokenNDim: Int
        /// Why the keyed request's render left the cache and the index, if
        /// it did; `nil` for an engaged render and for an unkeyed request.
        let renderIneligibility: String?
    }

    private static func runPhase(
        provider: ToyModelSessionProvider,
        conversation: HTTPPrefixCacheConversation,
        modelFingerprint: String? = nil,
        imageKeying: ModelIdentity.ImageKeying = visionFamilyKeying
    ) async throws -> OutcomeFacts {
        try await provider.withSession { session in
            switch try await RequestKeyingPhase.run(
                session: session,
                conversation: conversation,
                canonicalTools: nil,
                renderContext: .canonical,
                parameters: GenerateParameters(temperature: 0),
                modelID: "toy/model",
                modelFingerprint: modelFingerprint,
                imageKeying: imageKeying
            ) {
            case .keyed(let keyed, _):
                return OutcomeFacts(
                    isKeyed: true,
                    isIdentity: keyed.keySpace.isIdentity,
                    seedsPositionAnchor: keyed.seedsPositionAnchor,
                    unkeyedReason: nil,
                    fullTokens: keyed.facts.promptTokens,
                    tokenNDim: keyed.facts.tokenNDim,
                    renderIneligibility: keyed.render.ineligibility?.rawValue
                )
            case .unkeyed(let unkeyed, _):
                return OutcomeFacts(
                    isKeyed: false,
                    isIdentity: false,
                    seedsPositionAnchor: false,
                    unkeyedReason: unkeyed.reason.rawValue,
                    fullTokens: unkeyed.facts.promptTokens,
                    tokenNDim: unkeyed.facts.tokenNDim,
                    renderIneligibility: nil
                )
            }
        }
    }

    /// The fix: a text-only instance (LLM-class fallback of a vision-family
    /// checkpoint) keys an image-bearing request as text-only — keyed, with
    /// the identity key space and no anchor seeding — instead of degrading
    /// it to Unkeyed. The image bytes are deliberately undecodable: the
    /// text-only arm must not even reach the decode guard, because the
    /// dropped attachment never feeds the processor.
    @Test func textOnlyInstanceKeysImageRequestAsTextOnly() async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: ToySequencingTokenizer(),
            reportsFlatTextTokens: true
        )
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: Self.imageConversation(imageData: Data("not an image".utf8))
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity)
        #expect(outcome.seedsPositionAnchor == false)
        #expect(outcome.unkeyedReason == nil)
    }

    /// Issue #439: the dropped-image render is text-only by construction —
    /// the processor never sees the bytes, only the same content-array prompt
    /// the cache renders — so a text-only instance with a rendering tokenizer
    /// and a known fingerprint tokenizes an image-bearing conversation
    /// THROUGH the C25 Render+Token Cache: `prepare` never runs, and the
    /// keyed tokens are byte-identical to the fused `applyChatTemplate` the
    /// processor path would build. Before the fix, `hasMedia` keyed on the
    /// conversation's images and every such turn re-encoded the whole prompt.
    @Test func droppedImageRequestTokenizesThroughRenderTokenCache() async throws {
        let tokenizer = GreedyTokenizer(pieces: [
            "<|im_start|>", "<|im_end|>", "assistant", "user", "system",
            "\n", "look", " ",
        ])
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: tokenizer,
            reportsFlatTextTokens: true
        )
        let conversation = Self.imageConversation(imageData: Data("not an image".utf8))
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: conversation,
            modelFingerprint: "dropped-image-parity-\(UUID().uuidString)"
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity)
        #expect(outcome.seedsPositionAnchor == false)
        // The cache path was taken: no session verb — in particular no
        // `prepare` — was recorded for the whole keying phase.
        #expect(provider.recorder.verbs.isEmpty)
        // Render parity with the processor path: the toy processor's prepare
        // IS the fused `applyChatTemplate` over the same prompt messages.
        let truth = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        #expect(outcome.fullTokens == truth)
    }

    /// The 2026-09-18 Bonsai 2 27B session: a vision-container instance
    /// (its text-only `prepare` emits 2D tokens) serving a TEXT-ONLY
    /// request. The render cache and the Emitted Path Index it fronts must
    /// engage exactly as on a flat-token instance — `prepare` never runs,
    /// the keyed tokens are the fused render — and the prepared input must
    /// carry the processor's own rank, so every rank-keyed consumer
    /// downstream sees what the processor would have built. Before the fix
    /// the rank alone marked the render ineligible (`nonFlatTokens`), and a
    /// 18,939-token `write` turn re-prefilled in full.
    @Test func visionInstanceTextOnlyRequestTokenizesThroughRenderTokenCache() async throws {
        let tokenizer = GreedyTokenizer(pieces: [
            "<|im_start|>", "<|im_end|>", "assistant", "user", "system",
            "\n", "look", " ",
        ])
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: tokenizer,
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: Self.visionFamilyKeying.imagePadTokenId,
                padRunLength: 4,
                frame: THW(1, 8, 8)
            )
        )
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "look")]
        )
        let truth = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: conversation,
            modelFingerprint: "vision-text-only-\(UUID().uuidString)"
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity)
        // The container still seeds the Position Anchor on a text-only turn:
        // that reading is the instance's, not the render's.
        #expect(outcome.seedsPositionAnchor)
        #expect(outcome.fullTokens == truth)
        #expect(provider.recorder.verbs.isEmpty, "verbs: \(provider.recorder.verbs)")
        #expect(outcome.tokenNDim == 2)
    }

    /// Genuinely processed media — a vision-container instance, whose
    /// images survive the instance filter — always runs the processor's
    /// `prepare` (pixels, grids, the pad runs), and its own token list
    /// stands whenever the render's placeholders and the processor's runs
    /// do not pair up: here a template that renders no placeholder for the
    /// image part and a processor that puts its run after the messages.
    @Test func visionInstanceMediaWithoutARenderedPlaceholderKeepsTheProcessorsTokens()
        async throws
    {
        let tokenizer = GreedyTokenizer(pieces: [
            "<|im_start|>", "<|im_end|>", "assistant", "user", "system",
            "\n", "look", " ",
        ])
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: tokenizer,
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: Self.visionFamilyKeying.imagePadTokenId,
                padRunLength: 4,
                frame: THW(1, 8, 8)
            )
        )
        let conversation = Self.imageConversation(imageData: ImageTestFixtures.tinyPNGData)
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: conversation,
            modelFingerprint: "vision-media-guard-\(UUID().uuidString)"
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity == false)
        #expect(outcome.seedsPositionAnchor)
        #expect(provider.recorder.verbs == [.prepare])
        let pad = Self.visionFamilyKeying.imagePadTokenId
        let messagesEnd = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil,
            additionalContext: ["add_generation_prompt": false]
        ).count
        #expect(
            outcome.fullTokens
                == Array(render[..<messagesEnd]) + Array(repeating: pad, count: 4)
                + Array(render[messagesEnd...]))
        #expect(outcome.tokenNDim == 2)
        // The processor's tokens were fed, so the render leaves the index
        // for this request: it may neither register nor serve a path.
        #expect(outcome.renderIneligibility == "placeholderStructureMismatch")
    }

    /// The image-bearing request on a vision container (ADR-0063, amended
    /// 2026-09-18): the processor's `prepare` supplies pixels and grids; the
    /// token list is the render's — one pad per image, the Emitted Path
    /// composition — expanded at the pad into the run the processor placed,
    /// at the processor's rank. Keyed into the non-identity space.
    @Test func visionInstanceImageRequestExpandsTheRenderAtTheProcessorsRuns() async throws {
        let tokenizer = EmittedPathToyTokenizer()
        let pad = tokenizer.imagePadID
        let runLength = EmittedPathToyTokenizer.imagePadRunLength
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: tokenizer,
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: pad, padRunLength: runLength, frame: THW(1, 8, 8),
                expandsInPlace: true)
        )
        let conversation = Self.imageConversation(imageData: ImageTestFixtures.tinyPNGData)
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        #expect(render.count { $0 == pad } == 1)
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: conversation,
            modelFingerprint: "vision-media-render-\(UUID().uuidString)",
            imageKeying: ModelIdentity.ImageKeying(imagePadTokenId: pad, spatialMergeSize: 2)
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity == false)
        #expect(outcome.seedsPositionAnchor)
        #expect(provider.recorder.verbs == [.prepare])
        let expanded = render.flatMap { token in
            token == pad ? Array(repeating: token, count: runLength) : [token]
        }
        #expect(outcome.fullTokens == expanded)
        #expect(outcome.tokenNDim == 2)
        #expect(outcome.renderIneligibility == nil)
    }

    /// The guard the fix must NOT loosen: a vision-container instance whose
    /// `prepare` fails to attribute grids to the conversation's images (here:
    /// no vision stub, so no frames come back) still degrades to Unkeyed —
    /// grid mis-attribution on a real tower stays uncacheable.
    @Test func visionInstanceWithoutGridsStillDegradesToUnkeyed() async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: ToySequencingTokenizer()
        )
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: Self.imageConversation(imageData: ImageTestFixtures.tinyPNGData)
        )

        #expect(outcome.isKeyed == false)
        #expect(outcome.unkeyedReason == "image-grid-count-mismatch")
    }

    /// The genuine vision path stays whole: a vision-container instance whose
    /// `prepare` returns matching grids keys the image request into a
    /// non-identity key space with anchor seeding.
    @Test func visionInstanceWithGridsKeysImages() async throws {
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: [0]),
            tokenizer: ToySequencingTokenizer(),
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: Self.visionFamilyKeying.imagePadTokenId,
                padRunLength: 4,
                frame: THW(1, 8, 8)
            )
        )
        let outcome = try await Self.runPhase(
            provider: provider,
            conversation: Self.imageConversation(imageData: ImageTestFixtures.tinyPNGData)
        )

        #expect(outcome.isKeyed)
        #expect(outcome.isIdentity == false)
        #expect(outcome.seedsPositionAnchor)
        #expect(outcome.unkeyedReason == nil)
    }

    /// The end-to-end promise the incident broke: an image-bearing
    /// conversation on a text-only instance runs the KEYED spine and the
    /// second round restores the admitted leaf warm — the prompt cache works
    /// across a whole image-bearing session, not 0% forever.
    @Test func imageBearingConversationHitsCacheOnTextOnlyInstance() async throws {
        let tokenizer = ToySequencingTokenizer()
        let image = HTTPPrefixCacheImage(data: Data("opaque screenshot bytes".utf8))
        let round1 = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Hi", images: [image])]
        )
        let round2 = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: "Hi", images: [image]),
                .assistant(content: "Hello!"),
                HTTPPrefixCacheMessage(role: .user, content: "More?"),
            ]
        )

        let render1 = try tokenizer.applyChatTemplate(
            messages: round1.promptMessages, tools: nil, additionalContext: nil
        )
        let render2 = try tokenizer.applyChatTemplate(
            messages: round2.promptMessages, tools: nil, additionalContext: nil
        )
        #expect(Array(render2.prefix(render1.count)) == render1)
        let script = render2 + Array("Sure.".utf8).map(Int.init)

        // The 27B shape: a family identity that claims vision (imageKeying
        // non-nil) over a session that reports the LLM-class instance.
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5",
                "image_token_id": Self.visionFamilyKeying.imagePadTokenId,
                "vision_config": ["num_heads": 16, "spatial_merge_size": 2],
            ],
            chatTemplate: nil
        )
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script),
                tokenizer: tokenizer,
                reportsFlatTextTokens: true
            ),
            identity: identity
        )

        // -- Round 1: cold, but KEYED — the unkeyed arm's signature verb
        // (`makePreparingDecodeIterator`) must never appear.
        let handle1 = try await fixture.start(
            conversation: round1, parameters: Self.parameters()
        )
        #expect(handle1.diagnostics.cacheReason.hasPrefix("unkeyed") == false)
        #expect(handle1.cachedTokenCount == 0)
        let (text1, _) = try await collectServerText(handle1)
        #expect(text1 == "Hello!")
        let round1Verbs = fixture.provider.recorder.verbs
        #expect(
            round1Verbs == [
                .prepare, .newCache, .prefill, .quantizeKVCache, .makeDecodeIterator,
                .captureSnapshot,
            ]
        )

        // -- Round 2: warm. The admitted leaf restores and only the suffix
        // prefills — the incident's sessions never got here.
        let storedTokens1 = try tokenizer.applyChatTemplate(
            messages: round1.appendingAssistant(.assistant(content: "Hello!")).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        let handle2 = try await fixture.start(
            conversation: round2, parameters: Self.parameters()
        )
        #expect(handle2.diagnostics.cacheReason.hasPrefix("unkeyed") == false)
        #expect(handle2.cachedTokenCount == storedTokens1.count)
        let (text2, _) = try await collectServerText(handle2)
        #expect(text2 == "Sure.")
        let round2Verbs = Array(fixture.provider.recorder.verbs.dropFirst(round1Verbs.count))
        #expect(
            round2Verbs == [
                .prepare, .restore, .prefill, .quantizeKVCache, .makeDecodeIterator,
                .captureSnapshot,
            ]
        )

        await fixture.drain()
    }
}
