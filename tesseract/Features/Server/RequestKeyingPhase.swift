//
//  RequestKeyingPhase.swift
//  tesseract
//
//  The **Request Keying** phase of a cache-aware **Server Completion**: turn
//  one HTTP conversation into a **Keyed Request** — the identities every
//  later phase keys on (the global Marconi partition key, the request's
//  **Cache Key Space**, identity and free for text-only requests, and its
//  **Conversation Render**) plus every per-request fact those phases read,
//  derived once here (ADR-0070) — beside the prepared model input. A
//  key-space construction failure degrades the whole request to an
//  **Unkeyed Completion** — served normally, zero cache participation —
//  which carries the same facts and no keys, as a case of its own.
//
//  Runs inside the **Model Session** (ADR-0016): `prepare` is
//  tokenizer/processor-affine, so the phase is called from within the
//  session scope and hands back non-`Sendable` prepared input that must not
//  leave it.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// The per-request facts every phase after **Request Keying** reads,
/// derived once, where the request's instance truth lives (ADR-0070). Each
/// used to be re-derived by the phases that read it, and the copies drifted.
/// Constructible only by `RequestKeyingPhase.run`, so no phase can pair a
/// request with a fact it does not have.
nonisolated struct RequestFacts: Sendable {
    /// The prepared prompt as a flat token list (the 1D extraction of a
    /// possibly 2D vision tensor): real ids, safe to re-forward. The radix
    /// tree keys on the **Cache Key Path** instead, whose image runs are
    /// digest pseudo-tokens that must never reach an embedding.
    let promptTokens: [Int]
    /// Rank of the prepared token tensor: 1 on the LLM-class processor, 2
    /// (`[1, seq]`) on a vision container. The boundary residual re-prefill
    /// rebuilds inputs from a flat list at this rank.
    let tokenNDim: Int
    /// The partition the request's cache work routes to.
    let partitionKey: CachePartitionKey
    let renderContext: TemplateRenderContext
    /// The request's **Generation Prompt**, checked against what it fed.
    let generationPrompt: GenerationPrompt
    /// Whether the request defined tools: MTP engagement predicts they will
    /// be called, since emission is unknowable before decode.
    let toolsDefined: Bool
    /// Whether the request is text-only by instance truth: no image reached
    /// the model. A text-class instance drops a request's images (#439), so
    /// such a request is text-only by construction, however many images it
    /// carried. The Cache Claim check-out, the leaf capture, speculative-arm
    /// engagement and transient-boundary gating all read this one fact. For
    /// a keyed request it is exactly an identity **Cache Key Space**.
    let isTextOnly: Bool
    /// The chunked prefill step, for the request's own prefill and every
    /// re-prefill after it (the boundary residual, the speculative pass).
    let prefillStepSize: Int
    /// What every decode iterator runs with: the request's parameters with
    /// `kvBits` cleared, because the module quantizes the cache itself
    /// before the iterator sees it (ADR-0006).
    let decodeParameters: GenerateParameters
    /// The SSD tier gate, sampled once per request.
    let ssdEnabled: Bool
    /// The tool-call format the generation loop parses with, so the Emitted
    /// Path fidelity replay parses the emitted ids the same way.
    let toolCallFormat: ToolCallFormat

    var promptTokenCount: Int { promptTokens.count }

    /// The finished turn's leaf-store mode: the one rule over the think
    /// block and the render context (`LeafStorePhase.selectHTTPLeafStoreMode`).
    func leafStoreMode(emittedToolCalls: Bool) -> HTTPLeafStoreMode {
        LeafStorePhase.selectHTTPLeafStoreMode(
            generationPrompt: generationPrompt, renderContext: renderContext,
            emittedToolCalls: emittedToolCalls)
    }

    /// The mode MTP engagement predicts before decode: defined tools count
    /// as called.
    var predictedLeafStoreMode: HTTPLeafStoreMode {
        leafStoreMode(emittedToolCalls: toolsDefined)
    }

    fileprivate init(
        promptTokens: [Int], tokenNDim: Int, partitionKey: CachePartitionKey,
        renderContext: TemplateRenderContext, generationPrompt: GenerationPrompt,
        toolsDefined: Bool, isTextOnly: Bool, parameters: GenerateParameters, ssdEnabled: Bool,
        toolCallFormat: ToolCallFormat
    ) {
        self.promptTokens = promptTokens
        self.tokenNDim = tokenNDim
        self.partitionKey = partitionKey
        self.renderContext = renderContext
        self.generationPrompt = generationPrompt
        self.toolsDefined = toolsDefined
        self.isTextOnly = isTextOnly
        self.prefillStepSize = parameters.prefill.stepSize ?? 512
        var decodeParameters = parameters
        decodeParameters.kvBits = nil
        self.decodeParameters = decodeParameters
        self.ssdEnabled = ssdEnabled
        self.toolCallFormat = toolCallFormat
    }
}

/// A **Keyed Request** (CONTEXT.md): the identities every later phase keys
/// on, the request's **Cache Key Space** and **Conversation Render**, with
/// its facts. What resolution, planning, the drive, the Leaf Store phase and
/// the Speculative Canonical Prefill seed take whole.
nonisolated struct KeyedRequest: Sendable {
    let facts: RequestFacts
    /// The request's **Cache Key Space** — identity for text-only requests.
    /// Owns the key path the radix tree is driven with and translates every
    /// later render the same way.
    let keySpace: CacheKeySpace
    /// The request's **Conversation Render** — the one render authority
    /// every later phase (planner boundary, leaf-store measure, admission
    /// probes) draws on. Image-agnostic: its renders carry one pad per image,
    /// which `keySpace` translates.
    let render: ConversationRender
    /// The recognized vision container mis-positions M-RoPE on any
    /// nil-state warm forward — text-only restores included — so the
    /// Position Anchor is seeded whenever the family is recognized AND
    /// the loaded instance is that container, not just when this
    /// request carries images. A text-class instance of a vision family
    /// never reads the seeded state and gets `false`.
    let seedsPositionAnchor: Bool

    fileprivate init(
        facts: RequestFacts, keySpace: CacheKeySpace, render: ConversationRender,
        seedsPositionAnchor: Bool
    ) {
        self.facts = facts
        self.keySpace = keySpace
        self.render = render
        self.seedsPositionAnchor = seedsPositionAnchor
    }
}

/// An **Unkeyed Completion**'s request: no valid Cache Key Path could be
/// built, so it has no key space and no render, only its facts and why. A
/// case of its own, so no phase can read a placeholder key.
nonisolated struct UnkeyedRequest: Sendable {
    let facts: RequestFacts
    let reason: CacheKeySpace.UnkeyedReason

    fileprivate init(facts: RequestFacts, reason: CacheKeySpace.UnkeyedReason) {
        self.facts = facts
        self.reason = reason
    }
}

nonisolated enum RequestKeyingPhase {

    /// What keying made of the request. The prepared input rides beside the
    /// request and never leaves the **Model Session** (ADR-0016).
    enum Outcome {
        case keyed(KeyedRequest, input: LMInput)
        /// Key-space construction failed: serve an **Unkeyed Completion**
        /// from the prepared input with zero cache participation.
        case unkeyed(UnkeyedRequest, input: LMInput)
    }

    static func run(
        session: any ModelSession,
        conversation: HTTPPrefixCacheConversation,
        canonicalTools: [ToolSpec]?,
        renderContext: TemplateRenderContext,
        parameters: GenerateParameters,
        modelID: String,
        modelFingerprint: String?,
        imageKeying: ModelIdentity.ImageKeying?,
        ssdEnabled: Bool = false,
        emittedPathIndex: EmittedPathIndex = .shared,
        diagnostics: PrefixCacheDiagnostics.Context? = nil
    ) async throws -> Outcome {
        // 1. Tokenize the full conversation (BEFORE cache lookup). Images
        // ride along positionally: the renderer emits one `"image"` part
        // per attachment and the processor matches them in order.
        // `MessageConverter` already proved each payload `CIImage`-decodable.
        //
        // Instance truth over family intent: only a vision-container
        // instance routes images through a tower into the KV. An
        // `LLMModel`-class load — e.g. a vision-family checkpoint whose
        // weight layout the VLM factory rejects, silently falling back to
        // text (the mlx-community Qwen3.8-27B-4bit shape) — gets the
        // text-only processor, which ignores `UserInput.images` entirely:
        // image parts render as bare template placeholders and the bytes
        // never reach the model. Keying such a request on its images could
        // only degrade it to Unkeyed (`prepare` can never return grids),
        // zeroing cache participation for every turn of an image-bearing
        // conversation — so it keys text-only, matching what the model
        // actually sees, and the drop is logged as the served-degraded
        // fact it is.
        let requestImages = conversation.images
        let producesFlatTextTokens = session.producesFlatTextTokens
        let instanceProcessesImages = !producesFlatTextTokens
        if !requestImages.isEmpty, !instanceProcessesImages {
            Log.image.warning(
                "\(requestImages.count) image attachment(s) ignored — "
                    + "\(modelID) loaded as a text-only instance; serving and "
                    + "keying the request text-only")
        }
        let keyedImages = instanceProcessesImages ? requestImages : []
        // The family's vision claim, corrected once against the instance: a
        // text-class load never routes images or reads vision keying, so
        // every keying consumer below (key space, grid log, anchor) reads
        // this value, never the raw config-intent `imageKeying`.
        let effectiveImageKeying: ModelIdentity.ImageKeying? =
            instanceProcessesImages ? imageKeying : nil
        let userInputImages: [UserInput.Image] = try keyedImages.map { image in
            guard let decoded = CIImage(data: image.data) else {
                throw AgentEngineError.generationFailed(
                    "image attachment no longer decodes (digest \(image.digest.hexString.prefix(8)))"
                )
            }
            return .ciImage(decoded)
        }
        // C25 Render+Token Cache and the Emitted Path Resolve (ADR-0063):
        // every request renders through the cache on EITHER model class —
        // the `.messages` prompt reaches every processor's `generate(from:)`
        // unchanged, so the cache renders exactly what `prepare` would. This
        // is the ONE construction of the request's **Conversation Render**
        // — eligibility decided here, where instance truth lives, then
        // threaded to every later render. An unknown model fingerprint,
        // non-rendering tokenizers, and any render/encode failure fall back
        // to the processor.
        //
        // A text-only request (the instance-filtered image list is empty —
        // issue #439: a dropped-image request is text-only by construction)
        // is the rendered list at the processor's own rank (1D on the
        // LLM-class text processor, 2D `[1, seq]` on a vision container).
        // An image-bearing request runs the processor's `prepare` for its
        // pixels and grids, then takes the rendered list — the Emitted Path
        // composition, one pad per image — expanded at each pad into the run
        // the processor placed for that image. Until 2026-09-18 media was a
        // render ineligibility and every request after an image entered a
        // session re-encoded canonically: a Pi session against Bonsai 2 27B
        // read a PNG, diverged inside a 59 KB `write` turn and re-prefilled
        // 29,715 tokens from the system checkpoint.
        var render = ConversationRender.forRequest(
            tokenizer: session.tokenizer,
            toolSpecs: canonicalTools,
            renderContext: renderContext,
            modelFingerprint: modelFingerprint,
            emittedPathIndex: emittedPathIndex,
            diagnostics: diagnostics
        )
        let renderedTokens = render.fullRender(messages: conversation.promptMessages)
        let fullInput: LMInput
        if keyedImages.isEmpty, let renderedTokens {
            fullInput = session.textOnlyInput(tokens: renderedTokens)
        } else {
            let prepared = try await session.prepare(
                UserInput(
                    messages: conversation.promptMessages,
                    images: userInputImages,
                    tools: canonicalTools,
                    additionalContext: renderContext.additionalContext()
                )
            )
            switch imageBearingInput(
                rendered: renderedTokens,
                prepared: prepared,
                imagePadTokenId: effectiveImageKeying?.imagePadTokenId,
                diagnostics: diagnostics
            ) {
            case .expanded(let input):
                fullInput = input
            case .processorsOwn(let input, let mismatch):
                // The processor's tokens were fed, so nothing this request
                // renders may register or serve an emitted path.
                fullInput = input
                if mismatch { render = render.bypassing(.placeholderStructureMismatch) }
            }
        }
        // For LLM models tokens are 1D [seq], for VLM models (ParoQuant
        // Qwen35) they are 2D [batch, seq].
        let tokenNDim = fullInput.text.tokens.ndim

        // 2. Extract flat token sequence for radix tree operations.
        let fullTokens = LLMActor.extractTokenSequence(fullInput.text.tokens)

        // 3. Build the global Marconi partition key for this model
        //    configuration. Cross-session sharing is intentional:
        //    identical prompts under the same model config should
        //    reuse the same radix tree. The conversation's template-
        //    context digest separates render modes (issue #98) — it is
        //    the same digest the handler derived from `renderContext`.
        let partitionKey = CachePartitionKey(
            modelID: modelID,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            modelFingerprint: modelFingerprint,
            templateContextDigest: conversation.templateContextDigest
        )

        // The request's facts, once. Only the Generation Prompt differs by
        // outcome: it is checked against what the request feeds, the key
        // path when keyed, the prompt tokens when not.
        let toolCallFormat = session.configuration.toolCallFormat ?? .json
        func facts(checkedAgainst fed: [Int]) -> RequestFacts {
            RequestFacts(
                promptTokens: fullTokens, tokenNDim: tokenNDim, partitionKey: partitionKey,
                renderContext: renderContext,
                generationPrompt: render.checkedGenerationPrompt(
                    fed: fed, diagnostics: diagnostics),
                toolsDefined: canonicalTools?.isEmpty == false, isTextOnly: keyedImages.isEmpty,
                parameters: parameters,
                ssdEnabled: ssdEnabled, toolCallFormat: toolCallFormat)
        }

        // 3b. Build the request's **Cache Key Space** from the prepared
        // tokens, the conversation's images, and the family's image
        // keying. Identity (and free) for text-only requests. A
        // construction failure degrades the whole request to an **Unkeyed
        // Completion** — served normally, zero cache participation.
        let keySpace: CacheKeySpace
        switch CacheKeySpace.make(
            preparedTokens: fullTokens,
            imageDigests: keyedImages.map(\.digest),
            imageGrids: (fullInput.image?.frames ?? []).map { frame in
                let (t, h, w) = frame.values
                return (t: t, height: h, width: w)
            },
            imageKeying: effectiveImageKeying
        ) {
        case .success(let space):
            keySpace = space
        case .failure(let reason):
            return .unkeyed(
                UnkeyedRequest(facts: facts(checkedAgainst: fullTokens), reason: reason),
                input: fullInput)
        }
        // Grid instrumentation (ADR-0007 phase 2): the processed image grid
        // is the ground truth for the M-RoPE span and the pad-run length the
        // Cache Key Path expands. Logging it cheaply catches the deferred
        // "one screenshot → ~43,500 pad tokens" anomaly with real numbers if
        // it recurs — the chunked continuation already makes such an image
        // non-fatal, so this is observe-only, not a gate.
        if let frames = fullInput.image?.frames, !frames.isEmpty {
            let merge = effectiveImageKeying?.spatialMergeSize ?? 1
            let mergeArea = max(1, merge * merge)
            for (index, frame) in frames.enumerated() {
                let (t, h, w) = frame.values
                Log.image.debug(
                    "vision grid #\(index): t=\(t) h=\(h) w=\(w) "
                        + "patches=\(frame.product) padRun=\(frame.product / mergeArea) "
                        + "merge=\(merge)"
                )
            }
        }

        // 4. The keyed request. Its Generation Prompt is checked against the
        // key path it feeds, whose tail is text in every key space.
        return .keyed(
            KeyedRequest(
                facts: facts(checkedAgainst: keySpace.keyPath), keySpace: keySpace,
                render: render, seedsPositionAnchor: effectiveImageKeying != nil),
            input: fullInput)
    }

    /// The image-bearing request's model input.
    enum ImageBearingInput {
        /// The render's tokens — the Emitted Path composition, one pad per
        /// image — expanded at each pad into the run the processor placed,
        /// over the processor's pixels and grids.
        case expanded(LMInput)
        /// The processor's own prepared input: the render bypassed
        /// (`mismatch` false — nothing to expand, and the render is already
        /// out of the index), or the render's placeholders and the
        /// processor's runs did not pair up (`mismatch` true — the render
        /// must be taken out of the index for this request).
        case processorsOwn(LMInput, mismatch: Bool)
    }

    /// The image-bearing request's model input over the render's tokens:
    /// the processor's pixels and grids as prepared, its text replaced by
    /// the rendered list with each pad expanded into the run the processor
    /// placed for that image — in place, as `Qwen3VLProcessor`'s
    /// `replacePaddingTokens` does — at the processor's own rank and with
    /// its mask shape. The processor's own list stands when the render
    /// bypassed (`rendered` is nil), when the family has no placeholder
    /// identity, or when the render's placeholders and the processor's runs
    /// do not pair up (a processor that places its runs itself over a render
    /// that placed none — no production processor, but the toy's
    /// placeholder-free stub): that request keeps the processor's tokens,
    /// logs why, and its render leaves the index
    /// (`ConversationRender.bypassing(_:)`).
    private static func imageBearingInput(
        rendered: [Int]?,
        prepared: LMInput,
        imagePadTokenId: Int?,
        diagnostics: PrefixCacheDiagnostics.Context?
    ) -> ImageBearingInput {
        guard let rendered, let imagePadTokenId else {
            return .processorsOwn(prepared, mismatch: false)
        }
        let preparedTokens = LLMActor.extractTokenSequence(prepared.text.tokens)
        let runs = ImagePlaceholderRuns.runs(in: preparedTokens, padTokenId: imagePadTokenId)
        guard
            let expanded = ImagePlaceholderRuns.expand(
                renderTokens: rendered, padTokenId: imagePadTokenId,
                runLengths: runs.map(\.count))
        else {
            let placeholders = rendered.count { $0 == imagePadTokenId }
            diagnostics?.logSkip(
                stage: "imageRenderExpansion", reason: "placeholderStructureMismatch",
                extraFields: [
                    ("renderPlaceholders", "\(placeholders)"), ("preparedRuns", "\(runs.count)"),
                ])
            Log.image.debug(
                "image-bearing render kept the processor's tokens — "
                    + "renderPlaceholders=\(placeholders) preparedRuns=\(runs.count)")
            return .processorsOwn(prepared, mismatch: true)
        }
        let flat = MLXArray(expanded)
        let tokens = prepared.text.tokens.ndim == 1 ? flat : flat.expandedDimensions(axis: 0)
        let mask = prepared.text.mask.map { ones(like: tokens).asType($0.dtype) }
        return .expanded(
            LMInput(
                text: LMInput.Text(tokens: tokens, mask: mask),
                image: prepared.image,
                video: prepared.video,
                audio: prepared.audio
            ))
    }
}
