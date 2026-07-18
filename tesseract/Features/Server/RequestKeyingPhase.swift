//
//  RequestKeyingPhase.swift
//  tesseract
//
//  The **Request Keying** phase of a cache-aware **Server Completion**: turn
//  one HTTP conversation into the identities every later phase keys on — the
//  prepared model input, the flat token sequence, the global Marconi
//  partition key, and the request's **Cache Key Space** (identity and free
//  for text-only requests). A key-space construction failure degrades the
//  whole request to an **Unkeyed Completion** — served normally, zero cache
//  participation — surfaced here as data for the caller to act on.
//  Previously steps 1–3b inline in the generation builder.
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

nonisolated enum RequestKeyingPhase {

    /// The identities of a keyed request — what resolution, planning, plan
    /// application, and the leaf store all key on.
    struct Keyed {
        let fullInput: LMInput
        let fullTokens: [Int]
        let fullTokenCount: Int
        let tokenNDim: Int
        let partitionKey: CachePartitionKey
        let keySpace: CacheKeySpace
        /// The M-RoPE vision container mis-positions on any nil-state warm
        /// forward — text-only restores included — so the Position Anchor is
        /// seeded whenever *that* family is loaded, not just when this
        /// request carries images. Sequential-rule families (Gemma 4
        /// unified) restore with nil-state semantics like text models and
        /// never seed.
        let seedsPositionAnchor: Bool
    }

    enum Outcome {
        case keyed(Keyed)
        /// Key-space construction failed: serve an **Unkeyed Completion**
        /// from the prepared input, under the partition key (for
        /// diagnostics), with zero cache participation.
        case unkeyed(
            fullInput: LMInput,
            fullTokens: [Int],
            partitionKey: CachePartitionKey,
            reason: CacheKeySpace.UnkeyedReason
        )
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
        audioKeying: ModelIdentity.AudioKeying? = nil
    ) async throws -> Outcome {
        // 1. Tokenize the full conversation (BEFORE cache lookup). Images
        // ride along positionally: the renderer emits one `"image"` part
        // per attachment and the processor matches them in order.
        // `MessageConverter` already proved each payload `CIImage`-decodable.
        let requestImages = conversation.images
        let userInputImages: [UserInput.Image] = try requestImages.map { image in
            guard let decoded = CIImage(data: image.data) else {
                throw AgentEngineError.generationFailed(
                    "image attachment no longer decodes (digest \(image.digest.hexString.prefix(8)))"
                )
            }
            return .ciImage(decoded)
        }
        // Audio rides the same way — one `"audio"` part per clip, matched in
        // order. The vendor decode path is AVFoundation-over-URL (container
        // parsing + 16 kHz mono resample in one place), so each clip spools
        // to a content-addressed temp file for exactly the duration of
        // `prepare`.
        let requestAudios = conversation.audios
        let audioSpool = try AudioClipSpool(clips: requestAudios)
        defer { audioSpool.cleanUp() }
        let fullInput = try await session.prepare(
            UserInput(
                messages: conversation.promptMessages,
                images: userInputImages,
                audios: audioSpool.userInputAudios,
                tools: canonicalTools,
                additionalContext: renderContext.additionalContext()
            )
        )
        // Sequence length is always the LAST dim. For LLM models tokens are
        // 1D [seq], for VLM models (ParoQuant Qwen35) they are 2D [batch, seq].
        let fullTokenCount = fullInput.text.tokens.dim(-1)
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

        // 3b. Build the request's **Cache Key Space** from the prepared
        // tokens, the conversation's media, and the family's media keying.
        // Identity (and free) for text-only requests. A construction
        // failure degrades the whole request to an **Unkeyed Completion**
        // — served normally, zero cache participation.
        let keySpace: CacheKeySpace
        switch CacheKeySpace.make(
            preparedTokens: fullTokens,
            imageDigests: requestImages.map(\.digest),
            imageGrids: (fullInput.image?.frames ?? []).map { frame in
                let (t, h, w) = frame.values
                return (t: t, height: h, width: w)
            },
            imageKeying: imageKeying,
            audioDigests: requestAudios.map(\.digest),
            audioKeying: audioKeying
        ) {
        case .success(let space):
            keySpace = space
        case .failure(let reason):
            return .unkeyed(
                fullInput: fullInput,
                fullTokens: fullTokens,
                partitionKey: partitionKey,
                reason: reason
            )
        }
        // Grid instrumentation (ADR-0007 phase 2): the processed image grid
        // is the ground truth for the M-RoPE span and the pad-run length the
        // Cache Key Path expands. Logging it cheaply catches the deferred
        // "one screenshot → ~43,500 pad tokens" anomaly with real numbers if
        // it recurs — the chunked continuation already makes such an image
        // non-fatal, so this is observe-only, not a gate.
        if let frames = fullInput.image?.frames, !frames.isEmpty {
            let merge = imageKeying?.spatialMergeSize ?? 1
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
        // Audio run instrumentation: the run length is the clip's soft-token
        // count (40 ms per token) — the ground truth the audio prefill cost
        // scales with. Observe-only, like the grid log above.
        for (index, entry) in keySpace.audioTable.enumerated() {
            Log.server.debug(
                "audio run #\(index): tokens=\(entry.runLength) "
                    + "digest=\(entry.digest.hexString.prefix(8))"
            )
        }

        return .keyed(
            Keyed(
                fullInput: fullInput,
                fullTokens: fullTokens,
                fullTokenCount: fullTokenCount,
                tokenNDim: tokenNDim,
                partitionKey: partitionKey,
                keySpace: keySpace,
                seedsPositionAnchor: imageKeying?.anchorsWarmContinuations ?? false
            ))
    }
}
