import Foundation

/// The per-family image-placeholder identity — a **Model Identity** load-time
/// fact (CONTEXT.md). `nil` at the Model Identity level means the family is
/// not recognized for image keying; an image-bearing request then degrades to
/// an **Unkeyed Completion** through the Cache Key Space construction guard.
///
/// Covers single-pad-run families (Qwen-VL style: one pad token the processor
/// expands in place, framed by ordinary text tokens). Token-pair families
/// (framed image blocks à la Idefics3) are deliberately unrepresentable —
/// they bail at construction (PRD #72, out of scope).
nonisolated struct ImagePlaceholderIdentity: Hashable, Sendable {
    /// The token id whose runs in the *prepared* sequence are one image each,
    /// and whose single occurrences in a chat-template *render* are one image
    /// each (the processor expands the single pad into the run).
    let imagePadTokenId: Int
}

/// The **Cache Key Space** (CONTEXT.md → Image-aware prefix caching): the
/// per-request authority over the two token spaces, built once after prepare.
/// It owns the image table, produces the **Cache Key Path**, translates
/// render-space token sequences into key space, and reconstructs the
/// **Position Anchor** for warm restores. Every token path that touches the
/// radix tree passes through it; for text-only requests every operation is
/// the identity.
nonisolated struct CacheKeySpace: Sendable {

    /// One image of the request, in conversation order (which equals prompt
    /// order — the message renderer emits messages, and images within them,
    /// in sequence).
    struct RequestImage: Hashable, Sendable {
        let digest: ImageDigest
        /// The image's M-RoPE position span — how many position units its run
        /// occupies (vs `runLength` token/KV slots). See `positionSpan(...)`.
        let positionSpan: Int
    }

    struct ImageTableEntry: Hashable, Sendable {
        let digest: ImageDigest
        /// The placeholder run in prepared/key space. Key index == KV offset,
        /// so this range is valid in both.
        let runRange: Range<Int>
        let positionSpan: Int

        var runLength: Int { runRange.count }
    }

    /// One audio clip's run in prepared/key space. Audio positions are always
    /// sequential (span == run length, anchor delta 0), so the run itself is
    /// the whole geometry.
    struct AudioTableEntry: Hashable, Sendable {
        let digest: AudioDigest
        let runRange: Range<Int>

        var runLength: Int { runRange.count }
    }

    /// Whole-request degradation: no valid Cache Key Path can be built — the
    /// completion is served **Unkeyed** (zero cache participation, never a
    /// route bounce). Raw values are the wire strings for logs.
    enum UnkeyedReason: String, Error, Sendable {
        /// The loaded family has no image-placeholder identity (non-VLM, or a
        /// VLM family the app doesn't recognize) but the request has images.
        case unrecognizedPlaceholderFamily = "unrecognized-placeholder-family"
        /// Placeholder runs found in the prepared tokens ≠ images in the
        /// conversation — keying would mis-attribute content.
        case placeholderRunCountMismatch = "placeholder-run-count-mismatch"
        /// Processed image grids returned by prepare ≠ images in the
        /// conversation — the Position Anchor geometry cannot be attributed.
        case imageGridCountMismatch = "image-grid-count-mismatch"
        /// The loaded family has no audio-placeholder identity (not
        /// **Audio-capable**, or an audio family the app doesn't recognize)
        /// but the request has audio clips.
        case unrecognizedAudioPlaceholderFamily = "unrecognized-audio-placeholder-family"
        /// Audio placeholder runs found in the prepared tokens ≠ clips in the
        /// conversation — keying would mis-attribute content.
        case audioPlaceholderRunCountMismatch = "audio-placeholder-run-count-mismatch"
    }

    /// Feature-level degradation: one render could not be translated; only
    /// the consuming feature skips (LeafSkipReason pattern), the request and
    /// its lookup keep working.
    enum TranslationFailure: Error, Equatable, Sendable {
        case placeholderOccurrencesExceedImages(occurrences: Int, images: Int)
        case audioPlaceholderOccurrencesExceedClips(occurrences: Int, clips: Int)
    }

    /// The Cache Key Path — prepared tokens with each image's and clip's
    /// placeholder run replaced, length-preserving, by its digest's
    /// pseudo-token expansion.
    let keyPath: [Int]
    /// Per image, in prompt order.
    let imageTable: [ImageTableEntry]
    /// Per audio clip, in prompt order.
    let audioTable: [AudioTableEntry]
    private let placeholderIdentity: ImagePlaceholderIdentity?
    private let audioPadTokenId: Int?

    /// True for text-only requests: every operation is the identity.
    var isIdentity: Bool { imageTable.isEmpty && audioTable.isEmpty }

    /// The identity space over a known key path — what `make` produces for a
    /// text-only request. Translation returns inputs unchanged, anchors are
    /// zero. Also the natural stand-in for call sites and tests that predate
    /// image keying.
    static func identity(keyPath: [Int] = []) -> CacheKeySpace {
        CacheKeySpace(
            keyPath: keyPath, imageTable: [], audioTable: [],
            placeholderIdentity: nil, audioPadTokenId: nil
        )
    }

    // MARK: - Construction

    /// Build the request's key space from the loaded family's **Model
    /// Identity** media keying and the vendor-prepared image grids — the
    /// production entry point. Owns the whole construction guard: family
    /// recognition, grid attribution, and the position-span geometry, so the
    /// invariant "a key space exists iff prepared tokens, media, grids, and
    /// family identity all agree" lives in one place.
    ///
    /// Span geometry is the family's `PositionSpanRule`: the M-RoPE grid
    /// formula for Qwen-VL (grids required), run length for sequential
    /// families (Gemma 4 unified — grids ignored for spans, so a grid-less
    /// prepare cannot degrade the request). Audio spans are always run
    /// length.
    static func make(
        preparedTokens: [Int],
        imageDigests: [ImageDigest],
        imageGrids: [(t: Int, height: Int, width: Int)],
        imageKeying: ModelIdentity.ImageKeying?,
        audioDigests: [AudioDigest] = [],
        audioKeying: ModelIdentity.AudioKeying? = nil
    ) -> Result<CacheKeySpace, UnkeyedReason> {
        guard !imageDigests.isEmpty || !audioDigests.isEmpty else {
            return .success(.identity(keyPath: preparedTokens))
        }

        var keyPath = preparedTokens
        var imageTable: [ImageTableEntry] = []
        var audioTable: [AudioTableEntry] = []

        if !imageDigests.isEmpty {
            guard let imageKeying else {
                return .failure(.unrecognizedPlaceholderFamily)
            }
            let runs = placeholderRuns(
                in: preparedTokens, padTokenId: imageKeying.imagePadTokenId)
            guard runs.count == imageDigests.count else {
                return .failure(.placeholderRunCountMismatch)
            }
            let spans: [Int]
            switch imageKeying.spanRule {
            case .mropeGrid(let spatialMergeSize):
                guard imageGrids.count == imageDigests.count else {
                    return .failure(.imageGridCountMismatch)
                }
                spans = imageGrids.map { grid in
                    positionSpan(
                        t: grid.t, height: grid.height, width: grid.width,
                        spatialMergeSize: spatialMergeSize
                    )
                }
            case .sequential:
                spans = runs.map(\.count)
            }
            imageTable.reserveCapacity(imageDigests.count)
            for ((run, digest), span) in zip(zip(runs, imageDigests), spans) {
                keyPath.replaceSubrange(
                    run, with: ImagePseudoToken.expansion(digest: digest, runLength: run.count)
                )
                imageTable.append(
                    ImageTableEntry(digest: digest, runRange: run, positionSpan: span))
            }
        }

        if !audioDigests.isEmpty {
            guard let audioKeying else {
                return .failure(.unrecognizedAudioPlaceholderFamily)
            }
            let runs = placeholderRuns(
                in: preparedTokens, padTokenId: audioKeying.audioPadTokenId)
            guard runs.count == audioDigests.count else {
                return .failure(.audioPlaceholderRunCountMismatch)
            }
            audioTable.reserveCapacity(audioDigests.count)
            for (run, digest) in zip(runs, audioDigests) {
                keyPath.replaceSubrange(
                    run, with: AudioPseudoToken.expansion(digest: digest, runLength: run.count)
                )
                audioTable.append(AudioTableEntry(digest: digest, runRange: run))
            }
        }

        return .success(
            CacheKeySpace(
                keyPath: keyPath,
                imageTable: imageTable,
                audioTable: audioTable,
                placeholderIdentity: imageKeying.map {
                    ImagePlaceholderIdentity(imagePadTokenId: $0.imagePadTokenId)
                },
                audioPadTokenId: audioKeying?.audioPadTokenId
            ))
    }

    /// Build the request's key space from the prepared prompt tokens, the
    /// conversation's images, and the family's placeholder identity — the
    /// image-only internal shape (pre-audio call sites and tests).
    ///
    /// Text-only requests (no images) construct an identity space regardless
    /// of `placeholderIdentity` — image keying never taxes the text path.
    static func make(
        preparedTokens: [Int],
        images: [RequestImage],
        placeholderIdentity: ImagePlaceholderIdentity?
    ) -> Result<CacheKeySpace, UnkeyedReason> {
        guard !images.isEmpty else {
            return .success(
                CacheKeySpace(
                    keyPath: preparedTokens, imageTable: [], audioTable: [],
                    placeholderIdentity: placeholderIdentity, audioPadTokenId: nil
                ))
        }
        guard let identity = placeholderIdentity else {
            return .failure(.unrecognizedPlaceholderFamily)
        }

        let runs = placeholderRuns(in: preparedTokens, padTokenId: identity.imagePadTokenId)
        guard runs.count == images.count else {
            return .failure(.placeholderRunCountMismatch)
        }

        var keyPath = preparedTokens
        var table: [ImageTableEntry] = []
        table.reserveCapacity(images.count)
        for (run, image) in zip(runs, images) {
            keyPath.replaceSubrange(
                run, with: ImagePseudoToken.expansion(digest: image.digest, runLength: run.count)
            )
            table.append(
                ImageTableEntry(
                    digest: image.digest, runRange: run, positionSpan: image.positionSpan
                ))
        }
        return .success(
            CacheKeySpace(
                keyPath: keyPath, imageTable: table, audioTable: [],
                placeholderIdentity: placeholderIdentity, audioPadTokenId: nil
            ))
    }

    // MARK: - Translation (render space → key space)

    /// Translate a chat-template render — *unexpanded* space, one pad token
    /// per image and one per audio clip — into key space. Positional
    /// arithmetic per modality: the i-th image-pad occurrence maps to image
    /// i, the i-th audio-pad occurrence to clip i. Prefix renders (fewer
    /// media than the request) are valid; framing tokens are ordinary tokens
    /// in both spaces and pass through unchanged. Identity for text-only
    /// requests.
    func translate(renderTokens: [Int]) -> Result<[Int], TranslationFailure> {
        guard !isIdentity else {
            return .success(renderTokens)
        }
        let imagePadTokenId = placeholderIdentity?.imagePadTokenId

        var translated: [Int] = []
        translated.reserveCapacity(
            renderTokens.count
                + imageTable.reduce(0) { $0 + $1.runLength - 1 }
                + audioTable.reduce(0) { $0 + $1.runLength - 1 }
        )
        var imageIndex = 0
        var audioIndex = 0
        for token in renderTokens {
            if token == imagePadTokenId {
                guard imageIndex < imageTable.count else {
                    return .failure(
                        .placeholderOccurrencesExceedImages(
                            occurrences: imageIndex + 1, images: imageTable.count
                        ))
                }
                // The key path already holds this image's expansion — splice
                // the run instead of re-deriving every pseudo-token.
                translated.append(contentsOf: keyPath[imageTable[imageIndex].runRange])
                imageIndex += 1
            } else if token == audioPadTokenId {
                guard audioIndex < audioTable.count else {
                    return .failure(
                        .audioPlaceholderOccurrencesExceedClips(
                            occurrences: audioIndex + 1, clips: audioTable.count
                        ))
                }
                translated.append(contentsOf: keyPath[audioTable[audioIndex].runRange])
                audioIndex += 1
            } else {
                translated.append(token)
            }
        }
        return .success(translated)
    }

    /// The key-space length of a render — `translate` for consumers that only
    /// need the boundary offset, without materializing the translated path.
    func translatedLength(renderTokens: [Int]) -> Result<Int, TranslationFailure> {
        guard !isIdentity else {
            return .success(renderTokens.count)
        }
        let imagePadTokenId = placeholderIdentity?.imagePadTokenId

        var length = 0
        var imageIndex = 0
        var audioIndex = 0
        for token in renderTokens {
            if token == imagePadTokenId {
                guard imageIndex < imageTable.count else {
                    return .failure(
                        .placeholderOccurrencesExceedImages(
                            occurrences: imageIndex + 1, images: imageTable.count
                        ))
                }
                length += imageTable[imageIndex].runLength
                imageIndex += 1
            } else if token == audioPadTokenId {
                guard audioIndex < audioTable.count else {
                    return .failure(
                        .audioPlaceholderOccurrencesExceedClips(
                            occurrences: audioIndex + 1, clips: audioTable.count
                        ))
                }
                length += audioTable[audioIndex].runLength
                audioIndex += 1
            } else {
                length += 1
            }
        }
        return .success(length)
    }

    // MARK: - Position Anchor

    /// The rope-delta component of the **Position Anchor** for a cache warmed
    /// up to `offset`: Σ over images fully inside [0, offset) of
    /// (positionSpan − runLength). Zero for image-free prefixes, on the
    /// identity space, and for sequential-rule media (span == run length by
    /// construction — audio always, Gemma images always). `nil` when `offset`
    /// splits any media run — no admitted snapshot can sit there, so a
    /// mid-run offset means the caller is holding a corrupt boundary and
    /// must not restore.
    func positionAnchorDelta(upTo offset: Int) -> Int? {
        var delta = 0
        for entry in imageTable {
            if entry.runRange.upperBound <= offset {
                delta += entry.positionSpan - entry.runLength
            } else if entry.runRange.lowerBound < offset {
                return nil
            }
        }
        for entry in audioTable where entry.runRange.lowerBound < offset {
            // Sequential spans contribute 0 delta; only the split check bites.
            if entry.runRange.upperBound > offset {
                return nil
            }
        }
        return delta
    }

    /// The smallest restore offset whose remainder is media-free. Below it
    /// the remainder contains an image or audio run; phase 1 forced such hits
    /// cold, phase 2 (ADR-0007) continues warm *through* the media instead.
    /// It remains the boundary between the vendor-continued media span and
    /// the app-chunked text tail, and the cold fallback's media-prefix end.
    var minimumWarmOffset: Int {
        max(
            imageTable.last?.runRange.upperBound ?? 0,
            audioTable.last?.runRange.upperBound ?? 0
        )
    }

    /// The indices (into the request's image list, prompt order) of the images
    /// whose runs fall at or beyond `offset` — exactly the images a warm
    /// continuation from `offset` must re-run through the vision tower (those
    /// fully before `offset` are already in the restored cache). `nil` when
    /// `offset` splits a run (no valid restore there, mirroring
    /// `positionAnchorDelta`). An empty range means every image precedes
    /// `offset` (an image-free remainder — the ordinary text restore).
    func remainderImageIndices(from offset: Int) -> Range<Int>? {
        for (index, entry) in imageTable.enumerated() {
            if entry.runRange.upperBound <= offset {
                continue  // fully cached before the restore offset
            }
            if entry.runRange.lowerBound < offset {
                return nil  // offset splits this run
            }
            return index..<imageTable.count  // first image at or beyond offset
        }
        return imageTable.count..<imageTable.count  // all images precede offset
    }

    /// The audio sibling of `remainderImageIndices`: the clips whose runs
    /// fall at or beyond `offset` — exactly the clips a warm continuation
    /// from `offset` must re-feed (those fully before it are already in the
    /// restored cache). Same nil-on-split and empty-range semantics.
    func remainderAudioIndices(from offset: Int) -> Range<Int>? {
        for (index, entry) in audioTable.enumerated() {
            if entry.runRange.upperBound <= offset {
                continue  // fully cached before the restore offset
            }
            if entry.runRange.lowerBound < offset {
                return nil  // offset splits this run
            }
            return index..<audioTable.count  // first clip at or beyond offset
        }
        return audioTable.count..<audioTable.count  // all clips precede offset
    }

    // MARK: - Geometry

    /// M-RoPE position span of one image from its processed patch grid
    /// (`THW`, pre-merge) — max(t, h/m, w/m) for spatial merge size m. Pinned
    /// against the model's harvested rope delta by the VLM spike harness
    /// (`prepareStateCarriesRopeDelta`).
    static func positionSpan(t: Int, height: Int, width: Int, spatialMergeSize: Int) -> Int {
        max(t, max(height / spatialMergeSize, width / spatialMergeSize))
    }

    // MARK: - Internals

    private static func placeholderRuns(in tokens: [Int], padTokenId: Int) -> [Range<Int>] {
        var runs: [Range<Int>] = []
        var runStart: Int?
        for (index, token) in tokens.enumerated() {
            if token == padTokenId {
                if runStart == nil { runStart = index }
            } else if let start = runStart {
                runs.append(start..<index)
                runStart = nil
            }
        }
        if let start = runStart {
            runs.append(start..<tokens.count)
        }
        return runs
    }
}
