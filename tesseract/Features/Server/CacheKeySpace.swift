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

        init(digest: ImageDigest, positionSpan: Int) {
            self.digest = digest
            self.positionSpan = positionSpan
        }
    }

    struct ImageTableEntry: Hashable, Sendable {
        let digest: ImageDigest
        /// The placeholder run in prepared/key space. Key index == KV offset,
        /// so this range is valid in both.
        let runRange: Range<Int>
        let positionSpan: Int

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
    }

    /// Feature-level degradation: one render could not be translated; only
    /// the consuming feature skips (LeafSkipReason pattern), the request and
    /// its lookup keep working.
    enum TranslationFailure: Error, Equatable, Sendable {
        case placeholderOccurrencesExceedImages(occurrences: Int, images: Int)
    }

    /// The Cache Key Path — prepared tokens with each image's placeholder run
    /// replaced, length-preserving, by its digest's pseudo-token expansion.
    let keyPath: [Int]
    /// Per image, in prompt order.
    let imageTable: [ImageTableEntry]
    private let placeholderIdentity: ImagePlaceholderIdentity?

    /// True for text-only requests: every operation is the identity.
    var isIdentity: Bool { imageTable.isEmpty }

    /// The identity space over a known key path — what `make` produces for a
    /// text-only request. Translation returns inputs unchanged, anchors are
    /// zero. Also the natural stand-in for call sites and tests that predate
    /// image keying.
    static func identity(keyPath: [Int] = []) -> CacheKeySpace {
        CacheKeySpace(keyPath: keyPath, imageTable: [], placeholderIdentity: nil)
    }

    // MARK: - Construction

    /// Build the request's key space from the loaded family's **Model
    /// Identity** image keying and the vendor-prepared image grids — the
    /// production entry point. Owns the whole construction guard: family
    /// recognition, grid attribution, and the M-RoPE span geometry, so the
    /// invariant "a key space exists iff prepared tokens, images, grids, and
    /// family identity all agree" lives in one place.
    static func make(
        preparedTokens: [Int],
        imageDigests: [ImageDigest],
        imageGrids: [(t: Int, height: Int, width: Int)],
        imageKeying: ModelIdentity.ImageKeying?
    ) -> Result<CacheKeySpace, UnkeyedReason> {
        guard !imageDigests.isEmpty else {
            return .success(.identity(keyPath: preparedTokens))
        }
        guard let imageKeying else {
            return .failure(.unrecognizedPlaceholderFamily)
        }
        guard imageGrids.count == imageDigests.count else {
            return .failure(.imageGridCountMismatch)
        }
        return make(
            preparedTokens: preparedTokens,
            images: zip(imageDigests, imageGrids).map { digest, grid in
                RequestImage(
                    digest: digest,
                    positionSpan: positionSpan(
                        t: grid.t, height: grid.height, width: grid.width,
                        spatialMergeSize: imageKeying.spatialMergeSize
                    )
                )
            },
            placeholderIdentity: ImagePlaceholderIdentity(
                imagePadTokenId: imageKeying.imagePadTokenId
            )
        )
    }

    /// Build the request's key space from the prepared prompt tokens, the
    /// conversation's images, and the family's placeholder identity.
    ///
    /// Text-only requests (no images) construct an identity space regardless
    /// of `placeholderIdentity` — image keying never taxes the text path.
    static func make(
        preparedTokens: [Int],
        images: [RequestImage],
        placeholderIdentity: ImagePlaceholderIdentity?
    ) -> Result<CacheKeySpace, UnkeyedReason> {
        guard !images.isEmpty else {
            return .success(CacheKeySpace(
                keyPath: preparedTokens, imageTable: [], placeholderIdentity: placeholderIdentity
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
            table.append(ImageTableEntry(
                digest: image.digest, runRange: run, positionSpan: image.positionSpan
            ))
        }
        return .success(CacheKeySpace(
            keyPath: keyPath, imageTable: table, placeholderIdentity: placeholderIdentity
        ))
    }

    // MARK: - Translation (render space → key space)

    /// Translate a chat-template render — *unexpanded* space, one pad token
    /// per image — into key space. Positional arithmetic: the i-th pad
    /// occurrence maps to image i. Prefix renders (fewer images than the
    /// request) are valid; framing tokens are ordinary tokens in both spaces
    /// and pass through unchanged. Identity for text-only requests.
    func translate(renderTokens: [Int]) -> Result<[Int], TranslationFailure> {
        guard !imageTable.isEmpty, let identity = placeholderIdentity else {
            return .success(renderTokens)
        }

        var translated: [Int] = []
        translated.reserveCapacity(
            renderTokens.count + imageTable.reduce(0) { $0 + $1.runLength - 1 }
        )
        var imageIndex = 0
        for token in renderTokens {
            guard token == identity.imagePadTokenId else {
                translated.append(token)
                continue
            }
            guard imageIndex < imageTable.count else {
                return .failure(.placeholderOccurrencesExceedImages(
                    occurrences: imageIndex + 1, images: imageTable.count
                ))
            }
            // The key path already holds this image's expansion — splice the
            // run instead of re-deriving every pseudo-token.
            translated.append(contentsOf: keyPath[imageTable[imageIndex].runRange])
            imageIndex += 1
        }
        return .success(translated)
    }

    /// The key-space length of a render — `translate` for consumers that only
    /// need the boundary offset, without materializing the translated path.
    func translatedLength(renderTokens: [Int]) -> Result<Int, TranslationFailure> {
        guard !imageTable.isEmpty, let identity = placeholderIdentity else {
            return .success(renderTokens.count)
        }

        var length = 0
        var imageIndex = 0
        for token in renderTokens {
            guard token == identity.imagePadTokenId else {
                length += 1
                continue
            }
            guard imageIndex < imageTable.count else {
                return .failure(.placeholderOccurrencesExceedImages(
                    occurrences: imageIndex + 1, images: imageTable.count
                ))
            }
            length += imageTable[imageIndex].runLength
            imageIndex += 1
        }
        return .success(length)
    }

    // MARK: - Position Anchor

    /// The rope-delta component of the **Position Anchor** for a cache warmed
    /// up to `offset`: Σ over images fully inside [0, offset) of
    /// (positionSpan − runLength). Zero for image-free prefixes and on the
    /// identity space. `nil` when `offset` splits a placeholder run — no
    /// admitted snapshot can sit there, so a mid-run offset means the caller
    /// is holding a corrupt boundary and must not restore.
    func positionAnchorDelta(upTo offset: Int) -> Int? {
        var delta = 0
        for entry in imageTable {
            if entry.runRange.upperBound <= offset {
                delta += entry.positionSpan - entry.runLength
            } else if entry.runRange.lowerBound < offset {
                return nil
            }
        }
        return delta
    }

    /// The smallest restore offset whose remainder is image-free. Hits below
    /// this would put an image run in the remainder, which cannot be forwarded
    /// warm (ADR-0007 spike results) — the request serves cold instead.
    var minimumWarmOffset: Int {
        imageTable.last?.runRange.upperBound ?? 0
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
                runs.append(start ..< index)
                runStart = nil
            }
        }
        if let start = runStart {
            runs.append(start ..< tokens.count)
        }
        return runs
    }
}
