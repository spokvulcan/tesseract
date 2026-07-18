import Foundation
import Testing
@testable import Tesseract_Agent

/// The **Cache Key Space** seam (PRD #72): construction — (prepared tokens,
/// placeholder identity, image digests) → key path + image table *or* typed
/// unkeyed reason; translation — render-space tokens → key-space tokens *or*
/// typed skip; Position Anchor reconstruction. Pure, no model files —
/// mirrors the Completion Route and Prefill Planner suites.
struct CacheKeySpaceTests {

    // Qwen3.5's real ids, used as plausible constants — nothing here loads a model.
    private static let pad = 248_056
    private static let visionStart = 248_053
    private static let visionEnd = 248_054
    private static let identity = ImagePlaceholderIdentity(imagePadTokenId: pad)

    private static func digest(_ seed: String) -> ImageDigest {
        ImageDigest(imageBytes: Data(seed.utf8))
    }

    /// [text…] <vision_start> pad×runLength <vision_end> [text…]
    private static func prompt(runLengths: [Int], textChunk: Int = 4) -> [Int] {
        var tokens: [Int] = Array(1...textChunk)
        for (i, run) in runLengths.enumerated() {
            tokens.append(visionStart)
            tokens.append(contentsOf: Array(repeating: pad, count: run))
            tokens.append(visionEnd)
            tokens.append(contentsOf: Array(100 + i * 10..<100 + i * 10 + textChunk))
        }
        return tokens
    }

    // MARK: - Frozen expansion (golden values)

    /// **FROZEN FUNCTION GATE.** These values were computed independently of
    /// the Swift implementation and persist inside SSD admission paths across
    /// restarts. If this test fails, the change invalidates every
    /// image-bearing snapshot on disk — revert the function, don't update the
    /// goldens.
    @Test func pseudoTokenExpansionMatchesGoldenValues() throws {
        let digestA = try #require(ImageDigest(rawDigest: Data(0..<32)))
        let digestB = Self.digest("tesseract-image-a")
        let digestC = Self.digest("tesseract-image-b")

        // Digest bytes themselves are part of the frozen surface.
        #expect(
            digestB.hexString == "8a7136e7652bd150764b2cf0f3144e7014fbb90c2691dbef0b9fa20d97e254b6")
        #expect(
            digestC.hexString == "467e2ab3b7fa4141ccfe3fb3ace77350e1fc2b5c7f9a0e72a4700d9d7923a512")

        let golden: [(ImageDigest, Int, Int)] = [
            (digestA, 0, -3_964_090_289_384_830_885),
            (digestA, 1, -8_439_210_388_648_119_943),
            (digestA, 2, -7_691_130_605_843_516_506),
            (digestA, 63, -8_247_712_167_531_799_670),
            (digestB, 0, -8_941_181_424_613_796_520),
            (digestB, 1, -5_435_402_744_602_417_580),
            (digestB, 2, -8_991_916_012_570_296_300),
            (digestB, 63, -6_471_186_341_809_028_035),
            (digestC, 0, -1_764_138_204_917_859_022),
            (digestC, 1, -7_934_356_138_635_441_875),
            (digestC, 2, -5_307_446_864_424_869_516),
            (digestC, 63, -6_756_194_863_526_349_352),
        ]
        for (digest, index, expected) in golden {
            #expect(ImagePseudoToken.value(digest: digest, index: index) == expected)
        }
    }

    @Test func pseudoTokensAreAlwaysNegative() {
        let digests = ["a", "b", "c", "d"].map(Self.digest)
        for digest in digests {
            for index in 0..<256 {
                #expect(ImagePseudoToken.value(digest: digest, index: index) < 0)
            }
        }
    }

    /// The exact bug class this feature exists to prevent: two different
    /// images with identical placeholder geometry must diverge at the first
    /// expanded position.
    @Test func differentImagesSameSizeDivergeAtFirstPosition() {
        let a = ImagePseudoToken.expansion(digest: Self.digest("image-a"), runLength: 64)
        let b = ImagePseudoToken.expansion(digest: Self.digest("image-b"), runLength: 64)
        #expect(a.count == b.count)
        #expect(a[0] != b[0])
    }

    // MARK: - Construction

    @Test func textOnlyConstructsIdentitySpace() throws {
        let tokens = [1, 2, 3, 4, 5]
        let space = try CacheKeySpace.make(
            preparedTokens: tokens, images: [], placeholderIdentity: Self.identity
        ).get()
        #expect(space.isIdentity)
        #expect(space.keyPath == tokens)
        #expect(space.minimumWarmOffset == 0)
        #expect(space.positionAnchorDelta(upTo: 3) == 0)
    }

    /// A text-only request on an unrecognized family must stay keyable —
    /// image keying never taxes the text path.
    @Test func textOnlyWithoutIdentityStillConstructs() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: [1, 2, 3], images: [], placeholderIdentity: nil
        ).get()
        #expect(space.isIdentity)
        #expect(space.keyPath == [1, 2, 3])
    }

    @Test func imagesWithoutIdentityAreUnkeyed() {
        let result = CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4]),
            images: [.init(digest: Self.digest("a"), positionSpan: 2)],
            placeholderIdentity: nil
        )
        #expect(result.failureReason == .unrecognizedPlaceholderFamily)
    }

    @Test func runCountMismatchIsUnkeyed() {
        // Two images claimed, one run present.
        let result = CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4]),
            images: [
                .init(digest: Self.digest("a"), positionSpan: 2),
                .init(digest: Self.digest("b"), positionSpan: 2),
            ],
            placeholderIdentity: Self.identity
        )
        #expect(result.failureReason == .placeholderRunCountMismatch)

        // One image claimed, no runs present (non-VLM container dropped them).
        let textResult = CacheKeySpace.make(
            preparedTokens: [1, 2, 3],
            images: [.init(digest: Self.digest("a"), positionSpan: 2)],
            placeholderIdentity: Self.identity
        )
        #expect(textResult.failureReason == .placeholderRunCountMismatch)
    }

    @Test func keyPathReplacesRunsLengthPreservingAndLeavesFraming() throws {
        let tokens = Self.prompt(runLengths: [3])
        let digest = Self.digest("a")
        let space = try CacheKeySpace.make(
            preparedTokens: tokens,
            images: [.init(digest: digest, positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()

        #expect(space.keyPath.count == tokens.count)
        let run = try #require(space.imageTable.first?.runRange)
        #expect(
            Array(space.keyPath[run]) == ImagePseudoToken.expansion(digest: digest, runLength: 3))
        // Everything outside the run — framing included — is untouched.
        for index in tokens.indices where !run.contains(index) {
            #expect(space.keyPath[index] == tokens[index])
        }
        #expect(space.keyPath[run.lowerBound - 1] == Self.visionStart)
        #expect(space.keyPath[run.upperBound] == Self.visionEnd)
    }

    @Test func multiImageOrderingMapsRunsToImagesInPromptOrder() throws {
        let tokens = Self.prompt(runLengths: [3, 5])
        let digestA = Self.digest("first")
        let digestB = Self.digest("second")
        let space = try CacheKeySpace.make(
            preparedTokens: tokens,
            images: [
                .init(digest: digestA, positionSpan: 2),
                .init(digest: digestB, positionSpan: 3),
            ],
            placeholderIdentity: Self.identity
        ).get()

        #expect(space.imageTable.count == 2)
        #expect(space.imageTable[0].digest == digestA)
        #expect(space.imageTable[0].runLength == 3)
        #expect(space.imageTable[1].digest == digestB)
        #expect(space.imageTable[1].runLength == 5)
        #expect(space.imageTable[0].runRange.upperBound <= space.imageTable[1].runRange.lowerBound)
    }

    /// Same prompt, different image bytes → key paths diverge inside the run.
    @Test func sameSizeDifferentImageProducesDifferentKeyPath() throws {
        let tokens = Self.prompt(runLengths: [4])
        let spaceA = try CacheKeySpace.make(
            preparedTokens: tokens,
            images: [.init(digest: Self.digest("image-a"), positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()
        let spaceB = try CacheKeySpace.make(
            preparedTokens: tokens,
            images: [.init(digest: Self.digest("image-b"), positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()
        #expect(spaceA.keyPath != spaceB.keyPath)
        #expect(spaceA.keyPath.count == spaceB.keyPath.count)
    }

    // MARK: - Translation

    @Test func translationIsIdentityForTextOnly() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: [1, 2, 3], images: [], placeholderIdentity: Self.identity
        ).get()
        #expect(try space.translate(renderTokens: [9, 8, 7]).get() == [9, 8, 7])
    }

    /// Render space carries one pad per image; translation expands it to the
    /// full run, leaving framing tokens (ordinary tokens) untouched.
    @Test func translationExpandsPadsAndKeepsFraming() throws {
        let digest = Self.digest("a")
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [3]),
            images: [.init(digest: digest, positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()

        let render = [1, 2, Self.visionStart, Self.pad, Self.visionEnd, 100]
        let translated = try space.translate(renderTokens: render).get()
        #expect(
            translated == [1, 2, Self.visionStart]
                + ImagePseudoToken.expansion(digest: digest, runLength: 3)
                + [Self.visionEnd, 100])
    }

    /// A prefix render contains only the leading images — occurrence i maps
    /// to image i.
    @Test func prefixRenderTranslatesLeadingImagesOnly() throws {
        let digestA = Self.digest("first")
        let digestB = Self.digest("second")
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [2, 3]),
            images: [
                .init(digest: digestA, positionSpan: 2),
                .init(digest: digestB, positionSpan: 2),
            ],
            placeholderIdentity: Self.identity
        ).get()

        let prefixRender = [1, Self.visionStart, Self.pad, Self.visionEnd]
        let translated = try space.translate(renderTokens: prefixRender).get()
        #expect(
            translated == [1, Self.visionStart]
                + ImagePseudoToken.expansion(digest: digestA, runLength: 2)
                + [Self.visionEnd])
    }

    @Test func translationFailsTypedWhenRenderHasMoreImagesThanRequest() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [2]),
            images: [.init(digest: Self.digest("a"), positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()

        let render = [Self.pad, 5, Self.pad]
        switch space.translate(renderTokens: render) {
        case .success:
            Issue.record("expected typed translation failure")
        case .failure(let failure):
            #expect(failure == .placeholderOccurrencesExceedImages(occurrences: 2, images: 1))
        }
    }

    @Test func translatedLengthMatchesTranslateOnBothChannels() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [2, 5]),
            images: [
                .init(digest: Self.digest("a"), positionSpan: 2),
                .init(digest: Self.digest("b"), positionSpan: 3),
            ],
            placeholderIdentity: Self.identity
        ).get()

        let render = [1, Self.visionStart, Self.pad, Self.visionEnd, 7, Self.pad, 9]
        #expect(
            try space.translatedLength(renderTokens: render).get()
                == (try space.translate(renderTokens: render).get().count))

        let overflow = [Self.pad, 5, Self.pad, 6, Self.pad]
        switch space.translatedLength(renderTokens: overflow) {
        case .success:
            Issue.record("expected typed translation failure")
        case .failure(let failure):
            #expect(failure == .placeholderOccurrencesExceedImages(occurrences: 3, images: 2))
        }

        let identitySpace = CacheKeySpace.identity(keyPath: [1, 2, 3])
        #expect(try identitySpace.translatedLength(renderTokens: [9, 8]).get() == 2)
    }

    // MARK: - Production construction guard (Model Identity + prepared grids)

    @Test func gridConstructionBuildsSpansFromImageKeying() throws {
        let keying = ModelIdentity.ImageKeying(imagePadTokenId: Self.pad, spatialMergeSize: 2)
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [64]),
            imageDigests: [Self.digest("a")],
            imageGrids: [(t: 1, height: 16, width: 16)],
            imageKeying: keying
        ).get()
        // 256×256 spike geometry: span 8 over a 64-token run.
        #expect(space.imageTable[0].positionSpan == 8)
        #expect(space.minimumWarmOffset == space.imageTable[0].runRange.upperBound)
    }

    @Test func gridCountMismatchIsUnkeyed() {
        let keying = ModelIdentity.ImageKeying(imagePadTokenId: Self.pad, spatialMergeSize: 2)
        let result = CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4]),
            imageDigests: [Self.digest("a")],
            imageGrids: [],
            imageKeying: keying
        )
        #expect(result.failureReason == .imageGridCountMismatch)
    }

    @Test func imagesWithoutImageKeyingAreUnkeyedViaGridConstruction() {
        let result = CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4]),
            imageDigests: [Self.digest("a")],
            imageGrids: [(t: 1, height: 4, width: 4)],
            imageKeying: nil
        )
        #expect(result.failureReason == .unrecognizedPlaceholderFamily)
    }

    @Test func noImagesViaGridConstructionIsIdentity() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: [1, 2, 3],
            imageDigests: [],
            imageGrids: [],
            imageKeying: nil
        ).get()
        #expect(space.isIdentity)
        #expect(space.keyPath == [1, 2, 3])
    }

    // MARK: - Position Anchor

    @Test func anchorDeltaSumsImagesFullyBeforeOffset() throws {
        let tokens = Self.prompt(runLengths: [4, 6])  // spans below: 2 and 3
        let space = try CacheKeySpace.make(
            preparedTokens: tokens,
            images: [
                .init(digest: Self.digest("a"), positionSpan: 2),
                .init(digest: Self.digest("b"), positionSpan: 3),
            ],
            placeholderIdentity: Self.identity
        ).get()
        let run0 = space.imageTable[0].runRange
        let run1 = space.imageTable[1].runRange

        #expect(space.positionAnchorDelta(upTo: run0.lowerBound) == 0)
        #expect(space.positionAnchorDelta(upTo: run0.upperBound) == 2 - 4)
        #expect(space.positionAnchorDelta(upTo: run1.lowerBound) == 2 - 4)
        #expect(space.positionAnchorDelta(upTo: run1.upperBound) == (2 - 4) + (3 - 6))
        #expect(space.positionAnchorDelta(upTo: tokens.count) == (2 - 4) + (3 - 6))
    }

    @Test func anchorDeltaIsNilMidRun() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4]),
            images: [.init(digest: Self.digest("a"), positionSpan: 2)],
            placeholderIdentity: Self.identity
        ).get()
        let run = space.imageTable[0].runRange
        #expect(space.positionAnchorDelta(upTo: run.lowerBound + 1) == nil)
        #expect(space.positionAnchorDelta(upTo: run.upperBound - 1) == nil)
    }

    @Test func minimumWarmOffsetIsEndOfLastRun() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4, 6]),
            images: [
                .init(digest: Self.digest("a"), positionSpan: 2),
                .init(digest: Self.digest("b"), positionSpan: 3),
            ],
            placeholderIdentity: Self.identity
        ).get()
        #expect(space.minimumWarmOffset == space.imageTable[1].runRange.upperBound)
    }

    /// The continuation must receive exactly the images whose runs fall at or
    /// beyond the restore offset (ADR-0007 phase 2) — those before it are
    /// already in the restored cache.
    @Test func remainderImageIndicesSelectsImagesAtOrBeyondOffset() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.prompt(runLengths: [4, 6]),
            images: [
                .init(digest: Self.digest("a"), positionSpan: 2),
                .init(digest: Self.digest("b"), positionSpan: 3),
            ],
            placeholderIdentity: Self.identity
        ).get()
        let run0 = space.imageTable[0].runRange
        let run1 = space.imageTable[1].runRange

        // Below / at the first run: both images are in the remainder.
        #expect(space.remainderImageIndices(from: 0) == 0..<2)
        #expect(space.remainderImageIndices(from: run0.lowerBound) == 0..<2)
        // At the boundary after the first image: only the second.
        #expect(space.remainderImageIndices(from: run0.upperBound) == 1..<2)
        #expect(space.remainderImageIndices(from: run1.lowerBound) == 1..<2)
        // Past the last image: empty (an image-free remainder).
        #expect(space.remainderImageIndices(from: run1.upperBound) == 2..<2)
        // Splitting a run: nil — no valid restore there.
        #expect(space.remainderImageIndices(from: run0.lowerBound + 1) == nil)
        #expect(space.remainderImageIndices(from: run1.upperBound - 1) == nil)
    }

    @Test func remainderImageIndicesIsEmptyForTextOnly() {
        let space = CacheKeySpace.identity(keyPath: [1, 2, 3])
        #expect(space.remainderImageIndices(from: 0) == 0..<0)
    }

    /// Pinned against the model's harvested rope delta by the VLM spike
    /// harness: a 256×256 image → 16×16 patch grid, merge 2 → span 8,
    /// run 64, delta −56.
    @Test func positionSpanMatchesSpikeGeometry() {
        let span = CacheKeySpace.positionSpan(t: 1, height: 16, width: 16, spatialMergeSize: 2)
        #expect(span == 8)
        #expect(span - (1 * 16 * 16) / (2 * 2) == -56)
    }

    // MARK: - Audio keying + the sequential span rule (Gemma 4 unified)

    // Gemma 4 unified's real ids — nothing here loads a model.
    private static let gemmaImagePad = 258_880
    private static let gemmaAudioPad = 258_881
    private static let gemmaImageKeying = ModelIdentity.ImageKeying(
        imagePadTokenId: gemmaImagePad, spanRule: .sequential)
    private static let gemmaAudioKeying = ModelIdentity.AudioKeying(
        audioPadTokenId: gemmaAudioPad)

    private static func audioDigest(_ seed: String) -> AudioDigest {
        AudioDigest(audioBytes: Data(seed.utf8))
    }

    /// `[text…] boi image×n eoi boa audio×m eoa [text…]` — the unified
    /// processor's prepared shape: media runs framed by ordinary tokens.
    private static func gemmaPrompt(imageRuns: [Int], audioRuns: [Int]) -> [Int] {
        var tokens: [Int] = [1, 2, 3, 4]
        for run in imageRuns {
            tokens.append(255_999)  // boi
            tokens.append(contentsOf: Array(repeating: gemmaImagePad, count: run))
            tokens.append(258_882)  // eoi
            tokens.append(contentsOf: [10, 11])
        }
        for run in audioRuns {
            tokens.append(256_000)  // boa
            tokens.append(contentsOf: Array(repeating: gemmaAudioPad, count: run))
            tokens.append(258_883)  // eoa
            tokens.append(contentsOf: [20, 21])
        }
        return tokens
    }

    /// The audio digest space is domain-separated from images: identical
    /// bytes as image vs audio must never share a pseudo-token expansion.
    @Test func audioDigestIsDomainSeparatedFromImageDigest() {
        let bytes = Data("same-bytes".utf8)
        #expect(AudioDigest(audioBytes: bytes).rawBytes != ImageDigest(imageBytes: bytes).rawBytes)
    }

    /// An audio-bearing request keys: runs replace with negative pseudo-tokens
    /// (length-preserving), the audio table records the runs in prompt order,
    /// and framing tokens survive untouched.
    @Test func audioRunsKeyLengthPreservingInPromptOrder() throws {
        let prepared = Self.gemmaPrompt(imageRuns: [], audioRuns: [5, 3])
        let space = try CacheKeySpace.make(
            preparedTokens: prepared,
            imageDigests: [],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("clip-a"), Self.audioDigest("clip-b")],
            audioKeying: Self.gemmaAudioKeying
        ).get()

        #expect(space.keyPath.count == prepared.count)
        #expect(space.audioTable.count == 2)
        #expect(space.audioTable[0].runLength == 5)
        #expect(space.audioTable[1].runLength == 3)
        #expect(space.isIdentity == false)
        for entry in space.audioTable {
            for index in entry.runRange {
                #expect(space.keyPath[index] < 0)
            }
        }
        // Framing and text tokens are identical in both spaces.
        for (index, token) in prepared.enumerated()
        where !space.audioTable.contains(where: { $0.runRange.contains(index) }) {
            #expect(space.keyPath[index] == token)
        }
    }

    /// The sequential rule needs no grids: a Gemma image request keys off run
    /// lengths alone, and every image's span equals its run (anchor delta 0).
    @Test func sequentialImagesKeyWithoutGridsAndZeroDelta() throws {
        let prepared = Self.gemmaPrompt(imageRuns: [7, 4], audioRuns: [])
        let space = try CacheKeySpace.make(
            preparedTokens: prepared,
            imageDigests: [Self.digest("img-a"), Self.digest("img-b")],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying
        ).get()

        #expect(space.imageTable.count == 2)
        #expect(space.imageTable[0].positionSpan == 7)
        #expect(space.imageTable[1].positionSpan == 4)
        #expect(space.positionAnchorDelta(upTo: prepared.count) == 0)
    }

    /// Mixed media: images and audio key independently, the combined
    /// minimumWarmOffset covers the last run of either kind, and the anchor
    /// delta refuses offsets splitting an audio run.
    @Test func mixedMediaKeysBothTablesAndGuardsAudioSplits() throws {
        let prepared = Self.gemmaPrompt(imageRuns: [4], audioRuns: [6])
        let space = try CacheKeySpace.make(
            preparedTokens: prepared,
            imageDigests: [Self.digest("img")],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("clip")],
            audioKeying: Self.gemmaAudioKeying
        ).get()

        #expect(space.imageTable.count == 1)
        #expect(space.audioTable.count == 1)
        let audioRun = space.audioTable[0].runRange
        #expect(space.minimumWarmOffset == audioRun.upperBound)
        #expect(space.positionAnchorDelta(upTo: audioRun.lowerBound + 1) == nil)
        #expect(space.positionAnchorDelta(upTo: audioRun.upperBound) == 0)
        #expect(space.remainderAudioIndices(from: 0) == 0..<1)
        #expect(space.remainderAudioIndices(from: audioRun.upperBound) == 1..<1)
        #expect(space.remainderAudioIndices(from: audioRun.lowerBound + 1) == nil)
    }

    /// Audio without a recognized audio family degrades typed, mirroring the
    /// image guard.
    @Test func audioWithoutAudioKeyingIsUnkeyed() {
        let result = CacheKeySpace.make(
            preparedTokens: Self.gemmaPrompt(imageRuns: [], audioRuns: [4]),
            imageDigests: [],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("clip")],
            audioKeying: nil
        )
        #expect(result.failureReason == .unrecognizedAudioPlaceholderFamily)
    }

    /// Clip count ≠ run count degrades typed — keying would mis-attribute.
    @Test func audioRunCountMismatchIsUnkeyed() {
        let result = CacheKeySpace.make(
            preparedTokens: Self.gemmaPrompt(imageRuns: [], audioRuns: [4]),
            imageDigests: [],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("a"), Self.audioDigest("b")],
            audioKeying: Self.gemmaAudioKeying
        )
        #expect(result.failureReason == .audioPlaceholderRunCountMismatch)
    }

    /// Render translation splices both modalities: the i-th image pad maps to
    /// image i, the i-th audio pad to clip i, framing passes through.
    @Test func translationExpandsBothMediaPads() throws {
        let prepared = Self.gemmaPrompt(imageRuns: [4], audioRuns: [6])
        let space = try CacheKeySpace.make(
            preparedTokens: prepared,
            imageDigests: [Self.digest("img")],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("clip")],
            audioKeying: Self.gemmaAudioKeying
        ).get()

        // Render space: one pad per medium inside its framing.
        let render: [Int] =
            [1, 2, 3, 4]
            + [255_999, Self.gemmaImagePad, 258_882, 10, 11]
            + [256_000, Self.gemmaAudioPad, 258_883, 20, 21]
        let translated = try space.translate(renderTokens: render).get()
        #expect(translated == space.keyPath)
        #expect(space.translatedLength(renderTokens: render) == .success(space.keyPath.count))
    }

    /// A render with more audio pads than the request has clips fails typed.
    @Test func translationFailsTypedWhenRenderHasMoreClipsThanRequest() throws {
        let space = try CacheKeySpace.make(
            preparedTokens: Self.gemmaPrompt(imageRuns: [], audioRuns: [4]),
            imageDigests: [],
            imageGrids: [],
            imageKeying: Self.gemmaImageKeying,
            audioDigests: [Self.audioDigest("clip")],
            audioKeying: Self.gemmaAudioKeying
        ).get()

        let render = [1, Self.gemmaAudioPad, 2, Self.gemmaAudioPad]
        #expect(
            space.translate(renderTokens: render)
                == .failure(.audioPlaceholderOccurrencesExceedClips(occurrences: 2, clips: 1)))
    }

    /// Same run length, different clip bytes → different key paths (the
    /// digest drives the expansion).
    @Test func differentClipsSameLengthDivergeInKeyPath() throws {
        func space(_ seed: String) throws -> CacheKeySpace {
            try CacheKeySpace.make(
                preparedTokens: Self.gemmaPrompt(imageRuns: [], audioRuns: [4]),
                imageDigests: [],
                imageGrids: [],
                imageKeying: nil,
                audioDigests: [Self.audioDigest(seed)],
                audioKeying: Self.gemmaAudioKeying
            ).get()
        }
        #expect(try space("clip-a").keyPath != space("clip-b").keyPath)
    }
}

extension Result where Success == CacheKeySpace, Failure == CacheKeySpace.UnkeyedReason {
    fileprivate var failureReason: CacheKeySpace.UnkeyedReason? {
        if case .failure(let reason) = self { return reason }
        return nil
    }
}
