//
//  PrefillStrategyTests.swift
//  tesseractTests
//
//  The Prefill Strategy's decision table (ADR-0044): the chunked-vs-single-
//  shot route for one raw-generation prompt, pinned with no model and no
//  Metal. This rule used to live as three hand-written guards that had
//  already drifted apart; each table below is one leg of its one home.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite struct PrefillStrategyTests {

    private func decide(
        ndim: Int = 2,
        length: Int = 4096,
        image: Bool = false,
        video: Bool = false,
        audio: Bool = false,
        step: Int? = 1024
    ) -> PrefillStrategy {
        PrefillStrategy.decide(
            tokenNDim: ndim,
            sequenceLength: length,
            hasImage: image,
            hasVideo: video,
            hasAudio: audio,
            prefillStepSize: step
        )
    }

    @Test func aLongTextOnlyVLMClassPromptChunks() {
        #expect(decide() == .chunked(stepSize: 1024))
    }

    @Test func flatLLMClassTokensGoSingleShot() {
        // Upstream's TokenIterator chunks 1D prompts internally — the app
        // driver never takes them, however long.
        #expect(decide(ndim: 1) == .singleShot)
        #expect(decide(ndim: 1, length: 200_000) == .singleShot)
    }

    @Test func mediaKeepsThePromptSingleShot() {
        // The model's own `prepare` places media tokens; the chunked driver
        // only ever forwards text. Each medium alone forces the route.
        #expect(decide(image: true) == .singleShot)
        #expect(decide(video: true) == .singleShot)
        #expect(decide(audio: true) == .singleShot)
    }

    @Test func aPromptOfExactlyOneStepStaysSingleShot() {
        // Strict `>`: chunking a prompt that fits one step buys nothing.
        #expect(decide(length: 1024) == .singleShot)
        #expect(decide(length: 1025) == .chunked(stepSize: 1024))
    }

    @Test func theCarriedStepSizeIsTheBoundary() {
        #expect(decide(length: 2048, step: 2048) == .singleShot)
        #expect(decide(length: 2049, step: 2048) == .chunked(stepSize: 2048))
    }

    @Test func aMissingStepSizeFallsBackTo512() {
        #expect(decide(length: 512, step: nil) == .singleShot)
        #expect(decide(length: 513, step: nil) == .chunked(stepSize: 512))
    }

    @Test func theExtractorReadsThePreparedPromptShape() {
        // The `decide(for:)` overload is the exact place the drifted guards
        // lived — pin that it derives each fact from the input itself.
        let batched = LMInput(tokens: MLXArray.zeros([1, 600], type: Int32.self))
        #expect(
            PrefillStrategy.decide(for: batched, prefillStepSize: 512)
                == .chunked(stepSize: 512))

        let flat = LMInput(tokens: MLXArray.zeros([600], type: Int32.self))
        #expect(PrefillStrategy.decide(for: flat, prefillStepSize: 512) == .singleShot)

        let imageBearing = LMInput(
            text: .init(tokens: MLXArray.zeros([1, 600], type: Int32.self)),
            image: .init(pixels: MLXArray.zeros([1, 4, 4], type: Float.self))
        )
        #expect(
            PrefillStrategy.decide(for: imageBearing, prefillStepSize: 512)
                == .singleShot)
    }
}
