//
//  AgentLogitProcessorsTests.swift
//  tesseractTests
//
//  MLX-level coverage for the output-only presence penalty and the factory
//  that swaps it in for the vendor's prompt-seeded window.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct AgentLogitProcessorsTests {

    private let vocabSize = 64

    private func zeroLogits() -> MLXArray {
        MLXArray.zeros([1, vocabSize], type: Float32.self)
    }

    private func values(_ logits: MLXArray) -> [Float] {
        logits.eval()
        return logits.asArray(Float.self)
    }

    // MARK: - OutputPresencePenalty

    @Test func promptTokensAreNeverPenalized() {
        var processor = OutputPresencePenalty(penalty: 1.5)
        processor.prompt(MLXArray([Int32(1), 2, 3]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test func sampledTokensArePenalizedOnceEach() {
        var processor = OutputPresencePenalty(penalty: 1.5)
        processor.didSample(token: MLXArray([Int32(5)]))
        processor.didSample(token: MLXArray([Int32(9)]))
        // Re-sampling the same token must not stack the penalty.
        processor.didSample(token: MLXArray([Int32(5)]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -1.5)
        #expect(out[9] == -1.5)
        #expect(
            out.enumerated().allSatisfy { index, value in
                index == 5 || index == 9 ? true : value == 0
            })
    }

    /// The reason this type exists: the vendor's 20-token window forgets a
    /// token ~20 samples later; the whole-generation ring must not.
    @Test func penaltyPersistsFarBeyondTheVendorWindow() {
        var processor = OutputPresencePenalty(penalty: 1.5)
        for token in 0..<40 {
            processor.didSample(token: MLXArray([Int32(token)]))
        }

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[0] == -1.5)
        #expect(out[39] == -1.5)
        #expect(out[40] == 0)
    }

    @Test func ringWrapsAtCapacityInsteadOfGrowing() {
        var processor = OutputPresencePenalty(penalty: 1.0, capacity: 4)
        for token in [Int32(1), 2, 3, 4, 5] {
            processor.didSample(token: MLXArray([token]))
        }

        // Token 1 fell out of the 4-slot ring; 2...5 remain penalized.
        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[1] == 0)
        #expect(out[2] == -1.0)
        #expect(out[5] == -1.0)
    }

    // MARK: - Factory

    @Test func factoryReturnsNilWhenNoPenaltiesAreConfigured() {
        #expect(AgentLogitProcessors.processor(for: GenerateParameters()) == nil)
    }

    @Test func presenceOnlyParametersYieldOutputOnlySemantics() {
        let parameters = GenerateParameters(presencePenalty: 1.5, presenceContextSize: 20)
        var processor = AgentLogitProcessors.processor(for: parameters)!

        // Prompt seeding (what every generation path does) must not penalize.
        processor.prompt(MLXArray([Int32(7), 8]))
        let afterPrompt = values(processor.process(logits: zeroLogits()))
        #expect(afterPrompt.allSatisfy { $0 == 0 })

        // A sampled token stays penalized past the vendor's 20-token window.
        processor.didSample(token: MLXArray([Int32(7)]))
        for token in 30..<55 {
            processor.didSample(token: MLXArray([Int32(token)]))
        }
        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[7] == -1.5)
    }

    @Test func repetitionPenaltyKeepsVendorSemanticsAlongsidePresence() {
        let parameters = GenerateParameters(
            repetitionPenalty: 2.0, repetitionContextSize: 20,
            presencePenalty: 1.5
        )
        var processor = AgentLogitProcessors.processor(for: parameters)!

        // Repetition penalty IS prompt-seeded (vendor semantics): a positive
        // logit for a prompt token is divided by the penalty; presence adds
        // nothing for prompt tokens.
        processor.prompt(MLXArray([Int32(3)]))
        var logits = MLXArray.ones([1, vocabSize], type: Float32.self)
        logits = processor.process(logits: logits)
        let out = values(logits)
        #expect(out[3] == 0.5)
        #expect(out[4] == 1.0)
    }
}
