//
//  SuppressedTokensTests.swift
//  tesseractTests
//
//  `suppress_tokens` support: the fork's SuppressedTokensProcessor masks the
//  listed ids to -inf every decode step, GenerateParameters composes it with
//  the penalty chain, and the Gemma 4 preset carries the checkpoint's list
//  (generation_config.json: eoi 258882, eoa 258883).
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("Suppressed tokens")
struct SuppressedTokensTests {

    @Test("processor masks listed ids to -inf and leaves the rest untouched")
    func masksOnlyListedIds() {
        let processor = SuppressedTokensProcessor(tokens: [1, 3])
        let logits = MLXArray([0.5, 1.5, 2.5, 3.5, 4.5] as [Float])[.newAxis, 0...]

        let processed = processor.process(logits: logits)
        let values: [Float] = processed[0].asArray(Float.self)

        #expect(values[0] == 0.5)
        #expect(values[1] == -.infinity)
        #expect(values[2] == 2.5)
        #expect(values[3] == -.infinity)
        #expect(values[4] == 4.5)
    }

    @Test("mask survives a float16 logits dtype")
    func masksFloat16Logits() {
        let processor = SuppressedTokensProcessor(tokens: [0])
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float]).asType(.float16)[.newAxis, 0...]

        let processed = processor.process(logits: logits)
        let values: [Float] = processed[0].asType(.float32).asArray(Float.self)

        #expect(values[0] == -.infinity)
        #expect(values[1] == 2.0)
        #expect(values[2] == 3.0)
    }

    @Test("GenerateParameters with no penalties and no suppression yields no processor")
    func noProcessorWhenUnconfigured() {
        let params = GenerateParameters()
        #expect(params.processor() == nil)
    }

    @Test("GenerateParameters with only suppressedTokens yields a masking processor")
    func suppressionAloneYieldsProcessor() {
        var params = GenerateParameters()
        params.suppressedTokens = [2]
        let processor = params.processor()
        #expect(processor != nil)

        let logits = MLXArray([1.0, 1.0, 1.0, 1.0] as [Float])[.newAxis, 0...]
        let values: [Float] = (processor!.process(logits: logits))[0].asArray(Float.self)
        #expect(values[2] == -.infinity)
        #expect(values[0] == 1.0)
    }

    @Test("suppression composes with the penalty chain")
    func suppressionComposesWithPenalties() {
        var params = GenerateParameters()
        params.suppressedTokens = [1]
        params.presencePenalty = 0.5
        var processor = params.processor()
        #expect(processor != nil)

        // Seed the presence ring so the penalty half of the chain is active too.
        processor!.prompt(MLXArray([3] as [Int32]))
        let logits = MLXArray([2.0, 2.0, 2.0, 2.0] as [Float])[.newAxis, 0...]
        let values: [Float] = (processor!.process(logits: logits))[0].asArray(Float.self)

        #expect(values[1] == -.infinity)
        #expect(values[3] == 1.5)  // presence-penalized
        #expect(values[0] == 2.0)
    }

    @Test("gemma-4 preset carries the checkpoint's sampling and suppression")
    func gemma4PresetMatchesCheckpoint() {
        let params = AgentGenerateParameters.forModel("gemma-4-12b")
        #expect(params.temperature == 1.0)
        #expect(params.topP == 0.95)
        #expect(params.topK == 64)
        #expect(params.suppressedTokens == [258_882, 258_883])
        #expect(params.thinkingSafeguard.enabled == false)
    }

    @Test("suppressedTokens survives the fork parameter mapping")
    func mappingThreadsSuppressedTokens() {
        let params = AgentGenerateParameters.forModel("gemma-4-12b")
        let mapped = LLMActor.makeGenerateParameters(from: params)
        #expect(mapped.suppressedTokens == [258_882, 258_883])
    }

    @Test("non-gemma models keep an empty suppression list")
    func otherModelsUnsuppressed() {
        #expect(AgentGenerateParameters.forModel("qwen3.6-35b-a3b-paro").suppressedTokens.isEmpty)
        #expect(AgentGenerateParameters.forModel("ornith-9b").suppressedTokens.isEmpty)
    }

    @Test("gemma-4 preset runs its measured prefill step, others keep the default")
    func gemma4PrefillStepIsMeasuredOptimum() {
        #expect(AgentGenerateParameters.forModel("gemma-4-12b").prefillStepSize == 512)
        #expect(AgentGenerateParameters.forModel("qwen3.6-35b-a3b-paro").prefillStepSize == 1024)
    }
}
