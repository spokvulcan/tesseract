//
//  GenerationLogitProcessorTests.swift
//  tesseractTests
//
//  The one seam every raw-generation path routes through to decide which
//  logit processor it attaches (issue #405, enforcing ADR-0053's "every
//  generation path" clause): the decision rows at the seam's home. Which
//  processor each generation path ends up decoding with — the penalty
//  seed, the output-only presence, the `kvBits` carve-out — is pinned by
//  outcome in `DecodeStartTests`, through the one module that starts every
//  path's decode.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite struct GenerationLogitProcessorTests {

    @Test func noPenaltiesResolvesNil() {
        #expect(GenerationLogitProcessor.processor(for: GenerateParameters(temperature: 0)) == nil)
    }

    @Test func zeroPresencePenaltyResolvesNil() {
        let params = GenerateParameters(temperature: 0, presencePenalty: 0)
        #expect(GenerationLogitProcessor.processor(for: params) == nil)
    }

    @Test func presenceResolvesOutputOnlyPenalty() throws {
        let params = GenerateParameters(temperature: 0, presencePenalty: 1.5)
        let resolved = try #require(GenerationLogitProcessor.processor(for: params))
        #expect(resolved is OutputPresencePenalty)
    }

    @Test func multiplePenaltiesComposeThroughTheSeam() throws {
        let params = GenerateParameters(
            temperature: 0, repetitionPenalty: 2.0, presencePenalty: 1.5)
        let resolved = try #require(GenerationLogitProcessor.processor(for: params))
        #expect(resolved is CompositeLogitProcessor)
    }

    /// An iterator built from parameters keeps the vendor's KV plan (a KV
    /// Scheme's single-shot arm, the Speculation Plan): the penalties leave
    /// the parameters and the app processor rides a component instead.
    @Test func componentsCarryTheAppProcessorWithoutTheVendorPenalties() {
        var parameters = GenerateParameters(temperature: 0.6)
        parameters.presencePenalty = 1.5
        parameters.repetitionPenalty = 1.1
        parameters.kvScheme = KVScheme.turbo8v4.rawValue
        let (stripped, components) = GenerationLogitProcessor.components(for: parameters)
        #expect(stripped.presencePenalty == nil)
        #expect(stripped.repetitionPenalty == nil)
        #expect(stripped.kvScheme == "turbo8v4")
        #expect(components.logitProcessor(parameters: stripped) != nil)
        let (_, none) = GenerationLogitProcessor.components(for: GenerateParameters())
        #expect(none.logitProcessor(parameters: GenerateParameters()) == nil)
    }
}
