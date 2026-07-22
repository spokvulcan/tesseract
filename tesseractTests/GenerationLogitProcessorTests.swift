//
//  GenerationLogitProcessorTests.swift
//  tesseractTests
//
//  The one seam every raw-generation path routes through to decide which
//  logit processor it attaches (issue #405, enforcing ADR-0053's "every
//  generation path" clause). Two layers of coverage:
//
//  1. Decision rows at the seam's home — the `kvBits` carve-out and the
//     presence-on/off outcome, made once here so no call site re-derives it.
//  2. Per-constructible-path wiring — the state-threaded decode iterators run
//     their real inits over the toy model and are asserted to hold the seam's
//     output-only `OutputPresencePenalty`, not the vendor default. (The two
//     vendor-`TokenIterator` paths are constructible but their processor lives
//     in MLXLMCommon's non-`@testable` `TokenIterator.processor`, unreachable
//     from here — see the PR note; their decision is covered by the rows with
//     the exact `pathQuantizesKVUpFront` flag each passes.)
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@Suite struct GenerationLogitProcessorTests {

    // MARK: - Decision rows (the seam's home)

    private func presenceParams(kvBits: Int? = nil) -> GenerateParameters {
        GenerateParameters(kvBits: kvBits, temperature: 0, presencePenalty: 1.5)
    }

    @Test func noPenaltiesResolvesNilOnEitherPath() {
        let params = GenerateParameters(temperature: 0)
        #expect(
            GenerationLogitProcessor.resolve(for: params, pathQuantizesKVUpFront: true) == nil)
        #expect(
            GenerationLogitProcessor.resolve(for: params, pathQuantizesKVUpFront: false) == nil)
    }

    @Test func zeroPresencePenaltyResolvesNil() {
        let params = GenerateParameters(temperature: 0, presencePenalty: 0)
        #expect(
            GenerationLogitProcessor.resolve(for: params, pathQuantizesKVUpFront: true) == nil)
    }

    @Test func presenceResolvesOutputOnlyPenaltyWhenKVBitsNil() throws {
        let onQuantizingPath = try #require(
            GenerationLogitProcessor.resolve(
                for: presenceParams(), pathQuantizesKVUpFront: true))
        #expect(onQuantizingPath is OutputPresencePenalty)

        // The single-shot arm still gets the app processor when kvBits is nil —
        // the carve-out only bites when kvBits is set.
        let onSingleShot = try #require(
            GenerationLogitProcessor.resolve(
                for: presenceParams(), pathQuantizesKVUpFront: false))
        #expect(onSingleShot is OutputPresencePenalty)
    }

    @Test func appProcessorSurvivesKVBitsWhenPathQuantizesUpFront() throws {
        // Chunked-prefill / state-threaded paths quantize before their
        // iterator, so kvBits set does NOT divert them to the vendor.
        let resolved = try #require(
            GenerationLogitProcessor.resolve(
                for: presenceParams(kvBits: 8), pathQuantizesKVUpFront: true))
        #expect(resolved is OutputPresencePenalty)
    }

    @Test func kvBitsCarveOutDefersToVendorOnSingleShotPath() {
        // No up-front quantization point + kvBits set ⇒ defer to the vendor
        // init (nil), the theoretical carve-out (#252).
        #expect(
            GenerationLogitProcessor.resolve(
                for: presenceParams(kvBits: 8), pathQuantizesKVUpFront: false) == nil)
    }

    @Test func multiplePenaltiesComposeThroughTheSeam() throws {
        let params = GenerateParameters(
            temperature: 0, repetitionPenalty: 2.0, presencePenalty: 1.5)
        let resolved = try #require(
            GenerationLogitProcessor.resolve(for: params, pathQuantizesKVUpFront: true))
        #expect(resolved is CompositeLogitProcessor)
    }

    // MARK: - Per-path wiring (state-threaded decode, over the toy model)

    /// The cache-aware decode iterator (`makeDecodeIterator`, ADR-0007): its
    /// real init runs a prime forward on the toy model. The attached processor
    /// must be the seam's output-only presence, not the vendor window.
    @Test func stateThreadedDecodeIteratorAttachesTheSeamProcessor() async throws {
        let provider = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2, 3, 4, 5]))

        let withPresence = try await provider.withSession { session -> Bool in
            let params = GenerateParameters(temperature: 0, presencePenalty: 1.5)
            let iterator = session.makeDecodeIterator(
                remainder: LMInput.Text(tokens: MLXArray([Int32(5)])),
                fullText: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4])),
                cache: session.newCache(parameters: params),
                state: nil,
                parameters: params
            )
            return iterator.processor is OutputPresencePenalty
        }
        #expect(withPresence)

        let withoutPenalties = try await provider.withSession { session -> Bool in
            let params = GenerateParameters(temperature: 0)
            let iterator = session.makeDecodeIterator(
                remainder: LMInput.Text(tokens: MLXArray([Int32(5)])),
                fullText: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4])),
                cache: session.newCache(parameters: params),
                state: nil,
                parameters: params
            )
            return iterator.processor == nil
        }
        #expect(withoutPenalties)
    }

    /// The whole-prompt decode iterator (`makePreparingDecodeIterator`, the
    /// Unkeyed Completion): same seam contract through the `preparing:` init.
    @Test func stateThreadedPreparingIteratorAttachesTheSeamProcessor() async throws {
        let provider = ToyModelSessionProvider(model: ToyLanguageModel(script: [1, 2, 3, 4, 5]))

        let withPresence = try await provider.withSession { session -> Bool in
            let params = GenerateParameters(temperature: 0, presencePenalty: 1.5)
            let iterator = try session.makePreparingDecodeIterator(
                LMInput(tokens: MLXArray([Int32(1), 2, 3])),
                cache: session.newCache(parameters: params),
                parameters: params,
                prepare: nil
            )
            return iterator.processor is OutputPresencePenalty
        }
        #expect(withPresence)

        let withoutPenalties = try await provider.withSession { session -> Bool in
            let params = GenerateParameters(temperature: 0)
            let iterator = try session.makePreparingDecodeIterator(
                LMInput(tokens: MLXArray([Int32(1), 2, 3])),
                cache: session.newCache(parameters: params),
                parameters: params,
                prepare: nil
            )
            return iterator.processor == nil
        }
        #expect(withoutPenalties)
    }
}
