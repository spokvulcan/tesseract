//
//  SpeechEngineTests.swift
//  tesseractTests
//
//  Exercises `SpeechEngine`'s orchestration — the facade above the
//  `SpeechSynthesizer` port — through the public interface, with an
//  `InMemorySpeechSynthesizer` substituted below the seam. No MLX, no model
//  files: the default model repo, the `@Observable` load lifecycle, and the
//  mapping of model failures onto `SpeechError` all stay on the facade and are
//  asserted here against a hermetic adapter.
//

import Foundation
import os
import Testing

@testable import Tesseract_Agent

/// A `Sendable` factory spy: records how many synthesizers the engine built (one
/// per successful `loadModel`) and hands back each instance, so a test can assert
/// reload-via-factory and inspect the fresh adapter. `Sendable` without
/// `@unchecked` — its state lives behind an `OSAllocatedUnfairLock`.
final class SynthesizerFactorySpy: Sendable {
    private struct State {
        var builtCount = 0
        var synthesizers: [InMemorySpeechSynthesizer] = []
    }
    private let state = OSAllocatedUnfairLock(initialState: State())
    private let samples: [Float]
    private let sampleRate: Int

    init(samples: [Float] = [0.0], sampleRate: Int = 24_000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    func make() -> any SpeechSynthesizer {
        let synth = InMemorySpeechSynthesizer(samples: samples, sampleRate: sampleRate)
        state.withLock {
            $0.builtCount += 1
            $0.synthesizers.append(synth)
        }
        return synth
    }

    var builtCount: Int { state.withLock { $0.builtCount } }
    var synthesizers: [InMemorySpeechSynthesizer] { state.withLock { $0.synthesizers } }
}

@MainActor
struct SpeechEngineTests {

    // MARK: - Helpers

    /// Runs a throwing op and returns the `SpeechError` it threw, failing the test
    /// if it didn't throw one. (`SpeechError` is not `Equatable`, so cases are
    /// matched with `if case` at the call site.)
    private func captureSpeechError(
        _ op: () async throws -> (samples: [Float], sampleRate: Int),
        sourceLocation: SourceLocation = #_sourceLocation
    ) async -> SpeechError? {
        do {
            _ = try await op()
            Issue.record("expected a SpeechError to be thrown", sourceLocation: sourceLocation)
            return nil
        } catch let error as SpeechError {
            return error
        } catch {
            Issue.record("expected SpeechError, got \(error)", sourceLocation: sourceLocation)
            return nil
        }
    }

    // MARK: - Tracer: load + generate across the seam

    @Test
    func loadsAndGeneratesThroughTheSeam() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.5, 0.6], sampleRate: 24_000)
        let engine = SpeechEngine(makeSynthesizer: { synth })

        try await engine.loadModel()
        #expect(engine.isModelLoaded)

        let (samples, sampleRate) = try await engine.generate(
            text: "hi", voice: nil, language: nil, parameters: .default
        )

        #expect(samples == [0.5, 0.6])
        #expect(sampleRate == 24_000)
        #expect(await synth.loadCount == 1)
        #expect(await synth.generateCount == 1)
        #expect(await synth.recordedTexts == ["hi"])
        #expect(!engine.isLoading)
    }

    // MARK: - Argument passthrough across the seam

    @Test
    func generatePassesVoiceLanguageAndParametersThroughUnchanged() async throws {
        let synth = InMemorySpeechSynthesizer()
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        var params = TTSParameters.default
        params.seed = 99
        params.temperature = 0.42
        _ = try await engine.generate(
            text: "t", voice: "narrator", language: "fr", parameters: params)

        #expect(await synth.recordedVoices == ["narrator"])
        #expect(await synth.recordedLanguages == ["fr"])
        #expect(await synth.recordedParameters == [params])
    }

    @Test
    func generateStreamingPassesVoiceLanguageAndParametersThroughUnchanged() async throws {
        let synth = InMemorySpeechSynthesizer()
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        var params = TTSParameters.default
        params.seed = 7
        params.topP = 0.55
        let (stream, _) = try await engine.generateStreaming(
            text: "t", voice: "bard", language: "de", parameters: params, useVoiceAnchor: true
        )
        for try await _ in stream {}

        #expect(await synth.recordedStreamingVoices == ["bard"])
        #expect(await synth.recordedStreamingLanguages == ["de"])
        #expect(await synth.recordedStreamingParameters == [params])
        #expect(await synth.recordedUseVoiceAnchor == [true])
    }

    // MARK: - Generation-failure propagation (the facade does NOT remap generate errors)

    @Test
    func generateFailurePropagatesUnchanged() async throws {
        let synth = InMemorySpeechSynthesizer(generateError: FakeModelError(message: "boom"))
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        // Unlike `loadModel` (which maps onto `.modelLoadFailed`), `generate` is a
        // passthrough — the raw model error surfaces, not a `SpeechError`.
        await #expect(throws: FakeModelError.self) {
            _ = try await engine.generate(
                text: "t", voice: nil, language: nil, parameters: .default)
        }
    }

    @Test
    func generateStreamingFailurePropagatesUnchangedAsTheStreamsTerminatingError() async throws {
        let synth = InMemorySpeechSynthesizer(
            streamingError: FakeModelError(message: "decode failed"))
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        // The `generateStreaming` call itself succeeds (the stream is returned);
        // the raw model error surfaces as the stream's terminating error during
        // iteration, not re-wrapped as a `SpeechError`.
        let (stream, _) = try await engine.generateStreaming(
            text: "t", voice: nil, language: nil, parameters: .default
        )
        await #expect(throws: FakeModelError.self) {
            for try await _ in stream {}
        }
    }

    // MARK: - Not-loaded guard (stays on the facade)

    @Test
    func generateBeforeLoadThrowsModelNotLoadedWithoutTouchingTheSynthesizer() async throws {
        let synth = InMemorySpeechSynthesizer()
        let engine = SpeechEngine(makeSynthesizer: { synth })

        let error = await captureSpeechError {
            try await engine.generate(text: "hi", voice: nil, language: nil, parameters: .default)
        }

        guard case .modelNotLoaded = error else {
            Issue.record("expected .modelNotLoaded, got \(String(describing: error))")
            return
        }
        // The factory was never invoked and the synthesizer never saw a call.
        #expect(await synth.generateCount == 0)
        #expect(await synth.loadCount == 0)
        #expect(!engine.isModelLoaded)
    }

    // MARK: - Load-failure mapping (stays on the facade)

    @Test
    func modelLoadFailureIsMappedOntoModelLoadFailed() async throws {
        let synth = InMemorySpeechSynthesizer(loadError: FakeModelError(message: "no weights"))
        let engine = SpeechEngine(makeSynthesizer: { synth })

        var caught: SpeechError?
        do {
            try await engine.loadModel()
            Issue.record("expected loadModel to throw")
        } catch let error as SpeechError {
            caught = error
        }

        guard case .modelLoadFailed = caught else {
            Issue.record(
                "expected model failure mapped to .modelLoadFailed, got \(String(describing: caught))"
            )
            return
        }
        // The facade unwinds its load lifecycle on failure.
        #expect(!engine.isModelLoaded)
        #expect(!engine.isLoading)
        #expect(engine.loadingStatus == "")
        // A subsequent generate still reports not-loaded.
        let genError = await captureSpeechError {
            try await engine.generate(text: "x", voice: nil, language: nil, parameters: .default)
        }
        guard case .modelNotLoaded = genError else {
            Issue.record(
                "expected .modelNotLoaded after failed load, got \(String(describing: genError))")
            return
        }
    }

    // MARK: - Streaming across the seam

    @Test
    func generateStreamingYieldsChunksThroughTheSeamWithDefaultVoiceAnchor() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.7, 0.8], sampleRate: 16_000)
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        // `useVoiceAnchor` omitted — the facade's default (false) must flow through.
        let (stream, sampleRate) = try await engine.generateStreaming(
            text: "stream me", voice: nil, language: nil, parameters: .default
        )

        var chunks: [[Float]] = []
        for try await chunk in stream { chunks.append(chunk) }

        #expect(chunks == [[0.7, 0.8]])
        #expect(sampleRate == 16_000)
        #expect(await synth.streamingCount == 1)
        #expect(await synth.recordedStreamingTexts == ["stream me"])
        #expect(await synth.recordedUseVoiceAnchor == [false])
    }

    // MARK: - Unload lifecycle

    @Test
    func unloadReturnsFacadeToNotLoaded() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.1], sampleRate: 24_000)
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()
        _ = try await engine.generate(text: "ok", voice: nil, language: nil, parameters: .default)

        engine.unloadModel()
        #expect(!engine.isModelLoaded)
        #expect(engine.loadingStatus == "")

        let error = await captureSpeechError {
            try await engine.generate(
                text: "after", voice: nil, language: nil, parameters: .default)
        }
        guard case .modelNotLoaded = error else {
            Issue.record("expected .modelNotLoaded after unload, got \(String(describing: error))")
            return
        }
        // The post-unload generate never reached the (released) synthesizer.
        #expect(await synth.generateCount == 1)
    }

    // MARK: - Factory lifecycle

    @Test
    func loadsOnceWhileLoadedAndRebuildsViaFactoryAfterUnload() async throws {
        let spy = SynthesizerFactorySpy()
        let engine = SpeechEngine(makeSynthesizer: { spy.make() })

        try await engine.loadModel()
        // A redundant load while already loaded is a no-op — no second adapter.
        try await engine.loadModel()
        #expect(spy.builtCount == 1)
        #expect(await spy.synthesizers[0].loadCount == 1)

        // Unload, then load again: a *fresh* adapter is built via the factory.
        engine.unloadModel()
        try await engine.loadModel()

        #expect(spy.builtCount == 2)
        #expect(engine.isModelLoaded)
        #expect(await spy.synthesizers[1].loadCount == 1)
        // The default model repo is supplied by the facade to each fresh adapter.
        #expect(
            await spy.synthesizers[1].loadedRepos == [
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
            ])
    }

    // MARK: - Voice-anchor / cancel / offsets delegation across the seam

    @Test
    func voiceAnchorCancelAndOffsetsReachTheSynthesizerWhenLoaded() async throws {
        let synth = InMemorySpeechSynthesizer(tokenOffsets: [0, 3, 7])
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()

        await engine.buildVoiceAnchor(referenceCount: 2, voice: "narrator", language: "en")
        await engine.clearVoiceAnchor()
        await engine.cancelGeneration()
        let offsets = await engine.computeTokenCharOffsets(text: "align me")

        #expect(offsets == [0, 3, 7])
        #expect(await synth.buildVoiceAnchorCount == 1)
        #expect(await synth.recordedReferenceCounts == [2])
        #expect(await synth.clearVoiceAnchorCount == 1)
        #expect(await synth.cancelCount == 1)
        #expect(await synth.recordedOffsetTexts == ["align me"])
    }

    @Test
    func voiceAnchorCancelAndOffsetsAreSafeNoOpsWhenNotLoaded() async throws {
        let spy = SynthesizerFactorySpy()
        let engine = SpeechEngine(makeSynthesizer: { spy.make() })

        // Not loaded: these must not throw, must not lazily build an adapter, and
        // offsets must come back empty.
        await engine.buildVoiceAnchor(referenceCount: 2, voice: nil, language: nil)
        await engine.clearVoiceAnchor()
        await engine.cancelGeneration()
        let offsets = await engine.computeTokenCharOffsets(text: "x")

        #expect(offsets.isEmpty)
        #expect(spy.builtCount == 0)
        #expect(!engine.isModelLoaded)
    }
}
