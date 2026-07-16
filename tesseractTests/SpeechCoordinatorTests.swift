//
//  SpeechCoordinatorTests.swift
//  tesseractTests
//
//  Coordinator tests over the *real* v2 SpeechEngine (TesseractSpeech) driven
//  by a scripted synthesizer and pass-through lease — replace-don't-layer: the
//  engine's event grammar, supersession, and cancellation run for real; only
//  the model boundary and the audio device are peers (`ScriptedSpeechSynthesizer`,
//  `InMemoryAudioPlayback`, `RecordingHighlightSurface`).
//

import Foundation
import Testing
import TesseractSpeech

@testable import Tesseract_Agent

@MainActor
final class InMemoryTextExtractor: TextExtracting {
    var result: Result<String, Error> = .success("Hello world.")
    func extractSelectedText() async throws -> String { try result.get() }
}

@MainActor
final class CallbackProbe {
    private(set) var fireCount = 0
    func fire() { fireCount += 1 }
}

@MainActor
private struct Harness {
    let coordinator: SpeechCoordinator
    let synthesizer: ScriptedSpeechSynthesizer
    let playback: InMemoryAudioPlayback
    let overlay: RecordingHighlightSurface
    let settings: SettingsManager
    let presenter: SpeechEnginePresenter

    init(script: ScriptedSpeechSynthesizer.Script = .init()) async {
        synthesizer = ScriptedSpeechSynthesizer()
        await synthesizer.configure(script)
        let engine = SpeechEngine(
            model: .voiceDesign17B(.q8), synthesizer: synthesizer, gpu: ImmediateGPULease())
        presenter = SpeechEnginePresenter(engine: engine)
        playback = InMemoryAudioPlayback()
        overlay = RecordingHighlightSurface()
        settings = SettingsManager(store: InMemorySettingsStore())
        coordinator = SpeechCoordinator(
            textExtractor: InMemoryTextExtractor(),
            engine: presenter,
            playback: playback,
            settings: settings,
            notchOverlay: overlay
        )
    }
}

/// Poll until `condition` holds (the coordinator drains on its own task).
@MainActor
private func waitUntil(
    timeout: Duration = .seconds(5), _ condition: @MainActor () async -> Bool
) async -> Bool {
    let deadline = ContinuousClock.now + timeout
    while ContinuousClock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return await condition()
}

@MainActor
struct SpeechCoordinatorTests {

    @Test
    func speakTextDrainsEngineEventsIntoPlaybackAndOverlay() async throws {
        let harness = await Harness()
        let probe = CallbackProbe()

        harness.coordinator.speakText("Hello world.") { probe.fire() }
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })

        // Audio path: one streaming session at the engine's sample rate,
        // every scripted chunk scheduled.
        #expect(harness.playback.startedSampleRates == [24_000])
        #expect(harness.playback.appendedChunks.count == 3)

        // Overlay path: shown with alignment offsets, closed out as complete.
        #expect(
            harness.overlay.calls.contains { if case .show = $0 { true } else { false } })
        #expect(harness.overlay.calls.contains(.markGenerationComplete))

        // Residency mirrored for views/arbiter once the session opened.
        #expect(harness.presenter.isModelLoaded)
        #expect(await harness.synthesizer.loadCount == 1)

        // Completion fires only when the audio layer reports drained.
        #expect(probe.fireCount == 0)
        harness.playback.firePlaybackFinished()
        #expect(probe.fireCount == 1)
        #expect(harness.coordinator.state == .idle)
    }

    @Test
    func stopCancelsGenerationAndResetsPresentation() async throws {
        let harness = await Harness(
            script: .init(chunksPerSegment: 200, chunkDelayNanos: 2_000_000))

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { !harness.playback.appendedChunks.isEmpty })

        harness.coordinator.stop()

        #expect(harness.coordinator.state == .idle)
        #expect(harness.playback.stopCount >= 1)
        #expect(harness.overlay.calls.contains(.dismiss))
        // The stream is the cancellation token: the synthesizer observed the
        // cancel within a step.
        #expect(await waitUntil { await harness.synthesizer.sawCancellation })
    }

    @Test
    func pauseHoldsPlaybackAndResumeContinues() async throws {
        let harness = await Harness(
            script: .init(chunksPerSegment: 200, chunkDelayNanos: 2_000_000))

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.coordinator.state == .streaming })

        harness.coordinator.pause()
        #expect(harness.playback.pauseCount == 1)
        #expect(harness.playback.isPaused)
        if case .paused = harness.coordinator.state {
        } else {
            Issue.record("expected .paused, got \(harness.coordinator.state)")
        }

        harness.coordinator.resume()
        #expect(harness.playback.resumeCount == 1)
        #expect(!harness.playback.isPaused)
        #expect(harness.coordinator.state == .streaming)

        harness.coordinator.stop()
    }

    @Test
    func sessionReusedForSameVoiceReopenedOnVoiceChangeAndSeedFlowsThrough() async throws {
        let harness = await Harness()
        harness.settings.ttsVoiceDescription = ""
        harness.settings.ttsSeed = 42

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.playback.firePlaybackFinished()

        harness.coordinator.speakText("Hello again.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 2 })
        harness.playback.firePlaybackFinished()

        // Same voice: one session, one prime.
        #expect(await harness.synthesizer.primedVoices.count == 1)

        harness.settings.ttsVoiceDescription = "warm narrator"
        harness.coordinator.speakText("New voice.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 3 })

        let primed = await harness.synthesizer.primedVoices
        #expect(primed.count == 2)
        #expect(primed.last == "warm narrator")

        // The settings seed rides every request (reproducibility knob).
        let requests = await harness.synthesizer.requests
        #expect(requests.allSatisfy { $0.seed == 42 })
        #expect(requests.last?.voiceDescription == "warm narrator")
    }

    @Test
    func speakTextClaimsStateBeforeFirstAwait() async throws {
        let harness = await Harness()

        harness.coordinator.speakText("Hello world.")
        // Synchronous read, no waiting: the voice session's settled-engine
        // watchdog polls this state — a transient `.idle` during the
        // session-open await reopened the mic under live TTS (ADR-0041).
        #expect(harness.coordinator.state != .idle)

        harness.coordinator.stop()
    }

    @Test
    func voiceSessionRouteStreamsThroughTheVoiceSink() async throws {
        let harness = await Harness()
        let voiceSink = InMemoryAudioPlayback()
        harness.coordinator.voiceSessionPlayback = voiceSink

        harness.coordinator.speakText(
            "Hello world.", showsOverlay: false, route: .voiceSession)
        #expect(await waitUntil { voiceSink.finishStreamingCount == 1 })

        // Dual-Path Playback (ADR-0041): the voice sink got the utterance,
        // the standard sink stayed silent.
        #expect(voiceSink.startedSampleRates == [24_000])
        #expect(voiceSink.appendedChunks.count == 3)
        #expect(harness.playback.startedSampleRates.isEmpty)

        voiceSink.firePlaybackFinished()
        #expect(harness.coordinator.state == .idle)

        // The next standard utterance returns to the default sink.
        harness.coordinator.speakText("Hello again.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        #expect(voiceSink.finishStreamingCount == 1)
        harness.playback.firePlaybackFinished()
    }

    @Test
    func generationFailureSurfacesTransientError() async throws {
        let harness = await Harness(script: .init(failOnSegmentIndex: 0))

        harness.coordinator.speakText("Hello world.")
        #expect(
            await waitUntil {
                if case .error = harness.coordinator.state { true } else { false }
            })
        #expect(harness.overlay.calls.contains(.dismiss))
    }
}
