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
    private(set) var extractCount = 0
    func extractSelectedText() async throws -> String {
        extractCount += 1
        return try result.get()
    }
}

/// The Voice Engine's catalog status as the coordinator reads it.
@MainActor
final class VoiceEngineStatusStub {
    var status: ModelStatus = .downloaded(sizeOnDisk: 1)
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
    let pinnedVoices: PinnedVoiceStore
    let textExtractor = InMemoryTextExtractor()
    let voiceEngine = VoiceEngineStatusStub()
    /// Fires when the coordinator sends the user to the Models page.
    let modelsPage = CallbackProbe()

    /// `pinnedVoices`: pass an earlier harness's store to act as a relaunch.
    /// The default is a fresh temporary one, never the real app-support file.
    init(
        script: ScriptedSpeechSynthesizer.Script = .init(),
        pinnedVoices: PinnedVoiceStore? = nil
    ) async {
        self.pinnedVoices =
            pinnedVoices ?? PinnedVoiceStore(directory: makeTempDir("pinned-voices"))
        synthesizer = ScriptedSpeechSynthesizer()
        await synthesizer.configure(script)
        // The app's checkpoint, as DependencyContainer wires it: stored voices
        // are looked up under this spec.
        let engine = SpeechEngine(
            model: ModelDefinition.textToSpeechModelSpec, synthesizer: synthesizer,
            gpu: ImmediateGPULease())
        presenter = SpeechEnginePresenter(engine: engine)
        playback = InMemoryAudioPlayback()
        overlay = RecordingHighlightSurface()
        settings = SettingsManager(store: InMemorySettingsStore())
        coordinator = SpeechCoordinator(
            textExtractor: textExtractor,
            engine: presenter,
            voiceEngineStatus: { [voiceEngine] in voiceEngine.status },
            playback: playback,
            settings: settings,
            notchOverlay: overlay,
            pinnedVoices: self.pinnedVoices
        )
        coordinator.onVoiceEngineMissing = { [modelsPage] in modelsPage.fire() }
    }

    /// The message of a settled `.error` state, nil otherwise.
    var errorMessage: String? {
        if case .error(let message) = coordinator.state { return message }
        return nil
    }
}

@MainActor
struct SpeechCoordinatorTests {

    /// Poll until `condition` holds (the coordinator drains on its own task).
    /// The minute is a backstop, not a latency budget: every event of an
    /// utterance hops between the engine's actors and the main actor, and in
    /// the first seconds of a parallel run each hop can wait that long for a
    /// thread behind the other suites' work. A member, so it shadows the
    /// shared five-second `waitUntil`, which a file-level helper loses to for
    /// every condition that doesn't await.
    private func waitUntil(
        timeout: Duration = .seconds(60), _ condition: @MainActor () async -> Bool
    ) async -> Bool {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

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

        // Overlay path: shown, closed out as complete.
        #expect(
            harness.overlay.calls.contains { if case .show = $0 { true } else { false } })
        #expect(harness.overlay.calls.contains(.markGenerationComplete))

        // Residency mirrored for views/arbiter from the engine's readiness.
        #expect(await waitUntil { harness.presenter.isModelLoaded })
        #expect(await harness.synthesizer.loadCount == 1)

        // Completion fires only when the audio layer reports drained.
        #expect(probe.fireCount == 0)
        harness.playback.firePlaybackFinished()
        #expect(probe.fireCount == 1)
        #expect(harness.coordinator.state == .idle)
    }

    /// The engine's word starts reach the Read-Along in the utterance's
    /// seconds (ADR-0077).
    @Test
    func wordTimingReachesTheSurfaceInSeconds() async throws {
        let harness = await Harness(
            script: .init(wordStarts: [WordStart(word: 0, frame: 0), WordStart(word: 1, frame: 3)]))
        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        // 3 frames at 12.5 a second.
        #expect(
            harness.overlay.calls.contains(
                .timeWords(
                    [TimedWord(word: 0, start: 0), TimedWord(word: 1, start: 0.24)], segment: 0)))
        harness.coordinator.stop()
    }

    @Test
    func stopCancelsGenerationAndResetsPresentation() async throws {
        let harness = await Harness(script: .init(holdAfterChunks: 1))

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
        let harness = await Harness(script: .init(holdAfterChunks: 1))

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

    /// Offload Model unloads the engine but keeps the session (ADR-0038/0039),
    /// so reading again in the same voice reloads the checkpoint under that
    /// session. The presenter follows the engine's own Readiness, so the
    /// Models page and the next Offload Model see the voice model loaded.
    @Test
    func aSameVoiceReadAfterUnloadShowsTheModelLoadedAgain() async throws {
        let harness = await Harness()

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.playback.firePlaybackFinished()
        #expect(await waitUntil { harness.presenter.isModelLoaded })

        // What Offload Model does for the voice slot.
        await harness.presenter.unload()
        #expect(await waitUntil { !harness.presenter.isModelLoaded })

        harness.coordinator.speakText("Hello again.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 2 })
        harness.playback.firePlaybackFinished()

        #expect(await harness.synthesizer.loadCount == 2)
        #expect(await harness.presenter.engine.readiness == .warm)
        #expect(await waitUntil { harness.presenter.isModelLoaded })
    }

    /// ADR-0072: the first take of a designed voice is kept, and the next
    /// launch opens the voice pinned to it instead of rolling a new one.
    @Test
    func theFirstTakeIsRememberedAndARelaunchOpensWithIt() async throws {
        let harness = await Harness()
        harness.settings.ttsVoiceDescription = "warm narrator"
        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.playback.firePlaybackFinished()

        let stored = try #require(
            harness.pinnedVoices.voice(
                description: "warm narrator", language: harness.settings.ttsLanguage,
                model: ModelDefinition.textToSpeechModelSpec))
        #expect(stored.referenceText == "Hello world.")

        let relaunched = await Harness(pinnedVoices: harness.pinnedVoices)
        relaunched.settings.ttsVoiceDescription = "warm narrator"
        relaunched.coordinator.speakText("And again.")
        #expect(await waitUntil { relaunched.playback.finishStreamingCount == 1 })
        let first = try #require(await relaunched.synthesizer.requests.first)
        #expect(first.reference?.codeFrames == stored.codeFrames)
        #expect(!first.capturesReference, "a remembered voice needs no new take")
    }

    /// "Try another take" renders only the opening of the text, from the
    /// description alone and with a fresh seed, then replaces the stored voice.
    @Test
    func tryAnotherTakeRendersTheOpeningAndReplacesTheStoredVoice() async throws {
        let harness = await Harness()
        harness.settings.ttsVoiceDescription = "warm narrator"
        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.playback.firePlaybackFinished()
        let model = ModelDefinition.textToSpeechModelSpec
        let language = harness.settings.ttsLanguage
        let before = harness.pinnedVoices.voice(
            description: "warm narrator", language: language, model: model)

        let longText =
            "The rain had stopped by the time she reached the harbor. "
            + String(repeating: "The boats rocked quietly against the old stone pier. ", count: 20)
        harness.coordinator.tryAnotherTake(sampleFrom: longText)
        #expect(await waitUntil { harness.playback.finishStreamingCount == 2 })

        let take = try #require(await harness.synthesizer.requests.last)
        #expect(take.capturesReference)
        #expect(take.reference == nil)
        #expect(take.text.split(separator: " ").count <= 40, "only the opening is spoken")
        #expect(take.seed != UInt64(harness.settings.ttsSeed), "a retake rolls a fresh seed")
        let after = harness.pinnedVoices.voice(
            description: "warm narrator", language: language, model: model)
        #expect(after != before)
        #expect(after?.referenceText == take.text)
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

    // MARK: Soft Barge surface (ADR-0041)

    @Test
    func playbackLevelNowForwardsToTheActiveSink() async throws {
        let harness = await Harness()
        harness.playback.scriptedPlaybackLevel = 0.42
        #expect(harness.coordinator.playbackLevelNow() == 0.42)

        // The voice route swaps the active sink — the reading follows it.
        let voiceSink = InMemoryAudioPlayback()
        voiceSink.scriptedPlaybackLevel = 0.17
        harness.coordinator.voiceSessionPlayback = voiceSink
        harness.coordinator.speakText(
            "Hello world.", showsOverlay: false, route: .voiceSession)
        #expect(harness.coordinator.playbackLevelNow() == 0.17)
        harness.coordinator.stop()
    }

    @Test
    func fadePlaybackStepsTheSinkVolumeToTheTarget() async throws {
        let harness = await Harness()

        harness.coordinator.fadePlayback(to: 0.25, over: 0.1)
        #expect(await waitUntil { harness.playback.setVolumeCalls.last == 0.25 })
        // A ramp, not a jump — intermediate steps landed too.
        #expect(harness.playback.setVolumeCalls.count > 1)

        // Zero duration is an instant set.
        harness.coordinator.fadePlayback(to: 1.0, over: 0)
        #expect(harness.playback.setVolumeCalls.last == 1.0)
    }

    @Test
    func stopCancelsAnInFlightFade() async throws {
        let harness = await Harness()

        harness.coordinator.fadePlayback(to: 0.25, over: 2.0)
        _ = await waitUntil { !harness.playback.setVolumeCalls.isEmpty }
        harness.coordinator.stop()
        let countAtStop = harness.playback.setVolumeCalls.count
        try await Task.sleep(for: .milliseconds(100))
        // The fade died with the utterance — no further steps.
        #expect(harness.playback.setVolumeCalls.count == countAtStop)
    }

    // MARK: Voice Engine availability (the engine never downloads)

    @Test
    func userRequestWithoutVoiceEngineOpensModelsAndLoadsNothing() async throws {
        let harness = await Harness()
        harness.voiceEngine.status = .notDownloaded

        harness.coordinator.speakText("Hello world.", userInitiated: true)
        #expect(await waitUntil { harness.errorMessage != nil })

        #expect(harness.errorMessage?.contains("Voice Engine") == true)
        #expect(harness.modelsPage.fireCount == 1)
        #expect(await harness.synthesizer.loadCount == 0)
        #expect(harness.playback.startedSampleRates.isEmpty)
        #expect(!harness.presenter.isLoading)
    }

    /// Auto-speak and the Companion speak on their own; a missing Voice
    /// Engine shows the error but doesn't pull the window to Models.
    @Test
    func automaticSpeechWithoutVoiceEngineStaysPut() async throws {
        let harness = await Harness()
        harness.voiceEngine.status = .notDownloaded

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.errorMessage != nil })

        #expect(harness.modelsPage.fireCount == 0)
        #expect(await harness.synthesizer.loadCount == 0)
    }

    @Test
    func hotkeyWithoutVoiceEngineSkipsTheCaptureAndOpensModels() async throws {
        let harness = await Harness()
        harness.voiceEngine.status = .notDownloaded

        harness.coordinator.onHotkeyPressed()
        #expect(await waitUntil { harness.errorMessage != nil })

        #expect(harness.textExtractor.extractCount == 0, "selection left alone")
        #expect(harness.modelsPage.fireCount == 1)
    }

    /// Onboarding or the Models page is still fetching it: say so, and
    /// don't load a half-written checkpoint.
    @Test
    func downloadingVoiceEngineWaitsWithoutLoading() async throws {
        let harness = await Harness()
        harness.voiceEngine.status = .downloading(progress: 0.4)

        harness.coordinator.speakText("Hello world.", userInitiated: true)
        #expect(await waitUntil { harness.errorMessage != nil })

        #expect(harness.errorMessage?.contains("40%") == true)
        #expect(harness.modelsPage.fireCount == 0)
        #expect(await harness.synthesizer.loadCount == 0)
    }

    /// The catalog said downloaded, but the engine's disk check found the
    /// folder gone: the same outcome as a missing Voice Engine.
    @Test
    func engineFindingNoCheckpointIsTreatedAsMissing() async throws {
        let harness = await Harness()
        await harness.synthesizer.setCheckpointOnDisk(false)

        harness.coordinator.speakText("Hello world.", userInitiated: true)
        #expect(await waitUntil { harness.errorMessage != nil })

        #expect(harness.errorMessage?.contains("Voice Engine") == true)
        #expect(harness.modelsPage.fireCount == 1)
        #expect(await harness.synthesizer.loadCount == 0)
        #expect(!harness.presenter.isModelLoaded)
        #expect(!harness.presenter.isLoading)
    }

    /// A failed download or verify isn't proof the files are gone; the
    /// engine's disk check decides.
    @Test
    func errorStatusStillLetsTheEngineCheckTheDisk() async throws {
        let harness = await Harness()
        harness.voiceEngine.status = .error("The network connection was lost.")

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        #expect(harness.modelsPage.fireCount == 0)
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

    /// A new request cancels one parked between segments. The cancelled one
    /// wakes later: it once set the state idle and dropped the completion
    /// callback, both by then the new request's (the Reader's jump).
    @Test
    func aSupersededRequestLeavesTheNewOneAlone() async throws {
        let harness = await Harness()
        let long = String(
            repeating: "The boats rocked quietly against the old stone pier. ", count: 300)
        harness.coordinator.speakText(long)
        // Parked: 8 s of audio ahead of a head that doesn't move.
        #expect(await waitUntil { harness.playback.totalScheduledDuration >= 8 })

        let probe = CallbackProbe()
        harness.coordinator.speakText("Hello again.") { probe.fire() }
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        try? await Task.sleep(for: .milliseconds(100))
        #expect(harness.coordinator.state != .idle)

        harness.playback.firePlaybackFinished()
        #expect(probe.fireCount == 1)
    }

    /// A stop can land while the stopped request's stream still holds
    /// events the engine sent before it. The stopped request once drained
    /// them into what the next one owned by then: its text shown over the
    /// new reading, its audio appended to the new playback, and its finish
    /// ending the new request early.
    @Test
    func aStoppedRequestLeavesItsBufferedEventsAlone() async throws {
        let harness = await Harness()
        var buffered: DispatchTimeoutResult?
        // As the first request starts playing, hold the main actor until
        // the engine has sent its whole stream, then start the next one:
        // the stop lands with every event of the first still to drain.
        harness.playback.onStartStreaming = {
            harness.playback.onStartStreaming = nil
            buffered = harness.synthesizer.utteranceEnds.wait(timeout: .now() + 60)
            harness.coordinator.speakText("Hello again.")
        }
        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.overlay.calls.contains(.markGenerationComplete) })

        #expect(buffered == .success)
        #expect(harness.overlay.displayedTexts == ["Hello again."])
        #expect(harness.playback.appendedChunks.count == 3)
        #expect(harness.playback.finishStreamingCount == 1)
        harness.playback.firePlaybackFinished()
    }

    /// A stop can land while the session opens, which loads the voice model
    /// for the first request. The stopped request once carried on once it had
    /// opened: back to generating for good after the stop, speaking its text,
    /// and starting the sink again under the next request.
    @Test
    func aRequestStoppedWhileItsSessionOpensStaysStopped() async throws {
        let harness = await Harness()
        harness.coordinator.speakText("Hello world.")
        harness.coordinator.stop()
        // The stopped request has seen its session open: it clears the
        // loading note in the step that once set the state to generating.
        #expect(
            await waitUntil {
                await harness.synthesizer.primedVoices.count == 1
                    && !harness.presenter.isLoading
            })
        #expect(harness.coordinator.state == .idle)

        harness.coordinator.speakText("Hello again.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        #expect(await harness.synthesizer.requests.map(\.text) == ["Hello again."])
        #expect(harness.playback.startStreamingCount == 1)
        harness.playback.firePlaybackFinished()
    }

    /// A stop can also land while the engine admits the utterance, which
    /// loads the voice model again there after an unload. The stopped
    /// request once started the sink after the stop, as the request that
    /// stopped it was about to.
    @Test
    func aRequestStoppedWhileItsUtteranceIsAdmittedStaysStopped() async throws {
        let harness = await Harness()
        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.playback.firePlaybackFinished()

        // The next request is stopped mid-admission by the one after it.
        let coordinator = harness.coordinator
        await harness.synthesizer.onNextAudioFormat {
            await MainActor.run { coordinator.speakText("And again.") }
        }
        harness.coordinator.speakText("Hello again.")
        #expect(await waitUntil { harness.playback.finishStreamingCount == 2 })

        #expect(harness.playback.startStreamingCount == 2)
        harness.playback.firePlaybackFinished()
    }

    @Test
    func playbackSpeedAppliesAtTheStartAndWhenChanged() async throws {
        let harness = await Harness(script: .init(holdAfterChunks: 1))
        harness.settings.ttsPlaybackRate = 1.5

        harness.coordinator.speakText("Hello world.")
        #expect(await waitUntil { harness.playback.playbackRates == [1.5] })

        harness.coordinator.setPlaybackRate(2)
        #expect(harness.playback.playbackRates == [1.5, 2])
        #expect(harness.settings.ttsPlaybackRate == 2)
        harness.coordinator.stop()
    }

    /// Takes are compared side by side: each finished one is offered, and
    /// keeping an earlier take makes it the voice again.
    @Test
    func finishedTakesAreOfferedAndKeepingOneMakesItTheVoice() async throws {
        let harness = await Harness()
        harness.settings.ttsVoiceDescription = "warm narrator"
        let model = ModelDefinition.textToSpeechModelSpec
        let language = harness.settings.ttsLanguage

        harness.coordinator.tryAnotherTake(
            sampleFrom: "Here is how I sound. Every line keeps this voice.")
        #expect(await waitUntil { harness.coordinator.latestTake != nil })
        let first = try #require(harness.coordinator.latestTake)
        #expect(first.voice.voiceDescription == "warm narrator")
        #expect(first.samples.count == 3 * 1920 * 2)
        harness.playback.firePlaybackFinished()

        harness.coordinator.tryAnotherTake(
            sampleFrom: "Here is how I sound. Every line keeps this voice.")
        #expect(await waitUntil { harness.coordinator.latestTake?.id != first.id })
        harness.playback.firePlaybackFinished()

        await harness.coordinator.keep(first.voice)
        #expect(
            harness.pinnedVoices.voice(
                description: "warm narrator", language: language, model: model)
                == first.voice)
    }
}
