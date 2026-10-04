//
//  SpeechCoordinator.swift
//  tesseract
//
//  The presentation loop over the v2 speech engine (ADR-0038): open a session
//  for the settings voice, `speak`, and drain one typed event stream into
//  playback and the notch overlay. Segmentation, anchoring, pacing and
//  memory discipline all live behind the engine seam — v1's
//  six-responsibility, 432-line orchestration is deleted, not moved.
//
//  Pacing: the stream is the demand signal. After each segment lands we wait
//  until the scheduled-but-unplayed audio drops below a small window before
//  pulling the next event; the engine's `.lookahead` policy converts that
//  back-pressure into a park, so the GPU is free between bursts.
//  Pause is real: the player pauses instantly and we simply stop pulling.
//

import Foundation
import Observation
import TesseractSpeech
import os

@Observable @MainActor
final class SpeechCoordinator {
    private(set) var state: SpeechState = .idle
    private(set) var currentText: String = ""
    private(set) var currentSegmentIndex: Int = 0
    private(set) var totalSegments: Int = 0

    /// The last "Try another take" that played to the end: its audio, to
    /// replay beside earlier takes, and the voice it made, to bring back with
    /// `keep(_:)`. A stopped retake leaves this alone, as it leaves the voice.
    private(set) var latestTake: VoiceTake?

    private let textExtractor: any TextExtracting
    private let engine: SpeechEnginePresenter
    /// The Voice Engine's download status, read before each request. The
    /// engine only loads a checkpoint already on disk, so speech waits for
    /// the Models page download instead of starting its own.
    private let voiceEngineStatus: @MainActor () -> ModelStatus
    private let playback: any AudioPlayback
    private let settings: any SpeechSettings
    private let notchOverlay: (any WordHighlightSurface)?
    /// Each designed voice's Reference Take, kept across relaunches so the
    /// voice stays the same person (ADR-0072).
    private let pinnedVoices: PinnedVoiceStore

    private enum Pacing {
        /// Pull the next segment once less than this much scheduled audio
        /// remains unplayed — enough runway that generation (RTF ~0.15)
        /// always wins the race, small enough that stop/pause discard little.
        static let bufferAheadSeconds: TimeInterval = 8
        static let pollInterval: Duration = .milliseconds(150)
    }

    /// Called when a request the user made (the hotkey, a Speak button) finds
    /// no Voice Engine on disk; the app opens the Models page. Speech the app
    /// starts on its own (auto-speak, the Companion) only shows the error.
    var onVoiceEngineMissing: (@MainActor () -> Void)?

    private var activeTask: Task<Void, Never>?
    private var session: SpeechSession?
    private var sessionVoiceKey: String?
    private var isPaused = false
    private var speechCompletionCallback: (@MainActor @Sendable () -> Void)?

    init(
        textExtractor: any TextExtracting,
        engine: SpeechEnginePresenter,
        voiceEngineStatus: @escaping @MainActor () -> ModelStatus,
        playback: any AudioPlayback = AudioPlaybackManager(),
        settings: any SpeechSettings,
        notchOverlay: (any WordHighlightSurface)? = nil,
        pinnedVoices: PinnedVoiceStore = PinnedVoiceStore()
    ) {
        self.textExtractor = textExtractor
        self.engine = engine
        self.voiceEngineStatus = voiceEngineStatus
        self.playback = playback
        self.settings = settings
        self.notchOverlay = notchOverlay
        self.pinnedVoices = pinnedVoices

        playback.onPlaybackFinished = { [weak self] in
            guard let self else { return }
            self.state = .idle
            let callback = self.speechCompletionCallback
            self.speechCompletionCallback = nil
            callback?()
        }
    }

    /// Called by TTS hotkey press
    func onHotkeyPressed() {
        if state != .idle {
            stop()
            return
        }

        speechCompletionCallback = nil
        // Claim the state *before* the task's first await — a watcher polling
        // `state` must never read `.idle` while an utterance is in flight.
        state = .capturingText
        activeTask = Task {
            await captureAndSpeak()
        }
    }

    /// Speak text directly (for in-app usage). `showsOverlay: false` plays
    /// audio-only — for callers that bring their own visual surface (the
    /// Companion voice overlay, #328) and must not raise the TTS notch too.
    /// `userInitiated`: the user asked for this speech (a Speak button), so a
    /// missing Voice Engine sends them to the Models page. `retake`: speak
    /// only the opening as a new Reference Take (see `tryAnotherTake`).
    /// `language`: the text's own (a `TTSLanguage` raw value), when the caller
    /// knows it; otherwise the setting's.
    func speakText(
        _ text: String, showsOverlay: Bool = true,
        userInitiated: Bool = false,
        retake: Bool = false,
        language: String? = nil,
        onSuccess: (@MainActor @Sendable () -> Void)? = nil
    ) {
        guard !text.isEmpty else { return }

        stop()
        speechCompletionCallback = onSuccess
        // Claim the state synchronously: `stop()` above set `.idle`, and the
        // task below only reaches `.generating` after the session-open await.
        // The voice session's settled-engine watchdog polls this state — a
        // transient `.idle` here reads as a finished reply, which stops it
        // and reopens the mic (the 2026-07-16 trace).
        state = .generating(progress: "")
        activeTask = Task {
            guard await voiceEngineReady(userInitiated: userInitiated) else { return }
            await generateAndPlay(
                text: text, showsOverlay: showsOverlay, userInitiated: userInitiated,
                retake: retake, language: language)
        }
    }

    /// "Try another take": render the settings voice again from the opening
    /// of `text` (its first sentence or two, or a built-in sample when
    /// empty) with a fresh seed, play it, and keep it as the voice from now
    /// on (ADR-0072). Stopping it early keeps the previous take.
    func tryAnotherTake(sampleFrom text: String) {
        let sample =
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? Self.takeSample : text
        speakText(sample, userInitiated: true, retake: true)
    }

    /// What a new take reads when the composer is empty: two sentences with
    /// some range, long enough to carry the voice.
    private static let takeSample =
        "Here is how I sound when I read to you. "
        + "A story, an article, a long email: every line keeps this same voice, from the first word to the last."

    /// Keeps `voice`, an earlier take of its designed voice, from now on:
    /// stores it and drops the open session, so the next utterance opens
    /// pinned to it (ADR-0072).
    func keep(_ voice: PinnedVoice) async {
        stop()
        pinnedVoices.save(voice)
        await session?.close()
        session = nil
        sessionVoiceKey = nil
    }

    /// The take a designed voice speaks with now, if it has one.
    func pinnedVoice(description: String, language: String) -> PinnedVoice? {
        pinnedVoices.voice(
            description: description.isEmpty ? nil : description, language: language,
            model: ModelDefinition.textToSpeechModelSpec)
    }

    /// Read-aloud speed (the setting), applied at once to whatever is
    /// speaking — voice-session replies included.
    func setPlaybackRate(_ rate: Double) {
        settings.ttsPlaybackRate = rate
        playback.setPlaybackRate(Float(rate))
    }

    /// Cancelling the consuming task is the engine-side cancellation token
    /// (ADR-0038): generation stops within one decoder step.
    func stop() {
        Log.speech.info("[Coordinator] stop() called — state=\(String(describing: self.state))")
        speechCompletionCallback = nil
        activeTask?.cancel()
        activeTask = nil
        isPaused = false
        playback.stop()
        currentSegmentIndex = 0
        totalSegments = 0
        state = .idle
        currentText = ""
        notchOverlay?.dismiss()
    }

    /// Real pause: the player pauses instantly and the drain loop stops
    /// pulling, which parks the engine at the next segment
    /// boundary. (v1 could only finish the in-flight segment.)
    func pause() {
        guard !isPaused else { return }
        switch state {
        // `.generating` too: a pause can land before the first audio
        // chunk — it must hold that audio back (the sink won't start a
        // paused player).
        case .streaming, .streamingLongForm, .playing, .generating: break
        default: return
        }
        isPaused = true
        playback.pause()
        state = .paused(segment: currentSegmentIndex + 1, of: max(totalSegments, 1))
    }

    func resume() {
        guard isPaused else { return }
        isPaused = false
        playback.resume()
        state =
            totalSegments > 1
            ? .streamingLongForm(segment: currentSegmentIndex + 1, of: totalSegments)
            : .streaming
    }

    // MARK: - Private

    /// The per-request voice context derived from settings — the one home for
    /// the "empty voice description means no voice, never an empty prompt"
    /// rule every generate path shares. `language` overrides the setting's.
    private func ttsVoiceContext(language: String? = nil) -> (voice: String?, language: String) {
        (
            settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription,
            language ?? settings.ttsLanguage
        )
    }

    /// The shared transient-error presentation: show the error, linger, then
    /// auto-reset to idle unless cancelled. Lingers as long as dictation's
    /// errors do (`ErrorAutoReset`).
    private func presentTransientError(_ message: String) async {
        state = .error(message)
        try? await Task.sleep(for: ErrorAutoReset.delay)
        if !Task.isCancelled { state = .idle }
    }

    private func captureAndSpeak() async {
        state = .capturingText
        // Before the capture: without a Voice Engine there's no reason to
        // touch the user's selection.
        guard await voiceEngineReady(userInitiated: true) else { return }

        do {
            let text = try await textExtractor.extractSelectedText()
            currentText = text
            await generateAndPlay(text: text, userInitiated: true)
        } catch  where Task.isCancelled {
            // stop() set the state; a newer request may own it now.
        } catch is CancellationError {
            state = .idle
        } catch {
            Log.speech.error("Failed to capture text: \(error)")
            await presentTransientError(error.localizedDescription)
        }
    }

    /// Whether to go ahead with a request. Without the Voice Engine on disk
    /// the request stops here and says why.
    private func voiceEngineReady(userInitiated: Bool) async -> Bool {
        let message: String
        switch voiceEngineStatus() {
        case .downloaded, .error:
            // `.error` is a failed download or verify, not proof the files
            // are gone. The engine checks the disk itself before loading.
            return true
        case .downloading(let progress):
            message = "The Voice Engine is still downloading (\(Int(progress * 100))%)."
        case .verifying:
            message = "The Voice Engine is being verified. Try again in a moment."
        case .notDownloaded:
            await presentVoiceEngineMissing(userInitiated: userInitiated)
            return false
        }
        speechCompletionCallback = nil
        await presentTransientError(message)
        return false
    }

    private func presentVoiceEngineMissing(userInitiated: Bool) async {
        Log.speech.info(
            "Voice Engine not downloaded; speech request dropped (userInitiated=\(userInitiated))")
        speechCompletionCallback = nil
        if userInitiated { onVoiceEngineMissing?() }
        await presentTransientError("Download the Voice Engine in Models to hear speech.")
    }

    /// A session binds the settings voice to cached model state; reopen only
    /// when the voice changes (the instruct prefix re-primes off the hot path).
    /// A voice with a stored Reference Take opens pinned to it; otherwise the
    /// session's first segment becomes its take.
    private func openOrReuseSession(language: String? = nil) async throws -> SpeechSession {
        let (voiceDescription, language) = ttsVoiceContext(language: language)
        let preset = settings.presetVoice
        let key = "\(preset ?? voiceDescription ?? "")|\(language)"
        if let session, sessionVoiceKey == key { return session }

        await session?.close()
        session = nil
        sessionVoiceKey = nil

        if !engine.isModelLoaded {
            engine.noteLoading("Loading voice model…")
        }
        do {
            // A Preset Voice is the checkpoint's own speaker: nothing to pin.
            let pinned =
                preset == nil
                ? pinnedVoices.voice(
                    description: voiceDescription, language: language,
                    model: ModelDefinition.textToSpeechModelSpec)
                : nil
            let voice: Voice =
                preset.map { .preset(speaker: $0, language: language) }
                ?? pinned.map { .pinned($0) }
                ?? voiceDescription.map { .designed(description: $0, language: language) }
                ?? .standard(language: language)
            let opened = try await engine.engine.session(.readAloud, voice: voice)
            engine.noteReady()
            session = opened
            sessionVoiceKey = key
            return opened
        } catch {
            engine.noteFailed()
            throw error
        }
    }

    private func generateAndPlay(
        text: String, showsOverlay: Bool = true, userInitiated: Bool, retake: Bool = false,
        language: String? = nil
    ) async {
        // One resolution for the whole utterance: nil means audio-only, and
        // every overlay touch below no-ops.
        let overlay = showsOverlay ? notchOverlay : nil
        do {
            // Neither opening the session nor admitting the utterance ends at a
            // stop(), and either can spend seconds loading the voice model, so
            // check after each. The opened session stays for the next request;
            // the dropped utterance stops its generation.
            let session = try await openOrReuseSession(language: language)
            try Task.checkCancellation()
            state = .generating(progress: "")

            // A retake needs a fresh seed: the settings seed would render the
            // same take again.
            let options = SpeechOptions(
                seed: retake ? .entropy : .fixed(UInt64(clamping: settings.ttsSeed)),
                parameters: settings.ttsParameters)
            let utterance =
                retake
                ? try await session.retake(text, options: options)
                : try await session.speak(text, options: options)
            try Task.checkCancellation()
            totalSegments = utterance.segmentCount
            playback.startStreaming(sampleRate: utterance.sampleRate)
            playback.setPlaybackRate(Float(settings.ttsPlaybackRate))

            // A retake is one short segment: keep its audio for replay.
            var takeSamples: [Float] = []
            var overlayShown = false
            for try await event in utterance.events {
                // stop() cancels this task, but the stream still hands over
                // events the engine sent before it; by then the next request
                // owns the sink, the overlay and the state.
                try Task.checkCancellation()
                switch event {
                case .segment(let script):
                    currentSegmentIndex = script.index
                    state =
                        utterance.segmentCount > 1
                        ? .streamingLongForm(
                            segment: script.index + 1, of: utterance.segmentCount)
                        : .streaming
                    presentScript(
                        script, on: overlay, framesPerSecond: utterance.framesPerSecond,
                        overlayShown: &overlayShown)

                case .audio(let chunk):
                    playback.appendChunk(samples: chunk.samples)
                    if retake { takeSamples.append(contentsOf: chunk.samples) }

                case .words(let timing):
                    // Frames over the utterance, as the Read-Along's clock counts.
                    overlay?.timeWords(
                        timing.starts.map {
                            TimedWord(
                                word: $0.word,
                                start: Double($0.frame) / utterance.framesPerSecond)
                        },
                        segment: timing.segmentIndex)

                case .segmentDone(let index):
                    await rememberVoice(of: session)
                    try Task.checkCancellation()
                    overlay?.updateTotalDuration(playback.totalScheduledDuration)
                    Log.speech.info("Segment \(index + 1)/\(self.totalSegments) complete")
                    if index + 1 < utterance.segmentCount {
                        overlay?.markSegmentComplete()
                        // The demand signal: don't pull the next segment until
                        // playback needs it (or we're paused).
                        try await waitForPlaybackDemand()
                    }

                case .finished:
                    playback.finishStreaming()
                    overlay?.updateTotalDuration(playback.totalScheduledDuration)
                    overlay?.markGenerationComplete()
                    if retake, let voice = await session.exportPinnedVoice() {
                        latestTake = VoiceTake(
                            voice: voice, samples: takeSamples, sampleRate: utterance.sampleRate)
                    }
                // onPlaybackFinished advances state to .idle and fires
                // the completion callback once audio drains.
                }
            }
        } catch  where Task.isCancelled {
            // stop() cancelled this request and already tore playback and the
            // overlay down. A newer request may own the state, callback,
            // sink and overlay by now: resetting them here ended its reading.
        } catch is CancellationError {
            // Cancelled inside the engine, with no stop().
            speechCompletionCallback = nil
            if state != .idle { state = .idle }
        } catch SpeechEngineError.modelUnavailable(let detail) {
            // The catalog said downloaded, but the engine's disk check found
            // the checkpoint gone or partial.
            Log.speech.error("Voice Engine unavailable: \(detail)")
            playback.stop()
            notchOverlay?.dismiss()
            await presentVoiceEngineMissing(userInitiated: userInitiated)
        } catch {
            Log.speech.error("Speech generation failed: \(error)")
            speechCompletionCallback = nil
            playback.stop()
            notchOverlay?.dismiss()
            await presentTransientError(error.localizedDescription)
        }
    }

    /// Segment Windows arrive as data (`startFrame` is ground truth): the
    /// overlay switches exactly when the playback head crosses the boundary.
    private func presentScript(
        _ script: SegmentScript, on notchOverlay: (any WordHighlightSurface)?,
        framesPerSecond: Double, overlayShown: inout Bool
    ) {
        guard let notchOverlay else { return }
        if overlayShown {
            notchOverlay.switchText(
                script.text, segmentBase: Double(script.startFrame) / framesPerSecond)
        } else {
            notchOverlay.show(
                text: script.text,
                // What is heard, not what is rendering: over Bluetooth the
                // two are 150 ms or more apart (ADR-0077).
                playbackTimeProvider: { [weak self] in
                    self?.playback.heardPlaybackTime() ?? 0
                }
            )
            overlayShown = true
        }
    }

    /// Stores the session's Reference Take once it has one, so the next launch
    /// opens the same voice. Runs after every segment: the engine decides
    /// when a take forms, and the store skips a take it already holds.
    private func rememberVoice(of session: SpeechSession) async {
        guard let pinned = await session.exportPinnedVoice() else { return }
        pinnedVoices.save(pinned)
    }

    private func waitForPlaybackDemand() async throws {
        while true {
            try Task.checkCancellation()
            if !isPaused {
                let ahead = playback.totalScheduledDuration - playback.currentPlaybackTime()
                if ahead < Pacing.bufferAheadSeconds { return }
            }
            try await Task.sleep(for: Pacing.pollInterval)
        }
    }
}

/// One finished "Try another take": the voice it made and what it sounded
/// like (a sentence or two, so its samples are small).
struct VoiceTake: Identifiable, Sendable {
    let id = UUID()
    let voice: PinnedVoice
    let samples: [Float]
    let sampleRate: Int

    var duration: TimeInterval { Double(samples.count) / Double(max(sampleRate, 1)) }
}
