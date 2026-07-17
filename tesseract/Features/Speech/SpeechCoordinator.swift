//
//  SpeechCoordinator.swift
//  tesseract
//
//  The presentation loop over the v2 speech engine (ADR-0038): open a session
//  for the settings voice, `speak`, and drain one typed event stream into
//  playback and the notch overlay. Segmentation, anchoring, pacing, GPU
//  leasing, and memory discipline all live behind the engine seam — v1's
//  six-responsibility, 432-line orchestration is deleted, not moved.
//
//  Pacing: the stream is the demand signal. After each segment lands we wait
//  until the scheduled-but-unplayed audio drops below a small window before
//  pulling the next event; the engine's `.lookahead` policy converts that
//  back-pressure into a lease-free park, so the GPU is free between bursts.
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

    private let textExtractor: any TextExtracting
    private let engine: SpeechEnginePresenter
    private let playback: any AudioPlayback
    private let settings: SettingsManager
    private let notchOverlay: (any WordHighlightSurface)?

    private enum Pacing {
        /// Pull the next segment once less than this much scheduled audio
        /// remains unplayed — enough runway that generation (RTF ~0.27)
        /// always wins the race, small enough that stop/pause discard little.
        static let bufferAheadSeconds: TimeInterval = 8
        static let pollInterval: Duration = .milliseconds(150)
    }

    /// Which sink an utterance plays through (Dual-Path Playback, ADR-0041):
    /// `.standard` is the dedicated playback engine; `.voiceSession` renders
    /// through the VPIO capture engine so echo cancellation hears the reply
    /// as its own far-end reference.
    enum PlaybackRoute {
        case standard
        case voiceSession
    }

    /// The voice-session sink (ADR-0041), installed by the composition root.
    /// `nil` (or a `.standard` route) keeps the dedicated engine.
    var voiceSessionPlayback: (any AudioPlayback)? {
        didSet { voiceSessionPlayback.map(wireFinished) }
    }

    /// The sink the current utterance plays through. Every playback touch
    /// goes through this so stop/pause/resume always hit the engine that is
    /// actually speaking.
    private var activeSink: any AudioPlayback

    private var activeTask: Task<Void, Never>?
    private var session: SpeechSession?
    private var sessionVoiceKey: String?
    private var isPaused = false
    private var speechCompletionCallback: (@MainActor @Sendable () -> Void)?
    private var fadeTask: Task<Void, Never>?
    /// The coordinator's bookkeeping of the sink volume it last commanded —
    /// the protocol deliberately has no getter (sinks reset themselves to
    /// 1.0 at stream boundaries; this mirrors that contract).
    private var sinkVolume: Float = 1.0

    init(
        textExtractor: any TextExtracting,
        engine: SpeechEnginePresenter,
        playback: any AudioPlayback = AudioPlaybackManager(),
        settings: SettingsManager,
        notchOverlay: (any WordHighlightSurface)? = nil
    ) {
        self.textExtractor = textExtractor
        self.engine = engine
        self.playback = playback
        self.settings = settings
        self.notchOverlay = notchOverlay
        self.activeSink = playback

        wireFinished(playback)
    }

    private func wireFinished(_ sink: any AudioPlayback) {
        sink.onPlaybackFinished = { [weak self] in
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
    func speakText(
        _ text: String, showsOverlay: Bool = true,
        route: PlaybackRoute = .standard,
        onSuccess: (@MainActor @Sendable () -> Void)? = nil
    ) {
        guard !text.isEmpty else { return }

        stop()
        activeSink = route == .voiceSession ? (voiceSessionPlayback ?? playback) : playback
        speechCompletionCallback = onSuccess
        // Claim the state synchronously: `stop()` above set `.idle`, and the
        // task below only reaches `.generating` after the session-open await.
        // The voice session's settled-engine watchdog polls this state — a
        // transient `.idle` here reopened the mic under live TTS (the
        // 2026-07-16 self-echo trace, ADR-0041).
        state = .generating(progress: "")
        activeTask = Task {
            await generateAndPlay(text: text, showsOverlay: showsOverlay)
        }
    }

    /// Cancelling the consuming task is the engine-side cancellation token
    /// (ADR-0038): generation stops within one decoder step.
    func stop() {
        Log.speech.info("[Coordinator] stop() called — state=\(String(describing: self.state))")
        speechCompletionCallback = nil
        activeTask?.cancel()
        activeTask = nil
        fadeTask?.cancel()
        fadeTask = nil
        sinkVolume = 1.0
        isPaused = false
        activeSink.stop()
        currentSegmentIndex = 0
        totalSegments = 0
        state = .idle
        currentText = ""
        notchOverlay?.dismiss()
    }

    /// Real pause: the player pauses instantly and the drain loop stops
    /// pulling, which parks the engine lease-free at the next segment
    /// boundary. (v1 could only finish the in-flight segment.)
    func pause() {
        guard !isPaused else { return }
        switch state {
        // `.generating` too: a voice-session barge-in can land before the
        // first audio chunk — the pause must hold that audio back (the sink
        // won't start a paused player) or TTS would talk over the take.
        case .streaming, .streamingLongForm, .playing, .generating: break
        default: return
        }
        isPaused = true
        activeSink.pause()
        state = .paused(segment: currentSegmentIndex + 1, of: max(totalSegments, 1))
    }

    func resume() {
        guard isPaused else { return }
        isPaused = false
        activeSink.resume()
        state =
            totalSegments > 1
            ? .streamingLongForm(segment: currentSegmentIndex + 1, of: totalSegments)
            : .streaming
    }

    // MARK: - Soft Barge surface (ADR-0041)

    /// The reply's loudness at the playback head — the Echo Floor's playback
    /// envelope input.
    func playbackLevelNow() -> Float {
        activeSink.playbackLevel()
    }

    /// Ramps the active sink's volume to `target` — the Soft Barge duck and
    /// its fade-back. Linear steps at ~16 ms; a new fade supersedes the one
    /// in flight; `stop()` cancels outright (the sink resets itself to 1.0).
    func fadePlayback(to target: Float, over duration: TimeInterval) {
        fadeTask?.cancel()
        let start = sinkVolume
        guard duration > 0, abs(target - start) > 0.001 else {
            sinkVolume = target
            activeSink.setVolume(target)
            return
        }
        let stepInterval: Duration = .milliseconds(16)
        let steps = max(1, Int(duration / 0.016))
        fadeTask = Task { [weak self] in
            for step in 1...steps {
                try? await Task.sleep(for: stepInterval)
                guard !Task.isCancelled, let self else { return }
                let fraction = Float(step) / Float(steps)
                let volume = start + (target - start) * fraction
                self.sinkVolume = volume
                self.activeSink.setVolume(volume)
            }
        }
    }

    // MARK: - Private

    /// The per-request voice context derived from settings — the one home for
    /// the "empty voice description means no voice, never an empty prompt"
    /// rule every generate path shares.
    private var ttsVoiceContext: (voice: String?, language: String) {
        (
            settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription,
            settings.ttsLanguage
        )
    }

    /// The shared transient-error presentation: show the error, linger, then
    /// auto-reset to idle unless cancelled. Reads the same linger constant
    /// the dictation side single-sources, so the two families cannot drift.
    private func presentTransientError(_ message: String) async {
        state = .error(message)
        try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
        if !Task.isCancelled { state = .idle }
    }

    private func captureAndSpeak() async {
        state = .capturingText

        do {
            let text = try await textExtractor.extractSelectedText()
            currentText = text
            await generateAndPlay(text: text)
        } catch is CancellationError {
            state = .idle
        } catch {
            Log.speech.error("Failed to capture text: \(error)")
            await presentTransientError(error.localizedDescription)
        }
    }

    /// A session binds the settings voice to cached model state; reopen only
    /// when the voice changes (the instruct prefix re-primes off the hot path).
    private func openOrReuseSession() async throws -> SpeechSession {
        let (voiceDescription, language) = ttsVoiceContext
        let key = "\(voiceDescription ?? "")|\(language)"
        if let session, sessionVoiceKey == key { return session }

        await session?.close()
        session = nil
        sessionVoiceKey = nil

        if !engine.isModelLoaded {
            engine.noteLoading("Loading voice model…")
        }
        do {
            let voice: Voice =
                voiceDescription.map { .designed(description: $0, language: language) }
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

    private func generateAndPlay(text: String, showsOverlay: Bool = true) async {
        // One resolution for the whole utterance: nil means audio-only, and
        // every overlay touch below no-ops.
        let overlay = showsOverlay ? notchOverlay : nil
        do {
            let session = try await openOrReuseSession()
            state = .generating(progress: "")

            let seed = UInt64(clamping: settings.ttsSeed)
            let utterance = try await session.speak(
                text,
                options: SpeechOptions(seed: .fixed(seed), parameters: settings.ttsParameters)
            )
            totalSegments = utterance.segmentCount
            activeSink.startStreaming(sampleRate: utterance.sampleRate)

            var overlayShown = false
            for try await event in utterance.events {
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
                    activeSink.appendChunk(samples: chunk.samples)

                case .segmentDone(let index):
                    overlay?.updateTotalDuration(activeSink.totalScheduledDuration)
                    Log.speech.info("Segment \(index + 1)/\(self.totalSegments) complete")
                    if index + 1 < utterance.segmentCount {
                        overlay?.markSegmentComplete()
                        // The demand signal: don't pull the next segment until
                        // playback needs it (or we're paused).
                        try await waitForPlaybackDemand()
                    }

                case .finished:
                    activeSink.finishStreaming()
                    overlay?.updateTotalDuration(activeSink.totalScheduledDuration)
                    overlay?.markGenerationComplete()
                // onPlaybackFinished advances state to .idle and fires
                // the completion callback once audio drains.
                }
            }
        } catch is CancellationError {
            // stop() already tore playback and overlay down.
            speechCompletionCallback = nil
            if state != .idle { state = .idle }
        } catch {
            Log.speech.error("Speech generation failed: \(error)")
            speechCompletionCallback = nil
            activeSink.stop()
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
                script.text,
                tokenCharOffsets: script.tokenCharOffsets,
                segmentBase: Double(script.startFrame) / framesPerSecond
            )
        } else {
            notchOverlay.show(
                text: script.text,
                tokenCharOffsets: script.tokenCharOffsets,
                playbackTimeProvider: { [weak self] in
                    self?.activeSink.currentPlaybackTime() ?? 0
                }
            )
            overlayShown = true
        }
    }

    private func waitForPlaybackDemand() async throws {
        while true {
            try Task.checkCancellation()
            if !isPaused {
                let ahead = activeSink.totalScheduledDuration - activeSink.currentPlaybackTime()
                if ahead < Pacing.bufferAheadSeconds { return }
            }
            try await Task.sleep(for: Pacing.pollInterval)
        }
    }
}
