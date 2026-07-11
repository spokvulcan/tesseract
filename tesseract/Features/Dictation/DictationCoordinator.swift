//
//  DictationCoordinator.swift
//  tesseract
//

import AppKit
import Foundation
import Observation

/// The global system-wide dictation driver. A thin composer over the shared
/// **Voice Capture Session**: it maps the session's `StopResult`/`Outcome` onto
/// the **Overlay Feed**'s phases and beats and keeps only what is
/// dictation-specific — its commit (history + auto-insert text injection),
/// success/error sounds, the maximum-recording-duration auto-stop,
/// `DictationError` mapping, and the error auto-reset. Distinct from
/// **Voice Input** (`AgentVoiceInputController`), the agent composer leaf,
/// which composes the same session for its own presentation.
///
/// The coordinator is the feed's sole phase/beat writer; overlay variants and
/// the in-window dictation views read the feed, never the coordinator's
/// internals.
@Observable @MainActor
final class DictationCoordinator {
    let feed: DictationFeed
    private(set) var lastTranscription: String = ""

    /// The current lifecycle phase — a read-through to the feed, kept as the
    /// coordinator's public state surface for the in-window views and tests.
    var state: DictationFeed.Phase { feed.phase }

    private let session: VoiceCaptureSession
    private let textInjector: any TextInjecting
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    /// The maximum-recording-duration auto-stop. Caller-owned: it finalizes a stuck
    /// recording, which is a dictation-presentation concern, not part of the shared
    /// capture lifecycle.
    private var recordingTask: Task<Void, Never>?

    /// A hotkey press that arrived while a previous capture was still
    /// `.processing`. Push-to-talk must never swallow a press: the intent is
    /// honored the moment processing resolves — unless the key was released
    /// first, which abandons it (a tap fully inside the processing window has
    /// no audio to offer).
    private var startPending = false

    init(
        audioCapture: any AudioCapturing,
        transcriptionEngine: any Transcribing,
        textInjector: any TextInjecting,
        history: any TranscriptionStoring,
        settings: SettingsManager,
        feed: DictationFeed,
        captureDump: (any CaptureDumpStoring)? = nil
    ) {
        self.session = VoiceCaptureSession(
            audioCapture: audioCapture,
            transcriptionEngine: transcriptionEngine,
            captureDump: captureDump,
            isCaptureDumpEnabled: { settings.captureDumpEnabled }
        )
        self.textInjector = textInjector
        self.history = history
        self.settings = settings
        self.feed = feed
    }

    // MARK: - Public API

    func onHotkeyDown() {
        switch state {
        case .idle:
            DictationPerf.markPress()
            startRecording()
        case .error:
            // An error pill is feedback, never a gate: the press *is* the retry,
            // so recording starts immediately instead of waiting out the
            // error auto-reset.
            DictationPerf.markPress()
            feed.setPhase(.idle)
            startRecording()
        case .processing:
            startPending = true
        case .recording:
            break
        }
    }

    func onHotkeyUp() {
        startPending = false
        guard state == .recording else { return }
        stopRecordingAndProcess()
    }

    func toggleRecording() {
        switch state {
        case .idle:
            startRecording()
        case .recording:
            stopRecordingAndProcess()
        case .processing:
            // Can't stop while processing
            break
        case .error:
            // Reset and try again
            feed.setPhase(.idle)
            startRecording()
        }
    }

    func cancel() {
        startPending = false
        recordingTask?.cancel()
        recordingTask = nil
        session.cancel()
        feed.setPhase(.idle)
        feed.emit(.cancelled)
    }

    // MARK: - Private

    private func startRecording() {
        // Note (audit #285 item 6): the `.recording` emission lands *after*
        // the synchronous engine start below, but reordering would gain
        // nothing — emission and `AVAudioEngine.start()` complete inside one
        // main-actor job, so the pill's first frame can't precede either.
        // DictationPerf's press→visible measures the whole job.
        switch session.start() {
        case .started:
            feed.setPhase(.recording)

            // Start the maximum-duration timeout task.
            recordingTask = Task {
                let maxDuration = settings.maxRecordingDuration
                try? await Task.sleep(for: .seconds(maxDuration))

                if !Task.isCancelled && state == .recording {
                    stopRecordingAndProcess()
                }
            }

            if settings.playSounds {
                playSound(.startRecording)
            }
        case .micBusy:
            handleError(.microphoneBusy)
        case .captureFailed(let error):
            if let dictationError = error as? DictationError {
                handleError(dictationError)
            } else {
                handleError(.audioCaptureFailed(error.localizedDescription))
            }
        }
    }

    private func stopRecordingAndProcess() {
        recordingTask?.cancel()
        recordingTask = nil

        DictationPerf.markRelease()
        let stopStart = DispatchTime.now()
        let stopResult = session.stop()
        DictationPerf.record(span: "stop", ms: DictationPerf.msSince(stopStart))
        switch stopResult {
        case .noAudio:
            handleError(.audioCaptureFailed("No audio captured"))
            DictationPerf.markResolved("error(noAudio)")
        case .tooShort:
            handleError(.recordingTooShort)
            DictationPerf.markResolved("error(tooShort)")
        case .audio(let audioData):
            process(audioData)
        }
    }

    private func process(_ audioData: AudioData) {
        feed.setPhase(.processing)

        // Fire-and-forget: the session owns the in-flight task and its cancellation,
        // so this outer task is untracked — it only maps the outcome back to state.
        Task {
            let sessionStart = DispatchTime.now()
            var committedDuration: TimeInterval = 0
            let outcome = await session.transcribeAndCommit(
                audioData, language: settings.language
            ) { [self] text, duration in
                lastTranscription = text
                committedDuration = duration

                history.add(
                    text: text,
                    duration: duration,
                    model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                        ?? settings.selectedSpeechToTextModelID
                )

                if settings.autoInsertText {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    let injectStart = DispatchTime.now()
                    try await textInjector.inject(text + " ")
                    DictationPerf.record(
                        span: "inject", ms: DictationPerf.msSince(injectStart))
                }
            }
            DictationPerf.record(span: "session", ms: DictationPerf.msSince(sessionStart))

            switch outcome {
            case .committed:
                if settings.playSounds {
                    playSound(.success)
                }
                feed.setPhase(.idle)
                feed.emit(.committed(text: lastTranscription, duration: committedDuration))
                DictationPerf.markResolved("committed")
                drainPendingStart()
            case .empty:
                handleError(.noSpeechDetected)
                feed.emit(.empty)
                DictationPerf.markResolved("error(noSpeech)")
                drainPendingStart()
            case .failed(let error):
                if let dictationError = error as? DictationError {
                    handleError(dictationError)
                } else {
                    handleError(.transcriptionFailed(error.localizedDescription))
                }
                DictationPerf.markResolved("error(failed)")
                drainPendingStart()
            case .cancelled:
                feed.setPhase(.idle)
                feed.emit(.cancelled)
                DictationPerf.markResolved("cancelled")
            case .superseded:
                // A cancel-and-restart superseded this operation — the newer
                // operation owns the state; commit nothing and leave it untouched.
                feed.emit(.superseded)
                DictationPerf.markResolved("superseded")
            }
        }
    }

    /// Honors a hotkey press that arrived mid-`.processing`: the key is still
    /// held (release clears the flag), so recording starts now. An error the
    /// resolution just raised does not gate — same rule as a press on an idle
    /// error pill, and the new recording replaces it.
    private func drainPendingStart() {
        guard startPending else { return }
        startPending = false
        if case .error = state { feed.setPhase(.idle) }
        guard state == .idle else { return }
        startRecording()
    }

    private func handleError(_ error: DictationError) {
        feed.setPhase(.error(error))

        if settings.playSounds {
            playSound(.error)
        }

        // Auto-reset after a delay (shared duration so dictation and agent voice
        // input don't drift on how long an error lingers).
        Task {
            try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
            if case .error = state {
                feed.setPhase(.idle)
            }
        }
    }

    /// Preloaded once: `NSSound(named:)` loads from disk on first use, and the
    /// start-recording play sits on the same main-actor job that precedes the
    /// pill's state emission — a per-press load would delay the pill.
    private let sounds: [SystemSound: NSSound] = [
        SystemSound.startRecording: NSSound(named: "Tink"),
        SystemSound.success: NSSound(named: "Purr"),
        SystemSound.error: NSSound(named: "Funk"),
    ].compactMapValues { $0 }

    private func playSound(_ sound: SystemSound) {
        guard let nsSound = sounds[sound] else { return }
        if nsSound.isPlaying {
            nsSound.stop()
        }
        nsSound.play()
    }

    private enum SystemSound {
        case startRecording
        case success
        case error
    }
}
