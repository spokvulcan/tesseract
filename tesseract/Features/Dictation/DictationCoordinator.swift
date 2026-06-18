//
//  DictationCoordinator.swift
//  tesseract
//

import Foundation
import Observation
import AppKit

/// The global system-wide dictation overlay. A thin composer over the shared
/// **Voice Capture Session**: it maps the session's `StopResult`/`Outcome` onto
/// `DictationState` and keeps only what is dictation-specific — its commit
/// (history + auto-insert text injection), success/error sounds, the
/// maximum-recording-duration auto-stop, `DictationError` mapping, and the error
/// auto-reset. Distinct from **Voice Input** (`AgentVoiceInputController`), the
/// agent composer leaf, which composes the same session for its own presentation.
@Observable @MainActor
final class DictationCoordinator {
    private(set) var state: DictationState = .idle
    private(set) var lastTranscription: String = ""
    private(set) var lastError: DictationError?

    private let session: VoiceCaptureSession
    private let textInjector: any TextInjecting
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    /// The maximum-recording-duration auto-stop. Caller-owned: it finalizes a stuck
    /// recording, which is a dictation-presentation concern, not part of the shared
    /// capture lifecycle.
    private var recordingTask: Task<Void, Never>?

    init(
        audioCapture: any AudioCapturing,
        transcriptionEngine: any Transcribing,
        textInjector: any TextInjecting,
        history: any TranscriptionStoring,
        settings: SettingsManager
    ) {
        self.session = VoiceCaptureSession(
            audioCapture: audioCapture, transcriptionEngine: transcriptionEngine)
        self.textInjector = textInjector
        self.history = history
        self.settings = settings
    }

    // MARK: - Public API

    func onHotkeyDown() {
        guard state == .idle else { return }
        startRecording()
    }

    func onHotkeyUp() {
        guard state == .recording || state == .listening else { return }
        stopRecordingAndProcess()
    }

    func toggleRecording() {
        switch state {
        case .idle:
            startRecording()
        case .listening, .recording:
            stopRecordingAndProcess()
        case .processing:
            // Can't stop while processing
            break
        case .error:
            // Reset and try again
            state = .idle
            startRecording()
        }
    }

    func cancel() {
        recordingTask?.cancel()
        recordingTask = nil
        session.cancel()
        state = .idle
    }

    // MARK: - Private

    private func startRecording() {
        lastError = nil

        switch session.start() {
        case .started:
            state = .recording

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

        switch session.stop() {
        case .noAudio:
            handleError(.audioCaptureFailed("No audio captured"))
        case .tooShort:
            handleError(.recordingTooShort)
        case .audio(let audioData):
            process(audioData)
        }
    }

    private func process(_ audioData: AudioData) {
        state = .processing

        // Fire-and-forget: the session owns the in-flight task and its cancellation,
        // so this outer task is untracked — it only maps the outcome back to state.
        Task {
            let outcome = await session.transcribeAndCommit(
                audioData, language: settings.language
            ) { [self] text, duration in
                lastTranscription = text

                history.add(
                    text: text,
                    duration: duration,
                    model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                        ?? settings.selectedSpeechToTextModelID
                )

                if settings.autoInsertText {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    try await textInjector.inject(text + " ")
                }
            }

            switch outcome {
            case .committed:
                if settings.playSounds {
                    playSound(.success)
                }
                state = .idle
            case .empty:
                handleError(.noSpeechDetected)
            case .failed(let error):
                if let dictationError = error as? DictationError {
                    handleError(dictationError)
                } else {
                    handleError(.transcriptionFailed(error.localizedDescription))
                }
            case .cancelled:
                state = .idle
            case .superseded:
                // A cancel-and-restart superseded this operation — the newer
                // operation owns the state; commit nothing and leave it untouched.
                break
            }
        }
    }

    private func handleError(_ error: DictationError) {
        lastError = error
        state = .error(error.localizedDescription)

        if settings.playSounds {
            playSound(.error)
        }

        // Auto-reset after a delay (shared duration so dictation and agent voice
        // input don't drift on how long an error lingers).
        Task {
            try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
            if case .error = state {
                state = .idle
            }
        }
    }

    private func playSound(_ sound: SystemSound) {
        // Use NSSound for system sounds
        switch sound {
        case .startRecording:
            NSSound(named: "Tink")?.play()
        case .success:
            NSSound(named: "Purr")?.play()
        case .error:
            NSSound(named: "Funk")?.play()
        }
    }

    private enum SystemSound {
        case startRecording
        case success
        case error
    }
}
