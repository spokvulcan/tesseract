//
//  AgentVoiceInputController.swift
//  tesseract
//
//  The **Voice Input** module: the agent composer's push-to-talk
//  capture→transcribe→emit flow, carved out of `AgentCoordinator`. Distinct from
//  `DictationCoordinator`, the global system-wide dictation overlay — when
//  unqualified this is "agent voice input".
//
//  A thin composer over the shared **Voice Capture Session**: it maps the session's
//  outcomes onto `AgentVoiceState` and supplies a commit that **emits** the text to
//  the composer via `onVoiceTranscription`; it does **not** send. (The pre-carve
//  name `stopVoiceInputAndSend` was a misnomer.) Errors stay in `voiceState`, never
//  the coordinator's shared `error` banner. No sounds. No `Agent`, no arbiter.
//

import Foundation
import Observation

@Observable @MainActor
final class AgentVoiceInputController {

    // MARK: - Observable State

    private(set) var voiceState: AgentVoiceState = .idle

    /// Called when voice transcription completes, to populate the input bar. The
    /// composer feeds the emitted text into Agent Run's `send` when the user submits.
    @ObservationIgnored var onVoiceTranscription: ((String) -> Void)?

    // MARK: - Dependencies

    /// The shared capture lifecycle. `nil` when capture/transcription dependencies
    /// were not supplied — voice input then fails safe (`start()` reports it
    /// unavailable) rather than half-working.
    private let session: VoiceCaptureSession?
    private let settings: SettingsManager?

    @ObservationIgnored private var voiceErrorResetTask: Task<Void, Never>?

    // MARK: - Init

    init(
        audioCapture: (any AudioCapturing)? = nil,
        transcriptionEngine: (any Transcribing)? = nil,
        settings: SettingsManager? = nil
    ) {
        if let audioCapture, let transcriptionEngine {
            self.session = VoiceCaptureSession(
                audioCapture: audioCapture, transcriptionEngine: transcriptionEngine)
        } else {
            self.session = nil
        }
        self.settings = settings
    }

    // MARK: - Capture

    func start() {
        guard voiceState == .idle else { return }
        guard let session else {
            setVoiceError("Voice input not available")
            return
        }

        switch session.start() {
        case .started:
            voiceState = .recording
            Log.agent.info("Voice input started")
        case .micBusy:
            setVoiceError("Microphone in use")
        case .captureFailed(let error):
            setVoiceError("Mic error: \(error.localizedDescription)")
        }
    }

    /// Stops recording, transcribes, and emits the text to the composer via
    /// `onVoiceTranscription`. It does **not** send.
    func finishCapture() {
        guard voiceState == .recording else { return }
        guard let session else {
            cancel()
            return
        }

        switch session.stop() {
        case .noAudio, .tooShort:
            setVoiceError("Recording too short")
        case .audio(let audioData):
            voiceState = .transcribing
            Log.agent.info(
                "Voice input stopped, transcribing \(String(format: "%.1f", audioData.duration))s audio"
            )

            // Fire-and-forget: the session owns the in-flight task and its
            // cancellation; this outer task only maps the outcome.
            Task {
                let outcome = await session.transcribeAndCommit(
                    audioData, language: settings?.language ?? "en"
                ) { [self] text, _ in
                    Log.agent.info("Voice transcribed: \(text)")
                    voiceState = .idle
                    onVoiceTranscription?(text)
                }

                switch outcome {
                case .committed:
                    // The commit already set `.idle` and emitted the text.
                    break
                case .empty:
                    setVoiceError("No speech detected")
                case .failed:
                    setVoiceError("Transcription failed")
                    Log.agent.error("Voice transcription failed")
                case .cancelled:
                    voiceState = .idle
                    Log.agent.info("Voice transcription cancelled")
                case .superseded:
                    // A newer voice input owns the state — leave it untouched.
                    break
                }
            }
        }
    }

    func cancel() {
        session?.cancel()
        voiceState = .idle
        Log.agent.info("Voice input cancelled")
    }

    // MARK: - Private

    private func setVoiceError(_ message: String) {
        voiceState = .error(message)
        Log.agent.warning("Voice error: \(message)")

        voiceErrorResetTask?.cancel()
        voiceErrorResetTask = Task {
            try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
            if case .error = voiceState {
                voiceState = .idle
            }
        }
    }
}
