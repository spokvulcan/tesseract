//
//  AgentVoiceInputController.swift
//  tesseract
//
//  The **Voice Input** module: a field's push-to-talk capture→transcribe→emit
//  flow — the agent composer's, Today's, the capture and Jarvis panels', the
//  Onboarding Tour's. Distinct from `DictationCoordinator`, the global
//  system-wide dictation overlay — when unqualified this is "agent voice
//  input".
//
//  A thin presenter over a **Voice Capture Session** from **Voice Capture**,
//  so its takes run every stage (ADR-0096): it maps the session's outcomes
//  onto `AgentVoiceState` and supplies a commit that **emits** the text to the
//  surface via `onVoiceTranscription`; it does **not** send. Errors stay in
//  `voiceState`, never a shared banner. No sounds. No `Agent`, no arbiter.
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

    /// Called when a take ends without text (too short, no speech, a failed
    /// transcription), with the error line — so a surface waiting on the
    /// words can stop waiting.
    @ObservationIgnored var onVoiceFailure: ((String) -> Void)?

    /// Called instead of `onVoiceTranscription` for a take the Proofread Pass
    /// judged unintelligible, with its raw text — so a surface that acts on
    /// the words at once (saves them, sends them) can show them for checking
    /// first. Unset, the take is emitted like any other: the composer is an
    /// editable field.
    @ObservationIgnored var onVoiceRejected: ((String) -> Void)?

    // MARK: - Dependencies

    /// This surface's takes, with every stage.
    private let session: VoiceCaptureSession

    @ObservationIgnored private var voiceErrorResetTask: Task<Void, Never>?

    // MARK: - Init

    init(session: VoiceCaptureSession) {
        self.session = session
    }

    /// Spoken words join the words already in the field: after them, with one
    /// space between, or on their own when the field holds none.
    static func joining(_ spoken: String, to typed: String) -> String {
        let typed = typed.trimmingCharacters(in: .whitespacesAndNewlines)
        return typed.isEmpty ? spoken : typed + " " + spoken
    }

    // MARK: - Capture

    func start() {
        // An error is feedback, never a gate (same rule as dictation's hotkey):
        // starting again while one shows is the retry.
        if case .error = voiceState {
            voiceErrorResetTask?.cancel()
            voiceErrorResetTask = nil
            voiceState = .idle
        }
        guard voiceState == .idle else { return }

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

        switch session.stop() {
        case .noAudio, .tooShort:
            setVoiceError("Recording too short")
        case .silent:
            setVoiceError("Didn't hear anything")
        case .audio(let audioData, _):
            voiceState = .transcribing
            Log.agent.info(
                "Voice input stopped, transcribing \(String(format: "%.1f", audioData.duration))s audio"
            )

            // Fire-and-forget: the session owns the in-flight task and its
            // cancellation; this outer task only maps the outcome.
            Task {
                let outcome = await session.transcribeAndCommit(audioData) { [self] words in
                    Log.agent.info("Voice transcribed: \(words.text)")
                    voiceState = .idle
                    onVoiceTranscription?(words.text)
                }

                switch outcome {
                case .committed:
                    // The commit already set `.idle` and emitted the text.
                    break
                case .rejected(let raw, _):
                    // The composer is an editable field — a rejected take is
                    // still worth staging there; the user edits or clears it.
                    voiceState = .idle
                    if let onVoiceRejected {
                        onVoiceRejected(raw)
                    } else {
                        onVoiceTranscription?(raw)
                    }
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
        session.cancel()
        voiceState = .idle
        Log.agent.info("Voice input cancelled")
    }

    // MARK: - Private

    private func setVoiceError(_ message: String) {
        voiceState = .error(message)
        Log.agent.warning("Voice error: \(message)")
        onVoiceFailure?(message)

        voiceErrorResetTask?.cancel()
        voiceErrorResetTask = Task {
            try? await Task.sleep(for: ErrorAutoReset.delay)
            if case .error = voiceState {
                voiceState = .idle
            }
        }
    }
}
