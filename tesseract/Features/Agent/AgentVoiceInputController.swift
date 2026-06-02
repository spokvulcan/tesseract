//
//  AgentVoiceInputController.swift
//  tesseract
//
//  The **Voice Input** module: the agent composer's push-to-talk
//  capture→transcribe→emit flow, carved out of `AgentCoordinator`. Distinct from
//  `DictationCoordinator`, the global system-wide dictation overlay — when
//  unqualified this is "agent voice input".
//
//  `finishCapture()` stops recording, transcribes, and **emits** the text to the
//  composer via `onVoiceTranscription`; it does **not** send. (The pre-carve name
//  `stopVoiceInputAndSend` was a misnomer.) Errors stay in `voiceState`, never the
//  coordinator's shared `error` banner. No `Agent`, no arbiter.
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

    private let audioCapture: (any AudioCapturing)?
    private let transcriptionEngine: (any Transcribing)?
    private let settings: SettingsManager?
    private let postProcessor = TranscriptionPostProcessor()

    @ObservationIgnored private var voiceErrorResetTask: Task<Void, Never>?
    /// The **Operation Guard** for this controller's voice-input operations, so a
    /// background transcription task that completes after a cancel-and-restart
    /// recognizes it is stale (via its `OperationTicket`) and leaves the newer
    /// operation's state untouched. See CONTEXT.md → "Operation staleness".
    @ObservationIgnored private let operations = OperationGuard()

    private enum Defaults {
        nonisolated static let minimumRecordingDuration: TimeInterval = 0.5
        nonisolated static let errorAutoResetDelay: Duration = .seconds(3)
    }

    // MARK: - Init

    init(
        audioCapture: (any AudioCapturing)? = nil,
        transcriptionEngine: (any Transcribing)? = nil,
        settings: SettingsManager? = nil
    ) {
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.settings = settings
    }

    // MARK: - Capture

    func start() {
        guard voiceState == .idle else { return }
        guard let audioCapture else {
            setVoiceError("Voice input not available")
            return
        }
        guard !audioCapture.isCapturing else {
            setVoiceError("Microphone in use")
            return
        }

        do {
            // Bump for uniformity with DictationCoordinator + as defense-in-depth.
            // NOT load-bearing here: start() is `.idle`-gated (above), so — unlike
            // DictationCoordinator — no overlapping-restart path exists; any in-flight op
            // was already superseded by cancel()'s bump. See CONTEXT.md → "Operation staleness".
            operations.invalidate()
            try audioCapture.startCapture()
            voiceState = .recording
            Log.agent.info("Voice input started")
        } catch {
            setVoiceError("Mic error: \(error.localizedDescription)")
        }
    }

    /// Stops recording, transcribes, and emits the text to the composer via
    /// `onVoiceTranscription`. It does **not** send.
    func finishCapture() {
        guard voiceState == .recording else { return }
        guard let audioCapture, let transcriptionEngine else {
            cancel()
            return
        }

        let audioData = audioCapture.stopCapture()

        guard let audioData, audioData.duration >= Defaults.minimumRecordingDuration else {
            setVoiceError("Recording too short")
            return
        }

        voiceState = .transcribing
        Log.agent.info("Voice input stopped, transcribing \(String(format: "%.1f", audioData.duration))s audio")

        let ticket = operations.capture()
        Task {
            do {
                let language = settings?.language ?? "en"
                let result = try await transcriptionEngine.transcribe(audioData, language: language)

                // Stale-task guard: a cancel-and-restart since this operation began
                // means a newer voice input owns the state — drop this result.
                guard ticket.isCurrent else { return }

                let processedText = postProcessor.process(result.text)

                guard !processedText.isEmpty else {
                    setVoiceError("No speech detected")
                    return
                }

                Log.agent.info("Voice transcribed: \(processedText)")
                voiceState = .idle

                self.onVoiceTranscription?(processedText)
            } catch is CancellationError {
                // Cancelled (e.g. `cancel()` while transcribing) — not a failure.
                // Only return to idle if still the current operation; a newer
                // recording must not be clobbered by this stale task.
                guard ticket.isCurrent else { return }
                voiceState = .idle
                Log.agent.info("Voice transcription cancelled")
            } catch {
                guard ticket.isCurrent else { return }
                setVoiceError("Transcription failed")
                Log.agent.error("Voice transcription error: \(error)")
            }
        }
    }

    func cancel() {
        // Invalidate any in-flight transcription so a late success can't call
        // `onVoiceTranscription` (or overwrite state) after the user cancelled.
        operations.invalidate()
        if let audioCapture, audioCapture.isCapturing {
            _ = audioCapture.stopCapture()
        }
        transcriptionEngine?.cancelTranscription()
        voiceState = .idle
        Log.agent.info("Voice input cancelled")
    }

    // MARK: - Private

    private func setVoiceError(_ message: String) {
        voiceState = .error(message)
        Log.agent.warning("Voice error: \(message)")

        voiceErrorResetTask?.cancel()
        voiceErrorResetTask = Task {
            try? await Task.sleep(for: Defaults.errorAutoResetDelay)
            if case .error = voiceState {
                voiceState = .idle
            }
        }
    }
}
