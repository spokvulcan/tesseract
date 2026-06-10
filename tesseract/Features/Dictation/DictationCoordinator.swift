//
//  DictationCoordinator.swift
//  tesseract
//

import Foundation
import Observation
import AppKit

@Observable @MainActor
final class DictationCoordinator {
    private enum Defaults {
        static let minimumRecordingDuration: TimeInterval = 0.5
        static let errorAutoResetDelay: Duration = .seconds(3)
    }

    private(set) var state: DictationState = .idle
    private(set) var lastTranscription: String = ""
    private(set) var lastError: DictationError?

    private let audioCapture: any AudioCapturing
    private let transcriptionEngine: any Transcribing
    private let textInjector: any TextInjecting
    private let postProcessor: TranscriptionPostProcessor
    private let history: any TranscriptionStoring
    private let settings: SettingsManager

    private var recordingTask: Task<Void, Never>?
    private var recordingStartTime: Date?

    /// The in-flight transcribe→inject processing task. Tracked so `cancel()` can
    /// cancel it: the success path's `await textInjector.inject(...)` is itself the
    /// side effect, and is cancellation-aware (the real injector aborts before the
    /// paste). The post-await token guard alone can't stop an injection already
    /// suspended in flight — cancelling the task does.
    private var processingTask: Task<Void, Never>?

    /// The **Operation Guard** for this coordinator's dictation operations (a record →
    /// process attempt). `invalidate()`d when a new operation begins so a background
    /// transcription task that finishes *after* a cancel-and-restart recognizes it is
    /// stale (via its `OperationTicket`) and leaves the newer operation's state
    /// untouched. See CONTEXT.md → "Operation staleness".
    private let operations = OperationGuard()

    init(
        audioCapture: any AudioCapturing,
        transcriptionEngine: any Transcribing,
        textInjector: any TextInjecting,
        history: any TranscriptionStoring,
        settings: SettingsManager
    ) {
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.textInjector = textInjector
        self.postProcessor = TranscriptionPostProcessor()
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

        // Invalidate any in-flight processing so a transcription that completes
        // (or races cancellation to *success*) after this point is recognized as
        // stale and commits nothing — no history, no injection, no state change.
        operations.invalidate()

        // Cancel the processing task itself so a cancellation-aware side effect
        // suspended mid-flight (text injection) aborts rather than completing
        // after the user has cancelled. The token guard handles the complementary
        // case where the recognizer ignores cancellation and returns success.
        processingTask?.cancel()
        processingTask = nil

        if audioCapture.isCapturing {
            _ = audioCapture.stopCapture()
        }

        transcriptionEngine.cancelTranscription()
        state = .idle
    }

    // MARK: - Private

    private func startRecording() {
        lastError = nil
        // Advance the epoch at operation start: this coordinator can begin a new
        // recording while a prior transcription is still in flight (toggle/hotkey
        // restart) without cancelling it, so the start-bump is what supersedes that
        // overlapping prior operation.
        operations.invalidate()

        do {
            try audioCapture.startCapture()
            state = .recording
            recordingStartTime = Date()

            // Start timeout task
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
        } catch let error as DictationError {
            handleError(error)
        } catch {
            handleError(.audioCaptureFailed(error.localizedDescription))
        }
    }

    private func stopRecordingAndProcess() {
        recordingTask?.cancel()
        recordingTask = nil

        guard let audioData = audioCapture.stopCapture() else {
            handleError(.audioCaptureFailed("No audio captured"))
            return
        }

        // Check minimum duration
        if audioData.duration < Defaults.minimumRecordingDuration {
            handleError(.recordingTooShort)
            return
        }

        processAudio(audioData)
    }

    private func processAudio(_ audioData: AudioData) {
        state = .processing
        let ticket = operations.capture()

        processingTask = Task {
            do {
                // Transcribe with selected language
                let result = try await transcriptionEngine.transcribe(audioData, language: settings.language)

                // Stale-task guard: a cancel-and-restart since this operation
                // began means a newer operation owns the coordinator state — drop
                // this result instead of clobbering it (or injecting stale text).
                guard ticket.isCurrent else { return }

                // Post-process
                let processedText = postProcessor.process(result.text)

                guard !processedText.isEmpty else {
                    handleError(.noSpeechDetected)
                    return
                }

                lastTranscription = processedText

                // Add to history
                history.add(
                    text: processedText,
                    duration: audioData.duration,
                    model: ModelDefinition.withID(settings.selectedSpeechToTextModelID)?.displayName
                        ?? settings.selectedSpeechToTextModelID
                )

                // Inject text if enabled
                if settings.autoInsertText {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    try await textInjector.inject(processedText + " ")

                    // Injection suspends; a cancel-and-restart during it means a
                    // newer operation owns the state — don't play success or
                    // overwrite it.
                    guard ticket.isCurrent else { return }
                }

                if settings.playSounds {
                    playSound(.success)
                }

                state = .idle

            } catch is CancellationError {
                // Cancelled (e.g. `cancel()` while processing) — not a failure.
                // Only return to idle if this is still the current operation; a
                // newer recording must not be clobbered by this stale task.
                guard ticket.isCurrent else { return }
                state = .idle
            } catch let error as DictationError {
                guard ticket.isCurrent else { return }
                handleError(error)
            } catch {
                guard ticket.isCurrent else { return }
                handleError(.transcriptionFailed(error.localizedDescription))
            }
        }
    }

    private func handleError(_ error: DictationError) {
        lastError = error
        state = .error(error.localizedDescription)

        if settings.playSounds {
            playSound(.error)
        }

        // Auto-reset after a delay
        Task {
            try? await Task.sleep(for: Defaults.errorAutoResetDelay)
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
