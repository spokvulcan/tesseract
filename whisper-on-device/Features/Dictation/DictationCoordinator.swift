//
//  DictationCoordinator.swift
//  whisper-on-device
//

import Foundation
import Combine
import AppKit

@MainActor
final class DictationCoordinator: ObservableObject {
    @Published private(set) var state: DictationState = .idle
    @Published private(set) var lastTranscription: String = ""
    @Published private(set) var lastError: DictationError?

    private let audioCapture: AudioCaptureEngine
    private let transcriptionEngine: TranscriptionEngine
    private let textInjector: TextInjector
    private let postProcessor: TranscriptionPostProcessor
    private let history: TranscriptionHistory
    private let settings: SettingsManager

    private var recordingTask: Task<Void, Never>?
    private var recordingStartTime: Date?

    init(
        audioCapture: AudioCaptureEngine,
        transcriptionEngine: TranscriptionEngine,
        textInjector: TextInjector,
        history: TranscriptionHistory,
        settings: SettingsManager = .shared
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

        if audioCapture.isCapturing {
            _ = audioCapture.stopCapture()
        }

        transcriptionEngine.cancelTranscription()
        state = .idle
    }

    // MARK: - Private

    private func startRecording() {
        lastError = nil

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
        if audioData.duration < 0.5 {
            handleError(.recordingTooShort)
            return
        }

        processAudio(audioData)
    }

    private func processAudio(_ audioData: AudioData) {
        state = .processing

        Task {
            do {
                // Transcribe
                let result = try await transcriptionEngine.transcribe(audioData)

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
                    model: WhisperModel.displayName
                )

                // Inject text if enabled
                if settings.autoInsertText {
                    textInjector.restoreClipboard = settings.restoreClipboard
                    try await textInjector.inject(processedText)
                }

                if settings.playSounds {
                    playSound(.success)
                }

                state = .idle

            } catch let error as DictationError {
                handleError(error)
            } catch {
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
            try? await Task.sleep(for: .seconds(3))
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
