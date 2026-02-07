//
//  SpeechCoordinator.swift
//  tesseract
//

import Foundation
import Combine
import os

@MainActor
final class SpeechCoordinator: ObservableObject {
    @Published private(set) var state: SpeechState = .idle
    @Published private(set) var currentText: String = ""

    private let textExtractor: any TextExtracting
    private let speechEngine: SpeechEngine
    private let playbackManager: AudioPlaybackManager
    private let settings: SettingsManager

    private var activeTask: Task<Void, Never>?

    init(
        textExtractor: any TextExtracting,
        speechEngine: SpeechEngine,
        playbackManager: AudioPlaybackManager,
        settings: SettingsManager
    ) {
        self.textExtractor = textExtractor
        self.speechEngine = speechEngine
        self.playbackManager = playbackManager
        self.settings = settings

        playbackManager.onPlaybackFinished = { [weak self] in
            self?.state = .idle
        }
    }

    /// Called by TTS hotkey press
    func onHotkeyPressed() {
        if state != .idle {
            stop()
            return
        }

        activeTask = Task {
            await captureAndSpeak()
        }
    }

    /// Speak text directly (for in-app usage)
    func speakText(_ text: String) {
        guard !text.isEmpty else { return }

        stop()
        activeTask = Task {
            await generateAndPlay(text: text)
        }
    }

    func stop() {
        activeTask?.cancel()
        activeTask = nil
        playbackManager.stop()
        state = .idle
        currentText = ""
    }

    // MARK: - Private

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
            state = .error(error.localizedDescription)
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { state = .idle }
        }
    }

    private func generateAndPlay(text: String) async {
        do {
            // Load model if needed
            if !speechEngine.isModelLoaded {
                state = .loadingModel
                try await speechEngine.loadModel()
            }

            guard !Task.isCancelled else {
                state = .idle
                return
            }

            state = .generating(progress: "")

            let voiceDesc = settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription
            let language = settings.ttsLanguage

            let (samples, sampleRate) = try await speechEngine.generate(
                text: text,
                voice: voiceDesc,
                language: language,
                parameters: settings.ttsParameters
            )

            guard !Task.isCancelled else {
                state = .idle
                return
            }

            guard !samples.isEmpty else {
                state = .idle
                return
            }

            state = .playing
            playbackManager.play(samples: samples, sampleRate: sampleRate)
        } catch is CancellationError {
            playbackManager.stop()
            state = .idle
        } catch {
            Log.speech.error("Speech generation failed: \(error)")
            playbackManager.stop()
            state = .error(error.localizedDescription)
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { state = .idle }
        }
    }
}
