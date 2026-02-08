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
    @Published private(set) var currentSegmentIndex: Int = 0
    @Published private(set) var totalSegments: Int = 0

    private let textExtractor: any TextExtracting
    private let speechEngine: SpeechEngine
    private let playbackManager: AudioPlaybackManager
    private let settings: SettingsManager
    private let prepareForSpeech: (@MainActor () async -> Void)?

    private enum Defaults {
        static let voiceAnchorTokenCount = 48
    }

    private var activeTask: Task<Void, Never>?
    private var isLongFormActive = false
    private var pausedSegmentIndex: Int?
    private var segments: [TextSegment] = []

    init(
        textExtractor: any TextExtracting,
        speechEngine: SpeechEngine,
        playbackManager: AudioPlaybackManager,
        settings: SettingsManager,
        prepareForSpeech: (@MainActor () async -> Void)? = nil
    ) {
        self.textExtractor = textExtractor
        self.speechEngine = speechEngine
        self.playbackManager = playbackManager
        self.settings = settings
        self.prepareForSpeech = prepareForSpeech

        playbackManager.onPlaybackFinished = { [weak self] in
            guard let self, !self.isLongFormActive else { return }
            self.state = .idle
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
        playbackManager.debugDumpDisabled = false
        isLongFormActive = false
        pausedSegmentIndex = nil
        segments = []
        currentSegmentIndex = 0
        totalSegments = 0
        state = .idle
        currentText = ""
        Task {
            await speechEngine.cancelGeneration()
            await speechEngine.clearVoiceAnchor()
        }
    }

    func pause() {
        guard isLongFormActive else { return }
        // Mark where to resume — current segment will finish generating,
        // then the loop checks pausedSegmentIndex and breaks
        pausedSegmentIndex = currentSegmentIndex
        state = .paused(segment: currentSegmentIndex + 1, of: totalSegments)
    }

    func resume() {
        guard let resumeIndex = pausedSegmentIndex, !segments.isEmpty else { return }
        pausedSegmentIndex = nil

        activeTask = Task {
            await generateAndPlayLongForm(startingAt: resumeIndex + 1)
        }
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
        if settings.ttsStreamingEnabled && TextSegmenter.isLongForm(text) {
            segments = TextSegmenter.segment(text)
            totalSegments = segments.count
            Log.speech.info("Long-form detected: \(segments.count) segments")
            await generateAndPlayLongForm(startingAt: 0)
        } else if settings.ttsStreamingEnabled {
            await generateAndPlayStreaming(text: text)
        } else {
            await generateAndPlayBatch(text: text)
        }
    }

    private func generateAndPlayLongForm(startingAt startIndex: Int) async {
        // Capture segments locally — stop() may clear the property from another call site
        let localSegments = segments
        let segmentCount = localSegments.count

        guard startIndex < segmentCount else { return }

        do {
            if let prepareForSpeech {
                await prepareForSpeech()
            }

            if !speechEngine.isModelLoaded {
                state = .loadingModel
                try await speechEngine.loadModel()
            }

            guard !Task.isCancelled else {
                state = .idle
                return
            }

            let voiceDesc = settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription
            let language = settings.ttsLanguage

            isLongFormActive = true
            playbackManager.debugDumpDisabled = true

            // First segment in this session — discover sample rate and start streaming
            state = .generating(progress: "Segment \(startIndex + 1) of \(segmentCount)")

            let (firstStream, sampleRate) = try await speechEngine.generateStreaming(
                text: localSegments[startIndex].text,
                voice: voiceDesc,
                language: language,
                parameters: settings.ttsParameters
            )

            playbackManager.startStreaming(sampleRate: sampleRate)
            currentSegmentIndex = startIndex
            state = .streamingLongForm(segment: startIndex + 1, of: segmentCount)

            for try await chunk in firstStream {
                guard !Task.isCancelled else {
                    cleanupLongForm()
                    return
                }
                playbackManager.appendChunk(samples: chunk)
            }

            Log.speech.info("Segment \(startIndex + 1)/\(segmentCount) complete")

            // Build voice anchor from first segment's generated codes
            if startIndex == 0 && segmentCount > 1 {
                Log.speech.info("Building voice anchor from segment 1")
                await speechEngine.buildVoiceAnchor(
                    referenceCount: Defaults.voiceAnchorTokenCount,
                    voice: voiceDesc,
                    language: language
                )
            }

            if pausedSegmentIndex != nil { return }

            // Continue with remaining segments (with voice anchor for consistency)
            for i in (startIndex + 1)..<segmentCount {
                guard !Task.isCancelled else {
                    cleanupLongForm()
                    return
                }

                if pausedSegmentIndex != nil { return }

                currentSegmentIndex = i
                state = .streamingLongForm(segment: i + 1, of: segmentCount)
                Log.speech.info("Starting segment \(i + 1)/\(segmentCount) (with voice anchor)")

                let (segStream, _) = try await speechEngine.generateStreaming(
                    text: localSegments[i].text,
                    voice: voiceDesc,
                    language: language,
                    parameters: settings.ttsParameters,
                    useVoiceAnchor: true
                )

                for try await chunk in segStream {
                    guard !Task.isCancelled else {
                        cleanupLongForm()
                        return
                    }
                    playbackManager.appendChunk(samples: chunk)
                }

                Log.speech.info("Segment \(i + 1)/\(segmentCount) complete")

                if pausedSegmentIndex != nil { return }
            }

            // All segments done
            playbackManager.finishStreaming()
            isLongFormActive = false
            // onPlaybackFinished callback will set state = .idle
        } catch is CancellationError {
            cleanupLongForm()
        } catch {
            Log.speech.error("Long-form generation failed: \(error)")
            cleanupLongForm()
            state = .error(error.localizedDescription)
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { state = .idle }
        }
    }

    private func cleanupLongForm() {
        playbackManager.stop()
        playbackManager.debugDumpDisabled = false
        isLongFormActive = false
        state = .idle
    }

    private func generateAndPlayBatch(text: String) async {
        do {
            if let prepareForSpeech {
                await prepareForSpeech()
            }

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

    private func generateAndPlayStreaming(text: String) async {
        do {
            if let prepareForSpeech {
                await prepareForSpeech()
            }

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

            let (stream, sampleRate) = try await speechEngine.generateStreaming(
                text: text,
                voice: voiceDesc,
                language: language,
                parameters: settings.ttsParameters
            )

            playbackManager.startStreaming(sampleRate: sampleRate)

            for try await chunk in stream {
                guard !Task.isCancelled else {
                    playbackManager.stop()
                    state = .idle
                    return
                }

                playbackManager.appendChunk(samples: chunk)

                // Transition to streaming on first chunk
                if state != .streaming {
                    state = .streaming
                }
            }

            playbackManager.finishStreaming()
            // onPlaybackFinished callback will set state = .idle
        } catch is CancellationError {
            playbackManager.stop()
            state = .idle
        } catch {
            Log.speech.error("Streaming speech generation failed: \(error)")
            playbackManager.stop()
            state = .error(error.localizedDescription)
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { state = .idle }
        }
    }
}
