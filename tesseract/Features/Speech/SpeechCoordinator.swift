//
//  SpeechCoordinator.swift
//  tesseract
//

import Foundation
import Observation
import os

@Observable @MainActor
final class SpeechCoordinator {
    private(set) var state: SpeechState = .idle
    private(set) var currentText: String = ""
    private(set) var currentSegmentIndex: Int = 0
    private(set) var totalSegments: Int = 0

    private let textExtractor: any TextExtracting
    private let speechEngine: SpeechEngine
    private let playbackManager: AudioPlaybackManager
    private let settings: SettingsManager
    private let notchOverlay: TTSNotchPanelController?
    private let arbiter: InferenceArbiter?

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
        notchOverlay: TTSNotchPanelController? = nil,
        arbiter: InferenceArbiter? = nil
    ) {
        self.textExtractor = textExtractor
        self.speechEngine = speechEngine
        self.playbackManager = playbackManager
        self.settings = settings
        self.notchOverlay = notchOverlay
        self.arbiter = arbiter

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
        Log.speech.info("[Coordinator] stop() called — state=\(String(describing: self.state))")
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
        notchOverlay?.dismiss()
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

    /// Wraps a TTS operation in the arbiter lease (GPU serialization + model loading).
    /// Falls back to manual model loading when no arbiter is configured.
    private func withTTSReady<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        if let arbiter {
            return try await arbiter.withExclusiveGPU(.tts, body: body)
        }
        if !speechEngine.isModelLoaded {
            state = .loadingModel
            try await speechEngine.loadModel()
        }
        try Task.checkCancellation()
        return try await body()
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
            let completed = try await withTTSReady {
                try await self.generateLongFormSegments(
                    localSegments, startingAt: startIndex, count: segmentCount
                )
            }

            if completed {
                // All segments done — GPU no longer needed
                playbackManager.finishStreaming()
                notchOverlay?.markGenerationComplete()
                isLongFormActive = false
                // onPlaybackFinished callback will set state = .idle
            }
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

    /// Generates all long-form segments. Returns `true` if all segments completed,
    /// `false` if paused or cancelled mid-generation.
    private func generateLongFormSegments(
        _ localSegments: [TextSegment], startingAt startIndex: Int, count segmentCount: Int
    ) async throws -> Bool {
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

        // Show notch overlay with the first segment's text
        let firstOffsets = await speechEngine.computeTokenCharOffsets(text: localSegments[startIndex].text)
        notchOverlay?.show(text: localSegments[startIndex].text, tokenCharOffsets: firstOffsets, playbackTimeProvider: { [weak self] in
            self?.playbackManager.currentPlaybackTime() ?? 0
        })

        for try await chunk in firstStream {
            guard !Task.isCancelled else {
                cleanupLongForm()
                return false
            }
            playbackManager.appendChunk(samples: chunk)
            notchOverlay?.updateTotalDuration(playbackManager.totalScheduledDuration)
        }

        // Align duration estimate so segment 1 highlighting converges to 100%
        notchOverlay?.markSegmentComplete()

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

        if pausedSegmentIndex != nil { return false }

        // Continue with remaining segments (with voice anchor for consistency)
        for i in (startIndex + 1)..<segmentCount {
            guard !Task.isCancelled else {
                cleanupLongForm()
                return false
            }

            if pausedSegmentIndex != nil { return false }

            currentSegmentIndex = i
            state = .streamingLongForm(segment: i + 1, of: segmentCount)
            Log.speech.info("Starting segment \(i + 1)/\(segmentCount) (with voice anchor)")

            // Record where previous segment's audio ends in cumulative playback time.
            // The overlay stays on the previous segment's text until playback reaches this point.
            let prevSegEndDuration = playbackManager.totalScheduledDuration

            // Start generation immediately for throughput (don't wait for playback)
            let segOffsets = await speechEngine.computeTokenCharOffsets(text: localSegments[i].text)

            let (segStream, _) = try await speechEngine.generateStreaming(
                text: localSegments[i].text,
                voice: voiceDesc,
                language: language,
                parameters: settings.ttsParameters,
                useVoiceAnchor: true
            )

            var overlayUpdated = false

            for try await chunk in segStream {
                guard !Task.isCancelled else {
                    cleanupLongForm()
                    return false
                }
                playbackManager.appendChunk(samples: chunk)

                // Switch overlay text when playback reaches the previous segment boundary
                if !overlayUpdated && playbackManager.currentPlaybackTime() >= prevSegEndDuration - 0.1 {
                    notchOverlay?.updateText(
                        localSegments[i].text,
                        tokenCharOffsets: segOffsets,
                        segmentTimeBase: prevSegEndDuration,
                        segmentDurationBase: prevSegEndDuration
                    )
                    overlayUpdated = true
                }

                // Only update duration tracking after overlay has switched to this segment
                // (otherwise cumulative duration would corrupt the previous segment's pacing)
                if overlayUpdated {
                    notchOverlay?.updateTotalDuration(playbackManager.totalScheduledDuration)
                }
            }

            // If generation finished before playback caught up, wait for the boundary
            if !overlayUpdated {
                Log.speech.info("Segment \(i + 1) generated, waiting for playback (prevEnd=\(String(format: "%.1f", prevSegEndDuration))s, playback=\(String(format: "%.1f", self.playbackManager.currentPlaybackTime()))s)")
                while playbackManager.currentPlaybackTime() < prevSegEndDuration - 0.1 {
                    guard !Task.isCancelled else {
                        cleanupLongForm()
                        return false
                    }
                    if pausedSegmentIndex != nil { return false }
                    try await Task.sleep(for: .milliseconds(50))
                }
                notchOverlay?.updateText(
                    localSegments[i].text,
                    tokenCharOffsets: segOffsets,
                    segmentTimeBase: prevSegEndDuration,
                    segmentDurationBase: prevSegEndDuration
                )
            }

            // Final duration update and mark segment generation complete
            notchOverlay?.updateTotalDuration(playbackManager.totalScheduledDuration)
            notchOverlay?.markSegmentComplete()

            Log.speech.info("Segment \(i + 1)/\(segmentCount) complete")

            if pausedSegmentIndex != nil { return false }
        }

        return true
    }

    private func cleanupLongForm() {
        playbackManager.stop()
        playbackManager.debugDumpDisabled = false
        isLongFormActive = false
        notchOverlay?.dismiss()
        state = .idle
    }

    private func generateAndPlayBatch(text: String) async {
        do {
            let voiceDesc = settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription
            let language = settings.ttsLanguage

            let (samples, sampleRate) = try await withTTSReady {
                self.state = .generating(progress: "")
                return try await self.speechEngine.generate(
                    text: text,
                    voice: voiceDesc,
                    language: language,
                    parameters: self.settings.ttsParameters
                )
            }

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

            let duration = Double(samples.count) / Double(sampleRate)
            let tokenOffsets = await speechEngine.computeTokenCharOffsets(text: text)
            notchOverlay?.show(text: text, tokenCharOffsets: tokenOffsets, playbackTimeProvider: { [weak self] in
                self?.playbackManager.currentPlaybackTime() ?? 0
            })
            notchOverlay?.updateTotalDuration(duration)
            notchOverlay?.markGenerationComplete()
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
            let voiceDesc = settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription
            let language = settings.ttsLanguage

            try await withTTSReady {
                try await self.streamingGenerationBody(
                    text: text, voiceDesc: voiceDesc, language: language
                )
            }

            // GPU no longer needed — finish playback
            playbackManager.finishStreaming()
            notchOverlay?.updateTotalDuration(playbackManager.totalScheduledDuration)
            notchOverlay?.markGenerationComplete()
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

    private func streamingGenerationBody(
        text: String, voiceDesc: String?, language: String?
    ) async throws {
        state = .generating(progress: "")

        let (stream, sampleRate) = try await speechEngine.generateStreaming(
            text: text,
            voice: voiceDesc,
            language: language,
            parameters: settings.ttsParameters
        )

        playbackManager.startStreaming(sampleRate: sampleRate)
        let tokenOffsets = await speechEngine.computeTokenCharOffsets(text: text)
        notchOverlay?.show(text: text, tokenCharOffsets: tokenOffsets, playbackTimeProvider: { [weak self] in
            self?.playbackManager.currentPlaybackTime() ?? 0
        })

        for try await chunk in stream {
            guard !Task.isCancelled else {
                playbackManager.stop()
                notchOverlay?.dismiss()
                state = .idle
                return
            }

            playbackManager.appendChunk(samples: chunk)
            notchOverlay?.updateTotalDuration(playbackManager.totalScheduledDuration)

            // Transition to streaming on first chunk
            if state != .streaming {
                state = .streaming
            }
        }
    }
}
