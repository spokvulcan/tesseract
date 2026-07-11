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
    private let playback: any AudioPlayback
    private let settings: SettingsManager
    private let notchOverlay: (any WordHighlightSurface)?
    private let arbiter: any InferenceArbitrating

    private enum Defaults {
        static let voiceAnchorTokenCount = 48
    }

    private var activeTask: Task<Void, Never>?
    private var isLongFormActive = false
    private var pausedSegmentIndex: Int?
    private var segments: [TextSegment] = []
    private var speechCompletionCallback: (@MainActor @Sendable () -> Void)?

    init(
        textExtractor: any TextExtracting,
        speechEngine: SpeechEngine,
        playback: any AudioPlayback = AudioPlaybackManager(),
        settings: SettingsManager,
        notchOverlay: (any WordHighlightSurface)? = nil,
        arbiter: any InferenceArbitrating
    ) {
        self.textExtractor = textExtractor
        self.speechEngine = speechEngine
        self.playback = playback
        self.settings = settings
        self.notchOverlay = notchOverlay
        self.arbiter = arbiter

        playback.onPlaybackFinished = { [weak self] in
            guard let self, !self.isLongFormActive else { return }
            self.state = .idle
            let callback = self.speechCompletionCallback
            self.speechCompletionCallback = nil
            callback?()
        }
    }

    /// Called by TTS hotkey press
    func onHotkeyPressed() {
        if state != .idle {
            stop()
            return
        }

        speechCompletionCallback = nil
        activeTask = Task {
            await captureAndSpeak()
        }
    }

    /// Speak text directly (for in-app usage)
    func speakText(_ text: String, onSuccess: (@MainActor @Sendable () -> Void)? = nil) {
        guard !text.isEmpty else { return }

        stop()
        speechCompletionCallback = onSuccess
        activeTask = Task {
            await generateAndPlay(text: text)
            // Only keep the callback alive if playback actually started.
            // onPlaybackFinished will fire it on successful completion.
            // Clear it for all other exits: cancellation, error, empty samples, etc.
            if state != .playing && !isLongFormActive {
                speechCompletionCallback = nil
            }
        }
    }

    func stop() {
        Log.speech.info("[Coordinator] stop() called — state=\(String(describing: self.state))")
        speechCompletionCallback = nil
        activeTask?.cancel()
        activeTask = nil
        playback.stop()
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
    private func withTTSReady<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        try await arbiter.withExclusiveGPU(.tts, body: body)
    }

    /// The per-request voice context derived from settings — the one home for
    /// the "empty voice description means no voice, never an empty prompt"
    /// rule every generate path shares.
    private var ttsVoiceContext: (voice: String?, language: String) {
        (
            settings.ttsVoiceDescription.isEmpty ? nil : settings.ttsVoiceDescription,
            settings.ttsLanguage
        )
    }

    /// The shared transient-error presentation: show the error, linger, then
    /// auto-reset to idle unless cancelled. Reads the same linger constant
    /// the dictation side single-sources, so the two families cannot drift.
    private func presentTransientError(_ message: String) async {
        state = .error(message)
        try? await Task.sleep(for: VoiceCaptureSession.errorAutoResetDelay)
        if !Task.isCancelled { state = .idle }
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
            await presentTransientError(error.localizedDescription)
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
                // All segments done — GPU no longer needed.
                // Clear isLongFormActive BEFORE finishStreaming() because
                // finishStreaming() may fire onPlaybackFinished synchronously
                // when all buffers have already drained.
                isLongFormActive = false
                playback.finishStreaming()
                notchOverlay?.markGenerationComplete()
                // onPlaybackFinished callback will set state = .idle
            }
        } catch is CancellationError {
            cleanupLongForm()
        } catch {
            Log.speech.error("Long-form generation failed: \(error)")
            cleanupLongForm()
            await presentTransientError(error.localizedDescription)
        }
    }

    /// Generates all long-form segments. Returns `true` if all segments completed,
    /// `false` if paused or cancelled mid-generation.
    private func generateLongFormSegments(
        _ localSegments: [TextSegment], startingAt startIndex: Int, count segmentCount: Int
    ) async throws -> Bool {
        let (voiceDesc, language) = ttsVoiceContext

        isLongFormActive = true

        let segmentPlayback = SegmentPlayback(playback: playback, surface: notchOverlay)
        let onState: (SpeechState) -> Void = { [weak self] in self?.state = $0 }
        let isPaused: () -> Bool = { [weak self] in self?.pausedSegmentIndex != nil }

        // First segment in this session — discover sample rate and start streaming
        state = .generating(progress: "Segment \(startIndex + 1) of \(segmentCount)")

        let (firstStream, sampleRate) = try await speechEngine.generateStreaming(
            text: localSegments[startIndex].text,
            voice: voiceDesc,
            language: language,
            parameters: settings.ttsParameters
        )

        playback.startStreaming(sampleRate: sampleRate, diagnostics: .disabled)
        currentSegmentIndex = startIndex
        state = .streamingLongForm(segment: startIndex + 1, of: segmentCount)

        // Show notch overlay with the first segment's text, then drain it.
        let firstOffsets = await speechEngine.computeTokenCharOffsets(
            text: localSegments[startIndex].text)
        notchOverlay?.show(
            text: localSegments[startIndex].text, tokenCharOffsets: firstOffsets,
            playbackTimeProvider: { [weak self] in
                self?.playback.currentPlaybackTime() ?? 0
            })

        let firstDrained = try await segmentPlayback.run(
            .first(text: localSegments[startIndex].text, tokenOffsets: firstOffsets),
            stream: firstStream,
            onState: onState,
            isPaused: isPaused
        )
        guard firstDrained else {
            if Task.isCancelled { cleanupLongForm() }
            return false
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

        if pausedSegmentIndex != nil { return false }

        // Continue with remaining segments (with voice anchor for consistency).
        for i in (startIndex + 1)..<segmentCount {
            if Task.isCancelled {
                cleanupLongForm()
                return false
            }
            if pausedSegmentIndex != nil { return false }

            currentSegmentIndex = i
            state = .streamingLongForm(segment: i + 1, of: segmentCount)
            Log.speech.info("Starting segment \(i + 1)/\(segmentCount) (with voice anchor)")

            // Record where the previous segment's audio ends in cumulative playback time;
            // the overlay stays on the previous segment until the head reaches this point.
            let prevSegEndDuration = playback.totalScheduledDuration

            // Start generation immediately for throughput (don't wait for playback).
            let segOffsets = await speechEngine.computeTokenCharOffsets(text: localSegments[i].text)
            let (segStream, _) = try await speechEngine.generateStreaming(
                text: localSegments[i].text,
                voice: voiceDesc,
                language: language,
                parameters: settings.ttsParameters,
                useVoiceAnchor: true
            )

            let drained = try await segmentPlayback.run(
                .next(
                    text: localSegments[i].text, tokenOffsets: segOffsets,
                    boundary: prevSegEndDuration),
                stream: segStream,
                onState: onState,
                isPaused: isPaused
            )
            guard drained else {
                if Task.isCancelled { cleanupLongForm() }
                return false
            }

            Log.speech.info("Segment \(i + 1)/\(segmentCount) complete")
            if pausedSegmentIndex != nil { return false }
        }

        return true
    }

    private func cleanupLongForm() {
        playback.stop()
        isLongFormActive = false
        notchOverlay?.dismiss()
        state = .idle
    }

    private func generateAndPlayBatch(text: String) async {
        do {
            let (voiceDesc, language) = ttsVoiceContext

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
            playback.play(samples: samples, sampleRate: sampleRate)

            let duration = Double(samples.count) / Double(sampleRate)
            let tokenOffsets = await speechEngine.computeTokenCharOffsets(text: text)
            notchOverlay?.show(
                text: text, tokenCharOffsets: tokenOffsets,
                playbackTimeProvider: { [weak self] in
                    self?.playback.currentPlaybackTime() ?? 0
                })
            notchOverlay?.updateTotalDuration(duration)
            notchOverlay?.markGenerationComplete()
        } catch is CancellationError {
            playback.stop()
            state = .idle
        } catch {
            Log.speech.error("Speech generation failed: \(error)")
            playback.stop()
            await presentTransientError(error.localizedDescription)
        }
    }

    private func generateAndPlayStreaming(text: String) async {
        do {
            let (voiceDesc, language) = ttsVoiceContext

            try await withTTSReady {
                try await self.streamingGenerationBody(
                    text: text, voiceDesc: voiceDesc, language: language
                )
            }

            // GPU no longer needed — finish playback
            playback.finishStreaming()
            notchOverlay?.updateTotalDuration(playback.totalScheduledDuration)
            notchOverlay?.markGenerationComplete()
            // onPlaybackFinished callback will set state = .idle
        } catch is CancellationError {
            playback.stop()
            state = .idle
        } catch {
            Log.speech.error("Streaming speech generation failed: \(error)")
            playback.stop()
            await presentTransientError(error.localizedDescription)
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

        playback.startStreaming(sampleRate: sampleRate, diagnostics: .default)
        let tokenOffsets = await speechEngine.computeTokenCharOffsets(text: text)
        notchOverlay?.show(
            text: text, tokenCharOffsets: tokenOffsets,
            playbackTimeProvider: { [weak self] in
                self?.playback.currentPlaybackTime() ?? 0
            })

        let segmentPlayback = SegmentPlayback(playback: playback, surface: notchOverlay)
        let drained = try await segmentPlayback.run(
            .single(text: text, tokenOffsets: tokenOffsets, firstChunkState: .streaming),
            stream: stream,
            onState: { [weak self] in self?.state = $0 },
            isPaused: { false }
        )

        if !drained {
            // Cancelled mid-stream — tear down inline, matching the previous behavior.
            playback.stop()
            notchOverlay?.dismiss()
            state = .idle
        }
    }
}
