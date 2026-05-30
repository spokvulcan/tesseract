//
//  SpeechCoordinatorTests.swift
//  tesseractTests
//
//  Exercises `SpeechCoordinator` driving the `AudioPlayback` seam: that
//  synthesized audio reaches playback (one-shot, single streaming, long-form),
//  that the streaming *diagnostics policy* it picks matches the path, that
//  voice-anchor sequencing fires for long-form, and that `stop()` tears playback
//  down. Composed over the *real* `SpeechEngine` with an
//  `InMemorySpeechSynthesizer` below it and an `InMemoryAudioPlayback` below the
//  coordinator. No model files, no AVAudioEngine, no `UserDefaults`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Hermetic peer doubles for the coordinator's collaborators

@MainActor
final class FakeTextExtractor: TextExtracting {
    var canned: String
    init(_ canned: String = "") { self.canned = canned }
    func extractSelectedText() async throws -> String { canned }
}

/// MainActor-isolated (hence `Sendable`) probe so a `@MainActor @Sendable`
/// success callback can record that it fired without a mutable-capture race.
@MainActor
final class CallbackProbe {
    private(set) var fireCount = 0
    func fire() { fireCount += 1 }
}

@MainActor
struct SpeechCoordinatorTests {

    // MARK: - Helpers

    private struct WaitTimedOut: Error {}

    /// Awaits an `@Observable`-driven condition by yielding (no wall-clock sleep).
    private func waitUntil(
        _ condition: () -> Bool,
        attempts: Int = 100_000,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async throws {
        var n = 0
        while !condition() {
            n += 1
            if n > attempts {
                Issue.record("condition not met within \(attempts) yields", sourceLocation: sourceLocation)
                throw WaitTimedOut()
            }
            await Task.yield()
        }
    }

    private func makeLoadedEngine(_ synth: InMemorySpeechSynthesizer) async throws -> SpeechEngine {
        let engine = SpeechEngine(makeSynthesizer: { synth })
        try await engine.loadModel()
        return engine
    }

    private func makeSettings(streaming: Bool) -> SettingsManager {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.ttsStreamingEnabled = streaming
        return settings
    }

    // MARK: - C1 (tracer): one-shot batch playback through the seam

    @Test
    func batchPlaybackReachesTheSeamAndFinishesIdleFiringSuccess() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.1, 0.2, 0.3], sampleRate: 24_000)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()
        let probe = CallbackProbe()

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: false),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText("hello") { probe.fire() }

        try await waitUntil { coordinator.state == .playing }
        #expect(playback.playCount == 1)
        #expect(playback.playedSamples == [[0.1, 0.2, 0.3]])
        #expect(playback.playedSampleRates == [24_000])
        #expect(probe.fireCount == 0)

        // The audio layer reports drain — the coordinator returns to idle and fires success.
        playback.firePlaybackFinished()
        #expect(coordinator.state == .idle)
        #expect(probe.fireCount == 1)
    }

    // MARK: - C2: single (non-long-form) streaming uses the default diagnostics policy

    @Test
    func singleStreamingSchedulesChunksWithDefaultDiagnosticsThenFinishesIdle() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.5, 0.6], sampleRate: 16_000)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: true),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText("short text")

        try await waitUntil { playback.finishStreamingCount == 1 }
        #expect(playback.startStreamingCount == 1)
        #expect(playback.recordedDiagnostics == [.default])
        #expect(playback.startedSampleRates == [16_000])
        #expect(playback.appendedChunks == [[0.5, 0.6]])
        #expect(coordinator.state == .streaming)

        playback.firePlaybackFinished()
        #expect(coordinator.state == .idle)
    }

    // MARK: - C3: long-form wiring — one session, diagnostics disabled, voice anchor, per-segment offsets

    @Test
    func longFormStreamsOneSessionWithDiagnosticsDisabledAndBuildsVoiceAnchor() async throws {
        // Many sentences → long-form, and segments into more than one segment.
        let longText = Array(repeating: "This sentence has exactly eight words in it.", count: 60)
            .joined(separator: " ")
        let segments = TextSegmenter.segment(longText)
        #expect(TextSegmenter.isLongForm(longText))
        #expect(segments.count >= 2)

        let synth = InMemorySpeechSynthesizer(samples: [0.7, 0.8], sampleRate: 22_050)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()
        // Keep the virtual playback head ahead of every segment boundary so the wait
        // loop is a no-op here — this test isolates the session/policy/anchor/offset
        // *wiring*; C5 covers the boundary wait itself.
        playback.advance(by: 10_000)

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: true),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText(longText)

        try await waitUntil { playback.finishStreamingCount == 1 }
        // One streaming session for the whole long-form run, diagnostics disabled.
        #expect(playback.startStreamingCount == 1)
        #expect(playback.recordedDiagnostics == [.disabled])
        #expect(playback.startedSampleRates == [22_050])
        // One chunk per segment (the synthesizer yields a single chunk each).
        #expect(playback.appendedChunks.count == segments.count)
        // Voice anchor built exactly once, from the first segment.
        #expect(await synth.buildVoiceAnchorCount == 1)
        // Token-char offsets computed once per segment, in order, for alignment.
        #expect(await synth.recordedOffsetTexts == segments.map(\.text))

        playback.firePlaybackFinished()
        #expect(coordinator.state == .idle)
    }

    // MARK: - C4: stop() tears playback down through the seam and returns to idle

    @Test
    func stopTearsDownPlaybackThroughTheSeamAndReturnsToIdle() async throws {
        let synth = InMemorySpeechSynthesizer(samples: [0.1, 0.2, 0.3], sampleRate: 24_000)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: false),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText("hello")
        try await waitUntil { coordinator.state == .playing }

        let stopsBefore = playback.stopCount
        coordinator.stop()

        #expect(playback.stopCount == stopsBefore + 1)
        #expect(coordinator.state == .idle)
    }

    // MARK: - C5: long-form actually waits below a segment boundary until the clock advances past it

    @Test
    func longFormBlocksAtSegmentBoundaryThenCrossesWhenVirtualClockAdvances() async throws {
        // Exactly two segments, and a first-segment chunk long enough (> 0.1s) that
        // the boundary the coordinator waits on is genuinely above zero.
        let longText = Array(repeating: "This sentence has exactly eight words in it.", count: 30)
            .joined(separator: " ")
        #expect(TextSegmenter.segment(longText).count == 2)

        let sampleRate = 22_050
        let samples = [Float](repeating: 0.3, count: sampleRate / 5)  // 0.2s per segment
        let synth = InMemorySpeechSynthesizer(samples: samples, sampleRate: sampleRate)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()  // clock at 0 — below segment 0's end

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: true),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText(longText)

        // Both segments' chunks get scheduled, then the coordinator blocks waiting for
        // the playback head to reach segment 0's end before it advances past segment 1.
        try await waitUntil { playback.appendedChunks.count == 2 }
        // Genuinely blocked below the boundary: it never finishes while the clock sits
        // at 0, no matter how much the test loop yields.
        for _ in 0..<1000 { await Task.yield() }
        #expect(playback.finishStreamingCount == 0)
        #expect(coordinator.state == .streamingLongForm(segment: 2, of: 2))

        // Advance the virtual playback head past the boundary — the wait loop exits.
        playback.advance(by: 1.0)
        try await waitUntil { playback.finishStreamingCount == 1 }
        #expect(playback.startStreamingCount == 1)

        playback.firePlaybackFinished()
        #expect(coordinator.state == .idle)
    }

    // MARK: - C6: pausing below a boundary halts long-form; resume continues from the next segment

    @Test
    func pauseBelowBoundaryHaltsLongFormAndResumeContinuesNextSegment() async throws {
        // Three segments so there is a "next segment" to resume into.
        let longText = Array(repeating: "This sentence has exactly eight words in it.", count: 50)
            .joined(separator: " ")
        #expect(TextSegmenter.segment(longText).count == 3)

        let sampleRate = 22_050
        let samples = [Float](repeating: 0.3, count: sampleRate / 5)  // 0.2s per segment
        let synth = InMemorySpeechSynthesizer(samples: samples, sampleRate: sampleRate)
        let engine = try await makeLoadedEngine(synth)
        let playback = InMemoryAudioPlayback()  // clock at 0 — keeps it parked in segment 1's wait

        let coordinator = SpeechCoordinator(
            textExtractor: FakeTextExtractor(),
            speechEngine: engine,
            playback: playback,
            settings: makeSettings(streaming: true),
            notchOverlay: nil,
            arbiter: nil
        )

        coordinator.speakText(longText)

        // Park in segment 1's boundary wait (segments 0 and 1 scheduled, clock below boundary).
        try await waitUntil { playback.appendedChunks.count == 2 }
        coordinator.pause()
        #expect(coordinator.state == .paused(segment: 2, of: 3))

        // Give the 50ms-poll wait loop time to observe the pause and unwind that task,
        // so resume() starts cleanly rather than racing a still-spinning generation.
        try await Task.sleep(for: .milliseconds(150))
        #expect(playback.finishStreamingCount == 0)
        #expect(playback.appendedChunks.count == 2)   // segment 2 was NOT generated
        #expect(playback.startStreamingCount == 1)

        // Resume picks up at the next segment as a fresh streaming session (its first
        // segment, so no boundary wait) and runs to completion without the clock moving.
        coordinator.resume()
        try await waitUntil { playback.finishStreamingCount == 1 }
        #expect(playback.startStreamingCount == 2)
        #expect(playback.appendedChunks.count == 3)

        playback.firePlaybackFinished()
        #expect(coordinator.state == .idle)
    }
}
