//
//  AudioPlaybackTests.swift
//  tesseractTests
//
//  Light tests that the in-memory `AudioPlayback` honors its call-recording /
//  virtual-clock contract, so the coordinator tests built on it aren't a source of
//  false confidence. (The production `AudioPlaybackManager` drives a real
//  AVAudioEngine and isn't exercised here.)
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AudioPlaybackTests {

    @Test
    func recordsOneShotStreamingAndAppendsAndComputesScheduledDuration() {
        let playback = InMemoryAudioPlayback()

        playback.play(samples: [0.1, 0.2], sampleRate: 48_000)
        #expect(playback.playCount == 1)
        #expect(playback.playedSamples == [[0.1, 0.2]])
        #expect(playback.playedSampleRates == [48_000])

        playback.startStreaming(sampleRate: 24_000, diagnostics: .disabled)
        playback.appendChunk(samples: [0.0, 0.0, 0.0])
        playback.appendChunk(samples: [0.0])

        #expect(playback.startStreamingCount == 1)
        #expect(playback.startedSampleRates == [24_000])
        #expect(playback.recordedDiagnostics == [.disabled])
        #expect(playback.appendedChunks == [[0.0, 0.0, 0.0], [0.0]])
        // 4 scheduled samples at 24 kHz.
        #expect(playback.totalScheduledDuration == 4.0 / 24_000.0)

        playback.finishStreaming()
        #expect(playback.finishStreamingCount == 1)
    }

    @Test
    func virtualClockReadsZeroUntilAdvancedThenAccumulatesAndIgnoresLifecycle() {
        let playback = InMemoryAudioPlayback()
        #expect(playback.currentPlaybackTime() == 0)

        playback.advance(by: 1.5)
        #expect(playback.currentPlaybackTime() == 1.5)
        playback.advance(by: 2.0)
        #expect(playback.currentPlaybackTime() == 3.5)

        // The clock is purely test-controlled — lifecycle calls neither move nor reset it.
        playback.startStreaming(sampleRate: 16_000, diagnostics: .default)
        playback.appendChunk(samples: [0.0])
        playback.stop()
        #expect(playback.currentPlaybackTime() == 3.5)
    }

    @Test
    func firePlaybackFinishedInvokesTheInstalledCallback() {
        let playback = InMemoryAudioPlayback()
        let probe = CallbackProbe()
        playback.onPlaybackFinished = { probe.fire() }

        playback.firePlaybackFinished()
        playback.firePlaybackFinished()

        #expect(probe.fireCount == 2)
    }
}
