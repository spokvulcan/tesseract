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

        playback.startStreaming(sampleRate: 24_000)
        playback.appendChunk(samples: [0.0, 0.0, 0.0])
        playback.appendChunk(samples: [0.0])

        #expect(playback.startStreamingCount == 1)
        #expect(playback.startedSampleRates == [24_000])
        #expect(playback.appendedChunks == [[0.0, 0.0, 0.0], [0.0]])
        // 4 scheduled samples at 24 kHz.
        #expect(playback.totalScheduledDuration == 4.0 / 24_000.0)

        playback.finishStreaming()
        #expect(playback.finishStreamingCount == 1)

        playback.pause()
        #expect(playback.isPaused)
        playback.resume()
        #expect(!playback.isPaused)
        #expect(playback.pauseCount == 1)
        #expect(playback.resumeCount == 1)
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
        playback.startStreaming(sampleRate: 16_000)
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

    @Test
    func recordsVolumeAndScriptedPlaybackLevel() {
        let playback = InMemoryAudioPlayback()
        #expect(playback.playbackLevel() == 0)
        playback.scriptedPlaybackLevel = 0.4
        #expect(playback.playbackLevel() == 0.4)

        playback.setVolume(0.25)
        playback.setVolume(1.0)
        #expect(playback.setVolumeCalls == [0.25, 1.0])
    }
}

// MARK: - PlaybackEnvelope (the real sinks' loudness timeline, ADR-0041)

struct PlaybackEnvelopeTests {

    @Test
    func binsAtFiftyMillisecondGrainAcrossChunkBoundaries() {
        // One 50 ms bin at 24 kHz is 1200 samples; splitting it across two
        // appends must land in the same bin — boundaries are the envelope's,
        // never the chunks'.
        var envelope = PlaybackEnvelope()
        envelope.begin(sampleRate: 24_000)
        envelope.append(samples: [Float](repeating: 0.5, count: 600))
        #expect(envelope.level(at: 0.02) == 0)  // bin not complete yet
        envelope.append(samples: [Float](repeating: 0.5, count: 600))

        // RMS 0.5 → −6 dB → (−6+60)/60 ≈ 0.9 on the meter scale.
        let level = envelope.level(at: 0.02)
        #expect(level > 0.88)
        #expect(level < 0.92)
    }

    @Test
    func silenceReadsZeroAndPastTheEndReadsZero() {
        var envelope = PlaybackEnvelope()
        envelope.begin(sampleRate: 24_000)
        envelope.append(samples: [Float](repeating: 0, count: 2400))
        // Digital silence bottoms out the −60 dB floor.
        #expect(envelope.level(at: 0.05) == 0)
        // Beyond scheduled audio: nothing is playing there.
        #expect(envelope.level(at: 5.0) == 0)
        #expect(envelope.level(at: -1.0) == 0)
    }

    @Test
    func resetForgetsTheTimeline() {
        var envelope = PlaybackEnvelope()
        envelope.begin(sampleRate: 24_000)
        envelope.append(samples: [Float](repeating: 0.5, count: 2400))
        #expect(envelope.level(at: 0.02) > 0)
        envelope.reset()
        #expect(envelope.level(at: 0.02) == 0)
    }

    @Test
    func fullScaleReadsOne() {
        // The one shared normalization (mic meter + playback envelope).
        #expect(AudioConverter.meterLevel(rms: 1.0) == 1.0)
        #expect(AudioConverter.meterLevel(rms: 0.001) == 0)
    }
}
