//
//  StreamingSchedulerTests.swift
//  tesseractTests
//
//  The **Streaming Scheduler** at its own seam (ADR-0054): decision tables
//  over the push-based playback fold — the start gate, finish detection, and
//  the stream epoch that makes a stale buffer completion ignorable. Before the
//  cut this fold was declared and mutated inline in three places
//  (`AudioPlaybackManager`, `VoiceSessionPlayback`, `InMemoryAudioPlayback`),
//  had already drifted on the epoch guard, and no test constructed any of
//  them. All three now drive the value machine tested here.
//
//  Gotcha: Swift Testing's `#expect` can't wrap a call to a `mutating` method
//  directly — every verdict below is hoisted into a `let` first.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct StreamingSchedulerTests {

    /// The start gate fires exactly once per stream: the first chunk on an
    /// unpaused stream starts the player, every chunk after does not.
    @Test func startGateFiresExactlyOncePerStream() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)

        let first = scheduler.appendChunk(sampleCount: 100)
        #expect(first.startPlayer)
        #expect(scheduler.playerStarted)

        let second = scheduler.appendChunk(sampleCount: 100)
        #expect(!second.startPlayer)
        #expect(scheduler.pendingBufferCount == 2)
    }

    /// A chunk arriving while paused does not start the player; resuming does
    /// (it plays the node), and a chunk after resume must not restart it.
    @Test func pausedFirstChunkDoesNotStartAndResumeStartsExactlyOnce() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)

        let paused = scheduler.pause()
        #expect(paused)
        #expect(scheduler.isPaused)

        let whilePaused = scheduler.appendChunk(sampleCount: 100)
        #expect(!whilePaused.startPlayer)
        #expect(!scheduler.playerStarted)

        let resumed = scheduler.resume()
        #expect(resumed)
        #expect(scheduler.playerStarted)
        #expect(!scheduler.isPaused)

        let afterResume = scheduler.appendChunk(sampleCount: 100)
        #expect(!afterResume.startPlayer)
    }

    /// Pause and resume are idempotent — only the state-changing call reports
    /// `true`, so the adapter pauses/plays the node exactly once.
    @Test func pauseAndResumeReportOnlyRealTransitions() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)

        let firstPause = scheduler.pause()
        let secondPause = scheduler.pause()
        #expect(firstPause)
        #expect(!secondPause)

        let firstResume = scheduler.resume()
        let secondResume = scheduler.resume()
        #expect(firstResume)
        #expect(!secondResume)
    }

    /// Finish fires only when the stream is finished AND the last pending
    /// buffer lands — not on the earlier drains.
    @Test func finishFiresOnlyWhenFinishedAndLastBufferDrains() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)
        let a = scheduler.appendChunk(sampleCount: 100)
        let b = scheduler.appendChunk(sampleCount: 100)

        let finish = scheduler.finishStream()
        #expect(finish == .awaitingDrain)

        let first = scheduler.bufferCompleted(epoch: a.epoch)
        #expect(first == .pending)
        #expect(scheduler.pendingBufferCount == 1)

        let last = scheduler.bufferCompleted(epoch: b.epoch)
        #expect(last == .finished)
        #expect(scheduler.pendingBufferCount == 0)
    }

    /// Finish with nothing pending ends the utterance immediately — both when
    /// no chunk ever arrived and when every chunk has already drained.
    @Test func finishWithZeroPendingFiresImmediately() {
        var empty = StreamingScheduler()
        empty.beginStream(sampleRate: 24_000)
        let emptyFinish = empty.finishStream()
        #expect(emptyFinish == .finishedNow)

        var drained = StreamingScheduler()
        drained.beginStream(sampleRate: 24_000)
        let a = drained.appendChunk(sampleCount: 100)
        let completion = drained.bufferCompleted(epoch: a.epoch)
        #expect(completion == .pending)  // drained before finish → not final yet
        let drainedFinish = drained.finishStream()
        #expect(drainedFinish == .finishedNow)
    }

    /// A completion carrying a stale epoch (from a stopped-and-restarted
    /// stream) is ignored — it never decrements the *new* stream's counter.
    /// This is the guard `AudioPlaybackManager` shipped without.
    @Test func staleCompletionIgnoredAfterStopAndRestart() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)
        let old = scheduler.appendChunk(sampleCount: 100)

        scheduler.stop()
        scheduler.beginStream(sampleRate: 24_000)
        let fresh = scheduler.appendChunk(sampleCount: 100)
        #expect(old.epoch != fresh.epoch)

        // The old stream's buffer lands late — a stopped node flushes its
        // handlers — and must not touch the new stream's counter.
        let stale = scheduler.bufferCompleted(epoch: old.epoch)
        #expect(stale == .ignoreStale)
        #expect(scheduler.pendingBufferCount == 1)

        let current = scheduler.bufferCompleted(epoch: fresh.epoch)
        #expect(current == .pending)
        #expect(scheduler.pendingBufferCount == 0)
    }

    /// Stop mid-stream invalidates every in-flight completion: even a
    /// finished stream's buffers, landing after the stop, are ignored rather
    /// than firing a finish on the torn-down stream.
    @Test func stopMidStreamInvalidatesInFlightCompletions() {
        var scheduler = StreamingScheduler()
        scheduler.beginStream(sampleRate: 24_000)
        let a = scheduler.appendChunk(sampleCount: 100)
        let b = scheduler.appendChunk(sampleCount: 100)
        _ = scheduler.finishStream()

        scheduler.stop()
        #expect(scheduler.pendingBufferCount == 0)

        let staleA = scheduler.bufferCompleted(epoch: a.epoch)
        let staleB = scheduler.bufferCompleted(epoch: b.epoch)
        #expect(staleA == .ignoreStale)
        #expect(staleB == .ignoreStale)
    }

    /// The duration read is derived from the machine, not hand-copied: it
    /// tracks scheduled samples over the current stream's rate and resets on
    /// each new stream and on stop.
    @Test func durationTracksScheduledSamplesAndResetsPerStream() {
        var scheduler = StreamingScheduler()
        #expect(scheduler.totalScheduledDuration == 0)

        scheduler.beginStream(sampleRate: 24_000)
        _ = scheduler.appendChunk(sampleCount: 2_400)
        _ = scheduler.appendChunk(sampleCount: 1_200)
        #expect(scheduler.totalScheduledDuration == 3_600.0 / 24_000.0)

        // A new stream at a different sample rate starts the count fresh.
        scheduler.beginStream(sampleRate: 16_000)
        #expect(scheduler.totalScheduledDuration == 0)
        _ = scheduler.appendChunk(sampleCount: 1_600)
        #expect(scheduler.totalScheduledDuration == 1_600.0 / 16_000.0)

        scheduler.stop()
        #expect(scheduler.totalScheduledDuration == 0)
    }

    /// The `hasUndrainedAudio` gate (VoiceSessionPlayback's `hostedPlaying`)
    /// is true from stream start — even before any chunk — until a finished
    /// stream fully drains, and false while idle.
    @Test func hasUndrainedAudioGatesFromStartUntilDrained() {
        var scheduler = StreamingScheduler()
        #expect(!scheduler.hasUndrainedAudio)

        scheduler.beginStream(sampleRate: 24_000)
        #expect(scheduler.hasUndrainedAudio)

        let a = scheduler.appendChunk(sampleCount: 100)
        _ = scheduler.finishStream()
        #expect(scheduler.hasUndrainedAudio)  // finished but not yet drained

        _ = scheduler.bufferCompleted(epoch: a.epoch)
        #expect(!scheduler.hasUndrainedAudio)  // drained
    }
}
