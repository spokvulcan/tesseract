//
//  StreamingScheduler.swift
//  tesseract
//
//  The **Streaming Scheduler** (ADR-0054): the push-based playback fold as a
//  pure value machine under both `AudioPlayback` adapters and the in-memory
//  test peer. The counters (`pendingBufferCount`, `totalScheduledSamples`,
//  `streamingSampleRate`), the start gate, the finish detection, and the
//  stream epoch that makes a stale buffer completion ignorable all live
//  here; each driver calls one transition and performs its verdict. The
//  adapters keep only their `AVAudioPlayerNode`/engine calls (and the clock
//  position they hold while paused). This is the fourth application of the
//  policy/performer template — after the capture engine's lifecycle, duck,
//  and hold-wiring (ADR-0050) — and the seam that makes the three copies'
//  drift (a race guard shipped in one, absent in another) unrepresentable.
//

import Foundation

/// Pure state machine over the push scheduler's facts: how many scheduled
/// buffers are still draining, whether the producer has signalled its last
/// chunk, whether the player node has been told to start, whether the head
/// is paused, and the stream epoch. Holds no engine state — the
/// `AVAudioPlayerNode`, the audio format, the loudness `PlaybackEnvelope`,
/// and every AVFoundation call stay on the adapter; the adapter's drivers
/// call one transition each and perform its verdict.
nonisolated struct StreamingScheduler: Sendable, Equatable {

    /// Scheduled buffers that have not yet reported completion. The finish
    /// gate fires only when this reaches zero *and* the stream is finished.
    private(set) var pendingBufferCount = 0

    /// The producer has called `finishStream` — no more chunks will arrive,
    /// so the last buffer to drain ends the utterance.
    private(set) var streamFinished = false

    /// The player node has been told to `play()`. The start gate fires on
    /// the first chunk that arrives while not paused; every chunk after,
    /// and a chunk arriving on a paused-first stream, must not restart it.
    private(set) var playerStarted = false

    /// The head is paused. Held as a machine fact (not an adapter input) so
    /// the start gate stays a machine decision and the adapters stay
    /// verdict-only. The *clock position* to hold while paused is the
    /// adapter's — the machine has no notion of render time.
    private(set) var isPaused = false

    /// Every sample scheduled so far this stream, behind
    /// `totalScheduledDuration`.
    private(set) var totalScheduledSamples = 0

    /// The stream's sample rate, 0 while idle.
    private(set) var streamingSampleRate = 0

    /// Staleness guard for buffer completions. A stopped or restarted node
    /// flushes its scheduled handlers; each is stamped with the epoch that
    /// scheduled it, so a completion from a dead stream is ignored rather
    /// than decrementing the *new* stream's counter and firing its finish
    /// early. Bumped on every `beginStream` and `stop`, monotonically.
    private(set) var epoch = 0

    // MARK: - Reads

    /// Cumulative duration of all audio scheduled so far this stream, in
    /// seconds. The one formula behind every adapter's
    /// `totalScheduledDuration` — no longer hand-copied per sink.
    var totalScheduledDuration: TimeInterval {
        guard streamingSampleRate > 0 else { return 0 }
        return Double(totalScheduledSamples) / Double(streamingSampleRate)
    }

    /// A stream is live and still has audio to render — the
    /// `VoiceSessionPlayback.hostedPlaying` gate. False while idle (no
    /// stream) and false once a finished stream has fully drained.
    var hasUndrainedAudio: Bool {
        streamingSampleRate > 0 && !(streamFinished && pendingBufferCount <= 0)
    }

    // MARK: - Verdicts

    /// What `appendChunk` must do after the adapter has built the buffer.
    struct AppendOutcome: Sendable, Equatable {
        /// Stamp this buffer's completion callback with this epoch, so a
        /// landing after a stop/restart proves itself current or is ignored.
        let epoch: Int
        /// Start the player node now — the first chunk on an unpaused,
        /// not-yet-started stream. Later chunks and paused-first streams
        /// return `false` and must not touch the node.
        let startPlayer: Bool
    }

    /// What a buffer-completion callback must do.
    enum CompletionVerdict: Sendable, Equatable {
        /// The completion belongs to a dead stream (its epoch is stale) —
        /// ignore it. This is the guard `AudioPlaybackManager` lacked.
        case ignoreStale
        /// A buffer drained but the stream is not finished, or more remain —
        /// the counter dropped, nothing else to do.
        case pending
        /// The last buffer of a finished stream drained — fire
        /// `onPlaybackFinished`.
        case finished
    }

    /// What `finishStream` must do.
    enum FinishVerdict: Sendable, Equatable {
        /// Nothing was pending — the utterance ends immediately.
        case finishedNow
        /// Buffers are still draining — the last completion will end it.
        case awaitingDrain
    }

    // MARK: - Transitions

    /// Begins a stream at `sampleRate`: resets the counters and stales any
    /// completions still in flight from a previous stream. Adapters call
    /// this after tearing down the old engine; the in-memory peer calls it
    /// directly.
    mutating func beginStream(sampleRate: Int) {
        invalidate()
        streamingSampleRate = sampleRate
    }

    /// Records a scheduled chunk of `sampleCount` samples and decides the
    /// start gate. Called only after the adapter has a real buffer (the
    /// empty-chunk and buffer-creation guards are adapter-local, short of
    /// touching any counter).
    mutating func appendChunk(sampleCount: Int) -> AppendOutcome {
        totalScheduledSamples += sampleCount
        pendingBufferCount += 1
        let startPlayer = !playerStarted && !isPaused
        if startPlayer { playerStarted = true }
        return AppendOutcome(epoch: epoch, startPlayer: startPlayer)
    }

    /// A scheduled buffer reported completion carrying the `epoch` it was
    /// stamped with. Ignored if stale; otherwise the counter drops and the
    /// utterance ends if this was the last buffer of a finished stream.
    mutating func bufferCompleted(epoch: Int) -> CompletionVerdict {
        guard epoch == self.epoch else { return .ignoreStale }
        pendingBufferCount -= 1
        if streamFinished && pendingBufferCount <= 0 { return .finished }
        return .pending
    }

    /// The producer signals no more chunks. The utterance ends now if
    /// nothing is pending, otherwise when the last buffer drains.
    mutating func finishStream() -> FinishVerdict {
        streamFinished = true
        return pendingBufferCount <= 0 ? .finishedNow : .awaitingDrain
    }

    /// Pauses the head. Returns `true` when the state actually changed —
    /// the adapter then captures the clock position and pauses the node.
    mutating func pause() -> Bool {
        guard !isPaused else { return false }
        isPaused = true
        return true
    }

    /// Resumes from a pause. Returns `true` when the state actually changed;
    /// resuming plays the node, so `playerStarted` becomes true (a chunk
    /// arriving after resume must not restart it).
    mutating func resume() -> Bool {
        guard isPaused else { return false }
        isPaused = false
        playerStarted = true
        return true
    }

    /// Stops the stream: resets the counters and stales every completion
    /// still in flight, so none can decrement a later stream's counter.
    mutating func stop() {
        invalidate()
    }

    /// Resets every counter to idle and bumps the epoch so in-flight
    /// completions are stale from here on — the one path both `beginStream`
    /// and `stop` share, so they cannot drift as fields are added.
    private mutating func invalidate() {
        epoch += 1
        pendingBufferCount = 0
        streamFinished = false
        playerStarted = false
        isPaused = false
        totalScheduledSamples = 0
        streamingSampleRate = 0
    }
}
