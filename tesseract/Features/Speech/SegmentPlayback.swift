//
//  SegmentPlayback.swift
//  tesseract
//
//  The deep module owning the consume-one-TTS-stream-into-playback loop shared by
//  every speech path — the first long-form segment, each subsequent segment, and
//  single-shot streaming (CONTEXT.md → Segment Playback). Given a generated-sample
//  stream and a small `Segment` value, it drains the stream into
//  `AudioPlayback.appendChunk`, drives the `WordHighlightSurface`, performs the
//  boundary switch and drain-wait when the segment carries a boundary, and returns
//  `false` on cancellation/pause so each caller keeps its own cleanup.
//
//  The per-segment difference is the `Segment` *value*, not a bag of flags — the same
//  "the only injected difference is a value" move as Overlay Placement. The
//  duration-update timing and the boundary switch are *derived* from whether the
//  `Segment` carries a boundary, not from separate knobs.
//

import Foundation

@MainActor
struct SegmentPlayback {

    /// The shape of one segment to drain. `.first` / `.next` / `.single` make the three
    /// call sites zero-ceremony; the difference between them is this value, never flags.
    struct Segment {
        let text: String
        let tokenOffsets: [Int]

        /// The previous segment's cumulative scheduled duration — the **Segment Window**
        /// the playback head must reach before the surface switches to this segment's
        /// text. `nil` for a segment the caller has already `show`n (the first/only one),
        /// which both defers the switch *and* means duration updates flow from chunk one.
        let boundary: TimeInterval?

        /// `SpeechState` to publish when the first chunk arrives — single-streaming's
        /// `.streaming` flip. `nil` when the caller set the state before draining.
        let firstChunkState: SpeechState?

        /// Align this segment's duration estimate after draining so its highlight
        /// converges to 100% (long-form segments do; single-shot streaming leaves that
        /// to the caller's `markGenerationComplete`).
        let marksSegmentComplete: Bool

        /// The first/only segment of a long-form run: already `show`n, no boundary,
        /// aligned on completion.
        static func first(text: String, tokenOffsets: [Int]) -> Segment {
            Segment(text: text, tokenOffsets: tokenOffsets, boundary: nil,
                    firstChunkState: nil, marksSegmentComplete: true)
        }

        /// A subsequent long-form segment: switches the surface in once the head reaches
        /// `boundary`, then aligns on completion.
        static func next(text: String, tokenOffsets: [Int], boundary: TimeInterval) -> Segment {
            Segment(text: text, tokenOffsets: tokenOffsets, boundary: boundary,
                    firstChunkState: nil, marksSegmentComplete: true)
        }

        /// A single (non-long-form) streaming segment: already `show`n, flips to
        /// `firstChunkState` on the first chunk, and leaves the duration estimate to the
        /// caller's `markGenerationComplete`.
        static func single(text: String, tokenOffsets: [Int], firstChunkState: SpeechState) -> Segment {
            Segment(text: text, tokenOffsets: tokenOffsets, boundary: nil,
                    firstChunkState: firstChunkState, marksSegmentComplete: false)
        }
    }

    let playback: any AudioPlayback
    let surface: (any WordHighlightSurface)?

    /// How close the playback head must get to a boundary for the switch to fire.
    private static let boundaryTolerance: TimeInterval = 0.1
    /// Poll cadence while waiting for the head to reach a boundary.
    private static let drainPoll: Duration = .milliseconds(50)

    /// Drain one segment's sample stream into playback and the highlight surface.
    /// Returns `true` once fully drained, `false` if cancelled or paused mid-drain — in
    /// which case the caller performs its own teardown (this performs none).
    func run(
        _ segment: Segment,
        stream: AsyncThrowingStream<[Float], Error>,
        onState: (SpeechState) -> Void,
        isPaused: () -> Bool
    ) async throws -> Bool {
        // A boundary-less segment is already on screen, so it counts as "switched".
        var switched = segment.boundary == nil
        var sawFirstChunk = false

        for try await chunk in stream {
            guard !Task.isCancelled else { return false }
            playback.appendChunk(samples: chunk)

            if let boundary = segment.boundary {
                // Switch the moment the head reaches the boundary; only push duration
                // afterwards, so a not-yet-switched segment can't corrupt the previous
                // segment's pacing.
                if !switched && playback.currentPlaybackTime() >= boundary - Self.boundaryTolerance {
                    surface?.switchText(segment.text, tokenCharOffsets: segment.tokenOffsets, segmentBase: boundary)
                    switched = true
                }
                if switched {
                    surface?.updateTotalDuration(playback.totalScheduledDuration)
                }
            } else {
                surface?.updateTotalDuration(playback.totalScheduledDuration)
            }

            if !sawFirstChunk {
                sawFirstChunk = true
                if let firstChunkState = segment.firstChunkState { onState(firstChunkState) }
            }
        }

        // Generation finished before playback caught up: wait for the head to reach the
        // boundary, then switch.
        if let boundary = segment.boundary, !switched {
            while playback.currentPlaybackTime() < boundary - Self.boundaryTolerance {
                guard !Task.isCancelled else { return false }
                if isPaused() { return false }
                try await Task.sleep(for: Self.drainPoll)
            }
            surface?.switchText(segment.text, tokenCharOffsets: segment.tokenOffsets, segmentBase: boundary)
        }

        // A boundary segment's deferred duration updates didn't run for the chunks
        // scheduled before the switch — push the final cumulative duration once.
        if segment.boundary != nil {
            surface?.updateTotalDuration(playback.totalScheduledDuration)
        }
        if segment.marksSegmentComplete {
            surface?.markSegmentComplete()
        }
        return true
    }
}
