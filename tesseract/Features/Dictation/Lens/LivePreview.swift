//
//  LivePreview.swift
//  tesseract
//
//  The **Live Preview** (PRD #612, ADR-0085): what the Lens shows while the
//  owner is still talking, decoded by the same Whisper model as the final
//  pass. Each decode reads the audio from the end of the last confirmed
//  segment, so the Lens can show the whole take while a decode stays short.
//  WhisperKit's own confirmation rule decides what is settled: when a decode
//  returns more than two segments, all but the last two are confirmed and
//  the next decode starts where they end. The rest is the provisional tail,
//  shown dimmed, rewritten by the next decode.
//
//  The preview is only ever shown. What pastes is the full pass over the
//  whole take after release, exactly as before: pasting streamed text lost
//  words at segment boundaries in the 2026-10-03 replay.
//

import Foundation

/// One state of the preview, ready to show.
nonisolated struct LivePreview: Equatable, Sendable {
    /// The preview's text after the regex cleanup and the Learned Words.
    let text: String
    /// What the Learned Words caught in `text` (shown flipped; not counted:
    /// only a committed take counts its catches).
    let catches: [LearnedWordCatch]
    /// How many of `text`'s leading tokens are confirmed; the rest is the
    /// provisional tail.
    let confirmedTokens: Int
}

/// Folds decodes into a preview with WhisperKit's confirmation rule. Pure:
/// the coordinator's pump feeds it audio windows and decode results.
nonisolated struct LivePreviewAssembler: Sendable {
    /// No decode before this much new audio: sub-second snippets return
    /// noise, and the first words deserve one clean pass.
    static let minimumAudio: TimeInterval = 0.6
    /// Segments a decode keeps unconfirmed (WhisperKit's
    /// `requiredSegmentsForConfirmation`).
    static let unconfirmedSegments = 2
    /// An unconfirmed stretch longer than this confirms all but its last
    /// segment, so a decode never grows toward Whisper's 30 s window.
    static let maximumUnconfirmed: TimeInterval = 20

    /// The confirmed text so far, segment texts joined.
    private(set) var confirmed = ""
    /// Where the confirmed segments end, in seconds into the take.
    private(set) var confirmedEnd: TimeInterval = 0
    /// The newest decode's unconfirmed segments.
    private(set) var tail = ""

    init() {}

    /// The audio the next decode reads, given the take's audio from
    /// `confirmedEnd` to now (`AudioCapturing.captureSnapshot(from:)`). Nil
    /// until there is enough of it.
    func window(ofTail tail: AudioData) -> AudioData? {
        tail.duration >= Self.minimumAudio ? tail : nil
    }

    /// Folds one decode of `window(ofTail:)`. Segment times are relative to the
    /// window's start.
    mutating func fold(_ result: TranscriptionResult, windowDuration: TimeInterval) {
        let segments = result.segments.filter {
            !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let tooLong = windowDuration > Self.maximumUnconfirmed
        guard !segments.isEmpty else {
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if tooLong {
                // No segment boundaries to confirm at: settle what was heard
                // and move on, so the window never grows toward Whisper's 30 s.
                append([text])
                confirmedEnd += windowDuration
                tail = ""
            } else {
                tail = text
            }
            return
        }
        var keep = Self.unconfirmedSegments
        if tooLong { keep = segments.count > 1 ? 1 : 0 }
        if segments.count > keep, keep == 0 {
            append(segments.map(\.text))
            confirmedEnd += min(max(0, segments.last?.endTime ?? windowDuration), windowDuration)
            tail = ""
        } else if segments.count > keep {
            let settled = segments[..<(segments.count - keep)]
            append(settled.map(\.text))
            let end = min(settled.last?.endTime ?? 0, windowDuration)
            confirmedEnd += max(0, end)
            tail = Self.join(segments[(segments.count - keep)...].map(\.text))
        } else {
            tail = Self.join(segments.map(\.text))
        }
    }

    /// The whole preview: confirmed then tail.
    var rawText: String {
        Self.join([confirmed, tail])
    }

    private mutating func append(_ texts: [String]) {
        confirmed = Self.join([confirmed] + texts)
    }

    private static func join(_ parts: [String]) -> String {
        parts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
