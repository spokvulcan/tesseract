//
//  PocketControls.swift
//  tesseract
//
//  Reading in the pocket (#515): what a call, Siri, unplugged headphones,
//  the lock screen's buttons or the app going to the background do to the
//  reading. The phone's adapters turn the system's notifications and remote
//  commands into these events.
//
//  Everything that stops the audio from outside the app stops the reading at
//  the sentence being heard, and starting again reads from that sentence.
//  Only a pause made in the app, while it is in front, holds the audio
//  itself: once the app is in the background the system may stop the audio
//  engine under it, and a held stream can't come back from that.
//

import Foundation

/// Something outside the reading that wants it to stop, go on or move.
nonisolated enum PocketEvent: Equatable, Sendable {
    /// A call, Siri, an alarm took the audio.
    case interruptionBegan
    /// It gave the audio back; the system says whether to go on.
    case interruptionEnded(shouldResume: Bool)
    /// The headphones went away: the reading pauses, as in any audio app.
    case outputLost
    /// The output changed under the audio (new headphones), which stops the
    /// audio engine: the reading starts again where it was.
    case outputChanged
    /// The app left the screen.
    case movedToBackground
    /// A lock screen, Control Center or headphone button.
    case remote(RemoteCommand)
}

nonisolated enum RemoteCommand: Equatable, Sendable {
    case play, pause, togglePlayPause
    /// Previous and next track: a sentence back and forward.
    case previousTrack, nextTrack
    /// The lock screen's position bar, as a fraction of the text.
    case seek(Double)
}

@MainActor
final class PocketControls {
    /// The Reader the events act on: the text being read, or the one last
    /// opened.
    private let reader: @MainActor () -> SpeechReader?
    /// The interruption stopped a reading that was playing.
    private var resumesAfterInterruption = false

    init(reader: @escaping @MainActor () -> SpeechReader?) {
        self.reader = reader
    }

    func handle(_ event: PocketEvent) {
        guard let reader = reader() else { return }
        let isPlaying = reader.isReading && !reader.isPaused
        switch event {
        case .interruptionBegan:
            resumesAfterInterruption = isPlaying
            if reader.isReading { reader.stop() }
        case .interruptionEnded(let shouldResume):
            if resumesAfterInterruption, shouldResume { reader.play() }
            resumesAfterInterruption = false
        case .outputLost:
            if reader.isReading { reader.stop() }
        case .outputChanged:
            if isPlaying { restart(reader) }
        case .movedToBackground:
            if reader.isPaused { reader.stop() }
        case .remote(let command):
            handle(command, on: reader, isPlaying: isPlaying)
        }
    }

    private func handle(_ command: RemoteCommand, on reader: SpeechReader, isPlaying: Bool) {
        switch command {
        case .play where !isPlaying, .togglePlayPause where !isPlaying:
            restart(reader)
        case .pause where isPlaying, .togglePlayPause where isPlaying:
            reader.stop()
        case .previousTrack:
            reader.skip(sentences: -1)
        case .nextTrack:
            reader.skip(sentences: 1)
        case .seek(let fraction):
            reader.seek(toFraction: fraction)
        default:
            break
        }
    }

    /// Reads on from the sentence being heard (or the bookmark).
    private func restart(_ reader: SpeechReader) {
        if reader.isReading { reader.stop() }
        reader.play()
    }
}
