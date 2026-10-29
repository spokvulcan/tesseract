//
//  LFM3VoicePlayback.swift
//  tesseract
//
//  **LFM3 Voice**'s speech in the voice session. The model speaks its answer
//  as it writes it, so the sound comes from the turn, not from the text the
//  session hands its voice: the session's `speak` only waits for the turn's
//  audio to play out. Half-duplex holds as for Jarvis's voice: the session
//  keeps the mic closed while a turn runs or its audio plays.
//

import Foundation

@MainActor
final class LFM3VoicePlayback {

    private let player: any AudioPlayback
    /// A turn is under way, so more of its audio may come.
    private var producing = false
    /// The turn's audio stream is open and hasn't played out.
    private var playing = false
    /// The voice was stopped while the turn still runs: the rest of its
    /// audio is dropped.
    private var muted = false
    /// The session said no more is coming: its callback fires once the audio
    /// plays out.
    private var finishRequested = false
    private var onDone: (@MainActor @Sendable () -> Void)?

    init(player: any AudioPlayback = AudioPlaybackManager()) {
        self.player = player
        player.onPlaybackFinished = { [weak self] in self?.drained() }
    }

    /// A turn runs or its audio plays.
    var isActive: Bool { producing || playing }

    // MARK: - The turn

    /// Opens the turn's stream while its answer is still being written, so
    /// starting the audio engine never delays the first sound.
    func beginTurn() {
        producing = true
        muted = false
        player.startStreaming(sampleRate: LFM3VoiceEngine.sampleRate)
        playing = true
    }

    func play(_ samples: [Float]) {
        guard !muted, !samples.isEmpty else { return }
        player.appendChunk(samples: samples)
    }

    /// The turn made all its audio.
    func endTurn() {
        producing = false
        if playing { player.finishStreaming() } else { finishIfDone() }
    }

    // MARK: - The voice session's speech

    func speak(onDone: @escaping @MainActor @Sendable () -> Void) {
        self.onDone = onDone
    }

    func finishSpeaking() {
        finishRequested = true
        finishIfDone()
    }

    func stop() {
        player.stop()
        playing = false
        muted = producing
        finishRequested = false
        onDone = nil
    }

    // MARK: - Private

    private func drained() {
        playing = false
        finishIfDone()
    }

    private func finishIfDone() {
        guard finishRequested, !producing, !playing, let onDone else { return }
        self.onDone = nil
        finishRequested = false
        onDone()
    }
}
