//
//  LFM3VoicePlaybackTests.swift
//  tesseractTests
//
//  LFM3 Voice's speech in the voice session: the answer's audio comes from
//  the turn, and the session's `speak` finishes only once the turn made all
//  of it and it played out. Speech reads active the whole turn, so the mic
//  stays closed while a tool runs.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct LFM3VoicePlaybackTests {

    private final class Done {
        var calls = 0
    }

    private func make() -> (LFM3VoicePlayback, InMemoryAudioPlayback, Done) {
        let player = InMemoryAudioPlayback()
        let playback = LFM3VoicePlayback(player: player)
        return (playback, player, Done())
    }

    @Test func speechFinishesAfterTheTurnEndsAndItsAudioPlaysOut() {
        let (playback, player, done) = make()
        playback.beginTurn()
        // A tool round can come before any audio: the mic stays closed.
        #expect(playback.isActive)
        playback.play([0.1, 0.2])
        playback.speak { done.calls += 1 }
        playback.play([0.3])
        playback.endTurn()
        playback.finishSpeaking()
        #expect(player.startedSampleRates == [LFM3VoiceEngine.sampleRate])
        #expect(player.appendedChunks == [[0.1, 0.2], [0.3]])
        #expect(player.finishStreamingCount == 1)
        #expect(done.calls == 0)
        #expect(playback.isActive)

        player.firePlaybackFinished()
        #expect(done.calls == 1)
        #expect(!playback.isActive)
    }

    @Test func aTurnWithNoAudioFinishesWhenTheSessionSaysSo() {
        let (playback, player, done) = make()
        playback.beginTurn()
        playback.speak { done.calls += 1 }
        playback.endTurn()
        // The empty stream finishes at once, as the audio engine's does.
        player.firePlaybackFinished()
        #expect(done.calls == 0)
        playback.finishSpeaking()
        #expect(done.calls == 1)
        #expect(player.appendedChunks.isEmpty)
    }

    @Test func stoppingDropsTheRestOfTheTurnsAudioUntilTheNextTurn() {
        let (playback, player, done) = make()
        playback.beginTurn()
        playback.play([0.1])
        playback.speak { done.calls += 1 }
        playback.stop()
        playback.play([0.2])
        #expect(player.stopCount == 1)
        #expect(player.appendedChunks == [[0.1]])
        playback.endTurn()
        playback.finishSpeaking()
        #expect(done.calls == 0)

        playback.beginTurn()
        playback.play([0.3])
        #expect(player.startStreamingCount == 2)
        #expect(player.appendedChunks == [[0.1], [0.3]])
    }
}
