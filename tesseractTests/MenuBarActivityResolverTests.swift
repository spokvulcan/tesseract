//
//  MenuBarActivityResolverTests.swift
//  tesseractTests
//
//  The status glyph's activity is a pure derivation (#396): three
//  per-source phase mappings and a priority merge, extracted beside the
//  `MenuBarLanguagePins` precedent. These pin the ladder — dictation over
//  speech over Companion presence — and every mapping row; the icon
//  rendering stays dumb over the resolved value.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct MenuBarActivityResolverTests {

    // MARK: - The priority ladder

    @Test
    func dictationOutranksSpeechOutranksCompanion() {
        #expect(
            MenuBarActivityResolver.merged(
                dictation: .listening, speech: .speaking, companion: .thinking)
                == .listening)
        #expect(
            MenuBarActivityResolver.merged(
                dictation: .idle, speech: .speaking, companion: .thinking)
                == .speaking)
        #expect(
            MenuBarActivityResolver.merged(
                dictation: .idle, speech: .idle, companion: .thinking)
                == .thinking)
    }

    @Test
    func allIdleResolvesIdle() {
        #expect(
            MenuBarActivityResolver.merged(dictation: .idle, speech: .idle, companion: .idle)
                == .idle)
    }

    /// The quietest Companion rungs still surface when nothing acute runs —
    /// a summons or the sleep pass is visible presence.
    @Test
    func companionRungsSurfaceWhenAlone() {
        #expect(
            MenuBarActivityResolver.merged(dictation: .idle, speech: .idle, companion: .summoning)
                == .summoning)
        #expect(
            MenuBarActivityResolver.merged(dictation: .idle, speech: .idle, companion: .asleep)
                == .asleep)
    }

    // MARK: - Per-source mappings

    @Test
    func dictationPhasesMapToAcuteActivity() {
        #expect(MenuBarActivityResolver.activity(fromDictation: .idle) == .idle)
        #expect(MenuBarActivityResolver.activity(fromDictation: .recording) == .listening)
        #expect(MenuBarActivityResolver.activity(fromDictation: .processing) == .processing)
        #expect(MenuBarActivityResolver.activity(fromDictation: .proofreading) == .processing)
        #expect(MenuBarActivityResolver.activity(fromDictation: .error(.microphoneBusy)) == .idle)
    }

    @Test
    func speechStatesMapAudibleFlightToSpeaking() {
        #expect(MenuBarActivityResolver.activity(fromSpeech: .idle) == .idle)
        #expect(MenuBarActivityResolver.activity(fromSpeech: .capturingText) == .idle)
        #expect(MenuBarActivityResolver.activity(fromSpeech: .error("x")) == .idle)
        #expect(MenuBarActivityResolver.activity(fromSpeech: .paused(segment: 1, of: 3)) == .idle)
        #expect(
            MenuBarActivityResolver.activity(fromSpeech: .generating(progress: "…")) == .speaking)
        #expect(MenuBarActivityResolver.activity(fromSpeech: .streaming) == .speaking)
        #expect(
            MenuBarActivityResolver.activity(fromSpeech: .streamingLongForm(segment: 1, of: 3))
                == .speaking)
        #expect(MenuBarActivityResolver.activity(fromSpeech: .playing) == .speaking)
    }

    @Test
    func companionPresenceMapsOneToOne() {
        #expect(MenuBarActivityResolver.activity(fromCompanion: .idle) == .idle)
        #expect(MenuBarActivityResolver.activity(fromCompanion: .thinking) == .thinking)
        #expect(MenuBarActivityResolver.activity(fromCompanion: .summoning) == .summoning)
        #expect(MenuBarActivityResolver.activity(fromCompanion: .asleep) == .asleep)
    }
}
