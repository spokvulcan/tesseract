//
//  HotkeyMatcherTests.swift
//  tesseractTests
//
//  Decision table for the pure hotkey matcher — the one fire-or-not decision
//  both delivery paths (event tap and NSEvent fallback) share. Until this
//  extraction the decision was implemented twice, with divergent semantics
//  and no test surface.
//

import AppKit
import Testing

@testable import Tesseract_Agent

struct HotkeyMatcherTests {

    private static let space: UInt16 = 49
    private static let fKey: UInt16 = 3

    private static let dictation = KeyCombo(keyCode: space, modifiers: .option)
    private static let agent = KeyCombo(keyCode: space, modifiers: [.command, .shift])
    private static let tts = KeyCombo(keyCode: fKey, modifiers: .function)

    private static let bindings: [String: KeyCombo] = [
        "dictation": dictation,
        "agent": agent,
        "tts": tts,
    ]

    private func down(_ id: String) -> HotkeyMatcher.Fire { .init(id: id, direction: .down) }
    private func up(_ id: String) -> HotkeyMatcher.Fire { .init(id: id, direction: .up) }

    // MARK: - Press

    @Test
    func exactChordFiresDownAndSuppresses() {
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(verdict.fires == [down("dictation")])
        #expect(verdict.suppressKeyEvent)
        #expect(matcher.pressed == ["dictation"])
    }

    @Test
    func keyRepeatSuppressesWithoutRefiring() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        let repeated = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(repeated.fires.isEmpty)
        #expect(repeated.suppressKeyEvent)
    }

    @Test
    func extraModifierIsADifferentChord() {
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.option, .shift], bindings: Self.bindings)
        #expect(verdict.fires.isEmpty)
        #expect(!verdict.suppressKeyEvent)
    }

    @Test
    func irrelevantModifiersAreIgnoredOnBothSides() {
        var matcher = HotkeyMatcher()
        // Caps lock alongside the chord still matches.
        let verdict = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.option, .capsLock],
            bindings: Self.bindings)
        #expect(verdict.fires == [down("dictation")])
    }

    @Test
    func sameKeyDifferentModifiersSelectsTheRightBinding() {
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.command, .shift],
            bindings: Self.bindings)
        #expect(verdict.fires == [down("agent")])
    }

    // MARK: - Release by key up

    @Test
    func keyUpFiresUpAndSuppresses() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        let verdict = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(verdict.fires == [up("dictation")])
        #expect(verdict.suppressKeyEvent)
        #expect(matcher.pressed.isEmpty)
    }

    @Test
    func keyUpIsModifierIndependent() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        // Modifiers already changed by the time the key lifts — still an up.
        let verdict = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: [], bindings: Self.bindings)
        #expect(verdict.fires == [up("dictation")])
    }

    @Test
    func unpressedKeyUpPassesThrough() {
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(verdict.fires.isEmpty)
        #expect(!verdict.suppressKeyEvent)
    }

    // MARK: - Release by modifier drop

    @Test
    func droppingAChordModifierReleases() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        let verdict = matcher.handle(
            .flagsChanged, keyCode: 0, modifiers: [], bindings: Self.bindings)
        #expect(verdict.fires == [up("dictation")])
        #expect(!verdict.suppressKeyEvent)
        #expect(matcher.pressed.isEmpty)
    }

    @Test
    func addingAModifierWhileHeldDoesNotRelease() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        // Option still held, shift added: chord modifiers all still down.
        let verdict = matcher.handle(
            .flagsChanged, keyCode: 0, modifiers: [.option, .shift], bindings: Self.bindings)
        #expect(verdict.fires.isEmpty)
        #expect(matcher.pressed == ["dictation"])
    }

    @Test
    func flagsChangedNeverSuppresses() {
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .flagsChanged, keyCode: 0, modifiers: .option, bindings: Self.bindings)
        #expect(!verdict.suppressKeyEvent)
    }

    // MARK: - A held key belongs to its hotkey

    @Test
    func addingShiftMidHoldSwallowsTheRepeatsWithoutFiring() {
        // ⌥Space held, ⇧ tapped to keep the take: the Space repeats now carry
        // ⌥⇧, and none of them may reach the app or start ⌥⇧Space.
        var matcher = HotkeyMatcher()
        let bindings = ["dictation": Self.dictation, "other": KeyCombo.optionShiftSpace]
        _ = matcher.handle(.keyDown, keyCode: Self.space, modifiers: .option, bindings: bindings)
        _ = matcher.handle(
            .flagsChanged, keyCode: 56, modifiers: [.option, .shift], bindings: bindings)
        let repeated = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.option, .shift], isRepeat: true,
            bindings: bindings)
        #expect(repeated.fires.isEmpty)
        #expect(repeated.suppressKeyEvent)
        #expect(matcher.pressed == ["dictation"])
    }

    @Test
    func aRepeatNeverStartsAnotherBinding() {
        // ⌥Space held, then ⌃ pressed: the repeating Space must not fire ⌃⌥Space.
        var matcher = HotkeyMatcher()
        let bindings = ["dictation": Self.dictation, "fix": KeyCombo.controlOptionSpace]
        _ = matcher.handle(.keyDown, keyCode: Self.space, modifiers: .option, bindings: bindings)
        _ = matcher.handle(
            .flagsChanged, keyCode: 59, modifiers: [.option, .control], bindings: bindings)
        let repeated = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.option, .control], isRepeat: true,
            bindings: bindings)
        #expect(repeated.fires.isEmpty)
        #expect(repeated.suppressKeyEvent)
        #expect(matcher.pressed == ["dictation"])
    }

    @Test
    func aRepeatOfAnAppsKeyPassesThroughWithoutFiring() {
        // Space held in the app (typing), then ⌥ pressed: the repeat that
        // now reads ⌥Space is still the app's key, not a dictation press.
        var matcher = HotkeyMatcher()
        let verdict = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, isRepeat: true,
            bindings: Self.bindings)
        #expect(verdict.fires.isEmpty)
        #expect(!verdict.suppressKeyEvent)
        #expect(matcher.pressed.isEmpty)
    }

    @Test
    func optionSpaceAndControlOptionSpaceAreDistinct() {
        let bindings = ["dictation": Self.dictation, "fix": KeyCombo.controlOptionSpace]

        var matcher = HotkeyMatcher()
        let fix = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [.control, .option], bindings: bindings)
        #expect(fix.fires == [down("fix")])
        #expect(fix.suppressKeyEvent)
        let fixUp = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: [.control, .option], bindings: bindings)
        #expect(fixUp.fires == [up("fix")])

        let dictation = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: bindings)
        #expect(dictation.fires == [down("dictation")])
        #expect(matcher.pressed == ["dictation"])
    }

    @Test
    func aKeyStillDownAfterItsModifierLeftIsSwallowedUntilItComesUp() {
        // ⌥ let go before Space: dictation ends, and the Space that is still
        // down (the app never saw it go down) keeps repeating into nothing.
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        let release = matcher.handle(
            .flagsChanged, keyCode: 58, modifiers: [], bindings: Self.bindings)
        #expect(release.fires == [up("dictation")])

        let repeated = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [], isRepeat: true, bindings: Self.bindings)
        #expect(repeated.fires.isEmpty)
        #expect(repeated.suppressKeyEvent)
        let keyUp = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: [], bindings: Self.bindings)
        #expect(keyUp.fires.isEmpty)
        #expect(keyUp.suppressKeyEvent)

        // The next Space is the owner's again.
        let typed = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: [], bindings: Self.bindings)
        #expect(typed.fires.isEmpty)
        #expect(!typed.suppressKeyEvent)
    }

    @Test
    func aFreshPressAfterAMissedKeyUpWorksAgain() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        _ = matcher.handle(.flagsChanged, keyCode: 58, modifiers: [], bindings: Self.bindings)
        // No key up arrived; a new (non-repeat) press starts dictation again.
        let again = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(again.fires == [down("dictation")])
        #expect(again.suppressKeyEvent)
    }

    @Test
    func theFnGraveHotkeyHoldsItsKeyThroughAShiftTap() {
        // A dictation hotkey on fn rather than ⌥: fn+` (key code 50).
        let grave: UInt16 = 50
        let bindings = ["dictation": KeyCombo(keyCode: grave, modifiers: .function)]
        var matcher = HotkeyMatcher()
        let press = matcher.handle(
            .keyDown, keyCode: grave, modifiers: .function, bindings: bindings)
        #expect(press.fires == [down("dictation")])
        _ = matcher.handle(
            .flagsChanged, keyCode: 56, modifiers: [.function, .shift], bindings: bindings)
        let repeated = matcher.handle(
            .keyDown, keyCode: grave, modifiers: [.function, .shift], isRepeat: true,
            bindings: bindings)
        #expect(repeated.fires.isEmpty)
        #expect(repeated.suppressKeyEvent)
        let shiftUp = matcher.handle(
            .flagsChanged, keyCode: 56, modifiers: .function, bindings: bindings)
        #expect(shiftUp.fires.isEmpty)
        let release = matcher.handle(
            .keyUp, keyCode: grave, modifiers: .function, bindings: bindings)
        #expect(release.fires == [up("dictation")])
    }

    // MARK: - Bookkeeping

    @Test
    func forgetDropsHeldStateWithoutFiring() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        matcher.forget(id: "dictation")
        #expect(matcher.pressed.isEmpty)
        // A later key up passes through: nothing is held anymore.
        let verdict = matcher.handle(
            .keyUp, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        #expect(verdict.fires.isEmpty)
        #expect(!verdict.suppressKeyEvent)
    }

    @Test
    func resetDropsEverything() {
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        _ = matcher.handle(
            .keyDown, keyCode: Self.fKey, modifiers: .function, bindings: Self.bindings)
        _ = matcher.handle(.flagsChanged, keyCode: 58, modifiers: [], bindings: Self.bindings)
        matcher.reset()
        #expect(matcher.pressed.isEmpty)
        #expect(matcher.stillDown.isEmpty)
    }

    @Test
    func unboundPressedIdSurvivesFlagsChanged() {
        // A binding removed mid-hold: the release scan skips it rather than
        // crashing or firing; `forget` is the owner of that cleanup.
        var matcher = HotkeyMatcher()
        _ = matcher.handle(
            .keyDown, keyCode: Self.space, modifiers: .option, bindings: Self.bindings)
        let verdict = matcher.handle(.flagsChanged, keyCode: 0, modifiers: [], bindings: [:])
        #expect(verdict.fires.isEmpty)
    }

    // MARK: - Sentinels

    @Test
    func doubleCommandSentinelIsInertHere() {
        // The Appshot chord is DoubleCommandDetector's job; its sentinel key
        // code must never match a real key event in the generic matcher.
        var matcher = HotkeyMatcher()
        let bindings = ["appshot": KeyCombo.doubleCommand]
        let downVerdict = matcher.handle(
            .keyDown, keyCode: 0, modifiers: .command, bindings: bindings)
        #expect(downVerdict.fires.isEmpty)
        let flagsVerdict = matcher.handle(
            .flagsChanged, keyCode: 0, modifiers: .command, bindings: bindings)
        #expect(flagsVerdict.fires.isEmpty)
    }
}
