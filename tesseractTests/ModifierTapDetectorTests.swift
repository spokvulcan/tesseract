//
//  ModifierTapDetectorTests.swift
//  tesseractTests
//
//  The Shift tap as a decision table: Shift down and up again with nothing
//  pressed between and no other modifier changing is a tap (the "keep this
//  take in the Lens" gesture while the dictation hotkey is held); a shifted
//  key, a second Shift or a modifier joining or leaving is not. Raw flag
//  words in, taps out: no event tap, no NSEvent.
//

import Carbon.HIToolbox
import Testing

@testable import Tesseract_Agent

struct ModifierTapDetectorTests {

    // Raw flag words as CGEventFlags/NSEvent.ModifierFlags deliver them: the
    // device-independent masks plus the left/right NX_DEVICE bits.
    private static let option: UInt64 = 0x8_0000 | 0x20
    private static let function: UInt64 = 0x80_0000
    private static let control: UInt64 = 0x4_0000 | 0x01
    private static let leftShift: UInt64 = 0x2_0000 | 0x02
    private static let rightShift: UInt64 = 0x2_0000 | 0x04
    private static let bothShifts: UInt64 = 0x2_0000 | 0x06
    private static let capsLock: UInt64 = 0x1_0000

    private static let leftShiftKey = UInt16(kVK_Shift)
    private static let rightShiftKey = UInt16(kVK_RightShift)
    private static let controlKey = UInt16(kVK_Control)
    private static let optionKey = UInt16(kVK_Option)
    private static let capsLockKey = UInt16(kVK_CapsLock)

    @Test func shiftDownAndUpWhileOptionIsHeldIsATap() {
        var d = ModifierTapDetector()
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(tapped)
        }
    }

    @Test func eitherShiftKeyTaps() {
        var d = ModifierTapDetector()
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.rightShiftKey, rawFlags: Self.function | Self.rightShift)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.rightShiftKey, rawFlags: Self.function)
            #expect(tapped)
        }
    }

    @Test func theHeldHotkeysRepeatsDoNotSpoilATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        d.keyDown(isRepeat: true)
        d.keyDown(isRepeat: true)
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(tapped)
        }
    }

    @Test func aKeyPressedWithShiftIsAShiftedKeyNotATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        d.keyDown(isRepeat: false)
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(!tapped)
        }
        // And the next clean tap works again.
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(tapped)
        }
    }

    @Test func aModifierJoiningWhileShiftIsDownIsNotATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.controlKey, rawFlags: Self.option | Self.leftShift | Self.control)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.controlKey, rawFlags: Self.option | Self.leftShift)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(!tapped)
        }
    }

    @Test func theHotkeysModifierLeavingWhileShiftIsDownIsNotATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        do {
            let tapped = d.flagsChanged(keyCode: Self.optionKey, rawFlags: Self.leftShift)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: 0)
            #expect(!tapped)
        }
    }

    @Test func bothShiftKeysAreNotATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.rightShiftKey, rawFlags: Self.option | Self.bothShifts)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.rightShiftKey, rawFlags: Self.option | Self.leftShift)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(!tapped)
        }
    }

    @Test func capsLockDoesNotSpoilATap() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.capsLockKey, rawFlags: Self.option | Self.leftShift | Self.capsLock)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(
                keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.capsLock)
            #expect(tapped)
        }
    }

    @Test func otherModifiersAloneNeverTap() {
        var d = ModifierTapDetector()
        do {
            let tapped = d.flagsChanged(keyCode: Self.optionKey, rawFlags: Self.option)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.optionKey, rawFlags: 0)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.controlKey, rawFlags: Self.control)
            #expect(!tapped)
        }
        do {
            let tapped = d.flagsChanged(keyCode: Self.controlKey, rawFlags: 0)
            #expect(!tapped)
        }
    }

    @Test func resetForgetsAShiftInProgress() {
        var d = ModifierTapDetector()
        _ = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option | Self.leftShift)
        d.reset()
        do {
            let tapped = d.flagsChanged(keyCode: Self.leftShiftKey, rawFlags: Self.option)
            #expect(!tapped)
        }
    }
}
