//
//  ModifierKeyDetectorTests.swift
//  tesseractTests
//
//  The one-key hotkey as a decision table: a modifier pressed and released
//  on its own is down then up; another key or modifier joining while it is
//  held cancels it, so typing ⌥-something never calls Tesseract. Raw flag
//  words in, events out — no event tap, no NSEvent.
//

import Carbon.HIToolbox
import Testing

@testable import Tesseract_Agent

struct ModifierKeyDetectorTests {

    // Raw flag words as CGEventFlags/NSEvent.ModifierFlags deliver them: the
    // device-independent masks plus the left/right NX_DEVICE bits.
    private static let rightOptionDown: UInt64 = 0x8_0000 | 0x40
    private static let leftOptionDown: UInt64 = 0x8_0000 | 0x20
    private static let bothOptionsDown: UInt64 = 0x8_0000 | 0x60
    private static let rightOptionAndShift: UInt64 = 0x8_0000 | 0x40 | 0x2_0000 | 0x02
    private static let shiftDown: UInt64 = 0x2_0000 | 0x02
    private static let capsLockOnly: UInt64 = 0x1_0000
    private static let released: UInt64 = 0

    private static let rightOption = UInt16(kVK_RightOption)
    private static let leftOption = UInt16(kVK_Option)
    private static let shift = UInt16(kVK_Shift)

    private func detector() throws -> ModifierKeyDetector {
        try #require(ModifierKeyDetector(combo: .rightOption))
    }

    @Test func aTapIsDownThenUp() throws {
        var d = try detector()
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.rightOptionDown) == .down)
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.released) == .up)
    }

    @Test func capsLockDoesNotSpoilIt() throws {
        var d = try detector()
        #expect(
            d.flagsChanged(
                keyCode: Self.rightOption, rawFlags: Self.rightOptionDown | Self.capsLockOnly)
                == .down)
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.capsLockOnly) == .up)
    }

    @Test func typingWithItCancels() throws {
        var d = try detector()
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.rightOptionDown) == .down)
        #expect(d.keyDown() == .cancel)
        #expect(d.keyDown() == nil)
        // The release after a cancel reports nothing.
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.released) == nil)
        // And the next clean press works again.
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.rightOptionDown) == .down)
    }

    @Test func anotherModifierJoiningCancels() throws {
        var d = try detector()
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.rightOptionDown) == .down)
        #expect(d.flagsChanged(keyCode: Self.shift, rawFlags: Self.rightOptionAndShift) == .cancel)
        #expect(d.flagsChanged(keyCode: Self.shift, rawFlags: Self.rightOptionDown) == nil)
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.released) == nil)
    }

    @Test func pressedWhileAnotherModifierIsHeldIsNotAPress() throws {
        var d = try detector()
        #expect(d.flagsChanged(keyCode: Self.shift, rawFlags: Self.shiftDown) == nil)
        #expect(
            d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.rightOptionAndShift) == nil)
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.shiftDown) == nil)
    }

    @Test func theLeftOptionKeyIsNotTheRightOne() throws {
        var d = try detector()
        #expect(d.flagsChanged(keyCode: Self.leftOption, rawFlags: Self.leftOptionDown) == nil)
        #expect(d.flagsChanged(keyCode: Self.leftOption, rawFlags: Self.released) == nil)
        // Left held, then right: two Option keys, not the one-key hotkey.
        _ = d.flagsChanged(keyCode: Self.leftOption, rawFlags: Self.leftOptionDown)
        #expect(d.flagsChanged(keyCode: Self.rightOption, rawFlags: Self.bothOptionsDown) == nil)
    }

    @Test func onlyOneKeyCombosGetADetector() {
        #expect(ModifierKeyDetector(combo: .rightOption) != nil)
        #expect(ModifierKeyDetector(combo: KeyCombo(keyCode: UInt16(kVK_Function))) != nil)
        #expect(ModifierKeyDetector(combo: .optionShiftSpace) == nil)
        #expect(ModifierKeyDetector(combo: .doubleCommand) == nil)
        // The left Command and Shift keys carry too much typing.
        #expect(ModifierKeyDetector(combo: KeyCombo(keyCode: UInt16(kVK_Command))) == nil)
        #expect(ModifierKeyDetector(combo: KeyCombo(keyCode: UInt16(kVK_Shift))) == nil)
    }

    @Test func oneKeyCombosShowTheirKey() {
        #expect(KeyCombo.rightOption.displayString == "Right ⌥")
        #expect(KeyCombo.rightOption.isSingleModifier)
        #expect(KeyCombo(keyCode: UInt16(kVK_RightCommand)).displayString == "Right ⌘")
        #expect(!KeyCombo.optionShiftSpace.isSingleModifier)
    }

    @MainActor @Test func theCaptureHotkeyDefaultsToOneKey() {
        #expect(SettingsCatalogue.captureHotkeyKeyCode.default == Int(kVK_RightOption))
        #expect(SettingsCatalogue.captureHotkeyModifiers.default == 0)
    }
}
