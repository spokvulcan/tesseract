//
//  DoubleCommandDetectorTests.swift
//  tesseractTests
//
//  Tests the pure double-Command chord detector that backs the Appshot hotkey:
//  raw modifier-flag words in, a single fire per chord out. No event tap, no
//  NSEvent — the same detector serves both delivery paths.
//

import Testing

@testable import Tesseract_Agent

struct DoubleCommandDetectorTests {

    // Raw modifier-flag words as CGEventFlags/NSEvent.ModifierFlags deliver
    // them: the NX_ device-independent command mask plus the device-specific
    // left/right Command bits.
    private static let command: UInt64 = 0x10_0000
    private static let leftCommand = command | 0x08
    private static let rightCommand = command | 0x10
    private static let bothCommands = command | 0x18
    private static let shift: UInt64 = 0x2_0000
    private static let released: UInt64 = 0

    /// Run one flag-word sequence through a fresh detector, returning which
    /// events fired.
    private func fires(_ sequence: [UInt64]) -> [Bool] {
        var detector = DoubleCommandDetector()
        return sequence.map { detector.handleFlagsChanged(rawFlags: $0) }
    }

    @Test
    func bothCommandsInOneEventFiresOnce() {
        #expect(fires([Self.bothCommands, Self.bothCommands]) == [true, false])
    }

    @Test
    func leftThenRightFiresOnTheSecondEvent() {
        #expect(fires([Self.leftCommand, Self.bothCommands]) == [false, true])
    }

    @Test
    func singleCommandShortcutNeverFires() {
        // ⌘C: one Command down, keyDown for C changes no flags, Command up —
        // for either physical Command key.
        #expect(
            fires([Self.leftCommand, Self.released, Self.rightCommand, Self.released])
                == [false, false, false, false])
    }

    @Test
    func refireRequiresBothCommandsReleased() {
        // Fire; release only the right key and press it again: still one
        // chord. Full release re-arms.
        #expect(
            fires([
                Self.bothCommands, Self.leftCommand, Self.bothCommands,
                Self.released, Self.bothCommands,
            ]) == [true, false, false, false, true])
    }

    @Test
    func otherModifiersSuppressTheChord() {
        // Shift held alongside both Commands is a different chord. Releasing
        // shift while both Commands stay down completes a clean chord — the
        // detector is still armed, so it fires then.
        #expect(fires([Self.bothCommands | Self.shift, Self.bothCommands]) == [false, true])
    }

    @Test
    func addingAModifierAfterTheFireDoesNotRefire() {
        #expect(
            fires([Self.bothCommands, Self.bothCommands | Self.shift, Self.bothCommands])
                == [true, false, false])
    }

    @Test
    func commandWithoutDeviceBitsNeverFires() {
        // Synthetic events may carry the command mask with no device-specific
        // bits — there is no chord to detect there.
        #expect(fires([Self.command]) == [false])
    }
}
