//
//  DoubleCommandDetector.swift
//  tesseract
//

import Foundation

/// The pure chord detector behind the Appshot hotkey's double-Command default:
/// fed the raw modifier-flag word of every `flagsChanged` event, it reports the
/// moment both Command keys are down together — once per chord, re-armed only
/// after both keys are released.
///
/// Works on the raw flag word (not `NSEvent.ModifierFlags`) because the left/
/// right distinction lives in the device-specific NX_DEVICE bits, which both
/// `CGEventFlags` and `NSEvent.modifierFlags` carry in their raw values — one
/// detector serves the event-tap path and the NSEvent-monitor fallback.
nonisolated struct DoubleCommandDetector {

    /// NX_DEVICELCMDKEYMASK / NX_DEVICERCMDKEYMASK.
    private static let leftCommandBit: UInt64 = 0x08
    private static let rightCommandBit: UInt64 = 0x10
    private static let bothCommandBits = leftCommandBit | rightCommandBit

    /// The device-independent shift/control/option/fn masks — any of these held
    /// alongside the two Command keys makes it a different chord, not this one.
    private static let otherModifierMask: UInt64 = 0x2_0000 | 0x4_0000 | 0x8_0000 | 0x80_0000

    private var isArmed = true

    /// Feed one `flagsChanged` raw flag word; returns true when the chord fires.
    mutating func handleFlagsChanged(rawFlags: UInt64) -> Bool {
        let bothDown = rawFlags & Self.bothCommandBits == Self.bothCommandBits
        let othersClear = rawFlags & Self.otherModifierMask == 0

        if bothDown && othersClear {
            guard isArmed else { return false }
            isArmed = false
            return true
        }
        if rawFlags & Self.bothCommandBits == 0 {
            isArmed = true
        }
        return false
    }
}
