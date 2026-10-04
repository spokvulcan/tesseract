//
//  ModifierTapDetector.swift
//  tesseract
//
//  A tap of Shift while a hotkey is held: the gesture that keeps a take in
//  the **Lens** instead of pasting it (PRD #612, "tap ⇧ while talking").
//  Either Shift key counts. A tap is Shift going down and coming back up
//  with no key pressed in between and no other modifier changing; anything
//  else is the owner typing a capital or a shifted chord, and is not a tap.
//
//  Fed every `flagsChanged` event with its key code and raw flag word, and
//  every key-down. Auto-repeats do not spoil a tap: the held hotkey's own
//  key repeats the whole time it is held. Whether a registration is held is
//  the caller's question (the hotkey manager asks its matcher); this type
//  only says a tap happened.
//
//  Works on the raw flag word like `ModifierKeyDetector`: the NX_DEVICE bits
//  say which modifier keys are down, left and right.
//

import Carbon.HIToolbox

nonisolated struct ModifierTapDetector {

    /// The key codes of the two Shift keys.
    static let shiftKeys: Set<UInt16> = [UInt16(kVK_Shift), UInt16(kVK_RightShift)]
    /// Shift is down: the device-independent mask, or either side's device
    /// bit (left 0x02, right 0x04).
    static let shiftBits: UInt64 = 0x2_0000 | 0x02 | 0x04
    /// Every other modifier key, by its device bits (both sides of Control,
    /// Option and Command) plus the device-independent masks of Control,
    /// Option, Command and fn (fn has no device bit). Caps Lock is left out:
    /// it is a toggle, not a held modifier.
    static let otherModifierBits: UInt64 =
        (ModifierKeyDetector.deviceBits & ~0x06) | ModifierKeyDetector.functionMask
        | 0x4_0000 | 0x8_0000 | 0x10_0000

    private var state: State = .idle

    private enum State: Equatable {
        case idle
        /// Shift went down with these other modifiers held.
        case down(others: UInt64)
        /// Something else happened while Shift was down; waiting for it to
        /// come up.
        case spoiled
    }

    /// Feed one `flagsChanged` event (any modifier's). Returns true when it
    /// completes a tap.
    mutating func flagsChanged(keyCode changed: UInt16, rawFlags: UInt64) -> Bool {
        let shiftDown = rawFlags & Self.shiftBits != 0
        let others = rawFlags & Self.otherModifierBits
        switch state {
        case .idle:
            if Self.shiftKeys.contains(changed), shiftDown {
                state = .down(others: others)
            }
            return false
        case .down(let before):
            if others != before {
                state = shiftDown ? .spoiled : .idle
                return false
            }
            if !shiftDown {
                state = .idle
                return true
            }
            // The other Shift key joined or left: two Shifts, not a tap.
            if Self.shiftKeys.contains(changed) { state = .spoiled }
            return false
        case .spoiled:
            if !shiftDown { state = .idle }
            return false
        }
    }

    /// Feed one key-down (any key). A key pressed while Shift is down is a
    /// shifted key, not a tap; an auto-repeat is not a new press.
    mutating func keyDown(isRepeat: Bool) {
        guard !isRepeat, case .down = state else { return }
        state = .spoiled
    }

    /// Forget a press in progress (listening stopped).
    mutating func reset() {
        state = .idle
    }
}
