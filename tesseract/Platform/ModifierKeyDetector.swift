//
//  ModifierKeyDetector.swift
//  tesseract
//
//  The one-key hotkey: a modifier pressed on its own (Right Option, say).
//  Fed every `flagsChanged` event with its key code and raw flag word, and
//  every key-down: the hotkey is down when its key goes down with nothing
//  else held, up when it comes back up, and cancelled when any other key or
//  modifier joins while it is held — the owner is typing ⌥-something, not
//  calling Tesseract.
//
//  Works on the raw flag word for the same reason as the double-Command
//  detector: left and right live only in the device-specific NX_DEVICE bits,
//  which `CGEventFlags` and `NSEvent.modifierFlags` both carry.
//

import Foundation

nonisolated struct ModifierKeyDetector {

    enum Event: Equatable {
        case down
        case up
        case cancel
    }

    /// NX_DEVICE bits of every modifier key, left and right.
    static let deviceBits: UInt64 = 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x2000
    /// The device-independent fn (Globe) mask; fn has no device bit.
    static let functionMask: UInt64 = 0x80_0000

    let keyCode: UInt16
    private let mask: UInt64
    private var state: State = .idle

    private enum State {
        case idle
        /// Down alone; `down` was reported.
        case held
        /// Another key joined; `cancel` was reported, waiting for release.
        case spoiled
    }

    /// A detector for a one-key combo, or nil for any other combo.
    init?(combo: KeyCombo) {
        guard combo.isSingleModifier, let key = KeyCombo.singleModifierKeys[combo.keyCode] else {
            return nil
        }
        keyCode = combo.keyCode
        mask = key.mask
    }

    /// Feed one `flagsChanged` event (any modifier's).
    mutating func flagsChanged(keyCode changed: UInt16, rawFlags: UInt64) -> Event? {
        let isDown = rawFlags & mask != 0
        let others = (rawFlags & (Self.deviceBits | Self.functionMask)) & ~mask
        switch state {
        case .idle:
            guard changed == keyCode, isDown, others == 0 else { return nil }
            state = .held
            return .down
        case .held:
            if !isDown {
                state = .idle
                return .up
            }
            guard others != 0 else { return nil }
            state = .spoiled
            return .cancel
        case .spoiled:
            if !isDown { state = .idle }
            return nil
        }
    }

    /// Feed one key-down (any key): a key typed while held spoils it.
    mutating func keyDown() -> Event? {
        guard state == .held else { return nil }
        state = .spoiled
        return .cancel
    }

    /// Forget a press in progress (listening stopped).
    mutating func reset() {
        state = .idle
    }
}
