//
//  SyntheticKeyEvents.swift
//  tesseract
//
//  Every keyboard event Tesseract posts (the paste of a Clipboard Loan, the
//  copy that reads a selection, the Delete and backspaces of a fix put back
//  in an app) carries one marker in its source user-data field, so the
//  hotkey manager's event tap can tell them from the owner's typing: they
//  never match a hotkey and never count as a key the owner pressed in an
//  app since a paste (PRD #612).
//

import CoreGraphics

nonisolated enum SyntheticKeyEvents {

    /// Stamped in `.eventSourceUserData` of every key event Tesseract posts:
    /// "TESS" in ASCII. A real keyboard leaves the field zero.
    static let marker: Int64 = 0x5445_5353

    /// Post one key press, down then up, with exactly `flags` held (the
    /// owner's physical modifiers are not inherited) and the marker set.
    static func post(keyCode: CGKeyCode, flags: CGEventFlags = []) {
        let source = CGEventSource(stateID: .hidSystemState)
        for isDown in [true, false] {
            guard
                let event = CGEvent(
                    keyboardEventSource: source, virtualKey: keyCode, keyDown: isDown)
            else { continue }
            event.flags = flags
            event.setIntegerValueField(.eventSourceUserData, value: marker)
            event.post(tap: .cghidEventTap)
        }
    }

    /// Whether Tesseract posted this event. Cheap enough for the event tap's
    /// callback: one field read, no allocation.
    static func isOurs(_ event: CGEvent) -> Bool {
        event.getIntegerValueField(.eventSourceUserData) == marker
    }
}
