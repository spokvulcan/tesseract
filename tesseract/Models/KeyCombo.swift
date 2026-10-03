//
//  KeyCombo.swift
//  tesseract
//

import Foundation
import AppKit
import Carbon.HIToolbox

nonisolated struct KeyCombo: Codable, Equatable, Sendable {
    let keyCode: UInt16
    let modifiers: UInt

    init(keyCode: UInt16, modifiers: NSEvent.ModifierFlags = []) {
        self.keyCode = keyCode
        self.modifiers = modifiers.rawValue
    }

    var modifierFlags: NSEvent.ModifierFlags {
        NSEvent.ModifierFlags(rawValue: modifiers)
    }

    /// The double-Command chord (the Appshot default) — both Command keys
    /// pressed together. Not a real key: the sentinel key code never matches a
    /// keyboard event, so the chord is detected on modifier flags alone.
    var isDoubleCommand: Bool { keyCode == Self.doubleCommandKeyCode }

    /// A modifier key on its own — the one-key hotkey (Right Option, say).
    /// Stored as the modifier's own key code with no modifiers: a key press
    /// never carries a modifier's key code, so it can't collide with a key
    /// combo, and it is detected on modifier flags (`ModifierKeyDetector`).
    var isSingleModifier: Bool { modifiers == 0 && Self.singleModifierKeys[keyCode] != nil }

    var displayString: String {
        if isDoubleCommand { return "⌘⌘" }
        if isSingleModifier, let key = Self.singleModifierKeys[keyCode] { return key.name }

        var parts: [String] = []

        let flags = modifierFlags
        if flags.contains(.control) { parts.append("⌃") }
        if flags.contains(.option) { parts.append("⌥") }
        if flags.contains(.shift) { parts.append("⇧") }
        if flags.contains(.command) { parts.append("⌘") }
        if flags.contains(.function) { parts.append("fn") }

        parts.append(keyCodeToString(keyCode))

        return parts.joined()
    }

    private func keyCodeToString(_ keyCode: UInt16) -> String {
        if let special = KeyCombo.specialKeyStrings[keyCode] {
            return special
        }

        if let translated = KeyCombo.translateKeyCode(keyCode) {
            return translated.uppercased()
        }

        return "Key\(keyCode)"
    }

    private static let specialKeyStrings: [UInt16: String] = [
        UInt16(kVK_F1): "F1",
        UInt16(kVK_F2): "F2",
        UInt16(kVK_F3): "F3",
        UInt16(kVK_F4): "F4",
        UInt16(kVK_F5): "F5",
        UInt16(kVK_F6): "F6",
        UInt16(kVK_F7): "F7",
        UInt16(kVK_F8): "F8",
        UInt16(kVK_F9): "F9",
        UInt16(kVK_F10): "F10",
        UInt16(kVK_F11): "F11",
        UInt16(kVK_F12): "F12",
        UInt16(kVK_Space): "Space",
        UInt16(kVK_Return): "↩",
        UInt16(kVK_Escape): "⎋",
        UInt16(kVK_Delete): "⌫",
        UInt16(kVK_ForwardDelete): "⌦",
        UInt16(kVK_Tab): "⇥",
        UInt16(kVK_Home): "↖",
        UInt16(kVK_End): "↘",
        UInt16(kVK_PageUp): "⇞",
        UInt16(kVK_PageDown): "⇟",
        UInt16(kVK_LeftArrow): "←",
        UInt16(kVK_RightArrow): "→",
        UInt16(kVK_UpArrow): "↑",
        UInt16(kVK_DownArrow): "↓",
    ]

    private static func translateKeyCode(_ keyCode: UInt16) -> String? {
        guard let inputSource = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
            let layoutData = TISGetInputSourceProperty(
                inputSource,
                kTISPropertyUnicodeKeyLayoutData
            )
        else {
            return nil
        }

        let data = unsafeBitCast(layoutData, to: CFData.self)
        guard let layoutPtr = CFDataGetBytePtr(data) else {
            return nil
        }
        let keyboardLayout = layoutPtr.withMemoryRebound(to: UCKeyboardLayout.self, capacity: 1) {
            $0
        }

        var deadKeyState: UInt32 = 0
        var length: Int = 0
        var chars: [UniChar] = Array(repeating: 0, count: 8)

        let status = UCKeyTranslate(
            keyboardLayout,
            keyCode,
            UInt16(kUCKeyActionDisplay),
            0,
            UInt32(LMGetKbdType()),
            UInt32(kUCKeyTranslateNoDeadKeysBit),
            &deadKeyState,
            chars.count,
            &length,
            &chars
        )

        guard status == noErr, length > 0 else {
            return nil
        }

        return String(utf16CodeUnits: chars, count: length)
    }

    /// The modifiers that work as a hotkey on their own, with the flag bit
    /// that says each is down: the right-hand modifiers (rarely used for
    /// shortcuts) and fn. The left Command and Shift keys are left alone —
    /// too much typing rides on them.
    static let singleModifierKeys: [UInt16: (name: String, mask: UInt64)] = [
        UInt16(kVK_RightOption): ("Right ⌥", 0x40),
        UInt16(kVK_RightCommand): ("Right ⌘", 0x10),
        UInt16(kVK_RightControl): ("Right ⌃", 0x2000),
        UInt16(kVK_RightShift): ("Right ⇧", 0x04),
        UInt16(kVK_Function): ("fn", 0x80_0000),
    ]

    // Common presets
    static let doubleCommandKeyCode: UInt16 = .max
    static let doubleCommand = KeyCombo(keyCode: doubleCommandKeyCode, modifiers: .command)
    /// One key, alone: the capture default.
    static let rightOption = KeyCombo(keyCode: UInt16(kVK_RightOption))
    static let f5 = KeyCombo(keyCode: UInt16(kVK_F5))
    static let optionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .option)
    static let controlSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .control)
    /// ⌃⌥Space: the fix hotkey, which reopens the last take in the **Lens**.
    static let controlOptionSpace = KeyCombo(
        keyCode: UInt16(kVK_Space), modifiers: [.control, .option])
    static let functionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .function)
    static let optionShiftSpace = KeyCombo(
        keyCode: UInt16(kVK_Space), modifiers: [.option, .shift])
}
