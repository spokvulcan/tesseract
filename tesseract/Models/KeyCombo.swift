//
//  KeyCombo.swift
//  tesseract
//

import Foundation
import AppKit
import Carbon.HIToolbox

struct KeyCombo: Codable, Equatable, Sendable {
    let keyCode: UInt16
    let modifiers: UInt

    init(keyCode: UInt16, modifiers: NSEvent.ModifierFlags = []) {
        self.keyCode = keyCode
        self.modifiers = modifiers.rawValue
    }

    var modifierFlags: NSEvent.ModifierFlags {
        NSEvent.ModifierFlags(rawValue: modifiers)
    }

    var displayString: String {
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
        UInt16(kVK_DownArrow): "↓"
    ]

    private static func translateKeyCode(_ keyCode: UInt16) -> String? {
        guard let inputSource = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
              let layoutData = TISGetInputSourceProperty(
                inputSource,
                kTISPropertyUnicodeKeyLayoutData
              ) else {
            return nil
        }

        let data = unsafeBitCast(layoutData, to: CFData.self)
        guard let layoutPtr = CFDataGetBytePtr(data) else {
            return nil
        }
        let keyboardLayout = layoutPtr.withMemoryRebound(to: UCKeyboardLayout.self, capacity: 1) { $0 }

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

    // Common presets
    static let f5 = KeyCombo(keyCode: UInt16(kVK_F5))
    static let optionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .option)
    static let controlSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .control)
    static let functionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .function)
}
