//
//  KeyCombo.swift
//  whisper-on-device
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
        switch Int(keyCode) {
        case kVK_F1: return "F1"
        case kVK_F2: return "F2"
        case kVK_F3: return "F3"
        case kVK_F4: return "F4"
        case kVK_F5: return "F5"
        case kVK_F6: return "F6"
        case kVK_F7: return "F7"
        case kVK_F8: return "F8"
        case kVK_F9: return "F9"
        case kVK_F10: return "F10"
        case kVK_F11: return "F11"
        case kVK_F12: return "F12"
        case kVK_Space: return "Space"
        case kVK_Return: return "↩"
        case kVK_Escape: return "⎋"
        default:
            if let scalar = UnicodeScalar(keyCode) {
                return String(Character(scalar)).uppercased()
            }
            return "Key\(keyCode)"
        }
    }

    // Common presets
    static let f5 = KeyCombo(keyCode: UInt16(kVK_F5))
    static let optionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .option)
    static let controlSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .control)
    static let functionSpace = KeyCombo(keyCode: UInt16(kVK_Space), modifiers: .function)
}
