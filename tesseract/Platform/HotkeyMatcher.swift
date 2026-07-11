//
//  HotkeyMatcher.swift
//  tesseract
//
//  The single fire-or-not decision for registered hotkeys, shared by both
//  delivery paths (CGEventTap and the NSEvent-monitor fallback). The paths
//  only normalize their events and deliver the returned fires — press,
//  release, modifier-drop, and key-repeat semantics live here once, so they
//  cannot drift between paths.
//
//  Deliberately not this module's job: the double-Command chord (its sentinel
//  key code never matches a key event; `DoubleCommandDetector` owns it), and
//  delivery timing (the tap adapter defers fires to the next main-queue turn
//  to keep the tap callback fast; the monitor adapter fires synchronously).
//

import AppKit

nonisolated struct HotkeyMatcher {

    enum EventKind: Equatable {
        case keyDown
        case keyUp
        case flagsChanged
    }

    struct Fire: Equatable {
        enum Direction: Equatable {
            case down
            case up
        }

        let id: String
        let direction: Direction
    }

    struct Verdict: Equatable {
        let fires: [Fire]
        /// Whether a suppressing delivery path (the event tap) should swallow
        /// the key event. Repeats of a held hotkey suppress without firing.
        /// Never true for `flagsChanged` — modifier changes always pass
        /// through to the system.
        let suppressKeyEvent: Bool
    }

    /// The modifiers that participate in matching; everything else (caps
    /// lock, numeric pad, …) is ignored on both sides of the comparison.
    static let relevantModifiers: NSEvent.ModifierFlags = [
        .command, .option, .control, .shift, .function,
    ]

    private(set) var pressed: Set<String> = []

    /// Fold one normalized key event over the bindings. Mutates the pressed
    /// set synchronously; the caller delivers the returned fires (looking the
    /// registration up again at delivery, so an unregister that lands before
    /// a deferred delivery quietly drops the fire).
    mutating func handle(
        _ kind: EventKind,
        keyCode: UInt16,
        modifiers: NSEvent.ModifierFlags,
        bindings: [String: KeyCombo]
    ) -> Verdict {
        let current = modifiers.intersection(Self.relevantModifiers)

        switch kind {
        case .flagsChanged:
            // A pressed hotkey releases as soon as its modifiers are no
            // longer all held (the main key may still be down).
            var fires: [Fire] = []
            for id in pressed.sorted() {
                guard let combo = bindings[id] else { continue }
                let target = combo.modifierFlags.intersection(Self.relevantModifiers)
                if !current.contains(target) {
                    pressed.remove(id)
                    fires.append(Fire(id: id, direction: .up))
                }
            }
            return Verdict(fires: fires, suppressKeyEvent: false)

        case .keyUp:
            // Release is modifier-independent: whatever the flags are by now,
            // lifting the main key ends a pressed hotkey. An unpressed
            // binding's key passes through unsuppressed.
            var fires: [Fire] = []
            var matched = false
            for (id, combo) in bindings.sorted(by: { $0.key < $1.key }) {
                guard combo.keyCode == keyCode, pressed.contains(id) else { continue }
                pressed.remove(id)
                fires.append(Fire(id: id, direction: .up))
                matched = true
            }
            return Verdict(fires: fires, suppressKeyEvent: matched)

        case .keyDown:
            // Down requires exact equality on the relevant modifiers — extra
            // held modifiers make a different chord.
            var fires: [Fire] = []
            var matched = false
            for (id, combo) in bindings.sorted(by: { $0.key < $1.key }) {
                guard combo.keyCode == keyCode else { continue }
                let target = combo.modifierFlags.intersection(Self.relevantModifiers)
                guard current == target else { continue }
                matched = true
                if !pressed.contains(id) {
                    pressed.insert(id)
                    fires.append(Fire(id: id, direction: .down))
                }
            }
            return Verdict(fires: fires, suppressKeyEvent: matched)
        }
    }

    /// Drop any held state for one binding (unregister, rebind).
    mutating func forget(id: String) {
        pressed.remove(id)
    }

    /// Drop all held state (listening stopped, hotkey re-recorded).
    mutating func reset() {
        pressed.removeAll()
    }
}
