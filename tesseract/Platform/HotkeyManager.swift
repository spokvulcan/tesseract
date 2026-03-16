//
//  HotkeyManager.swift
//  tesseract
//

import Foundation
import AppKit
import Combine
import Carbon.HIToolbox

struct HotkeyRegistration {
    let id: String
    let combo: KeyCombo
    let onDown: () -> Void
    let onUp: (() -> Void)?
}

@MainActor
final class HotkeyManager: ObservableObject {
    @Published var currentHotkey: KeyCombo = .optionSpace {
        didSet {
            // Keep backward compat: updating currentHotkey updates the "dictation" registration
            if var reg = registrations["dictation"] {
                reg = HotkeyRegistration(id: "dictation", combo: currentHotkey, onDown: reg.onDown, onUp: reg.onUp)
                registrations["dictation"] = reg
            }
        }
    }
    @Published private(set) var isListening = false
    @Published private(set) var isUsingEventTap = false

    var onHotkeyDown: (() -> Void)? {
        didSet {
            // Backward compat: sync to dictation registration
            syncDictationRegistration()
        }
    }
    var onHotkeyUp: (() -> Void)? {
        didSet {
            syncDictationRegistration()
        }
    }

    private var registrations: [String: HotkeyRegistration] = [:]
    private var pressedHotkeys: Set<String> = []

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?

    // Fallback monitors for when Accessibility permission is denied
    private var globalMonitor: Any?
    private var localMonitor: Any?

    init() {}

    deinit {
        MainActor.assumeIsolated {
            stopListening()
        }
    }

    // MARK: - Multi-Hotkey Registration

    func registerHotkey(id: String, combo: KeyCombo, onDown: @escaping () -> Void, onUp: (() -> Void)? = nil) {
        registrations[id] = HotkeyRegistration(id: id, combo: combo, onDown: onDown, onUp: onUp)
    }

    func unregisterHotkey(id: String) {
        registrations.removeValue(forKey: id)
        pressedHotkeys.remove(id)
    }

    func updateRegisteredHotkey(id: String, combo: KeyCombo) {
        guard var reg = registrations[id] else { return }
        reg = HotkeyRegistration(id: id, combo: combo, onDown: reg.onDown, onUp: reg.onUp)
        registrations[id] = reg
        pressedHotkeys.remove(id)
    }

    // MARK: - Listening

    func startListening() {
        guard !isListening else { return }

        // Try CGEventTap first (requires Accessibility permission)
        if AXIsProcessTrusted() {
            startEventTap()
        } else {
            // Fall back to NSEvent monitors (cannot suppress events)
            startNSEventMonitors()
        }

        isListening = true
    }

    func stopListening() {
        stopEventTap()
        stopNSEventMonitors()

        isListening = false
        pressedHotkeys.removeAll()
        isUsingEventTap = false
    }

    // MARK: - CGEventTap Implementation

    private func startEventTap() {
        let eventMask = (1 << CGEventType.keyDown.rawValue) |
                        (1 << CGEventType.keyUp.rawValue) |
                        (1 << CGEventType.flagsChanged.rawValue)

        // Store self pointer for callback
        let refcon = Unmanaged.passUnretained(self).toOpaque()

        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,  // Enables suppression (return nil to suppress)
            eventsOfInterest: CGEventMask(eventMask),
            callback: { proxy, type, event, refcon -> Unmanaged<CGEvent>? in
                guard let refcon = refcon else { return Unmanaged.passUnretained(event) }

                let manager = Unmanaged<HotkeyManager>.fromOpaque(refcon).takeUnretainedValue()

                // Handle tap being disabled by the system
                if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
                    if let tap = manager.eventTap {
                        CGEvent.tapEnable(tap: tap, enable: true)
                    }
                    return Unmanaged.passUnretained(event)
                }

                let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
                let flags = event.flags

                // Convert CGEventFlags to NSEvent.ModifierFlags
                var modifiers: NSEvent.ModifierFlags = []
                if flags.contains(.maskCommand) { modifiers.insert(.command) }
                if flags.contains(.maskAlternate) { modifiers.insert(.option) }
                if flags.contains(.maskControl) { modifiers.insert(.control) }
                if flags.contains(.maskShift) { modifiers.insert(.shift) }
                if flags.contains(.maskSecondaryFn) { modifiers.insert(.function) }

                let relevantMods: NSEvent.ModifierFlags = [.command, .option, .control, .shift, .function]
                let currentRelevant = modifiers.intersection(relevantMods)

                // Handle flagsChanged - check for released modifiers on pressed hotkeys
                if type == .flagsChanged {
                    for id in manager.pressedHotkeys {
                        guard let reg = manager.registrations[id] else { continue }
                        let hotkeyRelevant = NSEvent.ModifierFlags(rawValue: reg.combo.modifiers).intersection(relevantMods)
                        if !currentRelevant.contains(hotkeyRelevant) {
                            DispatchQueue.main.async {
                                guard manager.pressedHotkeys.contains(id) else { return }
                                manager.pressedHotkeys.remove(id)
                                reg.onUp?()
                            }
                        }
                    }
                    // ALWAYS pass through flagsChanged events
                    return Unmanaged.passUnretained(event)
                }

                // For keyDown/keyUp, find matching registration(s)
                var matched = false
                for (id, reg) in manager.registrations {
                    let hotkeyRelevant = NSEvent.ModifierFlags(rawValue: reg.combo.modifiers).intersection(relevantMods)

                    guard keyCode == reg.combo.keyCode else { continue }

                    if type == .keyUp {
                        if manager.pressedHotkeys.contains(id) {
                            DispatchQueue.main.async {
                                manager.pressedHotkeys.remove(id)
                                reg.onUp?()
                            }
                            matched = true
                        }
                    } else if type == .keyDown {
                        if currentRelevant == hotkeyRelevant {
                            if !manager.pressedHotkeys.contains(id) {
                                DispatchQueue.main.async {
                                    manager.pressedHotkeys.insert(id)
                                    reg.onDown()
                                }
                            }
                            matched = true
                        }
                    }
                }

                // Suppress the event if any registration matched
                if matched {
                    return nil
                }

                return Unmanaged.passUnretained(event)
            },
            userInfo: refcon
        )

        guard let eventTap = eventTap else {
            // Failed to create event tap, fall back to NSEvent monitors
            startNSEventMonitors()
            return
        }

        runLoopSource = CFMachPortCreateRunLoopSource(nil, eventTap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: eventTap, enable: true)

        isUsingEventTap = true
    }

    private func stopEventTap() {
        if let eventTap = eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
            CFMachPortInvalidate(eventTap)
            self.eventTap = nil
        }

        if let runLoopSource = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
            self.runLoopSource = nil
        }
    }

    // MARK: - NSEvent Fallback (No Suppression)

    private func startNSEventMonitors() {
        // Global monitor for when app is not focused
        globalMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.keyDown, .keyUp, .flagsChanged]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleKeyEvent(event)
            }
        }

        // Local monitor for when app is focused
        localMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown, .keyUp, .flagsChanged]
        ) { [weak self] event in
            Task { @MainActor in
                self?.handleKeyEvent(event)
            }
            return event
        }

        isUsingEventTap = false
    }

    private func stopNSEventMonitors() {
        if let monitor = globalMonitor {
            NSEvent.removeMonitor(monitor)
            globalMonitor = nil
        }

        if let monitor = localMonitor {
            NSEvent.removeMonitor(monitor)
            localMonitor = nil
        }
    }

    private func handleKeyEvent(_ event: NSEvent) {
        let keyCode = event.keyCode
        let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift, .function])
        let relevantMods: NSEvent.ModifierFlags = [.command, .option, .control, .shift, .function]

        // Check for releases on pressed hotkeys
        for id in pressedHotkeys {
            guard let reg = registrations[id] else { continue }
            let targetModifiers = NSEvent.ModifierFlags(rawValue: reg.combo.modifiers)
            let relevantTargetModifiers = targetModifiers.intersection(relevantMods)

            // Main key released
            if event.type == .keyUp && keyCode == reg.combo.keyCode {
                pressedHotkeys.remove(id)
                reg.onUp?()
                continue
            }

            // Modifier released
            if event.type == .flagsChanged {
                let currentRelevant = modifiers.intersection(relevantMods)
                if !currentRelevant.contains(relevantTargetModifiers) {
                    pressedHotkeys.remove(id)
                    reg.onUp?()
                    continue
                }
            }
        }

        // Handle modifier-only hotkeys
        if event.type == .flagsChanged {
            for (id, reg) in registrations where reg.combo.keyCode == 0 {
                let targetModifiers = NSEvent.ModifierFlags(rawValue: reg.combo.modifiers)
                if modifiers == targetModifiers && !pressedHotkeys.contains(id) {
                    pressedHotkeys.insert(id)
                    reg.onDown()
                } else if modifiers != targetModifiers && pressedHotkeys.contains(id) {
                    pressedHotkeys.remove(id)
                    reg.onUp?()
                }
            }
            return
        }

        // Hotkey down detection for each registration
        for (id, reg) in registrations {
            guard reg.combo.keyCode != 0 else { continue }
            let targetModifiers = NSEvent.ModifierFlags(rawValue: reg.combo.modifiers)
            let relevantTargetModifiers = targetModifiers.intersection(relevantMods)

            guard keyCode == reg.combo.keyCode,
                  modifiers.intersection(relevantMods) == relevantTargetModifiers else {
                continue
            }

            if event.type == .keyDown && !pressedHotkeys.contains(id) {
                pressedHotkeys.insert(id)
                reg.onDown()
            }
        }
    }

    // MARK: - Permission Handling

    func refreshForAccessibilityPermission() {
        guard isListening else { return }

        if !isUsingEventTap && AXIsProcessTrusted() {
            stopNSEventMonitors()
            startEventTap()
        }
    }

    // MARK: - Hotkey Recording

    func recordHotkey() async -> KeyCombo? {
        return await withCheckedContinuation { continuation in
            var hasResumed = false
            var monitor: Any?
            let shouldResumeListening = isListening

            if shouldResumeListening {
                stopListening()
            }

            func finish(_ result: KeyCombo?) {
                guard !hasResumed else { return }
                hasResumed = true
                if let monitor {
                    NSEvent.removeMonitor(monitor)
                }
                if shouldResumeListening {
                    Task { @MainActor in
                        self.startListening()
                    }
                }
                continuation.resume(returning: result)
            }

            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
                let keyCode = event.keyCode
                let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift, .function])

                // Escape cancels recording
                if keyCode == UInt16(kVK_Escape) {
                    finish(nil)
                    return nil
                }

                finish(KeyCombo(keyCode: keyCode, modifiers: modifiers))
                return nil
            }

            // Timeout after 10 seconds
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(10))
                finish(nil)
            }
        }
    }

    func updateHotkey(_ combo: KeyCombo) {
        currentHotkey = combo
        pressedHotkeys.removeAll()
    }

    // MARK: - Private

    private func syncDictationRegistration() {
        if let onDown = onHotkeyDown {
            registrations["dictation"] = HotkeyRegistration(
                id: "dictation",
                combo: currentHotkey,
                onDown: onDown,
                onUp: onHotkeyUp
            )
        }
    }
}
