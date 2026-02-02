//
//  HotkeyManager.swift
//  whisper-on-device
//

import Foundation
import AppKit
import Combine
import Carbon.HIToolbox

@MainActor
final class HotkeyManager: ObservableObject {
    @Published var currentHotkey: KeyCombo = .optionSpace
    @Published private(set) var isListening = false
    @Published private(set) var isUsingEventTap = false

    var onHotkeyDown: (() -> Void)?
    var onHotkeyUp: (() -> Void)?

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var isHotkeyPressed = false

    // Fallback monitors for when Accessibility permission is denied
    private var globalMonitor: Any?
    private var localMonitor: Any?

    init() {}

    deinit {
        MainActor.assumeIsolated {
            stopListening()
        }
    }

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
        isHotkeyPressed = false
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

                // Check if this event matches our hotkey
                let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
                let flags = event.flags

                // Convert CGEventFlags to NSEvent.ModifierFlags for comparison
                var modifiers: NSEvent.ModifierFlags = []
                if flags.contains(.maskCommand) { modifiers.insert(.command) }
                if flags.contains(.maskAlternate) { modifiers.insert(.option) }
                if flags.contains(.maskControl) { modifiers.insert(.control) }
                if flags.contains(.maskShift) { modifiers.insert(.shift) }
                if flags.contains(.maskSecondaryFn) { modifiers.insert(.function) }

                // Get the expected modifiers (only the relevant ones for comparison)
                let relevantModifiers: NSEvent.ModifierFlags = [.command, .option, .control, .shift, .function]
                let currentModifiers = modifiers.intersection(relevantModifiers)

                // Capture hotkey configuration for comparison
                let hotkeyKeyCode = manager.currentHotkey.keyCode
                let hotkeyModifiers = NSEvent.ModifierFlags(rawValue: manager.currentHotkey.modifiers)
                let relevantHotkeyModifiers = hotkeyModifiers.intersection([.command, .option, .control, .shift])

                // Check for release conditions (main key released or modifier released)
                let isMainKeyUp = type == .keyUp && keyCode == hotkeyKeyCode
                let isModifierReleased = type == .flagsChanged && !currentModifiers.contains(relevantHotkeyModifiers)

                if isMainKeyUp || isModifierReleased {
                    DispatchQueue.main.async {
                        guard manager.isHotkeyPressed else { return }
                        manager.isHotkeyPressed = false
                        manager.onHotkeyUp?()
                    }
                    // Suppress main key up, pass through modifier changes
                    return isMainKeyUp ? nil : Unmanaged.passUnretained(event)
                }

                // Normal hotkey down detection (requires exact match)
                let matches = keyCode == hotkeyKeyCode &&
                              currentModifiers.rawValue == hotkeyModifiers.rawValue

                if matches && type == .keyDown {
                    DispatchQueue.main.async {
                        guard !manager.isHotkeyPressed else { return }
                        manager.isHotkeyPressed = true
                        manager.onHotkeyDown?()
                    }
                    return nil  // Suppress the key down event
                }

                // Pass through non-hotkey events
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
        // Note: This uses NSEvent.addGlobalMonitorForEvents which works in sandbox
        // but cannot suppress/intercept keys (only observe)
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
        let targetModifiers = NSEvent.ModifierFlags(rawValue: currentHotkey.modifiers)
        let relevantTargetModifiers = targetModifiers.intersection([.command, .option, .control, .shift])

        // When engaged, check for any release that should stop recording
        if isHotkeyPressed {
            // Main key released (regardless of modifier state)
            if event.type == .keyUp && keyCode == currentHotkey.keyCode {
                isHotkeyPressed = false
                onHotkeyUp?()
                return
            }

            // Modifier released (via flagsChanged)
            if event.type == .flagsChanged {
                let currentRelevant = modifiers.intersection([.command, .option, .control, .shift])
                if !currentRelevant.contains(relevantTargetModifiers) {
                    isHotkeyPressed = false
                    onHotkeyUp?()
                    return
                }
            }
        }

        // Handle modifier-only hotkeys (like Option key alone)
        if currentHotkey.keyCode == 0 && event.type == .flagsChanged {
            if modifiers == targetModifiers && !isHotkeyPressed {
                isHotkeyPressed = true
                onHotkeyDown?()
            } else if modifiers != targetModifiers && isHotkeyPressed {
                isHotkeyPressed = false
                onHotkeyUp?()
            }
            return
        }

        // Hotkey down detection (requires exact match)
        guard keyCode == currentHotkey.keyCode,
              modifiers.rawValue == currentHotkey.modifiers else {
            return
        }

        if event.type == .keyDown && !isHotkeyPressed {
            isHotkeyPressed = true
            onHotkeyDown?()
        }
    }

    // MARK: - Permission Handling

    /// Restarts listening with event tap if Accessibility permission was granted
    func refreshForAccessibilityPermission() {
        guard isListening else { return }

        // If we're not using event tap but now have permission, upgrade
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

            func finish(_ result: KeyCombo?) {
                guard !hasResumed else { return }
                hasResumed = true
                if let monitor {
                    NSEvent.removeMonitor(monitor)
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
}
