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
    @Published var currentHotkey: KeyCombo = .f5
    @Published private(set) var isListening = false

    var onHotkeyDown: (() -> Void)?
    var onHotkeyUp: (() -> Void)?

    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var isHotkeyPressed = false

    init() {}

    deinit {
        MainActor.assumeIsolated {
            stopListening()
        }
    }

    func startListening() {
        guard !isListening else { return }

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

        isListening = true
    }

    func stopListening() {
        if let monitor = globalMonitor {
            NSEvent.removeMonitor(monitor)
            globalMonitor = nil
        }

        if let monitor = localMonitor {
            NSEvent.removeMonitor(monitor)
            localMonitor = nil
        }

        isListening = false
        isHotkeyPressed = false
    }

    private func handleKeyEvent(_ event: NSEvent) {
        let keyCode = event.keyCode
        let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift])

        // Check if this matches our hotkey
        guard keyCode == currentHotkey.keyCode,
              modifiers.rawValue == currentHotkey.modifiers else {
            return
        }

        switch event.type {
        case .keyDown:
            if !isHotkeyPressed {
                isHotkeyPressed = true
                onHotkeyDown?()
            }

        case .keyUp:
            if isHotkeyPressed {
                isHotkeyPressed = false
                onHotkeyUp?()
            }

        case .flagsChanged:
            // Handle modifier-only hotkeys (like Option key alone)
            if currentHotkey.keyCode == 0 {
                let currentModifiers = event.modifierFlags.intersection([.command, .option, .control, .shift])
                let targetModifiers = NSEvent.ModifierFlags(rawValue: currentHotkey.modifiers)

                if currentModifiers == targetModifiers && !isHotkeyPressed {
                    isHotkeyPressed = true
                    onHotkeyDown?()
                } else if currentModifiers != targetModifiers && isHotkeyPressed {
                    isHotkeyPressed = false
                    onHotkeyUp?()
                }
            }

        default:
            break
        }
    }

    // MARK: - Hotkey Recording

    func recordHotkey() async -> KeyCombo? {
        return await withCheckedContinuation { continuation in
            var recordedCombo: KeyCombo?

            let monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
                let keyCode = event.keyCode
                let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift])

                // Escape cancels recording
                if keyCode == UInt16(kVK_Escape) {
                    continuation.resume(returning: nil)
                    return nil
                }

                recordedCombo = KeyCombo(keyCode: keyCode, modifiers: modifiers)
                continuation.resume(returning: recordedCombo)
                return nil
            }

            // Timeout after 10 seconds
            Task {
                try? await Task.sleep(for: .seconds(10))
                if let monitor {
                    NSEvent.removeMonitor(monitor)
                    continuation.resume(returning: nil)
                }
            }

            // Store reference to remove later
            _ = monitor
        }
    }
}
