//
//  TextInjector.swift
//  whisper-on-device
//

import Foundation
import Combine
import AppKit
import Carbon.HIToolbox

@MainActor
final class TextInjector: ObservableObject {
    @Published private(set) var lastInjectionSucceeded = false

    private var savedClipboardContents: [NSPasteboard.PasteboardType: Data]?
    var restoreClipboard = true

    func inject(_ text: String) async throws {
        guard !text.isEmpty else {
            throw DictationError.textInjectionFailed("Empty text")
        }

        // Save clipboard contents if restoration is enabled
        if restoreClipboard {
            saveClipboardContents()
        }

        // Copy text to clipboard
        copyToClipboard(text)

        // Small delay to ensure clipboard is updated
        try await Task.sleep(for: .milliseconds(50))

        // Simulate Cmd+V paste
        simulatePaste()

        // Small delay before restoring clipboard
        try await Task.sleep(for: .milliseconds(100))

        // Restore original clipboard contents
        if restoreClipboard {
            restoreClipboardContents()
        }

        lastInjectionSucceeded = true
    }

    // MARK: - Clipboard Operations

    private func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private func saveClipboardContents() {
        let pasteboard = NSPasteboard.general
        var contents: [NSPasteboard.PasteboardType: Data] = [:]

        for type in pasteboard.types ?? [] {
            if let data = pasteboard.data(forType: type) {
                contents[type] = data
            }
        }

        savedClipboardContents = contents.isEmpty ? nil : contents
    }

    private func restoreClipboardContents() {
        guard let contents = savedClipboardContents else { return }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()

        for (type, data) in contents {
            pasteboard.setData(data, forType: type)
        }

        savedClipboardContents = nil
    }

    // MARK: - Key Simulation

    private func simulatePaste() {
        let source = CGEventSource(stateID: .hidSystemState)

        // Virtual key code for 'V'
        let vKeyCode: CGKeyCode = CGKeyCode(kVK_ANSI_V)

        // Key down: Cmd+V
        if let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true) {
            keyDown.flags = .maskCommand
            keyDown.post(tap: .cghidEventTap)
        }

        // Key up: Cmd+V
        if let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false) {
            keyUp.flags = .maskCommand
            keyUp.post(tap: .cghidEventTap)
        }
    }
}
