//
//  TextInjector.swift
//  tesseract
//

import Foundation
import Combine
import AppKit
import Carbon.HIToolbox

@MainActor
protocol TextInjecting: AnyObject {
    var restoreClipboard: Bool { get set }
    func inject(_ text: String) async throws
}

@MainActor
final class TextInjector: ObservableObject, TextInjecting {
    private enum Defaults {
        static let clipboardSettleDelay: Duration = .milliseconds(50)
        static let clipboardRestoreDelay: Duration = .milliseconds(100)
    }

    @Published private(set) var lastInjectionSucceeded = false

    private var savedClipboardContents: [NSPasteboard.PasteboardType: Data]?
    var restoreClipboard = true

    func inject(_ text: String) async throws {
        guard !text.isEmpty else {
            throw DictationError.textInjectionFailed("Empty text")
        }

        // Our own app focused: a synthetic Cmd+V must not fire (it used to be
        // skipped outright, stranding the transcript on the clipboard — issue
        // #168). Insert in-process at the caret instead, no clipboard round-trip.
        let isOwnAppFocused =
            NSApp.isActive && NSApp.keyWindow != nil && !(NSApp.keyWindow is NSPanel)
        if isOwnAppFocused, insertIntoFocusedTextView(text) {
            lastInjectionSucceeded = true
            return
        }

        // Save clipboard contents if restoration is enabled
        if restoreClipboard {
            saveClipboardContents()
        }

        // Copy text to clipboard
        copyToClipboard(text)

        if !isOwnAppFocused {
            // Small delay to ensure clipboard is updated
            try await Task.sleep(for: Defaults.clipboardSettleDelay)

            // Simulate Cmd+V paste
            simulatePaste()

            // Small delay before restoring clipboard
            try await Task.sleep(for: Defaults.clipboardRestoreDelay)

            // Restore original clipboard contents
            if restoreClipboard {
                restoreClipboardContents()
            }
        }
        // Own app focused with no editable text field: text remains on the
        // clipboard for manual paste (a synthetic paste would only beep)

        lastInjectionSucceeded = true
    }

    /// Insert directly into our own key window's focused text view (the agent
    /// composer, or any field editor). Returns false when nothing editable has
    /// focus, in which case the caller falls back to leaving the text on the
    /// clipboard.
    private func insertIntoFocusedTextView(_ text: String) -> Bool {
        guard let textView = NSApp.keyWindow?.firstResponder as? NSTextView,
            textView.isEditable
        else { return false }
        textView.insertText(text, replacementRange: textView.selectedRange())
        return true
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
