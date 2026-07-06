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
        /// Pause between writing the transcript to the pasteboard and posting the
        /// synthetic Cmd+V. `setString` is synchronous IPC to the pasteboard
        /// server, so this mostly buys margin for slow pasteboard extensions;
        /// it is a fixed tax on every dictation, so it stays small.
        static let clipboardSettleDelay: Duration = .milliseconds(20)
        /// Pause between the paste and restoring the saved clipboard — the target
        /// app must read the transcript off the pasteboard first. Runs off the
        /// awaited path (see `inject`), so it delays nothing the user sees.
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

            // Restore the original clipboard after the target app has read the
            // transcript. Fire-and-forget: the caller's success feedback (sound,
            // pill leaving "processing") must not lag the visible text by the
            // restore delay, and the restore should complete even if the caller
            // is cancelled right after the paste.
            if restoreClipboard {
                let contents = savedClipboardContents
                savedClipboardContents = nil
                Task {
                    try? await Task.sleep(for: Defaults.clipboardRestoreDelay)
                    Self.restoreClipboardContents(contents)
                }
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

    private static func restoreClipboardContents(
        _ contents: [NSPasteboard.PasteboardType: Data]?
    ) {
        guard let contents else { return }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()

        for (type, data) in contents {
            pasteboard.setData(data, forType: type)
        }
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
