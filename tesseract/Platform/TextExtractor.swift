//
//  TextExtractor.swift
//  tesseract
//

import Foundation
import AppKit
import os

@MainActor
protocol TextExtracting: AnyObject {
    func extractSelectedText() async throws -> String
}

@MainActor
final class TextExtractor: TextExtracting {
    private enum Defaults {
        static let clipboardSettleDelay: Duration = .milliseconds(100)
    }

    private var savedClipboardContents: [NSPasteboard.PasteboardType: Data]?

    func extractSelectedText() async throws -> String {
        // Save current clipboard contents
        saveClipboardContents()

        // Clear clipboard so we can detect if copy succeeded
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()

        // Simulate Cmd+C to copy selected text
        simulateCopy()

        // Wait for clipboard to settle
        try await Task.sleep(for: Defaults.clipboardSettleDelay)

        // Read clipboard
        let text = pasteboard.string(forType: .string)

        // Restore original clipboard contents
        restoreClipboardContents()

        guard let text, !text.isEmpty else {
            throw SpeechError.noTextSelected
        }

        return text
    }

    // MARK: - Clipboard Operations

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

    private func simulateCopy() {
        let source = CGEventSource(stateID: .hidSystemState)

        let cKeyCode: CGKeyCode = 8  // kVK_ANSI_C

        if let keyDown = CGEvent(keyboardEventSource: source, virtualKey: cKeyCode, keyDown: true) {
            keyDown.flags = .maskCommand
            keyDown.post(tap: .cghidEventTap)
        }

        if let keyUp = CGEvent(keyboardEventSource: source, virtualKey: cKeyCode, keyDown: false) {
            keyUp.flags = .maskCommand
            keyUp.post(tap: .cghidEventTap)
        }
    }
}
