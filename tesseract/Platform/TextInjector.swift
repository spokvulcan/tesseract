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

/// Injects dictated text into the frontmost app via a **Clipboard Loan**: the
/// system pasteboard is borrowed as the transport for a synthetic Cmd+V and
/// then *always* returned — the pre-dictation contents restored, or, when
/// there was nothing to save, cleared. With restore mode on, a transcript
/// never lingers on the pasteboard where a later Cmd+V would re-paste it.
@MainActor
final class TextInjector: ObservableObject, TextInjecting {
    private enum Defaults {
        /// Pause between writing the transcript to the pasteboard and posting the
        /// synthetic Cmd+V. `setString` is synchronous IPC to the pasteboard
        /// server, so this mostly buys margin for slow pasteboard extensions;
        /// it is a fixed tax on every dictation, so it stays small.
        static let clipboardSettleDelay: Duration = .milliseconds(20)
        /// Pause between the paste and returning the loan — the target
        /// app must read the transcript off the pasteboard first. Runs off the
        /// awaited path (see `inject`), so it delays nothing the user sees.
        static let clipboardRestoreDelay: Duration = .milliseconds(100)
        /// Ceiling on the pre-injection clipboard snapshot. Sized for the
        /// dominant real payload — a copied screenshot, whose TIFF
        /// representation is width x height x 4 (~59 MB full-screen at 5K,
        /// ~81 MB at 6K) plus a PNG sibling. The original 10 MB cap silently
        /// dropped exactly that case and the dictation squatted on the
        /// pasteboard. Note the loop below reads each representation *before*
        /// counting it, so the cap bounds retained memory and the remaining
        /// reads, not the first big read. Over the cap the save is skipped
        /// *entirely* — never a partial snapshot, which a later restore would
        /// present as the full clipboard — and the loan return clears instead.
        static let maxSavedClipboardBytes = 128 * 1024 * 1024
    }

    /// The <http://nspasteboard.org> marker types (audit #285 item 9):
    /// `TransientType` tells clipboard managers not to record the write at
    /// all; `ConcealedType` marks it sensitive. Dictated text is exactly the
    /// private content a clipboard-history app must not retain.
    private enum PasteboardMarker {
        static let transient = NSPasteboard.PasteboardType("org.nspasteboard.TransientType")
        static let concealed = NSPasteboard.PasteboardType("org.nspasteboard.ConcealedType")
    }

    @Published private(set) var lastInjectionSucceeded = false

    private var savedClipboardContents: [NSPasteboard.PasteboardType: Data]?
    var restoreClipboard = true

    /// The in-flight deferred loan return, exposed so tests can await the
    /// restore-or-clear instead of sleeping past the restore delay.
    private(set) var clipboardReturnTask: Task<Void, Never>?

    private let pasteboard: NSPasteboard
    private let maxSavedClipboardBytes: Int
    private let ownAppFocused: @MainActor () -> Bool
    private let paste: @MainActor () -> Void

    /// The seams (pasteboard, focus probe, paste trigger) exist for tests:
    /// the loan contract runs against a uniquely-named pasteboard with no
    /// synthetic Cmd+V escaping to whatever app has focus on the test machine.
    init(
        pasteboard: NSPasteboard = .general,
        maxSavedClipboardBytes: Int = Defaults.maxSavedClipboardBytes,
        ownAppFocused: @escaping @MainActor () -> Bool = TextInjector.ownAppIsFocused,
        paste: @escaping @MainActor () -> Void = TextInjector.postSyntheticPaste
    ) {
        self.pasteboard = pasteboard
        self.maxSavedClipboardBytes = maxSavedClipboardBytes
        self.ownAppFocused = ownAppFocused
        self.paste = paste
    }

    func inject(_ text: String) async throws {
        guard !text.isEmpty else {
            throw DictationError.textInjectionFailed("Empty text")
        }

        // Our own app focused: a synthetic Cmd+V must not fire (it used to be
        // skipped outright, stranding the transcript on the clipboard — issue
        // #168). Insert in-process at the caret instead, no clipboard round-trip.
        let isOwnAppFocused = ownAppFocused()
        if isOwnAppFocused, insertIntoFocusedTextView(text) {
            lastInjectionSucceeded = true
            return
        }

        // Save clipboard contents if restoration is enabled
        if restoreClipboard {
            saveClipboardContents()
        }

        // Copy text to clipboard, remembering the generation our write minted.
        let transcriptChangeCount = copyToClipboard(text)

        if !isOwnAppFocused {
            // Small delay to ensure clipboard is updated
            try await Task.sleep(for: Defaults.clipboardSettleDelay)

            // Simulate Cmd+V paste
            paste()

            // Return the loan after the target app has read the transcript.
            // Fire-and-forget: the caller's success feedback (sound, pill
            // leaving "processing") must not lag the visible text by the
            // restore delay, and the return should complete even if the caller
            // is cancelled right after the paste.
            if restoreClipboard {
                let contents = savedClipboardContents
                savedClipboardContents = nil
                clipboardReturnTask = Task { [pasteboard] in
                    try? await Task.sleep(for: Defaults.clipboardRestoreDelay)
                    Self.returnClipboardLoan(
                        contents, to: pasteboard,
                        ifChangeCountStill: transcriptChangeCount)
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

    static func ownAppIsFocused() -> Bool {
        NSApp.isActive && NSApp.keyWindow != nil && !(NSApp.keyWindow is NSPanel)
    }

    // MARK: - Clipboard Operations

    /// Writes the transcript with the transient + concealed markers and
    /// returns the pasteboard generation the write minted, so the deferred
    /// loan return can prove the pasteboard is still ours.
    private func copyToClipboard(_ text: String) -> Int {
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        pasteboard.setString("", forType: PasteboardMarker.transient)
        pasteboard.setString("", forType: PasteboardMarker.concealed)
        return pasteboard.changeCount
    }

    private func saveClipboardContents() {
        var contents: [NSPasteboard.PasteboardType: Data] = [:]
        var totalBytes = 0

        for type in pasteboard.types ?? [] {
            if let data = pasteboard.data(forType: type) {
                totalBytes += data.count
                if totalBytes > maxSavedClipboardBytes {
                    savedClipboardContents = nil
                    return
                }
                contents[type] = data
            }
        }

        savedClipboardContents = contents.isEmpty ? nil : contents
    }

    /// Returns the Clipboard Loan: puts the pre-dictation snapshot back, or —
    /// when there was nothing to save (clipboard empty before the take, or
    /// over the snapshot cap) — clears the transcript, so the next Cmd+V can
    /// never re-paste the dictation. Restore mode off keeps the transcript
    /// (dictate-to-clipboard is a deliberate workflow); this only runs with
    /// it on.
    private static func returnClipboardLoan(
        _ contents: [NSPasteboard.PasteboardType: Data]?, to pasteboard: NSPasteboard,
        ifChangeCountStill expected: Int
    ) {
        // Another writer took the pasteboard after the transcript write (the
        // user copied, an app synced) — restoring or clearing now would stomp
        // *their* content: the wrong-content-pasted race (audit #285 item 9).
        // Their write wins; the snapshot is dropped.
        guard pasteboard.changeCount == expected else { return }
        pasteboard.clearContents()

        guard let contents else { return }
        for (type, data) in contents {
            pasteboard.setData(data, forType: type)
        }
    }

    // MARK: - Key Simulation

    static func postSyntheticPaste() {
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
