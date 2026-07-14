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
        /// Ceiling on the pre-injection clipboard snapshot, measured against
        /// *item-declared* content only. That distinction is the screenshot
        /// bug: a copied screenshot is one item declaring a ~2 MB PNG, but the
        /// pasteboard-level type list also offers server-derived
        /// representations (TIFF at width x height x 4+, each doubled by its
        /// legacy-name alias) — 65 MB of synthesized reads for 1.7 MB of
        /// content, and beyond any fixed cap for HDR captures. Derived types
        /// are never saved; the server re-derives them from the restored
        /// content. Over the cap the save is skipped *entirely* — never a
        /// partial snapshot, which a later restore would present as the full
        /// clipboard — and the loan return clears instead.
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

    /// Per-item snapshot of the pre-dictation clipboard: each element is one
    /// pasteboard item's declared types and data. Item-shaped so the restore
    /// preserves multi-item clipboards instead of flattening them.
    private var savedClipboardContents: [[NSPasteboard.PasteboardType: Data]]?
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

    /// Snapshots what each pasteboard item *declares* — never the
    /// pasteboard-level type list, whose server-derived representations
    /// (PNG→TIFF, modern/legacy aliases) multiply a screenshot's 1.7 MB of
    /// content into tens of megabytes of synthesized reads. The server
    /// re-derives those for whatever the restore puts back.
    private func saveClipboardContents() {
        var items: [[NSPasteboard.PasteboardType: Data]] = []
        var totalBytes = 0
        var shape: [String] = []

        for item in pasteboard.pasteboardItems ?? [] {
            var contents: [NSPasteboard.PasteboardType: Data] = [:]
            for type in item.types {
                if let data = item.data(forType: type) {
                    shape.append("\(type.rawValue):\(data.count)")
                    totalBytes += data.count
                    if totalBytes > maxSavedClipboardBytes {
                        savedClipboardContents = nil
                        Log.transcription.warning(
                            "clipboard loan: snapshot SKIPPED over cap at \(totalBytes)B [\(shape.joined(separator: " "))]"
                        )
                        return
                    }
                    contents[type] = data
                } else {
                    shape.append("\(type.rawValue):nil")
                }
            }
            if !contents.isEmpty {
                items.append(contents)
            }
        }

        savedClipboardContents = items.isEmpty ? nil : items
        Log.transcription.info(
            "clipboard loan: saved \(items.count) item(s), \(totalBytes)B [\(shape.joined(separator: " "))]"
        )
    }

    /// Returns the Clipboard Loan: puts the pre-dictation snapshot back, or —
    /// when there was nothing to save (clipboard empty before the take, or
    /// over the snapshot cap) — clears the transcript, so the next Cmd+V can
    /// never re-paste the dictation. Restore mode off keeps the transcript
    /// (dictate-to-clipboard is a deliberate workflow); this only runs with
    /// it on.
    private static func returnClipboardLoan(
        _ items: [[NSPasteboard.PasteboardType: Data]]?, to pasteboard: NSPasteboard,
        ifChangeCountStill expected: Int
    ) {
        // Another writer took the pasteboard after the transcript write (the
        // user copied, an app synced) — restoring or clearing now would stomp
        // *their* content: the wrong-content-pasted race (audit #285 item 9).
        // Their write wins; the snapshot is dropped.
        guard pasteboard.changeCount == expected else {
            Log.transcription.warning(
                "clipboard loan: return dropped — changeCount \(pasteboard.changeCount) != expected \(expected)"
            )
            return
        }
        pasteboard.clearContents()

        guard let items else {
            Log.transcription.info("clipboard loan: nothing saved — cleared transcript")
            return
        }
        // Pasteboard items are single-use once written, so the snapshot holds
        // raw data and the return mints fresh items from it.
        let rebuilt = items.map { contents -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in contents {
                item.setData(data, forType: type)
            }
            return item
        }
        pasteboard.writeObjects(rebuilt)
        Log.transcription.info(
            "clipboard loan: restored \(rebuilt.count) item(s), changeCount now \(pasteboard.changeCount)"
        )
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
