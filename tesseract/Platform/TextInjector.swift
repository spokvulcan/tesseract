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
/// then returned — the pre-dictation contents restored, cleared when there
/// was nothing to save, or left untouched in the one case the pasteboard
/// could not be read (never destroy what could not be seen). One loan is out
/// at a time: a new injection settles the pending return first, and the
/// return is scheduled before the first suspension so it survives the caller
/// being cancelled mid-injection.
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
        /// content. The cap bounds retained aggregate data, not a single
        /// representation's read: there is no API to size one without
        /// materializing it. Over the cap the save is skipped *entirely* —
        /// never a partial snapshot, which a later restore would present as
        /// the full clipboard — and the loan return clears instead.
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

    /// One Clipboard Loan, whole: what the snapshot found and the pasteboard
    /// generation the transcript write minted, so the return can prove the
    /// pasteboard is still ours.
    private struct ClipboardLoan {
        enum Snapshot {
            /// Ordered (type, data) pairs per pasteboard item. Order is the
            /// source app's declared fidelity preference; the restore replays
            /// it exactly.
            case items([[(type: NSPasteboard.PasteboardType, data: Data)]])
            /// Clipboard was empty before the take, or over the snapshot cap:
            /// the return clears, so the transcript never lingers.
            case nothingToSave
            /// The pasteboard could not be read — `pasteboardItems` returned
            /// nil (a retrieval error, distinct from empty), or a declared
            /// type refused its data. The return must leave the pasteboard
            /// alone: a lingering transcript is the lesser harm than
            /// destroying content that could not be read.
            case unreadable
        }

        let snapshot: Snapshot
        let transcriptChangeCount: Int
    }

    @Published private(set) var lastInjectionSucceeded = false

    var restoreClipboard = true

    /// The loan currently out, with its deferred return.
    private var activeLoan: (loan: ClipboardLoan, returnTask: Task<Void, Never>)?

    /// The in-flight deferred loan return, exposed so tests can await the
    /// restore-or-clear instead of sleeping past the restore delay.
    var clipboardReturnTask: Task<Void, Never>? { activeLoan?.returnTask }

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

        if isOwnAppFocused {
            // Own app focused with no editable text field: the text stays on
            // the clipboard for manual paste (a synthetic paste would only
            // beep). Deliberately not a loan — there is nothing to return.
            _ = copyToClipboard(text)
            lastInjectionSucceeded = true
            return
        }

        if restoreClipboard {
            // One loan at a time: a rapid re-injection settles the pending
            // return now, so this snapshot sees the restored original, never
            // the previous transcript.
            settleActiveLoan()
            let snapshot = takeSnapshot()
            let loan = ClipboardLoan(
                snapshot: snapshot, transcriptChangeCount: copyToClipboard(text))
            // The return is scheduled before this function first suspends:
            // VoiceCaptureSession.cancel() aborts an in-flight commit, and a
            // cancellation landing in the settle sleep below must not strand
            // the transcript with the loan unreturned.
            scheduleLoanReturn(loan)
        } else {
            _ = copyToClipboard(text)
        }

        // Small delay to ensure clipboard is updated
        try await Task.sleep(for: Defaults.clipboardSettleDelay)

        // Simulate Cmd+V paste
        paste()

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

    // MARK: - Clipboard Loan

    /// Writes the transcript with the transient + concealed markers and
    /// returns the pasteboard generation the write minted.
    private func copyToClipboard(_ text: String) -> Int {
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        pasteboard.setString("", forType: PasteboardMarker.transient)
        pasteboard.setString("", forType: PasteboardMarker.concealed)
        return pasteboard.changeCount
    }

    /// Snapshots what each pasteboard item *declares*, in declaration order.
    /// Never the pasteboard-level type list: its server-derived
    /// representations amplify the read tens of times over the true content
    /// (see `Defaults.maxSavedClipboardBytes`) and are re-derived from
    /// whatever the restore puts back.
    private func takeSnapshot() -> ClipboardLoan.Snapshot {
        guard let pasteboardItems = pasteboard.pasteboardItems else {
            Log.transcription.warning(
                "clipboard loan: snapshot unreadable — pasteboardItems query failed")
            return .unreadable
        }

        var items: [[(type: NSPasteboard.PasteboardType, data: Data)]] = []
        var totalBytes = 0
        var shape: [String] = []

        for item in pasteboardItems {
            var contents: [(type: NSPasteboard.PasteboardType, data: Data)] = []
            for type in item.types {
                guard let data = item.data(forType: type) else {
                    // A declared type refusing its data is a read failure, and
                    // restoring the readable remainder would silently present
                    // degraded content as the full clipboard.
                    Log.transcription.warning(
                        "clipboard loan: snapshot unreadable — no data for \(type.rawValue)")
                    return .unreadable
                }
                shape.append("\(type.rawValue):\(data.count)")
                totalBytes += data.count
                if totalBytes > maxSavedClipboardBytes {
                    Log.transcription.warning(
                        "clipboard loan: snapshot skipped over cap at \(totalBytes)B [\(shape.joined(separator: " "))]"
                    )
                    return .nothingToSave
                }
                contents.append((type, data))
            }
            if !contents.isEmpty {
                items.append(contents)
            }
        }

        guard !items.isEmpty else { return .nothingToSave }
        Log.transcription.info(
            "clipboard loan: saved \(items.count) item(s), \(totalBytes)B [\(shape.joined(separator: " "))]"
        )
        return .items(items)
    }

    /// Schedules the deferred return and records the loan as active. The task
    /// is unstructured on purpose: it must fire even if the injection that
    /// spawned it is cancelled, and it must not lag the caller's success
    /// feedback (sound, pill leaving "processing") by the restore delay.
    private func scheduleLoanReturn(_ loan: ClipboardLoan) {
        let returnTask = Task { [pasteboard] in
            // The settle delay plus the read window: the target app must take
            // the transcript off the pasteboard before the loan is returned.
            try? await Task.sleep(
                for: Defaults.clipboardSettleDelay + Defaults.clipboardRestoreDelay)
            // Cancelled means a newer injection settled this loan already.
            guard !Task.isCancelled else { return }
            Self.returnClipboardLoan(loan, to: pasteboard)
            self.finishLoan(loan)
        }
        activeLoan = (loan, returnTask)
    }

    /// Returns the pending loan immediately instead of waiting out its timer:
    /// a new injection is about to borrow the pasteboard, and its snapshot
    /// must see the restored original, not the previous transcript. MainActor
    /// serialization makes cancel-then-return race-free: the timer task either
    /// already ran (the changeCount guard drops this second return) or wakes
    /// cancelled and does nothing.
    private func settleActiveLoan() {
        guard let (loan, returnTask) = activeLoan else { return }
        returnTask.cancel()
        Self.returnClipboardLoan(loan, to: pasteboard)
        activeLoan = nil
    }

    private func finishLoan(_ loan: ClipboardLoan) {
        if activeLoan?.loan.transcriptChangeCount == loan.transcriptChangeCount {
            activeLoan = nil
        }
    }

    /// Returns the Clipboard Loan: restores the snapshot, clears the
    /// transcript when there was nothing to save (so the next Cmd+V can never
    /// re-paste the dictation), or — for an unreadable snapshot — leaves the
    /// pasteboard untouched. Restore mode off never takes a loan:
    /// dictate-to-clipboard is a deliberate workflow.
    private static func returnClipboardLoan(_ loan: ClipboardLoan, to pasteboard: NSPasteboard) {
        // Another writer took the pasteboard after the transcript write (the
        // user copied, an app synced) — restoring or clearing now would stomp
        // *their* content: the wrong-content-pasted race (audit #285 item 9).
        // Their write wins; the snapshot is dropped.
        guard pasteboard.changeCount == loan.transcriptChangeCount else {
            Log.transcription.warning(
                "clipboard loan: return dropped — changeCount \(pasteboard.changeCount) != expected \(loan.transcriptChangeCount)"
            )
            return
        }

        switch loan.snapshot {
        case .unreadable:
            Log.transcription.warning(
                "clipboard loan: snapshot was unreadable — transcript left in place")

        case .nothingToSave:
            pasteboard.clearContents()
            Log.transcription.info("clipboard loan: nothing saved — cleared transcript")

        case .items(let items):
            pasteboard.clearContents()
            // Pasteboard items are single-use once written, so the snapshot
            // holds raw data and the return mints fresh items, replaying each
            // item's declaration order.
            let rebuilt = items.map { contents -> NSPasteboardItem in
                let item = NSPasteboardItem()
                for (type, data) in contents {
                    item.setData(data, forType: type)
                }
                return item
            }
            if pasteboard.writeObjects(rebuilt) {
                Log.transcription.info(
                    "clipboard loan: restored \(rebuilt.count) item(s), changeCount now \(pasteboard.changeCount)"
                )
            } else {
                Log.transcription.error(
                    "clipboard loan: restore write FAILED — \(rebuilt.count) item(s) not returned"
                )
            }
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
