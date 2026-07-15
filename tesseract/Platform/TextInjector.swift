//
//  TextInjector.swift
//  tesseract
//

import Foundation
import Combine
import AppKit
import Carbon.HIToolbox

/// The AppKit pasteboard operations used by a Clipboard Loan. Keeping this
/// system boundary narrow lets failure behavior be exercised without touching
/// the user's general pasteboard.
@MainActor
protocol ClipboardPasteboard: AnyObject {
    var changeCount: Int { get }
    var pasteboardItems: [NSPasteboardItem]? { get }

    func clearContents() -> Int
    func setString(_ string: String, forType dataType: NSPasteboard.PasteboardType) -> Bool
    func writeObjects(_ objects: [any NSPasteboardWriting]) -> Bool
}

extension NSPasteboard: ClipboardPasteboard {}

@MainActor
protocol TextInjecting: AnyObject {
    var restoreClipboard: Bool { get set }
    func inject(_ text: String) async throws
}

/// Injects dictated text into the frontmost app via a **Clipboard Loan**: the
/// system pasteboard is borrowed as the transport for a synthetic Cmd+V and
/// then returned — the pre-dictation contents restored, or cleared when there
/// was nothing to save. A pasteboard that cannot be read aborts the injection
/// before anything is mutated: never destroy what could not be seen. One loan
/// is out at a time: a new injection waits for the pending return first, and
/// the transcript write + Cmd+V transport never suspends, while the return
/// runs in an independent task so it outlives cancellation of the caller.
@MainActor
final class TextInjector: ObservableObject, TextInjecting {
    private enum Defaults {
        /// Pause between the paste and returning the loan — the target
        /// app must read the transcript off the pasteboard first. The caller
        /// that took the loan does not await this; a rapid next caller does.
        static let clipboardRestoreDelay: Duration = .milliseconds(100)
        /// A failed clear-then-write return is retried while its pasteboard
        /// generation is still ours. This covers transient pasteboard-server
        /// failures without ever overwriting a newer user copy.
        static let clipboardRestoreRetryDelay: Duration = .milliseconds(50)
        static let maxClipboardRestoreAttempts = 3
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
        }

        let snapshot: Snapshot
        let transcriptChangeCount: Int
    }

    private enum LoanReturnResult {
        case finished
        case retry(expectedChangeCount: Int)
    }

    private struct ClipboardWriteFailure: Error {
        let expectedChangeCount: Int
        let reason: String
    }

    private struct FailedLoanReturn {
        let loan: ClipboardLoan
        var expectedChangeCount: Int
    }

    @Published private(set) var lastInjectionSucceeded = false

    var restoreClipboard = true

    /// The loan currently out, with its deferred return.
    private var activeLoan: (loan: ClipboardLoan, returnTask: Task<Void, Never>)?
    /// A snapshot AppKit repeatedly refused to write. It stays recoverable
    /// until the next clipboard use either restores it or observes that a
    /// newer user copy has made restoration unsafe.
    private var failedLoanReturn: FailedLoanReturn?

    /// The in-flight deferred loan return, exposed so tests can await the
    /// restore-or-clear instead of sleeping past the restore delay.
    var clipboardReturnTask: Task<Void, Never>? { activeLoan?.returnTask }

    private let pasteboard: any ClipboardPasteboard
    private let maxSavedClipboardBytes: Int
    private let ownAppFocused: @MainActor () -> Bool
    private let insertText: @MainActor (String) -> Bool
    private let paste: @MainActor () -> Void

    /// The seams (pasteboard, focus probe, direct insertion, paste trigger)
    /// exist for tests: the loan contract runs against a uniquely-named
    /// pasteboard with no synthetic Cmd+V escaping to whatever app has focus
    /// on the test machine, and no text landing in the test host's own UI.
    init(
        pasteboard: any ClipboardPasteboard = NSPasteboard.general,
        maxSavedClipboardBytes: Int = Defaults.maxSavedClipboardBytes,
        ownAppFocused: @escaping @MainActor () -> Bool = TextInjector.ownAppIsFocused,
        insertText: @escaping @MainActor (String) -> Bool =
            TextInjector.insertIntoFocusedTextView,
        paste: @escaping @MainActor () -> Void = TextInjector.postSyntheticPaste
    ) {
        self.pasteboard = pasteboard
        self.maxSavedClipboardBytes = maxSavedClipboardBytes
        self.ownAppFocused = ownAppFocused
        self.insertText = insertText
        self.paste = paste
    }

    func inject(_ text: String) async throws {
        lastInjectionSucceeded = false
        guard !text.isEmpty else {
            throw DictationError.textInjectionFailed("Empty text")
        }
        try Task.checkCancellation()

        // Our own app focused: a synthetic Cmd+V must not fire (it used to be
        // skipped outright, stranding the transcript on the clipboard — issue
        // #168). Insert in-process at the caret instead, no clipboard round-trip.
        var isOwnAppFocused = ownAppFocused()
        if isOwnAppFocused, insertText(text) {
            lastInjectionSucceeded = true
            return
        }

        // Clipboard-using paths queue behind the complete previous loan,
        // including its post-Cmd+V read window. Re-checking in a loop matters:
        // several rapid callers may all wake from the same return task, and
        // only the first may take the next loan.
        try await prepareClipboardForNextUse()

        // The wait can span a focus change. Re-probe before selecting the
        // delivery route, and retry direct insertion if Tesseract gained an
        // editable responder while the prior loan was returning.
        isOwnAppFocused = ownAppFocused()
        if isOwnAppFocused, insertText(text) {
            lastInjectionSucceeded = true
            return
        }

        if isOwnAppFocused {
            // Own app focused with no editable text field: the text stays on
            // the clipboard for manual paste (a synthetic paste would only
            // beep). Deliberately not a loan — there is nothing to return.
            try copyWithoutLoan(text)
            lastInjectionSucceeded = true
            return
        }

        if restoreClipboard {
            // Snapshot, write, and post Cmd+V without suspending. MainActor
            // isolation makes this transport phase atomic with respect to
            // rapid injections, so each paste observes its own transcript.
            let snapshot = try takeSnapshot()
            let transcriptChangeCount: Int
            switch copyToClipboard(text) {
            case .success(let changeCount):
                transcriptChangeCount = changeCount
            case .failure(let failure):
                // The destructive clear already happened. Return the saved
                // clipboard immediately, then surface the transport failure;
                // Cmd+V must never fire with partial or missing transcript data.
                let rollback = ClipboardLoan(
                    snapshot: snapshot,
                    transcriptChangeCount: failure.expectedChangeCount)
                scheduleLoanReturn(rollback, after: .zero)
                throw DictationError.textInjectionFailed(failure.reason)
            }
            let loan = ClipboardLoan(
                snapshot: snapshot, transcriptChangeCount: transcriptChangeCount)
            paste()
            // The target app's read window begins when Cmd+V is posted, not
            // when the transcript was written.
            scheduleLoanReturn(loan)
        } else {
            try copyWithoutLoan(text)
            paste()
        }

        lastInjectionSucceeded = true
    }

    /// Insert directly into our own key window's focused text view (the agent
    /// composer, or any field editor). Returns false when nothing editable has
    /// focus, in which case the caller falls back to leaving the text on the
    /// clipboard.
    static func insertIntoFocusedTextView(_ text: String) -> Bool {
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

    /// Writes the transcript with the transient + concealed markers. All
    /// three writes are one contract: partial success is a failed transport.
    private func copyToClipboard(_ text: String) -> Result<Int, ClipboardWriteFailure> {
        let clearedChangeCount = pasteboard.clearContents()
        let writes: [(value: String, type: NSPasteboard.PasteboardType)] = [
            (text, .string),
            ("", PasteboardMarker.transient),
            ("", PasteboardMarker.concealed),
        ]

        for write in writes {
            guard pasteboard.setString(write.value, forType: write.type) else {
                Log.transcription.error(
                    "clipboard loan: transcript write failed for \(write.type.rawValue)")
                return .failure(
                    ClipboardWriteFailure(
                        expectedChangeCount: clearedChangeCount,
                        reason: "The clipboard rejected the dictated text"))
            }
        }

        // Ownership check by content, not changeCount arithmetic: whether
        // adding data to a cleared pasteboard bumps the count is not
        // documented, and every dictation would die on a macOS that changed
        // it. If another writer took the pasteboard since our clear, the
        // read-back differs and Cmd+V must not deliver their content.
        guard pasteboard.pasteboardItems?.first?.string(forType: .string) == text else {
            Log.transcription.warning(
                "clipboard loan: transcript write lost ownership before paste")
            return .failure(
                ClipboardWriteFailure(
                    expectedChangeCount: clearedChangeCount,
                    reason: "The clipboard changed before the text could be pasted"))
        }
        // Read after the writes so the loan's generation is correct even if
        // the writes above did bump it.
        return .success(pasteboard.changeCount)
    }

    /// Dictate-to-clipboard and the own-app/no-field fallback deliberately do
    /// not take a loan. A failed partial write is still cleared if its
    /// generation remains ours, and is always reported to the caller.
    private func copyWithoutLoan(_ text: String) throws {
        guard case .failure(let failure) = copyToClipboard(text) else { return }
        if pasteboard.changeCount == failure.expectedChangeCount {
            _ = pasteboard.clearContents()
        }
        throw DictationError.textInjectionFailed(failure.reason)
    }

    /// Snapshots what each pasteboard item *declares*, in declaration order.
    /// Never the pasteboard-level type list: its server-derived
    /// representations amplify the read tens of times over the true content
    /// (see `Defaults.maxSavedClipboardBytes`) and are re-derived from
    /// whatever the restore puts back.
    private func takeSnapshot() throws -> ClipboardLoan.Snapshot {
        guard let pasteboardItems = pasteboard.pasteboardItems else {
            Log.transcription.warning(
                "clipboard loan: snapshot unreadable — pasteboardItems query failed")
            throw DictationError.textInjectionFailed(
                "Clipboard contents could not be read safely")
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
                    throw DictationError.textInjectionFailed(
                        "Clipboard contents could not be read safely")
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
    private func scheduleLoanReturn(
        _ loan: ClipboardLoan, after initialDelay: Duration = Defaults.clipboardRestoreDelay
    ) {
        let returnTask = Task { [pasteboard] in
            // The target app must take the transcript off the pasteboard
            // before the loan is returned.
            do {
                try await Task.sleep(for: initialDelay)
            } catch {
                return
            }

            var expectedChangeCount = loan.transcriptChangeCount
            for attempt in 1...Defaults.maxClipboardRestoreAttempts {
                switch Self.returnClipboardLoan(
                    loan, to: pasteboard, expectedChangeCount: expectedChangeCount)
                {
                case .finished:
                    self.finishLoan(loan)
                    return

                case .retry(let nextExpectedChangeCount):
                    expectedChangeCount = nextExpectedChangeCount
                    guard attempt < Defaults.maxClipboardRestoreAttempts else {
                        Log.transcription.error(
                            "clipboard loan: return deferred after \(attempt) failed attempts")
                        self.failedLoanReturn = FailedLoanReturn(
                            loan: loan,
                            expectedChangeCount: nextExpectedChangeCount)
                        self.finishLoan(loan)
                        return
                    }
                    do {
                        try await Task.sleep(for: Defaults.clipboardRestoreRetryDelay)
                    } catch {
                        return
                    }
                }
            }
        }
        activeLoan = (loan, returnTask)
    }

    private func prepareClipboardForNextUse() async throws {
        while true {
            if let returnTask = activeLoan?.returnTask {
                await returnTask.value
                try Task.checkCancellation()
                continue
            }

            guard var failedReturn = failedLoanReturn else { return }
            switch Self.returnClipboardLoan(
                failedReturn.loan,
                to: pasteboard,
                expectedChangeCount: failedReturn.expectedChangeCount)
            {
            case .finished:
                failedLoanReturn = nil
            case .retry(let nextExpectedChangeCount):
                failedReturn.expectedChangeCount = nextExpectedChangeCount
                failedLoanReturn = failedReturn
                throw DictationError.textInjectionFailed(
                    "The previous clipboard contents could not be restored safely")
            }
        }
    }

    private func finishLoan(_ loan: ClipboardLoan) {
        if activeLoan?.loan.transcriptChangeCount == loan.transcriptChangeCount {
            activeLoan = nil
        }
    }

    /// Returns the Clipboard Loan: restores the snapshot, clears the
    /// transcript when there was nothing to save (so the next Cmd+V can never
    /// re-paste the dictation). Unreadable snapshots abort before a loan is
    /// taken. Restore mode off never takes a loan: dictate-to-clipboard is a
    /// deliberate workflow.
    private static func returnClipboardLoan(
        _ loan: ClipboardLoan, to pasteboard: any ClipboardPasteboard,
        expectedChangeCount: Int
    ) -> LoanReturnResult {
        // Another writer took the pasteboard after the transcript write (the
        // user copied, an app synced) — restoring or clearing now would stomp
        // *their* content: the wrong-content-pasted race (audit #285 item 9).
        // Their write wins; the snapshot is dropped.
        guard pasteboard.changeCount == expectedChangeCount else {
            Log.transcription.warning(
                "clipboard loan: return dropped — changeCount \(pasteboard.changeCount) != expected \(expectedChangeCount)"
            )
            return .finished
        }

        switch loan.snapshot {
        case .nothingToSave:
            _ = pasteboard.clearContents()
            Log.transcription.info("clipboard loan: nothing saved — cleared transcript")
            return .finished

        case .items(let items):
            // Pasteboard items are single-use once written, so the snapshot
            // holds raw data and the return mints fresh items, replaying each
            // item's declaration order.
            var rebuilt: [NSPasteboardItem] = []
            for contents in items {
                let item = NSPasteboardItem()
                for (type, data) in contents {
                    guard item.setData(data, forType: type) else {
                        Log.transcription.error(
                            "clipboard loan: failed to rebuild item type \(type.rawValue)")
                        return .retry(expectedChangeCount: expectedChangeCount)
                    }
                }
                rebuilt.append(item)
            }

            let clearedGeneration = pasteboard.clearContents()
            if pasteboard.writeObjects(rebuilt) {
                Log.transcription.info(
                    "clipboard loan: restored \(rebuilt.count) item(s), changeCount now \(pasteboard.changeCount)"
                )
                return .finished
            } else {
                Log.transcription.error(
                    "clipboard loan: restore write FAILED — retrying \(rebuilt.count) item(s)"
                )
                guard pasteboard.changeCount == clearedGeneration else {
                    Log.transcription.warning(
                        "clipboard loan: retry dropped — pasteboard changed during failed write")
                    return .finished
                }
                return .retry(expectedChangeCount: clearedGeneration)
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
