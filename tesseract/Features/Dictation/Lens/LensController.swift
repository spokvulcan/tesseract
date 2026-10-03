//
//  LensController.swift
//  tesseract
//
//  The **Lens** (PRD #612, ADR-0085): ⌃⌥Space reopens the last take in a
//  glass card at the bottom of the screen, the owner types the word they
//  meant, and ↩ puts the fixed take back in the app while the pasted text
//  is still the last thing typed there. Every fix teaches a Learned Word
//  and shows it with Undo.
//
//  The card is a non-activating `GlassPanel`: it takes the keyboard only
//  while a take is being fixed and gives it back before anything is pasted,
//  so the app the owner was in stays in front throughout.
//

import AppKit
import Carbon.HIToolbox
import Observation
import SwiftUI

/// Where the Lens is drawn: the glass panel in the app, a recorder in tests.
@MainActor
protocol LensPresenting: AnyObject {
    /// Shows the card, taking the keyboard when asked.
    func show(takingKeyboard: Bool)
    /// Gives the keyboard back to the app in front; the card stays up.
    func releaseKeyboard()
    func hide()
}

@MainActor
final class LensController {

    /// Putting a fix back where the paste landed (`InAppReplacer` in the
    /// app; a fake in tests).
    struct PasteBack {
        enum Outcome: Equatable, Sendable {
            case replaced
            case notInApp(reason: String)
            /// The old words were erased but the fix could not be pasted.
            case lostOldText
        }

        /// Remembers where a take's paste landed. Called right after the
        /// paste; nil when nothing could ever be put back.
        let anchor: @MainActor (DictatedTake) -> Any?
        /// Replaces the anchored paste with `corrected` (the paste's shape:
        /// the text and a trailing space). `keysAllowed` is how many key
        /// presses and clicks since the paste were made in the Lens.
        let replace:
            @MainActor (_ anchor: Any, _ corrected: String, _ keysAllowed: Int) async -> Outcome

        static let none = PasteBack(
            anchor: { _ in nil }, replace: { _, _, _ in .notInApp(reason: "") })
    }

    let model: LensModel

    /// The last committed take: what ⌃⌥Space reopens.
    private(set) var lastTake: DictatedTake?
    private var anchor: Any?
    /// Key presses and clicks made in the Lens since the anchor: input the
    /// app never saw, so it does not mean the app moved on.
    private var lensInput = 0
    /// A fix is being put back in the app: the Lens's keys and the hotkey
    /// wait for it.
    private var isPuttingBack = false

    private let pasteBack: PasteBack
    private let inputCount: @MainActor () -> Int
    private let vocabulary: @MainActor () -> [String]
    private let isDictating: @MainActor () -> Bool

    /// Set once at composition (the panel presenter needs the controller for
    /// its keys, so it is attached after both exist).
    var presenter: (any LensPresenting)?
    private var dismissTask: Task<Void, Never>?
    private var dictationWatch: Task<Void, Never>?

    /// How long the result line stays before the Lens fades.
    static let resultLinger: Duration = .seconds(4)
    /// Above the dictation pill, which sits 60 pt over the bottom edge.
    static let bottomInset: CGFloat = 140

    init(
        model: LensModel, pasteBack: PasteBack, inputCount: @escaping @MainActor () -> Int,
        vocabulary: @escaping @MainActor () -> [String],
        isDictating: @escaping @MainActor () -> Bool = { false }
    ) {
        self.model = model
        self.pasteBack = pasteBack
        self.inputCount = inputCount
        self.vocabulary = vocabulary
        self.isDictating = isDictating
    }

    /// Closes the Lens when dictation starts: a new take must never paste
    /// into the Lens's own field.
    func watch(_ feed: DictationFeed) {
        dictationWatch?.cancel()
        dictationWatch = Task { [weak self, feed] in
            for await phase in Observations({ feed.phase }) {
                guard let self else { return }
                if phase == .recording, self.model.phase != .hidden {
                    self.keepFixes()
                    self.dismiss()
                }
            }
        }
    }

    // MARK: - Takes

    /// A take committed: it is the one ⌃⌥Space reopens. Called right after
    /// its paste, so the anchor sees the app exactly as the paste left it.
    func takeCommitted(_ take: DictatedTake) {
        lastTake = take
        anchor = take.pasted ? pasteBack.anchor(take) : nil
        lensInput = 0
    }

    /// One key press or click made in the Lens (the panel reports them).
    func noteLensInput() {
        lensInput += 1
    }

    /// The Lens stopped being key while a take was open (the owner clicked
    /// another window): close it, keeping what the fixes taught.
    func lensLostKeyboard() {
        guard model.phase == .fixing, !isPuttingBack else { return }
        cancelFix()
    }

    /// Fixes made with ⇥ stay with the take even when the Lens closes
    /// without finishing, so a reopen starts from them.
    private func keepFixes() {
        guard model.phase == .fixing, model.isChanged, let take = model.take,
            model.mode == .afterPaste
        else { return }
        lastTake = DictatedTake(
            pairID: take.pairID, text: model.text, catches: model.catches, app: take.app,
            pasted: take.pasted, at: take.at, pastedInto: take.pastedInto)
    }

    // MARK: - Hotkey

    /// ⌃⌥Space: reopen the last take, or close the Lens if it is open.
    func fixHotkeyPressed() {
        guard !isPuttingBack else { return }
        if model.phase == .fixing {
            cancelFix()
            return
        }
        reopenLastTake()
    }

    /// Opens the last take for fixing (⌃⌥Space, or the overlay's pencil).
    func reopenLastTake() {
        guard model.phase != .fixing, !isDictating() else { return }
        guard let take = lastTake else {
            showMessage("Nothing to fix yet")
            return
        }
        open(take, mode: .afterPaste)
    }

    /// Opens a take for fixing (⌃⌥Space, or a take on the Dictation page).
    func open(_ take: DictatedTake, mode: LensModel.Mode) {
        dismissTask?.cancel()
        model.open(take, mode: mode, vocabulary: vocabulary())
        show(takingKeyboard: true)
        announce("Fix a word. Type the word you meant.")
    }

    // MARK: - Keys

    /// The Lens's keys, seen before its field: ⇥ fix and stay, ↩ fix and
    /// finish, Esc clear then close, ← → pick a word, ⇧← ⇧→ widen the pick.
    func handleKey(keyCode: Int, modifiers: NSEvent.ModifierFlags) -> Bool {
        guard model.phase == .fixing, !isPuttingBack else { return false }
        let mods = modifiers.intersection([.command, .option, .control, .shift])
        switch keyCode {
        case kVK_Tab where mods.isEmpty:
            model.commit()
            return true
        case kVK_Return, kVK_ANSI_KeypadEnter:
            finish()
            return true
        case kVK_Escape:
            if !model.clearTyping() { cancelFix() }
            return true
        case kVK_LeftArrow where mods.isEmpty:
            model.moveTarget(by: -1)
            return true
        case kVK_RightArrow where mods.isEmpty:
            model.moveTarget(by: 1)
            return true
        case kVK_LeftArrow where mods == .shift:
            model.extendTarget(by: -1)
            return true
        case kVK_RightArrow where mods == .shift:
            model.extendTarget(by: 1)
            return true
        default:
            return false
        }
    }

    // MARK: - Finishing

    /// ↩: commits what is typed, then puts the fixed take back in the app
    /// while it is still the last thing typed there.
    func finish() {
        guard model.phase == .fixing, !isPuttingBack, let take = model.take else { return }
        if !model.typed.trimmingCharacters(in: .whitespaces).isEmpty {
            // Typed with nothing to replace: stay, the hint says why.
            guard model.commit() else { return }
        }
        guard model.isChanged else {
            dismiss()
            return
        }
        let corrected = model.text
        let fixed = DictatedTake(
            pairID: take.pairID, text: corrected, catches: model.catches, app: take.app,
            pasted: take.pasted, at: take.at, pastedInto: take.pastedInto)
        let appName = (take.pastedInto ?? take.app)?.name ?? "The app"

        // The keyboard goes back to the app before anything is pasted.
        releaseKeyboard()

        guard model.mode == .afterPaste, take.pasted, let anchor else {
            if model.mode == .afterPaste { lastTake = fixed }
            model.finish(LensModel.Result(line: "Fixed", detail: nil))
            settle()
            return
        }
        let keysInLens = lensInput
        isPuttingBack = true
        Task {
            let outcome = await pasteBack.replace(anchor, fixed.pastedText, keysInLens)
            self.isPuttingBack = false
            self.lastTake = fixed
            switch outcome {
            case .replaced:
                self.anchor = self.pasteBack.anchor(fixed)
                self.lensInput = 0
                self.model.finish(LensModel.Result(line: "Fixed in \(appName)", detail: nil))
            case .notInApp(let reason):
                // The app keeps the old text; the take and what it taught stay
                // fixed here, and a later fix still measures from the paste.
                self.model.finish(
                    LensModel.Result(
                        line: "\(appName) already has the old text",
                        detail: reason.isEmpty ? nil : reason))
            case .lostOldText:
                // The app no longer holds the take: nothing can be put back
                // there again.
                self.anchor = nil
                self.lastTake = DictatedTake(
                    pairID: fixed.pairID, text: fixed.text, catches: fixed.catches,
                    app: fixed.app, pasted: false, at: fixed.at)
                self.model.finish(
                    LensModel.Result(
                        line: "The fix could not be pasted into \(appName)",
                        detail: "the old words were erased; the fixed take is in the history"))
            }
            self.settle()
        }
    }

    /// Esc with nothing typed: close. Fixes already made (⇥) stay learned and
    /// recorded; the app keeps what was pasted.
    func cancelFix() {
        guard model.phase == .fixing else { return }
        guard model.isChanged, let take = model.take else {
            dismiss()
            return
        }
        releaseKeyboard()
        lastTake = DictatedTake(
            pairID: take.pairID, text: model.text, catches: model.catches, app: take.app,
            pasted: take.pasted, at: take.at, pastedInto: take.pastedInto)
        model.finish(
            LensModel.Result(
                line: take.pasted
                    ? "Not changed in \((take.pastedInto ?? take.app)?.name ?? "the app")"
                    : "Fixed",
                detail: nil))
        settle()
    }

    /// The result line's Undo: forget what the fixes taught.
    func undo() {
        model.undoLearning()
        if model.phase == .done { settle() }
    }

    // MARK: - Panel

    private func showMessage(_ line: String) {
        model.close()
        model.finish(LensModel.Result(line: line, detail: nil))
        show(takingKeyboard: false)
        settle(after: .seconds(1.5))
    }

    private func settle(after delay: Duration = LensController.resultLinger) {
        announce(
            model.result.map {
                [$0.line, model.learnedSummary].compactMap { $0 }.joined(separator: ". ")
            } ?? "")
        dismissTask?.cancel()
        dismissTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.dismiss()
        }
    }

    func dismiss() {
        dismissTask?.cancel()
        dismissTask = nil
        presenter?.hide()
        model.close()
    }

    /// Esc that reached the panel itself rather than the key intercept.
    func escape() {
        if model.phase == .fixing {
            if !model.clearTyping() { cancelFix() }
        } else {
            dismiss()
        }
    }

    private func show(takingKeyboard: Bool) {
        presenter?.show(takingKeyboard: takingKeyboard)
        if takingKeyboard { model.requestFocus() }
    }

    private func releaseKeyboard() {
        presenter?.releaseKeyboard()
    }

    private func announce(_ text: String) {
        guard !text.isEmpty else { return }
        NSAccessibility.post(
            element: NSApp as Any, notification: .announcementRequested,
            userInfo: [
                .announcement: text,
                .priority: NSAccessibilityPriorityLevel.high.rawValue,
            ])
    }
}

extension LensController.PasteBack {
    /// Puts fixes back through the app's `InAppReplacer`.
    init(replacer: InAppReplacer) {
        self.init(
            anchor: { take in
                // Where the paste went, which can differ from where the
                // take started if the owner switched apps meanwhile.
                guard let app = take.pastedInto ?? take.app else { return nil }
                return replacer.anchor(pasted: take.pastedText, in: app)
            },
            replace: { anchor, corrected, keysAllowed in
                guard let anchor = anchor as? InAppReplacer.Anchor else {
                    return .notInApp(reason: "")
                }
                switch await replacer.replace(anchor, with: corrected, keysAllowed: keysAllowed) {
                case .replaced, .retyped: return .replaced
                case .notInApp(let reason): return .notInApp(reason: reason)
                case .erasedNotPasted: return .lostOldText
                }
            })
    }
}

/// The Lens in its glass panel: bottom center, non-activating, key only
/// while a take is being fixed.
@MainActor
final class LensPanelPresenter: LensPresenting {
    private weak var controller: LensController?
    private var panel: GlassPanel?
    /// Set while the presenter hands the keyboard back itself, so that
    /// resign is not mistaken for the owner clicking away.
    private var isReleasing = false
    private var resignObserver: (any NSObjectProtocol)?

    init(controller: LensController) {
        self.controller = controller
    }

    func show(takingKeyboard: Bool) {
        guard let panel = panel ?? makePanel() else { return }
        self.panel = panel
        panel.placeBottomCenter(
            on: OverlayScreenLocator.preferredScreen(), fromBottom: LensController.bottomInset)
        panel.orderFrontRegardless()
        if takingKeyboard { panel.makeKey() }
    }

    /// A non-activating panel that stops being key hands the keys back to
    /// the active app's window; ordering it straight back keeps the card up.
    func releaseKeyboard() {
        guard let panel, panel.isKeyWindow else { return }
        isReleasing = true
        defer { isReleasing = false }
        panel.orderOut(nil)
        panel.orderFrontRegardless()
    }

    func hide() {
        panel?.orderOut(nil)
    }

    private func makePanel() -> GlassPanel? {
        guard let controller else { return nil }
        let panel = GlassPanel(
            size: NSSize(width: LensStyle.width, height: LensStyle.minHeight), cornerRadius: 24,
            becomesKeyOnlyIfNeeded: true)
        panel.onCancel = { [weak controller] in controller?.escape() }
        panel.onInput = { [weak controller] in controller?.noteLensInput() }
        // The owner clicked another window while a take was open.
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification, object: panel, queue: nil
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, !self.isReleasing else { return }
                self.controller?.lensLostKeyboard()
            }
        }
        panel.interceptKeyDown = { [weak controller] event in
            controller?.handleKey(keyCode: Int(event.keyCode), modifiers: event.modifierFlags)
                ?? false
        }
        panel.host(
            LensView(
                model: controller.model,
                actions: LensActions(
                    undo: { [weak controller] in controller?.undo() },
                    close: { [weak controller] in controller?.dismiss() }),
                onHeightChange: { [weak panel] height in
                    panel?.setHeightKeepingBottom(
                        min(LensStyle.maxHeight, max(LensStyle.minHeight, height)))
                }))
        return panel
    }
}
