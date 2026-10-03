//
//  LensController.swift
//  tesseract
//
//  The **Lens** (PRD #612, ADR-0084, ADR-0085): the one dictation overlay.
//  While the owner holds the dictation key it streams the **Live Preview**
//  in a glass card at the bottom of the screen; when the key comes up the
//  full pass lands and the words the preview had wrong settle into place.
//  A take held with ⇧ waits in the Lens until ↩ pastes it. ⌃⌥Space reopens
//  the last take, the owner types the word they meant, and ↩ puts the fixed
//  take back in the app while the pasted text is still the last thing typed
//  there. Every fix teaches a Learned Word and shows it with Undo.
//
//  The card is a non-activating `GlassPanel`: it never takes focus while
//  listening, takes the keyboard only while a take waits or is being fixed,
//  and gives it back before anything is pasted, so the app the owner was in
//  stays in front throughout.
//

import AppKit
import Carbon.HIToolbox
import Observation
import SwiftUI

/// Where the Lens is drawn: the glass panel in the app, a recorder in tests.
@MainActor
protocol LensPresenting: AnyObject {
    /// Builds the panel ahead of the first press, so it shows at once.
    func prepare()
    /// Shows the card, taking the keyboard when asked.
    func show(takingKeyboard: Bool)
    /// Gives the keyboard back to the app in front; the card stays up.
    func releaseKeyboard()
    func hide()
}

extension LensPresenting {
    func prepare() {}
}

@MainActor
final class LensController {

    /// Putting a fix back where the paste landed (`InAppReplacer` in the
    /// app; a fake in tests), and pasting a held take.
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
        /// Pastes a held take into the app in front; returns that app, or
        /// nil when the paste failed.
        let paste: @MainActor (_ text: String) async -> TargetApp?

        static let none = PasteBack(
            anchor: { _ in nil }, replace: { _, _, _ in .notInApp(reason: "") },
            paste: { _ in nil })
    }

    let model: LensModel

    /// The last committed take: what ⌃⌥Space reopens.
    private(set) var lastTake: DictatedTake?
    private var anchor: Any?
    /// Key presses and clicks made in the Lens since the anchor: input the
    /// app never saw, so it does not mean the app moved on.
    private var lensInput = 0
    /// A fix is being put back in the app (or a held take pasted): the
    /// Lens's keys and the hotkey wait for it.
    private var isPuttingBack = false

    private let pasteBack: PasteBack
    private let vocabulary: @MainActor () -> [String]
    private let isDictating: @MainActor () -> Bool
    private let checkBeforePasting: @MainActor () -> CheckBeforePasting
    private let fixHotkeyLabel: @MainActor () -> String
    /// Takes are pasted at all (Automatically Insert Text).
    private let pastes: @MainActor () -> Bool
    /// The dictation hotkey can see a ⇧ tap (a key combo, not a one-key
    /// hotkey, which another modifier spoils).
    private let canTapShift: @MainActor () -> Bool

    /// "Insert anyway" on a take the Proofread Pass rejected.
    var onInsertRawAnyway: (@MainActor () -> Void)?

    /// Set once at composition (the panel presenter needs the controller for
    /// its keys, so it is attached after both exist).
    var presenter: (any LensPresenting)?
    private var dismissTask: Task<Void, Never>?
    private var watches: [Task<Void, Never>] = []

    /// How long the result line stays before the Lens fades.
    static let resultLinger: Duration = .seconds(4)
    /// How long a landed take stays up ("Missed one? ⌃⌥Space").
    static let landedLinger: Duration = .seconds(2.8)
    /// Where the card sits: the dictation overlay's place, above the Dock.
    static let bottomInset: CGFloat = 60

    init(
        model: LensModel, pasteBack: PasteBack,
        vocabulary: @escaping @MainActor () -> [String],
        isDictating: @escaping @MainActor () -> Bool = { false },
        checkBeforePasting: @escaping @MainActor () -> CheckBeforePasting = { .whenShiftTapped },
        fixHotkeyLabel: @escaping @MainActor () -> String = { "⌃⌥Space" },
        pastes: @escaping @MainActor () -> Bool = { true },
        canTapShift: @escaping @MainActor () -> Bool = { true }
    ) {
        self.model = model
        self.pasteBack = pasteBack
        self.vocabulary = vocabulary
        self.isDictating = isDictating
        self.checkBeforePasting = checkBeforePasting
        self.fixHotkeyLabel = fixHotkeyLabel
        self.pastes = pastes
        self.canTapShift = canTapShift
    }

    /// Builds the panel ahead of the first press.
    func prepare() {
        presenter?.prepare()
    }

    // MARK: - Dictation

    /// Follows dictation: listening while the key is held, the preview as it
    /// streams, finishing after release, errors and a rejected take.
    func watch(_ feed: DictationFeed) {
        watches.forEach { $0.cancel() }
        watches = [
            Task { [weak self, feed] in
                for await phase in Observations({ feed.phase }) {
                    guard let self else { return }
                    self.phaseChanged(phase, app: feed.targetApp)
                }
            },
            Task { [weak self, feed] in
                for await preview in Observations({ feed.preview }) {
                    self?.model.show(preview)
                }
            },
            Task { [weak self, feed] in
                for await held in Observations({ feed.isHeld }) {
                    self?.model.setHeld(held)
                }
            },
            Task { [weak self, feed] in
                var last: UInt64?
                for await beat in Observations({ feed.beat }) {
                    guard let self, let beat, beat.id != last else { continue }
                    last = beat.id
                    self.beat(beat.outcome)
                }
            },
        ]
    }

    private func phaseChanged(_ phase: DictationFeed.Phase, app: TargetApp?) {
        switch phase {
        case .recording:
            // A new take: a fix in progress keeps what it made, and a new
            // take must never paste into the Lens's own field.
            keepFixes()
            keepPageFixes()
            dismissTask?.cancel()
            // A waiting or fixing Lens holds the keyboard: give it back
            // before the new take can paste.
            releaseKeyboard()
            model.listen(app: app)
            model.fixHotkeyLabel = fixHotkeyLabel()
            switch checkBeforePasting() {
            case _ where !pastes(): model.holdHint = nil
            case .whenShiftTapped:
                model.holdHint = canTapShift() ? "⇧ to check before pasting" : nil
            case .always: model.holdHint = "Waits for you before pasting"
            case .never: model.holdHint = nil
            }
            show(takingKeyboard: false)
            DictationPerf.markPanelShown()
            announce("Listening")
        case .processing, .proofreading:
            model.finishing()
        case .error(let error):
            // A take the owner is fixing, waiting on or putting back keeps
            // the Lens.
            guard model.phase != .fixing, !isPuttingBack else { return }
            let result = LensModel.Result(line: Self.line(for: error), detail: nil)
            if model.phase == .listening || model.phase == .finishing {
                model.finish(result)
            } else {
                // Raised before any recording (the microphone in use, a
                // capture that failed) or after the Lens closed ("Insert
                // anyway" could not paste): bring the card back for it.
                model.close()
                model.finish(result)
                show(takingKeyboard: false)
            }
            settle(after: .seconds(2.5))
        case .idle:
            break
        }
    }

    private func beat(_ outcome: DictationFeed.Outcome) {
        switch outcome {
        case .rejected(_, let reason):
            guard model.phase == .listening || model.phase == .finishing else { return }
            model.rejected(LensModel.Result(line: "Didn't catch that", detail: reason))
            settle()
        case .cancelled:
            if model.phase == .listening || model.phase == .finishing { dismiss() }
        case .committed, .empty, .superseded:
            break
        }
    }

    private static func line(for error: DictationError) -> String {
        switch error {
        case .noSpeechDetected: "Didn't hear anything"
        case .recordingTooShort: "Hold the key while you talk"
        case .microphoneBusy: "The microphone is in use"
        default: error.errorDescription ?? "Dictation failed"
        }
    }

    // MARK: - Takes

    /// A take committed: it is the one ⌃⌥Space reopens. Called right after
    /// its paste, so the anchor sees the app exactly as the paste left it. A
    /// held take opens for fixing and waits; any other lands in the Lens.
    func takeCommitted(_ take: DictatedTake) {
        lastTake = take
        anchor = take.pasted ? pasteBack.anchor(take) : nil
        lensInput = 0
        if take.held {
            open(take, mode: .held)
        } else if model.phase == .listening || model.phase == .finishing {
            model.land(take)
            if take.pasted {
                announce("Pasted into \((take.pastedInto ?? take.app)?.name ?? "the app")")
            }
            settle(after: Self.landedLinger, announcing: false)
        }
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
            model.mode != .fromPage
        else { return }
        lastTake = fixed(take)
    }

    /// A fix made from the Dictation page to the take ⌃⌥Space reopens
    /// carries into it, so the reopen starts from the fixed text; the take
    /// keeps where it was pasted.
    private func keepPageFixes() {
        guard model.mode == .fromPage, model.isChanged, let take = model.take,
            let last = lastTake, let pairID = take.pairID, last.pairID == pairID
        else { return }
        lastTake = DictatedTake(
            pairID: pairID, text: model.text, catches: model.catches, app: last.app,
            pasted: last.pasted, at: last.at, pastedInto: last.pastedInto, held: last.held)
    }

    private func fixed(_ take: DictatedTake, pasted: Bool? = nil) -> DictatedTake {
        DictatedTake(
            pairID: take.pairID, text: model.text, catches: model.catches, app: take.app,
            pasted: pasted ?? take.pasted, at: take.at, pastedInto: take.pastedInto,
            held: take.held && !(pasted ?? take.pasted))
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

    /// Opens the last take: a held one waits again, a pasted one opens for
    /// fixing.
    func reopenLastTake() {
        guard model.phase != .fixing, !isDictating() else { return }
        guard let take = lastTake else {
            showMessage("Nothing to fix yet")
            return
        }
        open(take, mode: take.held && !take.pasted ? .held : .afterPaste)
    }

    /// Opens one of the Dictation page's takes for fixing. A take open in
    /// the Lens is kept as a new take keeps it: a held take waits for
    /// ⌃⌥Space and fixes made stay with it (a click on the page has already
    /// closed it, when the Lens lost the keyboard; a context menu has not).
    /// Refused while a take is recorded or a fix is being put back.
    @discardableResult
    func openFromPage(_ take: DictatedTake) -> Bool {
        guard !isDictating(), !isPuttingBack else { return false }
        if model.phase == .fixing {
            keepFixes()
            keepPageFixes()
        }
        open(take, mode: .fromPage)
        return true
    }

    /// Opens a take for fixing (⌃⌥Space, a held take, or a take on the
    /// Dictation page).
    func open(_ take: DictatedTake, mode: LensModel.Mode) {
        dismissTask?.cancel()
        model.open(take, mode: mode, vocabulary: vocabulary())
        show(takingKeyboard: true)
        announce(
            mode == .held
                ? "Waiting. Type a word to fix it, or press Return to paste."
                : "Fix a word. Type the word you meant.")
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

    /// ↩: commits what is typed, then pastes a held take, or puts the fixed
    /// take back in the app while it is still the last thing typed there.
    func finish() {
        guard model.phase == .fixing, !isPuttingBack, let take = model.take else { return }
        if !model.typed.trimmingCharacters(in: .whitespaces).isEmpty {
            // Typed with nothing to replace: stay, the hint says why.
            guard model.commit() else { return }
        }
        if model.mode == .held {
            pasteHeld(take)
            return
        }
        guard model.isChanged else {
            dismiss()
            return
        }
        let fixed = fixed(take)
        let appName = (take.pastedInto ?? take.app)?.name ?? "The app"

        // The keyboard goes back to the app before anything is pasted.
        releaseKeyboard()

        guard model.mode == .afterPaste, take.pasted, let anchor else {
            if model.mode == .afterPaste { lastTake = fixed }
            keepPageFixes()
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
            // A new take started meanwhile owns the card: keep the bookkeeping,
            // leave the card to it.
            let ownsCard = self.model.phase == .fixing
            switch outcome {
            case .replaced:
                self.anchor = self.pasteBack.anchor(fixed)
                self.lensInput = 0
                self.finishIf(ownsCard, LensModel.Result(line: "Fixed in \(appName)", detail: nil))
            case .notInApp(let reason):
                // The app keeps the old text; the take and what it taught stay
                // fixed here, and a later fix still measures from the paste.
                self.finishIf(
                    ownsCard,
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
                self.finishIf(
                    ownsCard,
                    LensModel.Result(
                        line: "The fix could not be pasted into \(appName)",
                        detail: "the old words were erased; the fixed take is in the history"))
            }
            if ownsCard { self.settle() }
        }
    }

    /// ↩ on a held take: paste it (with its fixes) into the app in front,
    /// and anchor there so ⌃⌥Space can still fix it.
    private func pasteHeld(_ take: DictatedTake) {
        let text = model.text
        releaseKeyboard()
        isPuttingBack = true
        Task {
            let app = await pasteBack.paste(text + " ")
            self.isPuttingBack = false
            let ownsCard = self.model.phase == .fixing
            guard let app else {
                self.lastTake = self.fixed(take, pasted: false)
                self.finishIf(
                    ownsCard,
                    LensModel.Result(
                        line: "Couldn't paste the take",
                        detail: "it is kept: ⌃⌥Space brings it back"))
                if ownsCard { self.settle() }
                return
            }
            let pasted = DictatedTake(
                pairID: take.pairID, text: text, catches: self.model.catches, app: take.app,
                pasted: true, at: take.at, pastedInto: app)
            self.lastTake = pasted
            self.anchor = self.pasteBack.anchor(pasted)
            self.lensInput = 0
            self.finishIf(ownsCard, LensModel.Result(line: "Pasted into \(app.name)", detail: nil))
            if ownsCard { self.settle() }
        }
    }

    /// The result line, unless a new take took the card meanwhile.
    private func finishIf(_ ownsCard: Bool, _ result: LensModel.Result) {
        guard ownsCard else { return }
        model.finish(result)
    }

    /// Esc with nothing typed: close. A held take is kept unpasted for
    /// ⌃⌥Space; fixes already made (⇥) stay learned and recorded, and the
    /// app keeps what was pasted.
    func cancelFix() {
        guard model.phase == .fixing, let take = model.take else { return }
        if model.mode == .held {
            releaseKeyboard()
            lastTake = fixed(take, pasted: false)
            model.finish(
                LensModel.Result(
                    line: "Kept, not pasted", detail: "⌃⌥Space brings it back"))
            settle()
            return
        }
        guard model.isChanged else {
            dismiss()
            return
        }
        releaseKeyboard()
        if model.mode == .fromPage {
            keepPageFixes()
        } else {
            lastTake = fixed(take)
        }
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

    /// "Insert anyway" on a rejected take.
    func insertRawAnyway() {
        dismiss()
        onInsertRawAnyway?()
    }

    // MARK: - Panel

    private func showMessage(_ line: String) {
        model.close()
        model.finish(LensModel.Result(line: line, detail: nil))
        show(takingKeyboard: false)
        settle(after: .seconds(1.5))
    }

    private func settle(
        after delay: Duration = LensController.resultLinger, announcing: Bool = true
    ) {
        if announcing {
            announce(
                model.result.map {
                    [$0.line, model.learnedSummary].compactMap { $0 }.joined(separator: ". ")
                } ?? "")
        }
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
    /// Puts fixes back through the app's `InAppReplacer`, and pastes held
    /// takes through the dictation injector.
    init(
        replacer: InAppReplacer, injector: any TextInjecting,
        restoreClipboard: @escaping @MainActor () -> Bool
    ) {
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
            },
            paste: { text in
                // The Lens has just handed the keyboard back: let the app's
                // window take it before ⌘V goes out.
                try? await Task.sleep(for: .milliseconds(60))
                injector.restoreClipboard = restoreClipboard()
                do {
                    try await injector.inject(text)
                } catch {
                    return nil
                }
                return TargetApp.frontmost()
            })
    }
}

/// The Lens in its glass panel: bottom center, non-activating, key only
/// while a take waits or is being fixed.
@MainActor
final class LensPanelPresenter: LensPresenting {
    private weak var controller: LensController?
    private weak var feed: DictationFeed?
    private var panel: GlassPanel?
    /// Set while the presenter hands the keyboard back itself, so that
    /// resign is not mistaken for the owner clicking away.
    private var isReleasing = false
    private var resignObserver: (any NSObjectProtocol)?

    init(controller: LensController, feed: DictationFeed?) {
        self.controller = controller
        self.feed = feed
    }

    func prepare() {
        if panel == nil { panel = makePanel() }
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
                feed: feed,
                actions: LensActions(
                    undo: { [weak controller] in controller?.undo() },
                    close: { [weak controller] in controller?.dismiss() },
                    insertRawAnyway: { [weak controller] in controller?.insertRawAnyway() }),
                onHeightChange: { [weak panel] height in
                    panel?.setHeightKeepingBottom(
                        min(LensStyle.maxHeight, max(LensStyle.minHeight, height)))
                }))
        return panel
    }
}
