//
//  InAppReplacer.swift
//  tesseract
//
//  A fix put back where a pasted take landed (PRD #612, ADR-0085). After a
//  paste, ⌃⌥Space reopens the take in the **Lens**; when the owner fixes a
//  word there, the fix goes into the app too, but only while the pasted
//  text is still the last thing typed there: the same app in front, no key
//  pressed since the paste, the same focused element. Where the field
//  exposes its text through Accessibility, the pasted text must sit right
//  before the caret; the changed part is selected back and the fix pasted
//  over it. Where it does not (a terminal, some Electron apps), the changed
//  part is erased with backspaces and the fix pasted. Never in a password
//  field, and never while Secure Keyboard Entry hides the owner's typing.
//
//  The change always runs from the first changed Character to the end of
//  the pasted text, so the caret ends where the owner left it (after the
//  take and its space) on both paths. A region stopping before an unchanged
//  tail would leave the caret mid-take, and moving it back through
//  Accessibility races the paste, which reaches the app on another channel.
//
//  Chromium and Electron build their accessibility tree only when asked
//  (`AXManualAccessibility`), which also puts them in screen-reader mode;
//  this never asks, so those apps take the backspace path.
//

import AppKit
import ApplicationServices
import Carbon.HIToolbox

// MARK: - Focused text

/// An opaque handle on another app's focused text element (an
/// `AXUIElement` in production), so the element never leaks into the
/// value types that carry it. Two handles are the same element when
/// `CFEqual` says so.
nonisolated struct FocusedTextElement: @unchecked Sendable, Equatable {
    fileprivate let ref: AnyObject

    init(_ ref: AnyObject) {
        self.ref = ref
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        CFEqual(lhs.ref, rhs.ref)
    }
}

/// What Accessibility says about a focused text element right now.
nonisolated struct FocusedTextState: Equatable, Sendable {
    /// The element's text, when readable.
    var value: String?
    /// The selection in UTF-16 offsets into `value`; empty is the caret.
    var selection: Range<Int>?
    /// The value can be set: an editable field, where a paste replaces a
    /// selection. Terminals expose their text read-only.
    var isEditable: Bool
    /// A password field.
    var isSecure: Bool

    init(
        value: String? = nil, selection: Range<Int>? = nil, isEditable: Bool = false,
        isSecure: Bool = false
    ) {
        self.value = value
        self.selection = selection
        self.isEditable = isEditable
        self.isSecure = isSecure
    }
}

/// Reads and selects text in an app's focused element.
@MainActor
protocol FocusedTextAccessing: AnyObject {
    /// The focused element of the app `pid` now, or nil when Accessibility
    /// exposes none.
    func focusedElement(in pid: pid_t) -> FocusedTextElement?
    func state(of element: FocusedTextElement) -> FocusedTextState
    /// Sets the selection (UTF-16 offsets). True when the app accepted it,
    /// which does not mean it took: read the state back to know.
    @discardableResult
    func select(_ range: Range<Int>, in element: FocusedTextElement) -> Bool
}

/// The production accessor: the app's focused element through the
/// Accessibility API, asked of that app alone (never the system-wide
/// element, so the paste's app is the one read, and the timeout set here
/// stays on these elements instead of every Accessibility call in the
/// process). Each call is bounded by a 0.25 s messaging timeout so a hung
/// app cannot stall the main actor.
@MainActor
final class AXFocusedText: FocusedTextAccessing {
    private static let messagingTimeout: Float = 0.25

    func focusedElement(in pid: pid_t) -> FocusedTextElement? {
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, Self.messagingTimeout)
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(
                app, kAXFocusedUIElementAttribute as CFString, &value) == .success,
            let value, CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        AXUIElementSetMessagingTimeout(
            unsafeDowncast(value, to: AXUIElement.self), Self.messagingTimeout)
        return FocusedTextElement(value)
    }

    func state(of element: FocusedTextElement) -> FocusedTextState {
        guard let ax = Self.axElement(element) else { return FocusedTextState() }
        let role = Self.string(ax, kAXRoleAttribute)
        let subrole = Self.string(ax, kAXSubroleAttribute)
        let secure = kAXSecureTextFieldSubrole
        var settable: DarwinBoolean = false
        let editable =
            AXUIElementIsAttributeSettable(ax, kAXValueAttribute as CFString, &settable)
            == .success && settable.boolValue
        return FocusedTextState(
            value: Self.string(ax, kAXValueAttribute),
            selection: Self.selection(of: ax),
            isEditable: editable,
            isSecure: role == secure || subrole == secure)
    }

    func select(_ range: Range<Int>, in element: FocusedTextElement) -> Bool {
        guard let ax = Self.axElement(element) else { return false }
        var cfRange = CFRange(location: range.lowerBound, length: range.count)
        guard let value = AXValueCreate(.cfRange, &cfRange) else { return false }
        return AXUIElementSetAttributeValue(
            ax, kAXSelectedTextRangeAttribute as CFString, value) == .success
    }

    private static func axElement(_ element: FocusedTextElement) -> AXUIElement? {
        guard CFGetTypeID(element.ref) == AXUIElementGetTypeID() else { return nil }
        return unsafeDowncast(element.ref, to: AXUIElement.self)
    }

    private static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success
        else { return nil }
        return value as? String
    }

    private static func selection(of element: AXUIElement) -> Range<Int>? {
        var value: CFTypeRef?
        guard
            AXUIElementCopyAttributeValue(
                element, kAXSelectedTextRangeAttribute as CFString, &value) == .success,
            let value, CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        var range = CFRange()
        guard AXValueGetValue(unsafeDowncast(value, to: AXValue.self), .cfRange, &range),
            range.location >= 0, range.length >= 0
        else { return nil }
        return range.location..<(range.location + range.length)
    }
}

// MARK: - The edit

/// The pure text math of a fix put back in an app.
nonisolated enum InAppEdit {

    /// What turns the pasted text into the corrected one: everything from
    /// the first Character that differs to the end of the pasted text.
    struct Change: Equatable, Sendable {
        /// UTF-16 offset in the pasted text where the change starts.
        let offset: Int
        /// UTF-16 length from `offset` to the end of the pasted text: the
        /// range that gets selected.
        let length: Int
        /// Characters from the first change to the end of the pasted text:
        /// the backspaces that erase it.
        let erased: Int
        /// The corrected text from the first change on: what gets pasted.
        /// Empty when the fix only deletes.
        let replacement: String
    }

    /// The change from `old` to `new`, or nil when they are the same text.
    /// Characters compare as Swift does (canonical equivalence), so an
    /// emoji with a skin tone or a letter with a combining mark is one unit.
    static func change(from old: String, to new: String) -> Change? {
        var common = 0
        var commonUTF16 = 0
        for (a, b) in zip(old, new) {
            guard a == b else { break }
            common += 1
            commonUTF16 += a.utf16.count
        }
        let oldCount = old.count
        guard common < oldCount || common < new.count else { return nil }
        return Change(
            offset: commonUTF16,
            length: old.utf16.count - commonUTF16,
            erased: oldCount - common,
            replacement: String(new.dropFirst(common)))
    }

    /// Where `text` starts in `value` (a UTF-16 offset) when it ends exactly
    /// at `caret`, or nil when it is not right before the caret.
    static func start(of text: String, endingAt caret: Int, in value: String) -> Int? {
        let needle = text.utf16
        let haystack = value.utf16
        let count = needle.count
        guard count > 0, caret >= count, caret <= haystack.count else { return nil }
        let lower = haystack.index(haystack.startIndex, offsetBy: caret - count)
        let upper = haystack.index(lower, offsetBy: count)
        return haystack[lower..<upper].elementsEqual(needle) ? caret - count : nil
    }
}

// MARK: - The replacer

@MainActor
final class InAppReplacer {

    /// What was true right after a paste: the yardstick a later fix is
    /// measured against.
    nonisolated struct Anchor: Equatable, Sendable {
        /// What the paste typed (the take and the space after it).
        let pasted: String
        let pid: pid_t
        let bundleID: String?
        let appName: String
        /// The owner's key presses and clicks so far (`HotkeyManager.inputCount`).
        let keyCount: Int
        /// The focused text element, when Accessibility exposed one.
        let element: FocusedTextElement?
        /// Whether the key count could see typing (no Secure Keyboard Entry).
        let typingWasVisible: Bool
    }

    nonisolated enum Outcome: Equatable, Sendable {
        /// Selected back through Accessibility and pasted over.
        case replaced
        /// Erased with backspaces and pasted.
        case retyped
        /// The app has moved on; the fix stays in Tesseract only.
        case notInApp(reason: String)
        /// The old words were erased but the fix could not be pasted: the
        /// app no longer holds the take, so the anchor is spent.
        case erasedNotPasted
    }

    /// Tesseract itself: an Accessibility call into our own process from the
    /// main actor would wait on itself until the timeout.
    private static let ownPID = ProcessInfo.processInfo.processIdentifier

    private enum Timing {
        /// The Lens has just ordered out: let focus settle back in the app.
        static let settle: Duration = .milliseconds(50)
        /// Between setting a selection and reading it back.
        static let select: Duration = .milliseconds(40)
        /// Between the backspaces and the paste.
        static let erase: Duration = .milliseconds(60)
    }

    private let injector: any TextInjecting
    private let inputCount: @MainActor () -> Int
    private let frontmostPID: @MainActor () -> pid_t?
    private let focusedText: any FocusedTextAccessing
    private let typingIsVisible: @MainActor () -> Bool
    private let postKey: @MainActor (UInt16) -> Void
    private let sleep: @MainActor (Duration) async -> Void

    /// - Parameters:
    ///   - injector: pastes through a Clipboard Loan (the dictation injector).
    ///   - inputCount: the owner's key presses and clicks so far
    ///     (`HotkeyManager.inputCount`).
    ///   - typingIsVisible: false while Secure Keyboard Entry keeps key
    ///     presses from the event tap, so the count would miss them.
    ///   - postKey: posts one marked key press (`SyntheticKeyEvents`).
    init(
        injector: any TextInjecting,
        inputCount: @escaping @MainActor () -> Int,
        frontmostPID: @escaping @MainActor () -> pid_t? = {
            NSWorkspace.shared.frontmostApplication?.processIdentifier
        },
        focusedText: any FocusedTextAccessing = AXFocusedText(),
        typingIsVisible: @escaping @MainActor () -> Bool = { !IsSecureEventInputEnabled() },
        postKey: @escaping @MainActor (UInt16) -> Void = {
            SyntheticKeyEvents.post(keyCode: CGKeyCode($0))
        },
        sleep: @escaping @MainActor (Duration) async -> Void = { try? await Task.sleep(for: $0) }
    ) {
        self.injector = injector
        self.inputCount = inputCount
        self.frontmostPID = frontmostPID
        self.focusedText = focusedText
        self.typingIsVisible = typingIsVisible
        self.postKey = postKey
        self.sleep = sleep
    }

    /// Take the anchor right after a paste into the app `pid`.
    func anchor(pasted: String, pid: pid_t, bundleID: String?, appName: String) -> Anchor {
        Anchor(
            pasted: pasted, pid: pid, bundleID: bundleID, appName: appName,
            keyCount: inputCount(),
            element: pid == Self.ownPID ? nil : focusedText.focusedElement(in: pid),
            typingWasVisible: typingIsVisible())
    }

    func anchor(pasted: String, in app: TargetApp) -> Anchor {
        anchor(pasted: pasted, pid: app.pid, bundleID: app.bundleID, appName: app.name)
    }

    /// Put `corrected` (the whole corrected take, shaped like
    /// `anchor.pasted`: the text and its space) where the anchor's paste
    /// landed. `keysAllowed` is how many key presses since the paste were
    /// the owner's own fix (typed into the Lens), not typing in the app.
    func replace(_ anchor: Anchor, with corrected: String, keysAllowed: Int) async -> Outcome {
        guard let change = InAppEdit.change(from: anchor.pasted, to: corrected) else {
            return .replaced
        }
        await sleep(Timing.settle)
        let outcome = await put(change, at: anchor, keysAllowed: keysAllowed)
        switch outcome {
        case .replaced, .retyped:
            Log.transcription.info("in-app fix: \(outcome) in \(anchor.bundleID ?? "?")")
        case .notInApp(let reason):
            Log.transcription.info(
                "in-app fix: not in \(anchor.bundleID ?? "?"), \(reason)")
        case .erasedNotPasted:
            Log.transcription.error(
                "in-app fix: erased in \(anchor.bundleID ?? "?") but the paste failed")
        }
        return outcome
    }

    private func put(_ change: InAppEdit.Change, at anchor: Anchor, keysAllowed: Int) async
        -> Outcome
    {
        let app = anchor.appName
        guard frontmostPID() == anchor.pid else {
            return .notInApp(reason: "\(app) is no longer in front")
        }
        guard anchor.pid != Self.ownPID else {
            return .notInApp(reason: "the take went into Tesseract")
        }
        guard anchor.typingWasVisible, typingIsVisible() else {
            return .notInApp(reason: "Secure Keyboard Entry hides typing in \(app)")
        }
        guard inputCount() - anchor.keyCount <= keysAllowed else {
            return .notInApp(reason: "you typed or clicked in \(app) since")
        }
        let element = focusedText.focusedElement(in: anchor.pid)
        if let before = anchor.element, element != before {
            return .notInApp(reason: "the cursor moved to another field")
        }
        if let element {
            let state = focusedText.state(of: element)
            guard !state.isSecure else {
                return .notInApp(reason: "the cursor is in a password field")
            }
            if state.isEditable, let value = state.value, let selection = state.selection {
                return await selectAndPaste(
                    change, of: anchor, in: element, value: value, selection: selection)
            }
        }
        return await retype(change)
    }

    /// The Accessibility path: the pasted text must end at the caret.
    private func selectAndPaste(
        _ change: InAppEdit.Change, of anchor: Anchor, in element: FocusedTextElement,
        value: String, selection: Range<Int>
    ) async -> Outcome {
        let caret = selection.lowerBound
        guard selection.isEmpty,
            let start = InAppEdit.start(of: anchor.pasted, endingAt: caret, in: value)
        else {
            return .notInApp(reason: "\(anchor.appName) already has the old text")
        }
        let lower = start + change.offset
        let target = lower..<(lower + change.length)
        focusedText.select(target, in: element)
        await sleep(Timing.select)
        let now = focusedText.state(of: element).selection
        if now != target {
            // The app ignored the request: the caret is still after the
            // pasted text, so erasing it is as safe as selecting it.
            if now == caret..<caret { return await retype(change) }
            focusedText.select(caret..<caret, in: element)
            return .notInApp(reason: "\(anchor.appName) did not let Tesseract select the text")
        }
        guard !change.replacement.isEmpty else {
            postKey(UInt16(kVK_Delete))
            return .replaced
        }
        do {
            try await injector.inject(change.replacement)
        } catch {
            focusedText.select(caret..<caret, in: element)
            return .notInApp(reason: "the paste did not go through")
        }
        return .replaced
    }

    /// The keystroke path: one backspace per Character, then the paste.
    private func retype(_ change: InAppEdit.Change) async -> Outcome {
        for _ in 0..<change.erased {
            postKey(UInt16(kVK_Delete))
        }
        guard !change.replacement.isEmpty else { return .retyped }
        await sleep(Timing.erase)
        do {
            try await injector.inject(change.replacement)
        } catch {
            return .erasedNotPasted
        }
        return .retyped
    }
}
