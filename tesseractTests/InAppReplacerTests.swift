//
//  InAppReplacerTests.swift
//  tesseractTests
//
//  A fix put back where a pasted take landed (PRD #612). `InAppEditTests`
//  is the pure text math: where the change starts (by Character), how long
//  it is in UTF-16, how many backspaces erase it, what gets pasted.
//  `InAppReplacerTests` drives the replacer against a fake desktop: one app
//  in front with one text field that answers Accessibility reads, takes
//  selections, pastes and backspaces like a real field, so each test reads
//  the field's text afterwards instead of asserting on calls alone. No
//  Accessibility, no event posting, no pasteboard.
//

import AppKit
import Carbon.HIToolbox
import Testing

@testable import Tesseract_Agent

// MARK: - The text math

struct InAppEditTests {

    @Test func theChangeRunsFromTheFirstDifferentCharacterToTheEnd() {
        let change = InAppEdit.change(from: "use cloud code ", to: "use Claude code ")
        #expect(
            change == .init(offset: 4, length: 11, erased: 11, replacement: "Claude code "))
    }

    @Test func theSameTextIsNoChange() {
        #expect(InAppEdit.change(from: "use Claude code ", to: "use Claude code ") == nil)
    }

    @Test func theTrailingSpaceIsPartOfTheTake() {
        let change = InAppEdit.change(from: "open SRACT ", to: "open Tesseract ")
        #expect(change == .init(offset: 5, length: 6, erased: 6, replacement: "Tesseract "))
    }

    @Test func aJoinedWordReplacesFromTheSpace() {
        let change = InAppEdit.change(from: "a work tree ", to: "a worktree ")
        #expect(change == .init(offset: 6, length: 6, erased: 6, replacement: "tree "))
    }

    @Test func anEmojiIsOneCharacterAndTwoUTF16Units() {
        let change = InAppEdit.change(from: "ship it 🚀 now ", to: "ship it 🚢 now ")
        // "🚀 now " is 6 Characters and 7 UTF-16 units.
        #expect(change == .init(offset: 8, length: 7, erased: 6, replacement: "🚢 now "))
    }

    @Test func aSkinToneChangesTheWholeEmoji() {
        let change = InAppEdit.change(from: "I 👍 it ", to: "I 👍🏽 it ")
        #expect(change == .init(offset: 2, length: 6, erased: 5, replacement: "👍🏽 it "))
    }

    @Test func aCombiningMarkComparesByCharacter() {
        // A decomposed é (e + U+0301) equals a precomposed one as a Character,
        // and still counts two UTF-16 units where it stands.
        let change = InAppEdit.change(from: "cafe\u{301} x ", to: "café y ")
        #expect(change == .init(offset: 6, length: 2, erased: 2, replacement: "y "))
    }

    @Test func aDeletionPastesNothing() {
        let change = InAppEdit.change(from: "say it again again ", to: "say it again ")
        #expect(change == .init(offset: 13, length: 6, erased: 6, replacement: ""))
    }

    @Test func anInsertionAtTheEndErasesNothing() {
        let change = InAppEdit.change(from: "push the ", to: "push the PR ")
        #expect(change == .init(offset: 9, length: 0, erased: 0, replacement: "PR "))
    }

    @Test func thePastedTextIsFoundOnlyWhenItEndsAtTheCaret() {
        #expect(InAppEdit.start(of: "b c ", endingAt: 6, in: "a b c d") == 2)
        #expect(InAppEdit.start(of: "b c ", endingAt: 5, in: "a b c d") == nil)
        #expect(InAppEdit.start(of: "a b c d e", endingAt: 7, in: "a b c d") == nil)
        #expect(InAppEdit.start(of: "d", endingAt: 9, in: "a b c d") == nil)
        #expect(InAppEdit.start(of: "", endingAt: 0, in: "a b c d") == nil)
    }

    @Test func theCaretIsAUTF16Offset() {
        // "hi " is 3 units, "👍 " is 3 more.
        #expect(InAppEdit.start(of: "👍 ", endingAt: 6, in: "hi 👍 x") == 3)
        #expect(InAppEdit.start(of: "👍 ", endingAt: 5, in: "hi 👍 x") == nil)
    }
}

// MARK: - The replacer

@MainActor
struct InAppReplacerTests {

    // MARK: The Accessibility path

    @Test func aFixInAnEditableFieldIsSelectedBackAndPastedOver() async {
        let desktop = FakeDesktop(text: "Note: ", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .replaced)
        #expect(desktop.selections == [10..<21])
        #expect(desktop.pastes == ["Claude code "])
        #expect(desktop.field.value == "Note: use Claude code ")
        // The caret ends where the owner left it: after the take and its space.
        #expect(desktop.field.selection == 22..<22)
    }

    @Test func theSelectionIsCountedInUTF16() async {
        let desktop = FakeDesktop(text: "👋🏽 hi ", then: "say cloud ")
        let anchor = desktop.anchor(pasted: "say cloud ")

        let outcome = await desktop.replacer.replace(anchor, with: "say Claude ", keysAllowed: 0)

        #expect(outcome == .replaced)
        // "👋🏽 hi " is 8 UTF-16 units, "say " 4 more.
        #expect(desktop.selections == [12..<18])
        #expect(desktop.field.value == "👋🏽 hi say Claude ")
    }

    @Test func aDeletionSelectsTheOldWordsAndPressesDelete() async {
        let desktop = FakeDesktop(text: "", then: "say it again again ")
        let anchor = desktop.anchor(pasted: "say it again again ")

        let outcome = await desktop.replacer.replace(anchor, with: "say it again ", keysAllowed: 0)

        #expect(outcome == .replaced)
        #expect(desktop.selections == [13..<19])
        #expect(desktop.deletes == 1)
        #expect(desktop.pastes.isEmpty)
        #expect(desktop.field.value == "say it again ")
    }

    @Test func textNotRightBeforeTheCaretIsLeftAlone() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        // A click put the caret at the start: no key was pressed.
        desktop.field.selection = 0..<0

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "Notes already has the old text"))
        #expect(desktop.touched == false)
        #expect(desktop.field.value == "use cloud code ")
    }

    @Test func aSelectionInsteadOfACaretIsLeftAlone() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.field.selection = 4..<15

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "Notes already has the old text"))
        #expect(desktop.touched == false)
    }

    @Test func anAppThatIgnoresTheSelectionIsRetypedFromTheCaret() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        desktop.field.honorsSelection = false
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .retyped)
        #expect(desktop.deletes == 11)
        #expect(desktop.pastes == ["Claude code "])
        #expect(desktop.field.value == "use Claude code ")
    }

    @Test func aRefusedPasteGivesTheCaretBack() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        desktop.pasteFails = true
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "the paste did not go through"))
        #expect(desktop.field.value == "use cloud code ")
        #expect(desktop.field.selection == 15..<15)
    }

    // MARK: The backspace path

    @Test func aPasteRefusedAfterTheBackspacesSpendsTheAnchor() async {
        let desktop = FakeDesktop(text: "% echo ", then: "use cloud code ")
        desktop.field.isEditable = false
        desktop.pasteFails = true
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .erasedNotPasted)
        #expect(desktop.deletes == 11)
    }

    @Test func aTerminalIsErasedWithBackspacesAndRetyped() async {
        let desktop = FakeDesktop(text: "% echo ", then: "use cloud code ")
        desktop.field.isEditable = false
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .retyped)
        #expect(desktop.selections.isEmpty)
        #expect(desktop.deletes == 11)
        #expect(desktop.pastes == ["Claude code "])
        #expect(desktop.field.value == "% echo use Claude code ")
        // The paste waits for the backspaces to land.
        #expect(desktop.journal.suffix(2) == ["wait 60 ms", "paste"])
    }

    @Test func anAppWithNoFocusedElementIsRetyped() async {
        let desktop = FakeDesktop(text: "", then: "ship it 🚀 now ")
        desktop.exposesElement = false
        let anchor = desktop.anchor(pasted: "ship it 🚀 now ")
        #expect(anchor.element == nil)

        let outcome = await desktop.replacer.replace(
            anchor, with: "ship it 🚢 now ", keysAllowed: 0)

        #expect(outcome == .retyped)
        // One backspace per Character, not per UTF-16 unit.
        #expect(desktop.deletes == 6)
        #expect(desktop.field.value == "ship it 🚢 now ")
    }

    @Test func aDeletionWithBackspacesPastesNothing() async {
        let desktop = FakeDesktop(text: "", then: "say it again again ")
        desktop.exposesElement = false
        let anchor = desktop.anchor(pasted: "say it again again ")

        let outcome = await desktop.replacer.replace(anchor, with: "say it again ", keysAllowed: 0)

        #expect(outcome == .retyped)
        #expect(desktop.deletes == 6)
        #expect(desktop.pastes.isEmpty)
        #expect(desktop.field.value == "say it again ")
    }

    // MARK: When the app has moved on

    @Test func anotherAppInFrontIsLeftAlone() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.frontPID = 777

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "Notes is no longer in front"))
        #expect(desktop.touched == false)
    }

    @Test func theFrontAppIsCheckedAfterABeat() async {
        // The Lens has just ordered out; the app in front is read after the
        // beat, not before it.
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.onFirstWait = { desktop.frontPID = 777 }

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(desktop.journal.first == "wait 50 ms")
        #expect(outcome == .notInApp(reason: "Notes is no longer in front"))
    }

    @Test func typingInTheAppSinceThePasteIsLeftAlone() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.keyCount += 3

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 2)

        #expect(outcome == .notInApp(reason: "you typed or clicked in Notes since"))
        #expect(desktop.touched == false)
    }

    @Test func theKeysOfTheFixItselfAreAllowed() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.keyCount += 7

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 7)

        #expect(outcome == .replaced)
    }

    @Test func aCursorInAnotherFieldIsLeftAlone() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        // The same text and caret, but a different element.
        let other = FakeField(value: "use cloud code ")
        desktop.field = other

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "the cursor moved to another field"))
        #expect(desktop.touched == false)
    }

    @Test func aFieldThatStopsAnsweringCountsAsMoved() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.exposesElement = false

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "the cursor moved to another field"))
        #expect(desktop.touched == false)
    }

    @Test func aPasswordFieldIsNeverTouched() async {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        desktop.field.isSecure = true
        let anchor = desktop.anchor(pasted: "use cloud code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "the cursor is in a password field"))
        #expect(desktop.touched == false)
    }

    @Test func secureKeyboardEntryHidesTypingSoNothingIsTouched() async {
        let desktop = FakeDesktop(text: "% ", then: "use cloud code ")
        desktop.field.isEditable = false
        let anchor = desktop.anchor(pasted: "use cloud code ")
        desktop.typingVisible = false

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "Secure Keyboard Entry hides typing in Notes"))
        #expect(desktop.touched == false)
    }

    @Test func secureKeyboardEntryAtThePasteIsRemembered() async {
        let desktop = FakeDesktop(text: "% ", then: "use cloud code ")
        desktop.typingVisible = false
        let anchor = desktop.anchor(pasted: "use cloud code ")
        #expect(!anchor.typingWasVisible)
        desktop.typingVisible = true

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .notInApp(reason: "Secure Keyboard Entry hides typing in Notes"))
    }

    @Test func nothingToChangeTouchesNothing() async {
        let desktop = FakeDesktop(text: "", then: "use Claude code ")
        let anchor = desktop.anchor(pasted: "use Claude code ")

        let outcome = await desktop.replacer.replace(
            anchor, with: "use Claude code ", keysAllowed: 0)

        #expect(outcome == .replaced)
        #expect(desktop.journal.isEmpty)
    }

    @Test func theAnchorRemembersTheMomentAfterThePaste() {
        let desktop = FakeDesktop(text: "", then: "use cloud code ")
        desktop.keyCount = 41
        let anchor = desktop.anchor(pasted: "use cloud code ")

        #expect(anchor.pasted == "use cloud code ")
        #expect(anchor.pid == notesPID)
        #expect(anchor.bundleID == "com.apple.Notes")
        #expect(anchor.appName == "Notes")
        #expect(anchor.keyCount == 41)
        #expect(anchor.element == FocusedTextElement(desktop.field))
        #expect(anchor.typingWasVisible)
    }
}

// MARK: - Fakes

/// The app in front in these tests (never this test process).
private let notesPID: pid_t = 4242

/// One text field as an app would hold it: UTF-16 selection, a paste that
/// replaces the selection, a backspace that deletes the selection or one
/// Character before the caret.
@MainActor
private final class FakeField: NSObject {
    var value: String
    var selection: Range<Int>
    var isEditable = true
    var isSecure = false
    /// Whether a selection set through Accessibility takes.
    var honorsSelection = true

    /// A field with `value`, the caret at its end.
    init(value: String) {
        self.value = value
        let end = value.utf16.count
        selection = end..<end
    }

    func paste(_ text: String) {
        let range = NSRange(location: selection.lowerBound, length: selection.count)
        value = (value as NSString).replacingCharacters(in: range, with: text)
        let caret = selection.lowerBound + text.utf16.count
        selection = caret..<caret
    }

    func backspace() {
        guard selection.isEmpty else {
            paste("")
            return
        }
        let utf16 = value.utf16
        guard selection.lowerBound > 0,
            let end = utf16.index(utf16.startIndex, offsetBy: selection.lowerBound)
                .samePosition(in: value)
        else { return }
        let start = value.index(before: end)
        let caret = utf16.distance(from: utf16.startIndex, to: start)
        value.removeSubrange(start..<end)
        selection = caret..<caret
    }
}

/// One app in front (Notes, pid 4242) with one focused field. It answers
/// the replacer's Accessibility reads, takes its pastes (as the injector)
/// and its key presses, and counts the owner's keys.
@MainActor
private final class FakeDesktop: FocusedTextAccessing, TextInjecting {
    var restoreClipboard = true
    var frontPID: pid_t? = notesPID
    var keyCount = 0
    var typingVisible = true
    var field: FakeField
    /// Whether Accessibility exposes the field (some Electron apps do not).
    var exposesElement = true
    var pasteFails = false
    var onFirstWait: (() -> Void)?

    private(set) var journal: [String] = []
    private(set) var selections: [Range<Int>] = []
    private(set) var pastes: [String] = []
    private(set) var deletes = 0

    /// Whether anything reached the field: a selection, a key or a paste.
    var touched: Bool { !selections.isEmpty || !pastes.isEmpty || deletes > 0 }

    /// `before` was already in the field; `pasted` is the take just pasted.
    init(text before: String, then pasted: String) {
        field = FakeField(value: before + pasted)
    }

    /// A replacer wired to this desktop. It holds the desktop, not the
    /// other way round.
    var replacer: InAppReplacer {
        InAppReplacer(
            injector: self,
            inputCount: { [unowned self] in keyCount },
            frontmostPID: { [unowned self] in frontPID },
            focusedText: self,
            typingIsVisible: { [unowned self] in typingVisible },
            postKey: { [unowned self] in post($0) },
            sleep: { [unowned self] in await wait($0) })
    }

    func anchor(pasted: String) -> InAppReplacer.Anchor {
        replacer.anchor(
            pasted: pasted, pid: notesPID, bundleID: "com.apple.Notes",
            appName: "Notes")
    }

    // FocusedTextAccessing

    func focusedElement(in pid: pid_t) -> FocusedTextElement? {
        exposesElement && pid == notesPID ? FocusedTextElement(field) : nil
    }

    func state(of element: FocusedTextElement) -> FocusedTextState {
        guard element == FocusedTextElement(field) else { return FocusedTextState() }
        return FocusedTextState(
            value: field.value, selection: field.selection, isEditable: field.isEditable,
            isSecure: field.isSecure)
    }

    func select(_ range: Range<Int>, in element: FocusedTextElement) -> Bool {
        guard element == FocusedTextElement(field) else { return false }
        journal.append("select")
        selections.append(range)
        if field.honorsSelection { field.selection = range }
        return true
    }

    // TextInjecting

    func inject(_ text: String) async throws {
        if pasteFails { throw DictationError.textInjectionFailed("refused") }
        journal.append("paste")
        pastes.append(text)
        field.paste(text)
    }

    // Keys and time

    private func post(_ keyCode: UInt16) {
        #expect(keyCode == UInt16(kVK_Delete))
        journal.append("delete")
        deletes += 1
        field.backspace()
    }

    private func wait(_ duration: Duration) async {
        let ms =
            duration.components.seconds * 1000
            + duration.components.attoseconds / 1_000_000_000_000_000
        journal.append("wait \(ms) ms")
        if let onFirstWait {
            self.onFirstWait = nil
            onFirstWait()
        }
    }
}
