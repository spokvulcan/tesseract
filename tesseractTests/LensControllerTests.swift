//
//  LensControllerTests.swift
//  tesseractTests
//
//  Pins the **Lens**'s flow around a fix (PRD #612): ⌃⌥Space reopens the
//  last take, the Lens's own keys (⇥ ↩ Esc ← →), the keyboard handed back
//  before the fix is put back in the app, what the result line says when
//  the app still holds the paste and when it moved on, and the Lens closing
//  when dictation starts. The panel and the app are fakes: no window opens.
//

import AppKit
import Carbon.HIToolbox
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
final class RecordingLensPresenter: LensPresenting {
    private(set) var events: [String] = []

    func show(takingKeyboard: Bool) { events.append(takingKeyboard ? "show+key" : "show") }
    func releaseKeyboard() { events.append("release") }
    func hide() { events.append("hide") }
}

@MainActor
final class FakePasteBack {
    final class Anchor {
        let pasted: String
        init(pasted: String) { self.pasted = pasted }
    }

    var outcome: LensController.PasteBack.Outcome = .replaced
    private(set) var anchored: [String] = []
    private(set) var replaced: [(from: String, to: String, keysAllowed: Int)] = []

    var pasteBack: LensController.PasteBack {
        LensController.PasteBack(
            anchor: { [weak self] take in
                self?.anchored.append(take.pastedText)
                return Anchor(pasted: take.pastedText)
            },
            replace: { [weak self] anchor, corrected, keys in
                let from = (anchor as? Anchor)?.pasted ?? ""
                self?.replaced.append((from, corrected, keys))
                return self?.outcome ?? .replaced
            })
    }
}

@MainActor
struct LensControllerTests {

    private struct Fixture {
        let controller: LensController
        let presenter: RecordingLensPresenter
        let pasteBack: FakePasteBack
        let words: LearnedWordStore
        let keys: KeyCounter
    }

    @MainActor
    final class KeyCounter {
        var count = 0
    }

    private static let terminal = TargetApp(
        bundleID: "com.apple.Terminal", name: "Terminal", pid: 7)

    private func makeFixture(dictating: @escaping @MainActor () -> Bool = { false }) -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lens-controller-tests-\(UUID().uuidString)", isDirectory: true)
        let words = LearnedWordStore(directory: directory)
        let model = LensModel(learnedWords: words, pairs: nil, history: nil)
        let pasteBack = FakePasteBack()
        let keys = KeyCounter()
        let controller = LensController(
            model: model, pasteBack: pasteBack.pasteBack, inputCount: { keys.count },
            vocabulary: { ["Claude", "Tesseract"] }, isDictating: dictating)
        let presenter = RecordingLensPresenter()
        controller.presenter = presenter
        return Fixture(
            controller: controller, presenter: presenter, pasteBack: pasteBack, words: words,
            keys: keys)
    }

    private func take(_ text: String, pasted: Bool = true) -> DictatedTake {
        DictatedTake(pairID: nil, text: text, catches: [], app: Self.terminal, pasted: pasted)
    }

    private func press(_ f: Fixture, _ keyCode: Int) -> Bool {
        f.controller.handleKey(keyCode: keyCode, modifiers: [])
    }

    private func waitUntil(_ condition: () -> Bool) async {
        for _ in 0..<200 where !condition() {
            await Task.yield()
        }
    }

    // MARK: - Reopening

    @Test func theFixHotkeyReopensTheLastTakeWithTheKeyboard() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        #expect(f.pasteBack.anchored == ["Ask cloud why. "])

        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .afterPaste)
        #expect(f.controller.model.text == "Ask cloud why.")
        #expect(f.presenter.events == ["show+key"])
    }

    @Test func withNoTakeTheLensSaysSoWithoutTakingTheKeyboard() {
        let f = makeFixture()
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Nothing to fix yet")
        #expect(f.presenter.events == ["show"])
    }

    @Test func theFixHotkeyDoesNothingWhileDictating() {
        let f = makeFixture(dictating: { true })
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .hidden)
        #expect(f.presenter.events.isEmpty)
    }

    @Test func aTakeThatWasNotPastedIsNotAnchored() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why.", pasted: false))
        #expect(f.pasteBack.anchored.isEmpty)
    }

    // MARK: - Keys

    @Test func theLensKeysAreTakenAndLettersAreNot() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ship the build tonight."))
        f.controller.fixHotkeyPressed()

        #expect(press(f, kVK_LeftArrow))
        #expect(f.controller.model.target?.start == 3)
        #expect(press(f, kVK_LeftArrow))
        #expect(f.controller.model.target?.start == 2)
        #expect(press(f, kVK_RightArrow))
        #expect(f.controller.model.target?.start == 3)
        #expect(!press(f, kVK_ANSI_A))
        #expect(!f.controller.handleKey(keyCode: kVK_LeftArrow, modifiers: .command))
    }

    @Test func shiftArrowsWidenAndNarrowThePick() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Open the void design notes."))
        f.controller.fixHotkeyPressed()
        #expect(press(f, kVK_LeftArrow))
        #expect(press(f, kVK_LeftArrow))
        #expect(press(f, kVK_LeftArrow))
        #expect(f.controller.model.target == LensFix.Span(start: 2, count: 1))
        #expect(f.controller.handleKey(keyCode: kVK_RightArrow, modifiers: .shift))
        #expect(f.controller.model.target == LensFix.Span(start: 2, count: 2))
        #expect(f.controller.handleKey(keyCode: kVK_LeftArrow, modifiers: .shift))
        #expect(f.controller.model.target == LensFix.Span(start: 2, count: 1))

        f.controller.model.typed = "VoiceDesign"
        #expect(f.controller.handleKey(keyCode: kVK_RightArrow, modifiers: .shift))
        #expect(press(f, kVK_Tab))
        #expect(f.controller.model.text == "Open the VoiceDesign notes.")
    }

    @Test func tabFixesAndStays() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud about SRAX."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Tab))
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.text == "Ask Claude about SRAX.")
        #expect(f.pasteBack.replaced.isEmpty)
    }

    // MARK: - Finishing

    @Test func returnPutsTheFixBackAfterHandingTheKeyboardBack() async {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        for _ in 0..<6 { f.controller.noteLensInput() }  // "claude" typed into the Lens
        f.keys.count += 6
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Return))
        #expect(f.presenter.events == ["show+key", "release"])
        await waitUntil { f.controller.model.phase == .done }

        #expect(f.pasteBack.replaced.count == 1)
        #expect(f.pasteBack.replaced.first?.from == "Ask cloud why. ")
        #expect(f.pasteBack.replaced.first?.to == "Ask Claude why. ")
        #expect(f.pasteBack.replaced.first?.keysAllowed == 6)
        #expect(f.controller.model.result?.line == "Fixed in Terminal")
        #expect(f.controller.model.learnedSummary == "Learned cloud → Claude")
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        // Re-anchored on what the app now holds.
        #expect(f.pasteBack.anchored.last == "Ask Claude why. ")
    }

    @Test func whenTheAppMovedOnTheFixIsLearnedAndTheLensSaysSo() async {
        let f = makeFixture()
        f.pasteBack.outcome = .notInApp(reason: "you typed in Terminal since")
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Return))
        await waitUntil { f.controller.model.phase == .done }

        #expect(f.controller.model.result?.line == "Terminal already has the old text")
        #expect(f.controller.model.result?.detail == "you typed in Terminal since")
        #expect(f.words.word(heard: "cloud")?.meant == "Claude")
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        // Still anchored on the old paste: a later fix compares against it.
        #expect(f.pasteBack.anchored == ["Ask cloud why. "])
    }

    @Test func inputMadeInAnEarlierLensSessionStillCountsAsTheLenss() async {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        for _ in 0..<3 { f.controller.noteLensInput() }
        #expect(press(f, kVK_Escape))
        f.controller.fixHotkeyPressed()
        for _ in 0..<7 { f.controller.noteLensInput() }
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Return))
        await waitUntil { f.controller.model.phase == .done }
        #expect(f.pasteBack.replaced.first?.keysAllowed == 10)
    }

    @Test func theHotkeyWaitsWhileAFixIsBeingPutBack() async {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        // The put-back is in flight: the hotkey must not cancel or reopen.
        f.controller.fixHotkeyPressed()
        await waitUntil { f.controller.model.phase == .done }
        #expect(f.controller.model.result?.line == "Fixed in Terminal")
    }

    @Test func clickingAwayClosesTheLensAndKeepsItsFixes() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        f.controller.lensLostKeyboard()
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
    }

    @Test func aPasteLostAfterErasingDropsTheAnchor() async {
        let f = makeFixture()
        f.pasteBack.outcome = .lostOldText
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        await waitUntil { f.controller.model.phase == .done }

        #expect(f.controller.lastTake?.pasted == false)
        #expect(f.controller.model.result?.line == "The fix could not be pasted into Terminal")
    }

    @Test func returnWithNothingFixedJustCloses() {
        let f = makeFixture()
        f.controller.takeCommitted(take("All good."))
        f.controller.fixHotkeyPressed()

        #expect(press(f, kVK_Return))
        #expect(f.controller.model.phase == .hidden)
        #expect(f.presenter.events == ["show+key", "hide"])
        #expect(f.pasteBack.replaced.isEmpty)
    }

    @Test func returnWithATypedWordThatMatchesNothingStays() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Open the door."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "Tesseract"

        #expect(press(f, kVK_Return))
        #expect(f.controller.model.phase == .fixing)
    }

    @Test func aTakeThatWasNotPastedIsFixedAndLearnedOnly() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why.", pasted: false))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Return))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Fixed")
        #expect(f.pasteBack.replaced.isEmpty)
    }

    // MARK: - Esc

    @Test func escClearsTypingThenCloses() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Escape))
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.typed.isEmpty)
        #expect(press(f, kVK_Escape))
        #expect(f.controller.model.phase == .hidden)
    }

    @Test func escAfterAFixKeepsWhatItTaughtAndLeavesTheAppAlone() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        #expect(press(f, kVK_Escape))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Not changed in Terminal")
        #expect(f.pasteBack.replaced.isEmpty)
        #expect(f.words.word(heard: "cloud") != nil)
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
    }

    @Test func theFixHotkeyAgainClosesAnOpenLens() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .hidden)
    }

    // MARK: - Undo

    @Test func undoForgetsWhatTheFixTaught() async {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        await waitUntil { f.controller.model.phase == .done }

        f.controller.undo()
        #expect(f.words.word(heard: "cloud") == nil)
        #expect(f.controller.model.note == .undone)
    }

    // MARK: - Dictation

    @Test func startingToDictateClosesTheLens() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .fixing)

        feed.setPhase(.recording)
        await waitUntil { f.controller.model.phase == .hidden }
        #expect(f.controller.model.phase == .hidden)
        #expect(f.presenter.events.last == "hide")
    }
}
