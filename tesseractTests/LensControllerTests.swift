//
//  LensControllerTests.swift
//  tesseractTests
//
//  Pins the **Lens**'s flow around a fix (PRD #612): ⌃⌥Space reopens the
//  last take, the Lens's own keys (⇥ ↩ Esc ← →), the keyboard handed back
//  before the fix is put back in the app, what the result line says when
//  the app still holds the paste and when it moved on, the Lens closing
//  when dictation starts, and a take opened from the Dictation page
//  (refused while a take is recorded, keeping a take open in the Lens, its
//  fix carried into the take ⌃⌥Space reopens and nowhere else). The panel
//  and the app are fakes: no window opens.
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
    /// Where a held take's paste lands; nil fails the paste.
    var pasteApp: TargetApp? = TargetApp(
        bundleID: "com.apple.Terminal", name: "Terminal", pid: 7)
    private(set) var pastes: [String] = []
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
            },
            paste: { [weak self] text in
                self?.pastes.append(text)
                return self?.pasteApp
            })
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
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
            model: model, pasteBack: pasteBack.pasteBack,
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
        f.controller.model.typed = "claude"

        #expect(press(f, kVK_Return))
        #expect(f.presenter.events == ["show+key", "release"])
        await observe(until: { f.controller.model.phase == .done })

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
        await observe(until: { f.controller.model.phase == .done })

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
        await observe(until: { f.controller.model.phase == .done })
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
        await observe(until: { f.controller.model.phase == .done })
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
        await observe(until: { f.controller.model.phase == .done })

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
        await observe(until: { f.controller.model.phase == .done })

        f.controller.undo()
        #expect(f.words.word(heard: "cloud") == nil)
        #expect(f.controller.model.note == .undone)
    }

    // MARK: - Dictation

    @Test func startingToDictateTurnsAFixIntoListeningWithoutTheKeyboard() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })
        // The keyboard goes back before the new take can paste.
        #expect(Array(f.presenter.events.suffix(2)) == ["release", "show"])
        // The fix made with ⇥ stays with the take.
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
    }

    // MARK: - The Lens as the dictation overlay (slice 2)

    private func preview(_ text: String, confirmed: Int = 0) -> LivePreview {
        LivePreview(text: text, catches: [], confirmedTokens: confirmed)
    }

    @Test func theLensListensStreamsAndLandsTheTake() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)

        feed.setTargetApp(Self.terminal)
        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })
        #expect(f.presenter.events == ["release", "show"])
        #expect(f.controller.model.liveApp == Self.terminal)

        feed.setPreview(preview("Ask cloud why this", confirmed: 2))
        await observe(until: { f.controller.model.preview != nil })
        #expect(f.controller.model.liveTokens.count == 4)

        feed.setPhase(.processing)
        await observe(until: { f.controller.model.phase == .finishing })
        // The words stay up while the full pass runs.
        #expect(f.controller.model.preview?.text == "Ask cloud why this")

        f.controller.takeCommitted(take("Ask cloud why the"))
        #expect(f.controller.model.phase == .landed)
        // "this" became "the": that word settles.
        #expect(f.controller.model.settled == [3])
    }

    @Test func aHeldTakeWaitsAndReturnPastesIt() async {
        let f = makeFixture()
        let held = DictatedTake(
            pairID: nil, text: "Ask cloud why.", catches: [], app: Self.terminal, pasted: false,
            held: true)
        f.controller.takeCommitted(held)
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .held)
        #expect(f.presenter.events == ["show+key"])

        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        await observe(until: { f.controller.model.phase == .done })
        #expect(f.pasteBack.pastes == ["Ask Claude why. "])
        #expect(f.controller.model.result?.line == "Pasted into Terminal")
        #expect(f.controller.lastTake?.pasted == true)
        #expect(f.controller.lastTake?.held == false)
        // Anchored on the paste, so ⌃⌥Space can still fix it.
        #expect(f.pasteBack.anchored == ["Ask Claude why. "])
    }

    @Test func escKeepsAHeldTakeAndTheHotkeyBringsItBack() {
        let f = makeFixture()
        let held = DictatedTake(
            pairID: nil, text: "Ask cloud why.", catches: [], app: Self.terminal, pasted: false,
            held: true)
        f.controller.takeCommitted(held)

        #expect(press(f, kVK_Escape))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Kept, not pasted")
        #expect(f.pasteBack.pastes.isEmpty)

        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .held)
    }

    @Test func aFailedPasteKeepsTheHeldTake() async {
        let f = makeFixture()
        f.pasteBack.pasteApp = nil
        f.controller.takeCommitted(
            DictatedTake(
                pairID: nil, text: "Ship it.", catches: [], app: Self.terminal, pasted: false,
                held: true))
        #expect(press(f, kVK_Return))
        await observe(until: { f.controller.model.phase == .done })
        #expect(f.controller.model.result?.line == "Couldn't paste the take")
        #expect(f.controller.lastTake?.held == true)
    }

    @Test func aSilentTakeSaysSo() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })

        feed.setPhase(.error(.noSpeechDetected))
        await observe(until: { f.controller.model.phase == .done })
        #expect(f.controller.model.result?.line == "Didn't hear anything")
    }

    @Test func aRejectedTakeOffersInsertAnyway() async {
        let f = makeFixture()
        let feed = DictationFeed()
        var inserted = 0
        f.controller.onInsertRawAnyway = { inserted += 1 }
        f.controller.watch(feed)
        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })
        feed.setPhase(.processing)
        feed.emit(.rejected(raw: "asdf", reason: "unintelligible"))
        await observe(until: { f.controller.model.canInsertRaw })
        #expect(f.controller.model.result?.line == "Didn't catch that")

        f.controller.insertRawAnyway()
        #expect(inserted == 1)
        #expect(f.controller.model.phase == .hidden)
    }

    @Test func anErrorAtThePressShowsEvenWithTheLensClosed() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)

        feed.setPhase(.error(.microphoneBusy))
        await observe(until: { f.controller.model.phase == .done })
        #expect(f.controller.model.result?.line == "The microphone is in use")
        #expect(f.presenter.events.last == "show")
    }

    @Test func anErrorLeavesATakeBeingFixedAlone() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()

        feed.setPhase(.error(.textInjectionFailed("refused")))
        for _ in 0..<200 { await Task.yield() }
        #expect(f.controller.model.phase == .fixing)
    }

    @Test func aNewTakesCardCarriesNothingOfTheLastFix() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })
        #expect(f.controller.model.text.isEmpty)
        #expect(f.controller.model.receipts.isEmpty)
        #expect(f.controller.model.fixedTokens.isEmpty)
    }

    @Test func withAutoInsertOffNoHoldIsPromised() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lens-noinsert-\(UUID().uuidString)", isDirectory: true)
        let controller = LensController(
            model: LensModel(
                learnedWords: LearnedWordStore(directory: directory), pairs: nil, history: nil),
            pasteBack: .none, vocabulary: { [] }, pastes: { false })
        let presenter = RecordingLensPresenter()
        controller.presenter = presenter
        let feed = DictationFeed()
        controller.watch(feed)
        feed.setPhase(.recording)
        await observe(until: { controller.model.phase == .listening })
        #expect(controller.model.holdHint == nil)
    }

    // MARK: - A take opened from the Dictation page (slice 3)

    private static let terminalEntry = TranscriptionEntry.App(
        bundleID: "com.apple.Terminal", name: "Terminal")

    /// A take as the Dictation page opens it, from its history entry.
    private func pageTake(_ text: String, pairID: UUID? = UUID()) -> DictatedTake {
        DictatedTake.fromPage(
            pairID: pairID, text: text, catches: [], app: Self.terminalEntry, at: Date())
    }

    /// The take ⌃⌥Space reopens, pasted under its Correction Pair.
    private func pastedTake(
        _ text: String, pairID: UUID, pastedInto: TargetApp? = nil
    ) -> DictatedTake {
        DictatedTake(
            pairID: pairID, text: text, catches: [], app: Self.terminal, pasted: true,
            pastedInto: pastedInto)
    }

    @Test func aPageTakeOpensForFixingWithTheKeyboard() {
        let f = makeFixture()
        let opened = f.controller.openFromPage(pageTake("Ask cloud why."))

        #expect(opened)
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .fromPage)
        #expect(f.controller.model.text == "Ask cloud why.")
        #expect(f.presenter.events == ["show+key"])
        // Opening a take from the page does not make it the last take.
        #expect(f.controller.lastTake == nil)
    }

    @Test func aPageTakeIsRefusedWhileDictating() {
        let f = makeFixture(dictating: { true })
        let opened = f.controller.openFromPage(pageTake("Ask cloud why."))

        #expect(!opened)
        #expect(f.controller.model.phase == .hidden)
        #expect(f.presenter.events.isEmpty)
    }

    /// A click on the page closes a waiting take first (the Lens loses the
    /// keyboard); the context menu does not, so opening keeps it the same
    /// way: unpasted, with its fixes, for ⌃⌥Space.
    @Test func aPageTakeKeepsAHeldTakeWaitingForTheFixHotkey() {
        let f = makeFixture()
        let held = DictatedTake(
            pairID: UUID(), text: "Ask cloud why.", catches: [], app: Self.terminal,
            pasted: false, held: true)
        f.controller.takeCommitted(held)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        let page = pageTake("Ship the build tonight.")
        let opened = f.controller.openFromPage(page)

        #expect(opened)
        #expect(f.controller.model.mode == .fromPage)
        #expect(f.controller.model.take == page)
        #expect(f.pasteBack.pastes.isEmpty)
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        #expect(f.controller.lastTake?.held == true)
        #expect(f.controller.lastTake?.pasted == false)

        #expect(press(f, kVK_Escape))
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.mode == .held)
        #expect(f.controller.model.text == "Ask Claude why.")
    }

    @Test func aPageTakeKeepsTheFixesOfTheLastTakeBeingFixed() {
        let f = makeFixture()
        f.controller.takeCommitted(take("Ask cloud why."))
        f.controller.fixHotkeyPressed()
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        let opened = f.controller.openFromPage(pageTake("Ship it."))

        #expect(opened)
        #expect(f.controller.model.mode == .fromPage)
        #expect(f.controller.model.text == "Ship it.")
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
    }

    @Test func aSecondPageTakeReplacesTheFirst() {
        let f = makeFixture()
        let second = pageTake("Ship the build tonight.")
        let openedFirst = f.controller.openFromPage(pageTake("Ask cloud why."))
        let openedSecond = f.controller.openFromPage(second)

        #expect(openedFirst)
        #expect(openedSecond)
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .fromPage)
        #expect(f.controller.model.take == second)
        #expect(f.controller.model.text == "Ship the build tonight.")
    }

    @Test func aPageFixToTheLastTakeCarriesIntoTheReopen() {
        let f = makeFixture()
        let pairID = UUID()
        let textEdit = TargetApp(bundleID: "com.apple.TextEdit", name: "TextEdit", pid: 9)
        f.controller.takeCommitted(
            pastedTake("Ask cloud why.", pairID: pairID, pastedInto: textEdit))

        let opened = f.controller.openFromPage(pageTake("Ask cloud why.", pairID: pairID))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Fixed")
        // Nothing is put back in the app from the page.
        #expect(f.pasteBack.replaced.isEmpty)

        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .fixing)
        #expect(f.controller.model.mode == .afterPaste)
        #expect(f.controller.model.text == "Ask Claude why.")
        let reopened = f.controller.model.take
        #expect(reopened?.pairID == pairID)
        #expect(reopened?.pasted == true)
        #expect(reopened?.pastedInto == textEdit)
        #expect(reopened?.app == Self.terminal)
    }

    @Test func aPageFixToAnotherTakeLeavesTheLastTakeAlone() {
        let f = makeFixture()
        let last = pastedTake("Ship the build tonight.", pairID: UUID())
        f.controller.takeCommitted(last)

        let opened = f.controller.openFromPage(pageTake("Ask cloud why."))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.lastTake == last)

        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.text == "Ship the build tonight.")
    }

    @Test func aPageTakeWithNoPairNeverCarriesIntoTheLastTake() {
        let f = makeFixture()
        let last = take("Ask cloud why.")
        f.controller.takeCommitted(last)

        let opened = f.controller.openFromPage(pageTake("Ask cloud why.", pairID: nil))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Return))
        #expect(f.controller.lastTake == last)
    }

    @Test func cancellingAChangedPageTakeDoesNotMakeItTheLastTake() {
        let f = makeFixture()
        let opened = f.controller.openFromPage(pageTake("Ask cloud why."))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))
        #expect(f.controller.model.isChanged)

        #expect(press(f, kVK_Escape))
        #expect(f.controller.model.phase == .done)
        #expect(f.controller.model.result?.line == "Fixed")
        #expect(f.controller.lastTake == nil)
        // What the fix taught stays.
        #expect(f.words.word(heard: "cloud")?.meant == "Claude")

        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.result?.line == "Nothing to fix yet")
    }

    @Test func cancellingAChangedPageTakeLeavesAnotherLastTakeAlone() {
        let f = makeFixture()
        let last = pastedTake("Ship the build tonight.", pairID: UUID())
        f.controller.takeCommitted(last)

        let opened = f.controller.openFromPage(pageTake("Ask cloud why."))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))
        f.controller.cancelFix()

        #expect(f.controller.model.phase == .done)
        #expect(f.controller.lastTake == last)
    }

    @Test func cancellingAPageFixToTheLastTakeStillCarriesIt() {
        let f = makeFixture()
        let pairID = UUID()
        f.controller.takeCommitted(pastedTake("Ask cloud why.", pairID: pairID))

        let opened = f.controller.openFromPage(pageTake("Ask cloud why.", pairID: pairID))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))
        // ⌃⌥Space on an open take closes it.
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.phase == .done)

        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        #expect(f.controller.lastTake?.pasted == true)
        f.controller.fixHotkeyPressed()
        #expect(f.controller.model.mode == .afterPaste)
        #expect(f.controller.model.text == "Ask Claude why.")
    }

    @Test func openingAnotherPageTakeKeepsAFixToTheLastTake() {
        let f = makeFixture()
        let pairID = UUID()
        f.controller.takeCommitted(pastedTake("Ask cloud why.", pairID: pairID))

        let openedFirst = f.controller.openFromPage(pageTake("Ask cloud why.", pairID: pairID))
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))
        let openedSecond = f.controller.openFromPage(pageTake("Ship the build tonight."))

        #expect(openedFirst)
        #expect(openedSecond)
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        #expect(f.controller.lastTake?.pairID == pairID)
    }

    @Test func startingToDictateKeepsAPageFixToTheLastTake() async {
        let f = makeFixture()
        let feed = DictationFeed()
        f.controller.watch(feed)
        let pairID = UUID()
        f.controller.takeCommitted(pastedTake("Ask cloud why.", pairID: pairID))

        let opened = f.controller.openFromPage(pageTake("Ask cloud why.", pairID: pairID))
        #expect(opened)
        f.controller.model.typed = "claude"
        #expect(press(f, kVK_Tab))

        feed.setPhase(.recording)
        await observe(until: { f.controller.model.phase == .listening })
        #expect(f.controller.lastTake?.text == "Ask Claude why.")
        #expect(f.controller.lastTake?.pasted == true)
    }
}
