//
//  HotkeyTableTests.swift
//  tesseractTests
//
//  The **Hotkey Table** through its interface: settings written through the
//  facade (over the in-memory Settings Store Adapter), keys pressed on an
//  in-memory registrar — the peer of `HotkeyManager` at the registrar seam —
//  and the actions the table's rows name, recorded in order.
//

import AppKit
import Foundation
import Observation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct HotkeyTableTests {

    @Test func launchBindsEveryRowToItsSetting() {
        let h = TableHarness()
        defer { h.table.stop() }

        h.table.start()

        for hotkey in GlobalHotkey.allCases {
            #expect(h.registrar.combo(of: hotkey) == h.settings[keyPath: hotkey.row.combo])
        }
        #expect(h.registrar.registrationCount == GlobalHotkey.allCases.count)
    }

    @Test func eachKeyDoesWhatItsRowSays() {
        let h = TableHarness()
        defer { h.table.stop() }
        h.table.start()

        h.registrar.press(h.settings.hotkey)
        h.registrar.shiftTap(h.settings.hotkey)
        h.registrar.release(h.settings.hotkey)
        h.registrar.press(h.settings.hotkey)
        h.registrar.cancel(h.settings.hotkey)
        h.registrar.press(h.settings.fixHotkey)
        h.registrar.press(h.settings.ttsHotkey)
        h.registrar.press(h.settings.agentHotkey)
        h.registrar.release(h.settings.agentHotkey)
        h.registrar.press(h.settings.captureHotkey)
        h.registrar.release(h.settings.captureHotkey)
        h.registrar.press(h.settings.captureHotkey)
        h.registrar.cancel(h.settings.captureHotkey)
        h.registrar.press(h.settings.appshotHotkey)

        #expect(
            h.actions.done == [
                "startDictation", "holdTake", "finishDictation", "startDictation",
                "cancelDictation", "fixLastTake", "speakSelectedText", "startTalking",
                "finishTalking", "captureKeyDown", "captureKeyUp", "captureKeyDown",
                "captureKeyCancelled", "takeAppshot",
            ])
    }

    /// A changed setting moves its key: the old combo does nothing, the new
    /// one fires the row's action.
    @Test func aChangedSettingMovesItsKey() async {
        let h = TableHarness()
        defer { h.table.stop() }
        h.table.start()
        let old = h.settings.fixHotkey
        let new = KeyCombo(keyCode: 3, modifiers: [.control, .option])

        h.settings.fixHotkey = new

        #expect(await observe(until: { h.registrar.combo(of: .fixLastTake) == new }))
        h.registrar.press(old)
        #expect(h.actions.done.isEmpty)
        h.registrar.press(new)
        #expect(h.actions.done == ["fixLastTake"])
    }

    /// Only a change re-binds: launch binds each row once, and a setting
    /// written back to the combo it already has binds nothing again.
    @Test func anUnchangedComboIsNeverReBound() async {
        let h = TableHarness()
        defer { h.table.stop() }
        h.table.start()
        let rows = GlobalHotkey.allCases.count

        h.settings.hotkey = h.settings.hotkey
        h.settings.ttsHotkey = KeyCombo(keyCode: 17, modifiers: [.command, .shift])

        #expect(
            await observe(until: {
                h.registrar.combo(of: .speakSelectedText)
                    == KeyCombo(keyCode: 17, modifiers: [.command, .shift])
            }))
        for _ in 0..<100 { await Task.yield() }
        #expect(h.registrar.registrationCount == rows + 1)
    }

    /// ADR-0082: while a voice session is active, Speak Selected Text and
    /// Talk to Tesseract interrupt it instead of doing what they otherwise
    /// do; the other keys never do.
    @Test func theInterruptKeysInterruptAnActiveVoiceSession() {
        let h = TableHarness()
        defer { h.table.stop() }
        h.table.start()
        h.actions.voiceSessionActive = true

        h.registrar.press(h.settings.ttsHotkey)
        h.registrar.press(h.settings.agentHotkey)
        h.registrar.release(h.settings.agentHotkey)
        h.registrar.press(h.settings.hotkey)

        #expect(
            h.actions.done == [
                "interruptVoiceSession", "interruptVoiceSession", "finishTalking",
                "startDictation",
            ])
        #expect(
            GlobalHotkey.allCases.filter(\.row.interruptsVoiceSession) == [
                .speakSelectedText, .talkToTesseract,
            ])

        h.actions.voiceSessionActive = false
        h.registrar.press(h.settings.ttsHotkey)
        #expect(h.actions.done.last == "speakSelectedText")
    }

    /// Each default is the Settings Catalogue's (a fresh settings store reads
    /// it), and the pane's reset says it.
    @Test func everyDefaultIsTheCataloguesAndTheResetNamesIt() {
        let fresh = SettingsManager(store: InMemorySettingsStore())
        for hotkey in GlobalHotkey.allCases {
            #expect(hotkey.row.defaultCombo == fresh[keyPath: hotkey.row.combo], "\(hotkey)")
        }
        #expect(GlobalHotkey.fixLastTake.row.resetLabel == "Reset to ⌃⌥Space")
        #expect(GlobalHotkey.captureToReminders.row.resetLabel == "Reset to Right ⌥")
        #expect(GlobalHotkey.appshot.row.resetLabel == "Reset to ⌘⌘")
    }
}

// MARK: - Harness

@MainActor
private struct TableHarness {
    let settings = SettingsManager(store: InMemorySettingsStore())
    let registrar = InMemoryHotkeyRegistrar()
    let actions = ActionRecorder()
    let table: HotkeyTable

    init() {
        table = HotkeyTable(settings: settings, registrar: registrar, actions: actions.actions)
    }
}

/// The in-memory peer of `HotkeyManager`: keeps each id's registration and
/// fires the one bound to a combo when a test presses it.
@Observable @MainActor
private final class InMemoryHotkeyRegistrar: HotkeyRegistrar {
    private var registrations: [String: HotkeyRegistration] = [:]
    private(set) var registrationCount = 0

    func register(_ registration: HotkeyRegistration) {
        registrations[registration.id] = registration
        registrationCount += 1
    }

    func combo(of hotkey: GlobalHotkey) -> KeyCombo? {
        registrations[hotkey.rawValue]?.combo
    }

    private func bound(to combo: KeyCombo) -> HotkeyRegistration? {
        registrations.values.first { $0.combo == combo }
    }

    func press(_ combo: KeyCombo) { bound(to: combo)?.onDown() }
    func release(_ combo: KeyCombo) { bound(to: combo)?.onUp?() }
    func cancel(_ combo: KeyCombo) { bound(to: combo)?.onCancel?() }
    func shiftTap(_ combo: KeyCombo) { bound(to: combo)?.onShiftTap?() }
}

/// Records every named action the table fires.
@MainActor
private final class ActionRecorder {
    private(set) var done: [String] = []
    var voiceSessionActive = false

    var actions: HotkeyActions {
        HotkeyActions(
            startDictation: { [self] in done.append("startDictation") },
            finishDictation: { [self] in done.append("finishDictation") },
            cancelDictation: { [self] in done.append("cancelDictation") },
            holdTake: { [self] in done.append("holdTake") },
            fixLastTake: { [self] in done.append("fixLastTake") },
            speakSelectedText: { [self] in done.append("speakSelectedText") },
            startTalking: { [self] in done.append("startTalking") },
            finishTalking: { [self] in done.append("finishTalking") },
            captureKeyDown: { [self] in done.append("captureKeyDown") },
            captureKeyUp: { [self] in done.append("captureKeyUp") },
            captureKeyCancelled: { [self] in done.append("captureKeyCancelled") },
            takeAppshot: { [self] in done.append("takeAppshot") },
            isVoiceSessionActive: { [self] in voiceSessionActive },
            interruptVoiceSession: { [self] in done.append("interruptVoiceSession") })
    }
}
