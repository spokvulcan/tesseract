//
//  HotkeyTable.swift
//  tesseract
//
//  The **Hotkey Table**: the app's global hotkeys, one row each — the setting
//  that holds its combo, its default (the Settings Catalogue's), its label in
//  Settings, and what its key does, by the name of an action the composition
//  root supplies. `HotkeyTable` registers every row at launch and follows each
//  row's setting afterwards, so a changed combo re-binds that key at once. The
//  Hotkeys pane reads the same rows.
//
//  The registrar is the seam: `HotkeyManager` in the app, an in-memory peer in
//  the tests, which press a combo and see which action fires.
//

import AppKit
import Observation

/// One global hotkey: the table's rows, in the order the Hotkeys pane lists
/// them. The raw value is its registration id.
enum GlobalHotkey: String, CaseIterable {
    case dictation
    case fixLastTake
    case speakSelectedText = "tts"
    case talkToTesseract = "agent"
    case captureToReminders = "capture"
    case appshot

    /// Everything about this hotkey but what its actions are.
    var row: HotkeyRow {
        switch self {
        case .dictation:
            HotkeyRow(
                label: "Dictation Push-to-Talk", combo: \.hotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.hotkeyKeyCode, SettingsCatalogue.hotkeyModifiers),
                down: { $0.startDictation() }, up: { $0.finishDictation() },
                // A one-key dictation hotkey spoiled by another key: the owner
                // is typing with that modifier, not dictating.
                cancel: { $0.cancelDictation() },
                // ⇧ while talking: the take waits in the Lens (ADR-0086).
                shiftTap: { $0.holdTake() })
        case .fixLastTake:
            HotkeyRow(
                label: "Fix Last Take",
                help:
                    "Reopens the last dictated take so you can fix a word by typing the one you meant.",
                offersReset: true, combo: \.fixHotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.fixHotkeyKeyCode,
                    SettingsCatalogue.fixHotkeyModifiers),
                down: { $0.fixLastTake() })
        case .speakSelectedText:
            HotkeyRow(
                label: "Speak Selected Text", combo: \.ttsHotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.ttsHotkeyKeyCode,
                    SettingsCatalogue.ttsHotkeyModifiers),
                down: { $0.speakSelectedText() }, interruptsVoiceSession: true)
        case .talkToTesseract:
            HotkeyRow(
                label: "Talk to Tesseract", combo: \.agentHotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.agentHotkeyKeyCode,
                    SettingsCatalogue.agentHotkeyModifiers),
                down: { $0.startTalking() }, up: { $0.finishTalking() },
                interruptsVoiceSession: true)
        case .captureToReminders:
            // Tap to type, hold to speak. As one key (Right ⌥ by default),
            // another key pressed with it cancels: the owner is typing.
            HotkeyRow(
                label: "Capture to Reminders", offersReset: true, combo: \.captureHotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.captureHotkeyKeyCode,
                    SettingsCatalogue.captureHotkeyModifiers),
                down: { $0.captureKeyDown() }, up: { $0.captureKeyUp() },
                cancel: { $0.captureKeyCancelled() })
        case .appshot:
            // One-shot: no held state.
            HotkeyRow(
                label: "Capture Frontmost Window", offersReset: true, combo: \.appshotHotkey,
                defaultCombo: KeyCombo(
                    defaultOf: SettingsCatalogue.appshotHotkeyKeyCode,
                    SettingsCatalogue.appshotHotkeyModifiers),
                down: { $0.takeAppshot() })
        }
    }
}

/// A row of the Hotkey Table.
struct HotkeyRow {
    typealias Action = @MainActor (HotkeyActions) -> Void

    let label: String
    /// Effect-first help under the row in Settings, for a hotkey whose label
    /// alone doesn't say what it does.
    var help: String?
    /// Settings offers a reset to the default once the combo differs.
    var offersReset = false
    /// The setting that holds its combo.
    let combo: ReferenceWritableKeyPath<SettingsManager, KeyCombo>
    /// Its default, as the Settings Catalogue declares it.
    let defaultCombo: KeyCombo
    let down: Action
    var up: Action?
    /// A one-key hotkey spoiled: another key joined while it was held.
    var cancel: Action?
    /// ⇧ tapped while the key is held.
    var shiftTap: Action?
    /// While a voice session is active, a press interrupts it instead of
    /// doing `down` (ADR-0082).
    var interruptsVoiceSession = false

    /// The Hotkeys pane's way back to the default.
    var resetLabel: String { "Reset to \(defaultCombo.displayString)" }
}

/// What the hotkeys do, by name: one action per key event the table names,
/// supplied by the composition root.
struct HotkeyActions {
    var startDictation: @MainActor () -> Void
    var finishDictation: @MainActor () -> Void
    var cancelDictation: @MainActor () -> Void
    var holdTake: @MainActor () -> Void
    var fixLastTake: @MainActor () -> Void
    var speakSelectedText: @MainActor () -> Void
    var startTalking: @MainActor () -> Void
    var finishTalking: @MainActor () -> Void
    var captureKeyDown: @MainActor () -> Void
    var captureKeyUp: @MainActor () -> Void
    var captureKeyCancelled: @MainActor () -> Void
    var takeAppshot: @MainActor () -> Void
    /// Whether a voice session is active, read without building one.
    var isVoiceSessionActive: @MainActor () -> Bool
    /// Stops the session's speech and opens the mic for the owner's turn.
    var interruptVoiceSession: @MainActor () -> Void

    /// The interrupt key (ADR-0082): while a voice session is active, the
    /// press interrupts it — the speech stops and the mic opens for the
    /// owner's turn — instead of doing what the key otherwise does.
    func interruptingVoiceSession(otherwise press: () -> Void) {
        if isVoiceSessionActive() { interruptVoiceSession() } else { press() }
    }
}

/// Where the Hotkey Table puts its bindings: `HotkeyManager` in the app, an
/// in-memory peer in the tests.
protocol HotkeyRegistrar: AnyObject {
    /// Binds the registration's id to its combo and actions, replacing
    /// whatever that id had.
    func register(_ registration: HotkeyRegistration)
}

/// Registers every row of the table and follows each row's setting.
final class HotkeyTable {
    private let settings: SettingsManager
    private let registrar: any HotkeyRegistrar
    private let actions: HotkeyActions
    /// The combo each hotkey is bound to now.
    private var bound: [GlobalHotkey: KeyCombo] = [:]
    private var followers: [Task<Void, Never>] = []

    init(settings: SettingsManager, registrar: any HotkeyRegistrar, actions: HotkeyActions) {
        self.settings = settings
        self.registrar = registrar
        self.actions = actions
    }

    /// Following stops with the table: the followers hold it weakly.
    isolated deinit {
        for follower in followers { follower.cancel() }
    }

    /// Registers every row with its setting's combo, then follows each
    /// setting for as long as the table lives: a combo that changes re-binds
    /// its key. Idempotent.
    func start() {
        guard followers.isEmpty else { return }
        for hotkey in GlobalHotkey.allCases {
            bind(hotkey, to: settings[keyPath: hotkey.row.combo])
            followers.append(
                Task { [weak self, settings] in
                    let setting = hotkey.row.combo
                    for await combo in Observations({ settings[keyPath: setting] }) {
                        guard let self else { return }
                        if combo != self.bound[hotkey] { self.bind(hotkey, to: combo) }
                    }
                })
        }
    }

    private func bind(_ hotkey: GlobalHotkey, to combo: KeyCombo) {
        let row = hotkey.row
        let actions = actions
        let (up, cancel, shiftTap) = (row.up, row.cancel, row.shiftTap)
        bound[hotkey] = combo
        registrar.register(
            HotkeyRegistration(
                id: hotkey.rawValue, combo: combo,
                onDown: {
                    guard row.interruptsVoiceSession else { return row.down(actions) }
                    actions.interruptingVoiceSession { row.down(actions) }
                },
                onUp: up == nil ? nil : { up?(actions) },
                onCancel: cancel == nil ? nil : { cancel?(actions) },
                onShiftTap: shiftTap == nil ? nil : { shiftTap?(actions) }))
    }
}

extension KeyCombo {
    /// A hotkey's default as the Settings Catalogue declares it: the defaults
    /// of its key code and modifiers settings.
    init(defaultOf keyCode: Setting<Int>, _ modifiers: Setting<Int>) {
        self.init(
            keyCode: UInt16(keyCode.default),
            modifiers: NSEvent.ModifierFlags(rawValue: UInt(modifiers.default)))
    }
}
