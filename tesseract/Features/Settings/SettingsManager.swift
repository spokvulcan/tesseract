//
//  SettingsManager.swift
//  tesseract
//

import Foundation
import Observation
import ServiceManagement
import AppKit

@Observable @MainActor
final class SettingsManager {

    // MARK: - UserDefaults Keys

    private enum Key {
        static let launchAtLogin = "launchAtLogin"
        static let showInDock = "showInDock"
        static let showInMenuBar = "showInMenuBar"
        static let autoInsertText = "autoInsertText"
        static let restoreClipboard = "restoreClipboard"
        static let overlayStyle = "overlayStyle"
        static let glowTheme = "glowTheme"
        static let selectedMicrophoneUID = "selectedMicrophoneUID"
        static let language = "language"
        static let hotkeyKeyCode = "hotkeyKeyCode"
        static let hotkeyModifiers = "hotkeyModifiers"
        static let ttsHotkeyKeyCode = "ttsHotkeyKeyCode"
        static let ttsHotkeyModifiers = "ttsHotkeyModifiers"
        static let agentHotkeyKeyCode = "agentHotkeyKeyCode"
        static let agentHotkeyModifiers = "agentHotkeyModifiers"
        static let ttsTemperature = "ttsTemperature"
        static let ttsTopP = "ttsTopP"
        static let ttsRepetitionPenalty = "ttsRepetitionPenalty"
        static let ttsMaxTokens = "ttsMaxTokens"
        static let ttsSeed = "ttsSeed"
        static let ttsVoiceDescription = "ttsVoiceDescription"
        static let ttsLanguage = "ttsLanguage"
        static let ttsStreamingEnabled = "ttsStreamingEnabled"
        static let agentAutoSpeak = "agentAutoSpeak"
        static let selectedAgentModelID = "selectedAgentModelID"
        static let maxRecordingDuration = "maxRecordingDuration"
        static let playSounds = "playSounds"
        static let showNotifications = "showNotifications"
        static let hasCompletedOnboarding = "hasCompletedOnboarding"
    }

    // MARK: - General Settings

    var launchAtLogin = false {
        didSet {
            UserDefaults.standard.set(launchAtLogin, forKey: Key.launchAtLogin)
            updateLaunchAtLogin()
        }
    }

    var showInDock = true {
        didSet {
            UserDefaults.standard.set(showInDock, forKey: Key.showInDock)
            applyDockVisibility()
        }
    }

    var showInMenuBar = true {
        didSet { UserDefaults.standard.set(showInMenuBar, forKey: Key.showInMenuBar) }
    }

    var autoInsertText = true {
        didSet { UserDefaults.standard.set(autoInsertText, forKey: Key.autoInsertText) }
    }

    var restoreClipboard = true {
        didSet { UserDefaults.standard.set(restoreClipboard, forKey: Key.restoreClipboard) }
    }

    var overlayStyleRaw: String = OverlayStyle.pill.rawValue {
        didSet { UserDefaults.standard.set(overlayStyleRaw, forKey: Key.overlayStyle) }
    }

    var overlayStyle: OverlayStyle {
        get { OverlayStyle(rawValue: overlayStyleRaw) ?? .pill }
        set { overlayStyleRaw = newValue.rawValue }
    }

    var glowThemeRaw: String = GlowTheme.appleIntelligence.rawValue {
        didSet { UserDefaults.standard.set(glowThemeRaw, forKey: Key.glowTheme) }
    }

    var glowTheme: GlowTheme {
        get { GlowTheme(rawValue: glowThemeRaw) ?? .appleIntelligence }
        set { glowThemeRaw = newValue.rawValue }
    }

    // MARK: - Audio Settings

    var selectedMicrophoneUID: String = "" {
        didSet { UserDefaults.standard.set(selectedMicrophoneUID, forKey: Key.selectedMicrophoneUID) }
    }

    // MARK: - Language Settings

    var language: String = "en" {
        didSet { UserDefaults.standard.set(language, forKey: Key.language) }
    }

    var selectedLanguage: SupportedLanguage {
        SupportedLanguage.language(forCode: language) ?? .auto
    }

    // MARK: - Hotkey Settings

    var hotkeyKeyCode: Int = Int(KeyCombo.optionSpace.keyCode) {
        didSet { UserDefaults.standard.set(hotkeyKeyCode, forKey: Key.hotkeyKeyCode) }
    }

    var hotkeyModifiers: Int = Int(KeyCombo.optionSpace.modifiers) {
        didSet { UserDefaults.standard.set(hotkeyModifiers, forKey: Key.hotkeyModifiers) }
    }

    var hotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(hotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(hotkeyModifiers))
            )
        }
        set {
            hotkeyKeyCode = Int(newValue.keyCode)
            hotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - TTS Hotkey

    var ttsHotkeyKeyCode: Int = Int(KeyCombo.functionSpace.keyCode) {
        didSet { UserDefaults.standard.set(ttsHotkeyKeyCode, forKey: Key.ttsHotkeyKeyCode) }
    }

    var ttsHotkeyModifiers: Int = Int(KeyCombo.functionSpace.modifiers) {
        didSet { UserDefaults.standard.set(ttsHotkeyModifiers, forKey: Key.ttsHotkeyModifiers) }
    }

    var ttsHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(ttsHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(ttsHotkeyModifiers))
            )
        }
        set {
            ttsHotkeyKeyCode = Int(newValue.keyCode)
            ttsHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - Agent Hotkey

    var agentHotkeyKeyCode: Int = Int(KeyCombo.controlSpace.keyCode) {
        didSet { UserDefaults.standard.set(agentHotkeyKeyCode, forKey: Key.agentHotkeyKeyCode) }
    }

    var agentHotkeyModifiers: Int = Int(KeyCombo.controlSpace.modifiers) {
        didSet { UserDefaults.standard.set(agentHotkeyModifiers, forKey: Key.agentHotkeyModifiers) }
    }

    var agentHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(agentHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(agentHotkeyModifiers))
            )
        }
        set {
            agentHotkeyKeyCode = Int(newValue.keyCode)
            agentHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - TTS Settings

    var ttsTemperature: Double = 0.6 {
        didSet { UserDefaults.standard.set(ttsTemperature, forKey: Key.ttsTemperature) }
    }

    var ttsTopP: Double = 0.8 {
        didSet { UserDefaults.standard.set(ttsTopP, forKey: Key.ttsTopP) }
    }

    var ttsRepetitionPenalty: Double = 1.3 {
        didSet { UserDefaults.standard.set(ttsRepetitionPenalty, forKey: Key.ttsRepetitionPenalty) }
    }

    var ttsMaxTokens: Int = 4096 {
        didSet { UserDefaults.standard.set(ttsMaxTokens, forKey: Key.ttsMaxTokens) }
    }

    var ttsSeed: Int = 0 {
        didSet { UserDefaults.standard.set(ttsSeed, forKey: Key.ttsSeed) }
    }

    var ttsVoiceDescription: String = "" {
        didSet { UserDefaults.standard.set(ttsVoiceDescription, forKey: Key.ttsVoiceDescription) }
    }

    var ttsLanguage: String = "English" {
        didSet { UserDefaults.standard.set(ttsLanguage, forKey: Key.ttsLanguage) }
    }

    var ttsStreamingEnabled = true {
        didSet { UserDefaults.standard.set(ttsStreamingEnabled, forKey: Key.ttsStreamingEnabled) }
    }

    var agentAutoSpeak = false {
        didSet { UserDefaults.standard.set(agentAutoSpeak, forKey: Key.agentAutoSpeak) }
    }

    var selectedAgentModelID: String = ModelDefinition.defaultAgentModelID {
        didSet { UserDefaults.standard.set(selectedAgentModelID, forKey: Key.selectedAgentModelID) }
    }

    var ttsParameters: TTSParameters {
        get {
            TTSParameters(
                temperature: Float(ttsTemperature),
                topP: Float(ttsTopP),
                repetitionPenalty: Float(ttsRepetitionPenalty),
                maxTokens: ttsMaxTokens,
                seed: UInt64(ttsSeed)
            )
        }
        set {
            ttsTemperature = Double(newValue.temperature)
            ttsTopP = Double(newValue.topP)
            ttsRepetitionPenalty = Double(newValue.repetitionPenalty)
            ttsMaxTokens = newValue.maxTokens
            ttsSeed = Int(newValue.seed)
        }
    }

    // MARK: - Advanced Settings

    var maxRecordingDuration: Double = 300.0 {
        didSet { UserDefaults.standard.set(maxRecordingDuration, forKey: Key.maxRecordingDuration) }
    }

    var playSounds = true {
        didSet { UserDefaults.standard.set(playSounds, forKey: Key.playSounds) }
    }

    var showNotifications = true {
        didSet { UserDefaults.standard.set(showNotifications, forKey: Key.showNotifications) }
    }

    // MARK: - Onboarding

    var hasCompletedOnboarding = false {
        didSet { UserDefaults.standard.set(hasCompletedOnboarding, forKey: Key.hasCompletedOnboarding) }
    }

    // MARK: - Init

    init() {
        let ud = UserDefaults.standard

        // Register defaults so reads return correct values before first explicit write.
        ud.register(defaults: [
            Key.launchAtLogin: false,
            Key.showInDock: true,
            Key.showInMenuBar: true,
            Key.autoInsertText: true,
            Key.restoreClipboard: true,
            Key.overlayStyle: OverlayStyle.pill.rawValue,
            Key.glowTheme: GlowTheme.appleIntelligence.rawValue,
            Key.selectedMicrophoneUID: "",
            Key.language: "en",
            Key.hotkeyKeyCode: Int(KeyCombo.optionSpace.keyCode),
            Key.hotkeyModifiers: Int(KeyCombo.optionSpace.modifiers),
            Key.ttsHotkeyKeyCode: Int(KeyCombo.functionSpace.keyCode),
            Key.ttsHotkeyModifiers: Int(KeyCombo.functionSpace.modifiers),
            Key.agentHotkeyKeyCode: Int(KeyCombo.controlSpace.keyCode),
            Key.agentHotkeyModifiers: Int(KeyCombo.controlSpace.modifiers),
            Key.ttsTemperature: 0.6,
            Key.ttsTopP: 0.8,
            Key.ttsRepetitionPenalty: 1.3,
            Key.ttsMaxTokens: 4096,
            Key.ttsSeed: 0,
            Key.ttsVoiceDescription: "",
            Key.ttsLanguage: "English",
            Key.ttsStreamingEnabled: true,
            Key.agentAutoSpeak: false,
            Key.selectedAgentModelID: ModelDefinition.defaultAgentModelID,
            Key.maxRecordingDuration: 300.0,
            Key.playSounds: true,
            Key.showNotifications: true,
            Key.hasCompletedOnboarding: false,
        ])

        // Load persisted values (didSet does NOT fire during init).
        launchAtLogin = ud.bool(forKey: Key.launchAtLogin)
        showInDock = ud.bool(forKey: Key.showInDock)
        showInMenuBar = ud.bool(forKey: Key.showInMenuBar)
        autoInsertText = ud.bool(forKey: Key.autoInsertText)
        restoreClipboard = ud.bool(forKey: Key.restoreClipboard)
        overlayStyleRaw = ud.string(forKey: Key.overlayStyle) ?? OverlayStyle.pill.rawValue
        glowThemeRaw = ud.string(forKey: Key.glowTheme) ?? GlowTheme.appleIntelligence.rawValue
        selectedMicrophoneUID = ud.string(forKey: Key.selectedMicrophoneUID) ?? ""
        language = ud.string(forKey: Key.language) ?? "en"
        hotkeyKeyCode = ud.integer(forKey: Key.hotkeyKeyCode)
        hotkeyModifiers = ud.integer(forKey: Key.hotkeyModifiers)
        ttsHotkeyKeyCode = ud.integer(forKey: Key.ttsHotkeyKeyCode)
        ttsHotkeyModifiers = ud.integer(forKey: Key.ttsHotkeyModifiers)
        agentHotkeyKeyCode = ud.integer(forKey: Key.agentHotkeyKeyCode)
        agentHotkeyModifiers = ud.integer(forKey: Key.agentHotkeyModifiers)
        ttsTemperature = ud.double(forKey: Key.ttsTemperature)
        ttsTopP = ud.double(forKey: Key.ttsTopP)
        ttsRepetitionPenalty = ud.double(forKey: Key.ttsRepetitionPenalty)
        ttsMaxTokens = ud.integer(forKey: Key.ttsMaxTokens)
        ttsSeed = ud.integer(forKey: Key.ttsSeed)
        ttsVoiceDescription = ud.string(forKey: Key.ttsVoiceDescription) ?? ""
        ttsLanguage = ud.string(forKey: Key.ttsLanguage) ?? "English"
        ttsStreamingEnabled = ud.bool(forKey: Key.ttsStreamingEnabled)
        agentAutoSpeak = ud.bool(forKey: Key.agentAutoSpeak)
        selectedAgentModelID = ud.string(forKey: Key.selectedAgentModelID) ?? ModelDefinition.defaultAgentModelID
        maxRecordingDuration = ud.double(forKey: Key.maxRecordingDuration)
        playSounds = ud.bool(forKey: Key.playSounds)
        showNotifications = ud.bool(forKey: Key.showNotifications)
        hasCompletedOnboarding = ud.bool(forKey: Key.hasCompletedOnboarding)
    }

    // MARK: - Methods

    func resetToDefaults() {
        launchAtLogin = false
        showInDock = true
        showInMenuBar = true
        autoInsertText = true
        restoreClipboard = true
        overlayStyleRaw = OverlayStyle.pill.rawValue
        glowThemeRaw = GlowTheme.appleIntelligence.rawValue
        selectedMicrophoneUID = ""
        language = "en"
        hotkeyKeyCode = Int(KeyCombo.optionSpace.keyCode)
        hotkeyModifiers = Int(KeyCombo.optionSpace.modifiers)
        maxRecordingDuration = 300.0
        playSounds = true
        showNotifications = true
        ttsHotkeyKeyCode = Int(KeyCombo.functionSpace.keyCode)
        ttsHotkeyModifiers = Int(KeyCombo.functionSpace.modifiers)
        agentHotkeyKeyCode = Int(KeyCombo.controlSpace.keyCode)
        agentHotkeyModifiers = Int(KeyCombo.controlSpace.modifiers)
        ttsTemperature = 0.6
        ttsTopP = 0.8
        ttsRepetitionPenalty = 1.3
        ttsMaxTokens = 4096
        ttsSeed = 0
        ttsVoiceDescription = ""
        ttsLanguage = "English"
        ttsStreamingEnabled = true
        agentAutoSpeak = false
        selectedAgentModelID = ModelDefinition.defaultAgentModelID
    }

    // MARK: - Private

    private func updateLaunchAtLogin() {
        do {
            if launchAtLogin {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            Log.general.error("Failed to update launch at login: \(error)")
        }
    }

    func applyDockVisibility() {
        if showInDock {
            NSApp.setActivationPolicy(.regular)
        } else {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
