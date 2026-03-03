//
//  SettingsManager.swift
//  tesseract
//

import Foundation
import Combine
import SwiftUI
import ServiceManagement
import os

@MainActor
final class SettingsManager: ObservableObject {
    static let shared = SettingsManager()

    // MARK: - General Settings

    @AppStorage("launchAtLogin") var launchAtLogin = false {
        didSet { updateLaunchAtLogin() }
    }

    @AppStorage("showInDock") var showInDock = true {
        didSet { updateDockVisibility() }
    }

    @AppStorage("showInMenuBar") var showInMenuBar = true

    @AppStorage("autoInsertText") var autoInsertText = true

    @AppStorage("restoreClipboard") var restoreClipboard = true

    @AppStorage("overlayStyle") var overlayStyleRaw: String = OverlayStyle.pill.rawValue

    var overlayStyle: OverlayStyle {
        get { OverlayStyle(rawValue: overlayStyleRaw) ?? .pill }
        set { overlayStyleRaw = newValue.rawValue }
    }

    @AppStorage("glowTheme") var glowThemeRaw: String = GlowTheme.appleIntelligence.rawValue

    var glowTheme: GlowTheme {
        get { GlowTheme(rawValue: glowThemeRaw) ?? .appleIntelligence }
        set { glowThemeRaw = newValue.rawValue }
    }

    // MARK: - Audio Settings

    @AppStorage("selectedMicrophoneUID") var selectedMicrophoneUID: String = ""

    // MARK: - Language Settings

    @AppStorage("language") var language: String = "en"

    var selectedLanguage: SupportedLanguage {
        SupportedLanguage.language(forCode: language) ?? .auto
    }

    // MARK: - Hotkey Settings

    @AppStorage("hotkeyKeyCode") var hotkeyKeyCode: Int = Int(KeyCombo.optionSpace.keyCode)  // Option + Space

    @AppStorage("hotkeyModifiers") var hotkeyModifiers: Int = Int(KeyCombo.optionSpace.modifiers)

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

    // MARK: - TTS Settings

    @AppStorage("ttsHotkeyKeyCode") var ttsHotkeyKeyCode: Int = Int(KeyCombo.functionSpace.keyCode)
    @AppStorage("ttsHotkeyModifiers") var ttsHotkeyModifiers: Int = Int(KeyCombo.functionSpace.modifiers)

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

    @AppStorage("agentHotkeyKeyCode") var agentHotkeyKeyCode: Int = Int(KeyCombo.controlSpace.keyCode)
    @AppStorage("agentHotkeyModifiers") var agentHotkeyModifiers: Int = Int(KeyCombo.controlSpace.modifiers)

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

    @AppStorage("ttsTemperature") var ttsTemperature: Double = 0.6
    @AppStorage("ttsTopP") var ttsTopP: Double = 0.8
    @AppStorage("ttsRepetitionPenalty") var ttsRepetitionPenalty: Double = 1.3
    @AppStorage("ttsMaxTokens") var ttsMaxTokens: Int = 4096
    @AppStorage("ttsSeed") var ttsSeed: Int = 0
    @AppStorage("ttsVoiceDescription") var ttsVoiceDescription: String = ""
    @AppStorage("ttsLanguage") var ttsLanguage: String = "English"
    @AppStorage("ttsStreamingEnabled") var ttsStreamingEnabled = true
    @AppStorage("agentAutoSpeak") var agentAutoSpeak = false
    @AppStorage("selectedAgentModelID") var selectedAgentModelID: String = ModelDefinition.defaultAgentModelID

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

    @AppStorage("maxRecordingDuration") var maxRecordingDuration: Double = 300.0

    @AppStorage("playSounds") var playSounds = true

    @AppStorage("showNotifications") var showNotifications = true

    // MARK: - Onboarding

    @AppStorage("hasCompletedOnboarding") var hasCompletedOnboarding = false

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

    private func updateDockVisibility() {
        if showInDock {
            NSApp.setActivationPolicy(.regular)
        } else {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
