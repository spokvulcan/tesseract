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

    @AppStorage("ttsTemperature") var ttsTemperature: Double = 0.6
    @AppStorage("ttsTopP") var ttsTopP: Double = 0.8
    @AppStorage("ttsRepetitionPenalty") var ttsRepetitionPenalty: Double = 1.3
    @AppStorage("ttsRepetitionContextSize") var ttsRepetitionContextSize: Int = 20
    @AppStorage("ttsMaxTokens") var ttsMaxTokens: Int = 4096
    @AppStorage("ttsVoiceDescription") var ttsVoiceDescription: String = ""
    @AppStorage("ttsLanguage") var ttsLanguage: String = "English"

    var ttsParameters: TTSParameters {
        get {
            TTSParameters(
                temperature: Float(ttsTemperature),
                topP: Float(ttsTopP),
                repetitionPenalty: Float(ttsRepetitionPenalty),
                repetitionContextSize: ttsRepetitionContextSize,
                maxTokens: ttsMaxTokens
            )
        }
        set {
            ttsTemperature = Double(newValue.temperature)
            ttsTopP = Double(newValue.topP)
            ttsRepetitionPenalty = Double(newValue.repetitionPenalty)
            ttsRepetitionContextSize = newValue.repetitionContextSize
            ttsMaxTokens = newValue.maxTokens
        }
    }

    // MARK: - Advanced Settings

    @AppStorage("maxRecordingDuration") var maxRecordingDuration: Double = 60.0

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
        maxRecordingDuration = 60.0
        playSounds = true
        showNotifications = true
        ttsHotkeyKeyCode = Int(KeyCombo.functionSpace.keyCode)
        ttsHotkeyModifiers = Int(KeyCombo.functionSpace.modifiers)
        ttsTemperature = 0.6
        ttsTopP = 0.8
        ttsRepetitionPenalty = 1.3
        ttsRepetitionContextSize = 20
        ttsMaxTokens = 4096
        ttsVoiceDescription = ""
        ttsLanguage = "English"
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
