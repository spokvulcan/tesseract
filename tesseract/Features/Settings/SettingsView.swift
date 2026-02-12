//
//  SettingsView.swift
//  tesseract
//

import SwiftUI

// MARK: - General Settings Section

struct GeneralSettingsSection: View {
    @ObservedObject private var settings = SettingsManager.shared

    // Prevent disabling both dock and menu bar
    private var showInDockBinding: Binding<Bool> {
        Binding(
            get: { settings.showInDock },
            set: { newValue in
                // Only allow turning off dock if menu bar is enabled
                if !newValue && !settings.showInMenuBar {
                    return // Don't allow - would make app inaccessible
                }
                settings.showInDock = newValue
            }
        )
    }

    private var showInMenuBarBinding: Binding<Bool> {
        Binding(
            get: { settings.showInMenuBar },
            set: { newValue in
                // Only allow turning off menu bar if dock is enabled
                if !newValue && !settings.showInDock {
                    return // Don't allow - would make app inaccessible
                }
                settings.showInMenuBar = newValue
            }
        )
    }

    var body: some View {
        Form {
            Section {
                Toggle("Launch at Login", isOn: $settings.launchAtLogin)
                Toggle("Show in Dock", isOn: showInDockBinding)
                Toggle("Show in Menu Bar", isOn: showInMenuBarBinding)

                if !settings.showInDock || !settings.showInMenuBar {
                    Text("At least one must be enabled to access the app")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("After Transcription") {
                Toggle("Automatically insert text", isOn: $settings.autoInsertText)
                Toggle("Restore clipboard contents", isOn: $settings.restoreClipboard)
                    .disabled(!settings.autoInsertText)
            }

            Section("Feedback") {
                Toggle("Play sounds", isOn: $settings.playSounds)
                Toggle("Show notifications", isOn: $settings.showNotifications)
            }

            Section("Setup") {
                Button("Run Setup Wizard...") {
                    NotificationCenter.default.post(name: .showOnboarding, object: nil)
                }
            }

            Section("Recording Overlay") {
                Picker("Overlay Style", selection: $settings.overlayStyleRaw) {
                    ForEach(OverlayStyle.allCases) { style in
                        Text(style.displayName).tag(style.rawValue)
                    }
                }

                if let style = OverlayStyle(rawValue: settings.overlayStyleRaw) {
                    Text(style.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Show glow theme picker for full-screen border style
                if settings.overlayStyle == .fullScreenBorder {
                    Picker("Glow Theme", selection: $settings.glowThemeRaw) {
                        ForEach(GlowTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme.rawValue)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .navigationTitle("General")
    }
}

// MARK: - Audio Level Meter

struct AudioLevelMeter: View {
    @ObservedObject var audioCapture: AudioCaptureEngine
    @State private var isTestingMic = false

    var body: some View {
        HStack {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.secondary.opacity(0.2))

                    RoundedRectangle(cornerRadius: 4)
                        .fill(levelColor)
                        .frame(width: geometry.size.width * CGFloat(audioCapture.audioLevel))
                        .animation(.linear(duration: 0.1), value: audioCapture.audioLevel)
                }
            }
            .accessibilityElement()
            .accessibilityLabel("Audio level meter")
            .accessibilityValue("\(Int(audioCapture.audioLevel * 100)) percent")

            Button(isTestingMic ? "Stop" : "Test") {
                toggleMicTest()
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(isTestingMic ? "Stop microphone test" : "Test microphone")
            .accessibilityHint(isTestingMic ? "Stops the microphone level test" : "Starts monitoring microphone input level")
        }
    }

    private var levelColor: Color {
        let level = audioCapture.audioLevel
        if level > 0.8 {
            return .red
        } else if level > 0.5 {
            return .yellow
        } else {
            return .green
        }
    }

    private func toggleMicTest() {
        if isTestingMic {
            _ = audioCapture.stopCapture()
            isTestingMic = false
        } else {
            do {
                try audioCapture.startCapture()
                isTestingMic = true
            } catch {
                print("Failed to start mic test: \(error)")
            }
        }
    }
}

// MARK: - Model Settings Section

struct ModelSettingsSection: View {
    @EnvironmentObject private var container: DependencyContainer

    var body: some View {
        Form {
            Section("Transcription Model") {
                HStack {
                    Text("Model")
                    Spacer()
                    Text(WhisperModel.displayName)
                        .foregroundStyle(.secondary)
                }

                Text(WhisperModel.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Size: \(String(format: "%.1f", WhisperModel.sizeGB)) GB")
                    Spacer()
                    Text("RAM: \(WhisperModel.recommendedRAMGB) GB+")
                    Spacer()
                    Label("\(WhisperModel.languageCount) languages", systemImage: "globe")
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if container.transcriptionEngine.isModelLoaded {
                    Label("Model loaded and ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    Label("Model not loaded", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .navigationTitle("Model")
    }
}

// MARK: - Recording Settings Section

struct RecordingSettingsSection: View {
    @ObservedObject private var settings = SettingsManager.shared
    @EnvironmentObject private var container: DependencyContainer
    @State private var isRecordingHotkey = false
    @State private var isRecordingTTSHotkey = false

    var body: some View {
        Form {
            Section("Microphone") {
                Picker("Input Device", selection: $settings.selectedMicrophoneUID) {
                    Text("System Default").tag("")
                    ForEach(container.audioDeviceManager.availableDevices) { device in
                        Text(device.name).tag(device.uid)
                    }
                }

                AudioLevelMeter(audioCapture: container.audioCaptureEngine)
                    .frame(height: 20)
            }

            Section("Dictation Hotkey") {
                HStack {
                    Text("Push-to-Talk Key:")
                    Spacer()

                    if isRecordingHotkey {
                        Text("Press a key...")
                            .foregroundStyle(.secondary)
                            .accessibilityLabel("Waiting for key press")
                    } else {
                        Text(settings.hotkey.displayString)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                            .accessibilityLabel("Current hotkey: \(settings.hotkey.displayString)")
                    }

                    Button(isRecordingHotkey ? "Cancel" : "Change") {
                        if isRecordingHotkey {
                            isRecordingHotkey = false
                        } else {
                            recordNewHotkey()
                        }
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint(isRecordingHotkey ? "Cancel recording new hotkey" : "Record a new push-to-talk hotkey")
                }

                if settings.hotkey.modifierFlags.contains(.function) {
                    Text("Note: fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Speech Hotkey") {
                HStack {
                    Text("Speak Selected Text:")
                    Spacer()

                    if isRecordingTTSHotkey {
                        Text("Press a key...")
                            .foregroundStyle(.secondary)
                            .accessibilityLabel("Waiting for key press")
                    } else {
                        Text(settings.ttsHotkey.displayString)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                            .accessibilityLabel("Current TTS hotkey: \(settings.ttsHotkey.displayString)")
                    }

                    Button(isRecordingTTSHotkey ? "Cancel" : "Change") {
                        if isRecordingTTSHotkey {
                            isRecordingTTSHotkey = false
                        } else {
                            recordNewTTSHotkey()
                        }
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint(isRecordingTTSHotkey ? "Cancel recording new hotkey" : "Record a new speech hotkey")
                }

                if settings.ttsHotkey.modifierFlags.contains(.function) {
                    Text("Note: fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Duration") {
                VStack(alignment: .leading) {
                    Text("Maximum duration: \(Int(settings.maxRecordingDuration))s")
                    Slider(value: $settings.maxRecordingDuration, in: 10...120, step: 10)
                }
            }

            Section {
                LanguagePickerView(selectedLanguage: $settings.language)
            }

            Section {
                Button("Reset to Defaults") {
                    settings.resetToDefaults()
                }
                .foregroundStyle(.red)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .navigationTitle("Recording")
    }

    private func recordNewHotkey() {
        isRecordingHotkey = true

        Task {
            if let combo = await container.hotkeyManager.recordHotkey() {
                settings.hotkey = combo
            }
            isRecordingHotkey = false
        }
    }

    private func recordNewTTSHotkey() {
        isRecordingTTSHotkey = true

        Task {
            if let combo = await container.hotkeyManager.recordHotkey() {
                settings.ttsHotkey = combo
            }
            isRecordingTTSHotkey = false
        }
    }
}

#Preview("General") {
    GeneralSettingsSection()
}

#Preview("Recording") {
    RecordingSettingsSection()
        .environmentObject(DependencyContainer())
}

// MARK: - Notifications

extension Notification.Name {
    static let showOnboarding = Notification.Name("showOnboarding")
}
