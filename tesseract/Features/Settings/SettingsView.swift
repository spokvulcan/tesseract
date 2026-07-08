//
//  SettingsView.swift
//  tesseract
//

import SwiftUI

// MARK: - General Settings Section

struct GeneralSettingsSection: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.openWindow) private var openWindow

    // Prevent disabling both dock and menu bar
    private var showInDockBinding: Binding<Bool> {
        Binding(
            get: { settings.showInDock },
            set: { newValue in
                // Only allow turning off dock if menu bar is enabled
                if !newValue && !settings.showInMenuBar {
                    return  // Don't allow - would make app inaccessible
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
                    return  // Don't allow - would make app inaccessible
                }
                settings.showInMenuBar = newValue
            }
        )
    }

    var body: some View {
        @Bindable var settings = settings
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
            }

            Section("Setup") {
                Button("Show Welcome Tour\u{2026}") {
                    openWindow(id: WindowID.onboarding)
                }
            }

            Section("Recording Overlay") {
                Picker("Overlay Style", selection: $settings.overlayStyle) {
                    ForEach(OverlayStyle.allCases) { style in
                        Text(style.displayName).tag(style)
                    }
                }

                Text(settings.overlayStyle.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                // Show glow theme picker for full-screen border style
                if settings.overlayStyle == .fullScreenBorder {
                    Picker("Glow Theme", selection: $settings.glowTheme) {
                        ForEach(GlowTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)

        .navigationTitle("General")
    }
}

// MARK: - Audio Level Meter

struct AudioLevelMeter: View {
    var audioCapture: AudioCaptureEngine
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
            .accessibilityHint(
                isTestingMic
                    ? "Stops the microphone level test" : "Starts monitoring microphone input level"
            )
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
                // Metering-only: the meter wants the level, not a recording —
                // a long test must not accumulate audio or resample it on stop.
                try audioCapture.startLevelMetering()
                isTestingMic = true
            } catch {
                Log.general.error("Failed to start mic test: \(error)")
            }
        }
    }
}

// MARK: - Recording Settings Section

struct RecordingSettingsSection: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer
    @State private var isRecordingHotkey = false
    @State private var isRecordingTTSHotkey = false
    @State private var isRecordingAgentHotkey = false
    @State private var isRecordingAppshotHotkey = false
    @State private var selectedAgentModelDeclaresPreserveThinking = false

    private var selectedAgentModelStatus: ModelStatus {
        container.modelDownloadManager.status(for: settings.selectedAgentModelID)
    }

    /// Language options for the Translate skill's target picker: the canonical
    /// ``SupportedLanguage`` catalogue (minus the dictation-only "Auto-detect"
    /// pseudo-entry), with the current selection (the launch-time derivation
    /// from the macOS preferred languages, or a past choice) always present so
    /// the Picker never shows an unselectable value.
    private var translateLanguageOptions: [String] {
        var languages = SupportedLanguage.all
            .filter { $0.code != SupportedLanguage.auto.code }
            .map(\.name)
        if !languages.contains(settings.translateTargetLanguage) {
            languages.append(settings.translateTargetLanguage)
        }
        return languages.sorted()
    }

    private func refreshSelectedAgentModelCapabilities() {
        guard case .downloaded = selectedAgentModelStatus,
            let directory = container.modelDownloadManager.modelPath(
                for: settings.selectedAgentModelID
            )
        else {
            selectedAgentModelDeclaresPreserveThinking = false
            return
        }
        // `ModelIdentity.declares` runs the disk-reading probe off the MainActor
        // (ADR-0001) so opening or switching the settings pane can't stutter.
        // Publish back only while the same model is still selected, so a slow
        // read for a since-deselected model can't clobber a newer answer.
        let modelID = settings.selectedAgentModelID
        Task {
            let declares = await ModelIdentity.declares(
                .preserveThinking, atDirectory: directory
            )
            guard settings.selectedAgentModelID == modelID else { return }
            selectedAgentModelDeclaresPreserveThinking = declares
        }
    }

    var body: some View {
        @Bindable var settings = settings
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

                Text(
                    "Capture uses Apple's echo cancellation, noise suppression, and automatic gain. Other audio dips briefly while recording."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Recent Recordings") {
                Toggle("Keep Recent Recordings", isOn: $settings.captureDumpEnabled)
                Text(
                    "Keeps the most recent dictation recordings on disk (bounded, oldest deleted first) so a bad transcription can be diagnosed. Recordings never leave this Mac."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

                HStack {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [container.captureDumpStore.directory])
                    }
                    Button("Delete All Recordings", role: .destructive) {
                        container.captureDumpStore.deleteAll()
                    }
                }
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
                    .accessibilityHint(
                        isRecordingHotkey
                            ? "Cancel recording new hotkey" : "Record a new push-to-talk hotkey")
                }

                if settings.hotkey.modifierFlags.contains(.function) {
                    Text(
                        "Note: fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧."
                    )
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
                            .accessibilityLabel(
                                "Current TTS hotkey: \(settings.ttsHotkey.displayString)")
                    }

                    Button(isRecordingTTSHotkey ? "Cancel" : "Change") {
                        if isRecordingTTSHotkey {
                            isRecordingTTSHotkey = false
                        } else {
                            recordNewTTSHotkey()
                        }
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint(
                        isRecordingTTSHotkey
                            ? "Cancel recording new hotkey" : "Record a new speech hotkey")
                }

                if settings.ttsHotkey.modifierFlags.contains(.function) {
                    Text(
                        "Note: fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Agent Hotkey") {
                HStack {
                    Text("Talk to Tesse:")
                    Spacer()

                    if isRecordingAgentHotkey {
                        Text("Press a key...")
                            .foregroundStyle(.secondary)
                            .accessibilityLabel("Waiting for key press")
                    } else {
                        Text(settings.agentHotkey.displayString)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                            .accessibilityLabel(
                                "Current agent hotkey: \(settings.agentHotkey.displayString)")
                    }

                    Button(isRecordingAgentHotkey ? "Cancel" : "Change") {
                        if isRecordingAgentHotkey {
                            isRecordingAgentHotkey = false
                        } else {
                            recordNewAgentHotkey()
                        }
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint(
                        isRecordingAgentHotkey
                            ? "Cancel recording new hotkey" : "Record a new agent hotkey")
                }

                if settings.agentHotkey.modifierFlags.contains(.function) {
                    Text(
                        "Note: fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Appshot Hotkey") {
                HStack {
                    Text("Capture Frontmost Window:")
                    Spacer()

                    if isRecordingAppshotHotkey {
                        Text("Press a key...")
                            .foregroundStyle(.secondary)
                            .accessibilityLabel("Waiting for key press")
                    } else {
                        Text(settings.appshotHotkey.displayString)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                            .accessibilityLabel(
                                "Current appshot hotkey: \(settings.appshotHotkey.displayString)")
                    }

                    Button(isRecordingAppshotHotkey ? "Cancel" : "Change") {
                        if isRecordingAppshotHotkey {
                            isRecordingAppshotHotkey = false
                        } else {
                            recordNewAppshotHotkey()
                        }
                    }
                    .buttonStyle(.bordered)
                    .accessibilityHint(
                        isRecordingAppshotHotkey
                            ? "Cancel recording new hotkey" : "Record a new appshot hotkey")

                    if !settings.appshotHotkey.isDoubleCommand {
                        Button("Reset to ⌘⌘") {
                            settings.appshotHotkey = .doubleCommand
                        }
                        .buttonStyle(.bordered)
                        .help("Restore the double-Command default")
                    }
                }

                Text(
                    "Press both Command keys in any app to attach a shot of its frontmost window to the agent composer."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

                HStack {
                    Text("Screen Recording Permission:")
                    Spacer()
                    if container.permissionsManager.screenRecordingPermission == .granted {
                        Label("Granted", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .labelStyle(.titleAndIcon)
                    } else {
                        Text("Not granted")
                            .foregroundStyle(.secondary)
                        Button("Grant…") {
                            container.permissionsManager.requestScreenRecordingPermission()
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if container.permissionsManager.screenRecordingPermission != .granted {
                    Text(
                        "Appshots need Screen Recording. macOS applies the permission after Tesseract is relaunched."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Agent Model") {
                let agentModels = ModelDefinition.models(in: .agent)
                let downloadedAgentModels = container.modelDownloadManager.downloadedModels(
                    in: .agent)

                if downloadedAgentModels.isEmpty {
                    Text("No agent models downloaded. Download one from the Models page.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $settings.selectedAgentModelID) {
                        ForEach(downloadedAgentModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
                    }

                    if let selected = agentModels.first(where: {
                        $0.id == settings.selectedAgentModelID
                    }) {
                        Text(selected.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if selectedAgentModelDeclaresPreserveThinking {
                        Toggle(
                            "Preserve Thinking in Prompts",
                            isOn: Binding(
                                get: {
                                    settings.preserveThinkingRender(
                                        modelID: settings.selectedAgentModelID
                                    )
                                },
                                set: {
                                    settings.setPreserveThinkingRender(
                                        $0, modelID: settings.selectedAgentModelID
                                    )
                                }
                            ))
                        Text(
                            "Keeps each turn's thinking in the prompt so follow-up requests reuse the cache instead of re-reading the conversation. Uses more context window. Applies to new conversations."
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            Section("Sampling Preset") {
                Picker("Preset", selection: $settings.samplingPreset) {
                    ForEach(SamplingPreset.allCases) { preset in
                        Text(preset.displayName).tag(preset)
                    }
                }
                Text(settings.samplingPreset.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section("Web Access") {
                Toggle("Enable web access", isOn: $settings.webAccessEnabled)
                Text(
                    "Lets the agent search, read, and browse the web through a local browser. Only your search queries and the pages the agent visits leave your device — no conversation data."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Vision") {
                Toggle("Use vision models when available", isOn: $settings.useVisionWhenAvailable)
                Text(
                    "When on, a vision-capable model loads its image-aware container so you can attach images in chat. Prefill speed is unchanged — vision only keeps a small vision tower resident (~1 GB). Turn off to load the faster, text-only container instead; a model already loaded with vision keeps it until the next reload."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Skills") {
                Toggle("Show skill button", isOn: $settings.showSkillPills)
                Text(
                    "The floating ✦ button above the chat composer — hover to fan out your skills, tap one to run it on your text and attachments instantly. Every skill also stays available as a slash command."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

                Picker("Translate to", selection: $settings.translateTargetLanguage) {
                    ForEach(translateLanguageOptions, id: \.self) { language in
                        Text(language).tag(language)
                    }
                }
                Text(
                    "The Translate skill's default target. Text already in this language translates to English instead; naming a language in your message always wins."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Dictation Model") {
                let downloadedSpeechModels =
                    container.modelDownloadManager.downloadedModels(in: .speechToText)

                if downloadedSpeechModels.isEmpty {
                    Text("No dictation models downloaded. Download one from the Models page.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $settings.selectedSpeechToTextModelID) {
                        ForEach(downloadedSpeechModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
                    }

                    if let selected = ModelDefinition.withID(settings.selectedSpeechToTextModelID) {
                        Text(selected.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Duration") {
                VStack(alignment: .leading) {
                    let minutes = Int(settings.maxRecordingDuration) / 60
                    let seconds = Int(settings.maxRecordingDuration) % 60
                    if seconds == 0 {
                        Text("Maximum duration: \(minutes) min")
                    } else {
                        Text("Maximum duration: \(minutes) min \(seconds)s")
                    }
                    Slider(value: $settings.maxRecordingDuration, in: 30...1800, step: 30)
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
        .onAppear {
            refreshSelectedAgentModelCapabilities()
        }
        .onChange(of: settings.selectedAgentModelID) {
            refreshSelectedAgentModelCapabilities()
        }
        .onChange(of: selectedAgentModelStatus) {
            refreshSelectedAgentModelCapabilities()
        }

        .navigationTitle("Preferences")
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

    private func recordNewAppshotHotkey() {
        isRecordingAppshotHotkey = true

        Task {
            if let combo = await container.hotkeyManager.recordHotkey() {
                settings.appshotHotkey = combo
            }
            isRecordingAppshotHotkey = false
        }
    }

    private func recordNewAgentHotkey() {
        isRecordingAgentHotkey = true

        Task {
            if let combo = await container.hotkeyManager.recordHotkey() {
                settings.agentHotkey = combo
            }
            isRecordingAgentHotkey = false
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
