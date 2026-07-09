//
//  SettingsWindowPrototype.swift
//  tesseract
//
//  PROTOTYPE — Wayfinder ticket #215 (map #211). Throwaway code: the rough
//  native Settings scene the owner reacts to. Panes follow the ratified IA
//  (#213: General · Hotkeys · Agent · Dictation · Server) and the design
//  language (docs/design/design-language.md). Real knobs are wired straight
//  to SettingsManager (same bindings as the current sidebar pages); the only
//  stub is "Manage Models…", which opens the main window without navigating.
//  The accepted shape becomes the spec for the cutover ticket #216 — this
//  file is then deleted and rebuilt properly.
//

import SwiftUI

// MARK: - Root: TabView toolbar-of-panes (design language rule 7)

struct SettingsWindowPrototypeView: View {
    var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") {
                GeneralSettingsPane()
                    .frame(width: 620, height: 500)
            }
            Tab("Hotkeys", systemImage: "keyboard") {
                HotkeysSettingsPane()
                    .frame(width: 620, height: 560)
            }
            Tab("Agent", systemImage: "brain.head.profile") {
                AgentSettingsPane()
                    .frame(width: 620, height: 620)
            }
            Tab("Dictation", systemImage: "mic.fill") {
                DictationSettingsPane()
                    .frame(width: 620, height: 660)
            }
            Tab("Server", systemImage: "server.rack") {
                // The ratified Server pane is exactly the existing
                // configuration form (longest pane; scrolls).
                ServerConfigurationView()
                    .frame(width: 620, height: 700)
            }
        }
    }
}

// MARK: - General

private struct GeneralSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.openWindow) private var openWindow

    // Prevent disabling both dock and menu bar (same guard as the current page).
    private var showInDockBinding: Binding<Bool> {
        Binding(
            get: { settings.showInDock },
            set: { newValue in
                if !newValue && !settings.showInMenuBar { return }
                settings.showInDock = newValue
            }
        )
    }

    private var showInMenuBarBinding: Binding<Bool> {
        Binding(
            get: { settings.showInMenuBar },
            set: { newValue in
                if !newValue && !settings.showInDock { return }
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
            } footer: {
                if !settings.showInDock || !settings.showInMenuBar {
                    Text("At least one must stay on so the app remains reachable.")
                }
            }

            Section("Feedback") {
                Toggle("Play Sounds", isOn: $settings.playSounds)
            }

            Section {
                Picker("Overlay Style", selection: $settings.overlayStyle) {
                    ForEach(OverlayStyle.allCases) { style in
                        Text(style.displayName).tag(style)
                    }
                }
                if settings.overlayStyle == .fullScreenBorder {
                    Picker("Glow Theme", selection: $settings.glowTheme) {
                        ForEach(GlowTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme)
                        }
                    }
                }
            } header: {
                Text("Recording Overlay")
            } footer: {
                Text(settings.overlayStyle.description)
            }

            Section("Setup") {
                Button("Show Welcome Tour…") {
                    openWindow(id: WindowID.onboarding)
                }
            }

            Section {
                Button("Reset to Defaults", role: .destructive) {
                    settings.resetToDefaults()
                }
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - Hotkeys

private struct HotkeysSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section("Global Hotkeys") {
                HotkeyRecorderRow(label: "Dictation Push-to-Talk", combo: $settings.hotkey)
                HotkeyRecorderRow(label: "Speak Selected Text", combo: $settings.ttsHotkey)
                HotkeyRecorderRow(label: "Talk to Tesse", combo: $settings.agentHotkey)
            }

            Section {
                HotkeyRecorderRow(
                    label: "Capture Frontmost Window",
                    combo: $settings.appshotHotkey,
                    resetTo: .doubleCommand,
                    resetLabel: "Reset to ⌘⌘"
                )
            } header: {
                Text("Appshot")
            } footer: {
                Text(
                    "Press both Command keys in any app to attach a shot of its frontmost window to the agent composer."
                )
            }

            Section {
                HStack {
                    Text("Screen Recording")
                    Spacer()
                    if container.permissionsManager.screenRecordingPermission == .granted {
                        Label("Granted", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .labelStyle(.titleAndIcon)
                    } else {
                        Text("Not Granted")
                            .foregroundStyle(.secondary)
                        Button("Grant…") {
                            container.permissionsManager.requestScreenRecordingPermission()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            } header: {
                Text("Permissions")
            } footer: {
                if container.permissionsManager.screenRecordingPermission != .granted {
                    Text(
                        "Appshots need Screen Recording. macOS applies the permission after Tesseract is relaunched."
                    )
                }
            }
        }
        .formStyle(.grouped)
    }
}

/// One hotkey recorder line: current combo, Change/Cancel, optional reset.
private struct HotkeyRecorderRow: View {
    @EnvironmentObject private var container: DependencyContainer
    let label: String
    @Binding var combo: KeyCombo
    var resetTo: KeyCombo?
    var resetLabel: String = "Reset"
    @State private var isRecording = false

    var body: some View {
        HStack {
            Text(label)
            Spacer()

            if isRecording {
                Text("Press a key…")
                    .foregroundStyle(.secondary)
            } else {
                Text(combo.displayString)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.2))
                    .cornerRadius(4)
            }

            Button(isRecording ? "Cancel" : "Change") {
                toggleRecording()
            }
            .buttonStyle(.bordered)

            if let resetTo, combo != resetTo {
                Button(resetLabel) {
                    combo = resetTo
                }
                .buttonStyle(.bordered)
            }
        }

        if combo.modifierFlags.contains(.function) {
            Text(
                "fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func toggleRecording() {
        if isRecording {
            isRecording = false
            return
        }
        isRecording = true
        Task {
            if let newCombo = await container.hotkeyManager.recordHotkey() {
                combo = newCombo
            }
            isRecording = false
        }
    }
}

// MARK: - Agent

private struct AgentSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer
    @Environment(\.openWindow) private var openWindow
    @State private var selectedAgentModelDeclaresPreserveThinking = false

    private var selectedAgentModelStatus: ModelStatus {
        container.modelDownloadManager.status(for: settings.selectedAgentModelID)
    }

    private var translateLanguageOptions: [String] {
        var languages = SupportedLanguage.all
            .filter { $0.code != SupportedLanguage.auto.code }
            .map(\.name)
        if !languages.contains(settings.translateTargetLanguage) {
            languages.append(settings.translateTargetLanguage)
        }
        return languages.sorted()
    }

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section {
                let agentModels = ModelDefinition.models(in: .agent)
                let downloadedAgentModels = container.modelDownloadManager.downloadedModels(
                    in: .agent)

                if downloadedAgentModels.isEmpty {
                    Text("No agent models downloaded.")
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $settings.selectedAgentModelID) {
                        ForEach(downloadedAgentModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
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
                    }
                }

                // Stub: the IA links model pickers to the main-window Models
                // page; the prototype only opens the main window.
                Button("Manage Models…") {
                    openWindow(id: WindowID.main)
                }

                if let selected = agentModels.first(where: {
                    $0.id == settings.selectedAgentModelID
                }) {
                    Text(selected.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Model")
            } footer: {
                if selectedAgentModelDeclaresPreserveThinking {
                    Text(
                        "Preserve Thinking keeps each turn's thinking in the prompt so follow-up requests reuse the cache instead of re-reading the conversation. Uses more context window. Applies to new conversations."
                    )
                }
            }

            Section {
                Picker("Sampling Preset", selection: $settings.samplingPreset) {
                    ForEach(SamplingPreset.allCases) { preset in
                        Text(preset.displayName).tag(preset)
                    }
                }
            } footer: {
                Text(settings.samplingPreset.description)
            }

            Section {
                Toggle("Web Access", isOn: $settings.webAccessEnabled)
            } footer: {
                Text(
                    "Lets the agent search, read, and browse the web through a local browser. Only your search queries and the pages the agent visits leave your device — no conversation data."
                )
            }

            Section {
                Toggle("Use Vision Models When Available", isOn: $settings.useVisionWhenAvailable)
            } footer: {
                Text(
                    "Loads a vision-capable model with its image tower resident (~1 GB) so you can attach images in chat. Turn off to load the faster text-only container instead."
                )
            }

            Section {
                Toggle("Show Skill Button", isOn: $settings.showSkillPills)
                Picker("Translate To", selection: $settings.translateTargetLanguage) {
                    ForEach(translateLanguageOptions, id: \.self) { language in
                        Text(language).tag(language)
                    }
                }
            } header: {
                Text("Skills")
            } footer: {
                Text(
                    "The floating ✦ button above the composer fans out your skills on hover. Translate To sets the Translate skill's default target; naming a language in your message always wins."
                )
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
        let modelID = settings.selectedAgentModelID
        Task {
            let declares = await ModelIdentity.declares(
                .preserveThinking, atDirectory: directory
            )
            guard settings.selectedAgentModelID == modelID else { return }
            selectedAgentModelDeclaresPreserveThinking = declares
        }
    }
}

// MARK: - Dictation

private struct DictationSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section {
                Picker("Input Device", selection: $settings.selectedMicrophoneUID) {
                    Text("System Default").tag("")
                    ForEach(container.audioDeviceManager.availableDevices) { device in
                        Text(device.name).tag(device.uid)
                    }
                }

                AudioLevelMeter(audioCapture: container.audioCaptureEngine)
                    .frame(height: 20)
            } header: {
                Text("Microphone")
            } footer: {
                Text(
                    "Capture uses Apple's echo cancellation, noise suppression, and automatic gain. Other audio dips briefly while recording."
                )
            }

            Section {
                let downloadedSpeechModels =
                    container.modelDownloadManager.downloadedModels(in: .speechToText)

                if downloadedSpeechModels.isEmpty {
                    Text("No dictation models downloaded.")
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: $settings.selectedSpeechToTextModelID) {
                        ForEach(downloadedSpeechModels) { model in
                            Text(model.displayName).tag(model.id)
                        }
                    }
                }

                // Stub: opens the main window; the Models page lives there.
                Button("Manage Models…") {
                    openWindow(id: WindowID.main)
                }
            } header: {
                Text("Model")
            } footer: {
                if let selected = ModelDefinition.withID(settings.selectedSpeechToTextModelID) {
                    Text(selected.description)
                }
            }

            Section {
                LanguagePickerView(selectedLanguage: $settings.language)
            }

            Section("Duration") {
                VStack(alignment: .leading) {
                    let minutes = Int(settings.maxRecordingDuration) / 60
                    let seconds = Int(settings.maxRecordingDuration) % 60
                    if seconds == 0 {
                        Text("Maximum Duration: \(minutes) min")
                    } else {
                        Text("Maximum Duration: \(minutes) min \(seconds)s")
                    }
                    Slider(value: $settings.maxRecordingDuration, in: 30...1800, step: 30)
                }
            }

            Section {
                Toggle("Automatically Insert Text", isOn: $settings.autoInsertText)
                Toggle("Restore Clipboard Contents", isOn: $settings.restoreClipboard)
                    .disabled(!settings.autoInsertText)
            } header: {
                Text("After Transcription")
            } footer: {
                Text(
                    "Types the transcription into the frontmost app, then puts whatever was on the clipboard back."
                )
            }

            Section {
                Toggle("Keep Recent Recordings", isOn: $settings.captureDumpEnabled)
                HStack {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [container.captureDumpStore.directory])
                    }
                    Button("Delete All Recordings", role: .destructive) {
                        container.captureDumpStore.deleteAll()
                    }
                }
            } header: {
                Text("Recent Recordings")
            } footer: {
                Text(
                    "Keeps the most recent dictation recordings on disk (bounded, oldest deleted first) so a bad transcription can be diagnosed. Recordings never leave this Mac."
                )
            }
        }
        .formStyle(.grouped)
    }
}
