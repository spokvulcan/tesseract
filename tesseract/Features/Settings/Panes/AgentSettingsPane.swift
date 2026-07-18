//
//  AgentSettingsPane.swift
//  tesseract
//

import ServiceManagement
import SwiftUI

/// The Agent pane (#213): model choice (with the per-model Preserve-Thinking
/// toggle for declaring templates), sampling, web access, vision, and skills.
/// "Manage Models…" jumps to the main-window Models page — the download
/// manager is a task surface, not a Settings pane (#213).
struct AgentSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var container: DependencyContainer
    @State private var selectedAgentModelDeclaresPreserveThinking = false
    /// The one-time launch-at-login ask (ADR-0040 §3), raised on first enable.
    @State private var showingLaunchAtLoginAsk = false

    /// What sleep is doing right now, in the owner's words rather than the
    /// engine's.
    private var sleepStatus: String {
        switch container.memorySleep.phase {
        case .idle: "Working…"
        case .grading: "Judging what helped…"
        case .reexamining: "Re-reading what you disputed…"
        case .extracting: "Reading what you said…"
        case .reconciling: "Checking it against what I know…"
        case .sweeping: "Tidying…"
        }
    }

    private var selectedAgentModelStatus: ModelStatus {
        container.modelDownloadManager.status(for: settings.selectedAgentModelID)
    }

    /// Language options for the Translate skill's target picker — the one
    /// canonical list, shared with the status-bar menu's Translate To
    /// submenu.
    private var translateLanguageOptions: [String] {
        SupportedLanguage.translateTargetOptions(current: settings.translateTargetLanguage)
    }

    private var companionSection: some View {
        @Bindable var settings = settings
        return Section {
            Toggle("Companion", isOn: $settings.companionHeartbeatEnabled)
            Picker("Companion Model", selection: $settings.companionModelID) {
                ForEach(
                    container.modelDownloadManager.downloadedModels(in: .agent)
                ) { model in
                    Text(model.displayName).tag(model.id)
                }
            }
            .disabled(!settings.companionHeartbeatEnabled)
            // Never a silent login-item flip (ADR-0040 §3): the toggle
            // reads and writes the real SMAppService state.
            Toggle(
                "Launch at Login",
                isOn: Binding(
                    get: { SMAppService.mainApp.status == .enabled },
                    set: { wanted in
                        do {
                            if wanted {
                                try SMAppService.mainApp.register()
                            } else {
                                try SMAppService.mainApp.unregister()
                            }
                        } catch {
                            Log.companion.error(
                                "Launch-at-login change failed: \(error)")
                        }
                    }
                ))
            HStack {
                Button("Book Test Wake") {
                    container.companionLoop.bookTestWake()
                }
                .disabled(!settings.companionHeartbeatEnabled)
                Button("Edit Instructions…") {
                    openWindow(id: WindowID.companionInstructions)
                }
            }
        } header: {
            Text("Companion (Experimental)")
        } footer: {
            Text(
                "A mind that happens to live in your Mac. The Companion books his own day — morning planning, a midday pulse, an evening journal — and wakes for what he booked; every turn is a real conversation you can open in the chat list, and every delivery is recorded to his flight log. While the Companion is enabled his model also becomes the default agent model — one model, one mind, no swap cost between your chats and his turns; if it isn't downloaded he runs on the model selected above. If no notification appears, allow notifications in System Settings → Notifications → Tesseract."
            )
        }
    }

    private var modelSection: some View {
        @Bindable var settings = settings
        return Section {
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

            Button("Manage Models…") {
                (NSApp.delegate as? AppDelegate)?.navigateToModels()
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
    }

    private var companionVoiceSection: some View {
        @Bindable var settings = settings
        return Section {
            Picker("Voice Overlay Concept", selection: $settings.companionVoiceConceptRaw) {
                ForEach(CompanionVoiceConcepts.all) { concept in
                    Text(concept.displayName).tag(concept.id)
                }
            }
            Text(
                CompanionVoiceConcepts.concept(for: settings.companionVoiceConceptRaw).thesis
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            HStack {
                ForEach(CompanionVoiceScene.all) { scene in
                    Button(scene.title) {
                        container.companionVoicePrototype.play(scene)
                    }
                }
                Button("Stop") {
                    container.companionVoicePrototype.stopScene()
                }
            }
            Toggle("Summon Overlay for Beats", isOn: $settings.companionBeatsUseOverlay)
                .disabled(!settings.companionHeartbeatEnabled)
            // The voice session's taste ledger (#310) — tuned in wear.
            Toggle("Auto-Send Voice Turns", isOn: $settings.companionVoiceAutoSend)
            // The Native Audio Turn experiment (ADR-0042). Visible only when
            // the selected agent model is Audio-Capable (#358 story 9) —
            // impossible to trip over on a text-only model.
            if container.modelDownloadManager.isAudioCapable(settings.selectedAgentModelID) {
                Toggle(
                    "Native Audio Turns (Experimental)",
                    isOn: $settings.companionVoiceNativeAudio
                )
            }
            HStack {
                Text("End-of-Speech Silence")
                Slider(value: $settings.companionVoiceTrailingSilence, in: 1.0...3.0)
                Text(String(format: "%.1fs", settings.companionVoiceTrailingSilence))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            HStack {
                Text("Session Silence Timeout")
                Slider(value: $settings.companionVoiceSessionTimeout, in: 10...90, step: 5)
                Text(String(format: "%.0fs", settings.companionVoiceSessionTimeout))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            HStack {
                Text("Barge-In Sensitivity")
                Slider(value: $settings.companionVoiceBargeInLevel, in: 0.1...0.5)
                Text(String(format: "%.2f", settings.companionVoiceBargeInLevel))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        } header: {
            Text("Companion Voice")
        } footer: {
            Text(
                "Voice conversations ride the chat itself: the waveform button in the composer (or engaging a spoken summons) opens a session where the mic listens after each reply, silence sends your turn, and speaking over him stops him mid-word. The overlay concepts (ticket #328) are the session's face — pick one, preview with the scripted scenes. With Summon Overlay for Beats on, his spoken lines raise the picked concept as the summons surface; every reaction is recorded. Auto-Send off stages your words in the composer instead of sending. Native Audio Turns sends your voice to the model as audio — he hears you, not a transcript — when the selected agent model supports audio input (takes over 30 s fall back to transcription)."
            )
        }
    }

    var body: some View {
        @Bindable var settings = settings
        Form {
            modelSection

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

            // The living memory (ADR-0035, map #314). Dictation gets its own
            // switch because it is the one capture source whose text is usually
            // addressed to *other* apps — the owner's call was to capture it, and
            // a call like that is only really the owner's if he can take it back.
            Section {
                Toggle("Memory", isOn: $settings.memoryEnabled)
                Toggle("Remember Dictated Text", isOn: $settings.memoryCaptureDictation)
                    .disabled(!settings.memoryEnabled)
                Toggle("Consolidate While Idle", isOn: $settings.memorySleepEnabled)
                    .disabled(!settings.memoryEnabled)
                HStack {
                    Button("Open Memory…") { openWindow(id: WindowID.memory) }
                    Button("Consolidate Now") { container.memorySleep.start() }
                        .disabled(
                            !settings.memoryEnabled || !settings.memorySleepEnabled
                                || container.memorySleep.isRunning)
                    if container.memorySleep.isRunning {
                        ProgressView().controlSize(.small)
                        Text(sleepStatus)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("Memory")
            } footer: {
                Text(
                    "What you say is stored verbatim as it happens, and distilled into memories while the Mac is idle — nothing leaves this machine. Consolidation yields the moment you touch the keyboard. Open Memory to see what I believe about you, why, and to contest or delete any of it."
                )
            }

            // The Companion (ADR-0040): the entity's master switch, his model,
            // and the one test lever that exercises the whole pipe. Extracted
            // — the one Form body was past the type-checker's budget.
            companionSection

            // PROTOTYPE — the Companion voice-overlay concepts (map #301, #328).
            companionVoiceSection
        }
        .formStyle(.grouped)
        .onChange(of: settings.companionHeartbeatEnabled) { _, enabled in
            if enabled {
                container.companionLoop.activate()
                if !settings.companionLaunchAtLoginAsked {
                    settings.companionLaunchAtLoginAsked = true
                    showingLaunchAtLoginAsk = true
                }
            }
        }
        .alert("Keep Jarvis running?", isPresented: $showingLaunchAtLoginAsk) {
            Button("Launch at Login") {
                do { try SMAppService.mainApp.register() } catch {
                    Log.companion.error("Launch-at-login register failed: \(error)")
                }
            }
            Button("Not Now", role: .cancel) {}
        } message: {
            Text(
                "The Companion only runs while Tesseract is open. Start it at login so his day survives reboots — you can change this anytime with the Launch at Login toggle."
            )
        }
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

    /// `ModelIdentity.declares` runs the disk-reading probe off the MainActor
    /// (ADR-0001) so opening or switching the pane can't stutter. Publish back
    /// only while the same model is still selected, so a slow read for a
    /// since-deselected model can't clobber a newer answer.
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
