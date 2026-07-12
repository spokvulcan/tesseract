//
//  AgentSettingsPane.swift
//  tesseract
//

import SwiftUI

/// The Agent pane (#213): model choice (with the per-model Preserve-Thinking
/// toggle for declaring templates), sampling, web access, vision, and skills.
/// "Manage Models…" jumps to the main-window Models page — the download
/// manager is a task surface, not a Settings pane (#213).
struct AgentSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer
    @State private var selectedAgentModelDeclaresPreserveThinking = false

    private var selectedAgentModelStatus: ModelStatus {
        container.modelDownloadManager.status(for: settings.selectedAgentModelID)
    }

    /// Language options for the Translate skill's target picker — the one
    /// canonical list, shared with the status-bar menu's Translate To
    /// submenu.
    private var translateLanguageOptions: [String] {
        SupportedLanguage.translateTargetOptions(current: settings.translateTargetLanguage)
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

            // PROTOTYPE — the Companion walking skeleton (map #301, #303).
            Section {
                Toggle("Companion Heartbeat", isOn: $settings.companionHeartbeatEnabled)
                Toggle("Speak Pings Aloud", isOn: $settings.companionHeartbeatSpeaks)
                    .disabled(!settings.companionHeartbeatEnabled)
                Button("Send Test Ping") {
                    container.companionHeartbeat.sendTestPing()
                }
                .disabled(!settings.companionHeartbeatEnabled)
            } header: {
                Text("Companion (Experimental)")
            } footer: {
                Text(
                    "Walking-skeleton prototype: three fixed daily pings (9:00, 13:30, 21:30) as notifications you can click through, reply to, or dismiss. Every ping and outcome is recorded to companion/heartbeat.jsonl in the agent's folder. If no ping appears, allow notifications in System Settings → Notifications → Tesseract."
                )
            }
        }
        .formStyle(.grouped)
        .onChange(of: settings.companionHeartbeatEnabled) { _, enabled in
            if enabled { container.companionHeartbeat.activate() }
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
