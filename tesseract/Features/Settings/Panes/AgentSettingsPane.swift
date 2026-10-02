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
    @State private var selectedAgentModelDeclaresReasoningEffort = false

    private var selectedAgentModelStatus: ModelStatus {
        container.modelDownloadManager.status(for: settings.selectedAgentModelID)
    }

    /// Language options for the Translate skill's target picker — the one
    /// canonical list, shared with the status-bar menu's Translate To
    /// submenu.
    private var translateLanguageOptions: [String] {
        SupportedLanguage.translateTargetOptions(current: settings.translateTargetLanguage)
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

                if selectedAgentModelDeclaresReasoningEffort {
                    Picker("Reasoning Effort", selection: $settings.agentReasoningEffort) {
                        Text("Automatic (model default)").tag(ReasoningEffort?.none)
                        Text("Low").tag(ReasoningEffort?.some(.low))
                        Text("Medium").tag(ReasoningEffort?.some(.medium))
                        Text("Extra High").tag(ReasoningEffort?.some(.xhigh))
                    }
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
            VStack(alignment: .leading, spacing: 4) {
                if selectedAgentModelDeclaresPreserveThinking {
                    Text(
                        "Preserve Thinking keeps each turn's thinking in the prompt so follow-up requests reuse the cache instead of re-reading the conversation. Uses more context window. Applies to new conversations."
                    )
                }
                if selectedAgentModelDeclaresReasoningEffort {
                    Text(
                        "Reasoning Effort sets how deeply the model thinks before answering. Automatic uses the model's own default (Extra High for Qwen3.8). Changing it rewrites the start of the prompt, so the next turn re-reads the conversation once."
                    )
                }
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
            // The voice session's taste ledger (#310) — tuned in wear.
            Toggle("Auto-Send Voice Turns", isOn: $settings.companionVoiceAutoSend)
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
        } header: {
            Text("Companion Voice")
        } footer: {
            Text(
                "Voice conversations ride the chat itself: the waveform button in the composer opens a session where the mic listens after each reply and silence sends your turn. While he speaks the mic is off — press the Talk to Tesseract or Speak Selected Text hotkey, or click his words, and he stops and listens. The overlay concepts (ticket #328) are the session's face — pick one, preview with the scripted scenes. Auto-Send off stages your words in the composer instead of sending."
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
                Picker("Speculative Decoding", selection: $settings.speculationMode) {
                    ForEach(SpeculationMode.allCases, id: \.self) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
            } footer: {
                Text(
                    "Speeds up generation by drafting several tokens per step. Automatic loads every drafter the model supports and prefers DFlash2 (greedy and sampled, Qwen3.8-27B with its draft downloaded) over MTP (greedy, models that ship the head). Takes effect on the next model load; the Server activity page shows which algorithm served each request."
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

            // PROTOTYPE — the Companion voice-overlay concepts (map #301, #328).
            companionVoiceSection
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

    /// One disk-reading `ModelIdentity` construction off the MainActor
    /// (ADR-0001) yields every template capability the pane shows, so opening
    /// or switching the pane can't stutter and the template is parsed once.
    /// Publish back only while the same model is still selected, so a slow
    /// read for a since-deselected model can't clobber a newer answer.
    private func refreshSelectedAgentModelCapabilities() {
        guard case .downloaded = selectedAgentModelStatus,
            let directory = container.modelDownloadManager.modelPath(
                for: settings.selectedAgentModelID
            )
        else {
            selectedAgentModelDeclaresPreserveThinking = false
            selectedAgentModelDeclaresReasoningEffort = false
            return
        }
        let modelID = settings.selectedAgentModelID
        Task {
            let (declaresPreserve, declaresEffort) = await Task.detached {
                let identity = ModelIdentity(directory: directory)
                return (
                    identity.declaredTemplateFlags.contains(.preserveThinking),
                    identity.declaresReasoningEffort
                )
            }.value
            guard settings.selectedAgentModelID == modelID else { return }
            selectedAgentModelDeclaresPreserveThinking = declaresPreserve
            selectedAgentModelDeclaresReasoningEffort = declaresEffort
        }
    }
}
