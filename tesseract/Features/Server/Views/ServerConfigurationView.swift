//
//  ServerConfigurationView.swift
//  tesseract
//

import MLXLMCommon
import SwiftUI

struct ServerConfigurationView: View {
    @Environment(SettingsManager.self) private var settings
    @State private var portText: String = ""
    @FocusState private var isFocused: Bool

    var body: some View {
        @Bindable var settings = settings

        Form {
            Section {
                Toggle("Enable HTTP Server", isOn: $settings.isServerEnabled)

                LabeledContent("Port") {
                    TextField("8321", text: $portText)
                        .focused($isFocused)
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                        .onSubmit {
                            commitPort()
                        }
                }

                if settings.isServerEnabled {
                    LabeledContent("Endpoint") {
                        HStack {
                            Text(serverEndpointURL(port: settings.serverPort))
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)

                            Button {
                                copyServerEndpointToPasteboard(port: settings.serverPort)
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .help("Copy Endpoint")
                        }
                    }
                }
            } footer: {
                Text("The local API server provides an OpenAI-compatible /v1/chat/completions endpoint for integration with other tools.")
            }

            Section {
                if settings.isServerEnabled {
                    LabeledContent("OpenCode") {
                        HStack {
                            Text(OpenCodeSetupScript.oneLiner(port: settings.serverPort))
                                .font(.system(.callout, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)

                            Button {
                                copyOpenCodeSetupCommandToPasteboard(port: settings.serverPort)
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .help("Copy Setup Command")
                        }
                    }
                } else {
                    Text("Enable the server to set up clients.")
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Integrations")
            } footer: {
                Text("Run the command in a terminal to configure OpenCode for this server — every downloaded model, image input included. Re-run it after downloading models or changing the port.")
            }

            ServerPreserveThinkingSection()
        }
        .formStyle(.grouped)
        .navigationTitle("Configuration")
        .onAppear {
            portText = String(settings.serverPort)
        }
        .onChange(of: portText) { _, newValue in
            let filtered = newValue.filter { $0.isNumber }
            if filtered != newValue {
                portText = filtered
            }
        }
        .onChange(of: isFocused) { _, focused in
            if !focused {
                commitPort()
            }
        }
        .onChange(of: settings.serverPort) { _, newValue in
            if Int(portText) != newValue {
                portText = String(newValue)
            }
        }
    }

    private func commitPort() {
        if let port = Int(portText), port > 0, port <= 65535 {
            if settings.serverPort != port {
                settings.serverPort = port
            }
        } else {
            portText = String(settings.serverPort)
        }
    }
}

/// Per-model **Preserve-Thinking Render** opt-in for models the server serves
/// (issue #98, PRD #94). The server already resolves `preserve_thinking` per
/// request — request `chat_template_kwargs` win, this per-model app setting is
/// the fallback, and only templates that declare the flag participate
/// (`ModelIdentity.declaredTemplateFlags`). It writes the same
/// `preserveThinkingRender.<modelID>` key the Agent Preferences toggle does, so
/// a model served to a client (e.g. OpenCode) can be configured even when it is
/// not the selected agent model. Surfaced unconditionally — the setting is
/// per-model and shared with the agent, so it is meaningful with the server off.
private struct ServerPreserveThinkingSection: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var modelDownloadManager: ModelDownloadManager

    /// IDs of downloaded agent models whose template declares
    /// `preserve_thinking` — the models that get a toggle. Populated off the
    /// MainActor (`ModelIdentity.declares`, ADR-0001); empty until the first
    /// scan completes, so the empty-state shows briefly on first appearance.
    @State private var supportingModelIDs: Set<String> = []

    /// Downloaded agent models in catalogue order — the set the server can
    /// serve, mirroring the OpenCode one-liner / `/v1/models`.
    private var downloadedAgentModels: [ModelDefinition] {
        ModelDefinition.all.filter { definition in
            guard definition.category == .agent else { return false }
            if case .downloaded = modelDownloadManager.statuses[definition.id] {
                return true
            }
            return false
        }
    }

    var body: some View {
        let models = downloadedAgentModels
        // Only models whose template declares the flag get a toggle — matches
        // the Agent Preferences gating and the "models that support this" intent.
        let supported = models.filter { supportingModelIDs.contains($0.id) }
        return Section {
            if supported.isEmpty {
                Text("No downloaded model supports preserved thinking. Models such as Qwen3.6 support it.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(supported) { model in
                    Toggle(model.displayName, isOn: Binding(
                        get: { settings.preserveThinkingRender(modelID: model.id) },
                        set: { settings.setPreserveThinkingRender($0, modelID: model.id) }
                    ))
                }
            }
        } header: {
            Text("Preserve Thinking")
        } footer: {
            Text("Keeps each turn's thinking in the prompt so a client's follow-up requests reuse the prefix cache instead of re-prefilling the conversation. Uses more context window. Set per model; applies to new requests.")
        }
        // Re-scan whenever the downloaded agent-model set changes. `task(id:)`
        // cancels the prior scan, and the cancellation check guards against a
        // stale write clobbering the newer set's result.
        .task(id: models.map(\.id)) {
            await refreshCapabilities()
        }
    }

    private func refreshCapabilities() async {
        var supporting: Set<String> = []
        for model in downloadedAgentModels {
            guard let directory = modelDownloadManager.modelPath(for: model.id) else {
                continue
            }
            if await ModelIdentity.declares(.preserveThinking, atDirectory: directory) {
                supporting.insert(model.id)
            }
        }
        guard !Task.isCancelled else { return }
        supportingModelIDs = supporting
    }
}
