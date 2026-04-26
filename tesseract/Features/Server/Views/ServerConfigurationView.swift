//
//  ServerConfigurationView.swift
//  tesseract
//

import MLXLMCommon
import SwiftUI

struct ServerConfigurationView: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(InferenceArbiter.self) private var arbiter
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
                Toggle("Enable TriAttention", isOn: $settings.triattentionEnabled)

                if settings.triattentionEnabled {
                    LabeledContent("Active mode") {
                        Text(triAttentionModeDescription(for: arbiter))
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text("Sparse Attention (Advanced)")
            } footer: {
                Text("TriAttention reduces attention compute on long-context text inference. Supported only on PARO quantized Qwen3.5 models — other models automatically fall back to dense attention. Vision mode also falls back to dense. Toggling reloads the currently loaded model.")
            }

            Section {
                Toggle("Enable DFlash speculative decoding", isOn: $settings.dflashEnabled)
            } header: {
                Text("Speculative Decoding (Experimental)")
            } footer: {
                Text("DFlash uses a small block-diffusion draft model to speed up agent decoding by ~1.5× on supported targets. Currently supports only Qwen3.6-27B with greedy sampling — other models or non-greedy sampling silently fall back to autoregressive. The draft (~3.2 GB) must be downloaded separately from the Models tab. Toggling reloads the currently loaded model.")
            }
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
