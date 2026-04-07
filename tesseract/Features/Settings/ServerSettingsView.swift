//
//  ServerSettingsView.swift
//  tesseract
//

import SwiftUI

struct ServerSettingsView: View {
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
                            Text("http://127.0.0.1:\(String(settings.serverPort))")
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                            
                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString("http://127.0.0.1:\(String(settings.serverPort))", forType: .string)
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
        }
        .formStyle(.grouped)
        .navigationTitle("Server API")
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
            // Revert to the current valid port if input was invalid or empty
            portText = String(settings.serverPort)
        }
    }
}
