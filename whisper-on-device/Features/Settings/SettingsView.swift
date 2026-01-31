//
//  SettingsView.swift
//  whisper-on-device
//

import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsTab()
                .tabItem {
                    Label("General", systemImage: "gear")
                }

            AudioSettingsTab()
                .tabItem {
                    Label("Audio", systemImage: "mic")
                }

            ModelSettingsTab()
                .tabItem {
                    Label("Model", systemImage: "brain")
                }

            AdvancedSettingsTab()
                .tabItem {
                    Label("Advanced", systemImage: "slider.horizontal.3")
                }
        }
        .frame(width: 500, height: 400)
    }
}

// MARK: - General Settings Tab

struct GeneralSettingsTab: View {
    @ObservedObject private var settings = SettingsManager.shared

    var body: some View {
        Form {
            Section {
                Toggle("Launch at Login", isOn: $settings.launchAtLogin)
                Toggle("Show in Dock", isOn: $settings.showInDock)
                Toggle("Show in Menu Bar", isOn: $settings.showInMenuBar)
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
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Audio Settings Tab

struct AudioSettingsTab: View {
    @ObservedObject private var settings = SettingsManager.shared
    @StateObject private var audioDeviceManager = AudioDeviceManager()

    var body: some View {
        Form {
            Section("Microphone") {
                Picker("Input Device", selection: $settings.selectedMicrophoneUID) {
                    Text("System Default").tag("")
                    ForEach(audioDeviceManager.availableDevices) { device in
                        Text(device.name).tag(device.uid)
                    }
                }

                AudioLevelMeter()
                    .frame(height: 20)
            }

            Section("Voice Detection") {
                VStack(alignment: .leading) {
                    Text("Sensitivity: \(Int(settings.vadSensitivity * 100))%")
                    Slider(value: $settings.vadSensitivity, in: 0...1)
                }

                VStack(alignment: .leading) {
                    Text("Silence Threshold: \(String(format: "%.1f", settings.silenceThreshold))s")
                    Slider(value: $settings.silenceThreshold, in: 0.1...2.0, step: 0.1)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Audio Level Meter

struct AudioLevelMeter: View {
    @StateObject private var audioCapture = AudioCaptureEngine()
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

// MARK: - Model Settings Tab

struct ModelSettingsTab: View {
    @EnvironmentObject private var container: DependencyContainer
    @ObservedObject private var settings = SettingsManager.shared
    @State private var isLoadingModel = false

    private var modelManager: ModelManager {
        container.modelManager
    }

    var body: some View {
        Form {
            Section("Transcription Model") {
                Picker("Model", selection: $settings.selectedModel) {
                    ForEach(WhisperModel.allCases) { model in
                        HStack {
                            Text(model.displayName)
                            Spacer()
                            if modelManager.isModelDownloaded(model) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            }
                        }
                        .tag(model.rawValue)
                    }
                }
                .onChange(of: settings.selectedModel) { _, newValue in
                    loadSelectedModel(newValue)
                }

                if let selectedModel = WhisperModel(rawValue: settings.selectedModel) {
                    Text(selectedModel.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack {
                        Text("Size: \(String(format: "%.1f", selectedModel.sizeGB)) GB")
                        Spacer()
                        Text("RAM: \(selectedModel.recommendedRAMGB) GB+")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    // Show loading or status
                    if isLoadingModel {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text("Loading model...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else if container.transcriptionEngine.loadedModel == selectedModel {
                        Text("✓ Model loaded and ready")
                            .font(.caption)
                            .foregroundStyle(.green)
                    } else if modelManager.isModelDownloaded(selectedModel) {
                        Text("Model downloaded but not loaded")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }

            Section("Download") {
                ForEach(WhisperModel.allCases) { model in
                    ModelDownloadRow(model: model, modelManager: modelManager)
                }
            }

            Section {
                let usedSpace = modelManager.diskSpaceUsed()
                Text("Disk space used: \(ByteCountFormatter.string(fromByteCount: usedSpace, countStyle: .file))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func loadSelectedModel(_ rawValue: String) {
        guard let model = WhisperModel(rawValue: rawValue),
              modelManager.isModelDownloaded(model) else {
            return
        }

        isLoadingModel = true
        Task {
            defer { isLoadingModel = false }
            try? await container.transcriptionEngine.loadModel(model)
        }
    }
}

struct ModelDownloadRow: View {
    let model: WhisperModel
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(model.displayName)
                Text("\(String(format: "%.1f", model.sizeGB)) GB")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if modelManager.isDownloading[model] == true {
                ProgressView(value: modelManager.downloadProgress[model] ?? 0)
                    .frame(width: 100)

                Button("Cancel") {
                    modelManager.cancelDownload(model)
                }
                .buttonStyle(.bordered)
            } else if modelManager.isModelDownloaded(model) {
                Button("Delete") {
                    try? modelManager.deleteModel(model)
                }
                .buttonStyle(.bordered)
            } else {
                Button("Download") {
                    Task {
                        try? await modelManager.downloadModel(model)
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }
}

// MARK: - Advanced Settings Tab

struct AdvancedSettingsTab: View {
    @ObservedObject private var settings = SettingsManager.shared
    @StateObject private var hotkeyManager = HotkeyManager()
    @State private var isRecordingHotkey = false

    var body: some View {
        Form {
            Section("Global Hotkey") {
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
            }

            Section("Recording") {
                VStack(alignment: .leading) {
                    Text("Maximum duration: \(Int(settings.maxRecordingDuration))s")
                    Slider(value: $settings.maxRecordingDuration, in: 10...120, step: 10)
                }
            }

            Section("Language") {
                Picker("Transcription Language", selection: $settings.language) {
                    Text("Auto-detect").tag("auto")
                    Text("English").tag("en")
                    Text("Spanish").tag("es")
                    Text("French").tag("fr")
                    Text("German").tag("de")
                    Text("Italian").tag("it")
                    Text("Portuguese").tag("pt")
                    Text("Russian").tag("ru")
                    Text("Japanese").tag("ja")
                    Text("Chinese").tag("zh")
                }
            }

            Section {
                Button("Reset to Defaults") {
                    settings.resetToDefaults()
                }
                .foregroundStyle(.red)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func recordNewHotkey() {
        isRecordingHotkey = true

        Task {
            if let combo = await hotkeyManager.recordHotkey() {
                settings.hotkey = combo
            }
            isRecordingHotkey = false
        }
    }
}

#Preview {
    SettingsView()
}
