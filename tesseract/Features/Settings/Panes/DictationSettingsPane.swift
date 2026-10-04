//
//  DictationSettingsPane.swift
//  tesseract
//

import SwiftUI

/// The Dictation pane (#213): microphone, dictation model, transcription
/// language, duration, whether a take waits in the Lens before it pastes,
/// the after-transcription behavior, and the recent recordings diagnostics
/// store.
struct DictationSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer

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

                AudioLevelMeter(
                    audioCapture: container.audioCaptureEngine,
                    feed: container.dictationFeed
                )
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

                Button("Manage Models…") {
                    (NSApp.delegate as? AppDelegate)?.navigateToModels()
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
                Picker("Check Before Pasting", selection: $settings.checkBeforePasting) {
                    ForEach(CheckBeforePasting.allCases) { option in
                        Text(option.displayName).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            } footer: {
                Text(checkBeforePastingHelp)
            }

            Section {
                Toggle("Automatically Insert Text", isOn: $settings.autoInsertText)
                Toggle("Restore Clipboard Contents", isOn: $settings.restoreClipboard)
                    .disabled(!settings.autoInsertText)
                Toggle("Proofread Dictation", isOn: $settings.proofreadDictation)
            } header: {
                Text("After Transcription")
            } footer: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(
                        "Types the transcription into the frontmost app, then puts whatever was on the clipboard back."
                    )
                    if settings.proofreadDictation
                        && !container.modelDownloadManager.isDownloaded(
                            ModelDefinition.defaultProofreadModelID)
                    {
                        Text(
                            "Proofreading needs the Dictation Proofreader model: download it from the Models page."
                        )
                    } else {
                        Text(
                            "Proofreading rewrites each take with a small local model and can change what you meant, so it is off by default. Fix a misheard word with \(settings.fixHotkey.displayString) instead, and it becomes a Learned Word."
                        )
                    }
                }
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

    private var checkBeforePastingHelp: String {
        switch settings.checkBeforePasting {
        case .whenShiftTapped:
            "Tap ⇧ while you talk and the take waits in the Lens until you press ↩, so you can fix a word before it lands."
        case .always:
            "Every take waits in the Lens until you press ↩, so you can fix a word before it lands."
        case .never:
            "Every take pastes when you let go. Fix a word afterwards with \(settings.fixHotkey.displayString)."
        }
    }
}
