//
//  PhoneSettingsScreen.swift
//  tesseract-ios
//
//  Settings: how the Reader looks, the voice, and the privacy statement.
//

import SwiftUI

struct PhoneSettingsScreen: View {
    @Environment(PhoneSettings.self) private var settings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var settings = settings
        NavigationStack {
            Form {
                Section("Reading") {
                    LabeledContent("Text size") {
                        Stepper(
                            "\(Int(settings.readerTextSize)) pt", value: $settings.readerTextSize,
                            in: 14...30, step: 1)
                    }
                    Picker("Highlight", selection: $settings.readerHighlight) {
                        ForEach(ReadAlongHighlight.allCases) { style in
                            Text(style.label).tag(style)
                        }
                    }
                }
                Section("Voice") {
                    NavigationLink {
                        VoiceList()
                    } label: {
                        LabeledContent(
                            "Voice", value: PresetVoice.named(settings.phoneVoice)?.name ?? "")
                    }
                }
                Section("Privacy") {
                    Text(
                        "Tesseract reads on your iPhone. Your texts never leave it, and nothing you read is sent anywhere."
                    )
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

/// The speakers, as a settings page.
private struct VoiceList: View {
    @Environment(PhoneSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        List {
            Picker("Voice", selection: $settings.phoneVoice) {
                ForEach(PresetVoice.all) { voice in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(voice.name)
                        Text(voice.detail).font(.subheadline).foregroundStyle(.secondary)
                    }
                    .tag(voice.id)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        }
        .navigationTitle("Voice")
    }
}
