//
//  VoicePickerSheet.swift
//  tesseract-ios
//
//  The voice's nine speakers, each a **Preset Voice**. Until the neural voice
//  is on the phone, the system voice reads whichever is chosen.
//

import SwiftUI

struct VoicePickerSheet: View {
    @Environment(PhoneSettings.self) private var settings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(PresetVoice.all) { voice in
                        Button {
                            settings.phoneVoice = voice.id
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(voice.name).font(.body.weight(.medium))
                                    Text(voice.detail)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if settings.phoneVoice == voice.id {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                } footer: {
                    Text(
                        "Every voice reads all ten languages. Until the voice is downloaded, the system voice reads."
                    )
                }
            }
            .navigationTitle("Voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
