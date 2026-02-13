//
//  VoiceDesignView.swift
//  tesseract
//

import SwiftUI

struct VoiceDesignView: View {
    @Binding var voiceDescription: String

    private let characterLimit = 500

    private let presets: [(label: String, description: String)] = [
        ("Natural", "A natural, clear voice with a moderate pace and neutral tone, suitable for everyday conversations."),
        ("Warm", "A warm, friendly female voice with a gentle tone and smooth cadence, comforting and approachable."),
        ("Deep", "A deep, resonant male narrator voice with a measured pace and authoritative presence."),
        ("Calm", "A calm, soothing voice with a slow, deliberate pace, perfect for reading and relaxation.")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice Design")
                .font(.headline)

            ZStack(alignment: .topLeading) {
                TextEditor(text: $voiceDescription)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 80, maxHeight: 120)

                if voiceDescription.isEmpty {
                    Text("Describe the voice you want... e.g. \"A warm, friendly voice with a gentle tone\"")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            .padding(8)
            .background(.fill.quaternary)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                // Preset pills
                ForEach(presets, id: \.label) { preset in
                    Button(preset.label) {
                        voiceDescription = preset.description
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                }

                Spacer()

                // Character count
                Text("\(voiceDescription.count)/\(characterLimit)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }
        }
    }
}
