//
//  SpeechSettingsPane.swift
//  tesseract
//
//  The Speech pane: how the Voice Engine renders a voice. Two sliders with
//  plain ends for what changes the sound; the sampler's remaining knobs
//  under Advanced. Off the Speech page, which stays for reading.
//

import SwiftUI
import TesseractSpeech

struct SpeechSettingsPane: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section {
                EndLabeledSlider(
                    title: "Expressiveness", value: $settings.ttsTemperature, range: 0.3...1.5,
                    low: "Even", high: "Lively")
                EndLabeledSlider(
                    title: "Voice steadiness", value: $settings.ttsDetailTemperature,
                    range: 0.1...1.0, low: "Steady", high: "Loose")
            } header: {
                Text("Voice Rendering")
            } footer: {
                Text(
                    "Expressiveness is how much pacing, emphasis and intonation vary. Steadiness keeps the timbre the same person from passage to passage; lower is steadier."
                )
            }

            Section {
                EndLabeledSlider(
                    title: "Top-p", value: $settings.ttsTopP, range: 0.5...1.0, low: "Narrow",
                    high: "Full")
                EndLabeledSlider(
                    title: "Repetition penalty", value: $settings.ttsRepetitionPenalty,
                    range: 1.0...1.5, low: "Off", high: "Strong")
                Picker("Longest passage", selection: $settings.ttsMaxTokens) {
                    // Codec frames are 80 ms each.
                    Text("1 min 20 s").tag(1024)
                    Text("2 min 45 s").tag(2048)
                    Text("5 min 30 s").tag(4096)
                    Text("11 min").tag(8192)
                }
                LabeledContent("Seed") {
                    HStack(spacing: 6) {
                        TextField(
                            "Seed", value: $settings.ttsSeed, format: .number.grouping(.never)
                        )
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                        Button {
                            settings.ttsSeed = Int.random(in: 0...99_999)
                        } label: {
                            Image(systemName: "dice")
                        }
                        .buttonStyle(.borderless)
                        .help("New seed")
                        .accessibilityLabel("New seed")
                    }
                }
                Button("Reset to Recommended") {
                    settings.ttsParameters = TTSParameters()
                    settings.ttsSeed = SettingsCatalogue.ttsSeed.default
                }
            } header: {
                Text("Advanced")
            } footer: {
                Text(
                    "Top-p limits sampling to the likeliest sounds. The repetition penalty keeps the voice from looping. The same seed, text and voice render the same audio."
                )
            }
        }
        .formStyle(.grouped)
    }
}

/// A slider with its value, and what each end means under the track.
private struct EndLabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let low: String
    let high: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            // Snaps to 0.05 on set: a `step:` over these ranges would draw
            // dozens of tick marks.
            Slider(
                value: Binding(get: { value }, set: { value = ($0 / 0.05).rounded() * 0.05 }),
                in: range
            ) {
                Text(title)
            }
            .labelsHidden()
            HStack {
                Text(low)
                Spacer()
                Text(high)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
