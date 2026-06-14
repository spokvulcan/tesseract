//
//  TTSParametersInspector.swift
//  tesseract
//

import SwiftUI

/// Native inspector content for the Speech page's generation parameters.
struct TTSParametersInspector: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section("Sampling") {
                ParameterSliderRow(
                    title: "Temperature",
                    value: $settings.ttsTemperature,
                    range: 0.0...2.0,
                    summary: "Higher is more varied, less stable.",
                    helpText:
                        "Controls randomness. 0 always picks the most likely token; higher values produce more varied but less stable speech."
                )
                ParameterSliderRow(
                    title: "Top-P",
                    value: $settings.ttsTopP,
                    range: 0.0...1.0,
                    summary: "Lower is clearer, less expressive.",
                    helpText:
                        "Nucleus sampling. Restricts choices to the most likely tokens whose cumulative probability stays under this threshold."
                )
                ParameterSliderRow(
                    title: "Repetition Penalty",
                    value: $settings.ttsRepetitionPenalty,
                    range: 1.0...2.0,
                    summary: "Discourages repeats and audio loops.",
                    helpText:
                        "Penalizes previously generated tokens to prevent repetitive patterns. 1.0 applies no penalty."
                )
            }

            Section("Generation") {
                Picker("Max Tokens", selection: $settings.ttsMaxTokens) {
                    Text("1024").tag(1024)
                    Text("2048").tag(2048)
                    Text("4096").tag(4096)
                    Text("8192").tag(8192)
                }
                .help("Maximum codec tokens per segment — 4096 is roughly two minutes of audio.")

                LabeledContent("Seed") {
                    HStack(spacing: Theme.Spacing.xs) {
                        TextField("Seed", value: $settings.ttsSeed, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Button {
                            settings.ttsSeed = Int.random(in: 0...99_999)
                        } label: {
                            Image(systemName: "dice")
                        }
                        .buttonStyle(.borderless)
                        .help("Randomize seed")
                    }
                }
                .help("The same seed, text, and settings reproduce identical audio.")
            }

            Section("Playback") {
                Toggle("Stream audio", isOn: $settings.ttsStreamingEnabled)
                    .help("Play audio as it is generated and segment long-form text automatically.")
            }

            Section {
                Button("Reset to Defaults") {
                    settings.ttsTemperature = SettingsCatalogue.ttsTemperature.default
                    settings.ttsTopP = SettingsCatalogue.ttsTopP.default
                    settings.ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.default
                    settings.ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.default
                    settings.ttsSeed = SettingsCatalogue.ttsSeed.default
                    settings.ttsStreamingEnabled = SettingsCatalogue.ttsStreamingEnabled.default
                }
            }
        }
        .formStyle(.grouped)
        .inspectorColumnWidth(min: 250, ideal: 290, max: 340)
    }
}

// MARK: - Parameter Slider Row

private struct ParameterSliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let summary: String
    let helpText: String

    /// Snap granularity, applied on set instead of `Slider(step:)` so the
    /// track stays clean — a 0.05 step over these ranges would draw dozens
    /// of tick marks.
    private let granularity = 0.05

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            HStack {
                Text(title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: snappedValue, in: range) {
                Text(title)
            }
            .labelsHidden()
            Text(summary)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .help(helpText)
    }

    private var snappedValue: Binding<Double> {
        Binding(
            get: { value },
            set: { value = ($0 / granularity).rounded() * granularity }
        )
    }
}
