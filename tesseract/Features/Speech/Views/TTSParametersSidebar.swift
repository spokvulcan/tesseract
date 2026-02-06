//
//  TTSParametersSidebar.swift
//  tesseract
//

import SwiftUI

struct TTSParametersSidebar: View {
    @ObservedObject private var settings = SettingsManager.shared

    private let sidebarWidth: CGFloat = 260

    var body: some View {
        Form {
            Section("Sampling") {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Temperature")
                        Spacer()
                        Text(String(format: "%.2f", settings.ttsTemperature))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $settings.ttsTemperature, in: 0.0...2.0, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-P")
                        Spacer()
                        Text(String(format: "%.2f", settings.ttsTopP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $settings.ttsTopP, in: 0.0...1.0, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Repetition Penalty")
                        Spacer()
                        Text(String(format: "%.2f", settings.ttsRepetitionPenalty))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $settings.ttsRepetitionPenalty, in: 1.0...2.0, step: 0.05)
                }
            }

            Section("Limits") {
                Stepper(
                    "Context Size: \(settings.ttsRepetitionContextSize)",
                    value: $settings.ttsRepetitionContextSize,
                    in: 1...100
                )

                VStack(alignment: .leading, spacing: 4) {
                    Text("Max Tokens")
                    Picker("", selection: $settings.ttsMaxTokens) {
                        Text("1024").tag(1024)
                        Text("2048").tag(2048)
                        Text("4096").tag(4096)
                        Text("8192").tag(8192)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }

            Section {
                Button("Reset Defaults", role: .destructive) {
                    settings.ttsTemperature = 0.6
                    settings.ttsTopP = 0.8
                    settings.ttsRepetitionPenalty = 1.3
                    settings.ttsRepetitionContextSize = 20
                    settings.ttsMaxTokens = 4096
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: sidebarWidth)
        .background(.ultraThinMaterial)
    }
}
