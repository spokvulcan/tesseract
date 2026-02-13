//
//  TTSParametersSidebar.swift
//  tesseract
//

import AppKit
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
                    Text("Controls randomness. 0 = fully deterministic (always picks the most likely token). Higher values produce more varied but less stable speech.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
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
                    Text("Nucleus sampling. Limits token selection to the smallest set whose cumulative probability exceeds this threshold. Lower values restrict choices to higher-probability tokens, improving clarity at the cost of expressiveness.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
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
                    Text("Penalizes previously generated tokens to prevent repetitive patterns and audio loops. 1.0 = no penalty.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Seed")
                        Spacer()
                        Text("\(settings.ttsSeed)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    HStack {
                        TextField("", value: $settings.ttsSeed, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: .infinity)
                        Button {
                            settings.ttsSeed = Int.random(in: 0...99999)
                        } label: {
                            Image(systemName: "dice")
                        }
                        .buttonStyle(.borderless)
                    }
                    Text("Fixed seed makes generation reproducible. Same seed + same text + same settings = identical audio output.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Limits") {
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
                    Text("Maximum codec tokens per segment. Higher values allow longer audio per generation call. 4096 tokens ~ 2 min of audio at 12 Hz.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Playback") {
                Toggle("Stream audio", isOn: $settings.ttsStreamingEnabled)
                Text("Play audio progressively as it's generated instead of waiting for completion. Also enables automatic text segmentation for long-form content.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section {
                Button("Reset Defaults", role: .destructive) {
                    settings.ttsTemperature = 0.6
                    settings.ttsTopP = 0.8
                    settings.ttsRepetitionPenalty = 1.3
                    settings.ttsMaxTokens = 4096
                    settings.ttsSeed = 0
                    settings.ttsStreamingEnabled = true
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .frame(width: sidebarWidth)
        .frame(maxHeight: .infinity)
        .background {
            SidebarMaterial()
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.2), radius: 16, x: -2, y: 4)
    }
}

// MARK: - Sidebar Material (NSVisualEffectView bridge)

/// Uses AppKit's NSVisualEffectView with `.sidebar` material to get the exact same
/// translucent background as NavigationSplitView's left sidebar.
private struct SidebarMaterial: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = .sidebar
        view.blendingMode = .behindWindow
        view.state = .followsWindowActiveState
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
