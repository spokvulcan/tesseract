//
//  SpeechComposerView.swift
//  tesseract
//

import SwiftUI

/// The Speech page's main surface: a Mail-style composer card with a voice
/// description header row and the text to speak as the body.
struct SpeechComposerView: View {
    @Binding var text: String
    @Binding var voiceDescription: String
    @Binding var language: String

    private static let voicePresets: [(label: String, description: String)] = [
        ("Natural", "A natural, clear voice with a moderate pace and neutral tone, suitable for everyday conversations."),
        ("Warm", "A warm, friendly female voice with a gentle tone and smooth cadence, comforting and approachable."),
        ("Deep", "A deep, resonant male narrator voice with a measured pace and authoritative presence."),
        ("Calm", "A calm, soothing voice with a slow, deliberate pace, perfect for reading and relaxation.")
    ]

    var body: some View {
        VStack(spacing: 0) {
            voiceRow
                .padding(.horizontal, Theme.Spacing.lg)
                .padding(.vertical, Theme.Spacing.md)

            Divider()

            editor
        }
        .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: Theme.Radius.large))
        .frame(maxWidth: Theme.Layout.contentMaxWidth)
    }

    // MARK: - Voice row

    private var voiceRow: some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.sm) {
            Text("Voice")
                .foregroundStyle(.secondary)

            TextField(
                "Describe a voice…",
                text: $voiceDescription,
                axis: .vertical
            )
            .textFieldStyle(.plain)
            .lineLimit(1...3)
            .help("Describe the voice in natural language — e.g. “a warm, friendly narrator”. Leave empty for the default voice.")

            Menu {
                ForEach(Self.voicePresets, id: \.label) { preset in
                    Button(preset.label) { voiceDescription = preset.description }
                }
            } label: {
                Image(systemName: "sparkles")
            }
            .buttonStyle(.borderless)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Voice presets")

            Picker("Language", selection: $language) {
                ForEach(TTSLanguage.allCases) { lang in
                    Text("\(lang.flag) \(lang.displayName)").tag(lang.rawValue)
                }
            }
            .labelsHidden()
            .fixedSize()
        }
    }

    // MARK: - Text body

    private var editor: some View {
        TextEditor(text: $text)
            .font(.body)
            .lineSpacing(3)
            .scrollContentBackground(.hidden)
            .padding(Theme.Spacing.sm)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .overlay(alignment: .topLeading) {
                if text.isEmpty {
                    Text("Type or paste text to speak…")
                        .foregroundStyle(.tertiary)
                        .padding(.top, Theme.Spacing.sm)
                        .padding(.leading, Theme.Spacing.sm + 5)
                        .allowsHitTesting(false)
                }
            }
            .overlay(alignment: .bottomTrailing) {
                if !text.isEmpty {
                    Text("\(wordCount) words")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .monospacedDigit()
                        .padding(Theme.Spacing.sm)
                        .allowsHitTesting(false)
                }
            }
    }

    private var wordCount: Int {
        text.split(whereSeparator: \.isWhitespace).count
    }
}
