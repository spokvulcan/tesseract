//
//  ReaderDisplaySettings.swift
//  tesseract
//
//  The Speech page's one settings popover: how the text looks and lights up
//  while read, and the overlay that shows reading outside the app. Overlay
//  changes preview on screen.
//

import SwiftUI

struct ReaderDisplaySettings: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section("Text") {
                HStack(spacing: 10) {
                    Image(systemName: "textformat.size.smaller")
                        .accessibilityHidden(true)
                    Slider(value: $settings.readerTextSize, in: 14...28, step: 1) {
                        Text("Text size")
                    }
                    .labelsHidden()
                    Image(systemName: "textformat.size.larger")
                        .accessibilityHidden(true)
                }
                Picker("Typeface", selection: $settings.readerTypeface) {
                    ForEach(ReaderTypeface.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                Picker("Highlight", selection: $settings.readerHighlight) {
                    ForEach(ReadAlongHighlight.allCases) { Text($0.label).tag($0) }
                }
            }
            Section {
                Picker("Style", selection: $settings.speechOverlayStyle) {
                    ForEach(SpeechOverlayStyle.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                if settings.speechOverlayStyle != .off {
                    Picker("Size", selection: $settings.speechOverlaySize) {
                        ForEach(SpeechOverlaySize.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Picker("Word color", selection: $settings.speechOverlayTint) {
                        ForEach(SpeechOverlayTint.allCases) { tint in
                            Label {
                                Text(tint.label)
                            } icon: {
                                Image(systemName: "circle.fill").foregroundStyle(tint.color)
                            }
                            .tag(tint)
                        }
                    }
                    Toggle("Pause and stop on hover", isOn: $settings.speechOverlayShowsControls)
                    Picker("Show", selection: $settings.speechOverlayScope) {
                        ForEach(SpeechOverlayScope.allCases) { Text($0.label).tag($0) }
                    }
                }
            } header: {
                Text("Overlay")
            } footer: {
                Text(
                    "Shows the words as they're read, over every app. Speech from the hotkey uses it too."
                )
            }
        }
        .formStyle(.grouped)
        .frame(width: 340)
        .frame(minHeight: 380)
    }
}
