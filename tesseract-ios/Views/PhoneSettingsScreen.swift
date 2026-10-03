//
//  PhoneSettingsScreen.swift
//  tesseract-ios
//
//  Settings: how the Reader looks, the voice, and the privacy statement.
//

import SwiftUI
import UIKit

struct PhoneSettingsScreen: View {
    @Environment(PhoneSettings.self) private var settings
    @Environment(\.readingMeter) private var meter
    @Environment(\.dismiss) private var dismiss

    private var diagnostics: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let version = info["CFBundleShortVersionString"] as? String ?? "?"
        let build = info["CFBundleVersion"] as? String ?? "?"
        return ReadingMeter.report(
            app: "Tesseract \(version) (\(build))", device: Self.deviceModel,
            system: "iOS \(UIDevice.current.systemVersion)", voice: "System Voice",
            segments: meter?.segments ?? [])
    }

    /// The model identifier, such as iPhone17,2.
    private static var deviceModel: String {
        var system = utsname()
        uname(&system)
        return withUnsafeBytes(of: &system.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

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
                Section {
                    Text(diagnostics)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    Button("Copy Diagnostics", systemImage: "doc.on.doc") {
                        UIPasteboard.general.string = diagnostics
                    }
                } header: {
                    Text("Diagnostics")
                } footer: {
                    Text(
                        "Paste these into a TestFlight report: they say how fast and how warm the last reading was."
                    )
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

extension EnvironmentValues {
    /// The meter behind the Settings screen's diagnostics.
    @Entry var readingMeter: ReadingMeter?
}
