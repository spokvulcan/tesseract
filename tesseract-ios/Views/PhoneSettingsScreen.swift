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
    @Environment(PhoneVoice.self) private var voice
    @State private var confirmsDelete = false
    @Environment(\.readingMeter) private var meter
    @Environment(\.dismiss) private var dismiss

    private var diagnostics: String {
        let info = Bundle.main.infoDictionary ?? [:]
        let version = info["CFBundleShortVersionString"] as? String ?? "?"
        let build = info["CFBundleVersion"] as? String ?? "?"
        return ReadingMeter.report(
            app: "Tesseract \(version) (\(build))", device: Self.deviceModel,
            system: "iOS \(UIDevice.current.systemVersion)", voice: voiceName,
            segments: meter?.segments ?? [])
    }

    /// Which voice reads, and how fast the neural one measured.
    private var voiceName: String {
        guard case .ready(let check) = voice.state else { return "System Voice" }
        return String(
            format: "Neural voice (Speed Check RTF %.2f, first audio %.0f ms)",
            check.realTimeFactor, check.firstAudio * 1000)
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
                    Text(voice.summary)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    if case .downloading(let received, let total) = voice.state, total > 0 {
                        ProgressView(value: Double(received), total: Double(total))
                    }
                    switch voice.state {
                    case .notDownloaded:
                        Button("Download \(ModelDefinition.phoneVoice.sizeDescription)") {
                            voice.download()
                        }
                    case .downloading:
                        Button("Stop Downloading", role: .cancel) { voice.cancelDownload() }
                    case .preparing:
                        EmptyView()
                    case .ready, .unavailable:
                        if voice.isDownloaded {
                            Button("Delete the Neural Voice", role: .destructive) {
                                confirmsDelete = true
                            }
                        }
                    }
                    Toggle("Download over Cellular", isOn: $settings.allowsCellularDownloads)
                } header: {
                    Text("Neural Voice")
                } footer: {
                    Text(
                        "Generated on this iPhone's Neural Engine. Without it, or when the iPhone is warm, the system voice reads."
                    )
                }
                .confirmationDialog(
                    "Delete the neural voice?", isPresented: $confirmsDelete,
                    titleVisibility: .visible
                ) {
                    Button("Delete", role: .destructive) { Task { await voice.delete() } }
                } message: {
                    Text("The system voice reads until you download it again.")
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
