//
//  GeneralSettingsPane.swift
//  tesseract
//

import SwiftUI

/// The General pane (#213): app presence, feedback, the recording overlay,
/// onboarding re-entry, and Reset to Defaults.
struct GeneralSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.openWindow) private var openWindow

    // Prevent disabling both dock and menu bar — the app must stay reachable.
    private var showInDockBinding: Binding<Bool> {
        Binding(
            get: { settings.showInDock },
            set: { newValue in
                if !newValue && !settings.showInMenuBar { return }
                settings.showInDock = newValue
            }
        )
    }

    private var showInMenuBarBinding: Binding<Bool> {
        Binding(
            get: { settings.showInMenuBar },
            set: { newValue in
                if !newValue && !settings.showInDock { return }
                settings.showInMenuBar = newValue
            }
        )
    }

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section {
                Toggle("Launch at Login", isOn: $settings.launchAtLogin)
                Toggle("Show in Dock", isOn: showInDockBinding)
                Toggle("Show in Menu Bar", isOn: showInMenuBarBinding)
            } footer: {
                if !settings.showInDock || !settings.showInMenuBar {
                    Text("At least one must stay on so the app remains reachable.")
                }
            }

            Section("Feedback") {
                Toggle("Play Sounds", isOn: $settings.playSounds)
            }

            Section {
                Picker("Overlay Variant", selection: $settings.overlayVariantRaw) {
                    ForEach(OverlayVariants.all) { variant in
                        Text(variant.displayName).tag(variant.id)
                    }
                }
            } header: {
                Text("Recording Overlay")
            } footer: {
                Text("Dictation overlay design explorations — switch live, no restart needed.")
            }

            Section("Setup") {
                Button("Show Welcome Tour…") {
                    openWindow(id: WindowID.onboarding)
                }
            }

            Section {
                Button("Reset to Defaults", role: .destructive) {
                    settings.resetToDefaults()
                }
            }
        }
        .formStyle(.grouped)
    }
}
