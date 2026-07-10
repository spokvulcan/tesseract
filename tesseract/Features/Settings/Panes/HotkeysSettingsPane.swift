//
//  HotkeysSettingsPane.swift
//  tesseract
//

import SwiftUI

/// The Hotkeys pane (#213): every global hotkey in one section, the Appshot
/// gesture with its reset, and the Screen Recording permission it depends on.
struct HotkeysSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section("Global Hotkeys") {
                HotkeyRecorderRow(label: "Dictation Push-to-Talk", combo: $settings.hotkey)
                HotkeyRecorderRow(label: "Speak Selected Text", combo: $settings.ttsHotkey)
                HotkeyRecorderRow(label: "Talk to Tesseract", combo: $settings.agentHotkey)
            }

            Section {
                HotkeyRecorderRow(
                    label: "Capture Frontmost Window",
                    combo: $settings.appshotHotkey,
                    resetTo: .doubleCommand,
                    resetLabel: "Reset to ⌘⌘"
                )
            } header: {
                Text("Appshot")
            } footer: {
                Text(
                    "Press both Command keys in any app to attach a shot of its frontmost window to the agent composer."
                )
            }

            Section {
                HStack {
                    Text("Screen Recording")
                    Spacer()
                    if container.permissionsManager.screenRecordingPermission == .granted {
                        Label("Granted", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .labelStyle(.titleAndIcon)
                    } else {
                        Text("Not Granted")
                            .foregroundStyle(.secondary)
                        Button("Grant…") {
                            container.permissionsManager.requestScreenRecordingPermission()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            } header: {
                Text("Permissions")
            } footer: {
                if container.permissionsManager.screenRecordingPermission != .granted {
                    Text(
                        "Appshots need Screen Recording. macOS applies the permission after Tesseract is relaunched."
                    )
                }
            }
        }
        .formStyle(.grouped)
    }
}

/// One hotkey recorder line: current combo, Change/Cancel, optional reset —
/// the single generic row that replaced the four copy-pasted recorder
/// sections of the retired Preferences page.
struct HotkeyRecorderRow: View {
    @EnvironmentObject private var container: DependencyContainer
    let label: String
    @Binding var combo: KeyCombo
    var resetTo: KeyCombo?
    var resetLabel: String = "Reset"
    @State private var isRecording = false

    var body: some View {
        HStack {
            Text(label)
            Spacer()

            if isRecording {
                Text("Press a key…")
                    .foregroundStyle(.secondary)
            } else {
                Text(combo.displayString)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.2))
                    .cornerRadius(4)
                    .accessibilityLabel("Current hotkey: \(combo.displayString)")
            }

            Button(isRecording ? "Cancel" : "Change") {
                toggleRecording()
            }
            .buttonStyle(.bordered)
            .accessibilityHint(
                isRecording ? "Cancel recording new hotkey" : "Record a new hotkey")

            if let resetTo, combo != resetTo {
                Button(resetLabel) {
                    combo = resetTo
                }
                .buttonStyle(.bordered)
            }
        }

        if combo.modifierFlags.contains(.function) {
            Text(
                "fn is not reliably delivered for global hotkeys on macOS. Consider adding ⌘, ⌥, ⌃, or ⇧."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func toggleRecording() {
        if isRecording {
            isRecording = false
            return
        }
        isRecording = true
        Task {
            if let newCombo = await container.hotkeyManager.recordHotkey() {
                combo = newCombo
            }
            isRecording = false
        }
    }
}
