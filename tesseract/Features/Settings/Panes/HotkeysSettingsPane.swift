//
//  HotkeysSettingsPane.swift
//  tesseract
//

import Carbon.HIToolbox
import SwiftUI

/// The Hotkeys pane (#213): every row of the **Hotkey Table** — the global
/// hotkeys in one section, the Appshot gesture in its own — and the Screen
/// Recording permission Appshots depend on.
struct HotkeysSettingsPane: View {
    @EnvironmentObject private var container: DependencyContainer

    var body: some View {
        Form {
            Section("Global Hotkeys") {
                ForEach(GlobalHotkey.allCases.filter { $0 != .appshot }, id: \.self) {
                    HotkeyRecorderRow(hotkey: $0)
                }
            }

            Section {
                HotkeyRecorderRow(hotkey: .appshot)
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

/// One hotkey recorder line: the row's label, its current combo,
/// Change/Cancel, and a reset to its default once it differs — the single
/// generic row that replaced the four copy-pasted recorder sections of the
/// retired Preferences page.
struct HotkeyRecorderRow: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer
    let hotkey: GlobalHotkey
    @State private var isRecording = false

    private var row: HotkeyRow { hotkey.row }

    private var combo: KeyCombo {
        get { settings[keyPath: row.combo] }
        nonmutating set { settings[keyPath: row.combo] = newValue }
    }

    var body: some View {
        HStack {
            Text(row.label)
            Spacer()

            if isRecording {
                Text("Press a key, or a modifier on its own…")
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

            if combo != row.defaultCombo {
                Button(row.resetLabel) {
                    combo = row.defaultCombo
                }
                .buttonStyle(.bordered)
            }
        }

        if let help = row.help {
            Text(help)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }

        if combo.isSingleModifier {
            Text(
                combo.keyCode == UInt16(kVK_Function)
                    ? "Tap to type, hold to speak. fn alone needs System Settings → Keyboard → “Press 🌐 key to” set to Do Nothing."
                    : "One key: tap to type, hold to speak. Pressing another key with it types as usual."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        } else if combo.modifierFlags.contains(.function) {
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
            // Ends the pending recording, so the next key press isn't taken
            // and the other hotkeys work again at once.
            container.hotkeyManager.cancelRecording()
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
