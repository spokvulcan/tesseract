//
//  SettingsWindowView.swift
//  tesseract
//

import SwiftUI

/// The native Settings window (map #211, cutover #216): a `Settings` scene
/// hosting a `TabView` toolbar of panes — the ratified IA (#213) and design
/// language rule 7 (`docs/design/design-language.md`). The window sizes to
/// the current pane; the system restores the last-selected pane.
///
/// The Server pane is the existing `ServerConfigurationView` embedded
/// verbatim — the configuration form already is the ratified pane (its
/// grouped `Form` scrolls, being the longest).
struct SettingsWindowView: View {
    var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") {
                GeneralSettingsPane()
                    .frame(width: 620, height: 500)
            }
            Tab("Hotkeys", systemImage: "keyboard") {
                HotkeysSettingsPane()
                    .frame(width: 620, height: 560)
            }
            Tab("Agent", systemImage: "brain.head.profile") {
                AgentSettingsPane()
                    .frame(width: 620, height: 620)
            }
            Tab("Dictation", systemImage: "mic.fill") {
                DictationSettingsPane()
                    .frame(width: 620, height: 660)
            }
            Tab("Server", systemImage: "server.rack") {
                ServerConfigurationView()
                    .frame(width: 620, height: 700)
            }
        }
    }
}
