//
//  MenuBarManager.swift
//  tesseract
//

import Foundation
import AppKit
import SwiftUI
import Combine

@MainActor
final class MenuBarManager: ObservableObject {
    private var statusItem: NSStatusItem?
    private var cancellables = Set<AnyCancellable>()
    private weak var copyLastItem: NSMenuItem?

    @Published var isRecording = false
    @Published var hasHistory = false

    weak var coordinator: DictationCoordinator?
    weak var history: TranscriptionHistory?

    var onShowMainWindow: (() -> Void)?
    var onShowSettings: (() -> Void)?
    var onQuit: (() -> Void)?

    init() {}

    func setupMenuBar() {
        guard SettingsManager.shared.showInMenuBar else { return }

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        updateIcon()

        // Setup menu
        let menu = NSMenu()

        let hotkeyDisplay = SettingsManager.shared.hotkey.displayString
        let toggleItem = NSMenuItem(
            title: "Start Dictation (\(hotkeyDisplay))",
            action: #selector(toggleDictation),
            keyEquivalent: ""
        )
        toggleItem.target = self
        menu.addItem(toggleItem)

        let copyItem = NSMenuItem(
            title: "Copy Last Dictation",
            action: #selector(copyLastTranscription),
            keyEquivalent: ""
        )
        copyItem.target = self
        copyItem.isEnabled = hasHistory
        menu.addItem(copyItem)
        copyLastItem = copyItem

        menu.addItem(NSMenuItem.separator())

        let mainWindowItem = NSMenuItem(
            title: "Open Main Window",
            action: #selector(showMainWindow),
            keyEquivalent: ""
        )
        mainWindowItem.target = self
        menu.addItem(mainWindowItem)

        let settingsItem = NSMenuItem(
            title: "Settings…",
            action: #selector(showSettings),
            keyEquivalent: ""  // ⌘, handled by app-level Settings scene
        )
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(
            title: "Quit Tesseract",
            action: #selector(quit),
            keyEquivalent: ""  // ⌘Q handled by system
        )
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu

        // Subscribe to state changes for menu item text
        $isRecording
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateMenuItems()
            }
            .store(in: &cancellables)

        history?.$entries
            .receive(on: DispatchQueue.main)
            .sink { [weak self] entries in
                self?.hasHistory = !entries.isEmpty
                self?.updateMenuItems()
            }
            .store(in: &cancellables)
    }

    func teardownMenuBar() {
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
            statusItem = nil
        }
    }

    func updateState(from dictationState: DictationState) {
        switch dictationState {
        case .idle, .error, .processing:
            isRecording = false
        case .listening, .recording:
            isRecording = true
        }
    }

    // MARK: - Private

    private func updateIcon() {
        guard let button = statusItem?.button else { return }

        let config = NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        if let image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Tesseract")?
            .withSymbolConfiguration(config) {
            image.isTemplate = true
            button.image = image
        }
    }

    private func updateMenuItems() {
        guard let menu = statusItem?.menu,
              let toggleItem = menu.items.first else { return }

        let hotkeyDisplay = SettingsManager.shared.hotkey.displayString
        toggleItem.title = isRecording
            ? "Stop Dictation (\(hotkeyDisplay))"
            : "Start Dictation (\(hotkeyDisplay))"

        copyLastItem?.isEnabled = hasHistory
    }

    @objc private func toggleDictation() {
        coordinator?.toggleRecording()
    }

    @objc private func copyLastTranscription() {
        history?.copyLatestToPasteboard()
    }

    @objc private func showMainWindow() {
        onShowMainWindow?()
    }

    @objc private func showSettings() {
        onShowSettings?()
    }

    @objc private func quit() {
        onQuit?()
    }
}
