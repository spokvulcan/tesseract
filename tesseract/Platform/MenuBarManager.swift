//
//  MenuBarManager.swift
//  tesseract
//

import Foundation
import AppKit
import Observation
import SwiftUI
import Combine

@MainActor
final class MenuBarManager: ObservableObject {
    private var statusItem: NSStatusItem?
    private var cancellables = Set<AnyCancellable>()
    private var historyObservationTask: Task<Void, Never>?
    private var settingsObservationTask: Task<Void, Never>?
    private weak var toggleItem: NSMenuItem?
    private weak var copyLastItem: NSMenuItem?
    private weak var speakItem: NSMenuItem?
    private weak var talkItem: NSMenuItem?

    @Published var isRecording = false
    @Published var hasHistory = false

    let settings: SettingsManager
    weak var coordinator: DictationCoordinator?
    weak var history: TranscriptionHistory?
    weak var speechCoordinator: SpeechCoordinator?

    var onShowMainWindow: (() -> Void)?
    var onShowSettings: (() -> Void)?
    var onTalkToAgent: (() -> Void)?
    var onQuit: (() -> Void)?

    init(settings: SettingsManager) {
        self.settings = settings
    }

    func setupMenuBar() {
        // Subscribe to state changes for menu item text
        $isRecording
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateMenuItems()
            }
            .store(in: &cancellables)

        historyObservationTask = Task { @MainActor [weak self] in
            guard let self, let history = self.history else { return }
            for await entries in Observations({ history.entries }) {
                self.hasHistory = !entries.isEmpty
                self.updateMenuItems()
            }
        }

        settingsObservationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for await _ in Observations({
                (
                    self.settings.showInMenuBar,
                    self.settings.hotkey,
                    self.settings.ttsHotkey,
                    self.settings.agentHotkey
                )
            }) {
                self.applyCurrentSettings()
            }
        }

        applyCurrentSettings()
    }

    func teardownMenuBar() {
        historyObservationTask?.cancel()
        historyObservationTask = nil
        settingsObservationTask?.cancel()
        settingsObservationTask = nil
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
            statusItem = nil
        }
        toggleItem = nil
        copyLastItem = nil
        speakItem = nil
        talkItem = nil
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
        if let image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Tesseract Agent")?
            .withSymbolConfiguration(config) {
            image.isTemplate = true
            button.image = image
        }
    }

    private func updateMenuItems() {
        guard statusItem != nil else { return }

        let hotkeyDisplay = settings.hotkey.displayString
        toggleItem?.title = isRecording
            ? "Stop Dictation (\(hotkeyDisplay))"
            : "Start Dictation (\(hotkeyDisplay))"

        copyLastItem?.isEnabled = hasHistory
        speakItem?.title = "Speak Selected Text (\(settings.ttsHotkey.displayString))"
        talkItem?.title = "Talk to Tesse (\(settings.agentHotkey.displayString))"
    }

    private func applyCurrentSettings() {
        setMenuBarVisible(settings.showInMenuBar)
        updateMenuItems()
    }

    private func setMenuBarVisible(_ isVisible: Bool) {
        if isVisible {
            if statusItem == nil {
                createStatusItem()
            }
            updateIcon()
        } else if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
            statusItem = nil
            toggleItem = nil
            copyLastItem = nil
            speakItem = nil
            talkItem = nil
        }
    }

    private func createStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        let menu = NSMenu()

        let toggleItem = NSMenuItem(
            title: "Start Dictation",
            action: #selector(toggleDictation),
            keyEquivalent: ""
        )
        toggleItem.target = self
        menu.addItem(toggleItem)
        self.toggleItem = toggleItem

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

        let speakItem = NSMenuItem(
            title: "Speak Selected Text",
            action: #selector(speakSelectedText),
            keyEquivalent: ""
        )
        speakItem.target = self
        menu.addItem(speakItem)
        self.speakItem = speakItem

        let talkItem = NSMenuItem(
            title: "Talk to Tesse",
            action: #selector(talkToAgent),
            keyEquivalent: ""
        )
        talkItem.target = self
        menu.addItem(talkItem)
        self.talkItem = talkItem

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
            keyEquivalent: ""
        )
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(
            title: "Quit Tesseract Agent",
            action: #selector(quit),
            keyEquivalent: ""
        )
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu
    }

    @objc private func toggleDictation() {
        coordinator?.toggleRecording()
    }

    @objc private func copyLastTranscription() {
        history?.copyLatestToPasteboard()
    }

    @objc private func speakSelectedText() {
        speechCoordinator?.onHotkeyPressed()
    }

    @objc private func talkToAgent() {
        onTalkToAgent?()
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
