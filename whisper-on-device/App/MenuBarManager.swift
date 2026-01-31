//
//  MenuBarManager.swift
//  whisper-on-device
//

import Foundation
import AppKit
import SwiftUI
import Combine

@MainActor
final class MenuBarManager: ObservableObject {
    private var statusItem: NSStatusItem?
    private var cancellables = Set<AnyCancellable>()
    private var animationTimer: Timer?
    private var isAnimationHighlighted = true

    @Published var isRecording = false
    @Published var isProcessing = false

    weak var coordinator: DictationCoordinator?

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
            title: "Quit WhisperOnDevice",
            action: #selector(quit),
            keyEquivalent: ""  // ⌘Q handled by system
        )
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu

        // Subscribe to state changes
        $isRecording
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateIcon()
                self?.updateMenuItems()
            }
            .store(in: &cancellables)

        $isProcessing
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateIcon()
            }
            .store(in: &cancellables)
    }

    func teardownMenuBar() {
        stopRecordingAnimation()
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
            statusItem = nil
        }
    }

    func updateState(from dictationState: DictationState) {
        switch dictationState {
        case .idle, .error:
            isRecording = false
            isProcessing = false
        case .listening, .recording:
            isRecording = true
            isProcessing = false
        case .processing:
            isRecording = false
            isProcessing = true
        }
    }

    // MARK: - Private

    private func updateIcon() {
        guard let button = statusItem?.button else { return }

        let tintColor: NSColor

        if isRecording {
            tintColor = .systemRed
            startRecordingAnimation()
        } else if isProcessing {
            tintColor = .systemOrange
            stopRecordingAnimation()
        } else {
            // Use nil for idle state to let the system handle template image coloring
            // This ensures proper appearance in both Light and Dark menu bar
            tintColor = .secondaryLabelColor
            stopRecordingAnimation()
        }

        let config = NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        if let image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "WhisperOnDevice")?
            .withSymbolConfiguration(config) {
            // Make it a template image for proper menu bar appearance adaptation
            image.isTemplate = !isRecording && !isProcessing
            button.image = image
            button.contentTintColor = (isRecording || isProcessing) ? tintColor : nil
        }
    }

    private func startRecordingAnimation() {
        guard animationTimer == nil else { return }

        isAnimationHighlighted = true
        // Timer callback must dispatch to MainActor since Timer closures aren't actor-isolated
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.isAnimationHighlighted.toggle()
                self.statusItem?.button?.alphaValue = self.isAnimationHighlighted ? 1.0 : 0.5
            }
        }
    }

    private func stopRecordingAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
        statusItem?.button?.alphaValue = 1.0
    }

    private func updateMenuItems() {
        guard let menu = statusItem?.menu,
              let toggleItem = menu.items.first else { return }

        let hotkeyDisplay = SettingsManager.shared.hotkey.displayString
        toggleItem.title = isRecording
            ? "Stop Dictation (\(hotkeyDisplay))"
            : "Start Dictation (\(hotkeyDisplay))"
    }

    @objc private func toggleDictation() {
        coordinator?.toggleRecording()
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
