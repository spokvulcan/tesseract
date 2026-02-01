//
//  AppDelegate.swift
//  whisper-on-device
//

import Foundation
import AppKit
import SwiftUI
import Combine

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var cancellables = Set<AnyCancellable>()
    var container: DependencyContainer?
    var menuBarManager: MenuBarManager?
    var mainWindow: NSWindow?
    private var navigationSelection: Binding<NavigationItem?>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Setup will be done by the App struct after container is created
    }

    func setupWithContainer(_ container: DependencyContainer, navigationSelection: Binding<NavigationItem?>) {
        self.container = container
        self.navigationSelection = navigationSelection

        // Setup menu bar
        menuBarManager = MenuBarManager()
        menuBarManager?.coordinator = container.dictationCoordinator
        menuBarManager?.onShowMainWindow = { [weak self] in
            self?.showMainWindow()
        }
        menuBarManager?.onShowSettings = { [weak self] in
            self?.navigateToSettings()
        }
        menuBarManager?.onQuit = {
            NSApp.terminate(nil)
        }
        menuBarManager?.setupMenuBar()

        // Subscribe to dictation state changes
        container.dictationCoordinator.$state
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                // Ensure MainActor isolation for accessing @MainActor-isolated properties
                Task { @MainActor in
                    self?.menuBarManager?.updateState(from: state)
                }
            }
            .store(in: &cancellables)

        // Handle dock visibility
        updateDockVisibility()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Keep running for menu bar
        return false
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        // Could preload model here if needed
    }

    func applicationWillResignActive(_ notification: Notification) {
        // Could unload model on memory pressure
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Cleanup
        container?.hotkeyManager.stopListening()
        menuBarManager?.teardownMenuBar()
    }

    // MARK: - Window Management

    func showMainWindow() {
        if let window = mainWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        } else {
            // Window will be created by SwiftUI
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func navigateToSettings() {
        navigationSelection?.wrappedValue = .general
        showMainWindow()
    }

    private func updateDockVisibility() {
        if SettingsManager.shared.showInDock {
            NSApp.setActivationPolicy(.regular)
        } else {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
