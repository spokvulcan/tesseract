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
    var onOpenWindow: (() -> Void)?

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

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            // No visible windows - show the main window
            showMainWindow()
        }
        return true
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
        // Activate the app first to bring it to the foreground
        NSApp.activate(ignoringOtherApps: true)

        // Find content windows (excluding panels like status bar menus)
        let contentWindows = NSApp.windows.filter { window in
            !(window is NSPanel) && window.canBecomeMain
        }

        if let visibleWindow = contentWindows.first(where: { $0.isVisible }) {
            // If there's already a visible window, bring it to front
            visibleWindow.makeKeyAndOrderFront(nil)
        } else if let hiddenWindow = contentWindows.first {
            // If there's a hidden window, show it
            hiddenWindow.makeKeyAndOrderFront(nil)
        } else {
            // No windows exist - need to create one via SwiftUI
            // Use the callback provided by the App struct
            onOpenWindow?()
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
