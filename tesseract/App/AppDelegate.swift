//
//  AppDelegate.swift
//  tesseract
//

import Foundation
import AppKit
import SwiftUI
import Combine
import MLX

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var cancellables = Set<AnyCancellable>()
    var container: DependencyContainer?
    var menuBarManager: MenuBarManager?
    private weak var trackedMainWindow: NSWindow?
    private var navigationSelection: Binding<NavigationItem?>?
    var onOpenWindow: (() -> Void)?
    private var hasSetupWithContainer = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Ensure only one instance of the app runs at a time
        ensureSingleInstance()

        // Setup window lifecycle tracking
        setupWindowTracking()

        // Setup will be done by the App struct after container is created
    }

    private func setupWindowTracking() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowDidBecomeKey(_:)),
            name: NSWindow.didBecomeKeyNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowWillClose(_:)),
            name: NSWindow.willCloseNotification,
            object: nil
        )
    }

    @objc private func windowDidBecomeKey(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              !(window is NSPanel),
              window.canBecomeMain else { return }
        trackedMainWindow = window
    }

    @objc private func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              window === trackedMainWindow else { return }
        trackedMainWindow = nil
    }

    private func ensureSingleInstance() {
        let bundleID = Bundle.main.bundleIdentifier ?? "com.tesseract.app"
        let runningApps = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)

        // If more than one instance is running (including this one), terminate this instance
        if runningApps.count > 1 {
            // Activate the other instance
            if let otherInstance = runningApps.first(where: { $0 != NSRunningApplication.current }) {
                otherInstance.activate()
            }
            // Terminate this instance
            NSApp.terminate(nil)
        }
    }

    func setupWithContainer(_ container: DependencyContainer, navigationSelection: Binding<NavigationItem?>) {
        // Prevent duplicate setup from multiple window instances
        guard !hasSetupWithContainer else { return }
        hasSetupWithContainer = true

        self.container = container
        self.navigationSelection = navigationSelection

        // Setup menu bar
        menuBarManager = MenuBarManager()
        menuBarManager?.coordinator = container.dictationCoordinator
        menuBarManager?.history = container.transcriptionHistory
        menuBarManager?.speechCoordinator = container.speechCoordinator
        menuBarManager?.onShowMainWindow = { [weak self] in
            self?.showMainWindow()
        }
        menuBarManager?.onShowSettings = { [weak self] in
            self?.navigateToSettings()
        }
        menuBarManager?.onTalkToAgent = { [weak self] in
            self?.navigateToAgent()
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
            // Defer to next run loop iteration to let any SwiftUI window become visible first.
            // This prevents duplicate window creation from the race condition where both
            // SwiftUI and our code try to create windows simultaneously.
            DispatchQueue.main.async { [weak self] in
                let hasVisibleWindow = NSApp.windows.contains {
                    !($0 is NSPanel) && $0.canBecomeMain && $0.isVisible
                }
                if !hasVisibleWindow {
                    self?.showMainWindow()
                }
            }
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
        // Stop TTS generation and unload the model before exit() destroys MLX's
        // Metal device singleton. Pending GPU completion handlers would otherwise
        // crash trying to lock the destroyed mutex in Device::end_encoding.
        container?.speechCoordinator.stop()
        container?.speechEngine.unloadModel()

        // Drain any in-flight Metal command buffers so no completion handlers fire
        // after exit() destroys C++ static objects (including MLX's device mutex).
        Stream.gpu.synchronize()

        container?.hotkeyManager.stopListening()
        menuBarManager?.teardownMenuBar()
    }

    // MARK: - Window Management

    func showMainWindow() {
        // First check tracked window (fastest path)
        if let tracked = trackedMainWindow {
            NSApp.activate(ignoringOtherApps: true)
            tracked.makeKeyAndOrderFront(nil)
            return
        }

        // Fallback: find content windows (excluding panels like status bar menus)
        let contentWindows = NSApp.windows.filter { window in
            !(window is NSPanel) && window.canBecomeMain
        }

        // If we have any content window (visible or hidden), use it
        if let existingWindow = contentWindows.first {
            NSApp.activate(ignoringOtherApps: true)
            existingWindow.makeKeyAndOrderFront(nil)
            trackedMainWindow = existingWindow
            return
        }

        // No windows exist - need to create one via SwiftUI
        NSApp.activate(ignoringOtherApps: true)
        onOpenWindow?()
    }

    func navigateToSettings() {
        navigationSelection?.wrappedValue = .general
        showMainWindow()
    }

    func navigateToAgent() {
        navigationSelection?.wrappedValue = .agent
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
