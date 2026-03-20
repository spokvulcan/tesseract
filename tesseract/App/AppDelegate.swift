//
//  AppDelegate.swift
//  tesseract
//

import Foundation
import AppKit
import Observation
import SwiftUI
import MLX
import UserNotifications

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var observationTask: Task<Void, Never>?
    var container: DependencyContainer?
    var menuBarManager: MenuBarManager?
    private weak var trackedMainWindow: NSWindow?
    private var navigationSelection: Binding<NavigationItem?>?
    var onOpenWindow: (() -> Void)?
    /// Stores the session ID from a notification click. Survives cold launch —
    /// TesseractApp reads it after setup and forwards to schedulingService.
    var pendingBackgroundSessionId: UUID?
    private var hasSetupWithContainer = false
    private var isRunningUnderTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Ensure only one instance of the app runs at a time
        ensureSingleInstance()

        // Setup window lifecycle tracking
        setupWindowTracking()

        // Register as notification delegate so we handle clicks and foreground presentation
        UNUserNotificationCenter.current().delegate = self

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
        if isRunningUnderTests {
            return
        }

        let bundleID = Bundle.main.bundleIdentifier ?? "app.tesseract.agent"
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
        menuBarManager = MenuBarManager(settings: container.settingsManager)
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
        observationTask = Task { [weak self] in
            guard let container = self?.container else { return }
            for await state in Observations({ container.dictationCoordinator.state }) {
                self?.menuBarManager?.updateState(from: state)
            }
        }

        // Apply initial dock visibility (didSet doesn't fire during SettingsManager.init)
        container.settingsManager.applyDockVisibility()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Keep running for menu bar
        return false
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
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

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Cancel subscriptions synchronously (we're on MainActor)
        container?.schedulingService.cancelSubscriptions()

        // Persist interrupted task state asynchronously, then allow termination.
        // We use .terminateLater so MainActor stays free for the @MainActor-isolated
        // ScheduledTaskStore.saveRun call inside persistInterruptedTask.
        let actor = container?.schedulingActor
        guard actor != nil else { return .terminateNow }

        Task {
            await actor?.stopPolling()
            await actor?.stopHeartbeat()
            await actor?.persistInterruptedTask()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        observationTask?.cancel()
        observationTask = nil

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
}

// MARK: - UNUserNotificationCenterDelegate

extension AppDelegate: UNUserNotificationCenterDelegate {

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let userInfo = response.notification.request.content.userInfo
        let sessionIdString = userInfo[NotificationService.sessionIdKey] as? String

        await MainActor.run {
            navigateToAgent()

            if let sessionIdString, let sessionId = UUID(uuidString: sessionIdString) {
                if let service = container?.schedulingService {
                    // Warm launch: forward directly, no need to store on AppDelegate
                    service.pendingBackgroundSessionId = sessionId
                } else {
                    // Cold launch: stash for TesseractApp to forward after setup
                    pendingBackgroundSessionId = sessionId
                }
            }
        }
    }
}
