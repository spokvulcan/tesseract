//
//  WhisperOnDeviceApp.swift
//  whisper-on-device
//

import SwiftUI

@main
struct WhisperOnDeviceApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var container = DependencyContainer()
    @State private var showOnboarding = false
    @State private var selectedNavigation: NavigationItem? = .dictation

    var body: some Scene {
        WindowGroup {
            ContentView(
                coordinator: container.dictationCoordinator,
                transcriptionEngine: container.transcriptionEngine,
                history: container.transcriptionHistory,
                permissionsManager: container.permissionsManager,
                audioCapture: container.audioCaptureEngine,
                selectedNavigation: $selectedNavigation
            )
            .environmentObject(container)
            .focusedSceneValue(\.dictationActions, DictationActions(
                toggleRecording: { [weak container] in
                    container?.dictationCoordinator.toggleRecording()
                },
                clearHistory: { [weak container] in
                    container?.transcriptionHistory.clear()
                },
                isRecording: container.dictationCoordinator.state == .recording ||
                             container.dictationCoordinator.state == .listening,
                isModelLoaded: container.transcriptionEngine.isModelLoaded,
                hasHistory: !container.transcriptionHistory.entries.isEmpty
            ))
            .task {
                await container.setup()
                appDelegate.setupWithContainer(container, navigationSelection: $selectedNavigation)

                // Show onboarding if needed
                if !SettingsManager.shared.hasCompletedOnboarding {
                    showOnboarding = true
                }
            }
            .sheet(isPresented: $showOnboarding) {
                OnboardingView(
                    permissionsManager: container.permissionsManager,
                    modelManager: container.modelManager,
                    isPresented: $showOnboarding
                )
            }
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 700, height: 550)
        .commands {
            CommandGroup(replacing: .newItem) {}  // Remove "New Window" command

            CommandGroup(replacing: .appSettings) {
                Button("Settings...") {
                    selectedNavigation = .general
                    NSApp.activate(ignoringOtherApps: true)
                }
                .keyboardShortcut(",", modifiers: .command)
            }

            DictationCommands()
        }
    }
}

// MARK: - Focused Value for Dictation Commands

struct DictationActions {
    var toggleRecording: () -> Void
    var clearHistory: () -> Void
    var isRecording: Bool
    var isModelLoaded: Bool
    var hasHistory: Bool
}

struct DictationActionsKey: FocusedValueKey {
    typealias Value = DictationActions
}

extension FocusedValues {
    var dictationActions: DictationActions? {
        get { self[DictationActionsKey.self] }
        set { self[DictationActionsKey.self] = newValue }
    }
}

// MARK: - Dictation Menu Commands

struct DictationCommands: Commands {
    @FocusedValue(\.dictationActions) private var actions

    var body: some Commands {
        CommandMenu("Dictation") {
            Button(actions?.isRecording == true ? "Stop Recording" : "Start Recording") {
                actions?.toggleRecording()
            }
            .keyboardShortcut("d", modifiers: [.command, .shift])
            .disabled(actions?.isModelLoaded != true)

            Divider()

            Button("Clear History") {
                actions?.clearHistory()
            }
            .keyboardShortcut(.delete, modifiers: [.command])
            .disabled(actions?.hasHistory != true)
        }
    }
}
