//
//  TesseractApp.swift
//  tesseract
//

import SwiftUI

/// Helper view that captures the openWindow environment action and provides it to the AppDelegate
private struct WindowOpenerView: View {
    @Environment(\.openWindow) private var openWindow
    let appDelegate: AppDelegate

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onAppear {
                // Provide the open window callback with safety check
                appDelegate.onOpenWindow = { [openWindow] in
                    let hasContentWindow = NSApp.windows.contains { window in
                        !(window is NSPanel) && window.canBecomeMain
                    }
                    if !hasContentWindow {
                        openWindow(id: "main")
                    }
                }

                // Deduplicate windows on launch (handles Xcode debug state restoration)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    let contentWindows = NSApp.windows.filter { window in
                        !(window is NSPanel) && window.canBecomeMain
                    }
                    // Keep only the first window if duplicates exist
                    if contentWindows.count > 1 {
                        for window in contentWindows.dropFirst() {
                            window.close()
                        }
                    }
                }
            }
    }
}

@main
struct TesseractApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var container = DependencyContainer()
    @State private var showOnboarding = false
    @State private var selectedNavigation: NavigationItem? = .dictation

    init() {
        if CommandLine.arguments.contains("--benchmark") {
            Task { @MainActor in
                let runner = BenchmarkRunner()
                do {
                    try await runner.run()
                } catch {
                    Log.agent.error("Benchmark failed: \(error)")
                }
                exit(0)
            }
        }
    }

    var body: some Scene {
        WindowGroup(id: "main") {
            ContentView(
                coordinator: container.dictationCoordinator,
                transcriptionEngine: container.transcriptionEngine,
                history: container.transcriptionHistory,
                permissionsManager: container.permissionsManager,
                audioCapture: container.audioCaptureEngine,
                speechCoordinator: container.speechCoordinator,
                speechEngine: container.speechEngine,
                agentCoordinator: container.agentCoordinator,
                agentEngine: container.agentEngine,
                agentConversationStore: container.agentConversationStore,
                imageGenEngine: container.imageGenEngine,
                zimageGenEngine: container.zimageGenEngine,
                selectedNavigation: $selectedNavigation
            )
            .background {
                WindowOpenerView(appDelegate: appDelegate)
            }
            .environmentObject(container)
            .environmentObject(container.modelDownloadManager)
            .focusedSceneValue(\.dictationActions, DictationActions(
                toggleRecording: { [weak container] in
                    container?.dictationCoordinator.toggleRecording()
                },
                clearHistory: { [weak container] in
                    container?.transcriptionHistory.clear()
                },
                copyLastTranscription: { [weak container] in
                    container?.transcriptionHistory.copyLatestToPasteboard()
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
                    isPresented: $showOnboarding
                )
            }
            .onReceive(NotificationCenter.default.publisher(for: .showOnboarding)) { _ in
                showOnboarding = true
            }
        }
        // Prevent WindowGroup from creating multiple windows via external events
        .handlesExternalEvents(matching: Set<String>())
        .windowResizability(.contentMinSize)
        .defaultSize(width: 800, height: 700)
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
    var copyLastTranscription: () -> Void
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

            Button("Copy Last Transcription") {
                actions?.copyLastTranscription()
            }
            .disabled(actions?.hasHistory != true)

            Button("Clear History") {
                actions?.clearHistory()
            }
            .keyboardShortcut(.delete, modifiers: [.command])
            .disabled(actions?.hasHistory != true)
        }
    }
}
