//
//  TesseractApp.swift
//  tesseract
//

import SwiftUI

/// Bridges the SwiftUI `openWindow` environment action to the AppDelegate.
/// Needed because `@Environment(\.openWindow)` is only available inside a SwiftUI view hierarchy.
private struct WindowOpenerView: View {
    @Environment(\.openWindow) private var openWindow
    let appDelegate: AppDelegate

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onAppear {
                appDelegate.onOpenWindow = { [openWindow] in
                    openWindow(id: "main")
                }
            }
    }
}

@main
struct TesseractApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var container = DependencyContainer()
    @State private var showOnboarding = false
    @State private var selectedNavigation: NavigationItem? = .agent

    init() {
        let args = CommandLine.arguments
        if args.contains("--benchmark") {
            Task { @MainActor in
                do { try await BenchmarkRunner().run() }
                catch { Log.agent.error("Benchmark failed: \(error)") }
                exit(0)
            }
        } else if args.contains("--prefix-cache-e2e") {
            Self.runHarness("Prefix cache E2E") {
                try await PrefixCacheE2ERunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--hybrid-cache-correctness") {
            Self.runHarness("Hybrid cache correctness") {
                try await HybridCacheCorrectnessRunner(runner: BenchmarkRunner()).run()
            }
        }
    }

    /// Spawn a `@MainActor` task that runs a loaded-model verification
    /// harness, exits 0 on success, exits 1 on failure (after logging).
    @MainActor
    private static func runHarness(
        _ label: String,
        run: @MainActor @escaping () async throws -> Void
    ) {
        Task { @MainActor in
            do {
                try await run()
                exit(0)
            } catch {
                Log.agent.error("\(label) failed: \(error)")
                exit(1)
            }
        }
    }

    var body: some Scene {
        Window("Tesseract", id: "main") {
            ContentView(container: container, selectedNavigation: $selectedNavigation)
            .background {
                WindowOpenerView(appDelegate: appDelegate)
            }
            .injectCoreDependencies(from: container)
            .environment(container.schedulingService)
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

                // Forward any pending notification deep-link from cold launch
                if let sessionId = appDelegate.pendingBackgroundSessionId {
                    appDelegate.pendingBackgroundSessionId = nil
                    container.schedulingService.pendingBackgroundSessionId = sessionId
                }

                // Show onboarding if needed
                if !container.settingsManager.hasCompletedOnboarding {
                    showOnboarding = true
                }
            }
            .sheet(isPresented: $showOnboarding) {
                OnboardingView(isPresented: $showOnboarding)
                    .injectCoreDependencies(from: container)
            }
            .onReceive(NotificationCenter.default.publisher(for: .showOnboarding)) { _ in
                showOnboarding = true
            }
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 800, height: 700)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings...") {
                    selectedNavigation = .general
                    appDelegate.showMainWindow()
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
