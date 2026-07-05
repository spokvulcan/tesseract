//
//  TesseractApp.swift
//  tesseract
//

import SwiftUI

/// The app's window scene identifiers.
enum WindowID {
    static let main = "main"
    /// The Onboarding Tour's Welcome Window (see `CONTEXT.md` → Onboarding
    /// tour): the only window presented on a first launch, an ordinary
    /// on-demand window when relaunched from Settings.
    static let onboarding = "onboarding"
}

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
                    openWindow(id: WindowID.main)
                }
            }
    }
}

@main
struct TesseractApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var container = DependencyContainer()
    @State private var selectedNavigation: NavigationItem? = .agent

    init() {
        let args = CommandLine.arguments
        if args.contains("--benchmark") {
            Task { @MainActor in
                do { try await BenchmarkRunner().run() } catch {
                    Log.agent.error("Benchmark failed: \(error)")
                }
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
        } else if args.contains("--prefill-step-benchmark") {
            Self.runHarness("Prefill step benchmark") {
                try await PrefillStepBenchmarkRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--paroquant-vlm-smoke") {
            Self.runHarness("ParoQuant VLM smoke") {
                try await ParoQuantVLMSmokeRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--trace-replay") {
            Self.runHarness("Trace replay") {
                try await TraceReplayRunner(arguments: args).run()
            }
        } else if args.contains("--paged-kv-kernel-gate") {
            Self.runHarness("Paged-KV kernel gate") {
                try await PagedKVKernelGateRunner().run()
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
        Window("Tesseract", id: WindowID.main) {
            ContentView(container: container, selectedNavigation: $selectedNavigation)
                .background {
                    WindowOpenerView(appDelegate: appDelegate)
                }
                .injectCoreDependencies(from: container)
                .focusedSceneValue(
                    \.dictationActions,
                    DictationActions(
                        toggleRecording: { [weak container] in
                            container?.dictationCoordinator.toggleRecording()
                        },
                        clearHistory: { [weak container] in
                            container?.transcriptionHistory.clear()
                        },
                        copyLastTranscription: { [weak container] in
                            container?.transcriptionHistory.copyLatestToPasteboard()
                        },
                        isRecording: container.dictationCoordinator.state == .recording
                            || container.dictationCoordinator.state == .listening,
                        isModelLoaded: container.transcriptionEngine.isModelLoaded,
                        hasHistory: !container.transcriptionHistory.entries.isEmpty
                    )
                )
                .task {
                    await container.setup()
                    appDelegate.setupWithContainer(
                        container, navigationSelection: $selectedNavigation)
                }
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 800, height: 700)
        // First launch belongs to the Welcome Window; the main window arrives
        // at the Handoff (or via the menu bar). Every later launch is normal.
        .defaultLaunchBehavior(isFirstLaunch ? .suppressed : .automatic)
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

        Window("Welcome to Tesseract", id: WindowID.onboarding) {
            OnboardingTourView(container: container)
                .injectCoreDependencies(from: container)
                .injectDictationDependencies(from: container)
                .injectSpeechDependencies(from: container)
                .background {
                    WindowOpenerView(appDelegate: appDelegate)
                }
                .task {
                    // On a first launch this is the only window, so it owns
                    // the (idempotent) container setup and delegate wiring.
                    await container.setup()
                    appDelegate.setupWithContainer(
                        container, navigationSelection: $selectedNavigation)
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .restorationBehavior(.disabled)
        .defaultLaunchBehavior(isFirstLaunch ? .presented : .suppressed)
    }

    /// Read once per scene evaluation; only the value at launch matters for
    /// the two `defaultLaunchBehavior`s above.
    private var isFirstLaunch: Bool {
        !container.settingsManager.hasCompletedOnboarding
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
