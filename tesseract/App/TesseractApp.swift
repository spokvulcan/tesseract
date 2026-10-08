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
    /// The Markdown Gallery (see `CONTEXT.md`): the living style reference
    /// for chat markdown, reached from the Window menu.
    static let markdownGallery = "markdown-gallery"
    /// The Profile: everything Jarvis knows about the owner, editable.
    static let profile = "profile"
}

/// Bridges the SwiftUI `openWindow`/`openSettings` environment actions to the
/// AppDelegate. Needed because the actions are only available inside a SwiftUI
/// view hierarchy; the captured actions keep working after the hosting window
/// closes (the existing `showMainWindow` fallback relies on this).
private struct WindowOpenerView: View {
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings
    let appDelegate: AppDelegate

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onAppear {
                appDelegate.onOpenWindow = { [openWindow] in
                    openWindow(id: WindowID.main)
                }
                appDelegate.onOpenSettings = { [openSettings] in
                    openSettings()
                }
            }
    }
}

@main
struct TesseractApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var container = DependencyContainer()
    @State private var selectedNavigation: NavigationItem? = .today

    /// Launch arguments that select a headless harness run (dispatched in
    /// `init`). Harness instances are exempt from the single-instance guard
    /// so a bench can run while an interactive instance is open.
    static let harnessFlags: Set<String> = [
        "--paro-parity-bench", "--snapshot-bench", "--ssd-read-bench", "--warm-parity-bench",
        "--prefix-detect-bench",
        "--tokenize-cache-bench", "--agent-cpu-bench", "--dflash2-bench",
        "--prefix-cache-e2e", "--benchmark", "--hybrid-cache-correctness",
        "--prefill-step-benchmark", "--paroquant-vlm-smoke",
        "--prepared-checkpoint-parity", "--trace-replay",
        "--rotated-checkpoint-parity", "--turboquant-bench",
    ]
    static var isHarnessLaunch: Bool {
        CommandLine.arguments.contains { harnessFlags.contains($0) }
    }

    init() {
        // In-place dynamic slice updates (mlx fork C9): DFlash2 writes its
        // verify rows and the drafter's block K/V into cache slack rows whose
        // readers all precede the write on the stream; without the flag every
        // such write copies the whole store. The bench sets the same flag;
        // overwrite 0 keeps an explicit environment override.
        setenv("MLX_DYNSLICE_INPLACE", "1", 0)
        let args = CommandLine.arguments
        // `--paro-parity-bench` precedes `--benchmark`: scripts/bench.sh always
        // passes `--benchmark`, so the perf-ruler/gate harness is opt-in via
        // extra args forwarded by bench.sh (`bench.sh quick --model X
        // --paro-parity-bench`). Plain `--benchmark` runs are unaffected.
        if args.contains("--paro-parity-bench") {
            Self.runHarness("PARO parity bench", logSubdirectory: "paro-parity-bench") {
                try await ParoParityBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--turboquant-bench") {
            Self.runHarness("TurboQuant bench", logSubdirectory: "turboquant-bench") {
                try await TurboQuantBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--ssd-read-bench") {
            Self.runHarness("SSD read bench", logSubdirectory: nil) {
                try await SSDReadBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--warm-parity-bench") {
            Self.runHarness("Warm Body parity bench", logSubdirectory: nil) {
                try await WarmBodyParityBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--snapshot-bench") {
            Self.runHarness("Snapshot bench", logSubdirectory: "snapshot-bench") {
                try await SnapshotBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--prefix-detect-bench") {
            Self.runHarness("Prefix detect bench", logSubdirectory: "prefix-detect-bench") {
                try await PrefixDetectBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--tokenize-cache-bench") {
            Self.runHarness("Tokenize cache bench", logSubdirectory: "tokenize-cache-bench") {
                try await TokenizeCacheBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--agent-cpu-bench") {
            Self.runHarness("Agent CPU bench", logSubdirectory: "agent-cpu-bench") {
                try await AgentCpuBenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--dflash2-bench") {
            Self.runHarness("DFlash2 bench", logSubdirectory: nil) {
                try await DFlash2BenchRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--prefix-cache-e2e") {
            Self.runHarness("Prefix cache E2E", logSubdirectory: "prefix-cache-e2e") {
                try await PrefixCacheE2ERunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--benchmark") {
            Task { @MainActor in
                do { try await BenchmarkRunner().run() } catch {
                    Self.logHarnessFailure("Benchmark failed: \(error)", logSubdirectory: nil)
                }
                exit(0)
            }
        } else if args.contains("--hybrid-cache-correctness") {
            Self.runHarness(
                "Hybrid cache correctness", logSubdirectory: "hybrid-cache-correctness"
            ) {
                try await HybridCacheCorrectnessRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--prefill-step-benchmark") {
            Self.runHarness("Prefill step benchmark", logSubdirectory: "prefill-step-benchmark") {
                try await PrefillStepBenchmarkRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--paroquant-vlm-smoke") {
            Self.runHarness("ParoQuant VLM smoke", logSubdirectory: "paroquant-vlm-smoke") {
                try await ParoQuantVLMSmokeRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--prepared-checkpoint-parity") {
            Self.runHarness(
                "Prepared Checkpoint parity", logSubdirectory: "prepared-checkpoint-parity"
            ) {
                try await PreparedCheckpointParityRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--rotated-checkpoint-parity") {
            Self.runHarness(
                "Rotated Ternary Checkpoint parity", logSubdirectory: "rotated-checkpoint-parity"
            ) {
                try await RotatedCheckpointParityRunner(runner: BenchmarkRunner()).run()
            }
        } else if args.contains("--trace-replay") {
            Self.runHarness("Trace replay", logSubdirectory: "trace-replay") {
                try await TraceReplayRunner(arguments: args).run()
            }
        }
    }

    /// Spawn a `@MainActor` task that runs a loaded-model verification
    /// harness, exits 0 on success, exits 1 on failure (after logging).
    /// `logSubdirectory` is where the harness keeps its `latest.log` under
    /// the benchmark output directory (`nil`: the directory itself).
    @MainActor
    private static func runHarness(
        _ label: String,
        logSubdirectory: String?,
        run: @MainActor @escaping () async throws -> Void
    ) {
        Task { @MainActor in
            do {
                try await run()
                exit(0)
            } catch {
                logHarnessFailure("\(label) failed: \(error)", logSubdirectory: logSubdirectory)
                exit(1)
            }
        }
    }

    /// Log a harness failure and append it to the harness's `latest.log`,
    /// the file the scripts tail; a harness that failed before opening its
    /// log gets one.
    private static func logHarnessFailure(_ message: String, logSubdirectory: String?) {
        Log.agent.error("\(message)")
        var directory = BenchmarkConfig.fromCommandLine().outputDir
        if let logSubdirectory { directory.append(path: logSubdirectory) }
        let url = directory.appending(path: "latest.log")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("[harness] \(message)\n".utf8))
    }

    var body: some Scene {
        Window("Tesseract", id: WindowID.main) {
            ContentView(container: container, selectedNavigation: $selectedNavigation)
                .background {
                    WindowOpenerView(appDelegate: appDelegate)
                }
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
                        isRecording: container.dictationCoordinator.state == .recording,
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
        // The test host opens no window at all, restored or not (ADR-0073).
        .defaultLaunchBehavior(
            isFirstLaunch || ProcessEnvironment.isRunningTests ? .suppressed : .automatic
        )
        .restorationBehavior(ProcessEnvironment.isRunningTests ? .disabled : .automatic)
        .commands {
            DictationCommands()
        }

        // The native Settings window (map #211): ⌘, and the App menu reach it
        // through the standard system command; the status-bar "Settings…"
        // arrives via the AppDelegate's bridged `openSettings`.
        Settings {
            SettingsWindowView()
                .injectDependencies(from: container)
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
        // The test host's settings are in memory, so it always reads as a
        // first launch; the tour would start model downloads (ADR-0073).
        .defaultLaunchBehavior(
            isFirstLaunch && !ProcessEnvironment.isRunningTests ? .presented : .suppressed
        )

        // The Markdown Gallery: on-demand singleton, never presented at
        // launch; the system Window menu lists it automatically.
        Window("Markdown Gallery", id: WindowID.markdownGallery) {
            MarkdownGalleryView()
        }
        .defaultSize(width: 1280, height: 860)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        // The Profile: on-demand singleton, opened from Today and Settings.
        Window("Profile", id: WindowID.profile) {
            ProfileView()
                .environment(container.profileStore)
        }
        .defaultSize(width: 620, height: 560)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)
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
