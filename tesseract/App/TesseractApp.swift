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

    init() {
        // In-place dynamic slice updates (mlx fork C9): DFlash2 writes its
        // verify rows and the drafter's block K/V into cache slack rows whose
        // readers all precede the write on the stream; without the flag every
        // such write copies the whole store. The bench sets the same flag;
        // overwrite 0 keeps an explicit environment override.
        setenv("MLX_DYNSLICE_INPLACE", "1", 0)
        // A harness launch runs its harness and exits; it opens no window
        // and runs none of the owner's background work (ADR-0095).
        LaunchMode.current.harness?.start()
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
        // The Launch Mode picks the window a launch opens: the owner's first
        // launch belongs to the Welcome Window (the main window arrives at the
        // Handoff, or via the menu bar), every later one opens this. A test
        // host and a harness open none, restored or not (ADR-0073, ADR-0095).
        .defaultLaunchBehavior(launchWindow == .main ? .automatic : .suppressed)
        .restorationBehavior(launchWindow == nil ? .disabled : .automatic)
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
        .defaultLaunchBehavior(launchWindow == .welcome ? .presented : .suppressed)

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
    /// the `defaultLaunchBehavior`s above.
    private var launchWindow: LaunchMode.LaunchWindow? {
        LaunchMode.current.launchWindow(
            onboarded: container.settingsManager.hasCompletedOnboarding)
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
