//
//  LaunchMode.swift
//  tesseract
//
//  The **Launch Mode** (ADR-0095, extending ADR-0073): what this process is,
//  decided once from its arguments and environment — the owner's launch, a
//  scratch launch, the unit tests' host, or a headless harness — and the one
//  place that answers what such a process may do. Callers ask an answer
//  ("keep off the owner's data?", "open a window at launch?"), never the
//  arguments, the environment or the mode's name.
//
//  The decision and its answers are pure and shared with the iPhone app.
//  `LaunchMode.current` is the Mac's (`App/Harness.swift`): it knows which
//  flags name a harness. Shared code reads `StorageEnvironment`, whose one
//  answer the environment alone decides.
//

import Foundation

nonisolated enum LaunchMode: Equatable, Sendable {
    /// The owner's app: their data, their windows, their background work.
    case owner
    /// A scratch launch (`scripts/dev.sh dev --scratch`): the whole app, its
    /// windows, models and services, on scratch storage, for trying a change
    /// by hand or by an agent.
    case scratch
    /// The unit tests' host: the app the test runner launches to run the
    /// suites in.
    case testHost
    /// A headless harness run (a bench, a parity check, a trace replay),
    /// named by the flag that selects it.
    case harness(flag: String)

    /// The window a launch opens.
    enum LaunchWindow: Equatable, Sendable {
        case main
        /// The Onboarding Tour's Welcome window.
        case welcome
    }

    /// The variables the test runner sets in its host; the session variant
    /// covers runners that set only that one.
    static let testRunnerVariables = [
        "XCTestConfigurationFilePath", "XCTestBundlePath", "XCTestSessionIdentifier",
    ]

    /// The variable a scratch launch sets (`scripts/dev.sh … --scratch`).
    static let scratchVariable = "TESSERACT_SCRATCH_DATA"

    /// Decides the mode. The environment decides first: the test runner's
    /// variables make a test host, the scratch variable a scratch launch.
    /// Only then do the arguments name a harness: the first of
    /// `harnessFlags`, in their order, that the arguments carry. Anything else
    /// is the owner's launch.
    static func decide(
        arguments: [String], environment: [String: String], harnessFlags: [String]
    ) -> LaunchMode {
        if testRunnerVariables.contains(where: { environment[$0] != nil }) { return .testHost }
        if environment[scratchVariable] == "1" { return .scratch }
        if let flag = harnessFlags.first(where: { arguments.contains($0) }) {
            return .harness(flag: flag)
        }
        return .owner
    }

    // MARK: - Answers

    /// Keeps off the owner's data (ADR-0073): storage in a scratch folder,
    /// settings in memory, the in-memory Agenda, no Profile or day-state file,
    /// a non-persistent browser profile. A harness keeps the owner's data, as
    /// the owner's launch does, so the environment alone decides this.
    var keepsOffOwnersData: Bool {
        switch self {
        case .scratch, .testHost: true
        case .owner, .harness: false
        }
    }

    /// Runs the owner's background work: the hotkeys, the HTTP server, the
    /// Companion, the speech models' loads and prewarms, the MCP client. A
    /// harness owns the GPU and the model it loads, and a test host runs
    /// suites, not the app.
    var runsOwnersBackgroundWork: Bool {
        switch self {
        case .owner, .scratch: true
        case .testHost, .harness: false
        }
    }

    /// The window to open at launch, or nil for none and none restored. The
    /// owner's first launch belongs to the Welcome window; every later one
    /// opens the main window. A scratch launch opens the main window: its
    /// settings start at their defaults, and the tour would start model
    /// downloads.
    func launchWindow(onboarded: Bool) -> LaunchWindow? {
        switch self {
        case .owner: onboarded ? .main : .welcome
        case .scratch: .main
        case .testHost, .harness: nil
        }
    }

    /// Posts notifications to the OS. A scratch launch shares the installed
    /// app's notification center (one bundle id), so it keeps its nudges in
    /// memory.
    var postsNotifications: Bool { self == .owner }

    /// Quits when another instance is running. A test host and a harness run
    /// beside the owner's app.
    var enforcesSingleInstance: Bool {
        switch self {
        case .owner, .scratch: true
        case .testHost, .harness: false
        }
    }

    /// Moves or cleans the owner's data at launch: the sandbox model
    /// migration, retired checkpoints, the Companion's retired data. Only the
    /// owner's launch: a harness must not change what it measures against.
    var migratesOwnersData: Bool { self == .owner }

    /// Plays the app's own sounds. The test host drives hundreds of takes
    /// and nobody listens.
    var playsSounds: Bool { self != .testHost }
}
