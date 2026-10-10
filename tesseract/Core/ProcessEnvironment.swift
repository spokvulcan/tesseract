//
//  ProcessEnvironment.swift
//  tesseract
//
//  Process-level launch facts shared across the app. The single "are we
//  a test host?" definition — `AppDelegate` (skip app bootstrapping) and
//  `StorageEnvironment` (keep test runs off the owner's data, ADR-0073)
//  must agree, and the extra keys cover runners that set only the
//  session/bundle variants. A scratch launch keeps off the owner's data the
//  same way, with its windows and services running.
//

import Foundation

nonisolated enum ProcessEnvironment {
    /// True when the process is a test host.
    static let isRunningTests: Bool = {
        let env = ProcessInfo.processInfo.environment
        return env["XCTestConfigurationFilePath"] != nil
            || env["XCTestBundlePath"] != nil
            || env["XCTestSessionIdentifier"] != nil
    }()

    /// A scratch launch (`scripts/dev.sh dev --scratch`): the whole app, its
    /// windows, models and services, for trying a change by hand or by an
    /// agent.
    static let isScratchLaunch: Bool =
        ProcessInfo.processInfo.environment["TESSERACT_SCRATCH_DATA"] == "1"

    /// The process keeps off the owner's data (ADR-0073): a test host, or a
    /// scratch launch. Storage, settings, the Agenda, the Profile and day
    /// state, the browser profile and the OS notification center follow it.
    static let usesScratchData: Bool = isRunningTests || isScratchLaunch
}
