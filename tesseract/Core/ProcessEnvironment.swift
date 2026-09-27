//
//  ProcessEnvironment.swift
//  tesseract
//
//  Process-level launch facts shared across the app. The single "are we
//  a test host?" definition — `AppDelegate` (skip app bootstrapping) and
//  `StorageEnvironment` (keep test runs off the owner's data, ADR-0073)
//  must agree, and the extra keys cover runners that set only the
//  session/bundle variants.
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
}
