//
//  TelemetryEnvironment.swift
//  tesseract
//
//  Routing seam for durable telemetry homes (issue #159). Test processes
//  host the full app (the test scheme runs suites in parallel against the
//  app host), so any suite that spins up the server stack used to append
//  toy-model records into the *production* Application Support telemetry
//  files — the `toy/model` records and torn lines found in the 2026-07-05
//  trace-file forensics. Every durable telemetry default directory routes
//  through here; under a test runner it diverts to a per-process temp
//  directory (per-process so the parallel twin test processes never share
//  a file either).
//

import Foundation

nonisolated enum TelemetryEnvironment {
    /// True when the process is a test host. Mirrors `AppDelegate`'s
    /// launch-time check; the extra keys cover runners that set only the
    /// session/bundle variants.
    static let isRunningTests: Bool = {
        let env = ProcessInfo.processInfo.environment
        return env["XCTestConfigurationFilePath"] != nil
            || env["XCTestBundlePath"] != nil
            || env["XCTestSessionIdentifier"] != nil
    }()

    /// The durable home for a telemetry component (`"CacheDiagnostics"`,
    /// `"PrefixCacheTraces"`, ...): Application Support in production, an
    /// isolated per-process temp directory under a test runner.
    static func durableDirectory(component: String) -> URL {
        guard !isRunningTests else {
            return FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "TesseractTestTelemetry-\(ProcessInfo.processInfo.processIdentifier)",
                    isDirectory: true
                )
                .appendingPathComponent(component, isDirectory: true)
        }
        let base =
            FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent(component, isDirectory: true)
    }
}
