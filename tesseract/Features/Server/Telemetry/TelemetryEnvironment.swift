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
//  through here, on top of `StorageEnvironment` (ADR-0073), which keeps a
//  test process in its own scratch directory.
//

import Foundation

nonisolated enum TelemetryEnvironment {
    /// The variable a gate run sets (`scripts/gate.sh`): the process keeps
    /// all its durable telemetry under this folder, so a run's diagnostics
    /// and traces are its own files, apart from the owner's and from any
    /// other run's.
    static let directoryVariable = "TESSERACT_TELEMETRY_DIR"

    /// The durable home for a telemetry component (`"CacheDiagnostics"`,
    /// `"PrefixCacheTraces"`, ...): Application Support in production, the
    /// test process's scratch directory under a test runner, the gate run's
    /// folder when `directoryVariable` names one.
    static func durableDirectory(
        component: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        let root =
            environment[directoryVariable].flatMap { $0.isEmpty ? nil : $0 }
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? StorageEnvironment.applicationSupport
        return root.appendingPathComponent(component, isDirectory: true)
    }
}
