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
    /// The durable home for a telemetry component (`"CacheDiagnostics"`,
    /// `"PrefixCacheTraces"`, ...): Application Support in production, the
    /// test process's scratch directory under a test runner.
    static func durableDirectory(component: String) -> URL {
        StorageEnvironment.applicationSupport.appendingPathComponent(
            component, isDirectory: true)
    }
}
