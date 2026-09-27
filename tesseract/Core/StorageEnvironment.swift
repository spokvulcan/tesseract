//
//  StorageEnvironment.swift
//  tesseract
//
//  Where the app keeps what it stores (ADR-0073). In production these are the
//  owner's Application Support and Caches folders. Under a test runner they are
//  folders inside one scratch directory per test process, so no test run reads
//  or changes the owner's data. Per process, like the telemetry homes (#159):
//  the scheme runs suites in parallel processes against the same app host.
//
//  Every default storage location resolves through here, except the model
//  folder (`ModelDownloadManager.modelStorageURL`): suites load installed
//  models on purpose, and nothing under a test runner downloads one.
//

import Foundation

nonisolated enum StorageEnvironment {
    /// The Application Support root: the owner's in production, this test
    /// process's scratch copy under a test runner.
    static let applicationSupport: URL = root(
        .applicationSupportDirectory, scratchName: "Application Support")

    /// The Caches root, resolved the same way.
    static let caches: URL = root(.cachesDirectory, scratchName: "Caches")

    /// The scratch directory that holds a test process's storage roots.
    static let scratchRoot: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent(
            "TesseractTestStorage-\(ProcessInfo.processInfo.processIdentifier)",
            isDirectory: true
        )

    private static func root(
        _ directory: FileManager.SearchPathDirectory, scratchName: String
    ) -> URL {
        guard !ProcessEnvironment.isRunningTests else {
            return scratchRoot.appendingPathComponent(scratchName, isDirectory: true)
        }
        return FileManager.default.urls(for: directory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
    }
}
