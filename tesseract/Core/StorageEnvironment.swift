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

    /// The owner's home folder, for other apps' settings Tesseract edits on
    /// request (Claude Code's `~/.claude/settings.json`); a scratch folder
    /// under a test runner.
    static let home: URL =
        ProcessEnvironment.isRunningTests
        ? scratchRoot.appendingPathComponent("Home", isDirectory: true)
        : URL.homeDirectory

    /// The scratch directory that holds a test process's storage roots. It
    /// starts empty: a folder already at this path was left by an earlier
    /// process with the same pid, and a test must never start on another run's
    /// state. Folders of test processes that have exited go on the way.
    static let scratchRoot: URL = {
        let temporary = FileManager.default.temporaryDirectory
        let pid = ProcessInfo.processInfo.processIdentifier
        if ProcessEnvironment.isRunningTests {
            removeExitedScratchRoots(in: temporary, ownPID: pid, isAlive: isProcessAlive)
        }
        return temporary.appendingPathComponent("\(scratchPrefix)\(pid)", isDirectory: true)
    }()

    static let scratchPrefix = "TesseractTestStorage-"

    /// Removes the scratch folders in `directory` whose test process has
    /// exited, `ownPID`'s included: pids are unique among live processes, so a
    /// folder under this process's pid belongs to a dead one.
    static func removeExitedScratchRoots(
        in directory: URL, ownPID: pid_t, isAlive: (pid_t) -> Bool
    ) {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in names where name.hasPrefix(scratchPrefix) {
            guard let pid = pid_t(name.dropFirst(scratchPrefix.count)),
                pid == ownPID || !isAlive(pid)
            else { continue }
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    /// Whether a process with this pid exists. One another user owns (`EPERM`)
    /// counts as alive.
    private static func isProcessAlive(_ pid: pid_t) -> Bool {
        kill(pid, 0) == 0 || errno != ESRCH
    }

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
