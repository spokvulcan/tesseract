//
//  StorageEnvironment.swift
//  tesseract
//
//  Where the app keeps what it stores (ADR-0073). In production these are the
//  owner's Application Support and Caches folders. In a process whose **Launch
//  Mode** keeps off the owner's data (a test host, a scratch launch), they are
//  folders inside one scratch directory per process. Per process, like the
//  telemetry homes (#159): the scheme runs suites in parallel processes against
//  the same app host.
//
//  Every default storage location resolves through here, except the model
//  folder (`ModelDownloadManager.modelStorageURL`): suites load installed
//  models on purpose, and nothing under a test runner downloads one.
//

import Foundation

nonisolated enum StorageEnvironment {
    /// The Launch Mode's answer for this process (ADR-0095). Its environment
    /// alone decides it — a harness keeps the owner's data, as the owner's
    /// launch does — so shared code asks it without the Mac's harness table.
    static let keepsOffOwnersData: Bool = LaunchMode.decide(
        arguments: [], environment: ProcessInfo.processInfo.environment, harnessFlags: []
    ).keepsOffOwnersData

    /// The Application Support root: the owner's in production, this
    /// process's scratch copy under a test runner or in a scratch launch.
    static let applicationSupport: URL = root(
        .applicationSupportDirectory, scratchName: "Application Support")

    /// The Caches root, resolved the same way.
    static let caches: URL = root(.cachesDirectory, scratchName: "Caches")

    /// The owner's home folder, for other apps' settings Tesseract edits on
    /// request (Claude Code's `~/.claude/settings.json`); a scratch folder
    /// under a test runner or in a scratch launch.
    static let home: URL =
        keepsOffOwnersData
        ? scratchRoot.appendingPathComponent("Home", isDirectory: true)
        : URL.homeDirectory

    /// The scratch directory that holds a test process's (or a scratch
    /// launch's) storage roots. It starts empty: a folder already at this path
    /// was left by an earlier process with the same pid, and no run may start
    /// on another run's state. Folders of processes that have exited go on the
    /// way.
    static let scratchRoot: URL = {
        let temporary = FileManager.default.temporaryDirectory
        let pid = ProcessInfo.processInfo.processIdentifier
        if keepsOffOwnersData {
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
        guard !keepsOffOwnersData else {
            return scratchRoot.appendingPathComponent(scratchName, isDirectory: true)
        }
        return FileManager.default.urls(for: directory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
    }
}
