//
//  StorageEnvironmentTests.swift
//  tesseractTests
//
//  ADR-0073: a test run never reaches the owner's data. This suite runs in the
//  test host, so the scratch storage it checks is the storage every other
//  suite and the host's own container get.
//

import Foundation
import Testing
import WebKit

@testable import Tesseract_Agent

@Suite struct StorageEnvironmentTests {

    private static var scratch: String {
        StorageEnvironment.scratchRoot.standardizedFileURL.path + "/"
    }

    private static func ownerFolder(_ directory: FileManager.SearchPathDirectory) throws -> String {
        let url = try #require(FileManager.default.urls(for: directory, in: .userDomainMask).first)
        return url.standardizedFileURL.path + "/"
    }

    private static func isInScratch(_ url: URL) -> Bool {
        url.standardizedFileURL.path.hasPrefix(scratch)
    }

    @Test func rootsLiveInTheProcessScratchDirectory() throws {
        #expect(Self.isInScratch(StorageEnvironment.applicationSupport))
        #expect(Self.isInScratch(StorageEnvironment.caches))
        #expect(
            !StorageEnvironment.applicationSupport.standardizedFileURL.path
                .hasPrefix(try Self.ownerFolder(.applicationSupportDirectory)))
        #expect(
            !StorageEnvironment.caches.standardizedFileURL.path
                .hasPrefix(try Self.ownerFolder(.cachesDirectory)))
        // One scratch directory per process: parallel test processes never
        // share a store.
        #expect(
            StorageEnvironment.scratchRoot.lastPathComponent.hasSuffix(
                "-\(ProcessInfo.processInfo.processIdentifier)"))
    }

    @MainActor
    @Test func defaultLocationsResolveThroughTheSeam() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        for url in [
            PathSandbox.defaultRoot,
            TelemetryEnvironment.durableDirectory(component: "Probe"),
            SSDEnduranceAccumulator.defaultFileURL,
            settings.ssdPrefixCacheRootURL,
        ] {
            #expect(Self.isInScratch(url), "\(url.path) is outside the scratch directory")
        }
    }

    @MainActor
    @Test func agentProfileIsNonPersistent() {
        #expect(!AgentProfile().dataStore.isPersistent)
    }

    // MARK: - Source shape

    /// The files allowed to resolve a storage root without the seam: the seam
    /// itself, the model folder (the ADR's one exception), and the one-time
    /// move of that folder out of the retired sandbox container, which returns
    /// early under a test runner.
    private static let allowed: Set<String> = [
        "StorageEnvironment.swift",
        "ModelDownloadManager.swift",
        "SandboxMigration.swift",
    ]

    /// The ways to reach the owner's Application Support or Caches folder.
    private static let rootLookups = [
        "applicationSupportDirectory", "cachesDirectory",
        "NSHomeDirectory(", "homeDirectoryForCurrentUser",
    ]

    private static var appSourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // tesseractTests
            .deletingLastPathComponent()  // project root
            .appendingPathComponent("tesseract")
    }

    /// A new store that finds its folder by itself would take test runs back
    /// to the owner's data. Every default location goes through
    /// `StorageEnvironment` instead. Comment lines are skipped, so docs can
    /// still name the lookups. Off the main actor on purpose: the scan reads
    /// every app source file, and parallel main-actor suites wait on timeouts.
    @Test func onlyTheSeamResolvesStorageRoots() throws {
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: Self.appSourceRoot,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ))
        var scanned = 0
        var offenders: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            scanned += 1
            guard !Self.allowed.contains(url.lastPathComponent) else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            for (index, line) in source.components(separatedBy: "\n").enumerated() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("//") { continue }
                if Self.rootLookups.contains(where: { line.contains($0) }) {
                    offenders.append("\(url.lastPathComponent):\(index + 1)")
                }
            }
        }
        #expect(scanned > 100, "the scan found no app sources at \(Self.appSourceRoot.path)")
        #expect(offenders.isEmpty, "resolve storage through StorageEnvironment: \(offenders)")
    }
}
