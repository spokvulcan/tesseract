//
//  ModelDownloadLifecycleTests.swift
//  tesseractTests
//
//  Status-transition tests for the model download lifecycle (PRD #139),
//  through the public `ModelDownloadManager` entries only: construct the
//  manager with the in-memory **Model Fetching** peer and a temp storage
//  root, drive `download` / `verifyAndRepair` / `cancelDownload`, and assert
//  the status dictionary over time plus the resulting files on disk — never
//  internal task state. Prior art: `InMemorySettingsStore` (hermetic peer),
//  `VoiceCaptureSessionTests` (lifecycle + cancellation discipline), and
//  `ModelCatalogTests` (the pure sibling).
//

import Combine
import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ModelDownloadLifecycleTests {

    // MARK: - Download

    @Test func downloadRunsToDownloadedAndWritesFiles() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: [
                "fixture/alpha": [
                    .init(path: "model.safetensors", size: 64),
                    .init(path: "config.json", size: 16),
                ]
            ]
        )
        defer { harness.tearDown() }

        harness.manager.download(modelID: "alpha")
        #expect(harness.manager.status(for: "alpha") == .downloading(progress: 0))

        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .downloaded(sizeOnDisk: 80))

        let weights = harness.storageRoot
            .appendingPathComponent("fixture_alpha/model.safetensors")
        #expect(FileManager.default.fileExists(atPath: weights.path))
    }

    @Test func downloadReportsPerFileProgressFractions() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: [
                "fixture/alpha": [
                    .init(path: "a.safetensors", size: 8),
                    .init(path: "b.safetensors", size: 8),
                    .init(path: "config.json", size: 4),
                    .init(path: "tokenizer.json", size: 4),
                ]
            ]
        )
        defer { harness.tearDown() }

        harness.manager.download(modelID: "alpha")
        _ = try await harness.waitForSettled(id: "alpha")

        let fractions = harness.history["alpha", default: []].compactMap { status -> Double? in
            if case .downloading(let progress) = status { return progress }
            return nil
        }
        #expect(fractions == [0, 0.25, 0.5, 0.75, 1.0])
    }

    // MARK: - Occupancy

    @Test func secondDownloadAndVerifyMidDownloadAreIgnored() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: ["fixture/alpha": [.init(path: "model.safetensors", size: 8)]]
        )
        defer { harness.tearDown() }
        harness.fetching.holdListings = true

        harness.manager.download(modelID: "alpha")
        try await harness.waitUntil { harness.fetching.listedRepos.count == 1 }

        harness.manager.download(modelID: "alpha")
        harness.manager.verifyAndRepair(modelID: "alpha")

        // Give a wrongly-admitted second operation cycles to reach the peer.
        try await Task.sleep(for: .milliseconds(20))
        #expect(harness.fetching.listedRepos == ["fixture/alpha"])
        #expect(harness.manager.status(for: "alpha") == .downloading(progress: 0))

        harness.manager.cancelDownload(modelID: "alpha")
        _ = try await harness.waitForSettled(id: "alpha")
    }

    // MARK: - Cancel

    @Test func cancellingFreshDownloadReturnsToNotDownloaded() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: ["fixture/alpha": [.init(path: "model.safetensors", size: 8)]]
        )
        defer { harness.tearDown() }
        harness.fetching.holdListings = true

        harness.manager.download(modelID: "alpha")
        #expect(harness.manager.status(for: "alpha") == .downloading(progress: 0))

        harness.manager.cancelDownload(modelID: "alpha")

        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .notDownloaded)
    }

    @Test func cancellingVerifyOfDownloadedModelKeepsDownloadedStatus() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: ["fixture/alpha": [.init(path: "model.safetensors", size: 64)]]
        )
        defer { harness.tearDown() }

        try harness.placeFile(repo: "fixture/alpha", path: "model.safetensors", size: 64)
        harness.manager.refreshAllStatuses()
        #expect(harness.manager.status(for: "alpha") == .downloaded(sizeOnDisk: 64))

        harness.fetching.holdListings = true
        harness.manager.verifyAndRepair(modelID: "alpha")
        #expect(harness.manager.status(for: "alpha") == .verifying(progress: 0))
        let observedBeforeCancel = harness.history["alpha", default: []].count

        harness.manager.cancelDownload(modelID: "alpha")
        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .downloaded(sizeOnDisk: 64))

        // The regression this pins: cancel used to hard-code .notDownloaded,
        // flashing "not downloaded" over a model that is fully on disk.
        let afterCancel = harness.history["alpha", default: []].dropFirst(observedBeforeCancel)
        #expect(!afterCancel.contains(.notDownloaded))
    }

    // MARK: - Verify & Repair

    @Test func verifyOfIntactModelReturnsToDownloadedWithoutFetching() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: [
                "fixture/alpha": [
                    .init(path: "model.safetensors", size: 64),
                    .init(path: "config.json", size: 16),
                ]
            ]
        )
        defer { harness.tearDown() }

        try harness.placeFile(repo: "fixture/alpha", path: "model.safetensors", size: 64)
        try harness.placeFile(repo: "fixture/alpha", path: "config.json", size: 16)
        harness.manager.refreshAllStatuses()

        harness.manager.verifyAndRepair(modelID: "alpha")
        #expect(harness.manager.status(for: "alpha") == .verifying(progress: 0))

        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .downloaded(sizeOnDisk: 80))
        #expect(harness.fetching.fetchedFiles.isEmpty)
        #expect(harness.history["alpha", default: []].contains(.verifying(progress: 1.0)))
    }

    @Test func verifyRepairsOnlyMissingOrUndersizedFiles() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/alpha")],
            repos: [
                "fixture/alpha": [
                    .init(path: "intact.safetensors", size: 64),
                    .init(path: "truncated.safetensors", size: 32),
                    .init(path: "missing.json", size: 16),
                ]
            ]
        )
        defer { harness.tearDown() }

        try harness.placeFile(repo: "fixture/alpha", path: "intact.safetensors", size: 64)
        try harness.placeFile(repo: "fixture/alpha", path: "truncated.safetensors", size: 10)
        harness.manager.refreshAllStatuses()

        harness.manager.verifyAndRepair(modelID: "alpha")
        let settled = try await harness.waitForSettled(id: "alpha")

        #expect(settled == .downloaded(sizeOnDisk: 112))
        #expect(
            harness.fetching.fetchedFiles == [
                "fixture/alpha/truncated.safetensors",
                "fixture/alpha/missing.json",
            ])
        #expect(harness.sizeOnDisk(repo: "fixture/alpha", path: "truncated.safetensors") == 32)
        #expect(harness.sizeOnDisk(repo: "fixture/alpha", path: "missing.json") == 16)
    }

    // MARK: - Dependencies

    @Test func downloadAutoDownloadsDependenciesRecursively() async throws {
        let harness = try LifecycleHarness(
            definitions: [
                Self.model("parent", repo: "fixture/parent", dependencies: ["mid"]),
                Self.model("mid", repo: "fixture/mid", dependencies: ["leaf"]),
                Self.model("leaf", repo: "fixture/leaf"),
            ],
            repos: [
                "fixture/parent": [.init(path: "model.safetensors", size: 8)],
                "fixture/mid": [.init(path: "model.safetensors", size: 8)],
                "fixture/leaf": [.init(path: "model.safetensors", size: 8)],
            ]
        )
        defer { harness.tearDown() }

        harness.manager.download(modelID: "parent")

        for id in ["parent", "mid", "leaf"] {
            let settled = try await harness.waitForSettled(id: id)
            #expect(settled == .downloaded(sizeOnDisk: 8), "\(id) should be downloaded")
        }
    }

    // MARK: - Companion files

    @Test func companionFilesAreEnsuredOnDownloadAndVerify() async throws {
        let companion = CompanionFile(repo: "fixture/companions", path: "tokenizer.json")
        let harness = try LifecycleHarness(
            definitions: [
                Self.model("alpha", repo: "fixture/alpha", companionFiles: [companion])
            ],
            repos: [
                "fixture/alpha": [.init(path: "model.safetensors", size: 64)],
                "fixture/companions": [.init(path: "tokenizer.json", size: 5)],
            ]
        )
        defer { harness.tearDown() }

        // Download path: the companion lands next to the weights.
        harness.manager.download(modelID: "alpha")
        let downloaded = try await harness.waitForSettled(id: "alpha")
        #expect(downloaded == .downloaded(sizeOnDisk: 69))
        #expect(harness.sizeOnDisk(repo: "fixture/alpha", path: "tokenizer.json") == 5)

        // Verify path: a deleted companion is restored.
        try harness.removeFile(repo: "fixture/alpha", path: "tokenizer.json")
        harness.manager.verifyAndRepair(modelID: "alpha")
        let verified = try await harness.waitForSettled(id: "alpha")
        #expect(verified == .downloaded(sizeOnDisk: 69))
        #expect(harness.sizeOnDisk(repo: "fixture/alpha", path: "tokenizer.json") == 5)
    }

    // MARK: - Errors

    @Test func invalidRepositorySurfacesErrorStatus() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "not a repo")],
            repos: [:]
        )
        defer { harness.tearDown() }
        harness.fetching.errorsByRepo["not a repo"] =
            ModelFetchingError.invalidRepository("not a repo")

        harness.manager.download(modelID: "alpha")
        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .error("Invalid repository ID: not a repo"))
    }

    @Test func emptyRepositoryListingSurfacesErrorStatus() async throws {
        let harness = try LifecycleHarness(
            definitions: [Self.model("alpha", repo: "fixture/empty")],
            repos: ["fixture/empty": []]
        )
        defer { harness.tearDown() }

        harness.manager.download(modelID: "alpha")
        let settled = try await harness.waitForSettled(id: "alpha")
        #expect(settled == .error("No files found in fixture/empty"))
    }

    // MARK: - Snapshot resolution

    @Test func nonSafetensorsUnprefixedModelResolvesSnapshotWithProgress() async throws {
        let harness = try LifecycleHarness(
            definitions: [
                Self.model("snap", repo: "fixture/snap", requiredExtension: "bin")
            ],
            repos: [
                "fixture/snap": [
                    .init(path: "weights.bin", size: 24),
                    .init(path: "config.json", size: 8),
                ]
            ]
        )
        defer { harness.tearDown() }

        harness.manager.download(modelID: "snap")
        let settled = try await harness.waitForSettled(id: "snap")

        #expect(settled == .downloaded(sizeOnDisk: 32))
        let fractions = harness.history["snap", default: []].compactMap { status -> Double? in
            if case .downloading(let progress) = status { return progress }
            return nil
        }
        #expect(fractions == [0, 0.5, 1.0])
    }

    // MARK: - Fixtures

    static func model(
        _ id: String,
        repo: String,
        requiredExtension: String = "safetensors",
        pathPrefix: String? = nil,
        dependencies: [String] = [],
        companionFiles: [CompanionFile] = []
    ) -> ModelDefinition {
        ModelDefinition(
            id: id, displayName: "Display \(id)", description: "",
            category: .agent,
            source: .huggingFace(
                repo: repo, requiredExtension: requiredExtension, pathPrefix: pathPrefix),
            sizeDescription: "", dependencies: dependencies,
            companionFiles: companionFiles)
    }
}

/// Per-test world: a temp storage root, the in-memory peer, the manager under
/// test, and a status history recorded from the published `statuses` — the
/// observable contract.
@MainActor
final class LifecycleHarness {
    let storageRoot: URL
    let fetching: InMemoryModelFetching
    let manager: ModelDownloadManager

    /// Distinct consecutive statuses seen per model id, oldest first.
    private(set) var history: [String: [ModelStatus]] = [:]
    private var cancellable: AnyCancellable?

    init(
        definitions: [ModelDefinition],
        repos: [String: [InMemoryModelFetching.ScriptedFile]]
    ) throws {
        storageRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-lifecycle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: storageRoot, withIntermediateDirectories: true)
        fetching = InMemoryModelFetching(storageRoot: storageRoot, repos: repos)
        manager = ModelDownloadManager(
            fetching: fetching, storageRoot: storageRoot, definitions: definitions)
        cancellable = manager.$statuses.sink { [weak self] statuses in
            guard let self else { return }
            for (id, status) in statuses where self.history[id, default: []].last != status {
                self.history[id, default: []].append(status)
            }
        }
    }

    func tearDown() {
        cancellable = nil
        try? FileManager.default.removeItem(at: storageRoot)
    }

    private func fileURL(repo: String, path: String) -> URL {
        storageRoot.modelDirectory(forRepo: repo).appendingPathComponent(path)
    }

    /// Pre-place a file on disk under the model dir, as a completed download
    /// (or a truncated leftover) would have left it.
    func placeFile(repo: String, path: String, size: Int) throws {
        let target = fileURL(repo: repo, path: path)
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: size).write(to: target)
    }

    func removeFile(repo: String, path: String) throws {
        try FileManager.default.removeItem(at: fileURL(repo: repo, path: path))
    }

    func sizeOnDisk(repo: String, path: String) -> Int? {
        let target = fileURL(repo: repo, path: path)
        let attrs = try? FileManager.default.attributesOfItem(atPath: target.path)
        return attrs?[.size] as? Int
    }

    /// Wait until the model's status satisfies `predicate`, or time out and
    /// return the current status so the caller's `#expect` shows the actual.
    func waitFor(
        id: String,
        timeout: Duration = .seconds(10),
        _ predicate: (ModelStatus) -> Bool
    ) async throws -> ModelStatus {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            let status = manager.status(for: id)
            if predicate(status) { return status }
            try await Task.sleep(for: .milliseconds(2))
        }
        return manager.status(for: id)
    }

    /// Wait until an arbitrary condition holds (e.g. the peer saw a call).
    func waitUntil(
        timeout: Duration = .seconds(10), _ condition: () -> Bool
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(2))
        }
        Issue.record("condition not met within \(timeout)")
    }

    /// Wait until the model leaves its in-progress states.
    func waitForSettled(
        id: String, timeout: Duration = .seconds(10)
    ) async throws -> ModelStatus {
        try await waitFor(id: id, timeout: timeout) { status in
            switch status {
            case .downloading, .verifying: false
            default: true
            }
        }
    }
}
