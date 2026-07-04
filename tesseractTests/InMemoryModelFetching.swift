//
//  InMemoryModelFetching.swift
//  tesseractTests
//
//  A hermetic, in-memory **Model Fetching** peer for tests — a scripted repo
//  table, not a mock, and a peer implementation of `HuggingFaceModelFetching`
//  (the same pattern as `InMemorySettingsStore`). Each repo is scripted as a
//  list of (path, size) entries; fetches write that many zero bytes into the
//  temp storage root, so the manager's real file-system checks (sizes, status
//  computation) exercise real files. Sharing no global state, it runs
//  hermetically and in parallel.
//
//  It records the repos listed and files fetched, so tests can pin seam-level
//  behavior ("repair fetched only the undersized file") without reaching into
//  the manager. `holdListings` parks `listFiles` until the surrounding task is
//  cancelled — the hook for cancellation and occupancy tests.
//

import Foundation

@testable import Tesseract_Agent

@MainActor
final class InMemoryModelFetching: ModelFetching {
    struct ScriptedFile {
        let path: String
        let size: Int
    }

    var repos: [String: [ScriptedFile]]
    /// Any verb on this repo throws the scripted error instead.
    var errorsByRepo: [String: Error] = [:]
    /// Park `listFiles` until the surrounding task is cancelled.
    var holdListings = false

    /// Repos listed via `listFiles`, in order.
    private(set) var listedRepos: [String] = []
    /// Files fetched via `fetchFile`, in order, as "repo/path".
    private(set) var fetchedFiles: [String] = []

    private let storageRoot: URL

    init(storageRoot: URL, repos: [String: [ScriptedFile]] = [:]) {
        self.storageRoot = storageRoot
        self.repos = repos
    }

    func listFiles(in repo: String, recursive: Bool) async throws -> [RemoteModelFile] {
        listedRepos.append(repo)
        if let error = errorsByRepo[repo] { throw error }
        if holdListings {
            // Cancellation of the surrounding task throws CancellationError here.
            try await Task.sleep(for: .seconds(600))
        }
        return (repos[repo] ?? []).map { RemoteModelFile(path: $0.path, size: $0.size) }
    }

    func fetchFile(at path: String, from repo: String, to destination: URL) async throws {
        fetchedFiles.append("\(repo)/\(path)")
        if let error = errorsByRepo[repo] { throw error }
        guard let file = repos[repo]?.first(where: { $0.path == path }) else {
            throw NSError(
                domain: "InMemoryModelFetching", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Unscripted file \(repo)/\(path)"]
            )
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: file.size).write(to: destination)
    }

    func resolveSnapshot(
        of repo: String,
        requiredExtension: String,
        onProgress: @escaping @MainActor @Sendable (Double) -> Void
    ) async throws {
        if let error = errorsByRepo[repo] { throw error }
        let files = repos[repo] ?? []
        let modelDir = storageRoot.modelDirectory(forRepo: repo)
        for (index, file) in files.enumerated() {
            try Task.checkCancellation()
            let target = modelDir.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(count: file.size).write(to: target)
            onProgress(Double(index + 1) / Double(files.count))
        }
    }
}

extension URL {
    /// The on-disk model directory for a repo under a storage root — the same
    /// "repo with '/' flattened to '_'" layout the manager uses.
    func modelDirectory(forRepo repo: String) -> URL {
        appendingPathComponent(repo.replacingOccurrences(of: "/", with: "_"))
    }
}
