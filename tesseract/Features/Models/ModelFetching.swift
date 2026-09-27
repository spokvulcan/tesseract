//
//  ModelFetching.swift
//  tesseract
//

import Foundation
import HuggingFace

/// One file in a remote model repository, as the **Model Fetching** port
/// reports it: path relative to the repo root, plus the expected byte size
/// when the hub provides one.
struct RemoteModelFile: Equatable, Sendable {
    let path: String
    let size: Int?
}

/// **Model Fetching** — the narrow hub port below the model download
/// lifecycle: list a repo's files, fetch one file, resolve-or-download a
/// snapshot (see `CONTEXT.md` → Model catalog). Two adapters satisfy it —
/// `HuggingFaceModelFetching` (app) and `InMemoryModelFetching` (tests).
/// Disk deliberately stays outside the seam: size checks, stale-file
/// cleanup, and status computation run against the real file system,
/// because disk truth is the download manager's job.
protocol ModelFetching {
    /// The repo's files (directories excluded), optionally recursing into
    /// subdirectories.
    func listFiles(in repo: String, recursive: Bool) async throws -> [RemoteModelFile]

    /// Fetch a single file into `destination`.
    func fetchFile(at path: String, from repo: String, to destination: URL) async throws

    /// Resolve a full snapshot from the shared cache or download it,
    /// reporting fractional progress in [0, 1] on the main actor.
    func resolveSnapshot(
        of repo: String,
        requiredExtension: String,
        onProgress: @escaping @MainActor @Sendable (Double) -> Void
    ) async throws
}

enum ModelFetchingError: LocalizedError {
    case invalidRepository(String)
    case incompleteDownload(String)

    var errorDescription: String? {
        switch self {
        case .invalidRepository(let repo): "Invalid repository ID: \(repo)"
        case .incompleteDownload(let repo):
            "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared. Please try again."
        }
    }
}

/// The HuggingFace-backed production adapter: the hub client for listing,
/// per-file fetches and snapshots.
struct HuggingFaceModelFetching: ModelFetching {
    func listFiles(in repo: String, recursive: Bool) async throws -> [RemoteModelFile] {
        let entries = try await HubClient.default.listFiles(
            in: validated(repo), recursive: recursive)
        return
            entries
            .filter { $0.type == .file }
            .map { RemoteModelFile(path: $0.path, size: $0.size) }
    }

    func fetchFile(at path: String, from repo: String, to destination: URL) async throws {
        _ = try await HubClient.default.downloadFile(
            at: path, from: validated(repo), to: destination)
    }

    /// A directory that already holds a non-empty file of the required
    /// extension and a parseable `config.json` is reused as-is. Anything
    /// else under the repo's directory is cleared and the snapshot fetched
    /// again.
    func resolveSnapshot(
        of repo: String,
        requiredExtension: String,
        onProgress: @escaping @MainActor @Sendable (Double) -> Void
    ) async throws {
        let repoID = try validated(repo)
        let ext =
            requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst()) : requiredExtension
        let modelDir = ModelDownloadManager.modelStorageURL.appendingPathComponent(
            ModelDefinition.storageSubdirectory(forRepo: repo))
        let fm = FileManager.default

        if fm.fileExists(atPath: modelDir.path) {
            if Self.hasNonEmptyFile(withExtension: ext, in: modelDir) {
                let config = modelDir.appendingPathComponent("config.json")
                if fm.fileExists(atPath: config.path) {
                    if let data = try? Data(contentsOf: config),
                        (try? JSONSerialization.jsonObject(with: data)) != nil
                    {
                        return
                    }
                    Self.clear(modelDir, repoID: repoID)
                }
            } else {
                Self.clear(modelDir, repoID: repoID)
            }
        }

        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        _ = try await HubClient.default.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: "main",
            matching: ["*.\(ext)", "*.safetensors", "*.json", "*.txt", "*.wav"],
            progressHandler: { progress in
                onProgress(progress.fractionCompleted)
            }
        )

        guard Self.hasNonEmptyFile(withExtension: ext, in: modelDir) else {
            Self.clear(modelDir, repoID: repoID)
            throw ModelFetchingError.incompleteDownload(repo)
        }
    }

    private static func hasNonEmptyFile(withExtension ext: String, in directory: URL) -> Bool {
        let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey])
        return files?.contains { file in
            guard file.pathExtension == ext else { return false }
            return ((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0) > 0
        } ?? false
    }

    /// Drops a partial snapshot and the hub's cached copy of the repo, so the
    /// next attempt starts clean.
    private static func clear(_ modelDir: URL, repoID: Repo.ID) {
        let fm = FileManager.default
        try? fm.removeItem(at: modelDir)
        let hubRepoDir = HubCache.default.repoDirectory(repo: repoID, kind: .model)
        if fm.fileExists(atPath: hubRepoDir.path) {
            try? fm.removeItem(at: hubRepoDir)
        }
    }

    private func validated(_ repo: String) throws -> Repo.ID {
        guard let repoID = Repo.ID(rawValue: repo) else {
            throw ModelFetchingError.invalidRepository(repo)
        }
        return repoID
    }
}
