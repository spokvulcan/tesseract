//
//  ModelFetching.swift
//  tesseract
//

import Foundation
import HuggingFace
import MLXAudioCore

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

    var errorDescription: String? {
        switch self {
        case .invalidRepository(let repo): "Invalid repository ID: \(repo)"
        }
    }
}

/// The HuggingFace-backed production adapter: the hub client for listing and
/// per-file fetches, the shared resolve utility for snapshots. Both sit
/// behind this adapter unchanged.
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

    func resolveSnapshot(
        of repo: String,
        requiredExtension: String,
        onProgress: @escaping @MainActor @Sendable (Double) -> Void
    ) async throws {
        _ = try await ModelUtils.resolveOrDownloadModel(
            repoID: validated(repo),
            requiredExtension: requiredExtension,
            progressHandler: { @Sendable progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor in onProgress(fraction) }
            }
        )
    }

    private func validated(_ repo: String) throws -> Repo.ID {
        guard let repoID = Repo.ID(rawValue: repo) else {
            throw ModelFetchingError.invalidRepository(repo)
        }
        return repoID
    }
}
