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
/// lifecycle: list a repo's files and fetch one file (see `CONTEXT.md` →
/// Model catalog). Two adapters satisfy it —
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
/// per-file fetches.
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

    private func validated(_ repo: String) throws -> Repo.ID {
        guard let repoID = Repo.ID(rawValue: repo) else {
            throw ModelFetchingError.invalidRepository(repo)
        }
        return repoID
    }
}
