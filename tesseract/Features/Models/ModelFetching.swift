//
//  ModelFetching.swift
//  tesseract
//

import Foundation

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
/// `HuggingFaceModelFetching` (the Mac app) and `InMemoryModelFetching` (tests).
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
