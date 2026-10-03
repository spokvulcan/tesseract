//
//  HuggingFaceModelFetching.swift
//  tesseract
//

import Foundation
import HuggingFace

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
