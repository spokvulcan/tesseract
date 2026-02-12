//
//  ModelDownloadManager.swift
//  tesseract
//

import Foundation
import Combine
import os
import MLXAudioCore
import HuggingFace

enum ModelStatus: Equatable, Sendable {
    case notDownloaded
    case downloading(progress: Double)
    case downloaded(sizeOnDisk: Int64)
    case error(String)
}

@MainActor
final class ModelDownloadManager: ObservableObject {
    @Published private(set) var statuses: [String: ModelStatus] = [:]

    private var downloadTasks: [String: Task<Void, Never>] = [:]

    private static let cacheBaseURL: URL = {
        URL.cachesDirectory.appendingPathComponent("mlx-audio")
    }()

    init() {
        refreshAllStatuses()
    }

    // MARK: - Status

    func refreshAllStatuses() {
        for model in ModelDefinition.all {
            statuses[model.id] = Self.computeStatus(for: model)
        }
    }

    private static func computeStatus(for model: ModelDefinition) -> ModelStatus {
        guard case .huggingFace(_, let requiredExtension, let pathPrefix) = model.source else {
            return .notDownloaded
        }
        guard let subdir = model.cacheSubdirectory else { return .notDownloaded }

        var checkDir = cacheBaseURL.appendingPathComponent(subdir)
        if let pathPrefix {
            checkDir = checkDir.appendingPathComponent(pathPrefix)
        }

        guard FileManager.default.fileExists(atPath: checkDir.path) else {
            return .notDownloaded
        }

        let items = (try? FileManager.default.contentsOfDirectory(
            at: checkDir, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey]
        )) ?? []

        let hasRequired = items.contains { $0.pathExtension == requiredExtension }
        guard hasRequired else { return .notDownloaded }

        let totalSize = directorySize(at: checkDir)
        return .downloaded(sizeOnDisk: totalSize)
    }

    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
            if values?.isDirectory == false {
                total += Int64(values?.fileSize ?? 0)
            }
        }
        return total
    }

    // MARK: - Download

    func download(modelID: String) {
        guard let model = ModelDefinition.all.first(where: { $0.id == modelID }) else { return }
        guard case .huggingFace(let repo, let requiredExtension, let pathPrefix) = model.source else { return }

        if case .downloading = statuses[modelID] { return }

        // Auto-download dependencies
        for depID in model.dependencies {
            let depStatus = statuses[depID] ?? .notDownloaded
            if depStatus == .notDownloaded {
                download(modelID: depID)
            }
        }

        statuses[modelID] = .downloading(progress: 0)

        let task = Task { [weak self] in
            do {
                guard let repoID = Repo.ID(rawValue: repo) else {
                    self?.statuses[modelID] = .error("Invalid repository ID")
                    return
                }

                if let pathPrefix {
                    try await self?.downloadWithPrefix(
                        modelID: modelID,
                        repoID: repoID,
                        pathPrefix: pathPrefix
                    )
                } else {
                    let weakSelf = self
                    _ = try await ModelUtils.resolveOrDownloadModel(
                        repoID: repoID,
                        requiredExtension: requiredExtension,
                        progressHandler: { @Sendable progress in
                            let fraction = progress.fractionCompleted
                            Task { @MainActor in
                                weakSelf?.statuses[modelID] = .downloading(progress: fraction)
                            }
                        }
                    )
                }

                let status = Self.computeStatus(for: model)
                self?.statuses[modelID] = status
                Log.general.info("Model downloaded: \(model.displayName)")
            } catch is CancellationError {
                self?.statuses[modelID] = .notDownloaded
            } catch {
                self?.statuses[modelID] = .error(error.localizedDescription)
                Log.general.error("Download failed for \(model.displayName): \(error)")
            }

            self?.downloadTasks.removeValue(forKey: modelID)
        }
        downloadTasks[modelID] = task
    }

    private func downloadWithPrefix(
        modelID: String,
        repoID: Repo.ID,
        pathPrefix: String
    ) async throws {
        let client = HubClient.default

        let allEntries = try await client.listFiles(in: repoID, recursive: true)
        let filtered = allEntries.filter { entry in
            entry.path.hasPrefix(pathPrefix + "/") && entry.type == .file
        }

        guard !filtered.isEmpty else {
            throw NSError(
                domain: "ModelDownload", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No files found for prefix '\(pathPrefix)' in \(repoID)"]
            )
        }

        guard let subdir = ModelDefinition.all.first(where: { $0.id == modelID })?.cacheSubdirectory else {
            return
        }
        let modelDir = Self.cacheBaseURL.appendingPathComponent(subdir)

        // Clean up stale file left by a previous failed download
        // (downloadFile may have replaced the directory with a regular file)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir), !isDir.boolValue {
            try? FileManager.default.removeItem(at: modelDir)
        }

        let totalFiles = filtered.count

        for (index, entry) in filtered.enumerated() {
            try Task.checkCancellation()

            // downloadFile expects the exact file path, not a directory
            let targetFile = modelDir.appendingPathComponent(entry.path)

            _ = try await client.downloadFile(
                at: entry.path,
                from: repoID,
                to: targetFile
            )

            statuses[modelID] = .downloading(progress: Double(index + 1) / Double(totalFiles))
        }
    }

    // MARK: - Cancel

    func cancelDownload(modelID: String) {
        downloadTasks[modelID]?.cancel()
        downloadTasks.removeValue(forKey: modelID)
        statuses[modelID] = .notDownloaded
    }

    // MARK: - Delete

    func deleteModel(modelID: String) {
        guard let model = ModelDefinition.all.first(where: { $0.id == modelID }) else { return }
        guard let subdir = model.cacheSubdirectory else { return }

        var deleteDir = Self.cacheBaseURL.appendingPathComponent(subdir)
        if let prefix = model.pathPrefix {
            deleteDir = deleteDir.appendingPathComponent(prefix)
        }

        try? FileManager.default.removeItem(at: deleteDir)
        statuses[modelID] = .notDownloaded

        Log.general.info("Deleted cached model: \(model.displayName)")
    }

    // MARK: - Path Resolution

    func modelPath(for modelID: String) -> URL? {
        guard let model = ModelDefinition.all.first(where: { $0.id == modelID }) else { return nil }
        guard let subdir = model.cacheSubdirectory else { return nil }

        var path = Self.cacheBaseURL.appendingPathComponent(subdir)
        if let prefix = model.pathPrefix {
            path = path.appendingPathComponent(prefix)
        }

        return path
    }

    // MARK: - Computed

    var totalCacheSize: Int64 {
        statuses.values.reduce(Int64(0)) { sum, status in
            if case .downloaded(let size) = status { return sum + size }
            return sum
        }
    }
}
