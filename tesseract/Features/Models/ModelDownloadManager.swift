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
    case verifying(progress: Double)
    case error(String)
}

@MainActor
final class ModelDownloadManager: ObservableObject {
    @Published private(set) var statuses: [String: ModelStatus] = [:]

    private var downloadTasks: [String: Task<Void, Never>] = [:]

    /// Per-id cache of the vision probe (`isVisionCapable`). Capability is
    /// intrinsic to a model's `config.json`, so a known answer is cached
    /// permanently; an undownloaded model is answered `false` *uncached* so a
    /// later download re-probes. The **Vision Capability Memo** — one home,
    /// shared by every caller. See `CONTEXT.md` → Model catalog.
    private var visionCache: [String: Bool] = [:]

    static let modelStorageURL: URL = {
        let url = URL.applicationSupportDirectory.appendingPathComponent(
            ModelUtils.storageDirectoryName)
        // Ensure directory exists and is excluded from Time Machine / iCloud backup
        // (re-downloadable content per Apple guidelines). Setting on the parent
        // directory excludes all contents.
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
        return url
    }()

    init() {
        refreshAllStatuses()
    }

    // MARK: - Catalog queries

    /// The **Model Catalog** join: downloaded models in a category, in catalogue
    /// order. The one place callers reach for "which models can I select / serve."
    func downloadedModels(in category: ModelCategory) -> [ModelDefinition] {
        ModelCatalog.downloaded(in: category, definitions: ModelDefinition.all, statuses: statuses)
    }

    /// Whether a model id is present on disk.
    func isDownloaded(_ id: String) -> Bool {
        ModelCatalog.isDownloaded(id, statuses: statuses)
    }

    /// Raw download state for the download UI (progress / verifying / error),
    /// defaulting to `.notDownloaded`. Not a catalog question.
    func status(for id: String) -> ModelStatus {
        statuses[id] ?? .notDownloaded
    }

    /// Whether a downloaded model can serve images — the memoized **Vision
    /// Capability Memo**. Replaces the stranded `ModelVisionCapability` class.
    func isVisionCapable(_ id: String) -> Bool {
        if let cached = visionCache[id] { return cached }
        guard isDownloaded(id), let directory = modelPath(for: id) else { return false }
        let capable = ModelCatalog.isVisionCapable(directory: directory)
        visionCache[id] = capable
        return capable
    }

    // MARK: - Status

    func refreshAllStatuses() {
        for model in ModelDefinition.all {
            // Don't overwrite in-progress download or error states
            if let existing = statuses[model.id] {
                switch existing {
                case .downloading, .verifying, .error:
                    continue
                case .notDownloaded, .downloaded:
                    break
                }
            }
            statuses[model.id] = Self.computeStatus(for: model)
        }
    }

    private static func computeStatus(for model: ModelDefinition) -> ModelStatus {
        guard case .huggingFace(_, let requiredExtension, let pathPrefix) = model.source else {
            return .notDownloaded
        }
        guard let subdir = model.cacheSubdirectory else { return .notDownloaded }

        var checkDir = modelStorageURL.appendingPathComponent(subdir)
        if let pathPrefix {
            checkDir = checkDir.appendingPathComponent(pathPrefix)
        }

        guard FileManager.default.fileExists(atPath: checkDir.path) else {
            return .notDownloaded
        }

        // Search recursively for required files (handles nested dirs like transformer/)
        let hasRequired = hasFileRecursively(in: checkDir, withExtension: requiredExtension)
        guard hasRequired else { return .notDownloaded }

        let totalSize = directorySize(at: checkDir)
        return .downloaded(sizeOnDisk: totalSize)
    }

    private static func hasFileRecursively(in directory: URL, withExtension ext: String) -> Bool {
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        else { return false }

        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == ext {
                return true
            }
        }
        return false
    }

    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: url,
                includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        else { return 0 }

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
        guard case .huggingFace(let repo, let requiredExtension, let pathPrefix) = model.source
        else { return }

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
                    try await self?.downloadFileByFile(
                        modelID: modelID,
                        repoID: repoID,
                        pathPrefix: pathPrefix
                    )
                } else if requiredExtension == "safetensors" {
                    // Repos with nested directories (transformer/, vae/, scheduler/
                    // subdirs) fail with downloadSnapshot due to file/directory
                    // naming conflicts. Download files individually instead.
                    try await self?.downloadFileByFile(
                        modelID: modelID,
                        repoID: repoID,
                        pathPrefix: nil
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

                try await self?.ensureCompanionFiles(for: model)

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

    // MARK: - File Check

    private struct FileCheckResult {
        let filtered: [Git.TreeEntry]
        let pending: [(index: Int, entry: Git.TreeEntry)]
        let modelDir: URL

        var totalFiles: Int { filtered.count }
        var validFiles: Int { totalFiles - pending.count }
        var needsRepair: Bool { !pending.isEmpty }
    }

    private func checkFiles(
        modelID: String,
        repoID: Repo.ID,
        pathPrefix: String?
    ) async throws -> FileCheckResult {
        let client = HubClient.default

        let allEntries = try await client.listFiles(in: repoID, recursive: true)
        let filtered: [Git.TreeEntry]
        if let pathPrefix {
            filtered = allEntries.filter { entry in
                entry.path.hasPrefix(pathPrefix + "/") && entry.type == .file
            }
        } else {
            filtered = allEntries.filter { $0.type == .file }
        }

        guard !filtered.isEmpty else {
            let desc =
                pathPrefix.map { "No files found for prefix '\($0)' in \(repoID)" }
                ?? "No files found in \(repoID)"
            throw NSError(
                domain: "ModelDownload", code: 1,
                userInfo: [NSLocalizedDescriptionKey: desc]
            )
        }

        guard let subdir = ModelDefinition.all.first(where: { $0.id == modelID })?.cacheSubdirectory
        else {
            throw NSError(
                domain: "ModelDownload", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No cache subdirectory for \(modelID)"]
            )
        }
        let modelDir = Self.modelStorageURL.appendingPathComponent(subdir)

        var pending = [(index: Int, entry: Git.TreeEntry)]()
        for (index, entry) in filtered.enumerated() {
            let targetFile = modelDir.appendingPathComponent(entry.path)
            if FileManager.default.fileExists(atPath: targetFile.path) {
                if let expectedSize = entry.size {
                    let attrs = try? FileManager.default.attributesOfItem(atPath: targetFile.path)
                    let localSize = (attrs?[.size] as? Int64) ?? 0
                    if localSize >= Int64(expectedSize) {
                        continue
                    }
                } else {
                    continue
                }
            }
            pending.append((index, entry))
        }

        return FileCheckResult(filtered: filtered, pending: pending, modelDir: modelDir)
    }

    private func downloadPendingFiles(
        modelID: String,
        repoID: Repo.ID,
        result: FileCheckResult
    ) async throws {
        let client = HubClient.default
        let totalFiles = result.totalFiles
        let alreadyDone = result.validFiles

        if alreadyDone > 0 {
            statuses[modelID] = .downloading(progress: Double(alreadyDone) / Double(totalFiles))
        }

        for (i, (_, entry)) in result.pending.enumerated() {
            try Task.checkCancellation()

            let targetFile = result.modelDir.appendingPathComponent(entry.path)

            let parentDir = targetFile.deletingLastPathComponent()
            if !FileManager.default.fileExists(atPath: parentDir.path) {
                try FileManager.default.createDirectory(
                    at: parentDir, withIntermediateDirectories: true)
            }

            _ = try await client.downloadFile(
                at: entry.path,
                from: repoID,
                to: targetFile
            )

            statuses[modelID] = .downloading(
                progress: Double(alreadyDone + i + 1) / Double(totalFiles)
            )
        }
    }

    private func downloadFileByFile(
        modelID: String,
        repoID: Repo.ID,
        pathPrefix: String?
    ) async throws {
        let result = try await checkFiles(modelID: modelID, repoID: repoID, pathPrefix: pathPrefix)

        // Clean up stale file left by a previous failed download
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: result.modelDir.path, isDirectory: &isDir),
            !isDir.boolValue
        {
            try? FileManager.default.removeItem(at: result.modelDir)
        }

        if result.pending.isEmpty {
            statuses[modelID] = .downloading(progress: 1.0)
            return
        }

        try await downloadPendingFiles(modelID: modelID, repoID: repoID, result: result)
    }

    // MARK: - Companion Files

    /// Fetches the model's companion files (see `CompanionFile`) into the
    /// model folder. Idempotent: files already present at their expected size
    /// are skipped, truncated leftovers are re-downloaded.
    func ensureCompanionFiles(for model: ModelDefinition) async throws {
        guard !model.companionFiles.isEmpty, let folder = modelPath(for: model.id) else { return }
        let client = HubClient.default

        for (repo, files) in Dictionary(grouping: model.companionFiles, by: \.repo) {
            guard let repoID = Repo.ID(rawValue: repo) else {
                throw NSError(
                    domain: "ModelDownload", code: 3,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Invalid companion repository ID: \(repo)"
                    ]
                )
            }
            let entries = try await client.listFiles(in: repoID, recursive: false)

            for file in files {
                try Task.checkCancellation()

                let target = folder.appendingPathComponent(file.path)
                let expectedSize = entries.first { $0.path == file.path }?.size

                if FileManager.default.fileExists(atPath: target.path) {
                    if let expectedSize {
                        let attrs = try? FileManager.default.attributesOfItem(atPath: target.path)
                        let localSize = (attrs?[.size] as? Int64) ?? 0
                        if localSize >= Int64(expectedSize) { continue }
                    } else {
                        continue
                    }
                }

                _ = try await client.downloadFile(at: file.path, from: repoID, to: target)
                Log.general.info("Companion file downloaded: \(file.path) for \(model.displayName)")
            }
        }
    }

    // MARK: - Verify & Repair

    func verifyAndRepair(modelID: String) {
        guard let model = ModelDefinition.all.first(where: { $0.id == modelID }) else { return }
        guard case .huggingFace(let repo, _, let pathPrefix) = model.source else { return }

        // Don't verify if already verifying or downloading
        if let existing = statuses[modelID] {
            switch existing {
            case .downloading, .verifying:
                return
            default:
                break
            }
        }

        statuses[modelID] = .verifying(progress: 0)

        let task = Task { [weak self] in
            do {
                guard let repoID = Repo.ID(rawValue: repo) else {
                    self?.statuses[modelID] = .error("Invalid repository ID")
                    return
                }

                let result = try await self?.checkFiles(
                    modelID: modelID,
                    repoID: repoID,
                    pathPrefix: pathPrefix
                )

                guard let result else { return }

                self?.statuses[modelID] = .verifying(progress: 1.0)

                if !result.needsRepair {
                    try await self?.ensureCompanionFiles(for: model)
                    let status = Self.computeStatus(for: model)
                    self?.statuses[modelID] = status
                    Log.general.info(
                        "Verify OK: \(model.displayName) — \(result.totalFiles) files valid")
                } else {
                    Log.general.info(
                        "Verify: \(model.displayName) — \(result.pending.count)/\(result.totalFiles) files need repair"
                    )
                    try await self?.downloadPendingFiles(
                        modelID: modelID,
                        repoID: repoID,
                        result: result
                    )
                    try await self?.ensureCompanionFiles(for: model)
                    let status = Self.computeStatus(for: model)
                    self?.statuses[modelID] = status
                    Log.general.info("Repair complete: \(model.displayName)")
                }
            } catch is CancellationError {
                let status = Self.computeStatus(for: model)
                self?.statuses[modelID] = status
            } catch {
                self?.statuses[modelID] = .error(error.localizedDescription)
                Log.general.error("Verify failed for \(model.displayName): \(error)")
            }

            self?.downloadTasks.removeValue(forKey: modelID)
        }
        downloadTasks[modelID] = task
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

        var deleteDir = Self.modelStorageURL.appendingPathComponent(subdir)
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

        var path = Self.modelStorageURL.appendingPathComponent(subdir)
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
