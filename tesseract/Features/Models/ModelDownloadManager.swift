//
//  ModelDownloadManager.swift
//  tesseract
//

import Foundation
import Combine
import os
import MLXAudioCore

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

    private let fetching: any ModelFetching
    private let storageRoot: URL
    private let definitions: [ModelDefinition]

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

    init(
        fetching: any ModelFetching = HuggingFaceModelFetching(),
        storageRoot: URL = ModelDownloadManager.modelStorageURL,
        definitions: [ModelDefinition] = ModelDefinition.all
    ) {
        self.fetching = fetching
        self.storageRoot = storageRoot
        self.definitions = definitions
        refreshAllStatuses()
    }

    // MARK: - Catalog queries

    /// The **Model Catalog** join: downloaded models in a category, in catalogue
    /// order. The one place callers reach for "which models can I select / serve."
    func downloadedModels(in category: ModelCategory) -> [ModelDefinition] {
        ModelCatalog.downloaded(in: category, definitions: definitions, statuses: statuses)
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
        for model in definitions {
            // Don't overwrite in-progress download or error states
            if let existing = statuses[model.id] {
                switch existing {
                case .downloading, .verifying, .error:
                    continue
                case .notDownloaded, .downloaded:
                    break
                }
            }
            statuses[model.id] = computeStatus(for: model)
        }
    }

    /// Disk truth: status recomputed from what is actually on disk.
    private func recomputeStatus(for model: ModelDefinition) {
        statuses[model.id] = computeStatus(for: model)
    }

    private func computeStatus(for model: ModelDefinition) -> ModelStatus {
        guard case .huggingFace(_, let requiredExtension, let pathPrefix) = model.source else {
            return .notDownloaded
        }
        guard let subdir = model.cacheSubdirectory else { return .notDownloaded }

        var checkDir = storageRoot.appendingPathComponent(subdir)
        if let pathPrefix {
            checkDir = checkDir.appendingPathComponent(pathPrefix)
        }

        guard FileManager.default.fileExists(atPath: checkDir.path) else {
            return .notDownloaded
        }

        // Search recursively for required files (handles nested dirs like transformer/)
        let hasRequired = Self.hasFileRecursively(in: checkDir, withExtension: requiredExtension)
        guard hasRequired else { return .notDownloaded }

        let totalSize = Self.directorySize(at: checkDir)
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

        for case let fileURL as URL in enumerator where fileURL.pathExtension == ext {
            return true
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

    // MARK: - Lifecycle

    /// The one download/verify lifecycle both public entries parameterize:
    /// occupancy guard → in-progress status → tracked task running `body` →
    /// shared tail (companion files → recompute status from disk → task
    /// removal) — with a single cancellation arm. Cancel semantics unify on
    /// disk truth: any cancellation recomputes status from what is actually
    /// on disk, so a cancelled fresh download returns to "not downloaded"
    /// while a cancelled verify of a downloaded model stays "downloaded".
    private func runLifecycle(
        for model: ModelDefinition,
        occupiedWhen isOccupied: (ModelStatus) -> Bool,
        inProgress: ModelStatus,
        failureLabel: String,
        body: @escaping @MainActor () async throws -> Void
    ) {
        if let existing = statuses[model.id], isOccupied(existing) { return }

        statuses[model.id] = inProgress

        let task = Task { [weak self] in
            do {
                try await body()
                try await self?.ensureCompanionFiles(for: model)
                self?.recomputeStatus(for: model)
            } catch is CancellationError {
                self?.recomputeStatus(for: model)
            } catch {
                self?.statuses[model.id] = .error(error.localizedDescription)
                Log.general.error("\(failureLabel) failed for \(model.displayName): \(error)")
            }

            self?.downloadTasks.removeValue(forKey: model.id)
        }
        downloadTasks[model.id] = task
    }

    // MARK: - Download

    func download(modelID: String) {
        guard let model = definitions.first(where: { $0.id == modelID }) else { return }
        guard case .huggingFace(let repo, let requiredExtension, let pathPrefix) = model.source
        else { return }

        let occupied: (ModelStatus) -> Bool = {
            if case .downloading = $0 { return true }
            return false
        }
        // Checked here as well as in the lifecycle so dependencies don't kick
        // off when this model's own download is already in flight.
        if let existing = statuses[modelID], occupied(existing) { return }

        // Auto-download dependencies
        for depID in model.dependencies {
            let depStatus = statuses[depID] ?? .notDownloaded
            if depStatus == .notDownloaded {
                download(modelID: depID)
            }
        }

        runLifecycle(
            for: model,
            occupiedWhen: occupied,
            inProgress: .downloading(progress: 0),
            failureLabel: "Download"
        ) { [weak self] in
            if pathPrefix != nil || requiredExtension == "safetensors" {
                // Repos with nested directories (transformer/, vae/, scheduler/
                // subdirs) fail with snapshot resolution due to file/directory
                // naming conflicts. Download files individually instead.
                try await self?.downloadFileByFile(
                    model: model, repo: repo, pathPrefix: pathPrefix)
            } else {
                try await self?.fetching.resolveSnapshot(
                    of: repo,
                    requiredExtension: requiredExtension,
                    onProgress: { [weak self] fraction in
                        self?.statuses[modelID] = .downloading(progress: fraction)
                    }
                )
            }
            Log.general.info("Model downloaded: \(model.displayName)")
        }
    }

    // MARK: - File Check

    private struct FileCheckResult {
        let totalFiles: Int
        let pending: [RemoteModelFile]
        let modelDir: URL

        var validFiles: Int { totalFiles - pending.count }
        var needsRepair: Bool { !pending.isEmpty }
    }

    private func checkFiles(
        model: ModelDefinition,
        repo: String,
        pathPrefix: String?
    ) async throws -> FileCheckResult {
        let allFiles = try await fetching.listFiles(in: repo, recursive: true)
        let filtered: [RemoteModelFile]
        if let pathPrefix {
            filtered = allFiles.filter { $0.path.hasPrefix(pathPrefix + "/") }
        } else {
            filtered = allFiles
        }

        guard !filtered.isEmpty else {
            let desc =
                pathPrefix.map { "No files found for prefix '\($0)' in \(repo)" }
                ?? "No files found in \(repo)"
            throw NSError(
                domain: "ModelDownload", code: 1,
                userInfo: [NSLocalizedDescriptionKey: desc]
            )
        }

        guard let subdir = model.cacheSubdirectory else {
            throw NSError(
                domain: "ModelDownload", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No cache subdirectory for \(model.id)"]
            )
        }
        let modelDir = storageRoot.appendingPathComponent(subdir)

        var pending = [RemoteModelFile]()
        for file in filtered {
            let targetFile = modelDir.appendingPathComponent(file.path)
            if FileManager.default.fileExists(atPath: targetFile.path) {
                if let expectedSize = file.size {
                    let attrs = try? FileManager.default.attributesOfItem(atPath: targetFile.path)
                    let localSize = (attrs?[.size] as? Int64) ?? 0
                    if localSize >= Int64(expectedSize) {
                        continue
                    }
                } else {
                    continue
                }
            }
            pending.append(file)
        }

        return FileCheckResult(totalFiles: filtered.count, pending: pending, modelDir: modelDir)
    }

    private func downloadPendingFiles(
        modelID: String,
        repo: String,
        result: FileCheckResult
    ) async throws {
        let totalFiles = result.totalFiles
        let alreadyDone = result.validFiles

        if alreadyDone > 0 {
            statuses[modelID] = .downloading(progress: Double(alreadyDone) / Double(totalFiles))
        }

        for (index, file) in result.pending.enumerated() {
            try Task.checkCancellation()

            let targetFile = result.modelDir.appendingPathComponent(file.path)

            let parentDir = targetFile.deletingLastPathComponent()
            if !FileManager.default.fileExists(atPath: parentDir.path) {
                try FileManager.default.createDirectory(
                    at: parentDir, withIntermediateDirectories: true)
            }

            try await fetching.fetchFile(at: file.path, from: repo, to: targetFile)

            statuses[modelID] = .downloading(
                progress: Double(alreadyDone + index + 1) / Double(totalFiles)
            )
        }
    }

    private func downloadFileByFile(
        model: ModelDefinition,
        repo: String,
        pathPrefix: String?
    ) async throws {
        let result = try await checkFiles(model: model, repo: repo, pathPrefix: pathPrefix)

        // Clean up stale file left by a previous failed download
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: result.modelDir.path, isDirectory: &isDir),
            !isDir.boolValue
        {
            try? FileManager.default.removeItem(at: result.modelDir)
        }

        if result.pending.isEmpty {
            statuses[model.id] = .downloading(progress: 1.0)
            return
        }

        try await downloadPendingFiles(modelID: model.id, repo: repo, result: result)
    }

    // MARK: - Companion Files

    /// Fetches the model's companion files (see `CompanionFile`) into the
    /// model folder. Idempotent: files already present at their expected size
    /// are skipped, truncated leftovers are re-downloaded.
    private func ensureCompanionFiles(for model: ModelDefinition) async throws {
        guard !model.companionFiles.isEmpty, let folder = modelPath(for: model.id) else { return }

        for (repo, files) in Dictionary(grouping: model.companionFiles, by: \.repo) {
            let entries = try await fetching.listFiles(in: repo, recursive: false)

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

                try await fetching.fetchFile(at: file.path, from: repo, to: target)
                Log.general.info("Companion file downloaded: \(file.path) for \(model.displayName)")
            }
        }
    }

    // MARK: - Verify & Repair

    func verifyAndRepair(modelID: String) {
        guard let model = definitions.first(where: { $0.id == modelID }) else { return }
        guard case .huggingFace(let repo, _, let pathPrefix) = model.source else { return }

        runLifecycle(
            for: model,
            occupiedWhen: { status in
                switch status {
                case .downloading, .verifying: true
                default: false
                }
            },
            inProgress: .verifying(progress: 0),
            failureLabel: "Verify",
            body: { [weak self] in
                guard let self else { return }
                let result = try await self.checkFiles(
                    model: model, repo: repo, pathPrefix: pathPrefix)

                self.statuses[modelID] = .verifying(progress: 1.0)

                if result.needsRepair {
                    Log.general.info(
                        "Verify: \(model.displayName) — \(result.pending.count)/\(result.totalFiles) files need repair"
                    )
                    try await self.downloadPendingFiles(
                        modelID: modelID, repo: repo, result: result)
                    Log.general.info("Repair complete: \(model.displayName)")
                } else {
                    Log.general.info(
                        "Verify OK: \(model.displayName) — \(result.totalFiles) files valid")
                }
            }
        )
    }

    // MARK: - Cancel

    /// Cancels the in-flight operation. The final status is owned by the
    /// lifecycle's single cancellation arm, which recomputes it from disk —
    /// cancel never hard-codes a status, so a cancelled verify of a
    /// downloaded model keeps showing "downloaded".
    func cancelDownload(modelID: String) {
        downloadTasks[modelID]?.cancel()
    }

    // MARK: - Delete

    func deleteModel(modelID: String) {
        guard let model = definitions.first(where: { $0.id == modelID }) else { return }
        guard let subdir = model.cacheSubdirectory else { return }

        var deleteDir = storageRoot.appendingPathComponent(subdir)
        if let prefix = model.pathPrefix {
            deleteDir = deleteDir.appendingPathComponent(prefix)
        }

        try? FileManager.default.removeItem(at: deleteDir)
        statuses[modelID] = .notDownloaded

        Log.general.info("Deleted cached model: \(model.displayName)")
    }

    // MARK: - Path Resolution

    func modelPath(for modelID: String) -> URL? {
        guard let model = definitions.first(where: { $0.id == modelID }) else { return nil }
        guard let subdir = model.cacheSubdirectory else { return nil }

        var path = storageRoot.appendingPathComponent(subdir)
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
