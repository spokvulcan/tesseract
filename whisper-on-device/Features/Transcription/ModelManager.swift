//
//  ModelManager.swift
//  whisper-on-device
//

import Foundation
import Combine
@preconcurrency import WhisperKit

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var availableModels: [WhisperModel] = WhisperModel.allCases
    @Published private(set) var downloadedModels: Set<WhisperModel> = []
    @Published private(set) var downloadProgress: [WhisperModel: Double] = [:]
    @Published private(set) var isDownloading: [WhisperModel: Bool] = [:]

    private let fileManager = FileManager.default
    private var downloadTasks: [WhisperModel: Task<Void, Error>] = [:]

    /// Key for storing model paths in UserDefaults
    private static let downloadedModelPathsKey = "downloadedModelPaths"

    /// HuggingFace cache directory where WhisperKit stores downloaded models
    private var huggingFaceCacheDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("huggingface/models/argmaxinc/whisperkit-coreml", isDirectory: true)
    }

    init() {
        scanForDownloadedModels()
    }

    func downloadModel(_ model: WhisperModel) async throws {
        guard downloadTasks[model] == nil else { return }

        isDownloading[model] = true
        downloadProgress[model] = 0

        do {
            // Use WhisperKit's built-in model download with progress tracking
            // Capture model value and self reference for the sendable closure
            let modelVariant = model.rawValue
            let modelPath = try await WhisperKit.download(
                variant: modelVariant,
                progressCallback: { @Sendable [weak self] progress in
                    let fraction = progress.fractionCompleted
                    Task { @MainActor [weak self] in
                        self?.downloadProgress[model] = fraction
                    }
                }
            )

            // Persist the model path for reliable detection
            persistModelPath(model, path: modelPath)

            downloadedModels.insert(model)
            downloadProgress[model] = 1.0
            isDownloading[model] = false

            print("Model downloaded to: \(modelPath)")
        } catch {
            isDownloading[model] = false
            downloadProgress[model] = nil
            throw error
        }
    }

    func cancelDownload(_ model: WhisperModel) {
        downloadTasks[model]?.cancel()
        downloadTasks[model] = nil
        isDownloading[model] = false
        downloadProgress[model] = nil
    }

    func deleteModel(_ model: WhisperModel) throws {
        // Find the actual model path (could be in HuggingFace cache or persisted location)
        guard let modelPath = getLocalModelPath(model) else {
            // Model not found on disk, just remove from tracking
            downloadedModels.remove(model)
            removePersistedModelPath(model)
            return
        }

        // Delete the model directory
        if fileManager.fileExists(atPath: modelPath.path) {
            try fileManager.removeItem(at: modelPath)
        }

        downloadedModels.remove(model)
        removePersistedModelPath(model)
    }

    func getLocalModelPath(_ model: WhisperModel) -> URL? {
        // First, check the persisted path from UserDefaults
        if let persistedPath = getPersistedModelPath(model) {
            if fileManager.fileExists(atPath: persistedPath.path) && isValidModelDirectory(persistedPath) {
                return persistedPath
            }
        }

        // Fall back to scanning HuggingFace cache directory
        let modelPath = huggingFaceCacheDirectory.appendingPathComponent(model.rawValue)
        if fileManager.fileExists(atPath: modelPath.path) && isValidModelDirectory(modelPath) {
            return modelPath
        }

        return nil
    }

    func isModelDownloaded(_ model: WhisperModel) -> Bool {
        downloadedModels.contains(model)
    }

    func diskSpaceUsed() -> Int64 {
        var totalSize: Int64 = 0

        for model in downloadedModels {
            if let path = getLocalModelPath(model) {
                totalSize += directorySize(at: path)
            }
        }

        return totalSize
    }

    // MARK: - Private

    private func scanForDownloadedModels() {
        downloadedModels.removeAll()

        for model in WhisperModel.allCases {
            if getLocalModelPath(model) != nil {
                downloadedModels.insert(model)
            }
        }
    }

    /// Validates that a directory contains the required WhisperKit model files
    private func isValidModelDirectory(_ url: URL) -> Bool {
        // Check for presence of key model files (encoder and decoder)
        let encoderPath = url.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = url.appendingPathComponent("TextDecoder.mlmodelc")

        return fileManager.fileExists(atPath: encoderPath.path) ||
               fileManager.fileExists(atPath: decoderPath.path)
    }

    // MARK: - Model Path Persistence

    private func persistModelPath(_ model: WhisperModel, path: URL) {
        var paths = UserDefaults.standard.dictionary(forKey: Self.downloadedModelPathsKey) as? [String: String] ?? [:]
        paths[model.rawValue] = path.path
        UserDefaults.standard.set(paths, forKey: Self.downloadedModelPathsKey)
    }

    private func getPersistedModelPath(_ model: WhisperModel) -> URL? {
        guard let paths = UserDefaults.standard.dictionary(forKey: Self.downloadedModelPathsKey) as? [String: String],
              let pathString = paths[model.rawValue] else {
            return nil
        }
        return URL(fileURLWithPath: pathString)
    }

    private func removePersistedModelPath(_ model: WhisperModel) {
        var paths = UserDefaults.standard.dictionary(forKey: Self.downloadedModelPathsKey) as? [String: String] ?? [:]
        paths.removeValue(forKey: model.rawValue)
        UserDefaults.standard.set(paths, forKey: Self.downloadedModelPathsKey)
    }

    // MARK: - Utilities

    private func directorySize(at url: URL) -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var size: Int64 = 0

        for case let fileURL as URL in enumerator {
            guard let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize else {
                continue
            }
            size += Int64(fileSize)
        }

        return size
    }
}
