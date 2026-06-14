//
//  ModelManager.swift
//  tesseract
//

import Foundation

/// Manages the Whisper model path resolution.
/// The model is downloaded via the Models page and cached locally.
@MainActor
final class ModelManager {
    private let modelDownloadManager: ModelDownloadManager
    private let selectedModelID: () -> String

    init(modelDownloadManager: ModelDownloadManager, selectedModelID: @escaping () -> String) {
        self.modelDownloadManager = modelDownloadManager
        self.selectedModelID = selectedModelID
    }

    /// Returns the URL to the selected model's downloaded folder, or nil if
    /// not downloaded.
    func getModelPath() -> URL? {
        modelDownloadManager.modelPath(for: selectedModelID())
    }

    /// Validates that the downloaded model exists and has required files.
    func isModelAvailable() -> Bool {
        guard let modelPath = getModelPath() else { return false }

        let fileManager = FileManager.default
        let encoderPath = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = modelPath.appendingPathComponent("TextDecoder.mlmodelc")

        return fileManager.fileExists(atPath: encoderPath.path)
            && fileManager.fileExists(atPath: decoderPath.path)
    }
}
