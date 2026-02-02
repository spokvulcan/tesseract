//
//  ModelManager.swift
//  whisper-on-device
//

import Foundation

/// Manages the bundled Whisper model.
/// The model is bundled with the app, so no downloads are needed.
@MainActor
final class ModelManager {
    /// Returns the URL to the bundled model folder.
    /// The model is always available since it's bundled with the app.
    func getBundledModelPath() -> URL? {
        WhisperModel.bundledModelURL
    }

    /// Validates that the bundled model exists and has required files.
    func isModelAvailable() -> Bool {
        guard let modelPath = getBundledModelPath() else { return false }

        let fileManager = FileManager.default
        let encoderPath = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = modelPath.appendingPathComponent("TextDecoder.mlmodelc")

        return fileManager.fileExists(atPath: encoderPath.path) &&
               fileManager.fileExists(atPath: decoderPath.path)
    }
}
