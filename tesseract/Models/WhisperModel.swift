//
//  WhisperModel.swift
//  tesseract
//

import Foundation

/// The bundled Whisper model used for transcription.
/// Large V3 Turbo provides the best balance of accuracy and speed with 100+ language support.
enum WhisperModel {
    static let bundled = "openai_whisper-large-v3-v20240930_turbo"

    static let displayName = "Large V3 Turbo"
    static let sizeGB: Double = 1.5
    static let recommendedRAMGB = 8
    static let languageCount = 100

    static let description = "State-of-the-art multilingual model with best accuracy and speed."

    /// Returns the URL to the bundled model folder in the app bundle.
    /// The model files are copied directly to the Resources folder by Xcode's build system.
    static var bundledModelURL: URL? {
        Bundle.main.resourceURL
    }
}
