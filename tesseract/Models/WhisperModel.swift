//
//  WhisperModel.swift
//  tesseract
//

import Foundation

/// Metadata for the Whisper model used for transcription.
/// The model is downloaded on-demand via the Models page.
enum WhisperModel {
    static let modelID = "whisper-large-v3-turbo"
    static let folderName = "openai_whisper-large-v3-v20240930_turbo"

    static let displayName = "Large V3 Turbo"
    static let sizeGB: Double = 1.5
    static let recommendedRAMGB = 8
    static let languageCount = 100

    static let description = "State-of-the-art multilingual model with best accuracy and speed."
}
