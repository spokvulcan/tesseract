//
//  WhisperModelContract.swift
//  tesseract
//
//  The one home for "what files make a downloaded Whisper model loadable".
//  Both the availability query (App Bindings' model-path input) and the
//  engine's load-time verification consult this contract, so a model-format
//  change is a one-site edit.
//

import Foundation

nonisolated enum WhisperModelContract {
    /// The files a downloaded Whisper model directory must contain before
    /// WhisperKit can load it.
    static let requiredFiles = ["AudioEncoder.mlmodelc", "TextDecoder.mlmodelc"]

    /// The required files absent from the directory — empty means loadable.
    static func missingFiles(
        at modelPath: URL, fileManager: FileManager = .default
    ) -> [String] {
        requiredFiles.filter {
            !fileManager.fileExists(atPath: modelPath.appendingPathComponent($0).path)
        }
    }

    static func isComplete(at modelPath: URL, fileManager: FileManager = .default) -> Bool {
        missingFiles(at: modelPath, fileManager: fileManager).isEmpty
    }
}
