//
//  ModelDownloadManager+Vision.swift
//  tesseract
//

extension ModelDownloadManager {
    /// Whether a downloaded model can serve images — the memoized **Vision
    /// Capability Memo**. Replaces the stranded `ModelVisionCapability` class.
    func isVisionCapable(_ id: String) -> Bool {
        if let cached = visionCache[id] { return cached }
        guard isDownloaded(id), let directory = modelPath(for: id),
            let definition = definitions.first(where: { $0.id == id })
        else { return false }
        let capable = ModelCatalog.isVisionCapable(definition: definition, directory: directory)
        visionCache[id] = capable
        return capable
    }
}
