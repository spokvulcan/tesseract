//
//  ModelDownloadManager+Vision.swift
//  tesseract
//

extension ModelDownloadManager {
    /// Whether a downloaded model can serve images — the memoized **Vision
    /// Capability Memo**. Replaces the stranded `ModelVisionCapability` class.
    func isVisionCapable(_ id: String) -> Bool {
        if let cached = visionCache[id] { return cached }
        guard isDownloaded(id), let directory = modelPath(for: id) else { return false }
        let capable = ModelCatalog.isVisionCapable(directory: directory)
        visionCache[id] = capable
        return capable
    }
}
