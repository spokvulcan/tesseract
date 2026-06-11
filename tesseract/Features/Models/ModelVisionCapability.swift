//
//  ModelVisionCapability.swift
//  tesseract
//

import Foundation

/// Answers "can this downloaded model serve images?" without loading it —
/// the capability fact behind ADR-0008's vision-load rule and the
/// Integration snapshot's modality advertising.
///
/// The detection is `ModelIdentity`'s: the Qwen3.5 family with a
/// `vision_config` block in the model's own `config.json`. One rule, two
/// consumers — the arbiter's lease path and the client-config generators —
/// so the config a client receives can never disagree with what the server
/// will load.
@MainActor
final class ModelVisionCapability {

    private let downloads: ModelDownloadManager
    /// Probes hit disk (config.json + chat template) — memoized per model.
    /// Only positive-knowledge entries are cached: an undownloaded model is
    /// answered `false` without caching, so a later download is re-probed.
    private var cache: [String: Bool] = [:]

    init(downloads: ModelDownloadManager) {
        self.downloads = downloads
    }

    func isVisionCapable(_ modelID: String) -> Bool {
        if let cached = cache[modelID] { return cached }
        guard case .downloaded = downloads.statuses[modelID],
              let directory = downloads.modelPath(for: modelID)
        else { return false }
        let capable = Self.isVisionCapable(directory: directory)
        cache[modelID] = capable
        return capable
    }

    /// The single detection rule, applied to an on-disk model directory.
    nonisolated static func isVisionCapable(directory: URL) -> Bool {
        ModelIdentity(directory: directory).imageKeying != nil
    }
}
