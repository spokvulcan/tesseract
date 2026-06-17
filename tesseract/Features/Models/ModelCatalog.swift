//
//  ModelCatalog.swift
//  tesseract
//

import Foundation

/// The read-model joining the static `ModelDefinition` table with live download
/// state — the one home for the "which downloaded models are in category X / is
/// this id downloaded / is it vision-capable" questions that callers used to
/// re-derive inline as `filter { $0.category == … } × case .downloaded`.
///
/// Pure: data in, answers out. The live accessors that supply `ModelDefinition.all`
/// and the current `statuses` (and memoize the vision probe) live on
/// `ModelDownloadManager`, where a view that already observes it re-renders for
/// free. See `CONTEXT.md` → Model catalog.
nonisolated enum ModelCatalog {

    /// Downloaded models in a category, in catalogue order.
    static func downloaded(
        in category: ModelCategory,
        definitions: [ModelDefinition],
        statuses: [String: ModelStatus]
    ) -> [ModelDefinition] {
        definitions.filter { $0.category == category && isDownloaded($0.id, statuses: statuses) }
    }

    /// Whether a model id is present on disk.
    static func isDownloaded(_ id: String, statuses: [String: ModelStatus]) -> Bool {
        if case .downloaded = statuses[id] { return true }
        return false
    }

    /// The single vision-detection rule, applied to an on-disk model directory:
    /// the Qwen3.5 family with a `vision_config` block (via `ModelIdentity`).
    /// Pure given the directory; memoization is the caller's — the download
    /// manager holds the per-id cache.
    nonisolated static func isVisionCapable(directory: URL) -> Bool {
        ModelIdentity(directory: directory).imageKeying != nil
    }
}
