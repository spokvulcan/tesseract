//
//  ModelCatalog+Vision.swift
//  tesseract
//
//  The catalog's vision questions, which only the Mac's agent asks.
//

import Foundation

extension ModelCatalog {

    /// What the checkpoint on disk declares: the Qwen3.5 family with a
    /// `vision_config` block (via `ModelIdentity`). An observation, not the
    /// capability — that is ``isVisionCapable(definition:directory:)``.
    nonisolated static func checkpointDeclaresImageInput(directory: URL) -> Bool {
        ModelIdentity(directory: directory).imageKeying != nil
    }

    /// The **Vision-Capable Model** rule: the checkpoint declares image input
    /// *and* the catalog entry carries no **Text-Only Override**. Pure given
    /// its inputs; memoization is the caller's — the download manager holds
    /// the per-id cache.
    nonisolated static func isVisionCapable(definition: ModelDefinition, directory: URL) -> Bool {
        !definition.textOnlyOverride && checkpointDeclaresImageInput(directory: directory)
    }
}
