//
//  ModelCatalog+Vision.swift
//  tesseract
//
//  The catalog's vision questions, which only the Mac's agent asks.
//

import Foundation

extension ModelCatalog {

    /// The **Vision-Capable Model** rule: the checkpoint on disk declares
    /// image input — the Qwen3.5 family with a `vision_config` block (via
    /// `ModelIdentity`). Pure given its input; memoization is the caller's —
    /// the download manager holds the per-id cache.
    nonisolated static func isVisionCapable(directory: URL) -> Bool {
        ModelIdentity(directory: directory).imageKeying != nil
    }
}
