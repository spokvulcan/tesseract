import Foundation

@testable import Tesseract_Agent

/// Shared directory helpers for TriAttention test files. Every TriAttention
/// test needs a scratch workspace and, for the actor-load tests, a config.json
/// that passes `ModelFingerprint.computeFingerprint` and either
/// `isParoQuantModel` (PARO path) or `isTriAttentionEligibleModel` (non-PARO
/// Qwen3.5 family path) before the real container load fails.
enum TriAttentionTestFixtures {

    /// Shape of the fake model directory to produce.
    enum Kind {
        /// Empty directory — no config.json. Used to exercise the
        /// "unknown checkpoint" fallback.
        case empty
        /// PARO-quantized Qwen3.5 (e.g. `z-lab/Qwen3.5-*-PARO`). Passes
        /// `isParoQuantModel` and `isTriAttentionEligibleModel`.
        case paro
        /// Qwen3.5-family MoE with standard MLX-native affine quantization
        /// (e.g. `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`). Does NOT pass
        /// `isParoQuantModel`, and no longer passes
        /// `isTriAttentionEligibleModel` after the MoE gate — still needed
        /// to exercise the dense-gate fallback path.
        case qwen35MoeMlxNative
    }

    static func makeScratchDir(prefix: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Compatibility shim for the original `paro: Bool` API. New call sites
    /// should pass `kind:` directly.
    static func makeFakeModelDirectory(prefix: String, paro: Bool = false) throws -> URL {
        try makeFakeModelDirectory(prefix: prefix, kind: paro ? .paro : .empty)
    }

    static func makeFakeModelDirectory(prefix: String, kind: Kind) throws -> URL {
        let url = try makeScratchDir(prefix: prefix)
        let config: String? = switch kind {
        case .empty: nil
        case .paro:
            #"""
            {
              "model_type": "qwen3_5",
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "quantization_config": {
                "quant_method": "paroquant"
              }
            }
            """#
        case .qwen35MoeMlxNative:
            #"""
            {
              "model_type": "qwen3_5_moe",
              "architectures": ["Qwen3_5MoeForConditionalGeneration"],
              "quantization": {
                "mode": "affine",
                "bits": 4,
                "group_size": 64
              }
            }
            """#
        }
        if let config {
            try Data(config.utf8).write(to: url.appendingPathComponent("config.json", isDirectory: false))
        }
        return url
    }
}
