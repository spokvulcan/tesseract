import Foundation

@testable import Tesseract_Agent

/// Shared directory helpers for TriAttention test files. Every TriAttention
/// test needs a scratch workspace and, for the actor-load tests, a PARO-shaped
/// config.json that passes `ModelFingerprint.computeFingerprint` and
/// `isParoQuantModel` before the real container load fails.
enum TriAttentionTestFixtures {

    static func makeScratchDir(prefix: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func makeFakeModelDirectory(prefix: String, paro: Bool = false) throws -> URL {
        let url = try makeScratchDir(prefix: prefix)
        guard paro else { return url }
        let config = """
        {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "quantization_config": {
            "quant_method": "paroquant"
          }
        }
        """
        try Data(config.utf8).write(to: url.appendingPathComponent("config.json", isDirectory: false))
        return url
    }
}
