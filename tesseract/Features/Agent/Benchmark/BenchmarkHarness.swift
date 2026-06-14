//
//  BenchmarkHarness.swift
//  tesseract
//
//  Shared fixtures and plumbing for the loaded-model verification harnesses
//  (`HybridCacheCorrectnessRunner`, `ParoQuantVLMSmokeRunner`,
//  `PrefixCacheE2ERunner`): the deterministic inputs that must stay
//  byte-identical across harnesses, tensor comparison, and the check-report
//  JSON shape. One home so the harnesses cannot drift apart on the fixtures
//  their cross-referenced claims depend on.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon

nonisolated enum BenchmarkHarness {

    // MARK: - Check reports

    struct CheckResult: Codable, Sendable {
        let name: String
        let passed: Bool
        let detail: String
    }

    /// Write the harness check-report JSON (`date`, `model`, `passed`,
    /// `checks`) to `<reportDir>/<filePrefix>_<stamp>.json`; returns the
    /// written file's URL.
    @discardableResult
    static func writeReport(
        checks: [CheckResult],
        allPassed: Bool,
        modelName: String,
        reportDir: URL,
        filePrefix: String
    ) throws -> URL {
        struct Report: Codable {
            let date: String
            let model: String
            let passed: Bool
            let checks: [CheckResult]
        }
        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            model: modelName,
            passed: allPassed,
            checks: checks
        )
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let url = reportDir.appendingPathComponent(
            "\(filePrefix)_\(formatter.string(from: Date())).json"
        )
        try encoder.encode(report).write(to: url)
        return url
    }

    // MARK: - Tensor comparison

    /// Maximum absolute element-wise difference between two arrays. Casts
    /// bf16 to f32 to avoid silent precision loss in subtraction.
    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return Float.infinity }
        let aF = a.dtype == .bfloat16 ? a.asType(.float32) : a
        let bF = b.dtype == .bfloat16 ? b.asType(.float32) : b
        return (aF - bF).abs().max().item(Float.self)
    }

    // MARK: - Deterministic prompts

    /// Deterministic filler that encodes to at least `targetTokens` tokens.
    /// Repeats a fixed passage so the same target length yields identical
    /// bytes on every run — identical model state, bitwise-stable assertions.
    static func promptText(targetTokens: Int, tokenizer: any Tokenizer) -> String {
        let base = """
            The cache verification harness must produce a deterministic, \
            reproducible token sequence. We compose long passages from a \
            fixed lexicon so the same target length yields identical bytes \
            on every run, which in turn guarantees identical model state \
            and lets the bitwise equality assertions hold. Lorem ipsum dolor \
            sit amet, consectetur adipiscing elit. Sed do eiusmod tempor \
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim \
            veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip \
            ex ea commodo consequat. Duis aute irure dolor in reprehenderit \
            in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \
            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui \
            officia deserunt mollit anim id est laborum.
            """

        var combined = base
        while tokenizer.encode(text: combined, addSpecialTokens: false).count < targetTokens {
            combined += " " + base
        }
        return combined
    }

    /// `promptText` encoded and truncated to exactly `targetTokens` tokens.
    static func promptTokens(targetTokens: Int, tokenizer: any Tokenizer) -> [Int] {
        Array(
            tokenizer.encode(
                text: promptText(targetTokens: targetTokens, tokenizer: tokenizer),
                addSpecialTokens: false
            ).prefix(targetTokens)
        )
    }

    // MARK: - Deterministic images

    /// Deterministic RGBA byte-pattern image — identical pixels on every run,
    /// no asset files, enough spatial structure that patches differ. The
    /// spike-pinned geometry facts (256×256 seed 17 → 16×16 grid, 64-token
    /// run, rope delta −56) are facts about THIS generator; every harness
    /// must draw its images from it.
    static func deterministicImage(width: Int, height: Int, seed: Int) -> CIImage {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                bytes[i] = UInt8((x * 7 + seed) % 256)
                bytes[i + 1] = UInt8((y * 5 + seed * 2) % 256)
                bytes[i + 2] = UInt8((x + y + seed * 3) % 256)
                bytes[i + 3] = 255
            }
        }
        return CIImage(
            bitmapData: Data(bytes),
            bytesPerRow: width * 4,
            size: CGSize(width: width, height: height),
            format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
    }

    /// `deterministicImage` PNG-encoded (sRGB), for harnesses exercising the
    /// byte-level attachment path (the conversation shape carries digest-keyed
    /// attachment bytes, not decoded images). `nil` when encoding fails.
    static func deterministicPNG(width: Int, height: Int, seed: Int) -> Data? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CIContext().pngRepresentation(
            of: deterministicImage(width: width, height: height, seed: seed),
            format: .RGBA8,
            colorSpace: colorSpace
        )
    }
}
