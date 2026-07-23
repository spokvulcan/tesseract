import Foundation
import MLXHuggingFace
import MLXLMCommon
import os
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

/// Stable-prefix detect benchmark (`--prefix-detect-bench`) — measures the
/// per-request cost of `StablePrefixDetector.detect` (two Jinja renders +
/// BPE encodes of system+tools) at production scale vs the memoized hit,
/// and verifies the memo returns the identical boundary.
///
/// The memo's hit rate in production is near-100% (the stable prefix is
/// stable by definition), so the hit path is the steady state and the miss
/// path is what the memo saves per request.
@MainActor
final class PrefixDetectBenchRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("prefix-detect-bench")

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)

        let modelDir = try runner.resolveModelDirectory()
        log("Loading tokenizer from: \(modelDir.path)")
        let tokenizer = try await (#huggingFaceTokenizerLoader()).load(from: modelDir)

        // Production-scale inputs: ~10K-token system prompt + 40 tool specs.
        let filler =
            "You are a careful, methodical assistant working on the user's Mac. "
            + "You plan before acting, read files before editing them, and keep "
            + "answers short and factual. "
        let systemPrompt = String(repeating: filler, count: 320)
        let toolSpecs: [ToolSpec] = (0..<40).map { i in
            [
                "type": "function",
                "function": [
                    "name": "tool_\(i)",
                    "description": "Tool number \(i) used for agent operation \(i).",
                    "parameters": [
                        "type": "object",
                        "required": ["input"],
                        "properties": [
                            "input": [
                                "type": "string",
                                "description": "Input for tool_\(i).",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        }
        let fullTokens = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": "Summarize the repository status."],
            ],
            tools: toolSpecs,
            additionalContext: nil
        )
        log("full prompt = \(fullTokens.count) tokens")

        // Sanity: miss-path detect must produce a boundary.
        StablePrefixDetector.resetMemo()
        guard
            let boundary = try StablePrefixDetector.detect(
                systemPrompt: systemPrompt,
                toolSpecs: toolSpecs,
                additionalContext: nil,
                fullTokens: fullTokens,
                tokenizer: tokenizer
            )
        else {
            throw NSError(
                domain: "PrefixDetectBench", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "detect returned nil at production scale"])
        }
        log("stable-prefix boundary = \(boundary) tokens")

        // A/B: miss path (resetMemo + full two-probe detect) vs hit path.
        var missMs: [Double] = []
        var hitMs: [Double] = []
        var mismatches = 0
        for round in 0..<6 {
            let t0 = ContinuousClock.now
            StablePrefixDetector.resetMemo()
            let missResult = try StablePrefixDetector.detect(
                systemPrompt: systemPrompt,
                toolSpecs: toolSpecs,
                additionalContext: nil,
                fullTokens: fullTokens,
                tokenizer: tokenizer
            )
            missMs.append(Self.ms(since: t0))

            let t1 = ContinuousClock.now
            let hitResult = try StablePrefixDetector.detect(
                systemPrompt: systemPrompt,
                toolSpecs: toolSpecs,
                additionalContext: nil,
                fullTokens: fullTokens,
                tokenizer: tokenizer
            )
            hitMs.append(Self.ms(since: t1))
            if missResult != hitResult { mismatches += 1 }
            log(
                "round \(round): miss=\(String(format: "%.2f", missMs.last!))ms "
                    + "hit=\(String(format: "%.2f", hitMs.last!))ms "
                    + "identical=\(missResult == hitResult)"
            )
        }

        let missMean = missMs.reduce(0, +) / Double(missMs.count)
        let hitMean = hitMs.reduce(0, +) / Double(hitMs.count)
        log(
            String(
                format:
                    "SUMMARY: miss %.2f ms vs hit %.2f ms per request — saves %.2f ms (%.1f%% of the detect cost), mismatches=%d",
                missMean, hitMean, missMean - hitMean,
                missMean > 0 ? (missMean - hitMean) / missMean * 100 : 0, mismatches))
        log(mismatches == 0 ? "Overall: PASS" : "Overall: FAIL")
        logFileHandle?.closeFile()
        if mismatches > 0 {
            throw NSError(domain: "PrefixDetectBench", code: 2)
        }
    }

    private static func ms(since start: ContinuousClock.Instant) -> Double {
        let c = start.duration(to: .now).components
        return (Double(c.seconds) + Double(c.attoseconds) * 1e-18) * 1e3
    }

    private func log(_ message: String) {
        logger.info("\(message, privacy: .public)")
        if let data = (message + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }
}
