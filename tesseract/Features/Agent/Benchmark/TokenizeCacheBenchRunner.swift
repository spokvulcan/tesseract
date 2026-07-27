import Foundation
import MLXHuggingFace
import MLXLMCommon
import os
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

/// Render+token cache benchmark (`--tokenize-cache-bench`) — experiment C25.
///
/// Measures the C25 prepare path (`RenderTokenCache.resolve`) against the
/// fused `applyChatTemplate` baseline on a simulated 12-turn agent trajectory
/// at production scale (2K-token system prompt, 40 tool specs, history
/// growing ~700 tokens/turn). The previous render's generation-prompt tail
/// (`<|im_start|>assistant\n<think>\n`) plus adversarial junction classes
/// (letter / space / emoji / CRLF / digit reply starters) force the
/// trim-back path every growing turn.
///
/// The intrinsic gate: every turn asserts the C25 token list equals
/// `applyChatTemplate(...)` exactly, plus a per-turn assertion that
/// `renderChatTemplate` + `encode(rendered)` == `applyChatTemplate` (the
/// Layer 1/2 split). Any mismatch is a FAIL line and a non-zero exit.
///
/// Tokenizer-only by design: the cache path is render+encode, so the harness
/// exercises the real tokenizer/template through the same
/// `#huggingFaceTokenizerLoader` adaptor the app loads with, on a fresh
/// `RenderTokenCache` instance (isolated from `.shared`; identical code).
@MainActor
final class TokenizeCacheBenchRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("tokenize-cache-bench")

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    /// Which cache outcome a turn must produce.
    private enum Expectation {
        case missCold
        case hit
        case hitRepeat
        case miss
    }

    func run() async throws {
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)

        let modelDir = try runner.resolveModelDirectory()
        log("Loading tokenizer from: \(modelDir.path)")
        let tokenizer = try await (#huggingFaceTokenizerLoader()).load(from: modelDir)
        guard let rendering = tokenizer as? any ChatTemplateRendering else {
            log("FAIL: loaded tokenizer is not ChatTemplateRendering — Layer 2 wiring broken")
            logFileHandle?.closeFile()
            throw NSError(domain: "TokenizeCacheBench", code: 3)
        }
        let fingerprint =
            (try? ModelFingerprint.computeFingerprint(modelDir: modelDir)) ?? "unavailable"
        log("model fingerprint: \(fingerprint.prefix(16))…")

        let turns = Self.buildTrajectory()
        let toolSpecs = Self.makeToolSpecs()

        let cache = RenderTokenCache()
        var tokenMismatches = 0
        var parityFailures = 0
        var pathFailures = 0
        var totalBaselineMs = 0.0
        var totalC25Ms = 0.0
        var hitTurns = 0

        log(
            "turn             msgs  tokens  renderMs  baselineMs    c25Ms  path                  exact"
        )
        for turn in turns {
            // Layer 1/2 parity: render + encode(rendered) == applyChatTemplate.
            let renderStart = ContinuousClock.now
            let rendered = try rendering.renderChatTemplate(
                messages: turn.messages, tools: toolSpecs, additionalContext: nil)
            let renderMs = Self.ms(since: renderStart)
            let splitTokens = tokenizer.encode(text: rendered, addSpecialTokens: false)

            let baselineStart = ContinuousClock.now
            let baseline = try tokenizer.applyChatTemplate(
                messages: turn.messages, tools: toolSpecs, additionalContext: nil)
            let baselineMs = Self.ms(since: baselineStart)
            if splitTokens != baseline {
                parityFailures += 1
                log("FAIL: split render+encode != applyChatTemplate on \(turn.label)")
            }

            // C25 path.
            let c25Start = ContinuousClock.now
            let resolution = try cache.resolve(
                tokenizer: tokenizer,
                messages: turn.messages,
                tools: toolSpecs,
                additionalContext: nil,
                modelFingerprint: fingerprint
            )
            let c25Ms = Self.ms(since: c25Start)
            guard let resolution else {
                log("FAIL: resolve returned nil on \(turn.label) (tokenizer is rendering-capable)")
                tokenMismatches += 1
                continue
            }

            let exact = resolution.tokens == baseline
            if !exact {
                tokenMismatches += 1
                let divergence = zip(resolution.tokens, baseline).enumerated()
                    .first(where: { $0.element.0 != $0.element.1 })?.offset
                log(
                    "FAIL: \(turn.label) token mismatch at index \(divergence ?? -1) "
                        + "(c25=\(resolution.tokens.count) vs baseline=\(baseline.count))"
                )
            }
            if !Self.pathMatches(resolution.path, turn.expect) {
                pathFailures += 1
                log(
                    "FAIL: \(turn.label) path \(Self.describe(path: resolution.path)) "
                        + "did not meet expectation \(turn.expect)"
                )
            }
            let pathString = Self.describe(path: resolution.path)
            if case .hit = resolution.path {
                hitTurns += 1
                totalBaselineMs += baselineMs
                totalC25Ms += c25Ms
            }
            log(
                Self.pad(turn.label, 16)
                    + Self.pad("\(turn.messages.count)", 5)
                    + Self.pad("\(baseline.count)", 8)
                    + Self.pad(String(format: "%.2f", renderMs), 10)
                    + Self.pad(String(format: "%.2f", baselineMs), 12)
                    + Self.pad(String(format: "%.2f", c25Ms), 8)
                    + Self.pad(pathString, 22)
                    + (exact ? "PASS" : "FAIL")
            )
        }

        let stats = cache.statsSnapshot()
        let trimSummary = stats.trimHistogram.sorted(by: { $0.key < $1.key })
            .map { "k\($0.key):\($0.value)" }.joined(separator: ",")
        log("")
        log(
            "stats: hits=\(stats.hits) repeats=\(stats.repeats) misses=\(stats.misses) "
                + "trimHistogram=[\(trimSummary)] "
                + "junctionFailures=\(stats.junctionFailures) "
                + "windowEnlargements=\(stats.windowEnlargements)"
        )
        log(
            String(
                format:
                    "SUMMARY: hit turns=%d, baseline %.2f ms vs c25 %.2f ms per hit turn — saves %.2f ms (%.1f%%), token mismatches=%d, parity failures=%d, path failures=%d",
                hitTurns,
                hitTurns > 0 ? totalBaselineMs / Double(hitTurns) : 0,
                hitTurns > 0 ? totalC25Ms / Double(hitTurns) : 0,
                hitTurns > 0 ? (totalBaselineMs - totalC25Ms) / Double(hitTurns) : 0,
                totalBaselineMs > 0 ? (totalBaselineMs - totalC25Ms) / totalBaselineMs * 100 : 0,
                tokenMismatches,
                parityFailures,
                pathFailures
            ))
        let failed = tokenMismatches > 0 || parityFailures > 0 || pathFailures > 0
        log(failed ? "Overall: FAIL" : "Overall: PASS")
        logFileHandle?.closeFile()
        if failed {
            throw NSError(domain: "TokenizeCacheBench", code: 2)
        }
    }

    /// Production-scale stable head: 40 tool specs.
    private static func makeToolSpecs() -> [ToolSpec] {
        (0..<40).map { i in
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
    }

    /// The 12-turn trajectory: turn N holds user_1...user_N with interleaved
    /// assistant replies (~500 tokens each, users ~200) whose starters rotate
    /// through the adversarial junction classes; plus the identical-repeat,
    /// edited-history, and unrelated turns.
    private static func buildTrajectory() -> [(
        label: String, messages: [[String: any Sendable]], expect: Expectation
    )] {
        let systemFiller =
            "You are a careful, methodical assistant working on the user's Mac. "
            + "You plan before acting, read files before editing them, and keep "
            + "answers short and factual. "
        let systemPrompt = String(repeating: systemFiller, count: 160)

        // Adversarial junction classes for the assistant-reply starters.
        let replyStarters = [
            "The analysis is complete. ",  // letter
            " 42 files matched the query. ",  // leading space + digits
            "🙂 Done with that step. ",  // emoji
            "\r\nNext, I checked the logs. ",  // CRLF
            "7/7 checks passed. ",  // digit
            " Ordinary continuation after a space.",  // trailing-space junction
        ]
        let replyFiller =
            "I read the file, compared it against the expected output, and "
            + "recorded the difference in the working notes before moving on. "
        let userFiller =
            "Please continue with the next step of the investigation and "
            + "report anything unusual you find along the way. "

        var turns: [(label: String, messages: [[String: any Sendable]], expect: Expectation)] = []
        var history: [[String: any Sendable]] = [
            ["role": "user", "content": "Step 1: " + String(repeating: userFiller, count: 3)]
        ]
        for turn in 1...12 {
            let messages: [[String: any Sendable]] =
                [["role": "system", "content": systemPrompt]] + history
            turns.append(
                (
                    label: "turn \(turn)",
                    messages: messages,
                    expect: turn == 1 ? .missCold : .hit
                ))
            let starter = replyStarters[(turn - 1) % replyStarters.count]
            history.append([
                "role": "assistant",
                "content": starter + String(repeating: replyFiller, count: 10),
            ])
            history.append([
                "role": "user",
                "content": "Step \(turn + 1): " + String(repeating: userFiller, count: 3),
            ])
        }
        // Identical repeat of the last turn.
        turns.append((label: "repeat", messages: turns.last!.messages, expect: .hitRepeat))
        // Edited history: turn 12 with the 3rd user message edited — must miss.
        var edited = turns[11].messages
        edited[5] = [
            "role": "user",
            "content": "EDITED: " + (edited[5]["content"] as? String ?? ""),
        ]
        turns.append((label: "edited-history", messages: edited, expect: .miss))
        // Unrelated prompt: fresh conversation — must miss.
        turns.append(
            (
                label: "unrelated",
                messages: [
                    ["role": "system", "content": "You are a different assistant."],
                    ["role": "user", "content": "Unrelated request."],
                ],
                expect: .miss
            ))
        return turns
    }

    private static func pathMatches(_ path: RenderTokenCache.Path, _ expect: Expectation) -> Bool {
        switch (path, expect) {
        case (.miss(.cold), .missCold):
            return true
        case (.hit, .hit), (.hitRepeat, .hitRepeat), (.miss, .miss):
            return true
        default:
            return false
        }
    }

    private static func describe(path: RenderTokenCache.Path) -> String {
        switch path {
        case .hit(let trimmedBy):
            return trimmedBy == 0 ? "hit(k=0)" : "hit(trim=\(trimmedBy))"
        case .hitRepeat:
            return "hitRepeat"
        case .miss(let reason):
            return "miss(\(reason))"
        }
    }

    private static func pad(_ string: String, _ width: Int) -> String {
        string.count >= width
            ? string + " " : string + String(repeating: " ", count: width - string.count)
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
