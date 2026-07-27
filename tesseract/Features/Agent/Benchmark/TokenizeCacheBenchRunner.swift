import Foundation
import MLXHuggingFace
import MLXLMCommon
import os
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

/// Render+token cache benchmark (`--tokenize-cache-bench`) — experiments
/// C25/C26/C27/C28.
///
/// Measures the C25 prepare path (`RenderTokenCache.resolve`) against the
/// fused `applyChatTemplate` baseline on a simulated 12-turn agent trajectory
/// at production scale (2K-token system prompt, 40 tool specs, history
/// growing ~700 tokens/turn). The previous render's generation-prompt tail
/// (`<|im_start|>assistant\n<think>\n`) plus adversarial junction classes
/// (letter / space / emoji / CRLF / digit reply starters) force the
/// trim-back path every growing turn.
///
/// The C27 leg: after each turn's full prepare (which stores the cache
/// entry), the planner-style truncated-at-last-user resolve runs against
/// that entry and is timed against the standalone
/// `applyChatTemplate(..., add_generation_prompt: false)` the Prefill
/// Planner pays today — asserted token-exact per turn. Turns whose last
/// message is NOT the last user (assistant tail) or that resolve against a
/// stale entry (run before their own prepare) must fall back — still exact.
///
/// The C28 post-gen leg: for each turn carrying an assistant reply, the
/// three post-generation encodes the Leaf Store phase and the Leaf Admission
/// Builder pay — the stored conversation (request + reply, generation
/// prompt OFF) and that conversation plus a synthetic user probe — run
/// through `resolveReplacingTail` against the turn's entry, timed against
/// the standalone `applyChatTemplate` encodes and asserted token-exact.
///
/// The intrinsic gate: every turn asserts the C25 token list equals
/// `applyChatTemplate(...)` exactly, plus a per-turn assertion that
/// `renderChatTemplate` + `encode(rendered)` == `applyChatTemplate` (the
/// Layer 1/2 split), plus the C27 leg's and the C28 legs' exactness. Any
/// mismatch is a FAIL line and a non-zero exit.
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
    /// Internal (not private) so `--agent-cpu-bench` can reuse the trajectory.
    enum Expectation {
        case missCold
        case hit
        case hitRepeat
        case miss
    }

    /// Which outcome a turn's C27 truncated leg must produce.
    enum TruncExpectation {
        case hit
        case fallback
    }

    /// One simulated request: the C25 prepare leg plus the C27 truncated leg.
    /// Internal (not private) so `--agent-cpu-bench` reuses the same trajectory.
    struct Turn {
        let label: String
        let messages: [[String: any Sendable]]
        let expect: Expectation
        let truncExpect: TruncExpectation
        /// Run the truncated leg BEFORE the C25 prepare — leaves the cache
        /// holding the previous turn's entry, the stale-entry fallback shape.
        let truncFirst: Bool
        /// The assistant reply appended to the conversation this turn — its
        /// presence arms the C28 post-gen legs (leaf-store + the two
        /// admission renders), all expected to hit against this turn's entry.
        let postGenReply: [String: any Sendable]?

        /// The planner's truncated input: messages through the last user
        /// message, or the whole list when there is none (production skips
        /// that call; the bench still resolves it to prove the fallback).
        var truncMessages: [[String: any Sendable]] {
            guard
                let lastUser = messages.lastIndex(where: {
                    ($0["role"] as? String) == "user"
                })
            else {
                return messages
            }
            return Array(messages[...lastUser])
        }

        init(
            label: String, messages: [[String: any Sendable]], expect: Expectation,
            truncExpect: TruncExpectation, truncFirst: Bool = false,
            postGenReply: [String: any Sendable]? = nil
        ) {
            self.label = label
            self.messages = messages
            self.expect = expect
            self.truncExpect = truncExpect
            self.truncFirst = truncFirst
            self.postGenReply = postGenReply
        }
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
        var totals = RunTotals()

        /// The C28 post-gen legs: the Leaf Store phase's stored-conversation
        /// encode and the Leaf Admission Builder's two reusable-prefix
        /// renders, each resolved through the cache against this turn's entry
        /// and timed against the standalone `applyChatTemplate` — asserted
        /// token-exact on every hit, expected to hit on every leg.
        func runPostGenLegs(_ turn: Turn, reply: [String: any Sendable]) throws -> [String] {
            let mergedContext: [String: any Sendable] = ["add_generation_prompt": false]
            let storedMessages = turn.messages + [reply]
            let probeMessage: [String: any Sendable] = [
                "role": "user", "content": "Aqkz_strip_probe",
            ]
            // (label, messages): the leaf-store encode and the two admission
            // renders — admission-stored is the same render production runs
            // twice (once in the phase, once in the builder).
            let legs: [(String, [[String: any Sendable]])] = [
                ("c28 leaf-store", storedMessages),
                ("c28 admission", storedMessages),
                ("c28 probe", storedMessages + [probeMessage]),
            ]
            var lines: [String] = []
            for (index, leg) in legs.enumerated() {
                let cachedStart = ContinuousClock.now
                let resolved = try cache.resolveReplacingTail(
                    tokenizer: tokenizer,
                    messages: leg.1,
                    tools: toolSpecs,
                    baseAdditionalContext: nil,
                    mergedAdditionalContext: mergedContext,
                    modelFingerprint: fingerprint
                )
                let cachedMs = Self.ms(since: cachedStart)
                let standaloneStart = ContinuousClock.now
                let standalone = try tokenizer.applyChatTemplate(
                    messages: leg.1, tools: toolSpecs,
                    additionalContext: mergedContext)
                let standaloneMs = Self.ms(since: standaloneStart)

                var exact = true
                var pathOK = true
                let path: String
                if let resolved {
                    path = "hit"
                    totals.postGenLegHits[index] += 1
                    totals.postGenLegMs[index] += cachedMs
                    totals.postGenLegStandaloneMs[index] += standaloneMs
                    if resolved != standalone {
                        exact = false
                        totals.postGenMismatches += 1
                        let divergence = zip(resolved, standalone).enumerated()
                            .first(where: { $0.element.0 != $0.element.1 })?.offset
                        log(
                            "FAIL: \(turn.label) \(leg.0) token mismatch at index "
                                + "\(divergence ?? -1) (c28=\(resolved.count) vs "
                                + "standalone=\(standalone.count))"
                        )
                    }
                } else {
                    // Every post-gen leg resolves against its own turn's
                    // entry — a fallback here is a path failure, not an
                    // expected shape (the unit suites own the fallback
                    // shapes).
                    path = "fallback"
                    pathOK = false
                    totals.postGenPathFailures += 1
                    log("FAIL: \(turn.label) \(leg.0) unexpectedly fell back")
                }
                lines.append(
                    Self.pad("  " + leg.0, 16)
                        + Self.pad("\(leg.1.count)", 5)
                        + Self.pad("\(standalone.count)", 8)
                        + Self.pad("", 10)
                        + Self.pad(String(format: "%.2f", standaloneMs), 12)
                        + Self.pad(String(format: "%.2f", cachedMs), 8)
                        + Self.pad(path, 22)
                        + (exact && pathOK ? "PASS" : "FAIL")
                )
            }
            return lines
        }

        /// The C27 leg: resolve the planner's truncated-at-last-user render
        /// through the cache, time it against the standalone truncated
        /// `applyChatTemplate`, and assert exact equality on a hit. C31: the
        /// leg passes the same `messagesAreEntryPrefix` assertion
        /// `PrefillPlanner` passes in production — the turn's truncated
        /// messages are a prefix of the messages the prepare leg resolved.
        func runTruncatedLeg(_ turn: Turn) throws -> String {
            let mergedContext: [String: any Sendable] = ["add_generation_prompt": false]
            let truncStart = ContinuousClock.now
            let resolved = try cache.resolveTruncated(
                tokenizer: tokenizer,
                messages: turn.truncMessages,
                tools: toolSpecs,
                baseAdditionalContext: nil,
                mergedAdditionalContext: mergedContext,
                modelFingerprint: fingerprint,
                messagesAreEntryPrefix: true
            )
            let truncMs = Self.ms(since: truncStart)
            let standaloneStart = ContinuousClock.now
            let standalone = try tokenizer.applyChatTemplate(
                messages: turn.truncMessages, tools: toolSpecs,
                additionalContext: mergedContext)
            let standaloneMs = Self.ms(since: standaloneStart)

            var exact = true
            let truncPath: String
            if let resolved {
                truncPath = "hit"
                if resolved != standalone {
                    exact = false
                    totals.truncMismatches += 1
                    let divergence = zip(resolved, standalone).enumerated()
                        .first(where: { $0.element.0 != $0.element.1 })?.offset
                    log(
                        "FAIL: \(turn.label) truncated token mismatch at index "
                            + "\(divergence ?? -1) (c27=\(resolved.count) vs "
                            + "standalone=\(standalone.count))"
                    )
                }
                totals.truncHitTurns += 1
                totals.totalTruncMs += truncMs
                totals.totalTruncStandaloneMs += standaloneMs
            } else {
                // The production fallback: the planner pays the standalone
                // encode — exact by construction.
                truncPath = "fallback"
            }
            let pathOK =
                switch turn.truncExpect {
                case .hit: resolved != nil
                case .fallback: resolved == nil
                }
            if !pathOK {
                totals.truncPathFailures += 1
                log(
                    "FAIL: \(turn.label) truncated path \(truncPath) "
                        + "did not meet expectation \(turn.truncExpect)"
                )
            }
            return Self.pad("  c27 truncated", 16)
                + Self.pad("\(turn.truncMessages.count)", 5)
                + Self.pad("\(standalone.count)", 8)
                + Self.pad("", 10)
                + Self.pad(String(format: "%.2f", standaloneMs), 12)
                + Self.pad(String(format: "%.2f", truncMs), 8)
                + Self.pad(truncPath, 22)
                + (exact && pathOK ? "PASS" : "FAIL")
        }

        log(
            "turn             msgs  tokens  renderMs  baselineMs      cMs  path                  exact"
        )
        for turn in turns {
            // A throwing leg must leave a FAIL line in the log, not a silent
            // exit — the summary is the harness's only verdict channel.
            do {
                // The stale-entry turn resolves its truncated leg against the
                // previous turn's entry (stale-entry fallback shape).
                var truncLine: String?
                if turn.truncFirst {
                    truncLine = try runTruncatedLeg(turn)
                }

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
                    totals.parityFailures += 1
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
                    log(
                        "FAIL: resolve returned nil on \(turn.label) (tokenizer is rendering-capable)"
                    )
                    totals.tokenMismatches += 1
                    continue
                }

                let exact = resolution.tokens == baseline
                if !exact {
                    totals.tokenMismatches += 1
                    let divergence = zip(resolution.tokens, baseline).enumerated()
                        .first(where: { $0.element.0 != $0.element.1 })?.offset
                    log(
                        "FAIL: \(turn.label) token mismatch at index \(divergence ?? -1) "
                            + "(c25=\(resolution.tokens.count) vs baseline=\(baseline.count))"
                    )
                }
                if !Self.pathMatches(resolution.path, turn.expect) {
                    totals.pathFailures += 1
                    log(
                        "FAIL: \(turn.label) path \(Self.describe(path: resolution.path)) "
                            + "did not meet expectation \(turn.expect)"
                    )
                }
                let pathString = Self.describe(path: resolution.path)
                if case .hit = resolution.path {
                    totals.hitTurns += 1
                    totals.totalBaselineMs += baselineMs
                    totals.totalC25Ms += c25Ms
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
                if let truncLine {
                    log(truncLine)
                } else {
                    log(try runTruncatedLeg(turn))
                }
                // The C28 post-gen legs run against this turn's own entry
                // (stored by the C25 leg above; the C27 leg never mutates).
                if let postGenReply = turn.postGenReply {
                    for line in try runPostGenLegs(turn, reply: postGenReply) {
                        log(line)
                    }
                }
            } catch {
                log("FAIL: \(turn.label) threw: \(error)")
                logFileHandle?.closeFile()
                throw error
            }
        }

        let failed = logSummaryAndEvaluate(totals, stats: cache.statsSnapshot())
        logFileHandle?.closeFile()
        if failed {
            throw NSError(domain: "TokenizeCacheBench", code: 2)
        }
    }

    /// Accumulated per-leg counters and timings for one `run()` — factored
    /// out so the turn loop, the nested legs, and the summary printer each
    /// stay small.
    private struct RunTotals {
        var tokenMismatches = 0
        var parityFailures = 0
        var pathFailures = 0
        var totalBaselineMs = 0.0
        var totalC25Ms = 0.0
        var hitTurns = 0
        var truncMismatches = 0
        var truncPathFailures = 0
        var truncHitTurns = 0
        var totalTruncMs = 0.0
        var totalTruncStandaloneMs = 0.0
        var postGenMismatches = 0
        var postGenPathFailures = 0
        /// Per-leg hit counts and timings: leaf-store, admission-stored,
        /// admission-probe — the three post-generation encodes.
        var postGenLegHits = [0, 0, 0]
        var postGenLegMs = [0.0, 0.0, 0.0]
        var postGenLegStandaloneMs = [0.0, 0.0, 0.0]
    }

    /// Prints the stats + per-leg summaries and returns the overall failure
    /// flag (`true` = any mismatch or path failure).
    private func logSummaryAndEvaluate(
        _ totals: RunTotals, stats: RenderTokenCache.Stats
    ) -> Bool {
        let trimSummary = stats.trimHistogram.sorted(by: { $0.key < $1.key })
            .map { "k\($0.key):\($0.value)" }.joined(separator: ",")
        func reasons(_ counts: [String: Int]) -> String {
            counts.sorted(by: { $0.key < $1.key })
                .map { "\($0.key):\($0.value)" }.joined(separator: ",")
        }
        log("")
        log(
            "stats: hits=\(stats.hits) repeats=\(stats.repeats) misses=\(stats.misses) "
                + "trimHistogram=[\(trimSummary)] "
                + "missReasons=[\(reasons(stats.missReasons))] "
                + "junctionFailures=\(stats.junctionFailures) "
                + "replacedJunctionFailures=\(stats.replacedJunctionFailures) "
                + "junctionWindowEnlargements=\(stats.junctionWindowEnlargements) "
                + "cutWindowEnlargements=\(stats.cutWindowEnlargements) "
                + "truncatedHits=\(stats.truncatedHits) "
                + "truncatedFallbacks=\(stats.truncatedFallbacks) "
                + "truncatedFallbackReasons=[\(reasons(stats.truncatedFallbackReasons))] "
                + "replacedHits=\(stats.replacedHits) "
                + "replacedFallbacks=\(stats.replacedFallbacks) "
                + "replacedFallbackReasons=[\(reasons(stats.replacedFallbackReasons))]"
        )
        log(
            String(
                format:
                    "SUMMARY: hit turns=%d, baseline %.2f ms vs c25 %.2f ms per hit turn — saves %.2f ms (%.1f%%), token mismatches=%d, parity failures=%d, path failures=%d",
                totals.hitTurns,
                totals.hitTurns > 0 ? totals.totalBaselineMs / Double(totals.hitTurns) : 0,
                totals.hitTurns > 0 ? totals.totalC25Ms / Double(totals.hitTurns) : 0,
                totals.hitTurns > 0
                    ? (totals.totalBaselineMs - totals.totalC25Ms) / Double(totals.hitTurns) : 0,
                totals.totalBaselineMs > 0
                    ? (totals.totalBaselineMs - totals.totalC25Ms) / totals.totalBaselineMs * 100
                    : 0,
                totals.tokenMismatches,
                totals.parityFailures,
                totals.pathFailures
            ))
        log(
            String(
                format:
                    "C27 SUMMARY: truncated hit turns=%d, standalone %.2f ms vs c27 %.2f ms per hit turn — saves %.2f ms (%.1f%%), token mismatches=%d, path failures=%d",
                totals.truncHitTurns,
                totals.truncHitTurns > 0
                    ? totals.totalTruncStandaloneMs / Double(totals.truncHitTurns) : 0,
                totals.truncHitTurns > 0 ? totals.totalTruncMs / Double(totals.truncHitTurns) : 0,
                totals.truncHitTurns > 0
                    ? (totals.totalTruncStandaloneMs - totals.totalTruncMs)
                        / Double(totals.truncHitTurns) : 0,
                totals.totalTruncStandaloneMs > 0
                    ? (totals.totalTruncStandaloneMs - totals.totalTruncMs)
                        / totals.totalTruncStandaloneMs * 100 : 0,
                totals.truncMismatches,
                totals.truncPathFailures
            ))
        let legNames = ["leaf-store", "admission", "probe"]
        for (index, name) in legNames.enumerated() {
            let hits = totals.postGenLegHits[index]
            log(
                String(
                    format:
                        "C28 SUMMARY (%@): hit legs=%d, standalone %.2f ms vs c28 %.2f ms per hit leg — saves %.2f ms (%.1f%%)",
                    name,
                    hits,
                    hits > 0 ? totals.postGenLegStandaloneMs[index] / Double(hits) : 0,
                    hits > 0 ? totals.postGenLegMs[index] / Double(hits) : 0,
                    hits > 0
                        ? (totals.postGenLegStandaloneMs[index] - totals.postGenLegMs[index])
                            / Double(hits) : 0,
                    totals.postGenLegStandaloneMs[index] > 0
                        ? (totals.postGenLegStandaloneMs[index] - totals.postGenLegMs[index])
                            / totals.postGenLegStandaloneMs[index] * 100 : 0
                ))
        }
        log(
            String(
                format: "C28 SUMMARY: token mismatches=%d, path failures=%d",
                totals.postGenMismatches, totals.postGenPathFailures
            ))
        let failed =
            totals.tokenMismatches > 0 || totals.parityFailures > 0 || totals.pathFailures > 0
            || totals.truncMismatches > 0 || totals.truncPathFailures > 0
            || totals.postGenMismatches > 0 || totals.postGenPathFailures > 0
        log(failed ? "Overall: FAIL" : "Overall: PASS")
        return failed
    }

    /// Production-scale stable head: 40 tool specs.
    /// Internal (not private) so `--agent-cpu-bench` reuses the same specs.
    static func makeToolSpecs() -> [ToolSpec] {
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
    /// edited-history, and unrelated turns. Every turn ends in a user
    /// message, so its C27 truncated leg hits — except the two fallback
    /// turns appended last: an assistant tail (the trim's tail is dropped
    /// content, not a generation prompt) and a fresh conversation resolved
    /// before its own prepare (against the previous turn's stale entry).
    /// (A system-tail conversation is unrenderable on the loaded template —
    /// "System message must be at the beginning" — so that fallback shape
    /// lives in the unit tests, not here.)
    /// Internal (not private) so `--agent-cpu-bench` runs the same trajectory.
    static func buildTrajectory() -> [Turn] {
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

        var turns: [Turn] = []
        var history: [[String: any Sendable]] = [
            ["role": "user", "content": "Step 1: " + String(repeating: userFiller, count: 3)]
        ]
        for turn in 1...12 {
            let messages: [[String: any Sendable]] =
                [["role": "system", "content": systemPrompt]] + history
            let starter = replyStarters[(turn - 1) % replyStarters.count]
            let reply: [String: any Sendable] = [
                "role": "assistant",
                "content": starter + String(repeating: replyFiller, count: 10),
            ]
            turns.append(
                Turn(
                    label: "turn \(turn)",
                    messages: messages,
                    expect: turn == 1 ? .missCold : .hit,
                    truncExpect: .hit,
                    postGenReply: reply
                ))
            history.append(reply)
            history.append([
                "role": "user",
                "content": "Step \(turn + 1): " + String(repeating: userFiller, count: 3),
            ])
        }
        // Identical repeat of the last turn — its entry is untouched by the
        // hitRepeat, so the post-gen legs hit exactly as on turn 12.
        turns.append(
            Turn(
                label: "repeat", messages: turns.last!.messages, expect: .hitRepeat,
                truncExpect: .hit, postGenReply: turns.last!.postGenReply))
        // Edited history: turn 12 with the 3rd user message edited — must miss.
        // The miss stores the edited conversation's entry, so the truncated
        // and post-gen legs still hit.
        var edited = turns[11].messages
        edited[5] = [
            "role": "user",
            "content": "EDITED: " + (edited[5]["content"] as? String ?? ""),
        ]
        turns.append(
            Turn(
                label: "edited-history", messages: edited, expect: .miss, truncExpect: .hit,
                postGenReply: [
                    "role": "assistant",
                    "content": "Edited findings: " + String(repeating: replyFiller, count: 10),
                ]))
        // Unrelated prompt: fresh conversation — must miss; same reasoning:
        // the miss stores this conversation's entry, so the truncated and
        // post-gen legs hit.
        turns.append(
            Turn(
                label: "unrelated",
                messages: [
                    ["role": "system", "content": "You are a different assistant."],
                    ["role": "user", "content": "Unrelated request."],
                ],
                expect: .miss,
                truncExpect: .hit,
                postGenReply: [
                    "role": "assistant",
                    "content": "Unrelated reply: " + String(repeating: replyFiller, count: 10),
                ]
            ))
        // C27 fallback turns (no post-gen legs: their shapes exist for the
        // truncated leg's fallback coverage — the unit suites own the C28
        // fallback shapes).
        turns.append(
            Turn(
                label: "assistant-tail",
                messages: [
                    ["role": "system", "content": "You are a careful assistant."],
                    ["role": "user", "content": "Begin the audit."],
                    [
                        "role": "assistant",
                        "content": String(repeating: replyFiller, count: 10),
                    ],
                ],
                expect: .miss,
                truncExpect: .fallback
            ))
        turns.append(
            Turn(
                label: "stale-entry",
                messages: [
                    ["role": "system", "content": "A fresh system prompt."],
                    ["role": "user", "content": "A fresh user request."],
                ],
                expect: .miss,
                truncExpect: .fallback,
                truncFirst: true
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
