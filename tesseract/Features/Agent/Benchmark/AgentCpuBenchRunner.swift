import Foundation
import MLXHuggingFace
import MLXLMCommon
import os
import Tokenizers  // referenced by the #huggingFaceTokenizerLoader macro expansion

/// Per-turn CPU attribution benchmark (`--agent-cpu-bench`) — measurement
/// only. The tokenize side of the request path is covered by
/// `--tokenize-cache-bench` (C25–C28); this runner times everything ELSE the
/// agent/server turn pays on the CPU, per turn, over the same 12-turn
/// growing-conversation trajectory (2K-token system prompt, 40 tool specs,
/// ~700 tokens/turn — `TokenizeCacheBenchRunner.buildTrajectory`, shared).
///
/// Timed per turn (median of `reps` reps, interleaved, warmed):
/// 1. `p1 conv-build`   — `AgentConversationBuilder.conversation(...)`: the
///    agent-side canonicalization into `HTTPPrefixCacheConversation`
///    (per-message mapping + the tool-specs JSON digest).
/// 2. `p2 canonicalize` — the HTTP edge: `MessageConverter.normalizeRequest`
///    (includes the `toolDefinitionsDigest` JSONEncoder+SHA256 pass) plus
///    `TemplateRenderContext.resolve`.
/// 3. `p3 keying`       — `CacheKeySpace.make` on the text-only (identity)
///    path + the `CachePartitionKey` construction the Request Keying phase
///    does per request.
/// 4. `p4 boundary`     — `PrefillPlanner.detectBoundaries` in the production
///    steady state: StablePrefixDetector memo HIT (SHA-256 memo key over the
///    system prompt + canonical tools JSON, then a prefix token-hash
///    verification that scales with the stable-prefix length), the
///    generation-prompt encode, the C27 `resolveTruncated` re-render
///    (Jinja + digest chain + trim walk + cut verification — the full BPE
///    encode is what C27 eliminates), and the identity `translatedLength`.
/// 5. `p5 detok`        — the production streaming detokenizer
///    (`NaiveStreamingDetokenizer`, the same one `TokenGenerationLoop` drives
///    per generated token) over a ~700-token assistant reply; reported as
///    ms/turn and ms/token.
/// 6. `p6 digests`      — the `RenderTokenCache` per-message SHA-256 digest
///    chain + the tools/context `canonicalForm` digests, timed directly
///    (this cost lives INSIDE the C25 resolve — it overlaps the tokenize
///    bucket and is itemized here for attribution).
/// 7. `p7 radix`        — `TokenRadixTree` CPU: `findBestSnapshot` lookup +
///    `insertPath` + two `storeSnapshot` bookkeeping calls (leaf at the
///    terminal offset, system at the stable-prefix offset) against a tree
///    pre-populated with the trajectory's earlier turns, snapshots carrying
///    empty layer state (no GPU).
///
/// SKIPPED (uncallable without heavy fixtures — noted, not faked):
/// - `HybridCacheSnapshot` capture/restore and the SSD `SnapshotManifest`
///   bookkeeping: both need live GPU KV-cache arrays / the SSD writer.
///
/// Tokenizer-only by design (same harness shape as
/// `--tokenize-cache-bench`): every phase above is render/encode/SHA/trie
/// CPU work — no GPU weights are loaded.
@MainActor
final class AgentCpuBenchRunner {

    private enum PhaseID: String, CaseIterable {
        case conversationBuild = "p1 conv-build"
        case canonicalize = "p2 canonicalize"
        case keying = "p3 keying"
        case boundaryDetect = "p4 boundary"
        case detokStream = "p5 detok"
        case digestChain = "p6 digests"
        case radixOps = "p7 radix"
    }

    /// One turn's pre-built fixtures: everything the timed phases consume,
    /// built once (untimed) from the trajectory's prompt-message dicts.
    private struct TurnFixture {
        let label: String
        let promptMessages: [[String: any Sendable]]
        let systemPrompt: String
        let llmMessages: [LLMMessage]
        let openAIMessages: [OpenAI.ChatMessage]
        var fullTokens: [Int] = []
        var conversation: HTTPPrefixCacheConversation?
    }

    /// One turn's per-phase samples plus the radix-op breakdown.
    private struct TurnSamples {
        var phases: [PhaseID: [Double]] = [:]
        var radixLookup: [Double] = []
        var radixInsert: [Double] = []
        var radixStore: [Double] = []
        var detokPerTokenUs: [Double] = []

        mutating func append(_ phase: PhaseID, _ ms: Double) {
            phases[phase, default: []].append(ms)
        }

        func median(_ phase: PhaseID) -> Double {
            Self.median(phases[phase] ?? [])
        }

        static func median(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
    }

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("agent-cpu-bench")

    /// Median reps per phase per turn (spec: ≥3, interleaved, warmed).
    private static let reps = 5
    /// The turns the summary table reports (all 12 are timed and logged).
    private static let tableTurns = [1, 4, 8, 12]
    /// A phase is a TARGET at ≥2 ms/turn on the ~11K-token turn.
    private static let targetThresholdMs = 2.0

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
        let fingerprint =
            (try? ModelFingerprint.computeFingerprint(modelDir: modelDir)) ?? "unavailable"
        log("model fingerprint: \(fingerprint.prefix(16))…")

        let turns = Array(TokenizeCacheBenchRunner.buildTrajectory().prefix(12))
        let toolSpecs = TokenizeCacheBenchRunner.makeToolSpecs()
        let canonicalTools = LLMActor.canonicalizeToolSpecs(toolSpecs)
        let openAITools = Self.makeOpenAIToolDefinitions()
        let modelID = runner.activeConfig.resolvedModelID

        // The ~700-token assistant reply the detok phase streams (starter +
        // filler in the trajectory's shape; the encode is untimed setup).
        // Paragraph breaks matter: `NaiveStreamingDetokenizer.next()`
        // re-decodes the whole segment accumulated since the last "\n", so a
        // reply's cost is O(segment²) between newlines — the markdown-ish
        // shape here is the realistic case; a newline-free variant is logged
        // alongside as the worst case (tool-call JSON buffers, etc.).
        let replyFiller =
            "I read the file, compared it against the expected output, and "
            + "recorded the difference in the working notes before moving on. "
        let replyText =
            "The analysis is complete. "
            + String(repeating: replyFiller + replyFiller + "\n\n", count: 14)
        let replyTokens = tokenizer.encode(text: replyText, addSpecialTokens: false)
        let flatReplyTokens = tokenizer.encode(
            text: "The analysis is complete. " + String(repeating: replyFiller, count: 28),
            addSpecialTokens: false)
        log(
            "detok reply: \(replyTokens.count) tokens (realistic), "
                + "\(flatReplyTokens.count) tokens (no-newline worst case)"
        )

        var fixtures = turns.map { Self.makeFixture(turn: $0) }
        var allSamples: [TurnSamples] = []
        var stablePrefixOffset = 0

        for (index, _) in fixtures.enumerated() {
            var samples = TurnSamples()
            try prepareTurn(
                index: index, fixtures: &fixtures, tokenizer: tokenizer,
                canonicalTools: canonicalTools, fingerprint: fingerprint,
                stablePrefixOffset: &stablePrefixOffset
            )
            // Warm every phase once (untimed) before the interleaved reps.
            try runPhases(
                index: index, fixtures: fixtures, tokenizer: tokenizer,
                canonicalTools: canonicalTools, openAITools: openAITools,
                fingerprint: fingerprint, modelID: modelID,
                replyTokens: replyTokens, stablePrefixOffset: stablePrefixOffset,
                samples: &samples, record: false
            )
            for _ in 0..<Self.reps {
                try runPhases(
                    index: index, fixtures: fixtures, tokenizer: tokenizer,
                    canonicalTools: canonicalTools, openAITools: openAITools,
                    fingerprint: fingerprint, modelID: modelID,
                    replyTokens: replyTokens, stablePrefixOffset: stablePrefixOffset,
                    samples: &samples, record: true
                )
            }
            allSamples.append(samples)
            logTurnLine(index: index, fixtures: fixtures, samples: samples)
        }

        // One-off worst case: the same-length reply with no newline resets —
        // the detokenizer's O(segment²) upper bound, turn-independent.
        let flatDetokMs = Self.timeDetok(tokens: flatReplyTokens, tokenizer: tokenizer)
        log(
            String(
                format:
                    "detok worst case (no newlines): %.2f ms over %d tokens (%.2f us/token)",
                flatDetokMs, flatReplyTokens.count,
                flatDetokMs * 1000 / Double(max(flatReplyTokens.count, 1))
            ))

        let report = buildReport(fixtures: fixtures, allSamples: allSamples)
        log("")
        for line in report {
            log(line)
        }
        let reportURL = reportDir.appendingPathComponent("report.md")
        try report.joined(separator: "\n").write(
            to: reportURL, atomically: true, encoding: .utf8)
        log("Report written: \(reportURL.path)")
        logFileHandle?.closeFile()
    }

    // MARK: - Per-turn setup (untimed)

    /// Store the turn's entry in the shared render+token cache exactly as the
    /// Request Keying phase's C25 resolve does — over
    /// `conversation.promptMessages` (the conversation shape trims assistant
    /// whitespace; resolving the raw trajectory dicts instead would make the
    /// C27 truncated resolve miss on a digest/render mismatch production never
    /// sees) — take its exact tokens as the request's token list, and warm
    /// the boundary detect (StablePrefixDetector memo + C27 entry).
    private func prepareTurn(
        index: Int,
        fixtures: inout [TurnFixture],
        tokenizer: any MLXLMCommon.Tokenizer,
        canonicalTools: [ToolSpec]?,
        fingerprint: String,
        stablePrefixOffset: inout Int
    ) throws {
        fixtures[index].conversation = AgentConversationBuilder.conversation(
            systemPrompt: fixtures[index].systemPrompt,
            messages: fixtures[index].llmMessages,
            toolSpecs: canonicalTools
        )
        guard let conversation = fixtures[index].conversation else {
            log("FAIL: conversation build returned nil on \(fixtures[index].label)")
            throw NSError(domain: "AgentCpuBench", code: 2)
        }
        let resolution = try RenderTokenCache.shared.resolve(
            tokenizer: tokenizer,
            messages: conversation.promptMessages,
            tools: canonicalTools,
            additionalContext: nil,
            modelFingerprint: fingerprint
        )
        guard let resolution else {
            log("FAIL: resolve returned nil on \(fixtures[index].label)")
            throw NSError(domain: "AgentCpuBench", code: 1)
        }
        fixtures[index].fullTokens = resolution.tokens
        let boundaries = try PrefillPlanner.detectBoundaries(
            conversation: conversation,
            toolSpecs: canonicalTools,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: fixtures[index].fullTokens),
            renderContext: .canonical,
            modelFingerprint: fingerprint
        )
        if index == 0, let offset = boundaries.stablePrefixOffset {
            stablePrefixOffset = offset
        }
        log(
            "\(fixtures[index].label): tokens=\(fixtures[index].fullTokens.count) "
                + "stablePrefix=\(boundaries.stablePrefixOffset ?? -1) "
                + "lastMessage=\(boundaries.lastMessageOffset ?? -1) "
                + "lastUser=\(boundaries.lastUserOffset ?? -1)"
        )
    }

    // MARK: - Timed phases

    /// One pass over all seven phases for one turn; each phase's median-of-reps
    /// lands in `samples` when `record` is set. Phases run in a fixed order so
    /// every rep interleaves them against identical cache state.
    // swiftlint:disable:next function_parameter_count
    private func runPhases(
        index: Int,
        fixtures: [TurnFixture],
        tokenizer: any MLXLMCommon.Tokenizer,
        canonicalTools: [ToolSpec]?,
        openAITools: [OpenAI.ToolDefinition],
        fingerprint: String,
        modelID: String,
        replyTokens: [Int],
        stablePrefixOffset: Int,
        samples: inout TurnSamples,
        record: Bool
    ) throws {
        let fixture = fixtures[index]
        guard let conversation = fixture.conversation else { return }

        // p1 — conversation building (agent-side adapter).
        let p1 = Self.ms {
            _ = AgentConversationBuilder.conversation(
                systemPrompt: fixture.systemPrompt,
                messages: fixture.llmMessages,
                toolSpecs: canonicalTools
            )
        }

        // p2 — request canonicalization + digests (HTTP edge).
        let p2 = Self.ms {
            _ = MessageConverter.normalizeRequest(fixture.openAIMessages, tools: openAITools)
            _ = TemplateRenderContext.resolve(
                requestKwargs: nil, appEnabledFlags: [], declaredFlags: [.preserveThinking])
        }

        // p3 — keying: Cache Key Space (text-only identity path) + partition key.
        let p3 = Self.ms {
            _ = CacheKeySpace.make(
                preparedTokens: fixture.fullTokens,
                imageDigests: [],
                imageGrids: [],
                imageKeying: nil
            )
            _ = CachePartitionKey(
                modelID: modelID,
                kvBits: nil,
                kvGroupSize: 64,
                modelFingerprint: fingerprint,
                templateContextDigest: conversation.templateContextDigest
            )
        }

        // p4 — boundary detection, memo-warm steady state.
        var p4 = 0.0
        let p4Start = ContinuousClock.now
        _ = try PrefillPlanner.detectBoundaries(
            conversation: conversation,
            toolSpecs: canonicalTools,
            promptStartsThinking: true,
            tokenizer: tokenizer,
            keySpace: .identity(keyPath: fixture.fullTokens),
            renderContext: .canonical,
            modelFingerprint: fingerprint
        )
        p4 = Self.ms(since: p4Start)

        // p5 — per-token streaming detokenization of the assistant reply.
        let p5 = Self.timeDetok(tokens: replyTokens, tokenizer: tokenizer)

        // p6 — RenderTokenCache digest chain + canonicalForm digests, over
        // the same prompt-message dicts the Request Keying resolve digests.
        let p6 = Self.ms {
            _ = RenderTokenCache.digestChain(conversation.promptMessages)
            _ = RenderTokenCache.sha256Hex(
                RenderTokenCache.canonicalForm(optional: canonicalTools))
            _ = RenderTokenCache.sha256Hex(
                RenderTokenCache.canonicalForm(optional: nil as Any?))
        }

        // p7 — radix-tree CPU against the settled tree of earlier turns.
        let tree = buildTree(
            through: index, fixtures: fixtures, stablePrefixOffset: stablePrefixOffset)
        let lookupStart = ContinuousClock.now
        let hit = tree.findBestSnapshot(tokens: fixture.fullTokens)
        let lookupMs = Self.ms(since: lookupStart)
        let insertStart = ContinuousClock.now
        let node = tree.insertPath(tokens: fixture.fullTokens)
        let insertMs = Self.ms(since: insertStart)
        let storeStart = ContinuousClock.now
        tree.storeSnapshot(
            Self.emptySnapshot(offset: fixture.fullTokens.count, type: .leaf), on: node)
        if stablePrefixOffset > 0 {
            tree.storeSnapshot(
                Self.emptySnapshot(offset: stablePrefixOffset, type: .system),
                forTokens: fixture.fullTokens, atOffset: stablePrefixOffset)
        }
        let storeMs = Self.ms(since: storeStart)

        if record {
            samples.append(.conversationBuild, p1)
            samples.append(.canonicalize, p2)
            samples.append(.keying, p3)
            samples.append(.boundaryDetect, p4)
            samples.append(.detokStream, p5)
            samples.append(.digestChain, p6)
            samples.append(.radixOps, lookupMs + insertMs + storeMs)
            samples.radixLookup.append(lookupMs)
            samples.radixInsert.append(insertMs)
            samples.radixStore.append(storeMs)
            samples.detokPerTokenUs.append(p5 * 1000 / Double(max(replyTokens.count, 1)))
            if index > 0, hit == nil {
                log("note: \(fixture.label) radix lookup missed (expected a prefix hit)")
            }
        }
    }

    /// Stream one token list through the production detokenizer exactly as
    /// `TokenGenerationLoop` drives it — `append` + `next` per token — and
    /// return the total milliseconds.
    private static func timeDetok(tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer) -> Double {
        var detok = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        let start = ContinuousClock.now
        for token in tokens {
            detok.append(token: token)
            _ = detok.next()
        }
        return ms(since: start)
    }

    /// The settled prefix-cache tree the request's lookup/insert runs against:
    /// every earlier turn's token path with a leaf snapshot at its terminal
    /// offset and a system snapshot at the shared stable-prefix offset.
    private func buildTree(
        through index: Int,
        fixtures: [TurnFixture],
        stablePrefixOffset: Int
    ) -> TokenRadixTree {
        let tree = TokenRadixTree()
        for prior in fixtures.prefix(index) {
            let node = tree.insertPath(tokens: prior.fullTokens)
            tree.storeSnapshot(
                Self.emptySnapshot(offset: prior.fullTokens.count, type: .leaf), on: node)
            if stablePrefixOffset > 0 {
                tree.storeSnapshot(
                    Self.emptySnapshot(offset: stablePrefixOffset, type: .system),
                    forTokens: prior.fullTokens, atOffset: stablePrefixOffset)
            }
        }
        return tree
    }

    /// A GPU-free snapshot: empty layer state, so only the tree's CPU
    /// bookkeeping (budget reconciliation, state transition) is exercised.
    private static func emptySnapshot(
        offset: Int, type: HybridCacheSnapshot.CheckpointType
    ) -> HybridCacheSnapshot {
        HybridCacheSnapshot(
            tokenOffset: offset, layers: [], checkpointType: type, memoryBytes: 0,
            createdAt: .now)
    }

    // MARK: - Fixtures

    /// Split a trajectory turn into the shapes the timed phases consume: the
    /// leading system message becomes the system prompt; the rest map to the
    /// agent-side `LLMMessage` list and the HTTP-side `OpenAI.ChatMessage`
    /// list (text-only — the trajectory's shape).
    private static func makeFixture(turn: TokenizeCacheBenchRunner.Turn) -> TurnFixture {
        let systemPrompt = turn.messages.first?["content"] as? String ?? ""
        var llmMessages: [LLMMessage] = []
        var openAIMessages: [OpenAI.ChatMessage] = []
        for message in turn.messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            guard let chatRole = OpenAI.ChatRole(rawValue: role) else { continue }
            openAIMessages.append(
                OpenAI.ChatMessage(role: chatRole, content: .text(content)))
            switch chatRole {
            case .system:
                continue  // the builder takes the system prompt separately
            case .user:
                llmMessages.append(.user(content: content))
            case .assistant:
                llmMessages.append(.assistant(content: content, toolCalls: nil))
            case .tool:
                llmMessages.append(.toolResult(toolCallId: "", content: content))
            }
        }
        return TurnFixture(
            label: turn.label,
            promptMessages: turn.messages,
            systemPrompt: systemPrompt,
            llmMessages: llmMessages,
            openAIMessages: openAIMessages
        )
    }

    /// The HTTP edge's view of the 40 tool specs (same content as
    /// `makeToolSpecs`, as `OpenAI.ToolDefinition` values).
    private static func makeOpenAIToolDefinitions() -> [OpenAI.ToolDefinition] {
        (0..<40).map { i in
            OpenAI.ToolDefinition(
                type: "function",
                function: OpenAI.FunctionDefinition(
                    name: "tool_\(i)",
                    description: "Tool number \(i) used for agent operation \(i).",
                    parameters: .object([
                        "type": .string("object"),
                        "required": .array([.string("input")]),
                        "properties": .object([
                            "input": .object([
                                "type": .string("string"),
                                "description": .string("Input for tool_\(i)."),
                            ])
                        ]),
                    ])
                )
            )
        }
    }

    // MARK: - Reporting

    private func logTurnLine(index: Int, fixtures: [TurnFixture], samples: TurnSamples) {
        var parts = [
            "\(fixtures[index].label) (\(fixtures[index].fullTokens.count) tok):"
        ]
        for phase in PhaseID.allCases {
            parts.append(
                "\(phase.rawValue)=\(String(format: "%.3f", samples.median(phase)))")
        }
        parts.append(
            "radix(lookup=\(String(format: "%.3f", TurnSamples.median(samples.radixLookup)))"
                + " insert=\(String(format: "%.3f", TurnSamples.median(samples.radixInsert)))"
                + " store=\(String(format: "%.3f", TurnSamples.median(samples.radixStore))))")
        parts.append(
            "detokPerToken=\(String(format: "%.2f", TurnSamples.median(samples.detokPerTokenUs)))us"
        )
        log(parts.joined(separator: " "))
    }

    /// The summary table: phase × turn(1/4/8/12) medians, per-phase verdict,
    /// and the total accounted non-tokenize CPU per turn.
    private func buildReport(fixtures: [TurnFixture], allSamples: [TurnSamples]) -> [String] {
        let columns = Self.tableTurns
        var lines: [String] = []
        lines.append("# --agent-cpu-bench report")
        lines.append("")
        lines.append(
            "Median of \(Self.reps) interleaved reps per phase. Turn token counts: "
                + columns.map { "turn \($0) = \(fixtures[$0 - 1].fullTokens.count)" }
                .joined(separator: ", "))
        lines.append("")
        var header = Self.pad("phase", 18)
        for turn in columns {
            header += Self.pad("turn \(turn)", 12)
        }
        header += "verdict"
        lines.append(header)
        lines.append(String(repeating: "-", count: header.count))

        var totals = [Double](repeating: 0, count: columns.count)
        for phase in PhaseID.allCases {
            let medians = columns.map { allSamples[$0 - 1].median(phase) }
            for (column, median) in medians.enumerated() {
                totals[column] += median
            }
            let atScale = medians[columns.count - 1]
            let verdict =
                atScale >= Self.targetThresholdMs
                ? "TARGET (\(String(format: "%.2f", atScale)) ms ≥ "
                    + "\(String(format: "%.0f", Self.targetThresholdMs)) ms)"
                : "CLOSED (\(String(format: "%.2f", atScale)) ms)"
            var row = Self.pad(phase.rawValue, 18)
            for median in medians {
                row += Self.pad(String(format: "%.3f", median), 12)
            }
            row += verdict
            lines.append(row)
        }
        lines.append(String(repeating: "-", count: header.count))
        var totalRow = Self.pad("TOTAL (p1–p7)", 18)
        for total in totals {
            totalRow += Self.pad(String(format: "%.3f", total), 12)
        }
        lines.append(totalRow)
        var outsideRow = Self.pad("outside p6 digests", 18)
        let p6 = PhaseID.digestChain
        for (column, turn) in columns.enumerated() {
            outsideRow += Self.pad(
                String(format: "%.3f", totals[column] - allSamples[turn - 1].median(p6)), 12)
        }
        lines.append(outsideRow)
        lines.append("")
        lines.append(
            "Verdict rule: TARGET when the phase's median at the ~11K-token turn reaches "
                + "\(String(format: "%.0f", Self.targetThresholdMs)) ms, else CLOSED. "
                + "p6 lives inside the C25 resolve (tokenize bucket) — the second total "
                + "excludes it. p4 includes the C27 truncated resolve (Jinja render + "
                + "digests + trim/cut verification), not the full re-encode C27 eliminates. "
                + "p5 streams a realistic paragraph-shaped reply (segments reset at "
                + "newlines); the no-newline worst case is logged once above the table."
        )
        lines.append(
            "SKIPPED: HybridCacheSnapshot capture/restore and SSD SnapshotManifest "
                + "bookkeeping — both need live GPU KV-cache arrays / the SSD writer; "
                + "the radix phase above covers the tree's CPU bookkeeping with "
                + "empty-layer snapshots."
        )
        return lines
    }

    // MARK: - Timing + logging helpers

    private static func ms(since start: ContinuousClock.Instant) -> Double {
        let c = start.duration(to: .now).components
        return (Double(c.seconds) + Double(c.attoseconds) * 1e-18) * 1e3
    }

    private static func ms(_ body: () throws -> Void) rethrows -> Double {
        let start = ContinuousClock.now
        try body()
        return ms(since: start)
    }

    private static func pad(_ string: String, _ width: Int) -> String {
        string.count >= width
            ? string + " " : string + String(repeating: " ", count: width - string.count)
    }

    private func log(_ message: String) {
        logger.info("\(message, privacy: .public)")
        if let data = (message + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }
}
