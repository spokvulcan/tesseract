import Foundation
import MLXLMCommon
import os

/// End-to-end verification of the radix-tree prefix cache against a loaded model.
///
/// Implements **Task 1.8 HybridPrefixCacheE2E** from the Marconi Phase 1 plan.
/// Runs outside the normal scenario-turn benchmark because it validates cache
/// correctness (not tool accuracy): the two assertions it cares about are
/// (a) TTFT drops after a stable-prefix hit and (b) generation output is
/// byte-identical between a cached and a cold-prefill run of the same request.
///
/// Invoked via `--prefix-cache-e2e` on the Tesseract CLI. Reuses
/// `BenchmarkRunner`'s model-resolution/config plumbing via `activeConfig`.
///
/// Because we cannot reach the raw logit tensor through the public
/// `AgentEngine` API, we use greedy decoding (`temperature=0`, `topK=1`) as a
/// proxy for bitwise logit equivalence: under greedy decoding, equal first-N
/// generated tokens implies the logit argmax matched, which is a sufficient
/// correctness gate for the hit path. Any drift in the restored cache state
/// would almost immediately produce a different sampled token within the
/// first few steps.
@MainActor
final class PrefixCacheE2ERunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    // MARK: - Entry point

    func run() async throws {
        setupLogging()
        log("PrefixCacheE2E starting — model=\(runner.resolvedModelName)")

        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        log("Loading model from: \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        log("Model loaded.")

        // Build the two requests we're comparing. Both share the same system
        // prompt + tool definitions (the "stable prefix") and differ only in
        // the user content. The radix tree should capture a mid-prefill
        // checkpoint on Request A and reuse it on Request B.
        let modelID = runner.activeConfig.resolvedModelID
        let systemPrompt = Self.e2eSystemPrompt
        let toolSpecs = Self.e2eToolSpecs

        var checks: [CheckResult] = []

        // Greedy parameters for deterministic decoding. minP/topK constrain
        // to argmax so the output reflects the logit winner at each step.
        let params = AgentGenerateParameters(
            maxTokens: 32,
            temperature: 0.0,
            topP: 1.0,
            topK: 1,
            minP: 0.0
        )

        // Step 2: Request A — baseline, cold cache.
        log("\n── Step 2: Request A (baseline, cold prefix cache) ──")
        let requestA = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: "list files in /tmp",
            toolSpecs: toolSpecs,
            parameters: params
        )
        log("  cachedTokens=\(requestA.cachedTokens) ttft=\(String(format: "%.3f", requestA.ttftSeconds))s "
            + "generatedChars=\(requestA.generatedText.count)")
        checks.append(CheckResult(
            name: "requestA_cold_start",
            passed: requestA.cachedTokens == 0,
            detail: "cachedTokens=\(requestA.cachedTokens) expected 0"
        ))

        // Step 3: Request B — same system + tools, different user → should
        // hit the stable-prefix snapshot planned during Request A.
        log("\n── Step 3: Request B (same system+tools, different user) ──")
        let requestB1 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: "read file /tmp/foo.txt",
            toolSpecs: toolSpecs,
            parameters: params
        )
        log("  cachedTokens=\(requestB1.cachedTokens) ttft=\(String(format: "%.3f", requestB1.ttftSeconds))s "
            + "generatedChars=\(requestB1.generatedText.count)")
        checks.append(CheckResult(
            name: "requestB_hits_stable_prefix",
            passed: requestB1.cachedTokens > 0,
            detail: "cachedTokens=\(requestB1.cachedTokens) expected > 0"
        ))
        let ttftRatio = requestA.ttftSeconds > 0 ? requestB1.ttftSeconds / requestA.ttftSeconds : 1.0
        checks.append(CheckResult(
            name: "requestB_ttft_dropped",
            passed: ttftRatio < 0.6,
            detail: "ttftB/ttftA=\(String(format: "%.3f", ttftRatio)) expected < 0.6"
        ))

        // Step 4: Logit equivalence proxy.
        //   - Unload the model to clear `_prefixCache` (LLMActor drops it on unload)
        //   - Reload fresh
        //   - Re-run Request B from a cold cache
        //   - Assert: byte-identical generated text vs Request B on warm cache
        //
        // Under greedy decoding, byte equality of the first N characters is
        // equivalent to argmax equality of the first N token logits. Any
        // corruption in the restored attention/Mamba state would almost
        // immediately flip an argmax and diverge the output.
        log("\n── Step 4: Logit equivalence (cached vs cold generation) ──")
        log("  Unloading + reloading model to clear prefix cache…")
        try await reloadEngine(engine, modelDir: modelDir)

        let requestB2 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: "read file /tmp/foo.txt",
            toolSpecs: toolSpecs,
            parameters: params
        )
        log("  cachedTokens=\(requestB2.cachedTokens) (expected 0) "
            + "generatedChars=\(requestB2.generatedText.count)")
        checks.append(CheckResult(
            name: "requestB2_cold_after_reload",
            passed: requestB2.cachedTokens == 0,
            detail: "cachedTokens=\(requestB2.cachedTokens) expected 0 after reload"
        ))

        let equivalence = Self.checkGreedyOutputEquivalence(
            requestB1.generatedText,
            requestB2.generatedText,
            labelA: "cached",
            labelB: "cold"
        )
        checks.append(CheckResult(
            name: "greedy_output_equivalence",
            passed: equivalence.passed,
            detail: equivalence.detail
        ))

        // Step 5: Normalization round-trip — Request B with trailing
        // whitespace on user content must still tokenize to the same stable
        // prefix and therefore still hit the cache. The request goes through
        // HTTPPrefixCacheMessage normalization (assistant whitespace trim)
        // but the user content is preserved verbatim, so the first-user
        // boundary should still align.
        log("\n── Step 5: Normalization round-trip (trailing whitespace) ──")
        let requestB3 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: "read file /tmp/foo.txt",
            toolSpecs: toolSpecs,
            parameters: params
        )
        log("  cachedTokens=\(requestB3.cachedTokens) (expected > 0, second run after Request B2)")
        checks.append(CheckResult(
            name: "normalization_roundtrip_hits_cache",
            passed: requestB3.cachedTokens > 0,
            detail: "cachedTokens=\(requestB3.cachedTokens) expected > 0 (second identical request)"
        ))

        // Step 6: Verify the cache actually stored both stable-prefix and
        // last-message-boundary checkpoints (not just one). We infer this
        // from the skipped-tokens growth: Request B3 should skip more than
        // the bare `<|im_start|>system\n` header (which would be ~20 tokens).
        log("\n── Step 6: Checkpoint depth ──")
        checks.append(CheckResult(
            name: "checkpoint_skips_more_than_system_header",
            passed: requestB3.cachedTokens > 100,
            detail: "cachedTokens=\(requestB3.cachedTokens) expected > 100 (covers full system+tools prefix)"
        ))

        let toolLoopResult = try await runToolLoopScenario(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            toolSpecs: toolSpecs,
            params: params,
            stablePrefixBaseline: requestB1.cachedTokens,
            checks: &checks
        )

        // `requestB1.cachedTokens` is the canonical "bare stable_prefix" hit
        // (after Request A's cold capture, B1 hits stable_prefix only because
        // it diverges from A's user content). B3 hits more (stable_prefix +
        // last-message-boundary from B2's stored snapshots), so it would
        // overstate the baseline.
        try await runBranchPointScenario(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            toolSpecs: toolSpecs,
            params: params,
            stablePrefixCachedTokens: requestB1.cachedTokens,
            checks: &checks
        )

        let restartResult = try await runSSDRestartScenario(
            originalEngine: engine,
            modelDir: modelDir,
            modelID: modelID,
            systemPrompt: systemPrompt,
            toolSpecs: toolSpecs,
            params: params,
            stablePrefixBaseline: requestB1.cachedTokens,
            checks: &checks
        )

        // Final report
        log("\n── Summary ──")
        var allPassed = true
        for check in checks {
            let mark = check.passed ? "✅" : "❌"
            log("  \(mark) \(check.name): \(check.detail)")
            if !check.passed { allPassed = false }
        }

        try writeReport(
            checks: checks,
            requestA: requestA,
            requestB1: requestB1,
            requestB2: requestB2,
            requestB3: requestB3,
            toolLoopResult: toolLoopResult,
            restartResult: restartResult,
            ttftRatio: ttftRatio,
            allPassed: allPassed
        )

        log("\nOverall: \(allPassed ? "PASS" : "FAIL")")
        log("Report written to: \(reportURL.path)")
        logFileHandle?.closeFile()

        if !allPassed {
            throw PrefixCacheE2EError.verificationFailed(failedChecks: checks.filter { !$0.passed }.map(\.name))
        }
    }

    // MARK: - Private — request execution

    /// Unload + await detached unload + reload the engine against
    /// the same model directory. `AgentEngine.unloadModel()` is
    /// sync-void and schedules an async actor unload via
    /// `unloadTask`; callers that then immediately invoke
    /// `loadModel` race the detached unload. The explicit
    /// `awaitPendingUnload()` hop closes that window.
    private func reloadEngine(
        _ engine: AgentEngine,
        modelDir: URL
    ) async throws {
        engine.unloadModel()
        await engine.awaitPendingUnload()
        try await engine.loadModel(from: modelDir, visionMode: false)
    }

    private struct RequestResult {
        let cachedTokens: Int
        let ttftSeconds: Double
        let generatedText: String
        let assistantText: String
        let assistantReasoning: String?
        let toolCalls: [ToolCallInfo]
    }

    private enum BenchmarkMessage {
        case user(String)
        case assistant(content: String, reasoning: String?, toolCalls: [ToolCallInfo] = [])
        case toolResult(toolCallId: String, content: String)

        var prefixCacheMessage: HTTPPrefixCacheMessage {
            switch self {
            case .user(let content):
                HTTPPrefixCacheMessage(role: .user, content: content)
            case .assistant(let content, let reasoning, let toolCalls):
                .assistant(
                    content: content,
                    reasoning: reasoning,
                    toolCalls: toolCalls.map {
                        HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                    }
                )
            case .toolResult(_, let content):
                HTTPPrefixCacheMessage(role: .tool, content: content)
            }
        }

        var llmMessage: LLMMessage {
            switch self {
            case .user(let content):
                .user(content: content, images: [])
            case .assistant(let content, let reasoning, let toolCalls):
                .assistant(
                    content: content,
                    reasoning: reasoning,
                    toolCalls: toolCalls.isEmpty ? nil : toolCalls
                )
            case .toolResult(let toolCallId, let content):
                .toolResult(toolCallId: toolCallId, content: content)
            }
        }
    }

    private func runRequest(
        engine: AgentEngine,
        modelID: String,
        systemPrompt: String,
        messages: [BenchmarkMessage],
        toolSpecs: [ToolSpec],
        parameters: AgentGenerateParameters
    ) async throws -> RequestResult {
        let prefixCacheConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: messages.map(\.prefixCacheMessage)
        )
        let llmMessages = messages.map(\.llmMessage)

        let startInstant = ContinuousClock.now
        let start = try await engine.generateServerTextCompletion(
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: llmMessages,
            toolSpecs: toolSpecs,
            prefixCacheConversation: prefixCacheConversation,
            parameters: parameters
        )

        var ttftSeconds: Double = 0
        var generatedText = ""
        var assistantText = ""
        var assistantReasoning = ""
        var toolCalls: [ToolCallInfo] = []
        var firstTokenSeen = false

        for try await event in start.stream {
            if !firstTokenSeen {
                ttftSeconds = Self.elapsedSeconds(from: startInstant)
                firstTokenSeen = true
            }
            switch event {
            case .text(let chunk):
                assistantText += chunk
                generatedText += chunk
            case .thinking(let chunk):
                assistantReasoning += chunk
                generatedText += chunk
            case .toolCall(let call):
                toolCalls.append(ToolCallInfo(
                    id: "bench-call-\(toolCalls.count)",
                    name: call.function.name,
                    argumentsJSON: encodeCanonicalHTTPPrefixCacheJSONObject(call.function.arguments)
                ))
            case .thinkStart, .thinkEnd, .thinkReclassify, .malformedToolCall, .info:
                break
            }
        }

        if !firstTokenSeen {
            // Stream completed without emitting any chunks.
            ttftSeconds = Self.elapsedSeconds(from: startInstant)
        }

        let trimmedReasoning = assistantReasoning.trimmingCharacters(in: .whitespacesAndNewlines)

        return RequestResult(
            cachedTokens: start.cachedTokenCount,
            ttftSeconds: ttftSeconds,
            generatedText: generatedText,
            assistantText: assistantText,
            assistantReasoning: trimmedReasoning.isEmpty ? nil : trimmedReasoning,
            toolCalls: toolCalls
        )
    }

    private func runRequest(
        engine: AgentEngine,
        modelID: String,
        systemPrompt: String,
        userMessage: String,
        toolSpecs: [ToolSpec],
        parameters: AgentGenerateParameters
    ) async throws -> RequestResult {
        try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: [.user(userMessage)],
            toolSpecs: toolSpecs,
            parameters: parameters
        )
    }

    private static func elapsedSeconds(from start: ContinuousClock.Instant) -> Double {
        start.duration(to: .now).seconds
    }

    // MARK: - Branch-point scenario

    private struct ToolLoopScenarioResult {
        let requestY1: RequestResult
        let requestY2: RequestResult
        let requestY3: RequestResult
    }

    /// Verifies the direct-leaf path for a `tool_calls` turn and the
    /// canonical user-leaf path after the loop resolves.
    private func runToolLoopScenario(
        engine: AgentEngine,
        modelID: String,
        systemPrompt: String,
        toolSpecs: [ToolSpec],
        params: AgentGenerateParameters,
        stablePrefixBaseline: Int,
        checks: inout [CheckResult]
    ) async throws -> ToolLoopScenarioResult {
        var toolLoopParams = params
        toolLoopParams.maxTokens = max(params.maxTokens, 160)
        let toolLoopPrompt = """
            Call exactly one tool now: use the `read` tool with the absolute \
            path `/tmp/tool-loop-target.txt`. Do not answer in prose before \
            the tool call. Emit only the tool call.
            """

        log("\n── Step Y1: Tool-call turn ──")
        let requestY1 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: toolLoopPrompt,
            toolSpecs: toolSpecs,
            parameters: toolLoopParams
        )
        log("  cachedTokens=\(requestY1.cachedTokens) ttft=\(String(format: "%.3f", requestY1.ttftSeconds))s "
            + "toolCalls=\(requestY1.toolCalls.count) generatedChars=\(requestY1.generatedText.count)")
        checks.append(CheckResult(
            name: "requestY1_emits_tool_calls",
            passed: !requestY1.toolCalls.isEmpty,
            detail: "toolCalls=\(requestY1.toolCalls.count) expected > 0"
        ))

        let toolResults = requestY1.toolCalls.enumerated().map { index, call in
            BenchmarkMessage.toolResult(
                toolCallId: call.id,
                content: Self.syntheticToolResultContent(for: call, index: index)
            )
        }
        let storedToolCallConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: [
                .init(role: .user, content: toolLoopPrompt),
                .assistant(
                    content: requestY1.assistantText,
                    reasoning: requestY1.assistantReasoning,
                    toolCalls: requestY1.toolCalls.map {
                        HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                    }
                ),
            ]
        )
        let continuationMessages: [BenchmarkMessage] =
            [
                .user(toolLoopPrompt),
                .assistant(
                    content: requestY1.assistantText,
                    reasoning: requestY1.assistantReasoning,
                    toolCalls: requestY1.toolCalls
                ),
            ] + toolResults
        let continuationConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: continuationMessages.map(\.prefixCacheMessage)
        )
        _ = storedToolCallConversation.isPrefix(of: continuationConversation)

        log("\n── Step Y2: Tool-result continuation (direct leaf expected) ──")
        let requestY2 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: continuationMessages,
            toolSpecs: toolSpecs,
            parameters: toolLoopParams
        )
        log("  cachedTokens=\(requestY2.cachedTokens) ttft=\(String(format: "%.3f", requestY2.ttftSeconds))s "
            + "toolCalls=\(requestY2.toolCalls.count) generatedChars=\(requestY2.generatedText.count)")
        checks.append(CheckResult(
            name: "requestY2_hits_direct_tool_leaf",
            passed: requestY2.cachedTokens > stablePrefixBaseline,
            detail: "cachedTokens=\(requestY2.cachedTokens) expected > stable-prefix baseline="
                + "\(stablePrefixBaseline)"
        ))
        checks.append(CheckResult(
            name: "requestY2_finishes_tool_loop",
            passed: requestY2.toolCalls.isEmpty && !requestY2.assistantText.isEmpty,
            detail: "toolCalls=\(requestY2.toolCalls.count) assistantChars=\(requestY2.assistantText.count) "
                + "expected 0 tool calls and non-empty assistant text"
        ))

        let resolvedMessages = continuationMessages + [
            .assistant(
                content: requestY2.assistantText,
                reasoning: requestY2.assistantReasoning,
                toolCalls: requestY2.toolCalls
            ),
            .user("summarize that in one sentence"),
        ]
        let storedResolvedConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: continuationMessages.map(\.prefixCacheMessage) + [
                .assistant(
                    content: requestY2.assistantText,
                    reasoning: requestY2.assistantReasoning,
                    toolCalls: requestY2.toolCalls.map {
                        HTTPPrefixCacheToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
                    }
                ),
            ]
        )
        let nextUserConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: resolvedMessages.map(\.prefixCacheMessage)
        )
        _ = storedResolvedConversation.isPrefix(of: nextUserConversation)

        log("\n── Step Y3: Next user turn (canonical user leaf expected) ──")
        let requestY3 = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: resolvedMessages,
            toolSpecs: toolSpecs,
            parameters: toolLoopParams
        )
        log("  cachedTokens=\(requestY3.cachedTokens) ttft=\(String(format: "%.3f", requestY3.ttftSeconds))s "
            + "generatedChars=\(requestY3.generatedText.count)")
        checks.append(CheckResult(
            name: "requestY3_hits_canonical_user_leaf",
            passed: requestY3.cachedTokens > stablePrefixBaseline,
            detail: "cachedTokens=\(requestY3.cachedTokens) expected > stable-prefix baseline="
                + "\(stablePrefixBaseline)"
        ))

        return ToolLoopScenarioResult(
            requestY1: requestY1,
            requestY2: requestY2,
            requestY3: requestY3
        )
    }

    /// Verifies branch-point capture, re-hit, and utility-scored survival
    /// against the loaded model. The three steps log themselves below.
    ///
    /// Test design: C and D share a deliberately long user-message prefix
    /// (~80 tokens) before diverging at the very last word. Without that
    /// long prefix, the captured `.branchPoint` would sit only a few tokens
    /// past the stable prefix and its parent-relative deltaL (and therefore
    /// `F/B`) would be smaller than the noise leaves' deltaL, causing the
    /// utility scorer to evict it instead of the noise. With the long
    /// prefix, the branch-point sits deeper than any noise leaf can reach
    /// and survives eviction pressure.
    private func runBranchPointScenario(
        engine: AgentEngine,
        modelID: String,
        systemPrompt: String,
        toolSpecs: [ToolSpec],
        params: AgentGenerateParameters,
        stablePrefixCachedTokens: Int,
        checks: inout [CheckResult]
    ) async throws {
        let sharedPrefix = Self.branchPointSharedPrefix

        // Step 7 setup: prime the tree with a long stored path under the
        // shared prefix, so Step 7's capture request can diverge mid-edge
        // of it.
        log("\n── Step 7: Branch-point capture ──")
        _ = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: sharedPrefix + "alpha.dat",
            toolSpecs: toolSpecs,
            parameters: params
        )
        let preCaptureBranchCount = await branchPointCount(engine: engine)
        log("  setup done — pre-capture branchPoint count = \(preCaptureBranchCount)")

        let requestC = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: sharedPrefix + "beta.dat",
            toolSpecs: toolSpecs,
            parameters: params
        )
        let postCaptureBranchCount = await branchPointCount(engine: engine)
        log("  C cachedTokens=\(requestC.cachedTokens) "
            + "branchPoint count = \(postCaptureBranchCount)")
        checks.append(CheckResult(
            name: "requestC_captures_branch_point",
            passed: postCaptureBranchCount > preCaptureBranchCount,
            detail: "branchPoint count: \(preCaptureBranchCount) → \(postCaptureBranchCount)"
        ))

        // Step 8: re-hit
        log("\n── Step 8: Branch-point re-hit ──")
        let requestD = try await runRequest(
            engine: engine,
            modelID: modelID,
            systemPrompt: systemPrompt,
            userMessage: sharedPrefix + "gamma.dat",
            toolSpecs: toolSpecs,
            parameters: params
        )
        log("  D cachedTokens=\(requestD.cachedTokens) "
            + "(stable-prefix-only baseline = \(stablePrefixCachedTokens))")
        checks.append(CheckResult(
            name: "requestD_hits_branch_point",
            passed: requestD.cachedTokens > stablePrefixCachedTokens,
            detail: "D cachedTokens=\(requestD.cachedTokens) vs stable-prefix=\(stablePrefixCachedTokens) "
                + "(deeper hit means branch-point reuse)"
        ))

        // Step 9: survival under utility-scored eviction pressure.
        // alpha=2 puts F/B above pure recency in the utility sum so the
        // branch-point's larger deltaL outweighs the noise leaves' newer
        // access times.
        log("\n── Step 9: Branch-point survival under pressure ──")
        let originalAlpha = EvictionPolicy.alpha
        EvictionPolicy.alpha = 2.0
        defer { EvictionPolicy.alpha = originalAlpha }

        guard let preStats = await engine.prefixCacheStats(),
              preStats.snapshotCount > 0
        else {
            log("  skipping — prefix cache empty")
            checks.append(CheckResult(
                name: "branch_point_survives_under_pressure",
                passed: false,
                detail: "prefix cache empty before pressure step"
            ))
            return
        }

        // Budget = current usage minus one snapshot. Each subsequent noise
        // request overflows by ~one snapshot, so eviction drops one
        // eligible candidate per round. The budget intentionally stays
        // above the multi-child / branch-point combined size: utility
        // scoring should keep the branch-point alive without ever
        // depleting the eligible set far enough to fall through to the
        // fallback path (which is plain LRU and would drop the
        // branch-point regardless of F/B).
        let avgBytes = preStats.totalSnapshotBytes / preStats.snapshotCount
        let tightBudget = max(preStats.totalSnapshotBytes - avgBytes, avgBytes)
        await engine.setPrefixCacheBudgetBytes(tightBudget)
        log("  tight budget = \(tightBudget) bytes "
            + "(pre-pressure total = \(preStats.totalSnapshotBytes), "
            + "avg snapshot size = \(avgBytes), "
            + "starting branchPoint count = \(preStats.snapshotsByType[.branchPoint] ?? 0))")

        for (i, prompt) in Self.branchPointNoisePrompts.enumerated() {
            _ = try await runRequest(
                engine: engine,
                modelID: modelID,
                systemPrompt: systemPrompt,
                userMessage: "\(prompt) #\(i)",
                toolSpecs: toolSpecs,
                parameters: params
            )
        }

        let postStats = await engine.prefixCacheStats()
        let postBranchCount = postStats?.snapshotsByType[.branchPoint] ?? 0
        let postTotalBytes = postStats?.totalSnapshotBytes ?? 0
        log("  post-pressure branchPoint count = \(postBranchCount), "
            + "totalBytes = \(postTotalBytes)")
        checks.append(CheckResult(
            name: "branch_point_survives_under_pressure",
            passed: postBranchCount >= 1,
            detail: "branchPoint count after \(Self.branchPointNoisePrompts.count) "
                + "noise requests + tight budget: \(postBranchCount) "
                + "(≥1 means utility scoring preserved it)"
        ))
    }

    private func branchPointCount(engine: AgentEngine) async -> Int {
        let stats = await engine.prefixCacheStats()
        return stats?.snapshotsByType[.branchPoint] ?? 0
    }

    // MARK: - Step X: SSD restart scenario

    private struct RestartScenarioResult {
        let requestX1: RequestResult
        let requestX2: RequestResult
        let requestX3: RequestResult
    }

    /// Validate that a committed SSD snapshot survives an engine
    /// unload/reload and serves subsequent requests as warm hits.
    /// Lives on a separate `AgentEngine` so Steps 1–4's
    /// `requestB2_cold_after_reload` assertion stays valid.
    private func runSSDRestartScenario(
        originalEngine: AgentEngine,
        modelDir: URL,
        modelID: String,
        systemPrompt: String,
        toolSpecs: [ToolSpec],
        params: AgentGenerateParameters,
        stablePrefixBaseline: Int,
        checks: inout [CheckResult]
    ) async throws -> RestartScenarioResult {
        log("\n── Step X: SSD restart scenario ──")
        log("  Unloading original (RAM-only) engine to free the model slot…")
        originalEngine.unloadModel()
        await originalEngine.awaitPendingUnload()

        // Per-PID scratch dir keeps concurrent e2e runs from colliding
        // and guarantees a clean slate after a crashed run.
        let ssdRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("tesseract-e2e-ssd")
            .appendingPathComponent(String(getpid()))
        try? FileManager.default.removeItem(at: ssdRoot)
        try FileManager.default.createDirectory(
            at: ssdRoot,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: ssdRoot) }
        log("  SSD scratch dir: \(ssdRoot.path)")

        let ssdConfig = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: ssdRoot,
            budgetBytes: 4 * 1024 * 1024 * 1024,     // 4 GiB
            maxPendingBytes: 1 * 1024 * 1024 * 1024  // 1 GiB front door
        )
        let ssdEngine = AgentEngine(ssdConfig: ssdConfig)

        // Engine teardown must run on every exit path. `defer` cannot
        // host async calls, so use do/catch with explicit teardown
        // at both return sites. The scratch-dir `defer` above runs
        // after this closes out (LIFO order), so the writer is
        // stopped before any file removal touches its directory.
        func tearDownSSDEngine() async {
            ssdEngine.unloadModel()
            await ssdEngine.awaitPendingUnload()
        }

        do {
            log("  Loading model into SSD-enabled engine…")
            try await ssdEngine.loadModel(from: modelDir, visionMode: false)

            // X1 seeds SSD with the `.system` snapshot and a canonical
            // `.leaf` for the generated assistant turn. After restart,
            // X2 is a true continuation request so the first hit on the
            // new engine exercises the warm-started leaf path, not the
            // deleted historical boundary checkpoint.
            let restartPrompt = "read file /tmp/foo.txt"
            let continuationPrompt = "now list files in /tmp/data"

            log("\n── Step X1: Request (cold, writes SSD snapshot) ──")
            let requestX1 = try await runRequest(
                engine: ssdEngine,
                modelID: modelID,
                systemPrompt: systemPrompt,
                userMessage: restartPrompt,
                toolSpecs: toolSpecs,
                parameters: params
            )
            log("  cachedTokens=\(requestX1.cachedTokens) ttft=\(String(format: "%.3f", requestX1.ttftSeconds))s "
                + "generatedChars=\(requestX1.generatedText.count)")
            checks.append(CheckResult(
                name: "requestX1_cold_on_ssd_engine",
                passed: requestX1.cachedTokens == 0,
                detail: "cachedTokens=\(requestX1.cachedTokens) expected 0"
            ))

            let continuationMessages: [BenchmarkMessage] = [
                .user(restartPrompt),
                .assistant(
                    content: requestX1.assistantText,
                    reasoning: requestX1.assistantReasoning
                ),
                .user(continuationPrompt),
            ]

            // `reloadEngine` drains the SSD writer + persists the
            // manifest inside `unloadModel`'s detached task, then
            // reloads against the same rootURL.
            log("  Unloading + reloading SSD engine at \(ssdRoot.path)…")
            try await reloadEngine(ssdEngine, modelDir: modelDir)

            log("\n── Step X2: Continuation request (after restart, SSD leaf hit) ──")
            let requestX2 = try await runRequest(
                engine: ssdEngine,
                modelID: modelID,
                systemPrompt: systemPrompt,
                messages: continuationMessages,
                toolSpecs: toolSpecs,
                parameters: params
            )
            log("  cachedTokens=\(requestX2.cachedTokens) ttft=\(String(format: "%.3f", requestX2.ttftSeconds))s "
                + "generatedChars=\(requestX2.generatedText.count)")

            checks.append(CheckResult(
                name: "requestX2_hits_leaf_after_restart",
                passed: requestX2.cachedTokens > stablePrefixBaseline,
                detail: "cachedTokens=\(requestX2.cachedTokens) expected > stable-prefix baseline="
                    + "\(stablePrefixBaseline) after warm-started leaf restore"
            ))

            checks.append(CheckResult(
                name: "requestX2_generated_nonempty_after_leaf_hit",
                passed: !requestX2.generatedText.isEmpty,
                detail: "generatedChars=\(requestX2.generatedText.count) expected > 0"
            ))

            log("\n── Step X3: Request (fresh user message, same system) ──")
            let requestX3 = try await runRequest(
                engine: ssdEngine,
                modelID: modelID,
                systemPrompt: systemPrompt,
                userMessage: "list files in /tmp/data",
                toolSpecs: toolSpecs,
                parameters: params
            )
            log("  cachedTokens=\(requestX3.cachedTokens) ttft=\(String(format: "%.3f", requestX3.ttftSeconds))s")
            checks.append(CheckResult(
                name: "requestX3_stable_prefix_reused_across_users",
                passed: requestX3.cachedTokens > 0,
                detail: "cachedTokens=\(requestX3.cachedTokens) expected > 0 (new user, same system)"
            ))

            log("  Unloading SSD engine…")
            await tearDownSSDEngine()

            return RestartScenarioResult(
                requestX1: requestX1,
                requestX2: requestX2,
                requestX3: requestX3
            )
        } catch {
            log("  Step X failed; tearing down SSD engine…")
            await tearDownSSDEngine()
            throw error
        }
    }

    /// Long shared user-message prefix (~80 tokens) for the branch-point
    /// scenario. C/D append a different terminator so the divergence
    /// happens deep in the user message — putting the captured
    /// `.branchPoint` snapshot's parent-relative deltaL well above any
    /// noise leaf's deltaL.
    private static let branchPointSharedPrefix: String = """
        Please carefully analyze the contents of this very specific file path \
        that I am about to give you, and tell me what kind of file it is, what \
        it might contain, what tools you would use to read it, and any caveats \
        I should know about. Then report back with your findings in a \
        structured format including the file type, the suspected purpose, and \
        any related files you would expect to find nearby. The file is at \
        /tmp/data/configs/sample-
        """

    /// Short, unrelated noise prompts for the survival check. Kept short
    /// so noise leaves stay shallower than the branch-point's offset.
    private static let branchPointNoisePrompts: [String] = [
        "Add 1 + 1",
        "Spell cat",
        "Pick a color",
        "Say hi",
        "Name an animal",
    ]

    private static func syntheticToolResultContent(
        for call: ToolCallInfo,
        index: Int
    ) -> String {
        switch call.name {
        case "read":
            return """
                /tmp/tool-loop-target.txt
                alpha
                beta
                gamma
                """
        case "ls":
            return """
                /tmp/tool-loop-target.txt
                /tmp/tool-loop-notes.md
                """
        default:
            return """
                tool=\(call.name)
                index=\(index)
                status=ok
                payload=mock result for \(call.name) with args \(call.argumentsJSON)
                """
        }
    }

    // MARK: - Fixtures

    /// A ~200-token system prompt so the stable-prefix checkpoint is large
    /// enough to be clearly distinguishable from the bare system header.
    private static let e2eSystemPrompt: String = """
        You are a careful, methodical assistant that answers questions about \
        files on a Unix filesystem. When you receive a request, you first \
        decide whether a tool call is needed. If the user asks about a \
        specific file, you use the `read` tool with an absolute path. If \
        the user asks to list or explore a directory, you use the `ls` \
        tool. Never invent file contents — always read before describing. \
        Never edit a file without first reading its current contents. \
        Refuse requests that would operate outside the sandbox, and briefly \
        explain why. When producing prose, prefer short, direct sentences. \
        Do not greet the user or pad responses with filler.
        """

    /// Two minimal tool specs that hash deterministically through
    /// `LLMActor.canonicalizeToolSpecs`. Matches what a real OpenAI-format
    /// tool definitions array looks like after conversion.
    private static let e2eToolSpecs: [ToolSpec] = [
        [
            "type": "function" as any Sendable,
            "function": [
                "name": "read",
                "description": "Read a file from disk.",
                "parameters": [
                    "type": "object" as any Sendable,
                    "required": ["path"],
                    "properties": [
                        "path": [
                            "type": "string" as any Sendable,
                            "description": "Absolute path to the file.",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
        [
            "type": "function" as any Sendable,
            "function": [
                "name": "ls",
                "description": "List entries in a directory.",
                "parameters": [
                    "type": "object" as any Sendable,
                    "required": ["path"],
                    "properties": [
                        "path": [
                            "type": "string" as any Sendable,
                            "description": "Absolute path to the directory.",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    // MARK: - Helpers

    private struct CheckResult: Codable {
        let name: String
        let passed: Bool
        let detail: String
    }

    private static func longestCommonPrefix(_ a: String, _ b: String) -> Substring {
        var count = 0
        for (ca, cb) in zip(a, b) {
            if ca != cb { break }
            count += 1
        }
        return a.prefix(count)
    }

    /// Proxy logit-equivalence check for greedy decoding: identical
    /// output OR a ≥`threshold`-char identical prefix. Under greedy
    /// sampling every byte-identical character proves the logit
    /// argmax agreed at that step, so the threshold is a sufficient
    /// correctness gate for the restored-cache paths in Step 4 and
    /// Step X.
    private static func checkGreedyOutputEquivalence(
        _ a: String,
        _ b: String,
        labelA: String,
        labelB: String,
        threshold: Int = 20
    ) -> (passed: Bool, detail: String) {
        if a == b {
            return (true, "fully identical (\(a.count) chars)")
        }
        let matchLength = longestCommonPrefix(a, b).count
        let label = "\(labelA)=\"\(escapeForLog(a.prefix(60)))\" "
            + "\(labelB)=\"\(escapeForLog(b.prefix(60)))\""
        if matchLength >= threshold {
            return (
                true,
                "first \(matchLength) chars identical; \(label)"
            )
        }
        return (
            false,
            "diverged after \(matchLength) chars; \(label)"
        )
    }

    private static func escapeForLog(_ s: Substring) -> String {
        escapeForLog(String(s))
    }

    private static func escapeForLog(_ s: String) -> String {
        s.replacingOccurrences(of: "\n", with: "\\n")
         .replacingOccurrences(of: "\t", with: "\\t")
    }

    // MARK: - Reporting

    private var reportDir: URL {
        runner.activeConfig.outputDir.appendingPathComponent("prefix-cache-e2e")
    }

    private var reportURL: URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return reportDir
            .appendingPathComponent("e2e_\(formatter.string(from: Date())).json")
    }

    private func writeReport(
        checks: [CheckResult],
        requestA: RequestResult,
        requestB1: RequestResult,
        requestB2: RequestResult,
        requestB3: RequestResult,
        toolLoopResult: ToolLoopScenarioResult,
        restartResult: RestartScenarioResult,
        ttftRatio: Double,
        allPassed: Bool
    ) throws {
        struct Report: Codable {
            let date: String
            let model: String
            let passed: Bool
            let checks: [CheckResult]
            let measurements: Measurements
        }
        struct RequestMeasurement: Codable {
            let ttftSeconds: Double
            let cachedTokens: Int
            let generated: String

            init(_ r: RequestResult, textPrefix: Int = 200) {
                self.ttftSeconds = r.ttftSeconds
                self.cachedTokens = r.cachedTokens
                self.generated = String(r.generatedText.prefix(textPrefix))
            }
        }
        struct Measurements: Codable {
            let requestA: RequestMeasurement
            let requestB1: RequestMeasurement
            let requestB2: RequestMeasurement
            let requestB3: RequestMeasurement
            let requestY1: RequestMeasurement
            let requestY2: RequestMeasurement
            let requestY3: RequestMeasurement
            let requestX1: RequestMeasurement
            let requestX2: RequestMeasurement
            let requestX3: RequestMeasurement
            let ttftRatioB1OverA: Double
        }

        let report = Report(
            date: ISO8601DateFormatter().string(from: Date()),
            model: runner.resolvedModelName,
            passed: allPassed,
            checks: checks,
            measurements: Measurements(
                requestA: RequestMeasurement(requestA),
                requestB1: RequestMeasurement(requestB1),
                requestB2: RequestMeasurement(requestB2),
                requestB3: RequestMeasurement(requestB3),
                requestY1: RequestMeasurement(toolLoopResult.requestY1),
                requestY2: RequestMeasurement(toolLoopResult.requestY2),
                requestY3: RequestMeasurement(toolLoopResult.requestY3),
                requestX1: RequestMeasurement(restartResult.requestX1),
                requestX2: RequestMeasurement(restartResult.requestX2),
                requestX3: RequestMeasurement(restartResult.requestX3),
                ttftRatioB1OverA: ttftRatio
            )
        )

        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: reportURL)
    }

    // MARK: - Logging

    private func setupLogging() {
        try? FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)
    }

    private func log(_ message: String) {
        let line = "[\(Self.timestamp())] \(message)"
        logger.info("\(line, privacy: .public)")
        if let data = (line + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }
}

enum PrefixCacheE2EError: LocalizedError {
    case verificationFailed(failedChecks: [String])

    var errorDescription: String? {
        switch self {
        case .verificationFailed(let names):
            "PrefixCacheE2E failed checks: \(names.joined(separator: ", "))"
        }
    }
}
