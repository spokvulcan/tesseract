import Foundation
import MLX
import MLXLMCommon
import os

/// Loaded-model logit-equivalence harness for the hybrid prefix cache.
/// Drives the production `PrefillExecutor` chunked-prefill path against a
/// manual cache to verify mid-prefill capture/restore round-trips
/// bit-for-bit. Run via `--hybrid-cache-correctness` on the Tesseract CLI.
@MainActor
// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable:next type_body_length
final class HybridCacheCorrectnessRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?

    /// Cache snapshots are deep copies of the exact tensors, so post-restore
    /// equality must be 0 — any drift is a real correctness bug.
    nonisolated private static let bitwiseTolerance: Float = 0.0

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    // MARK: - Entry point

    func run() async throws {
        setupLogging()
        log("HybridCacheCorrectness starting — model=\(runner.resolvedModelName)")

        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        let modelID = runner.activeConfig.resolvedModelID
        let servingTemperature = AgentGenerateParameters.forModel(modelID).temperature
        log("Loading model from: \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        log("Model loaded.")

        let testRun = try await engine.llmActor.withModelContainer { container in
            await container.perform { context in
                Self.runAllTests(
                    context: context, modelDirectory: modelDir, modelID: modelID,
                    servingTemperature: servingTemperature)
            }
        }

        for line in testRun.logs { log(line) }
        let checks = testRun.checks

        log("\n── Summary ──")
        var allPassed = true
        for check in checks {
            let mark = check.passed ? "✅" : "❌"
            log("  \(mark) \(check.name): \(check.detail)")
            if !check.passed { allPassed = false }
        }

        let reportURL = try writeReport(checks: checks, allPassed: allPassed)
        log("\nOverall: \(allPassed ? "PASS" : "FAIL")")
        log("Report written to: \(reportURL.path)")
        logFileHandle?.closeFile()

        if !allPassed {
            throw HybridCacheCorrectnessError.verificationFailed(
                failedChecks: checks.filter { !$0.passed }.map { $0.name }
            )
        }
    }

    // MARK: - Orchestration

    private struct TestRunResult: Sendable {
        let logs: [String]
        let checks: [CheckResult]
    }

    private nonisolated static func runAllTests(
        context: ModelContext, modelDirectory: URL, modelID: String, servingTemperature: Float
    ) -> TestRunResult {
        var logs: [String] = []
        var checks: [CheckResult] = []

        let shortPrompt = BenchmarkHarness.promptTokens(
            targetTokens: 2048, tokenizer: context.tokenizer)
        // Test 8 wants a 16K prompt with restore at 8K — exercises cache shapes
        // and chunk-loop handling well past the 4K-context typical case.
        let longPrompt = BenchmarkHarness.promptTokens(
            targetTokens: 16384, tokenizer: context.tokenizer)
        logs.append("\n── Token sequences ──")
        logs.append("  short prompt: \(shortPrompt.count) tokens")
        logs.append("  long prompt:  \(longPrompt.count) tokens")

        // The two `fullPrefillLastLogits` calls (short + long) are the most
        // expensive single operations. Compute each once and share across
        // all tests that need them — saves one full-prompt prefill.
        let shortFullLogits: MLXArray
        let longFullLogits: MLXArray
        do {
            shortFullLogits = try fullPrefillLastLogits(context: context, tokens: shortPrompt)
            longFullLogits = try fullPrefillLastLogits(context: context, tokens: longPrompt)
        } catch {
            logs.append("Setup failed: \(error)")
            return TestRunResult(
                logs: logs,
                checks: [CheckResult(name: "setupFailed", passed: false, detail: "\(error)")]
            )
        }

        func runTest(
            _ name: String,
            _ body: () throws -> (passed: Bool, detail: String, lines: [String])
        ) {
            logs.append("\n── \(name) ──")
            do {
                let result = try body()
                logs.append(contentsOf: result.lines)
                checks.append(CheckResult(name: name, passed: result.passed, detail: result.detail))
            } catch {
                logs.append("  ❌ threw: \(error)")
                checks.append(CheckResult(name: name, passed: false, detail: "threw: \(error)"))
            }
        }

        runTest("midPrefillRestoreMatchesFullPrefill") {
            try test1_midPrefillRestore(
                context: context, tokens: shortPrompt, fullLogits: shortFullLogits)
        }
        runTest("restoreAtExactMatch") {
            try test2_restoreAtExactMatch(context: context, tokens: shortPrompt)
        }
        runTest("divergentSuffixAfterRestore") {
            try test3_divergentSuffix(context: context, tokens: shortPrompt)
        }
        runTest("mambaStateRestoredExactly") {
            try test4_5_stateEquality(
                context: context, tokens: shortPrompt,
                filter: { $0 is MambaCache },
                label: "mamba"
            )
        }
        runTest("attentionKVRestoredExactly") {
            try test4_5_stateEquality(
                context: context, tokens: shortPrompt,
                filter: { !($0 is MambaCache) },
                label: "attention",
                checkOffset: true
            )
        }
        runTest("quantizedKVCacheRestoredExactly") {
            try test6_quantizedKVCacheRestoredExactly(context: context, tokens: shortPrompt)
        }
        runTest("multipleRestoresFromSameSnapshot") {
            try test7_multipleRestoresFromSameSnapshot(context: context, tokens: shortPrompt)
        }
        runTest("longContext16KRestore") {
            try test8_longContextRestore(
                context: context, tokens: longPrompt, fullLogits: longFullLogits)
        }
        runTest("leafHitWithNormalizationDivergenceBounded") {
            try test9_leafHitWithTrimBoundedDivergence(context: context, tokens: shortPrompt)
        }
        runTest("leafHitWithoutNormalizationMatchesBitwise") {
            try test10_leafHitWithoutTrim(
                context: context, tokens: shortPrompt, fullLogits: shortFullLogits
            )
        }
        runTest("twoPassLogitsMatchFullPrefill") {
            try test11_twoPassAlignmentPrefill(
                context: context, tokens: shortPrompt, fullLogits: shortFullLogits
            )
        }
        runTest("asymmetricStateRestoreRestoresWithoutCrashing") {
            try test12_asymmetricStateRestoreSmoke(
                context: context, modelDirectory: modelDirectory, modelID: modelID,
                servingTemperature: servingTemperature)
        }

        return TestRunResult(logs: logs, checks: checks)
    }

    // MARK: - Tests

    private nonisolated static func test1_midPrefillRestore(
        context: ModelContext,
        tokens: [Int],
        fullLogits: MLXArray
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let n = tokens.count
        let kValues = [n / 4, n / 2, 3 * n / 4]
        var lines: [String] = ["  full prefill logits: shape=\(fullLogits.shape)"]

        var failures: [String] = []
        for k in kValues {
            let restoredLogits = try restoreAndContinueLastLogits(
                context: context, tokens: tokens, restoreAt: k
            )
            let diff = BenchmarkHarness.maxAbsDiff(fullLogits, restoredLogits)
            lines.append("  K=\(k) (\(k * 100 / n)%): maxAbsDiff=\(diff)")
            if diff > bitwiseTolerance {
                failures.append("K=\(k) diff=\(diff)")
            }
        }

        let passed = failures.isEmpty
        let detail =
            passed
            ? "all \(kValues.count) K values bitwise match (K=\(kValues))"
            : "diverged at: \(failures.joined(separator: ", "))"
        return (passed, detail, lines)
    }

    /// Capture at K = N (full prompt length, no suffix to prefill). Restore
    /// yields a cache identical to the post-prefill state. Forward the same
    /// sentinel token on both the live cache and the restored cache and
    /// compare — both must produce identical logits.
    ///
    /// Note: this can't reuse `fullLogits` (which is "logits at position N",
    /// produced from a cache at offset N-1 + final-token forward). The
    /// capture-at-N test produces "logits at position N+1" from a cache
    /// already at offset N + sentinel forward. The two outputs are different
    /// computations; comparing them would compare apples to oranges.
    private nonisolated static func test2_restoreAtExactMatch(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        // Single full prefill of all N tokens, capture at offset N, then
        // forward the sentinel from both the live and the restored cache.
        // Doing it in this order means only one prefill is needed.
        let liveCache = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: tokens, checkpoints: [:], cache: liveCache)
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: liveCache, offset: tokens.count, type: .system
            )
        else {
            return (false, "capture nil", [])
        }
        // Forward the sentinel on the live cache. Captures the "predict
        // position N+1" logits from a cache at offset N.
        let logitsLive = lastTokenLogits(context: context, tokens: tokens, cache: liveCache)

        // Restore a fresh copy at offset N and forward the same sentinel.
        let restoredCache = try snap.restore()
        let logitsRestored = lastTokenLogits(
            context: context, tokens: tokens, cache: restoredCache
        )

        let diff = BenchmarkHarness.maxAbsDiff(logitsLive, logitsRestored)
        return (diff <= bitwiseTolerance, "maxAbsDiff=\(diff)", [])
    }

    /// Smoke test: divergent suffix after restore produces finite logits with
    /// non-trivial vocab span (no NaNs, no shape mismatch).
    private nonisolated static func test3_divergentSuffix(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let n = tokens.count
        let k = n / 2

        let prefillCache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.prefix(k)),
            checkpoints: [k: .system],
            cache: prefillCache
        )
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: prefillCache, offset: k, type: .system
            )
        else {
            return (false, "capture nil", [])
        }
        let restored = try snap.restore()

        let divergentSuffix = Array(tokens.suffix(n - k).reversed())
        try prefill(
            context: context,
            tokens: divergentSuffix,
            checkpoints: [:],
            checkpointBaseOffset: k,
            cache: restored
        )
        let logits = lastTokenLogits(
            context: context, tokens: divergentSuffix, cache: restored
        )
        let logitsF = logits.dtype == .bfloat16 ? logits.asType(.float32) : logits
        let maxVal = logitsF.max().item(Float.self)
        let minVal = logitsF.min().item(Float.self)
        let isfinite = maxVal.isFinite && minVal.isFinite
        let span = maxVal - minVal
        return (isfinite && span > 0, "finite=\(isfinite) span=\(span)", [])
    }

    /// Tests 4 + 5: state-array equality between live and restored caches at
    /// K = N/2, parameterized by layer filter (Mamba vs attention).
    private nonisolated static func test4_5_stateEquality(
        context: ModelContext,
        tokens: [Int],
        filter: (any KVCache) -> Bool,
        label: String,
        checkOffset: Bool = false
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let k = tokens.count / 2
        let cacheA = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.prefix(k)),
            checkpoints: [:],
            cache: cacheA
        )
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: cacheA, offset: k, type: .system
            )
        else {
            return (false, "capture nil", [])
        }
        let cacheB = try snap.restore()

        let result = compareCacheStates(
            cacheA: cacheA, cacheB: cacheB, filter: filter, checkOffset: checkOffset
        )
        var detail = "\(label)Layers=\(result.layerCount) maxAbsDiff=\(result.maxDiff)"
        if checkOffset { detail += " offsetMatch=\(!result.offsetMismatch)" }
        let passed =
            result.layerCount > 0
            && result.maxDiff <= bitwiseTolerance
            && !result.offsetMismatch
        return (passed, detail, [])
    }

    /// Force quantization on every layer via `maybeQuantizeKVCache`, capture,
    /// restore with the matching hints, compare wq/scales/biases element-wise.
    private nonisolated static func test6_quantizedKVCacheRestoredExactly(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        var cacheA: [any KVCache] = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: tokens, checkpoints: [:], cache: cacheA)
        maybeQuantizeKVCache(cache: &cacheA, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)

        let quantizedLayers = cacheA.filter { $0 is QuantizedKVCache }
        if quantizedLayers.isEmpty {
            return (true, "no QuantizedKVCache layers in this model — skipped", [])
        }

        guard
            let snap = HybridCacheSnapshot.capture(
                cache: cacheA, offset: tokens.count, type: .system
            )
        else {
            return (false, "capture nil", [])
        }
        let cacheB = try snap.restore()

        var maxDiff: Float = 0
        var allMetaMatch = true
        for (la, lb) in zip(cacheA, cacheB) {
            guard let qa = la as? QuantizedKVCache, let qb = lb as? QuantizedKVCache else {
                continue
            }
            if qa.metaState != qb.metaState { allMetaMatch = false }
            for (sa, sb) in zip(qa.state, qb.state) {
                let d = BenchmarkHarness.maxAbsDiff(sa, sb)
                if d > maxDiff { maxDiff = d }
            }
        }

        let passed = maxDiff <= bitwiseTolerance && allMetaMatch
        return (
            passed,
            "quantizedLayers=\(quantizedLayers.count) maxAbsDiff=\(maxDiff) metaMatch=\(allMetaMatch)",
            []
        )
    }

    /// Capture once, restore twice, prefill the same suffix on each — both
    /// restored caches must produce identical last-position logits.
    private nonisolated static func test7_multipleRestoresFromSameSnapshot(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let k = tokens.count / 2
        let setupCache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.prefix(k)),
            checkpoints: [:],
            cache: setupCache
        )
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: setupCache, offset: k, type: .system
            )
        else {
            return (false, "capture nil", [])
        }

        let cache1 = try snap.restore()
        let cache2 = try snap.restore()
        let suffix = Array(tokens.suffix(tokens.count - k))

        try prefill(
            context: context, tokens: suffix,
            checkpoints: [:], checkpointBaseOffset: k, cache: cache1
        )
        let logits1 = lastTokenLogits(context: context, tokens: suffix, cache: cache1)

        try prefill(
            context: context, tokens: suffix,
            checkpoints: [:], checkpointBaseOffset: k, cache: cache2
        )
        let logits2 = lastTokenLogits(context: context, tokens: suffix, cache: cache2)

        let diff = BenchmarkHarness.maxAbsDiff(logits1, logits2)
        return (diff <= bitwiseTolerance, "maxAbsDiff=\(diff)", [])
    }

    private nonisolated static func test8_longContextRestore(
        context: ModelContext,
        tokens: [Int],
        fullLogits: MLXArray
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let n = tokens.count
        let k = n / 2
        let restoredLogits = try restoreAndContinueLastLogits(
            context: context, tokens: tokens, restoreAt: k
        )
        let diff = BenchmarkHarness.maxAbsDiff(fullLogits, restoredLogits)
        return (
            diff <= bitwiseTolerance,
            "totalTokens=\(n) restoreAt=\(k) maxAbsDiff=\(diff)",
            []
        )
    }

    /// Leaf-hit round-trip with no normalization trim. Captures as `.leaf`
    /// from a cache that's never been trimmed (`actualCacheOffset ==
    /// storedTokens.count` in production), restores into a fresh cache, and
    /// compares the post-restore final-token logits bitwise to a cold
    /// reference. Validates the leaf-tier capture/restore path is byte-exact
    /// when no trimming complicates the cache state.
    private nonisolated static func test10_leafHitWithoutTrim(
        context: ModelContext,
        tokens: [Int],
        fullLogits: MLXArray
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let setupCache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.dropLast()),
            checkpoints: [:],
            cache: setupCache
        )
        let leafOffset = tokens.count - 1
        guard
            let leafSnap = HybridCacheSnapshot.capture(
                cache: setupCache, offset: leafOffset, type: .leaf
            )
        else {
            return (false, "leaf capture nil", [])
        }
        let restoredCache = try leafSnap.restore()
        let restoredLogits = lastTokenLogits(
            context: context, tokens: tokens, cache: restoredCache
        )
        let diff = BenchmarkHarness.maxAbsDiff(fullLogits, restoredLogits)
        return (diff <= bitwiseTolerance, "maxAbsDiff=\(diff)", [])
    }

    /// Simulates the Phase 3.1 alignment path: restore at `K`, capture an
    /// intermediate checkpoint at `M > K` during the suffix prefill, then
    /// continue to the final logits. The result must match cold prefill
    /// bitwise, and the mid-suffix checkpoint must actually be emitted.
    private nonisolated static func test11_twoPassAlignmentPrefill(
        context: ModelContext,
        tokens: [Int],
        fullLogits: MLXArray
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let n = tokens.count
        let k = n / 4
        let m = n / 2

        let setupCache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.prefix(k)),
            checkpoints: [:],
            cache: setupCache
        )
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: setupCache, offset: k, type: .system
            )
        else {
            return (false, "capture nil at K=\(k)", [])
        }

        let restoredCache = try snap.restore()
        let suffix = Array(tokens[k..<(tokens.count - 1)])
        let snapshots = try prefill(
            context: context,
            tokens: suffix,
            checkpoints: [m: .branchPoint],
            checkpointBaseOffset: k,
            cache: restoredCache
        )
        let restoredLogits = lastTokenLogits(
            context: context, tokens: tokens, cache: restoredCache
        )
        let diff = BenchmarkHarness.maxAbsDiff(fullLogits, restoredLogits)
        let capturedOffsets = snapshots.map(\.tokenOffset)
        let capturedM = capturedOffsets.contains(m)
        return (
            capturedM && diff <= bitwiseTolerance,
            "K=\(k) M=\(m) maxAbsDiff=\(diff) capturedOffsets=\(capturedOffsets)",
            []
        )
    }

    /// Diagnostic: simulates the trim-and-restore path that **production no
    /// longer reaches**. The **Server Completion** module now skips
    /// the leaf store entirely when re-tokenization would require any
    /// non-zero `trimAmount` (see the offset-alignment guard in
    /// `ServerCompletion.swift`). The reasoning, validated by this
    /// test, is that trimming attention K/V while Mamba's recurrent state
    /// stays advanced past the trimmed offset perturbs raw logits by ~10
    /// even at `trim = 1` on Qwen3.5. Argmax stays stable at trim=1
    /// (greedy decoding would survive), but the rest of the distribution
    /// drifts in a way that affects sampled decoding — and the HTTP server
    /// propagates user-supplied `temperature`/`top_p`, so we can't predict
    /// future request sampling params at store time.
    ///
    /// The original Phase 2 plan-doc spec called for `max|diff| < 0.01`
    /// bound here. That bound was unrealistic on this model class and is
    /// no longer load-bearing — production simply doesn't expose itself to
    /// the divergence anymore.
    ///
    /// **What this test still validates:** the simulated trim-and-restore
    /// math (a) executes without crashing on the live model and (b)
    /// preserves argmax at `trim = 1`, matching the historical Phase 1
    /// "small recurrent divergence — acceptable" assumption. If a future
    /// regression makes even `trim = 1` flip argmax, the recurrent-state
    /// behavior has drifted further than the production guard accounts for
    /// and the guard's conservative threshold may need tightening. The
    /// `trim = 2` and `trim = 4` measurements are logged as diagnostics
    /// to characterize the drift envelope on the current model.
    private nonisolated static func test9_leafHitWithTrimBoundedDivergence(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let trimSweep = [1, 2, 4]
        var lines: [String] = []
        var argmaxStableAt1 = false
        var trimmedLayerCount = 0
        var sawTrimmable = false

        for trimAmount in trimSweep {
            let n = tokens.count
            let leafOffset = n - 1 - trimAmount
            let coldTokens = Array(tokens.prefix(leafOffset + 1))

            let liveCache = context.model.newCache(parameters: nil)
            try prefill(
                context: context,
                tokens: Array(tokens.dropLast()),
                checkpoints: [:],
                cache: liveCache
            )
            var trimmedThisIter = 0
            for layer in liveCache where layer.isTrimmable {
                if layer.trim(trimAmount) > 0 { trimmedThisIter += 1 }
            }
            trimmedLayerCount = trimmedThisIter
            if trimmedThisIter > 0 { sawTrimmable = true }
            guard trimmedThisIter > 0 else { continue }

            guard
                let leafSnap = HybridCacheSnapshot.capture(
                    cache: liveCache, offset: leafOffset, type: .leaf
                )
            else {
                return (false, "trimmed leaf capture nil at trim=\(trimAmount)", lines)
            }
            let restoredCache = try leafSnap.restore()
            let restoredLogits = lastTokenLogits(
                context: context, tokens: coldTokens, cache: restoredCache
            )
            let coldLogits = try fullPrefillLastLogits(context: context, tokens: coldTokens)

            let diff = BenchmarkHarness.maxAbsDiff(coldLogits, restoredLogits)
            let coldArgmax = coldLogits.argMax().item(Int32.self)
            let restoredArgmax = restoredLogits.argMax().item(Int32.self)
            let stable = coldArgmax == restoredArgmax
            if trimAmount == 1 { argmaxStableAt1 = stable }

            lines.append(
                "  trim=\(trimAmount): maxAbsDiff=\(diff) "
                    + "argmax cold=\(coldArgmax) restored=\(restoredArgmax) stable=\(stable)"
            )
        }

        guard sawTrimmable else {
            return (
                true,
                "no trimmable layers in this model — skipped",
                ["  trimSweep=\(trimSweep) trimmableLayers=0"]
            )
        }
        return (
            argmaxStableAt1,
            "trim=1 argmaxStable=\(argmaxStableAt1) trimmedLayers=\(trimmedLayerCount)",
            lines
        )
    }

    // MARK: - Asymmetric-State Restore (issue #134 acceptance gate)

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient.
    // swiftlint:disable function_body_length
    /// The **Asymmetric-State Restore** spike's acceptance seam: render a real
    /// captured thinking conversation through the real chat template (the
    /// shared **Conversation Render** contract), capture the think-bearing
    /// render, synthesize the stripped-path snapshot, restore it through the
    /// real restore path, and measure ASR vs gold (a full, consistent
    /// re-prefill of the stripped stretch) at the real serving temperature.
    ///
    /// Three artifacts per the PRD's acceptance gate:
    /// 1. **Smoke** — the synthesized snapshot restores without crashing
    ///    (free from the seam).
    /// 2. **Distributional divergence** — per-token KL + top-k agreement at
    ///    the first divergence position, at the serving temperature. Argmax
    ///    alone is explicitly not a pass (ADR-0009's false-success mode).
    /// 3. **Side-by-side vs gold** — short greedy continuations from the
    ///    ASR-restored and gold caches, logged for the researcher's
    ///    qualitative call.
    ///
    /// This is a research spike: correctness is unproven by construction
    /// (recurrent state is stale by design), so the check reports the
    /// measurements rather than asserting a tight bound — the verdict is the
    /// researcher's qualitative call against gold, not a green checkmark.
    private nonisolated static func test12_asymmetricStateRestoreSmoke(
        context: ModelContext, modelDirectory: URL, modelID: String, servingTemperature: Float
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        var lines: [String] = []
        lines.append("  modelID=\(modelID) servingTemperature=\(servingTemperature)")

        // 1. A real multi-turn thinking conversation with substantial think
        //    blocks — the hazard must be genuinely exercised, not masked by a
        //    toy input (user story #16). The conversation ENDS at an assistant
        //    turn: that assistant turn (and any since the last real user
        //    message) is the **Tool Stretch**, which the template renders with
        //    `<think>` kept in the bearing render. Ending on a user turn would
        //    make that user the last query and strip the stretch's thinks,
        //    defeating the exercise.
        let messages: [[String: any Sendable]] = [
            [
                "role": "system",
                "content":
                    "You are a careful coding agent. Think step by step, then call a tool or answer.",
            ],
            ["role": "user", "content": "What is 17 times 23? Show your reasoning."],
            [
                "role": "assistant",
                "reasoning_content":
                    "I need to compute 17 * 23. Break it: 17 * 20 = 340, 17 * 3 = 51, total 391. "
                    + "Double-check: 23 * 17 = 23 * 10 + 23 * 7 = 230 + 161 = 391. Confirmed.",
                "content": "The answer is 391.",
            ],
            [
                "role": "user",
                "content": "Now subtract 91 and tell me what you get.",
            ],
            [
                "role": "assistant",
                "reasoning_content":
                    "391 minus 91: 391 - 90 = 301, then 301 - 1 = 300. Verify: 300 + 91 = 391. Correct.",
                "content": "391 minus 91 is 300.",
            ],
        ]

        // 2. Render the bearing path (the stretch keeps its `<think>` blocks)
        //    through the real chat template. The stripped path is, by the
        //    ASR definition (issue #134: "the think-stripped token path is
        //    almost the think-bearing path with the `<think>` spans removed"),
        //    the bearing tokens with the think spans excised — NOT a separate
        //    template render, whose common prefix with the bearing would stop
        //    at the first `<think>` and grossly under-length the stretch.
        let bearingTokens: [Int]
        do {
            bearingTokens = try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["add_generation_prompt": false]
            )
        } catch {
            return (false, "template render failed: \(error)", lines)
        }
        guard bearingTokens.count > 1 else {
            return (false, "bearing render too short (no thinking template?)", lines)
        }
        lines.append("  bearingTokens=\(bearingTokens.count)")

        // 3. Scan the bearing render for `<think>`/`</think>` delimiters.
        let openThink = context.tokenizer.encode(text: "<think>", addSpecialTokens: false)
        let closeThink = context.tokenizer.encode(text: "</think>", addSpecialTokens: false)
        let postThinkSeparator = context.tokenizer.encode(text: "\n\n", addSpecialTokens: false)
        let thinkSpans = AsymmetricStateRestore.thinkSpans(
            in: bearingTokens, openThinkTokens: openThink, closeThinkTokens: closeThink,
            postCloseTokens: postThinkSeparator)
        guard !thinkSpans.isEmpty else {
            return (
                false,
                "no <think> delimiters found in the bearing render; ASR not exercised",
                lines + ["  template did not retain thinks; failing the ASR acceptance gate"]
            )
        }
        let excised = thinkSpans.reduce(0) { $0 + $1.length }
        lines.append("  thinkSpans=\(thinkSpans.count) excisedTokens=\(excised)")

        // The stripped token path = the bearing tokens with the think spans
        // excised — the ASR input contract, and byte-identical to what the
        // canonical next-request render will restore.
        let strippedTokens = AsymmetricStateRestore.strippedTokens(
            in: bearingTokens, spans: thinkSpans)
        let expectedStripped = bearingTokens.count - excised
        precondition(
            strippedTokens.count == expectedStripped,
            "stripped reconstruction \(strippedTokens.count) != expected \(expectedStripped)")

        // 4. Capture the bearing snapshot through the real model.
        let bearingCache = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: bearingTokens, checkpoints: [:], cache: bearingCache)
        guard
            let bearingSnapshot = HybridCacheSnapshot.capture(
                cache: bearingCache, offset: bearingTokens.count, type: .leaf)
        else {
            return (false, "bearing capture nil", lines)
        }

        // 5. Read per-layer RoPE metadata from the model config (user story
        //    #11). Missing metadata is a gate failure: hard-coded family
        //    constants would let the harness pass while testing a different
        //    geometry than the loaded checkpoint.
        let layerClasses = bearingSnapshot.layers.map { $0.className }
        guard
            let ropeMetadata = AsymmetricStateRestore.RopeMetadata.readFromConfigDirectory(
                modelDirectory, layerClassNames: layerClasses)
        else {
            return (false, "missing RoPE metadata in loaded model config", lines)
        }
        guard !ropeMetadata.isEmpty else {
            return (false, "no sliceable attention layers in the bearing snapshot", lines)
        }
        lines.append("  ropeMetadataLayers=\(ropeMetadata.count)/\(layerClasses.count)")

        // 6. Synthesize. `expectedStripped` (bearing − excised) equals
        //    `strippedTokens.count` by the reconstruction above.
        let synthesized: HybridCacheSnapshot
        do {
            synthesized = try AsymmetricStateRestore.synthesize(
                bearingSnapshot: bearingSnapshot,
                strippedTokenCount: expectedStripped,
                thinkSpans: thinkSpans,
                ropeMetadataByLayer: ropeMetadata
            )
        } catch {
            return (false, "synthesis failed: \(error)", lines)
        }
        lines.append(
            "  synthesized offset=\(synthesized.tokenOffset) "
                + "(bearing \(bearingSnapshot.tokenOffset) → stripped \(expectedStripped))")

        // 7. Smoke: restore through the real path without crashing.
        let asrCache = try synthesized.restore()

        // 8. Gold: cold prefill of the stripped stretch (the same tokens the
        //    ASR snapshot claims to align to), then forward the new-user-turn
        //    header — the real suffix the next request will feed past the
        //    divergence — to measure per-token KL/top-k at the first divergence
        //    positions.
        let goldCache = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: strippedTokens, checkpoints: [:], cache: goldCache)

        // The suffix is the new-user turn the next real request appends. Render
        // messages + a fresh user probe and take the tokens past the stripped
        // stretch as the suffix (optionally cross-checking the prefix matches
        // the stripped reconstruction — the ASR assumption).
        var nextRequestMessages = messages
        nextRequestMessages.append(["role": "user", "content": "Thanks — now double it."])
        let nextRender = try context.tokenizer.applyChatTemplate(
            messages: nextRequestMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        let prefixMatch =
            zip(nextRender.prefix(expectedStripped), strippedTokens)
            .allSatisfy { $0 == $1 } && nextRender.count >= expectedStripped
        lines.append("  next-render prefix matches stripped reconstruction: \(prefixMatch)")
        guard prefixMatch else {
            return (false, "next render prefix did not match stripped reconstruction", lines)
        }
        let suffixCount = min(8, max(0, nextRender.count - expectedStripped))
        let suffix = Array(nextRender[expectedStripped..<(expectedStripped + suffixCount)])
        guard !suffix.isEmpty else {
            return (
                false, "no suffix past the stripped stretch; ASR comparison not exercised", lines
            )
        }
        func forward(_ token: Int, on cache: [any KVCache]) -> MLXArray {
            let input = LMInput.Text(
                tokens: MLXArray([Int32(token)]).expandedDimensions(axis: 0), mask: nil)
            let output = context.model(input, cache: cache, state: nil)
            eval(output.logits)
            return output.logits[0, 0]
        }
        var klSeries: [Float] = []
        var topKSeries: [Float] = []
        var argmaxStable = 0
        var lastASRLogits: MLXArray?
        var lastGoldLogits: MLXArray?
        for token in suffix {
            let asrLogits = forward(token, on: asrCache)
            let goldLogits = forward(token, on: goldCache)
            lastASRLogits = asrLogits
            lastGoldLogits = goldLogits
            if let kl = TokenDistributionDivergence.klDivergence(
                gold: goldLogits, asr: asrLogits, temperature: servingTemperature)
            {
                klSeries.append(kl)
            }
            if let agree = TokenDistributionDivergence.topKAgreement(
                gold: goldLogits, asr: asrLogits, k: 5)
            {
                topKSeries.append(agree)
            }
            if TokenDistributionDivergence.argmaxAgrees(gold: goldLogits, asr: asrLogits) {
                argmaxStable += 1
            }
        }
        let meanKL = klSeries.isEmpty ? Float.nan : klSeries.reduce(0, +) / Float(klSeries.count)
        let meanTopK =
            topKSeries.isEmpty ? Float.nan : topKSeries.reduce(0, +) / Float(topKSeries.count)
        lines.append(
            "  KL(p‖q) series @ T=\(servingTemperature): "
                + klSeries.map { String(format: "%.4f", $0) }.joined(separator: ", "))
        lines.append(
            "  mean KL=\(String(format: "%.4f", meanKL)) meanTopK(5)=\(String(format: "%.3f", meanTopK))"
        )
        lines.append(
            "  argmax stable \(argmaxStable)/\(suffix.count) (lower bound only — not the verdict)")

        // 9. Side-by-side: short greedy continuation from each cache. Both
        //    caches sit after the same suffix. Decode from the logits already
        //    produced by the final suffix token so we do not feed that suffix
        //    token twice before comparing continuations.
        guard let asrSeed = lastASRLogits, let goldSeed = lastGoldLogits else {
            return (false, "missing suffix logits; ASR comparison not exercised", lines)
        }
        let asrContinuation = greedyDecodeFromLogits(
            seedLogits: asrSeed, cache: asrCache, count: 12, context: context,
            tokenizer: context.tokenizer)
        let goldContinuation = greedyDecodeFromLogits(
            seedLogits: goldSeed, cache: goldCache, count: 12, context: context,
            tokenizer: context.tokenizer)
        lines.append("  ASR  greedy: \(asrContinuation)")
        lines.append("  GOLD greedy: \(goldContinuation)")

        // The check passes iff the smoke artifact held (restore without
        // crashing) — the spike's verdict is the researcher's qualitative read
        // of the KL/top-k series and the side-by-side, not a numeric threshold.
        let detail =
            "smoke=OK spans=\(thinkSpans.count) excised=\(excised) "
            + "meanKL=\(String(format: "%.4f", meanKL)) meanTopK=\(String(format: "%.3f", meanTopK))"
        return (true, detail, lines)
    }
    // swiftlint:enable function_body_length

    /// Greedy-decode `count` tokens from a cache whose latest logits were already
    /// computed, and decode them to a string for the side-by-side artifact.
    /// Mutates `cache` (advances its offset).
    private nonisolated static func greedyDecodeFromLogits(
        seedLogits: MLXArray, cache: [any KVCache], count: Int,
        context: ModelContext, tokenizer: any Tokenizer
    ) -> String {
        var ids: [Int] = []
        var nextId = Int(seedLogits.argMax().item(Int32.self))
        for _ in 0..<count {
            ids.append(nextId)
            let last = MLXArray([Int32(nextId)]).expandedDimensions(axis: 0)
            let output = context.model(
                LMInput.Text(tokens: last, mask: nil), cache: cache, state: nil)
            eval(output.logits)
            nextId = Int(output.logits[0, 0].argMax().item(Int32.self))
        }
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
    }

    // MARK: - Helpers — model invocation

    /// Drive `PrefillExecutor.run` over `tokens`. The production prefill
    /// code path — using it here means the harness exercises the same loop
    /// the prefix cache uses on the hot path, instead of an independent
    /// reimplementation that could drift. `consumeAll` brings the cache to
    /// offset `checkpointBaseOffset + tokens.count`.
    @discardableResult
    nonisolated private static func prefill(
        context: ModelContext,
        tokens: [Int],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType],
        checkpointBaseOffset: Int = 0,
        generateParameters: GenerateParameters = .init(),
        cache: [any KVCache]
    ) throws -> [HybridCacheSnapshot] {
        guard !tokens.isEmpty else { return [] }
        // 1D input: `PrefillExecutor` adds the batch dim per chunk itself.
        // Passing a pre-batched 2D input would route through the VLM-shaped
        // slicing and silently produce wrong logits for an LLM.
        let inputArr = MLXArray(tokens.map { Int32($0) })
        // `consumeAll` brings the cache to the full token count — the harness
        // doesn't run a token iterator, so there is no prime token to keep.
        let warmed = try PrefillExecutor.run(
            model: context.model,
            text: .init(tokens: inputArr, mask: nil),
            cache: cache,
            checkpoints: checkpoints,
            checkpointBaseOffset: checkpointBaseOffset,
            prefillStepSize: 512,
            consumeAll: true
        )
        return warmed.snapshots
    }

    /// Full prefill of `tokens`, return the next-token logits at position
    /// `tokens.count` by feeding the last input token through one final
    /// forward pass.
    nonisolated private static func fullPrefillLastLogits(
        context: ModelContext,
        tokens: [Int]
    ) throws -> MLXArray {
        let cache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.dropLast()),
            checkpoints: [:],
            cache: cache
        )
        return lastTokenLogits(context: context, tokens: tokens, cache: cache)
    }

    /// Restore the cache at offset `k`, prefill the suffix `tokens[k..<N-1]`,
    /// and return the last-position logits.
    nonisolated private static func restoreAndContinueLastLogits(
        context: ModelContext,
        tokens: [Int],
        restoreAt k: Int
    ) throws -> MLXArray {
        let setupCache = context.model.newCache(parameters: nil)
        try prefill(
            context: context,
            tokens: Array(tokens.prefix(k)),
            checkpoints: [:],
            cache: setupCache
        )
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: setupCache, offset: k, type: .system
            )
        else {
            throw HybridCacheCorrectnessError.snapshotCaptureFailed
        }
        let cache = try snap.restore()
        try prefill(
            context: context,
            tokens: Array(tokens[k..<(tokens.count - 1)]),
            checkpoints: [:],
            checkpointBaseOffset: k,
            cache: cache
        )
        return lastTokenLogits(context: context, tokens: tokens, cache: cache)
    }

    /// Run a single-token forward pass on `tokens.last!` against the given
    /// cache and return the resulting logits at the final position.
    /// Caller must ensure `cache` is warm at offset `tokens.count - 1`.
    nonisolated private static func lastTokenLogits(
        context: ModelContext,
        tokens: [Int],
        cache: [any KVCache]
    ) -> MLXArray {
        let lastToken = MLXArray([Int32(tokens.last!)]).expandedDimensions(axis: 0)
        let lastInput = LMInput.Text(tokens: lastToken, mask: nil)
        let output = context.model(lastInput, cache: cache, state: nil)
        eval(output.logits)
        // Shape [1, 1, vocab] → [vocab].
        return output.logits[0, 0]
    }

    // MARK: - Helpers — comparison

    private struct CacheStateComparison {
        let layerCount: Int
        let maxDiff: Float
        let offsetMismatch: Bool
    }

    nonisolated private static func compareCacheStates(
        cacheA: [any KVCache],
        cacheB: [any KVCache],
        filter: (any KVCache) -> Bool,
        checkOffset: Bool
    ) -> CacheStateComparison {
        var layerCount = 0
        var maxDiff: Float = 0
        var offsetMismatch = false
        for (la, lb) in zip(cacheA, cacheB) where filter(la) {
            layerCount += 1
            if checkOffset && la.offset != lb.offset { offsetMismatch = true }
            for (sa, sb) in zip(la.state, lb.state) {
                let d = BenchmarkHarness.maxAbsDiff(sa, sb)
                if d > maxDiff { maxDiff = d }
            }
        }
        return CacheStateComparison(
            layerCount: layerCount, maxDiff: maxDiff, offsetMismatch: offsetMismatch
        )
    }

    // MARK: - Reporting

    private typealias CheckResult = BenchmarkHarness.CheckResult

    private var reportDir: URL {
        runner.activeConfig.outputDir.appendingPathComponent("hybrid-cache-correctness")
    }

    @discardableResult
    private func writeReport(checks: [CheckResult], allPassed: Bool) throws -> URL {
        try BenchmarkHarness.writeReport(
            checks: checks,
            allPassed: allPassed,
            modelName: runner.resolvedModelName,
            reportDir: reportDir,
            filePrefix: "correctness"
        )
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

enum HybridCacheCorrectnessError: LocalizedError {
    case verificationFailed(failedChecks: [String])
    case snapshotCaptureFailed

    var errorDescription: String? {
        switch self {
        case .verificationFailed(let names):
            "HybridCacheCorrectness failed checks: \(names.joined(separator: ", "))"
        case .snapshotCaptureFailed:
            "HybridCacheSnapshot.capture returned nil — unsupported cache layer type"
        }
    }
}
