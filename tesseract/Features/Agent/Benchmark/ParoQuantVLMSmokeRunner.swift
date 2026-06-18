import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import os

/// VLM smoke + warmed-prefix-cache spike harness (`--paroquant-vlm-smoke`).
///
/// Original scope (PR #164 review comment C5): load the VLM container of
/// `loadParoQuantModel` and assert key resolution. Extended for PRD #72
/// (image-aware prefix caching) into the **warmed-cache spike**: before any
/// keying machinery is built, empirically pin how Qwen3.5-VLM M-RoPE
/// positions behave across snapshot restore boundaries, using only the
/// vanilla vendor surface.
///
/// The checks encode the spike's hypotheses, derived from reading the
/// vendored position machinery:
///
/// 1. `prepare()` hardcodes `state: nil`, and the recompute branch derives
///    positions from its own inputs starting at zero — so any forward that
///    starts on a warmed cache without threaded/seeded state is mis-positioned
///    by the restore offset (`…Diverges` checks).
/// 2. The continuation branch (sequential positions = arange + cache offset +
///    RoPE delta) is reachable app-side by seeding `LMOutput.State` with the
///    vendor's `"qwen35.ropeDeltas"` key (public `LMOutput.Key(String)` init)
///    — so **text** remainders restore correctly with zero vendor changes
///    (`…Matches` checks).
/// 3. The RoPE delta at any boundary is reconstructible app-side from the
///    processed image grids (Σ per image: max(t, h/m, w/m) − t·h·w/m²), so
///    restoring *after* a cached image needs no extra persistence
///    (`prepareStateCarriesRopeDelta`).
///
/// A diverging `…Matches` check or a matching `…Diverges` check falsifies the
/// analysis and reshapes PRD #72.
@MainActor
// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable:next type_body_length
final class ParoQuantVLMSmokeRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private lazy var reportDir: URL = runner.activeConfig.outputDir
        .appendingPathComponent("paroquant-vlm-smoke")

    /// Snapshot restore is a deep tensor copy; with correctly anchored
    /// positions the warm path must reproduce the cold path bit-for-bit.
    nonisolated private static let bitwiseTolerance: Float = 0.0

    /// App-side handle to the vendor's Qwen3.5 rope-delta state slot. The
    /// string is vendor-internal but stable (`Qwen35.swift`); the
    /// `prepareStateCarriesRopeDelta` check pins it — if the vendor renames
    /// the key, the harvested delta comes back nil and the check fails.
    nonisolated private static let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

    init(runner: BenchmarkRunner) {
        self.runner = runner
    }

    func run() async throws {
        setupLogging()
        log("ParoQuantVLMSmoke + warmed-prefix spike starting — model=\(runner.resolvedModelName)")

        let modelDir = try runner.resolveModelDirectory()
        log("Loading VLM model from: \(modelDir.path)")

        let engine = AgentEngine()
        do {
            try await engine.loadModel(from: modelDir, visionMode: true)
        } catch {
            log("❌ loadModel(visionMode: true) failed: \(error)")
            throw ParoQuantVLMSmokeError.loadFailed(String(describing: error))
        }

        guard engine.isModelLoaded else {
            log("❌ engine reports not loaded after loadModel")
            throw ParoQuantVLMSmokeError.engineNotReady
        }
        log("✅ VLM model loaded — in_proj_ba split + all PARO rotation keys resolved cleanly.")

        let testRun = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.runAllChecks(context: context)
            }
        }

        for line in testRun.logs { log(line) }

        log("\n── Summary ──")
        var allPassed = true
        for check in testRun.checks {
            let mark = check.passed ? "✅" : "❌"
            log("  \(mark) \(check.name): \(check.detail)")
            if !check.passed { allPassed = false }
        }
        try writeReport(checks: testRun.checks, allPassed: allPassed)

        engine.unloadModel()
        await engine.awaitPendingUnload()
        log("✅ Engine unloaded cleanly.")

        log("Overall: \(allPassed ? "PASS" : "FAIL")")
        if !allPassed {
            throw ParoQuantVLMSmokeError.spikeChecksFailed(
                testRun.checks.filter { !$0.passed }.map { $0.name }
            )
        }
    }

    // MARK: - Orchestration

    private struct TestRunResult: Sendable {
        let logs: [String]
        let checks: [CheckResult]
    }

    private typealias CheckResult = BenchmarkHarness.CheckResult

    nonisolated private static func runAllChecks(context: ModelContext) async throws
        -> TestRunResult
    {
        var logs: [String] = []
        var checks: [CheckResult] = []

        guard let vlm = context.model as? Qwen35 else {
            return TestRunResult(
                logs: [
                    "model is \(type(of: context.model)), not Qwen35 — spike requires the Qwen3.5 VLM container"
                ],
                checks: [CheckResult(name: "vlmContainerType", passed: false, detail: "not Qwen35")]
            )
        }

        // Shared fixtures. One deterministic image, one image-bearing prompt
        // (image up front, agent-attachment shape), one text-only prompt.
        let image = BenchmarkHarness.deterministicImage(width: 256, height: 256, seed: 17)
        let image2 = BenchmarkHarness.deterministicImage(width: 256, height: 256, seed: 41)
        let imageInput = try await prepareImageInput(context: context, image: image)
        let textTokens = BenchmarkHarness.promptTokens(
            targetTokens: 1536, tokenizer: context.tokenizer)
        logs.append("image prompt: \(imageInput.text.tokens.dim(-1)) tokens")
        logs.append("text prompt:  \(textTokens.count) tokens")

        func runCheck(
            _ name: String,
            _ body: () async throws -> (passed: Bool, detail: String, lines: [String])
        ) async {
            logs.append("\n── \(name) ──")
            do {
                let result = try await body()
                logs.append(contentsOf: result.lines)
                checks.append(CheckResult(name: name, passed: result.passed, detail: result.detail))
            } catch {
                logs.append("  ❌ threw: \(error)")
                checks.append(CheckResult(name: name, passed: false, detail: "threw: \(error)"))
            }
        }

        await runCheck("coldImagePrepareDeterministic") {
            try coldImagePrepareDeterministic(context: context, input: imageInput)
        }
        await runCheck("prepareStateCarriesRopeDelta") {
            try prepareStateCarriesRopeDelta(context: context, vlm: vlm, input: imageInput)
        }
        await runCheck("iteratorStateDropDiverges") {
            try iteratorStateDropDiverges(context: context, input: imageInput)
        }
        await runCheck("chunkShapeNoiseControl") {
            try chunkShapeNoiseControl(context: context, tokens: textTokens)
        }
        await runCheck("textOnlyNaiveRestoreDiverges") {
            try textOnlyRestore(context: context, tokens: textTokens, seeded: false)
        }
        await runCheck("textOnlySeededRestoreMatches") {
            try textOnlyRestore(context: context, tokens: textTokens, seeded: true)
        }
        await runCheck("imageLeafRestoreTextRemainderMatches") {
            try imageLeafRestoreTextRemainder(context: context, input: imageInput)
        }
        await runCheck("naiveImageRemainderPrepareDiverges") {
            try await naiveImageRemainderPrepare(context: context, vlm: vlm, image: image)
        }
        await runCheck("imageRemainderContinuationMatches") {
            try await imageRemainderContinuationMatches(context: context, vlm: vlm, image: image)
        }
        await runCheck("stackedImageRemainderContinuationMatches") {
            try await stackedImageRemainderContinuationMatches(
                context: context, vlm: vlm, image1: image, image2: image2)
        }
        await runCheck("unkeyedImageWholeContinuationMatches") {
            try unkeyedImageWholeContinuationMatches(context: context, vlm: vlm, input: imageInput)
        }

        return TestRunResult(logs: logs, checks: checks)
    }

    // MARK: - Checks

    /// Same image input prepared twice on fresh caches must produce identical
    /// last-position logits — pins vision-tower + prepare determinism, the
    /// precondition for every bitwise comparison below.
    nonisolated private static func coldImagePrepareDeterministic(
        context: ModelContext,
        input: LMInput
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let (logitsA, _) = try coldLastLogits(context: context, input: input)
        let (logitsB, _) = try coldLastLogits(context: context, input: input)
        let diff = BenchmarkHarness.maxAbsDiff(logitsA, logitsB)
        return (diff <= bitwiseTolerance, "maxAbsDiff=\(diff)", [])
    }

    /// `prepare()`'s returned state must carry the M-RoPE delta, and the delta
    /// must equal the app-side reconstruction from the processed image grids:
    /// Σ per image (max(t, h/m, w/m) − t·h·w/m²). This is the formula
    /// production uses to anchor positions when restoring after cached images.
    nonisolated private static func prepareStateCarriesRopeDelta(
        context: ModelContext,
        vlm: Qwen35,
        input: LMInput
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let cache = context.model.newCache(parameters: nil)
        let result = try context.model.prepare(input, cache: cache, windowSize: nil)
        guard case .logits(let output) = result else {
            return (false, "prepare returned .tokens — unexpected for the VLM container", [])
        }
        guard let state = output.state, let deltas = state[ropeDeltasKey] else {
            return (false, "no \"qwen35.ropeDeltas\" in prepare state — vendor key drifted?", [])
        }
        eval(deltas)
        let harvested = deltas.asArray(Int32.self).map(Int.init)

        let merge = vlm.config.visionConfiguration.spatialMergeSize
        var computed = 0
        for frame in input.image?.frames ?? [] {
            let (t, h, w) = frame.values
            let span = max(t, max(h / merge, w / merge))
            let runLength = (t * h * w) / (merge * merge)
            computed += span - runLength
        }

        let matches = harvested == [computed]
        return (
            matches,
            "harvested=\(harvested) computedFromGrids=\(computed)",
            [
                "  frames=\((input.image?.frames ?? []).map(\.values).map { [$0.0, $0.1, $0.2] }) merge=\(merge)"
            ]
        )
    }

    /// Evidence for the vanilla decode bug: the upstream `TokenIterator`
    /// discards `prepare()`'s returned state, so its first decode forward runs
    /// with nil state and recomputes positions from zero. Forwarding the same
    /// sentinel with threaded vs nil state must differ — if it doesn't,
    /// positions don't matter here and the whole spike premise is wrong.
    nonisolated private static func iteratorStateDropDiverges(
        context: ModelContext,
        input: LMInput
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let (threaded, _) = try coldLastLogits(context: context, input: input)

        // Same computation, but the final single-token forward drops state —
        // exactly what TokenIterator's `.logits` path does today.
        let cache = context.model.newCache(parameters: nil)
        let tokens = LLMActor.extractTokenSequence(input.text.tokens)
        let prefix = sliced2D(input.text.tokens, to: tokens.count - 1)
        let prefixInput = LMInput(
            text: .init(tokens: prefix, mask: ones(like: prefix).asType(.int8)),
            image: input.image
        )
        guard case .logits = try context.model.prepare(prefixInput, cache: cache, windowSize: nil)
        else {
            return (false, "prepare returned .tokens", [])
        }
        let (dropped, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1], cache: cache, state: nil
        )

        let diff = BenchmarkHarness.maxAbsDiff(threaded, dropped)
        return (diff > 0, "maxAbsDiff=\(diff) (expected > 0: nil-state decode mis-positions)", [])
    }

    /// Pure chunk-shape control — no restore, no reseeding, one continuously
    /// threaded state on one cache, but the prefill split into two calls at a
    /// non-chunk-aligned offset. Positions are identical by construction, so
    /// any difference vs the single continuous run is floating-point noise
    /// from differing kernel shapes (attention / GatedDeltaNet scans at
    /// different chunk lengths). Bounds what "bitwise" can even mean for
    /// misaligned restore offsets on this stack.
    nonisolated private static func chunkShapeNoiseControl(
        context: ModelContext,
        tokens: [Int]
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let split = 768  // deliberately not a multiple of the 512 prefill step

        let cacheA = context.model.newCache(parameters: nil)
        let outA = try prefill(context: context, tokens: Array(tokens.dropLast()), cache: cacheA)
        let (logitsA, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1], cache: cacheA, state: outA.state
        )

        let cacheB = context.model.newCache(parameters: nil)
        let outB1 = try prefill(
            context: context, tokens: Array(tokens.prefix(split)), cache: cacheB)
        let outB2 = try prefill(
            context: context, tokens: Array(tokens[split..<(tokens.count - 1)]),
            checkpointBaseOffset: split, cache: cacheB, initialState: outB1.state
        )
        let (logitsB, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1], cache: cacheB, state: outB2.state
        )

        let diff = BenchmarkHarness.maxAbsDiff(logitsA, logitsB)
        return (
            true,
            "maxAbsDiff=\(diff) (informational: FP noise floor from chunk shapes alone)",
            [
                "  argmax A=\(logitsA.argMax().item(Int32.self)) B=\(logitsB.argMax().item(Int32.self))"
            ]
        )
    }

    /// Text-only prompt on the VLM container, cold full prefill vs
    /// restore-at-K + remainder prefill, K chunk-aligned so both paths run
    /// identical kernel shapes and FP-noise cannot mask position behavior.
    ///
    /// - `seeded == false` is today's production path (`PrefillExecutor` with
    ///   nil state): the first remainder chunk recomputes positions from zero
    ///   → expected to diverge. This is the latent vision-mode text bug.
    /// - `seeded == true` seeds rope delta 0 (image-free prefix) → the
    ///   continuation branch anchors at the restore offset → expected bitwise.
    nonisolated private static func textOnlyRestore(
        context: ModelContext,
        tokens: [Int],
        seeded: Bool
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let k = 512  // chunk-aligned: cold and warm chunk boundaries coincide

        let coldCache = context.model.newCache(parameters: nil)
        let coldOut = try prefill(
            context: context, tokens: Array(tokens.dropLast()), cache: coldCache
        )
        let (coldLogits, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1],
            cache: coldCache, state: coldOut.state
        )

        let setupCache = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: Array(tokens.prefix(k)), cache: setupCache)
        guard let snap = HybridCacheSnapshot.capture(cache: setupCache, offset: k, type: .system)
        else {
            return (false, "capture nil", [])
        }
        let restored = try snap.restore()

        var initialState: LMOutput.State?
        if seeded {
            var state = LMOutput.State()
            state[ropeDeltasKey] = MLXArray([Int32(0)])
            initialState = state
        }
        let warmOut = try prefill(
            context: context, tokens: Array(tokens[k..<(tokens.count - 1)]),
            checkpointBaseOffset: k, cache: restored, initialState: initialState
        )
        let (warmLogits, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1],
            cache: restored, state: warmOut.state
        )

        let diff = BenchmarkHarness.maxAbsDiff(coldLogits, warmLogits)
        if seeded {
            return (
                diff <= bitwiseTolerance,
                "maxAbsDiff=\(diff) (expected 0: seeded delta anchors positions)", []
            )
        }
        return (
            diff > 0,
            "maxAbsDiff=\(diff) (expected > 0: nil-state remainder restarts positions at 0)", []
        )
    }

    /// Turn-over-turn chaining: prefill an image turn cold, capture the leaf
    /// *after* the image, restore it, and prefill a text continuation with the
    /// rope delta harvested from the image turn (in production: reconstructed
    /// from the image grids — `prepareStateCarriesRopeDelta` pins the two
    /// equal). This is the case the PRD's user stories live or die on — text
    /// turns chaining off a cached image turn.
    ///
    /// The bitwise reference is the *shape-matched* cold chain: the same
    /// prepare-then-continue computation with one continuously threaded state
    /// and no capture/restore round trip. Against it, the restored path
    /// isolates exactly two things: snapshot restore fidelity and the
    /// app-side delta reconstruction — the spike's actual questions. The
    /// single-shot cold prepare diff is reported as information; it includes
    /// chunk-shape FP noise (see `chunkShapeNoiseControl`).
    nonisolated private static func imageLeafRestoreTextRemainder(
        context: ModelContext,
        input: LMInput
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let turn1Tokens = LLMActor.extractTokenSequence(input.text.tokens)
        let suffixText = """
            The image shows a synthetic gradient pattern. Now, setting the image aside: \
            list three properties a deterministic test fixture must have, and explain \
            why reproducibility matters for bitwise cache verification.
            """
        let suffix = context.tokenizer.encode(text: suffixText, addSpecialTokens: false)
        let full = turn1Tokens + suffix

        // Shape-matched cold chain: prepare(turn 1) → threaded-state suffix
        // prefill → sentinel forward. One cache, no restore.
        let coldCache = context.model.newCache(parameters: nil)
        guard
            case .logits(let coldTurn1) = try context.model.prepare(
                input, cache: coldCache, windowSize: nil
            )
        else {
            return (false, "cold turn-1 prepare returned .tokens", [])
        }
        eval(coldCache)
        let coldOut = try prefill(
            context: context, tokens: Array(suffix.dropLast()),
            checkpointBaseOffset: turn1Tokens.count, cache: coldCache,
            initialState: coldTurn1.state
        )
        let (coldLogits, _) = lastTokenLogits(
            model: context.model, token: full[full.count - 1],
            cache: coldCache, state: coldOut.state
        )

        // Informational: true single-shot cold prepare over the full
        // two-turn sequence (production cold shape for a turn-2 request).
        let singleCache = context.model.newCache(parameters: nil)
        let singlePrefix = MLXArray(full.dropLast().map { Int32($0) }).expandedDimensions(axis: 0)
        let singleInput = LMInput(
            text: .init(tokens: singlePrefix, mask: ones(like: singlePrefix).asType(.int8)),
            image: input.image
        )
        guard
            case .logits(let singleOut) = try context.model.prepare(
                singleInput, cache: singleCache, windowSize: nil
            )
        else {
            return (false, "single-shot cold prepare returned .tokens", [])
        }
        eval(singleCache)
        let (singleLogits, _) = lastTokenLogits(
            model: context.model, token: full[full.count - 1],
            cache: singleCache, state: singleOut.state
        )

        // Warm path: image turn prefilled cold (full turn-1 prompt), leaf
        // captured at its end, restored, then the text continuation prefills
        // with the harvested delta seeded into fresh state.
        let warmSetup = context.model.newCache(parameters: nil)
        guard
            case .logits(let turn1Out) = try context.model.prepare(
                input, cache: warmSetup, windowSize: nil
            )
        else {
            return (false, "turn-1 prepare returned .tokens", [])
        }
        eval(warmSetup)
        guard let turn1State = turn1Out.state, let turn1Deltas = turn1State[ropeDeltasKey] else {
            return (false, "no rope delta in turn-1 prepare state", [])
        }
        guard
            let snap = HybridCacheSnapshot.capture(
                cache: warmSetup, offset: turn1Tokens.count, type: .system
            )
        else {
            return (false, "leaf capture nil", [])
        }
        let restored = try snap.restore()

        var seeded = LMOutput.State()
        seeded[ropeDeltasKey] = turn1Deltas
        let warmOut = try prefill(
            context: context, tokens: Array(suffix.dropLast()),
            checkpointBaseOffset: turn1Tokens.count, cache: restored, initialState: seeded
        )
        let (warmLogits, _) = lastTokenLogits(
            model: context.model, token: full[full.count - 1],
            cache: restored, state: warmOut.state
        )

        let diff = BenchmarkHarness.maxAbsDiff(coldLogits, warmLogits)
        let singleShotDiff = BenchmarkHarness.maxAbsDiff(singleLogits, warmLogits)
        return (
            diff <= bitwiseTolerance,
            "maxAbsDiff=\(diff) vs shape-matched cold (expected 0); "
                + "singleShotColdDiff=\(singleShotDiff) (informational, incl. chunk-shape FP noise)",
            [
                "  turn1=\(turn1Tokens.count) tokens, suffix=\(suffix.count) tokens",
                "  argmax warm=\(warmLogits.argMax().item(Int32.self)) "
                    + "singleShotCold=\(singleLogits.argMax().item(Int32.self))",
            ]
        )
    }

    /// The negative that bounds phase 1: restore an image-free text prefix and
    /// run the image-bearing remainder through `prepare()` — the only public
    /// entry that runs the vision tower. `prepare` starts from nil state, so
    /// its M-RoPE recompute is 0-based instead of anchored at the restore
    /// offset → expected to diverge. Image-bearing remainders therefore cannot
    /// restore through the vanilla vendor surface; they serve cold.
    nonisolated private static func naiveImageRemainderPrepare(
        context: ModelContext,
        vlm: Qwen35,
        image: CIImage
    ) async throws -> (passed: Bool, detail: String, lines: [String]) {
        // Long text prefix, then the image: the HTTP debug-flow shape.
        let filler = BenchmarkHarness.promptText(targetTokens: 900, tokenizer: context.tokenizer)
        let chat: [Chat.Message] = [
            .system("You are a meticulous visual analyst. Context notes: " + filler),
            .user(
                "Describe the attached image: dominant colors, gradients, any structure.",
                images: [.ciImage(image)]
            ),
        ]
        let prepared = try await context.processor.prepare(input: UserInput(chat: chat))
        let tokens = LLMActor.extractTokenSequence(prepared.text.tokens)

        guard let visionStart = tokens.firstIndex(of: vlm.config.visionStartTokenId),
            visionStart > 64
        else {
            return (
                false, "no vision start token (or image not in remainder) — prompt shape wrong", []
            )
        }
        let k = visionStart - 8

        let coldCache = context.model.newCache(parameters: nil)
        let coldPrefixArr = sliced2D(prepared.text.tokens, to: tokens.count - 1)
        let coldInput = LMInput(
            text: .init(tokens: coldPrefixArr, mask: ones(like: coldPrefixArr).asType(.int8)),
            image: prepared.image
        )
        guard
            case .logits(let coldOut) = try context.model.prepare(
                coldInput, cache: coldCache, windowSize: nil
            )
        else {
            return (false, "cold prepare returned .tokens", [])
        }
        eval(coldCache)
        let (coldLogits, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1],
            cache: coldCache, state: coldOut.state
        )

        // Warm-naive: prefill the text prefix, capture/restore at K, then
        // vendor prepare over the image-bearing remainder.
        let setupCache = context.model.newCache(parameters: nil)
        try prefill(context: context, tokens: Array(tokens.prefix(k)), cache: setupCache)
        guard let snap = HybridCacheSnapshot.capture(cache: setupCache, offset: k, type: .system)
        else {
            return (false, "capture nil", [])
        }
        let restored = try snap.restore()

        let remainder = Array(tokens[k..<(tokens.count - 1)])
        let remainderArr = MLXArray(remainder.map { Int32($0) }).expandedDimensions(axis: 0)
        let remainderInput = LMInput(
            text: .init(tokens: remainderArr, mask: ones(like: remainderArr).asType(.int8)),
            image: prepared.image
        )
        guard
            case .logits(let warmOut) = try context.model.prepare(
                remainderInput, cache: restored, windowSize: nil
            )
        else {
            return (false, "remainder prepare returned .tokens", [])
        }
        eval(restored)
        let (warmLogits, _) = lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1],
            cache: restored, state: warmOut.state
        )

        let diff = BenchmarkHarness.maxAbsDiff(coldLogits, warmLogits)
        return (
            diff > 0,
            "maxAbsDiff=\(diff) (expected > 0: prepare cannot anchor at the restore offset)",
            ["  prompt=\(tokens.count) tokens, restore K=\(k), visionStart=\(visionStart)"]
        )
    }

    /// The phase-2 positive that inverts `naiveImageRemainderPrepare`: the same
    /// long-text-prefix-then-image prompt, but the image-bearing remainder is
    /// continued through the **windowed `prepareContinuation`** with the restore
    /// offset's Position Anchor seeded (rope delta 0 — the prefix is text).
    /// Three frozen facts, together proving warm image continuation is correct:
    ///
    /// 1. **Bitwise vs shape-matched** — the restored continuation reproduces
    ///    the restore-free continuation (same prefill + same chunked
    ///    continuation, no capture/restore round trip) bit-for-bit, isolating
    ///    snapshot restore fidelity and chunk determinism.
    /// 2. **Argmax vs production-cold** — it lands the same top token as the
    ///    single-shot cold `prepare` over the whole prefix, proving the
    ///    offset-aware M-RoPE image positions are correct, not merely
    ///    self-consistent. (Bitwise against single-shot cold is impossible — the
    ///    chunk shapes differ; see `chunkShapeNoiseControl`.)
    /// 3. **Resumed anchor** — the continuation's returned state carries the
    ///    rope delta the post-image text tail resumes with, equal to the
    ///    app-side reconstruction from the image grids (Σ max(t,h/m,w/m) −
    ///    t·h·w/m²) that `prepareStateCarriesRopeDelta` pins for cold prepare.
    ///
    /// A diverging bitwise diff, a cold-mismatched argmax, or a wrong resumed
    /// delta falsifies ADR-0007 phase 2 and the swap in `ServerCompletion`.
    nonisolated private static func imageRemainderContinuationMatches(
        context: ModelContext,
        vlm: Qwen35,
        image: CIImage
    ) async throws -> (passed: Bool, detail: String, lines: [String]) {
        // Identical prompt shape to the `…Diverges` negative: long text prefix,
        // then the image — the HTTP debug-flow shape.
        let filler = BenchmarkHarness.promptText(targetTokens: 900, tokenizer: context.tokenizer)
        let chat: [Chat.Message] = [
            .system("You are a meticulous visual analyst. Context notes: " + filler),
            .user(
                "Describe the attached image: dominant colors, gradients, any structure.",
                images: [.ciImage(image)]
            ),
        ]
        let prepared = try await context.processor.prepare(input: UserInput(chat: chat))
        let tokens = LLMActor.extractTokenSequence(prepared.text.tokens)

        guard let visionStart = tokens.firstIndex(of: vlm.config.visionStartTokenId),
            visionStart > 64
        else {
            return (
                false, "no vision start token (or image not in remainder) — prompt shape wrong", []
            )
        }
        let k = visionStart - 8  // clean text boundary below the image run

        // Production-cold reference (argmax target): single-shot prepare over
        // the whole [0, last) prefix, image included.
        let (coldLogits, _) = try coldLastLogits(context: context, input: prepared)

        // Continue the image-bearing remainder [k, last) through the windowed
        // continuation, the image-free prefix's anchor (delta 0) seeded.
        // `restore` toggles only the capture/restore round trip; the prefill
        // shape, chunk shape, and anchor are otherwise identical, so the two
        // runs can differ only by restore fidelity.
        func continueRemainder(restore: Bool) throws -> (logits: MLXArray, state: LMOutput.State?) {
            let cache = context.model.newCache(parameters: nil)
            try prefill(context: context, tokens: Array(tokens.prefix(k)), cache: cache)

            let working: [any KVCache]
            if restore {
                guard let snap = HybridCacheSnapshot.capture(cache: cache, offset: k, type: .system)
                else {
                    throw ParoQuantVLMSmokeError.unexpectedPrepareResult
                }
                working = try snap.restore()
            } else {
                eval(cache)
                working = cache
            }

            var anchor = LMOutput.State()
            anchor[ropeDeltasKey] = MLXArray([Int32(0)])  // [0, k) is text → delta 0

            let remainder = Array(tokens[k..<(tokens.count - 1)])
            let remainderArr = MLXArray(remainder.map { Int32($0) }).expandedDimensions(axis: 0)
            let remainderInput = LMInput(
                text: .init(tokens: remainderArr, mask: ones(like: remainderArr).asType(.int8)),
                image: prepared.image
            )
            guard
                case .logits(let out) = try vlm.prepareContinuation(
                    remainderInput, cache: working, state: anchor, windowSize: 512
                )
            else {
                throw ParoQuantVLMSmokeError.unexpectedPrepareResult
            }
            eval(working)
            let (logits, _) = lastTokenLogits(
                model: context.model, token: tokens[tokens.count - 1], cache: working,
                state: out.state
            )
            return (logits, out.state)
        }

        let reference = try continueRemainder(restore: false)
        let warm = try continueRemainder(restore: true)

        let bitwiseDiff = BenchmarkHarness.maxAbsDiff(reference.logits, warm.logits)
        let argmaxWarm = warm.logits.argMax().item(Int32.self)
        let argmaxCold = coldLogits.argMax().item(Int32.self)

        // Resumed anchor: the post-image text tail's rope delta, reconstructed
        // from the image grids exactly as production does.
        let merge = vlm.config.visionConfiguration.spatialMergeSize
        var computed = 0
        for frame in prepared.image?.frames ?? [] {
            let (t, h, w) = frame.values
            computed += max(t, max(h / merge, w / merge)) - (t * h * w) / (merge * merge)
        }
        var harvested: [Int] = []
        if let state = warm.state, let deltas = state[ropeDeltasKey] {
            eval(deltas)
            harvested = deltas.asArray(Int32.self).map(Int.init)
        }

        let passed =
            bitwiseDiff <= bitwiseTolerance && argmaxWarm == argmaxCold && harvested == [computed]
        return (
            passed,
            "bitwiseVsShapeMatched=\(bitwiseDiff) (expected 0); "
                + "argmax warm=\(argmaxWarm) cold=\(argmaxCold) (expected equal); "
                + "resumedDelta=\(harvested) computedFromGrids=\(computed)",
            [
                "  prompt=\(tokens.count) tokens, restore K=\(k), visionStart=\(visionStart)",
                "  merge=\(merge) frames=\((prepared.image?.frames ?? []).map(\.values).map { [$0.0, $0.1, $0.2] })",
            ]
        )
    }

    /// The **stacked-image** continuation — the riskiest position path, and the
    /// one `imageRemainderContinuationMatches` (anchor delta 0, single image)
    /// does not reach. Two images: the first is *cached* (so the anchor delta is
    /// non-zero — the whole reason the Position Anchor exists), the second lands
    /// in the continued remainder. This exercises the two genuinely-new pieces
    /// of logic together:
    ///
    /// 1. **Non-zero anchor** — `prepareContinuation` positions the second
    ///    image's diverging t/h/w indices from `cacheOffset + image1Delta`, not
    ///    zero and not `cacheOffset` alone.
    /// 2. **Pixel-row skipping** — only the *second* image's pixel patches are
    ///    fed (the first image's `THW.product` rows are sliced off, exactly as
    ///    `ServerCompletion.imageSpan` does for already-cached images). A
    ///    misaligned skip silently feeds the wrong patches.
    ///
    /// Same three frozen facts as the single-image case: bitwise vs a
    /// shape-matched restore-free continuation, argmax vs single-shot cold over
    /// *both* images, and the resumed anchor equal to the *sum* of both image
    /// deltas. A wrong skip or a wrong anchor diverges the argmax from cold.
    nonisolated private static func stackedImageRemainderContinuationMatches(
        context: ModelContext,
        vlm: Qwen35,
        image1: CIImage,
        image2: CIImage
    ) async throws -> (passed: Bool, detail: String, lines: [String]) {
        let chat: [Chat.Message] = [
            .system("You are a meticulous visual analyst."),
            .user(
                "Compare these two images: dominant colors, gradients, and any structure they share.",
                images: [.ciImage(image1), .ciImage(image2)]
            ),
        ]
        let prepared = try await context.processor.prepare(input: UserInput(chat: chat))
        let tokens = LLMActor.extractTokenSequence(prepared.text.tokens)

        var visionStarts: [Int] = []
        for (i, t) in tokens.enumerated() where t == vlm.config.visionStartTokenId {
            visionStarts.append(i)
        }
        guard visionStarts.count >= 2,
            let pixels = prepared.image?.pixels,
            let frames = prepared.image?.frames, frames.count == 2
        else {
            return (
                false,
                "expected 2 images / 2 vision starts; got \(visionStarts.count) starts, "
                    + "\(prepared.image?.frames?.count ?? 0) frames",
                []
            )
        }
        // Restore boundary = the second image's <|vision_start|>: image 1 (and
        // any inter-image text) is fully below it, image 2 fully at/above it.
        let k = visionStarts[1]
        let merge = vlm.config.visionConfiguration.spatialMergeSize
        func gridDelta(_ frame: THW) -> Int {
            let (t, h, w) = frame.values
            return max(t, max(h / merge, w / merge)) - (t * h * w) / (merge * merge)
        }
        let image1Delta = gridDelta(frames[0])
        let totalDelta = image1Delta + gridDelta(frames[1])
        let image1Patches = frames[0].product  // pixel rows belonging to image 1

        // Production-cold reference (argmax target): single-shot prepare over
        // the whole [0, last) prefix with BOTH images.
        let (coldLogits, _) = try coldLastLogits(context: context, input: prepared)

        // Prefill [0, k) with image 1's vision features (so the cache holds the
        // image, not raw pad-token embeddings), restore optionally, then
        // continue [k, last) through image 2 with image 1's anchor seeded and
        // only image 2's pixels fed.
        func continueStacked(restore: Bool) throws -> (logits: MLXArray, state: LMOutput.State?) {
            let cache = context.model.newCache(parameters: nil)
            let prefixTokens = sliced2D(prepared.text.tokens, to: k)
            let prefixInput = LMInput(
                text: .init(tokens: prefixTokens, mask: ones(like: prefixTokens).asType(.int8)),
                image: LMInput.ProcessedImage(
                    pixels: pixels[0..<image1Patches, 0...], frames: [frames[0]])
            )
            guard
                case .logits = try context.model.prepare(prefixInput, cache: cache, windowSize: nil)
            else {
                throw ParoQuantVLMSmokeError.unexpectedPrepareResult
            }

            let working: [any KVCache]
            if restore {
                guard let snap = HybridCacheSnapshot.capture(cache: cache, offset: k, type: .system)
                else {
                    throw ParoQuantVLMSmokeError.unexpectedPrepareResult
                }
                working = try snap.restore()
            } else {
                eval(cache)
                working = cache
            }

            var anchor = LMOutput.State()
            anchor[ropeDeltasKey] = MLXArray([Int32(image1Delta)])  // image 1 is cached

            let remainder = Array(tokens[k..<(tokens.count - 1)])
            let remainderArr = MLXArray(remainder.map { Int32($0) }).expandedDimensions(axis: 0)
            let remainderInput = LMInput(
                text: .init(tokens: remainderArr, mask: ones(like: remainderArr).asType(.int8)),
                // Skip image 1's patch rows — only image 2's pixels are new.
                image: LMInput.ProcessedImage(
                    pixels: pixels[image1Patches..., 0...], frames: [frames[1]])
            )
            guard
                case .logits(let out) = try vlm.prepareContinuation(
                    remainderInput, cache: working, state: anchor, windowSize: 512
                )
            else {
                throw ParoQuantVLMSmokeError.unexpectedPrepareResult
            }
            eval(working)
            let (logits, _) = lastTokenLogits(
                model: context.model, token: tokens[tokens.count - 1], cache: working,
                state: out.state
            )
            return (logits, out.state)
        }

        let reference = try continueStacked(restore: false)
        let warm = try continueStacked(restore: true)

        let bitwiseDiff = BenchmarkHarness.maxAbsDiff(reference.logits, warm.logits)
        let argmaxWarm = warm.logits.argMax().item(Int32.self)
        let argmaxCold = coldLogits.argMax().item(Int32.self)
        var harvested: [Int] = []
        if let state = warm.state, let deltas = state[ropeDeltasKey] {
            eval(deltas)
            harvested = deltas.asArray(Int32.self).map(Int.init)
        }

        let passed =
            bitwiseDiff <= bitwiseTolerance && argmaxWarm == argmaxCold && harvested == [totalDelta]
        return (
            passed,
            "bitwiseVsShapeMatched=\(bitwiseDiff) (expected 0); "
                + "argmax warm=\(argmaxWarm) cold=\(argmaxCold) (expected equal); "
                + "resumedDelta=\(harvested) sumOfImageDeltas=\(totalDelta)",
            [
                "  prompt=\(tokens.count) tokens, restore K=\(k) (image-2 vision start), "
                    + "image1Patches=\(image1Patches) image1Delta=\(image1Delta)",
                "  frames=\(frames.map(\.values).map { [$0.0, $0.1, $0.2] }) merge=\(merge)",
            ]
        )
    }

    /// The **unkeyed image fallback** (ADR-0007 phase 2, the P2 gap closed in
    /// `ServerCompletion.makeUnkeyedGeneration`): when cache keying fails on an
    /// image-bearing request (e.g. a placeholder/grid mismatch), the request
    /// must not fall back to the vendor's single-shot `[heads, L, L]` `prepare`
    /// and crash. Instead the *whole* `[0, last)` input — vision tower, image →
    /// token merge, and text tail — is driven through the windowed
    /// `prepareContinuation` on a fresh cache anchored at zero (`state: nil`),
    /// the exact shape `StateThreadedTokenIterator(preparing:prepare:)` injects.
    ///
    /// Unlike the remainder checks, nothing is split off; this is the only check
    /// that runs `prepareContinuation` over a *full* image-bearing prompt from
    /// zero. It must land the same top token as the single-shot cold `prepare`
    /// over the same prefix — proving the from-zero whole-input continuation
    /// positions the image correctly. (Bitwise is impossible: the chunked
    /// continuation runs different kernel shapes than single-shot `prepare`; see
    /// `chunkShapeNoiseControl`.) A diverging argmax means the unkeyed fallback
    /// would serve mis-positioned, wrong tokens.
    nonisolated private static func unkeyedImageWholeContinuationMatches(
        context: ModelContext,
        vlm: Qwen35,
        input: LMInput
    ) throws -> (passed: Bool, detail: String, lines: [String]) {
        let tokens = LLMActor.extractTokenSequence(input.text.tokens)

        // Production-cold reference (argmax target): single-shot prepare over
        // the whole [0, last) prefix, image included, then a threaded forward.
        let (coldLogits, _) = try coldLastLogits(context: context, input: input)

        // Unkeyed-fallback shape: the whole prompt through the windowed
        // continuation on a fresh cache, anchored at zero — the first generated
        // token is sampled from the final position's logits, exactly as the
        // iterator's `.logits` branch does.
        let cache = context.model.newCache(parameters: nil)
        let wholeTokens = sliced2D(input.text.tokens, to: tokens.count)
        let wholeInput = LMInput(
            text: .init(tokens: wholeTokens, mask: ones(like: wholeTokens).asType(.int8)),
            image: input.image
        )
        guard
            case .logits(let out) = try vlm.prepareContinuation(
                wholeInput, cache: cache, state: nil, windowSize: 512
            )
        else {
            return (false, "prepareContinuation returned .tokens", [])
        }
        let lastPosition = out.logits[0..., -1, 0...]
        eval(lastPosition)

        let argmaxWarm = lastPosition.argMax().item(Int32.self)
        let argmaxCold = coldLogits.argMax().item(Int32.self)
        return (
            argmaxWarm == argmaxCold,
            "argmax wholeContinuation=\(argmaxWarm) cold=\(argmaxCold) (expected equal)",
            [
                "  prompt=\(tokens.count) tokens, one prepareContinuation over the whole image+text input from zero"
            ]
        )
    }

    // MARK: - Helpers — model invocation

    /// Cold reference: vendor `prepare` over everything but the last token
    /// (image included), then one single-token forward with the *threaded*
    /// state — the position-correct decode shape.
    nonisolated private static func coldLastLogits(
        context: ModelContext,
        input: LMInput
    ) throws -> (MLXArray, LMOutput.State?) {
        let tokens = LLMActor.extractTokenSequence(input.text.tokens)
        let cache = context.model.newCache(parameters: nil)
        let prefix = sliced2D(input.text.tokens, to: tokens.count - 1)
        let prefixInput = LMInput(
            text: .init(tokens: prefix, mask: ones(like: prefix).asType(.int8)),
            image: input.image
        )
        guard
            case .logits(let out) = try context.model.prepare(
                prefixInput, cache: cache, windowSize: nil
            )
        else {
            throw ParoQuantVLMSmokeError.unexpectedPrepareResult
        }
        eval(cache)
        return lastTokenLogits(
            model: context.model, token: tokens[tokens.count - 1], cache: cache, state: out.state
        )
    }

    /// Drive the production `PrefillExecutor` over 1D `tokens` (consume-all).
    @discardableResult
    nonisolated private static func prefill(
        context: ModelContext,
        tokens: [Int],
        checkpointBaseOffset: Int = 0,
        cache: [any KVCache],
        initialState: LMOutput.State? = nil
    ) throws -> PrefillExecutor.Output {
        let inputArr = MLXArray(tokens.map { Int32($0) })
        return try PrefillExecutor.run(
            model: context.model,
            text: .init(tokens: inputArr, mask: nil),
            cache: cache,
            checkpointBaseOffset: checkpointBaseOffset,
            prefillStepSize: 512,
            consumeAll: true,
            initialState: initialState
        )
    }

    nonisolated private static func lastTokenLogits(
        model: any LanguageModel,
        token: Int,
        cache: [any KVCache],
        state: LMOutput.State?
    ) -> (MLXArray, LMOutput.State?) {
        let input = LMInput.Text(
            tokens: MLXArray([Int32(token)]).expandedDimensions(axis: 0), mask: nil
        )
        let output = model(input, cache: cache, state: state)
        eval(output.logits)
        return (output.logits[0, 0], output.state)
    }

    // MARK: - Helpers — fixtures

    /// The agent-attachment prompt shape: image-bearing user turn after a
    /// short system prompt.
    nonisolated private static func prepareImageInput(
        context: ModelContext,
        image: CIImage
    ) async throws -> LMInput {
        let chat: [Chat.Message] = [
            .system("You are a meticulous visual analyst."),
            .user(
                "Describe the image: dominant colors, gradient direction, repeating structure.",
                images: [.ciImage(image)]
            ),
        ]
        return try await context.processor.prepare(input: UserInput(chat: chat))
    }

    // MARK: - Helpers — tensors

    /// Slice a token tensor (1D or 2D) to its first `count` positions,
    /// returned 2D ([1, count]).
    nonisolated private static func sliced2D(_ tokens: MLXArray, to count: Int) -> MLXArray {
        let sliced =
            tokens.ndim > 1 ? tokens[0..., ..<count] : tokens[..<count].expandedDimensions(axis: 0)
        return sliced
    }

    // MARK: - Reporting

    private func writeReport(checks: [CheckResult], allPassed: Bool) throws {
        try BenchmarkHarness.writeReport(
            checks: checks,
            allPassed: allPassed,
            modelName: runner.resolvedModelName,
            reportDir: reportDir,
            filePrefix: "spike"
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

enum ParoQuantVLMSmokeError: LocalizedError {
    case loadFailed(String)
    case engineNotReady
    case unexpectedPrepareResult
    case spikeChecksFailed([String])

    var errorDescription: String? {
        switch self {
        case .loadFailed(let detail):
            "ParoQuant VLM smoke: loadModel(visionMode: true) failed — \(detail)"
        case .engineNotReady:
            "ParoQuant VLM smoke: engine.isModelLoaded == false after loadModel"
        case .unexpectedPrepareResult:
            "ParoQuant VLM smoke: model.prepare returned .tokens — unexpected for the VLM container"
        case .spikeChecksFailed(let names):
            "Warmed-prefix spike failed checks: \(names.joined(separator: ", "))"
        }
    }
}
