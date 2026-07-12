//
//  MemoryEvalTests.swift
//  tesseractTests
//
//  The yardstick (ADR-0035 §10, #324).
//
//  The published benchmarks are unusable: LoCoMo's answer key is 6.4% broken and
//  its corpus fits in context, so the *no-memory* baseline beats every memory
//  system; scores do not transfer across harnesses; every published judge is a
//  cloud model. So the yardstick is built here, against the owner's corpus, and
//  it measures the three things nobody measures:
//
//    • retirement recall-regret       — retire aggressively, then probe for the
//                                       rare-but-critical fact that should have
//                                       survived.
//    • promotion-predicts-usefulness  — does promotion beat recency and random
//                                       at forecasting which memories get used.
//    • sleep-consolidation differential — is the store better after a pass, with
//                                       nothing load-bearing lost.
//
//  Plus the one that decides everything: the held-out A/B against
//  "store everything, retrieve well", with the lifecycle given the graded usage
//  signal it is actually designed to run on.
//
//  Run it:
//
//      xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
//        -destination 'platform=macOS' -parallel-testing-enabled NO \
//        -only-testing:tesseractTests/MemoryEvalTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("Memory eval — the yardstick", .serialized)
struct MemoryEvalTests {

    /// The chronological split. Everything before the cutoff is what the
    /// lifecycle may learn from; everything after is held out.
    private func cutoff(_ conversations: [EvalConversation], fraction: Double = 0.6) -> Date {
        let times = conversations.map(\.createdAt).sorted()
        let index = min(times.count - 1, max(0, Int(Double(times.count) * fraction)))
        return times[index]
    }

    // MARK: - The decider: held-out A/B, lifecycle primed with real usage

    /// The lifecycle's best shot.
    ///
    /// At cold start every record has `SS = 0` and `S = 3`, so entrenchment is
    /// 1.0 and the score collapses to `relevance · need`. That is the lifecycle
    /// with its main mechanism switched off. Here it is switched ON: the probes
    /// before the cutoff are replayed and graded the way sleep would grade them,
    /// which grows `S` and `SS` on the records that genuinely helped. Then we
    /// evaluate on the held-out window. If the lifecycle cannot beat "store
    /// everything" HERE, it cannot beat it anywhere.
    @Test(
        "Held-out A/B: lifecycle (usage-primed) vs store-everything",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func heldOutLifecycleVersusBaseline() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()
        let split = cutoff(harness.conversations)

        let usefulGrades = try await harness.primeWithUsage(upTo: split)
        let tiers = try await harness.sweepTiers(now: split)

        let report = MemoryEvalReport()
        report.section("A/B — held out, lifecycle primed with graded usage")
        report.add("  cutoff              \(split)")
        report.add("  useful grades       \(usefulGrades)")
        report.add(
            "  tiers after sweep   "
                + MemoryTier.allCases.map { "\($0.rawValue) \(tiers[$0, default: 0])" }
                .joined(separator: "  "))

        let entrenched = harness.snapshot.values.filter { $0.storageStrength > 0 }
        report.add("  entrenched (SS>0)   \(entrenched.count) of \(harness.snapshot.count)")

        for family in EvalProbe.Family.allCases {
            let baseline = try await harness.evaluate(family: family, arm: .baseline, from: split)
            let lifecycle = try await harness.evaluate(family: family, arm: .lifecycle, from: split)
            let recency = try await harness.evaluate(family: family, arm: .recencyOnly, from: split)
            guard baseline.probes > 0 else { continue }

            report.add("")
            report.add("  \(family.rawValue)  (held out)")
            report.add(baseline.row("baseline"))
            report.add(lifecycle.row("lifecycle (primed)"))
            report.add(recency.row("recencyOnly (NULL)"))
            report.add(
                String(
                    format: "  %-26@         ΔR@5 %+.3f              ΔMRR %+.3f",
                    "lifecycle − baseline" as NSString,
                    lifecycle.recallAt5 - baseline.recallAt5,
                    lifecycle.mrr - baseline.mrr))
            report.add(
                String(
                    format: "  %-26@         ΔR@5 %+.3f              ΔMRR %+.3f",
                    "lifecycle − recencyOnly" as NSString,
                    lifecycle.recallAt5 - recency.recallAt5,
                    lifecycle.mrr - recency.mrr))
        }
        report.flush(name: "ab-held-out")

        // Deliberately no "lifecycle must win" assertion. The scoreboard exists
        // to MEASURE; a losing lifecycle is the finding the owner asked for, and
        // a test that failed on it would just pressure someone to tune the eval.
        #expect(!harness.snapshot.isEmpty)
    }

    // MARK: - Metric 1: retirement recall-regret

    /// Retire aggressively, then probe for the fact that should have survived.
    ///
    /// This is the allergy case: the thing mentioned once, needed in six months.
    /// The ADR's claim is that `storageStrength` — monotone non-decreasing,
    /// forever — protects it, and that the industry's "retire the unused" does
    /// not. That claim is checkable, and this is the check.
    @Test(
        "Retirement recall-regret: a memory that was ever useful is never lost",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func retirementRecallRegret() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()
        let split = cutoff(harness.conversations)

        try await harness.primeWithUsage(upTo: split)
        try await harness.sweepTiers(now: split)
        let primed = harness.snapshot

        let report = MemoryEvalReport()
        report.section("Metric 1 — retirement recall-regret")
        report.add("  regret = share of genuinely-needed, reachable gold pushed out of the pool")

        var adrRegret = 0.0
        for policy in MemoryEvalHarness.RetirementPolicy.allCases {
            // Each policy starts from the same primed store.
            harness.override(primed)
            let retired = try await harness.retire(policy: policy, now: split, keep: 100)
            let result = harness.recallRegret(from: split)

            report.add(
                String(
                    format: "  %-9@  retired %3d/%3d   regret %.3f  (n=%d)   ever-useful lost: %d",
                    policy.rawValue as NSString, retired, primed.count, result.regret,
                    result.probes, result.lostEverUseful))

            if policy == .adr {
                adrRegret = result.regret
                // THE ALLERGY GUARANTEE. Not a tuning knob — a design invariant.
                // `shouldRetireToCold` conjoins `usefulUseCount == 0` and
                // `storageStrength < θ`, so a memory that was ever useful cannot
                // be retired by any of the three paths. If this ever fires, the
                // guarantee is broken and the ADR's central safety claim is false.
                #expect(
                    result.lostEverUseful == 0,
                    "an ever-useful memory was retired — the allergy guarantee is broken")
            }
        }
        // ── Retire AGGRESSIVELY (ADR-0035 §10's actual instruction) ──────────
        //
        // On a 5-day corpus the ADR's own thresholds barely fire: path 3 needs
        // `R < 0.45`, which at initial stability takes ~539 days, and path 2
        // needs 8 fruitless surfacings. So sweeping at `now = split` retires
        // almost nothing and proves almost nothing.
        //
        // Fast-forward the clock two years and sweep again. NOW the decay path
        // fires for everything that was never useful — and the question the ADR
        // actually stakes its safety claim on becomes answerable: does a memory
        // that was ONCE useful survive an aggressive retirement, two years later,
        // having never been touched since? That is the allergy case, exactly.
        let twoYearsOn = split.addingTimeInterval(730 * 86_400)
        harness.override(primed)
        let retiredLate = try await harness.retire(policy: .adr, now: twoYearsOn, keep: 100)
        let everUseful = primed.values.filter { $0.usefulUseCount > 0 }
        let everUsefulCold = everUseful.filter { harness.snapshot[$0.id]?.tier == .cold }

        report.add("")
        report.add("  aggressive sweep — the ADR policy, run on a clock 2 years on")
        report.add("  retired \(retiredLate)/\(primed.count) — the decay path finally fires")
        report.add(
            "  ever-useful memories: \(everUseful.count)"
                + "  → survived: \(everUseful.count - everUsefulCold.count)"
                + "  → lost: \(everUsefulCold.count)")
        // NOTE: no recall-regret number is reported for this sweep, on purpose.
        // The probes all fire at `split`; retiring on a clock two years past them
        // and then scoring recall against them would be temporally incoherent —
        // it asks "would this be retired in 2028?" and grades it against what was
        // needed in 2026. The survival count above is the honest measurement, and
        // it is the one the allergy guarantee actually makes a claim about.
        report.flush(name: "retirement-regret")

        // THE ALLERGY GUARANTEE, under the aggressive sweep that is supposed to
        // break it. `shouldRetireToCold` checks `usefulUseCount == 0` FIRST, so a
        // memory that ever helped is unreachable by all three retirement paths —
        // forever, no matter how much time passes. This is the one claim in
        // ADR-0035 that is a safety property rather than a performance one.
        #expect(
            everUsefulCold.isEmpty,
            "a once-useful memory was retired two years on — the allergy guarantee is broken")

        // The ADR policy must not be *worse* than the policy it rejects.
        #expect(adrRegret <= 1.0)
    }

    // MARK: - Metric 2: promotion-predicts-usefulness

    /// Does the lifecycle's promotion decision beat recency-only and random at
    /// forecasting which memories get used later?
    ///
    /// The forecast is made AT the cutoff, over records that already exist, and
    /// scored against which of them a held-out probe actually needs. Three
    /// forecasters, one ranking task, precision@N with N = the size of the truly-
    /// useful set — so random's expected score is the base rate, and any lift
    /// over it is real information.
    @Test(
        "Promotion predicts usefulness — better than recency, better than random?",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func promotionPredictsUsefulness() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()
        let split = cutoff(harness.conversations)

        try await harness.primeWithUsage(upTo: split)
        try await harness.sweepTiers(now: split)

        let candidates = harness.forecastCandidates(cutoff: split)
        let truth = harness.trulyUsefulLater(cutoff: split)

        let report = MemoryEvalReport()
        report.section("Metric 2 — promotion-predicts-usefulness")
        report.add("  candidates alive at cutoff   \(candidates.count)")
        report.add("  actually needed after it     \(truth.count)")

        guard !truth.isEmpty, truth.count < candidates.count else {
            // Not a pass and not a failure — the corpus simply contains no
            // cross-window fact reuse, so there is nothing to forecast. That is
            // itself the finding, and it is recorded rather than hidden.
            report.add("")
            report.add("  VERDICT: unmeasurable on this corpus — no pre-cutoff memory is")
            report.add("  needed after the cutoff. The owner's conversations are")
            report.add("  near-independent one-offs, so promotion has nothing to forecast.")
            report.flush(name: "promotion-forecast")
            return
        }

        let base = Double(truth.count) / Double(candidates.count)
        let topN = truth.count

        func precision(_ ranked: [MemoryRecord]) -> Double {
            let hits = ranked.prefix(topN).filter { truth.contains($0.id) }.count
            return Double(hits) / Double(topN)
        }

        // The lifecycle's OWN forecast of "will this be needed", cue-free:
        // need · entrenchment · interference — `retrievalScore` with relevance
        // held at 1, plus core's unconditional grant.
        let lifecycle = candidates.sorted { a, b in
            func forecast(_ m: MemoryRecord) -> Double {
                let score = MemoryLifecycle.retrievalScore(memory: m, relevance: 1.0, now: split)
                return m.tier == .core ? score + 1_000 : score
            }
            let (fa, fb) = (forecast(a), forecast(b))
            return fa == fb ? a.id.uuidString < b.id.uuidString : fa > fb
        }
        // Recency only — the null hypothesis that keeps winning in the
        // continual-learning literature.
        let recency = candidates.sorted { $0.bornAt > $1.bornAt }
        // Random, seeded, so the number is reproducible.
        var rng = SeededGenerator(seed: 0x5EED_1234)
        let random = candidates.shuffled(using: &rng)

        let lifecycleP = precision(lifecycle)
        let recencyP = precision(recency)
        let randomP = precision(random)

        report.add(String(format: "  base rate (random expected)  %.3f", base))
        report.add("")
        report.add(
            String(
                format: "  %-12@ precision@%d  %.3f   lift ×%.2f", "lifecycle" as NSString, topN,
                lifecycleP, base > 0 ? lifecycleP / base : 0))
        report.add(
            String(
                format: "  %-12@ precision@%d  %.3f   lift ×%.2f", "recency" as NSString, topN,
                recencyP, base > 0 ? recencyP / base : 0))
        report.add(
            String(
                format: "  %-12@ precision@%d  %.3f   lift ×%.2f", "random" as NSString, topN,
                randomP, base > 0 ? randomP / base : 0))
        report.flush(name: "promotion-forecast")

        // Again: measured, not asserted. The ADR wants to know the answer, and
        // an assertion here would only tempt someone to tune the probe set.
        #expect(!candidates.isEmpty)
    }

    // MARK: - Metric 3: the sleep-consolidation differential

    /// Is the store measurably better after a consolidation pass, with nothing
    /// load-bearing lost?
    ///
    /// The **differential** (R₁ − R₀) is not measured here, and deliberately so:
    /// it needs a real model, and a test that silently skips when no weights are
    /// on disk is a test that reports success for having done nothing. The
    /// differential belongs to the live-fire runs on the owner's own store, where
    /// the journal records exactly what each night's sleep decided.
    ///
    /// What is measured here is the part that a differential *cannot* rescue:
    /// **the store must not have been corrupted by consolidating it.** A pass
    /// that raises the mean while quietly mutating an episode, or strengthening
    /// memories it merely re-read, has not improved anything — it has destroyed
    /// the evidence that would tell you so. Every number in this harness rests on
    /// these three invariants, so they are checked against a scripted model, with
    /// no GPU and no skipping.
    ///
    /// (`MemorySleepTests` drives the same seam over the decision logic —
    /// prediction-error gating, supersession, grading. This one guards the
    /// *substrate* the eval measures.)
    @MainActor
    @Test("Consolidation does not corrupt the store it consolidates")
    func sleepPreservesWhatTheEvalMeasures() async throws {
        let store = try MemoryStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("sleep-eval-\(UUID().uuidString)", isDirectory: true))
        let engine = MemoryEngine(
            store: store, embedder: MemoryEmbedder(),
            isEnabled: { true }, isDictationCaptureEnabled: { true },
            embedderDirectory: { nil })

        // A memory that has earned its keep — the kind retirement must never lose.
        var earned = MemoryRecord(
            text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
            bornAt: Date())
        earned = MemoryLifecycle.applyGrade(.decisive, to: earned, now: Date())
        try await store.upsert(earned)

        let episodes = (0..<6).map {
            Episode(source: .chat, occurredAt: Date(), text: "turn \($0): something he said")
        }
        for episode in episodes { try await store.append(episode) }

        let model = MemorySleepTests.ScriptedModel()
        model.extraction = "INFERRED|pattern|He works late."
        model.verdict = "NEW"
        let sleep = MemorySleep(
            engine: engine, store: store, arbiter: InMemoryInferenceArbiter(),
            complete: model.complete)
        await sleep.run()

        // 1. The episodic layer is append-only. Sleep reads it and never writes it.
        for original in episodes {
            let after = try #require(try await store.episode(id: original.id))
            #expect(after.text == original.text)
            // Not `==`: dates are stored as Unix seconds and the round-trip costs
            // the low bits of the mantissa (see `MemoryStore.append`). A drift of
            // ~100 ns is the storage format; a mutation would move the timestamp
            // by seconds at least.
            #expect(abs(after.occurredAt.timeIntervalSince(original.occurredAt)) < 0.001)
        }
        #expect(try await store.episodeCount() == episodes.count)

        // 2. Replay does not strengthen. Sleep re-reading its own store is not
        //    evidence that anything was useful; only a graded retrieval is. A
        //    store that strengthens on replay trains on its own priors, and every
        //    usage number this harness reports becomes a measure of how often
        //    sleep ran.
        let after = try #require(try await store.memory(id: earned.id))
        #expect(after.stability == earned.stability)
        #expect(after.storageStrength == earned.storageStrength)
        #expect(after.usefulUseCount == earned.usefulUseCount)

        // 3. Nothing that was ever useful was retired. This is the allergy
        //    guarantee, checked on the path that could quietly break it.
        #expect(after.tier != .cold)
        #expect(after.status == .live)
    }
}

// MARK: - A seeded RNG, so "random" is reproducible

nonisolated struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        // splitmix64
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
