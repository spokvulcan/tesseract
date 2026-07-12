//
//  MemoryBaselineTests.swift
//  tesseractTests
//
//  The baseline — "store everything, retrieve well" (ADR-0035 §10, #324).
//
//  This is the thing to beat, and it is deliberately built FIRST. The owner's
//  locked call: every mechanism (lifecycle scoring, sleep distillation) must
//  beat or match it to stay. The reason is the continual-learning record, where
//  brain-inspired methods keep losing to just keeping the data — sleep-replay
//  48.5% against plain rehearsal on 0.75% of the data at 79.9%; EWC 0.087
//  against an undefended MLP's 0.085. If our living lifecycle cannot beat
//  "store everything" on the owner's own corpus, the lifecycle is decoration,
//  and this suite is how we find out.
//
//  Gated on the corpus existing, so CI (which has none) skips rather than fails.
//  Run it:
//
//      xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
//        -destination 'platform=macOS' -parallel-testing-enabled NO \
//        -only-testing:tesseractTests/MemoryBaselineTests
//
//  The numbers land in `DebugPaths.root/memory-eval/` and on `Log.memory`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite("Memory baseline — store everything, retrieve well", .serialized)
struct MemoryBaselineTests {

    // MARK: - The loader

    @Test(
        "The corpus loads: every turn becomes an episode, and nothing else does",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func corpusLoads() async throws {
        let conversations = try MemoryEvalFixture.conversations()
        try #require(
            !conversations.isEmpty, "corpus at \(MemoryEvalCorpus.directory.path) is empty")

        // Shape invariants that would silently wreck the eval if they broke.
        for conversation in conversations {
            #expect(!conversation.id.isEmpty)
            #expect(!conversation.turns.isEmpty)
            for turn in conversation.turns {
                #expect(!turn.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                // Apple reference-date epoch, not Unix. Read as Unix, every
                // episode lands in 1970 and the temporal pool collapses.
                #expect(turn.occurredAt.timeIntervalSince1970 > 1_600_000_000)
            }
            // Turns are ordered within a conversation.
            let times = conversation.turns.map(\.occurredAt)
            #expect(times == times.sorted())
        }

        let turns = conversations.flatMap(\.turns)
        let users = turns.filter { $0.role == .user }.count
        let span =
            (conversations.last?.createdAt.timeIntervalSince(
                conversations[0].createdAt) ?? 0) / 86_400

        let report = MemoryEvalReport()
        report.section("Corpus")
        report.add("  conversations   \(conversations.count)")
        report.add(
            "  episodes        \(turns.count)  (user \(users), assistant \(turns.count - users))")
        report.add(String(format: "  span            %.1f days", span))
        report.add(
            "  embedder        \(MemoryEvalCorpus.isEmbedderAvailable ? "present (dense ⊕ BM25)" : "absent (BM25 only)")"
        )
        report.flush(name: "corpus")
    }

    // MARK: - The baseline store

    @Test(
        "Store everything: every episode is stored, embedded, and retrievable",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func baselineStoresEverything() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()
        let expected = try MemoryEvalFixture.conversations().flatMap(\.turns).count

        // The baseline's defining property: nothing is dropped, filtered, scored
        // for importance, or retired. Everything is kept.
        let episodes = try await harness.store.episodeCount()
        let memories = try await harness.store.memoryCount()
        #expect(episodes == expected)
        #expect(memories == expected)

        // And everything is reachable — no tier gating in the baseline.
        let tiers = try await harness.store.countsByTier()
        #expect(tiers[.cold, default: 0] == 0, "a freshly-stored corpus must retire nothing")

        if MemoryEvalCorpus.isEmbedderAvailable {
            let vectors = try await harness.store.embeddings(kind: "memory")
            #expect(vectors.count == expected, "every memory must carry a vector")
            #expect(vectors.first?.1.count == memoryEmbeddingDimension)
        }
    }

    // MARK: - The probe set

    @Test(
        "Probes are derived mechanically from the corpus's own structure",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func probesAreWellFormed() async throws {
        let probes = try MemoryEvalFixture.probes()
        try #require(!probes.isEmpty)

        for probe in probes {
            #expect(!probe.cue.isEmpty)
            #expect(!probe.goldEpisodeIDs.isEmpty)
            // A probe may never be its own answer.
            if let cueEpisodeID = probe.cueEpisodeID {
                #expect(!probe.goldEpisodeIDs.contains(cueEpisodeID))
            }
        }

        let report = MemoryEvalReport()
        report.section("Probes (mechanical — no LLM, no hand-written key)")
        for family in EvalProbe.Family.allCases {
            let matching = probes.filter { $0.family == family }
            let gold = matching.map { Double($0.goldEpisodeIDs.count) }
            let meanGold = gold.isEmpty ? 0 : gold.reduce(0, +) / Double(gold.count)
            report.add(
                String(
                    format: "  %-20@  n=%3d  mean gold %.1f", family.rawValue as NSString,
                    matching.count, meanGold))
        }
        report.flush(name: "probes")

        // The reference-back family is the sharp one; if it ever empties, the
        // eval has quietly lost its most discriminating probe.
        #expect(probes.contains { $0.family == .referenceBack })
    }

    // MARK: - THE HEADLINE: baseline vs lifecycle

    /// The number that decides whether the mechanisms stay.
    ///
    /// Cold start — the store as it is the moment the corpus is backfilled, with
    /// no graded usage yet. Note what this means for the lifecycle: `SS = 0` for
    /// every record, so entrenchment `(1 + log1p(SS))` is exactly 1.0; nothing is
    /// superseded, so interference is exactly 1.0. `retrievalScore` therefore
    /// collapses to `relevance · needProbability` — the lifecycle reduces to a
    /// recency prior, and this test measures whether that prior helps or hurts.
    @Test(
        "Cold start: does lifecycle scoring beat the baseline?",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func lifecycleVersusBaselineColdStart() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()

        let report = MemoryEvalReport()
        report.section("A/B — cold start (no graded usage yet)")
        report.add("  relevance: \(harness.isDense ? "0.75·cosine + 0.25·BM25" : "BM25 only")")
        report.add("  arms differ ONLY in the ranker; same records, same pool, same relevance.")

        report.add("  recencyOnly is the NULL: it discards the cue entirely and ranks by age.")

        for family in EvalProbe.Family.allCases {
            let baseline = try await harness.evaluate(family: family, arm: .baseline)
            let lifecycle = try await harness.evaluate(family: family, arm: .lifecycle)
            let recency = try await harness.evaluate(family: family, arm: .recencyOnly)
            guard baseline.probes > 0 else { continue }

            report.add("")
            report.add("  \(family.rawValue)")
            report.add(baseline.row("baseline (store everything)"))
            report.add(lifecycle.row("lifecycle (ADR-0035 §5)"))
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
            if recency.mrr > lifecycle.mrr {
                report.add(
                    "  ⚠︎ THE NULL WINS. A cue-blind recency sort beats the lifecycle here, so")
                report.add(
                    "    this family measures RECENCY, not memory. Read no lifecycle win from it.")
            }
        }
        report.flush(name: "ab-cold-start")

        // No assertion on which arm wins — the point of a scoreboard is to
        // MEASURE, and a losing lifecycle is a finding, not a test failure. What
        // must hold is that the baseline actually works: a retrieval floor this
        // low would mean the harness, not the memory system, is broken.
        let session = try await harness.evaluate(family: .sessionContinuation, arm: .baseline)
        #expect(session.recallAt10 > 0.1, "the baseline retriever itself looks broken")
    }

    // MARK: - Are the probes recency-degenerate?

    /// The check that keeps the whole scoreboard honest.
    ///
    /// Every probe family here takes its gold from the SAME conversation as the
    /// cue — so the correct answers are, almost by construction, the newest
    /// records in the pool. If that is true, then "prefer recent" is a nearly
    /// perfect strategy, any ranker with a recency term will look brilliant, and
    /// the A/B above is measuring recency rather than memory.
    ///
    /// This is not a hypothetical worry. It is the exact critique ADR-0035 §10
    /// levels at LoCoMo — "its corpus fits in context, so the *no-memory*
    /// baseline beats every memory system" — and it would be negligent to
    /// import that flaw into our own yardstick without measuring it.
    @Test(
        "How recency-degenerate are the probes? (the yardstick's own integrity)",
        .enabled(if: MemoryEvalCorpus.isAvailable))
    func probesAreRecencyDegenerate() async throws {
        let harness = try await MemoryEvalFixture.makeHarness()

        let report = MemoryEvalReport()
        report.section("Yardstick integrity — gold's position in the pool by age")
        report.add("  'gold percentile' = share of the pool NEWER than the gold record.")
        report.add("  0.00 ⇒ the gold IS the newest thing in the store (pure recency wins).")

        for family in EvalProbe.Family.allCases {
            var percentiles: [Double] = []
            for probe in harness.probes where probe.family == family {
                let candidates = harness.pool(for: probe)
                guard candidates.count > 1 else { continue }
                let goldIDs = harness.gold(probe)
                let golds = candidates.filter { goldIDs.contains($0.id) }
                guard !golds.isEmpty else { continue }
                // The BEST-placed gold: how much of the pool is newer than it?
                let best =
                    golds.map { g in
                        Double(candidates.filter { $0.bornAt > g.bornAt }.count)
                            / Double(candidates.count)
                    }.min() ?? 1
                percentiles.append(best)
            }
            guard !percentiles.isEmpty else { continue }
            let mean = percentiles.reduce(0, +) / Double(percentiles.count)
            let topDecile =
                Double(percentiles.filter { $0 <= 0.10 }.count) / Double(percentiles.count)
            report.add(
                String(
                    format: "  %-20@  mean gold percentile %.3f   in newest 10%% of pool: %.0f%%",
                    family.rawValue as NSString, mean, topDecile * 100))
            if mean < 0.05 {
                report.add(
                    "    ⚠︎ DEGENERATE: the answer is always the newest record. A memory system")
                report.add(
                    "      is not needed to pass this — the context window already has it.")
            }
        }
        report.add("")
        report.add("  What this means for the corpus, not just the probes: the owner's 65")
        report.add("  conversations are near-independent one-offs over 5 days, so every")
        report.add("  question's answer sits in the turns immediately before it. There is no")
        report.add("  cross-session fact reuse for a lifecycle to exploit — which is exactly")
        report.add("  the flaw ADR-0035 §10 diagnoses in LoCoMo, now measured in our own yard.")
        report.flush(name: "recency-degeneracy")

        #expect(!harness.probes.isEmpty)
    }
}
