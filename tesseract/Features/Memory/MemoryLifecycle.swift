//
//  MemoryLifecycle.swift
//  tesseract
//
//  The living lifecycle: promotion, decay, retirement (ADR-0035 §3–§4, #320).
//
//  Pure math over value types — no store, no model, no clock of its own.
//  Every function takes `now` explicitly so the whole lifecycle is
//  deterministically testable.
//
//  The spine is ACT-R base-level activation, which two research tickets
//  reached independently from different literatures (#316 from the cognitive
//  architectures, #317 from Anderson & Schooler's rational analysis, whose
//  log-odds *is* base-level activation). It is shipped here in FSRS-6's
//  power-law parameterization, because FSRS is the same idea with 20 years of
//  fitting behind it and a local, vendorable implementation.
//
//  The one thing to hold on to: **R is need-probability, not recall
//  probability.** Our reads never fail — a memory in SQLite is always there.
//  What decays is the *probability that this memory is needed now*, which is
//  the quantity Anderson & Schooler showed the environment actually obeys.
//

import Foundation

// The equations below are transcribed from ACT-R and FSRS-6, where the
// quantities are named `S` (stability), `R` (retrievability/need) and `D`
// (difficulty). Renaming them to satisfy a linter would make the code harder
// to check against the papers it came from, which is the only way anyone can
// ever verify it. Same call, same reason, as `EvictionPolicy.swift`.
// swiftlint:disable identifier_name

nonisolated enum MemoryLifecycle: Sendable {

    // MARK: - Curve constants

    /// FSRS-6's fitted decay exponent. Our ONE tunable curve knob.
    /// Power-law, not exponential — the heavy tail is the psychologically
    /// correct shape (Ebbinghaus/Wickelgren), and it is what lets an
    /// old-but-once-important fact survive.
    static let decay: Double = -0.1542

    /// Defined so that `R(S, S) = 0.9` — i.e. after `S` days, need-probability
    /// has fallen to 0.9.
    static let factor: Double = pow(0.9, 1.0 / decay) - 1.0

    /// A fact just learned is worth remembering for a few days before it must
    /// prove itself.
    static let initialStability: Double = 3.0
    static let initialDifficulty: Double = 5.0

    // FSRS-6 fitted shape constants. A **prior on the shape**, not scripture:
    // in FSRS's own benchmark a zero-parameter moving average beats FSRS-6
    // (log loss 0.337 vs 0.346) and a small learned sequence model crushes
    // both (0.277). Re-fit on our own graded-event log once ~10⁴ events exist
    // (ADR-0035, Consequences).
    static let w6: Double = 3.0194
    static let w8: Double = 1.8722
    static let w9: Double = 0.1666
    static let w10: Double = 0.796
    static let w11: Double = 1.4835
    static let w12: Double = 0.0614
    static let w13: Double = 0.2629
    static let w14: Double = 1.6483
    /// FSRS w16 — the "easy bonus". A decisive use is worth more than a
    /// merely-used one.
    static let decisiveBonus: Double = 1.87

    // MARK: - Need probability

    /// `R(t, S) = (1 + FACTOR · t/S)^DECAY`
    ///
    /// A prior on **need**, not on recall. Falls off as a power law, so it
    /// decelerates: the fact you last needed a year ago is not much less
    /// likely to be needed today than the one from eleven months ago. That
    /// heavy tail is the whole point — it is what an exponential kernel
    /// (MemoryBank's `e^(−t/S)`, the most-cited decay design in the agent
    /// literature) gets wrong.
    static func needProbability(daysSinceUsefulUse t: Double, stability S: Double) -> Double {
        let t = max(0, t)
        let S = max(0.01, S)
        return pow(1.0 + factor * t / S, decay)
    }

    /// Convenience over a memory and a clock.
    static func needProbability(of memory: MemoryRecord, now: Date) -> Double {
        needProbability(
            daysSinceUsefulUse: days(from: memory.lastUsefulUseAt, to: now),
            stability: memory.stability
        )
    }

    static func days(from: Date, to: Date) -> Double {
        max(0, to.timeIntervalSince(from) / 86_400)
    }

    // MARK: - The retrieval score

    /// `score = relevance · needProbability · entrenchment · interference`
    ///
    /// **Multiplicative**, because the terms are probabilities of independent
    /// failure modes: a superseded or irrelevant memory cannot be rescued by
    /// sheer age or entrenchment. (Generative Agents used an additive form
    /// with all weights 1; multiplicative is the better default here.)
    static func retrievalScore(memory: MemoryRecord, relevance: Double, now: Date) -> Double {
        let need = needProbability(of: memory, now: now)
        let entrenchment = 1.0 + log1p(max(0, memory.storageStrength))
        let interference = (memory.supersededBy == nil && memory.status != .superseded) ? 1.0 : 0.1
        return relevance * need * entrenchment * interference
    }

    // MARK: - The update rule

    /// Apply a graded outcome to a memory.
    ///
    /// The grade comes from the **sleep judge**, not from "was it retrieved".
    /// This is the single most important boundary in the lifecycle: a system
    /// that strengthens on retrieval trains on its own retriever's beliefs and
    /// never learns anything.
    ///
    /// Note what `.ignored` does *not* do: it does not touch S, SS, or D.
    /// Retrieved-and-ignored is not a lapse — it is a retriever miss, and the
    /// memory is not at fault.
    static func applyGrade(_ grade: UseGrade, to memory: MemoryRecord, now: Date) -> MemoryRecord {
        var m = memory
        let R = needProbability(of: memory, now: now)

        switch grade {
        case .decisive, .used:
            let bonus = grade == .decisive ? decisiveBonus : 1.0
            // Bjork's asymmetry, verbatim from FSRS-6: the gain is
            // proportional to (e^{(1−R)·w} − 1), so a memory revived after
            // long dormancy gains far more than one used twice in an hour.
            let gain =
                exp(w8) * (11.0 - m.difficulty) * pow(max(0.01, m.stability), -w9)
                * (exp((1.0 - R) * w10) - 1.0) * bonus
            m.stability *= (1.0 + gain)

            // MONOTONE NON-DECREASING, forever. ΔSS is a decreasing function
            // of R — the deletion guard grows most when the memory was least
            // expected to be needed.
            m.storageStrength += (1.0 - R)

            let g = Double(grade == .decisive ? 4 : 3)
            m.difficulty = clamp(m.difficulty - w6 * (g - 3.0) * (10.0 - m.difficulty) / 9.0, 1, 10)

            m.lastUsefulUseAt = now
            m.usefulUseCount += 1

        case .ignored:
            // NOT a lapse. The lifecycle is untouched; only the per-cue
            // affinity (held in the store, not here) decays.
            break

        case .harmful:
            // Stale or contradicted — interference, not decay. Note it does
            // NOT reset to zero (Jost's law), and SS is untouched, ever.
            let post =
                w11 * pow(m.difficulty, -w12) * (pow(m.stability + 1.0, w13) - 1.0)
                * exp((1.0 - R) * w14)
            m.stability = min(post, m.stability / 1.05)
            m.difficulty = clamp(m.difficulty + w6 * (10.0 - m.difficulty) / 9.0, 1, 10)
        }

        m.stability = max(0.01, m.stability)
        return m
    }

    /// A re-encounter with no surprise. The cheap path — the rewriter is never
    /// invoked, and this is the biggest compute saving on the write path.
    static func confirm(_ memory: MemoryRecord) -> MemoryRecord {
        var m = memory
        m.confirmations += 1
        return m
    }

    /// Retrieved into context. **Diagnostic only** — this must never feed the
    /// lifecycle, or "retrieved" silently becomes "useful".
    static func markSeen(_ memory: MemoryRecord, now: Date) -> MemoryRecord {
        var m = memory
        m.seenCount += 1
        m.lastSeenAt = now
        return m
    }

    // MARK: - Promotion

    /// Promotion to `.core` requires useful uses spread over **≥ 3 distinct
    /// days**. This is the spacing effect, and it matters: three uses inside
    /// one conversation are one massed episode and are worth roughly one.
    static let corePromotionStability: Double = 60
    static let corePromotionUsefulUses: Int = 3
    static let corePromotionDistinctDays: Int = 3

    /// What promotion concretely grants: unconditional presence in the
    /// injected block, every conversation. A core memory has stopped being a
    /// retrieval and become identity.
    static func shouldPromoteToCore(
        _ memory: MemoryRecord, distinctUsefulDays: Int
    ) -> Bool {
        guard memory.status == .live, memory.tier != .core else { return false }
        return memory.stability > corePromotionStability
            && memory.usefulUseCount >= corePromotionUsefulUses
            && distinctUsefulDays >= corePromotionDistinctDays
    }

    // MARK: - Retirement

    /// The need-probability below which a never-useful memory leaves the
    /// default retrieval pool.
    ///
    /// **Calibrated against the curve, not picked round.** A power-law tail
    /// never collapses: with `DECAY = −0.1542`, need-probability at initial
    /// stability takes ~2.3 *million* years to reach 0.05, and still sits at
    /// 0.34 after a decade. Any low absolute threshold is therefore dead code —
    /// a trap this design walked into and a test caught. 0.45 is reached after
    /// roughly 18 months of a memory never once being useful, which is the
    /// intended pace: slow, because disuse is not evidence of uselessness.
    static let coldNeedThreshold: Double = 0.45
    /// The storage-strength floor. A memory that has *ever* been decisive has
    /// SS > 0 and is protected by the conjunction below — forever.
    static let coldStorageStrengthThreshold: Double = 0.5
    /// How many times a memory may be surfaced and graded useless before that
    /// counts as *evidence* about the memory rather than about the retriever.
    static let coldIgnoredEvidenceCount: Int = 8

    /// Retirement is **demotion, never deletion** (ADR-0035 §4).
    ///
    /// Three paths, and the *primary* one is interference — not disuse:
    ///
    ///  1. **Superseded.** Age is not consulted. This is how memories actually
    ///     retire: something newer replaced them.
    ///  2. **Repeatedly surfaced and never once useful.** Note the difference
    ///     from the Law of Disuse: *"never retrieved"* is a fact about the
    ///     index and is no evidence at all, but *"retrieved eight times and it
    ///     never once helped"* is evidence about the memory. This is the path
    ///     that actually clears extraction noise.
    ///  3. **Never useful, and its need-probability has finally decayed.** The
    ///     slow path — see `coldNeedThreshold` on why it is slow by design.
    ///
    /// Paths 2 and 3 both require `usefulUseCount == 0` *and* no entrenchment.
    /// The allergy mentioned once and acted on has `SS > 0` and outlives all
    /// of them.
    static func shouldRetireToCold(_ memory: MemoryRecord, now: Date) -> Bool {
        if memory.tier == .cold { return false }
        if memory.supersededBy != nil || memory.status == .superseded { return true }

        guard memory.usefulUseCount == 0 else { return false }
        guard memory.storageStrength < coldStorageStrengthThreshold else { return false }

        if memory.seenCount >= coldIgnoredEvidenceCount { return true }
        return needProbability(of: memory, now: now) < coldNeedThreshold
    }

    /// The full tier sweep. Runs in sleep, never on the read path.
    ///
    /// `distinctUsefulDays` is supplied by the store (it needs the retrieval
    /// log to compute), which is why this is a parameter and not a field.
    static func sweepTier(_ memory: MemoryRecord, distinctUsefulDays: Int, now: Date)
        -> MemoryRecord
    {
        var m = memory
        if shouldRetireToCold(m, now: now) {
            m.tier = .cold
            return m
        }
        if shouldPromoteToCore(m, distinctUsefulDays: distinctUsefulDays) {
            m.tier = .core
            return m
        }
        // Between the two poles, need-probability sorts hot from warm. The
        // boundary sits above `coldNeedThreshold` so the ladder is monotone:
        // hot ≥ 0.6 > warm ≥ 0.45 > cold.
        if m.tier != .core {
            let need = needProbability(of: m, now: now)
            m.tier = need >= warmNeedThreshold ? .hot : .warm
        }
        return m
    }

    /// Above this, a memory is hot (roughly: usefully used within the last
    /// ~3 months at initial stability).
    static let warmNeedThreshold: Double = 0.6

    // MARK: - The sleep priority queue

    /// Rank consolidation candidates by how fast their need-probability is
    /// *falling* — `R(t−1) − R(t)`, the memories on the edge of slipping —
    /// scaled by entrenchment.
    ///
    /// Inherits both of prioritized-experience-replay's post-mortem fixes:
    /// priorities are **recomputed**, never cached (PER's "outdated
    /// priorities"), and the caller must mix in a uniform floor (PER's
    /// "insufficient coverage of the sample space") so the cold tail is ever
    /// visited at all.
    static func consolidationPriority(_ memory: MemoryRecord, now: Date) -> Double {
        let t = days(from: memory.lastUsefulUseAt, to: now)
        let rNow = needProbability(daysSinceUsefulUse: t, stability: memory.stability)
        let rPrev = needProbability(
            daysSinceUsefulUse: max(0, t - 1), stability: memory.stability)
        let slope = max(0, rPrev - rNow)
        return slope * (1.0 + log1p(max(0, memory.storageStrength)))
    }

    // MARK: - ε-exploration

    /// ~1 retrieval slot in 20 is drawn from `.warm`/`.cold`.
    ///
    /// **Not optional.** The counterfactual — "which memories would have been
    /// useful had we surfaced them?" — is otherwise unobservable, so without
    /// this the lifecycle trains on its own priors forever and nothing in the
    /// cold tier can ever come back. It is the only thing standing between us
    /// and a memory system that confidently forgets whatever it once decided
    /// to forget.
    static let explorationRate: Double = 0.05

    /// How many of `slots` should be exploration draws. Always ≥ 1 when the
    /// budget allows, so exploration cannot round to zero on small reads.
    static func explorationSlots(of slots: Int) -> Int {
        guard slots >= 4 else { return 0 }
        return max(1, Int((Double(slots) * explorationRate).rounded()))
    }

    // MARK: - Helpers

    private static func clamp(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
        min(max(x, lo), hi)
    }
}

// swiftlint:enable identifier_name
