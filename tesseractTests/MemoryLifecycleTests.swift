//
//  MemoryLifecycleTests.swift
//  tesseractTests
//
//  The lifecycle math (ADR-0035 §3–§4, #320).
//
//  These tests pin the *claims the design makes*, not the arithmetic. Each one
//  is named after the property it defends, because the whole argument of map
//  #314 rests on them: if storage strength can decrease, or retrieval can
//  strengthen a memory, or a decisive memory can be retired, the design is
//  simply a different design.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory lifecycle")
struct MemoryLifecycleTests {

    private let day: TimeInterval = 86_400

    private func makeMemory(
        stability: Double = 3,
        storageStrength: Double = 0,
        usefulUseCount: Int = 0,
        bornDaysAgo: Double = 0,
        lastUsefulUseDaysAgo: Double? = nil,
        status: MemoryStatus = .live,
        tier: MemoryTier = .hot,
        supersededBy: UUID? = nil,
        now: Date = Date()
    ) -> MemoryRecord {
        let born = now.addingTimeInterval(-bornDaysAgo * day)
        return MemoryRecord(
            text: "I've noticed Bohdan starts with the hardest task.",
            kind: .belief,
            provenance: .inferred,
            status: status,
            tier: tier,
            bornAt: born,
            stability: stability,
            storageStrength: storageStrength,
            lastUsefulUseAt: now.addingTimeInterval(-(lastUsefulUseDaysAgo ?? bornDaysAgo) * day),
            usefulUseCount: usefulUseCount,
            supersededBy: supersededBy)
    }

    // MARK: - The decay curve

    @Test("Need-probability is 0.9 after exactly one stability period")
    func needProbabilityAtStability() {
        // FACTOR is defined so R(S, S) == 0.9. If this drifts, every threshold
        // in the design has silently moved.
        let r = MemoryLifecycle.needProbability(daysSinceUsefulUse: 10, stability: 10)
        #expect(abs(r - 0.9) < 0.001)
    }

    @Test("Need-probability decays, but as a power law — the tail stays heavy")
    func decayIsPowerLawNotExponential() {
        let s = 10.0
        let r1 = MemoryLifecycle.needProbability(daysSinceUsefulUse: 1, stability: s)
        let r100 = MemoryLifecycle.needProbability(daysSinceUsefulUse: 100, stability: s)
        let r1000 = MemoryLifecycle.needProbability(daysSinceUsefulUse: 1_000, stability: s)

        #expect(r1 > r100)
        #expect(r100 > r1000)

        // The point of a power law: decay *decelerates*. An exponential kernel
        // (MemoryBank's e^(−t/S) — the most-cited decay design in the agent
        // literature) would have annihilated this memory long before day 1000.
        #expect(r1000 > 0.2, "a power-law tail must not collapse to zero")

        // And the drop from 100→1000 days is smaller than 1→100.
        #expect((r1 - r100) > (r100 - r1000))
    }

    // MARK: - Storage strength: the deletion guard

    @Test("Storage strength NEVER decreases — not on any grade, ever")
    func storageStrengthIsMonotone() {
        let now = Date()
        var memory = makeMemory(stability: 20, storageStrength: 3, usefulUseCount: 2, now: now)
        let initial = memory.storageStrength

        for grade in [UseGrade.decisive, .used, .ignored, .harmful, .harmful, .ignored] {
            let before = memory.storageStrength
            memory = MemoryLifecycle.applyGrade(grade, to: memory, now: now)
            #expect(
                memory.storageStrength >= before,
                "grade \(grade.rawValue) decreased storage strength — it is the deletion guard, and Bjork's asymmetry says it cannot"
            )
        }
        #expect(memory.storageStrength >= initial)
    }

    @Test("A dormant memory revived gains far more stability than one used twice in an hour")
    func bjorkAsymmetry() {
        let now = Date()
        // Same memory, two histories: one just used, one long dormant.
        let fresh = makeMemory(stability: 30, lastUsefulUseDaysAgo: 0.04, now: now)  // ~1 hour
        let dormant = makeMemory(stability: 30, lastUsefulUseDaysAgo: 200, now: now)

        let freshAfter = MemoryLifecycle.applyGrade(.used, to: fresh, now: now)
        let dormantAfter = MemoryLifecycle.applyGrade(.used, to: dormant, now: now)

        let freshGain = freshAfter.stability - fresh.stability
        let dormantGain = dormantAfter.stability - dormant.stability

        #expect(
            dormantGain > freshGain * 2,
            "Bjork: gain scales inversely with current retrievability")
    }

    @Test("A decisive use is worth more than a merely-used one")
    func decisiveBeatsUsed() {
        let now = Date()
        let memory = makeMemory(stability: 10, lastUsefulUseDaysAgo: 5, now: now)
        let used = MemoryLifecycle.applyGrade(.used, to: memory, now: now)
        let decisive = MemoryLifecycle.applyGrade(.decisive, to: memory, now: now)
        #expect(decisive.stability > used.stability)
        #expect(decisive.storageStrength == used.storageStrength)  // ΔSS is grade-independent
    }

    // MARK: - Retrieved ≠ useful

    @Test("`.ignored` is NOT a lapse: it touches nothing")
    func ignoredIsNotALapse() {
        let now = Date()
        let memory = makeMemory(stability: 25, storageStrength: 2, usefulUseCount: 1, now: now)
        let after = MemoryLifecycle.applyGrade(.ignored, to: memory, now: now)

        // Retrieved-and-ignored indicts the *retriever*, not the memory.
        #expect(after.stability == memory.stability)
        #expect(after.storageStrength == memory.storageStrength)
        #expect(after.difficulty == memory.difficulty)
        #expect(after.usefulUseCount == memory.usefulUseCount)
        #expect(after.lastUsefulUseAt == memory.lastUsefulUseAt)
    }

    @Test("Being seen does not make a memory useful")
    func seenIsDiagnosticOnly() {
        let now = Date()
        let memory = makeMemory(now: now)
        let seen = MemoryLifecycle.markSeen(memory, now: now)

        #expect(seen.seenCount == 1)
        #expect(seen.usefulUseCount == 0, "retrieval must never count as use")
        #expect(seen.stability == memory.stability)
        #expect(seen.storageStrength == memory.storageStrength)
        #expect(seen.lastUsefulUseAt == memory.lastUsefulUseAt)
    }

    @Test("A confirmation costs nothing but a counter")
    func confirmationIsCheap() {
        let memory = makeMemory()
        let confirmed = MemoryLifecycle.confirm(memory)
        #expect(confirmed.confirmations == 1)
        #expect(confirmed.stability == memory.stability)
        #expect(confirmed.storageStrength == memory.storageStrength)
    }

    // MARK: - Harmful

    @Test("`.harmful` weakens stability but never resets it to zero (Jost's law)")
    func harmfulDoesNotAnnihilate() {
        let now = Date()
        let memory = makeMemory(stability: 60, storageStrength: 5, usefulUseCount: 4, now: now)
        let after = MemoryLifecycle.applyGrade(.harmful, to: memory, now: now)

        #expect(after.stability < memory.stability)
        #expect(after.stability > 0)
        #expect(
            after.storageStrength == memory.storageStrength,
            "SS is untouched by harm — only the owner deletes")
    }

    // MARK: - Retirement

    @Test("Retirement is by supersession, not by disuse — age is not consulted")
    func retirementIsBySupersession() {
        let now = Date()
        // Ancient, never used, but NOT superseded, and it was once decisive
        // (SS > 0) — the allergy case. It must survive.
        let allergy = makeMemory(
            stability: 5, storageStrength: 1.2, usefulUseCount: 1,
            bornDaysAgo: 400, lastUsefulUseDaysAgo: 380, now: now)
        #expect(
            !MemoryLifecycle.shouldRetireToCold(allergy, now: now),
            "a memory that was ever useful is protected by storage strength, forever")

        // Superseded yesterday, high stability, recently used — retires anyway.
        let stale = makeMemory(
            stability: 90, storageStrength: 8, usefulUseCount: 9,
            bornDaysAgo: 30, lastUsefulUseDaysAgo: 1,
            supersededBy: UUID(), now: now)
        #expect(
            MemoryLifecycle.shouldRetireToCold(stale, now: now),
            "supersession retires regardless of age, strength, or recency")
    }

    @Test("The never-useful path needs the full conjunction to fire")
    func neverUsefulRetirementNeedsAllThree() {
        let now = Date()
        // Never useful, no entrenchment, need finally decayed → retires.
        let noise = makeMemory(
            stability: 3, storageStrength: 0, usefulUseCount: 0,
            bornDaysAgo: 900, lastUsefulUseDaysAgo: 900, now: now)
        #expect(MemoryLifecycle.shouldRetireToCold(noise, now: now))

        // Same age, but it was useful once → protected by storage strength,
        // forever. This is the allergy guarantee.
        let usedOnce = makeMemory(
            stability: 3, storageStrength: 0.8, usefulUseCount: 1,
            bornDaysAgo: 900, lastUsefulUseDaysAgo: 900, now: now)
        #expect(!MemoryLifecycle.shouldRetireToCold(usedOnce, now: now))
    }

    @Test("A power-law tail never collapses — so a low absolute threshold would be dead code")
    func theTailNeverCollapses() {
        // This is the trap the design walked into: with DECAY = −0.1542,
        // need-probability at initial stability is still 0.34 after a DECADE,
        // and would take ~2.3 million years to reach 0.05. Any retirement rule
        // keyed to a small absolute R can therefore never fire. The threshold
        // is calibrated against this curve, not chosen round.
        let tenYears = MemoryLifecycle.needProbability(daysSinceUsefulUse: 3_650, stability: 3)
        #expect(tenYears > 0.3, "the heavy tail is the design; it must not be sanded off")
        #expect(
            MemoryLifecycle.coldNeedThreshold > tenYears,
            "the cold threshold must be reachable within a human lifetime")
    }

    @Test("Surfaced repeatedly and never once useful is evidence — and it retires the memory")
    func repeatedUselessnessIsEvidence() {
        let now = Date()
        // Young, so the time path cannot fire. But it has been put in front of
        // the model eight times and never once helped. "Never retrieved" is a
        // fact about the index and proves nothing; *this* is evidence about
        // the memory.
        var noise = makeMemory(
            stability: 3, storageStrength: 0, usefulUseCount: 0, bornDaysAgo: 5, now: now)
        noise.seenCount = MemoryLifecycle.coldIgnoredEvidenceCount
        #expect(MemoryLifecycle.shouldRetireToCold(noise, now: now))

        // Seen just as often, but it *did* help once → protected.
        var earned = noise
        earned.usefulUseCount = 1
        earned.storageStrength = 0.7
        #expect(!MemoryLifecycle.shouldRetireToCold(earned, now: now))
    }

    // MARK: - Promotion

    @Test("Promotion to core requires useful uses spread over distinct days (the spacing effect)")
    func promotionRequiresSpacing() {
        let now = Date()
        let strong = makeMemory(stability: 90, usefulUseCount: 5, now: now)

        // Three uses inside one conversation are one massed episode. They are
        // worth roughly one, and must not buy promotion.
        #expect(!MemoryLifecycle.shouldPromoteToCore(strong, distinctUsefulDays: 1))
        #expect(!MemoryLifecycle.shouldPromoteToCore(strong, distinctUsefulDays: 2))
        #expect(MemoryLifecycle.shouldPromoteToCore(strong, distinctUsefulDays: 3))
    }

    @Test("Promotion also requires real stability and repeat usefulness")
    func promotionRequiresStabilityAndCount() {
        let now = Date()
        let young = makeMemory(stability: 10, usefulUseCount: 5, now: now)
        #expect(!MemoryLifecycle.shouldPromoteToCore(young, distinctUsefulDays: 9))

        let rarelyUseful = makeMemory(stability: 90, usefulUseCount: 1, now: now)
        #expect(!MemoryLifecycle.shouldPromoteToCore(rarelyUseful, distinctUsefulDays: 9))
    }

    // MARK: - The retrieval score

    @Test("The score is multiplicative: a superseded memory cannot be rescued by entrenchment")
    func supersededIsSuppressed() {
        let now = Date()
        let live = makeMemory(stability: 30, storageStrength: 5, now: now)
        var dead = live
        dead.status = .superseded
        dead.supersededBy = UUID()

        let liveScore = MemoryLifecycle.retrievalScore(memory: live, relevance: 0.9, now: now)
        let deadScore = MemoryLifecycle.retrievalScore(memory: dead, relevance: 0.9, now: now)

        #expect(deadScore < liveScore / 5)
    }

    @Test("Entrenchment lifts an old memory, but relevance still dominates")
    func entrenchmentDoesNotOverrideRelevance() {
        let now = Date()
        let entrenched = makeMemory(
            stability: 100, storageStrength: 20, lastUsefulUseDaysAgo: 30, now: now)
        let fresh = makeMemory(stability: 3, storageStrength: 0, now: now)

        // Irrelevant-but-entrenched must lose to relevant-but-new.
        let irrelevant = MemoryLifecycle.retrievalScore(
            memory: entrenched, relevance: 0.05, now: now)
        let relevant = MemoryLifecycle.retrievalScore(memory: fresh, relevance: 0.95, now: now)
        #expect(relevant > irrelevant)
    }

    // MARK: - ε-exploration

    @Test("Exploration always claims a slot once the budget can afford one")
    func explorationNeverRoundsToZero() {
        // The cold tail can only come back if something draws from it.
        #expect(MemoryLifecycle.explorationSlots(of: 8) >= 1)
        #expect(MemoryLifecycle.explorationSlots(of: 20) >= 1)
        #expect(MemoryLifecycle.explorationSlots(of: 4) >= 1)
        // Tiny budgets spend everything on exploitation.
        #expect(MemoryLifecycle.explorationSlots(of: 3) == 0)
    }

    // MARK: - The tier sweep

    @Test("The sweep promotes, demotes, and retires in one pass")
    func sweepMovesTiers() {
        let now = Date()

        let promoted = MemoryLifecycle.sweepTier(
            makeMemory(stability: 90, usefulUseCount: 4, now: now),
            distinctUsefulDays: 4, now: now)
        #expect(promoted.tier == .core)

        let retired = MemoryLifecycle.sweepTier(
            makeMemory(stability: 60, supersededBy: UUID(), now: now),
            distinctUsefulDays: 0, now: now)
        #expect(retired.tier == .cold)

        // Fading but not gone: hot → warm as need-probability crosses 0.5.
        let fading = MemoryLifecycle.sweepTier(
            makeMemory(
                stability: 2, storageStrength: 1, usefulUseCount: 1,
                lastUsefulUseDaysAgo: 300, now: now),
            distinctUsefulDays: 1, now: now)
        #expect(fading.tier == .warm, "still reachable — demoted, not deleted")
    }

    @Test("Consolidation priority favours memories whose need is falling fastest")
    func consolidationPriorityTracksTheSlope() {
        let now = Date()
        // Steep part of the curve — recently used, low stability.
        let slipping = makeMemory(stability: 2, lastUsefulUseDaysAgo: 2, now: now)
        // Flat part — long dormant, so its need has already bottomed out.
        let settled = makeMemory(stability: 2, lastUsefulUseDaysAgo: 900, now: now)

        #expect(
            MemoryLifecycle.consolidationPriority(slipping, now: now)
                > MemoryLifecycle.consolidationPriority(settled, now: now))
    }

    // MARK: - Tier ordering

    @Test("Bigger tier means more important — every caller reads it that way")
    func tiersOrderByImportance() {
        // This is not pedantry. When this was inverted, sleep counted a retirement
        // as a promotion and told the owner a memory it had just pushed out of the
        // pool was "always present now" — and the injected block led with the cold
        // tail instead of the core beliefs. One `switch` decided both.
        #expect(MemoryTier.core > MemoryTier.hot)
        #expect(MemoryTier.hot > MemoryTier.warm)
        #expect(MemoryTier.warm > MemoryTier.cold)
        #expect(MemoryTier.allCases.max() == .core)
        #expect(MemoryTier.allCases.min() == .cold)
    }
}
