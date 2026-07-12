//
//  MemorySleepTests.swift
//  tesseractTests
//
//  Consolidation, driven by a **scripted model** (ADR-0035 §7).
//
//  `MemorySleep` takes its model as an injected closure, which means a whole
//  sleep pass — grade, extract, reconcile, sweep — can be run to completion with
//  no GPU and no weights, and its decisions asserted exactly. That is the only
//  way to test the part of the system that matters: not "did it call the model",
//  but "given what the model said, what did it *do* to the store".
//
//  Three invariants here would each, on their own, invalidate every number the
//  eval harness produces:
//
//    - the episodic layer is append-only, and sleep never touches it;
//    - consolidation never increments stability or storage strength — a store
//      that strengthens its own memories on replay is a self-licking ice cream
//      cone, and its usage statistics mean nothing;
//    - no prediction error, no rewrite.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory sleep")
@MainActor
struct MemorySleepTests {

    /// A model that says exactly what the test tells it to, chosen by what it is
    /// being asked. Sleep's four prompts are distinguishable by a phrase apiece,
    /// which is all the routing this needs.
    final class ScriptedModel: @unchecked Sendable {
        var extraction = "NOTHING"
        var verdict = "NEW"
        var grades = ""
        var reread = "NOTHING"
        private(set) var prompts: [String] = []

        var complete: @Sendable (String) async throws -> String {
            { [self] prompt in
                await MainActor.run { self.prompts.append(prompt) }
                if prompt.contains("grading your own memory") { return grades }
                if prompt.contains("whether it is news") { return verdict }
                if prompt.contains("memories is WRONG") { return reread }
                return extraction
            }
        }
    }

    private func makeStore() throws -> MemoryStore {
        try MemoryStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("sleep-\(UUID().uuidString)", isDirectory: true))
    }

    private func makeEngine(_ store: MemoryStore) -> MemoryEngine {
        MemoryEngine(
            store: store, embedder: MemoryEmbedder(),
            isEnabled: { true }, isDictationCaptureEnabled: { true },
            embedderDirectory: { nil })
    }

    private func makeSleep(
        _ engine: MemoryEngine, _ store: MemoryStore, _ model: ScriptedModel
    ) -> MemorySleep {
        MemorySleep(
            engine: engine, store: store, arbiter: InMemoryInferenceArbiter(),
            complete: model.complete)
    }

    // MARK: - Extraction

    @Test("Sleep distils episodes into first-person beliefs, with provenance")
    func sleepDistilsEpisodes() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = """
            STATED|belief|He is allergic to shellfish.
            INFERRED|pattern|He works in long focused blocks late at night.
            """

        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "remember I can't eat shellfish"))
        await makeSleep(engine, store, model).run()

        let memories = try await store.allLiveMemories()
        #expect(memories.count == 2)

        let stated = try #require(memories.first { $0.text.contains("shellfish") })
        #expect(stated.provenance == .stated)
        #expect(stated.kind == .belief)

        let inferred = try #require(memories.first { $0.text.contains("focused blocks") })
        // The distinction the whole safety story rests on: what he said versus
        // what I concluded. A guess recorded as testimony is the one error that
        // can never be undone from the inside.
        #expect(inferred.provenance == .inferred)
        #expect(inferred.kind == .pattern)

        // Consumed — the next sleep does not re-read them.
        #expect(try await store.unconsolidatedEpisodes().isEmpty)
    }

    @Test("An episode sleep has read is never altered — the lower layer is immutable")
    func sleepNeverMutatesEpisodes() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "STATED|belief|He is allergic to shellfish."

        let episode = Episode(
            source: .chat, occurredAt: Date(), text: "remember I can't eat shellfish")
        try await store.append(episode)

        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.episode(id: episode.id))
        // Verbatim, forever. The episodic layer is the only thing that can ever
        // correct a belief that has drifted, so nothing may rewrite it — not
        // even the process whose whole job is rewriting.
        #expect(after.text == episode.text)
        #expect(after.source == episode.source)
        // Not `==`: the Unix-seconds round-trip costs sub-microsecond precision
        // (see `MemoryStore.append`). This passed by luck of the low bits until
        // its sibling in `MemoryEvalTests` did not.
        #expect(abs(after.occurredAt.timeIntervalSince(episode.occurredAt)) < 0.001)
    }

    @Test("A model that has nothing to say produces no memories")
    func nothingIsAGoodAnswer() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "NOTHING"

        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "what's 2+2"))
        await makeSleep(engine, store, model).run()

        // Most turns are not worth remembering. A consolidation that finds
        // something profound in "what's 2+2" is a consolidation that will fill
        // the store with noise.
        #expect(try await store.memoryCount() == 0)
        #expect(try await store.unconsolidatedEpisodes().isEmpty)
    }

    // MARK: - The prediction-error gate

    @Test("No prediction error, no rewrite — a known claim confirms, it does not re-add")
    func sameClaimConfirmsAndNeverRewrites() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let existing = MemoryRecord(
            text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
            bornAt: Date())
        try await store.upsert(existing)

        model.extraction = "STATED|belief|He cannot eat shellfish."
        model.verdict = "SAME 1"
        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "no shellfish for me"))

        await makeSleep(engine, store, model).run()

        let memories = try await store.allLiveMemories()
        // The store did NOT grow. This is the whole gate: a system whose rewriter
        // fires on every observation will drift into fiction on rephrasing alone,
        // each rewrite one plausible step from the last.
        #expect(memories.count == 1)
        let after = try #require(memories.first)
        #expect(after.id == existing.id)
        #expect(after.text == "He is allergic to shellfish.", "the wording is untouched")
        #expect(after.confirmations == existing.confirmations + 1)

        // And it is written down. The *absence* of a rewrite is the whole design,
        // so it had better be something the owner can watch happen rather than
        // something he has to take on faith.
        let journal = try await store.journal(limit: 20)
        let confirmation = try #require(journal.first { $0.mutation == .confirmed })
        #expect(confirmation.memoryID == existing.id)
    }

    @Test("A step up is journalled as a promotion — and does not claim core's standing")
    func aStepUpIsAPromotion() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        // Warm, fresh, and never useless — the sweep lifts it back to hot. That is
        // a promotion, and it must be *called* one. But it is nowhere near core
        // (which wants 60 days of stability and useful uses on three separate
        // days), so the journal may not promise core's standing — "always present
        // now" is what core concretely grants, and a hot memory has not earned it.
        var warm = MemoryRecord(
            text: "He prefers automatic defaults over manual configuration.",
            kind: .belief, provenance: .inferred, bornAt: Date())
        warm.tier = .warm
        try await store.upsert(warm)

        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: warm.id))
        #expect(after.tier == .hot)

        let entry = try #require(
            (try await store.journal(limit: 20)).first { $0.mutation == .promoted })
        #expect(entry.memoryID == warm.id)
        #expect(entry.detail.contains("hot"))
        #expect(
            !entry.detail.contains("Always present"),
            "only core is unconditionally present — do not promise it of a hot memory")
    }

    @Test("A retirement is journalled as a demotion — never as a promotion")
    func aRetirementIsNeverCalledAPromotion() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        // Surfaced eight times, never once useful: retirement path two. It goes
        // cold — and the owner must be *told* it went cold. The live store had
        // four journal rows announcing "always present now" about memories that
        // had in fact just been moved out of the default pool, because the tier
        // comparison ran backwards. Never again.
        var noise = MemoryRecord(
            text: "He says 'Continue, please' a lot.",
            kind: .pattern, provenance: .inferred, bornAt: Date())
        noise.seenCount = 9
        noise.usefulUseCount = 0
        noise.storageStrength = 0
        try await store.upsert(noise)

        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: noise.id))
        #expect(after.tier == .cold, "shown nine times, never once useful")

        let journal = try await store.journal(limit: 20)
        let entry = try #require(journal.first { $0.memoryID == noise.id && $0.mutation != .added })
        #expect(entry.mutation == .demoted)
        #expect(journal.allSatisfy { $0.mutation != .promoted }, "nothing here was promoted")
        #expect(!entry.detail.contains("Always present"))
    }

    @Test("A contradiction supersedes — and supersession is not deletion")
    func contradictionSupersedes() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let old = MemoryRecord(
            text: "He lives in Kyiv.", kind: .belief, provenance: .stated, bornAt: Date())
        try await store.upsert(old)

        model.extraction = "STATED|belief|He lives in Lisbon."
        model.verdict = "REPLACES 1"
        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "I moved to Lisbon last month"))

        await makeSleep(engine, store, model).run()

        let superseded = try #require(try await store.memory(id: old.id))
        // Still there. What I used to believe is evidence about the past, and
        // about me — deleting it would erase the only record that I changed my
        // mind.
        #expect(superseded.status == .superseded)
        #expect(superseded.text == "He lives in Kyiv.")

        let successor = try #require(
            try await store.allLiveMemories().first { $0.text.contains("Lisbon") })
        #expect(superseded.supersededBy == successor.id)
        // The successor inherits what the old belief earned: it is its
        // continuation, not a stranger.
        #expect(successor.storageStrength >= superseded.storageStrength)
    }

    // MARK: - The owner's veto

    /// The whole point of two layers. The episode is testimony and cannot be
    /// wrong; the belief drawn from it is inference and can be. Contest sends the
    /// inference back to the testimony.
    private func contestedMemory(
        _ text: String, from episodeText: String, in store: MemoryStore, engine: MemoryEngine
    ) async throws -> (MemoryRecord, Episode) {
        let episode = Episode(source: .chat, occurredAt: Date(), text: episodeText)
        try await store.append(episode)
        try await store.markConsolidated([episode.id], at: Date())

        let memory = MemoryRecord(
            text: text, kind: .belief, provenance: .inferred,
            sourceEpisodeIDs: [episode.id], bornAt: Date())
        try await store.upsert(memory)
        await engine.contest(memory)

        let contested = try #require(try await store.memory(id: memory.id))
        #expect(contested.status == .contested)
        return (contested, episode)
    }

    @Test("Contesting sends the belief back to what he actually said — and the re-read wins")
    func contestingRereadsTheSources() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let (contested, _) = try await contestedMemory(
            "He goes running every morning.",
            from: "did my 100 sit-ups and 50 push-ups before breakfast again",
            in: store, engine: engine)
        // It had earned something while it was wrong.
        var strengthened = contested
        strengthened.storageStrength = 3.0
        strengthened.confirmations = 4
        try await store.upsert(strengthened)

        model.reread = "STATED|belief|He does 100 sit-ups and 50 push-ups before breakfast."
        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: contested.id))
        #expect(after.status == .superseded, "contested is not a resting state")
        #expect(after.text == "He goes running every morning.", "what I used to think, verbatim")

        let successor = try #require(
            try await store.allLiveMemories().first { $0.text.contains("sit-ups") })
        #expect(after.supersededBy == successor.id)
        // Nothing is inherited across a veto. Strength the old belief accrued
        // *while he considered it wrong* is not credit its replacement gets to
        // spend — unlike an ordinary supersession, where the old belief was right
        // for a while and the new one is its continuation.
        #expect(successor.confirmations == 0)
        #expect(successor.storageStrength < 3.0)
    }

    @Test("A re-read that just restates the rejected memory is the model arguing — he wins")
    func aRestatementIsNotACorrection() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let (contested, _) = try await contestedMemory(
            "He goes running every morning.",
            from: "went for a run this morning",
            in: store, engine: engine)

        // Same claim back, with a full stop and a different case. The evidence
        // does support it — and he still says it is wrong, and he is the authority
        // on his own life.
        model.reread = "STATED|belief|he goes running every morning"
        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: contested.id))
        #expect(after.status == .contested, "not resurrected")
        #expect(after.tier == .cold, "and not offered again")
        #expect(try await store.memoryCount() == 1, "no successor was invented")
    }

    @Test("A contested memory the evidence cannot carry goes cold — it is never deleted")
    func aContestedMemoryGoesColdNotAway() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let (contested, episode) = try await contestedMemory(
            "He is bored by his own project.",
            from: "ugh, this refactor is dragging",
            in: store, engine: engine)

        model.reread = "NOTHING"
        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: contested.id))
        // Still here. Deletion is his hand alone (ADR-0035 §9) — sleep may move a
        // belief out of reach, never out of existence, and the dispute rides along
        // with it if he ever recalls it outright.
        #expect(after.tier == .cold)
        #expect(after.status == .contested)
        #expect(after.storageStrength == contested.storageStrength, "strength is monotone")
        // And the testimony it was drawn from is untouched by any of it.
        let source = try #require(try await store.episode(id: episode.id))
        #expect(source.text == "ugh, this refactor is dragging")
    }

    @Test("A contested belief that went cold is not re-tried every night")
    func aColdContestedBeliefStaysDisposedOf() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let (contested, _) = try await contestedMemory(
            "He is bored by his own project.",
            from: "ugh, this refactor is dragging",
            in: store, engine: engine)

        model.reread = "NOTHING"
        await makeSleep(engine, store, model).run()
        #expect(try #require(try await store.memory(id: contested.id)).tier == .cold)
        let promptsAfterVerdict = model.prompts.count

        // Night two. The verdict does not expire: no re-read, no fresh chance
        // for the model to mint a successor to a belief already disposed of,
        // no second "Retired" journal line.
        await makeSleep(engine, store, model).run()
        #expect(
            model.prompts.count == promptsAfterVerdict,
            "sleep re-read a contested belief it had already retired")
        let retirements = await engine.journal(limit: 100).filter {
            $0.memoryID == contested.id && $0.mutation == .demoted
        }
        #expect(retirements.count == 1)
    }

    @Test("A claim matching a vetoed belief stands on its own — it never confirms the veto away")
    func aVetoedBeliefIsNotConfirmedByReconcile() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        // A contested belief, still hot — the owner vetoed it moments ago and
        // sleep has not re-examined it yet when reconcile meets a lookalike.
        let vetoed = MemoryRecord(
            text: "He goes running every morning.", kind: .belief, provenance: .inferred,
            bornAt: Date())
        try await store.upsert(vetoed)
        await engine.contest(vetoed)

        // The extractor produces the same claim again; the adjudicator would
        // call it SAME — but a contested belief must never be offered as a
        // neighbour, so the claim is judged against nothing and enters as NEW.
        model.reread = "NOTHING"
        model.extraction = "INFERRED|belief|He goes running every morning."
        model.verdict = "SAME 1"
        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "went for a run"))
        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: vetoed.id))
        #expect(after.confirmations == 0, "reconcile confirmed a belief the owner vetoed")
        #expect(after.status == .contested)
    }

    @Test("A restatement is caught through punctuation and case")
    func sameClaimNormalises() {
        #expect(
            MemorySleep.isSameClaim(
                "He goes running every morning.", as: "he goes running every morning"))
        #expect(
            !MemorySleep.isSameClaim(
                "He goes running every morning.", as: "He walks every morning."))
    }

    // MARK: - Grading

    @Test("Grading is where the lifecycle moves — and only useful use strengthens")
    func gradingDrivesTheLifecycle() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "NOTHING"

        let helpful = MemoryRecord(
            text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
            bornAt: Date())
        let irrelevant = MemoryRecord(
            text: "He likes rain.", kind: .belief, provenance: .stated, bornAt: Date())
        try await store.upsert(helpful)
        try await store.upsert(irrelevant)

        let episode = Episode(source: .chat, occurredAt: Date(), text: "what should I order?")
        try await store.append(episode)
        try await store.markConsolidated([episode.id], at: Date())

        try await store.log([
            RetrievalEvent(
                memoryID: helpful.id, episodeID: episode.id, retrievedAt: Date(), cue: "order"),
            RetrievalEvent(
                memoryID: irrelevant.id, episodeID: episode.id, retrievedAt: Date(), cue: "order"),
        ])
        model.grades = "1: decisive\n2: ignored"

        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: helpful.id))
        #expect(after.usefulUseCount == 1)
        #expect(after.stability > MemoryLifecycle.initialStability)
        #expect(after.storageStrength > 0)

        let ignored = try #require(try await store.memory(id: irrelevant.id))
        // `.ignored` is NOT a lapse. Being retrieved and not helping says nothing
        // bad about a memory — only that it was a poor answer to *this* cue. It
        // touches the cue affinity and nothing else.
        #expect(ignored.usefulUseCount == 0)
        #expect(ignored.stability == irrelevant.stability)
        #expect(ignored.storageStrength == irrelevant.storageStrength)

        // The queue drained: a graded event is never re-graded.
        #expect(try await store.ungradedRetrievals().isEmpty)
    }

    /// The invariant that makes every number in the eval harness meaningful.
    @Test("Consolidation never strengthens a memory it merely re-read")
    func replayNeverStrengthens() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "STATED|belief|He is allergic to shellfish."
        model.verdict = "SAME 1"

        let existing = MemoryRecord(
            text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
            bornAt: Date())
        try await store.upsert(existing)
        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "no shellfish"))

        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: existing.id))
        // A confirmation raises *confidence*, not strength. Sleep re-reading its
        // own store is not evidence that a memory was useful — only a real
        // retrieval that a judge graded useful is, and that is the one path by
        // which stability may rise. Otherwise the store trains on its own priors
        // and the lifecycle measures nothing but how often sleep ran.
        #expect(after.stability == existing.stability)
        #expect(after.storageStrength == existing.storageStrength)
        #expect(after.usefulUseCount == 0)
        #expect(after.confirmations == 1)
    }

    /// Sleep must not feed the counter it retires against.
    ///
    /// Reconcile looks up each new claim's nearest neighbours. Those lookups are
    /// bookkeeping — nobody *saw* the results. When they marked the neighbours
    /// "seen", sleep inflated `seenCount` on its own store, and the third
    /// retirement path ("shown eight times and never once helped") began firing on
    /// memories that had never been shown to anyone. Measured on the owner's real
    /// store: `He runs an X/Twitter account: @spok_vulkan` reached `seenCount 8`
    /// and went cold after a single night, having never been surfaced in a
    /// conversation. Left alone, sleep retires everything the owner ever told it.
    @Test("Sleep's internal lookups never mark a memory as seen")
    func reconcileDoesNotInflateSeenCount() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let told = MemoryRecord(
            text: "He runs an X/Twitter account: @spok_vulkan.", kind: .belief,
            provenance: .stated, bornAt: Date())
        try await store.upsert(told)

        // Ten nights of consolidation, each producing a claim that reconcile will
        // compare against everything already in the store.
        model.extraction = "INFERRED|belief|He posts about Star Trek."
        model.verdict = "NEW"
        for night in 0..<10 {
            try await store.append(
                Episode(source: .chat, occurredAt: Date(), text: "night \(night)"))
            await makeSleep(engine, store, model).run()
        }

        let after = try #require(try await store.memory(id: told.id))
        // Never surfaced to anyone, so never seen — and therefore never retired.
        #expect(after.seenCount == 0, "sleep marked a memory seen that nobody saw")
        #expect(after.tier != .cold, "sleep retired a memory it had only ever looked at itself")
        #expect(after.status == .live)
    }

    // MARK: - Yield

    @Test("The owner coming back cancels the run, and the unread episodes survive")
    func yieldingLosesNoWork() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "STATED|belief|He is allergic to shellfish."

        for index in 0..<20 {
            try await store.append(
                Episode(source: .chat, occurredAt: Date(), text: "turn number \(index)"))
        }

        let sleep = makeSleep(engine, store, model)
        sleep.start()
        sleep.yield()

        // The work item that died is simply redone next sleep. Episodes are
        // marked consumed only once the call that read them *returned*, so a
        // cancelled batch stays in the queue — which is what makes yielding free.
        #expect(!sleep.isRunning)
        #expect(!(try await store.unconsolidatedEpisodes().isEmpty))
    }

    /// A cancelled run outlives the `yield()` that killed it — it keeps executing
    /// until its `await` unwinds. If its cleanup clears the handle of the run that
    /// replaced it, `isRunning` goes false while a run is live, and the next idle
    /// tick starts a **second concurrent sleep** over the same episodes. Two
    /// consolidations distilling the same day is how a store fills with duplicates.
    @Test("A yielded run cannot clear the handle of the run that replaced it")
    func aDyingRunCannotDisownItsSuccessor() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()
        model.extraction = "NOTHING"
        for index in 0..<20 {
            try await store.append(
                Episode(source: .chat, occurredAt: Date(), text: "turn \(index)"))
        }

        let sleep = makeSleep(engine, store, model)
        sleep.start()
        sleep.yield()  // run 1 is cancelled but still unwinding
        sleep.start()  // run 2 takes the handle

        // Give the cancelled run every chance to run its cleanup and clobber run 2.
        for _ in 0..<20 { await Task.yield() }

        #expect(sleep.isRunning, "run 2 still owns the handle")
    }

    // MARK: - Parsing (what the model actually returns is never clean)

    @Test("Claim parsing rejects fragments and unmarked provenance")
    func claimParsingIsStrict() {
        let claims = MemorySleep.parseClaims(
            """
            STATED|belief|He is allergic to shellfish.
            he likes cats
            MAYBE|belief|He might like dogs.
            INFERRED|nonsense|He works late.
            STATED|belief|short
            NOTHING
            """,
            sourceEpisodeIDs: [])

        // Two survive: the well-formed STATED line, and the INFERRED one whose
        // *kind* is junk but whose provenance is not — kind falls back, provenance
        // never does. A line with no provenance marker is not a memory.
        #expect(claims.count == 2)
        #expect(claims.contains { $0.provenance == .stated && $0.text.contains("shellfish") })
        #expect(claims.contains { $0.provenance == .inferred && $0.kind == .belief })
        #expect(!claims.contains { $0.text == "short" })
    }

    @Test("A verdict the model garbles drops the claim rather than inventing a memory")
    func unparseableVerdictsAreConservative() {
        #expect(MemorySleep.parseVerdict("REPLACES 2", count: 3) == .replaces([1]))
        #expect(MemorySleep.parseVerdict("SAME 1", count: 3) == .same(0))
        #expect(MemorySleep.parseVerdict("NEW", count: 3) == .new)
        // Local models preface and hedge. "I think this is probably SAME 1..."
        #expect(MemorySleep.parseVerdict("Looking at these, I'd say SAME 1.", count: 3) == .same(0))
        // Garbage with existing neighbours: drop the claim outright. Adding a
        // memory the judge could not vouch for is the worse failure — and so is
        // confirming a neighbour the judge never actually matched, which is
        // what folding garbage into `SAME 1` used to do.
        #expect(MemorySleep.parseVerdict("¯\\_(ツ)_/¯", count: 3) == .drop)
        #expect(MemorySleep.parseVerdict("¯\\_(ツ)_/¯", count: 0) == .new)
    }

    @Test("REPLACES reads every listed belief — one correction can falsify several")
    func replacesParsesLists() {
        // The #333 shape: "that's not my name" contradicted a stated belief AND
        // an inferred event. A single-index verdict could retire only one.
        #expect(MemorySleep.parseVerdict("REPLACES 1, 3", count: 3) == .replaces([0, 2]))
        #expect(MemorySleep.parseVerdict("REPLACES 2 and 3", count: 3) == .replaces([1, 2]))
        #expect(MemorySleep.parseVerdict("I'd say REPLACES 1,2.", count: 3) == .replaces([0, 1]))
        // A digit inside trailing prose is not a target: the list ends at the
        // first word that is not a separator.
        #expect(
            MemorySleep.parseVerdict("REPLACES 1 because 2 is older", count: 3) == .replaces([0]))
        // Repeats collapse; out-of-range ends the list.
        #expect(MemorySleep.parseVerdict("REPLACES 2, 2", count: 3) == .replaces([1]))
        #expect(MemorySleep.parseVerdict("REPLACES 9", count: 3) == .drop)
    }

    @Test("A garbled verdict leaves the store exactly as it was")
    func unparseableVerdictTouchesNothing() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        let existing = MemoryRecord(
            text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
            bornAt: Date())
        try await store.upsert(existing)
        try await store.append(
            Episode(source: .chat, occurredAt: Date(), text: "no shellfish for me"))

        model.extraction = "STATED|belief|He cannot eat shellfish."
        model.verdict = "¯\\_(ツ)_/¯"
        await makeSleep(engine, store, model).run()

        let after = try #require(try await store.memory(id: existing.id))
        // Neither confirmed on the strength of garbage…
        #expect(after.confirmations == 0)
        // …nor a new memory added the judge could not vouch for.
        #expect(try await store.memoryCount() == 1)
    }

    @Test("Grades default to ignored when the judge does not say")
    func gradeParsingDefaultsToIgnored() {
        let grades = MemorySleep.parseGrades("1: decisive\n3: harmful", count: 3)
        #expect(grades == [.decisive, .ignored, .harmful])
        // Not "useful by default": a memory has to *earn* its strength, and the
        // absence of a verdict is not evidence of one.
        #expect(MemorySleep.parseGrades("", count: 2) == [.ignored, .ignored])
    }

    // MARK: - Multi-belief contradictions (#333)

    @Test("One correction retires every belief it falsifies — no live contradiction survives")
    func replacesRetiresAllListedNeighbours() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        // The live #333 store: the same wrong fact as a stated belief AND an
        // inferred event. A single-index REPLACES could retire only one, and
        // recall would keep serving the other beside its own correction.
        let stated = MemoryRecord(
            text: "He gave me the nickname \"Pelican.\"", kind: .belief,
            provenance: .stated, bornAt: Date())
        let inferred = MemoryRecord(
            text: "He gave the assistant the nickname \"Pelican.\"", kind: .event,
            provenance: .inferred, bornAt: Date())
        try await store.upsert(stated)
        try await store.upsert(inferred)

        try await store.append(
            Episode(
                source: .chat, occurredAt: Date(),
                text: "Pelican is not your name — that was the SVG image I asked for."))
        model.extraction =
            "STATED|belief|\"Pelican\" referred to an SVG image he requested, not a name."
        model.verdict = "REPLACES 1, 2"
        await makeSleep(engine, store, model).run()

        let oldStated = try #require(try await store.memory(id: stated.id))
        let oldInferred = try #require(try await store.memory(id: inferred.id))
        #expect(oldStated.status == .superseded)
        #expect(oldInferred.status == .superseded)
        // Both retired in favour of the SAME successor.
        #expect(oldStated.supersededBy != nil)
        #expect(oldStated.supersededBy == oldInferred.supersededBy)

        let live = try await store.allLiveMemories()
        #expect(live.count == 1)
        #expect(live.first?.text.contains("SVG image") == true)
    }

    @Test("The re-read sees what he said when he rejected the belief")
    func rereadCarriesTheContestNote() async throws {
        let store = try makeStore()
        let engine = makeEngine(store)
        let model = ScriptedModel()

        // The evidence alone re-derives the mistake — the July 7 episode reads
        // like a christening. His rejection is what points the other way.
        let episode = Episode(
            source: .chat, occurredAt: Date(),
            text: "Very nice, Pelican. You are an artist.")
        try await store.append(episode)
        try await store.markConsolidated([episode.id], at: Date())
        let wrong = MemoryRecord(
            text: "He gave me the nickname \"Pelican.\"", kind: .belief,
            provenance: .inferred, sourceEpisodeIDs: [episode.id], bornAt: Date())
        try await store.upsert(wrong)
        _ = try await store.contest(
            id: wrong.id, at: Date(),
            reason: "It referred to the SVG pelican image, not a name for the assistant.")

        model.reread = "NOTHING"
        await makeSleep(engine, store, model).run()

        let rereadPrompt = try #require(
            model.prompts.first { $0.contains("memories is WRONG") })
        #expect(rereadPrompt.contains("When he rejected it:"))
        #expect(rereadPrompt.contains("SVG pelican image"))
        // And NOTHING still retires it cold, note or no note.
        #expect(try #require(try await store.memory(id: wrong.id)).tier == .cold)
    }
}
