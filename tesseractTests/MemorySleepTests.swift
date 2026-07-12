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

    private func makeSleep(_ engine: MemoryEngine, _ model: ScriptedModel) -> MemorySleep {
        MemorySleep(
            engine: engine, arbiter: InMemoryInferenceArbiter(), complete: model.complete)
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
        await makeSleep(engine, model).run()

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

        await makeSleep(engine, model).run()

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
        await makeSleep(engine, model).run()

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

        await makeSleep(engine, model).run()

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

        await makeSleep(engine, model).run()

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
        await makeSleep(engine, model).run()

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
        await makeSleep(engine, model).run()

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
        await makeSleep(engine, model).run()

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

        await makeSleep(engine, model).run()

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

        await makeSleep(engine, model).run()

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
            await makeSleep(engine, model).run()
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

        let sleep = makeSleep(engine, model)
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

        let sleep = makeSleep(engine, model)
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
        #expect(MemorySleep.parseVerdict("REPLACES 2", count: 3) == .replaces(1))
        #expect(MemorySleep.parseVerdict("SAME 1", count: 3) == .same(0))
        #expect(MemorySleep.parseVerdict("NEW", count: 3) == .new)
        // Local models preface and hedge. "I think this is probably SAME 1..."
        #expect(MemorySleep.parseVerdict("Looking at these, I'd say SAME 1.", count: 3) == .same(0))
        // Garbage with existing neighbours: fold into the nearest rather than add.
        // Silently adding a memory the judge could not vouch for is the worse
        // failure — an unrecoverable one.
        #expect(MemorySleep.parseVerdict("¯\\_(ツ)_/¯", count: 3) == .same(0))
        #expect(MemorySleep.parseVerdict("¯\\_(ツ)_/¯", count: 0) == .new)
    }

    @Test("Grades default to ignored when the judge does not say")
    func gradeParsingDefaultsToIgnored() {
        let grades = MemorySleep.parseGrades("1: decisive\n3: harmful", count: 3)
        #expect(grades == [.decisive, .ignored, .harmful])
        // Not "useful by default": a memory has to *earn* its strength, and the
        // absence of a verdict is not evidence of one.
        #expect(MemorySleep.parseGrades("", count: 2) == [.ignored, .ignored])
    }
}
