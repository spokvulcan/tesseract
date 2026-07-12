//
//  MemoryCallbackTests.swift
//  tesseractTests
//
//  The acceptance bar of map #301 ticket #302 — "one specific, true, first-person
//  callback every morning" — expressed as tests.
//
//  These exist because the first live morning beat cleared none of them. It said:
//  "Morning. What's the one hard thing today, the AI agent or that pending task?"
//  Composed, not scripted; grounded in his real work; and still a failure, because
//  offering him two things is telling him you know neither. That line is the
//  fixture below.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory callback", .serialized)
@MainActor
struct MemoryCallbackTests {

    /// The composer and the check are one closure with two prompts; the critic's
    /// is the one that tells the model to stop me embarrassing myself.
    final class ScriptedModel: @unchecked Sendable {
        var line = "PASS"
        var verdict = "KEEP"
        private(set) var critiqued: [String] = []

        var complete: @Sendable (String) async throws -> String {
            { [self] prompt in
                if prompt.contains("stop me from embarrassing myself") {
                    await MainActor.run { self.critiqued.append(prompt) }
                    return verdict
                }
                return line
            }
        }
    }

    private func makeStore() throws -> MemoryStore {
        try MemoryStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("callback-\(UUID().uuidString)", isDirectory: true))
    }

    private func makeEngine(_ store: MemoryStore) -> MemoryEngine {
        MemoryEngine(
            store: store, embedder: MemoryEmbedder(),
            isEnabled: { true }, isDictationCaptureEnabled: { true },
            embedderDirectory: { nil })
    }

    /// A store with something real in it to be grounded in.
    ///
    /// Worth saying out loud, because it cost a red test to learn: with no
    /// embedder loaded, retrieval is keyword-only, so the beat's cue has to
    /// actually overlap what is stored or nothing comes back at all. That is not
    /// a test artifact — it is how the beat behaves on a cold install, and the
    /// fallback it takes then (the plain line) is the correct one.
    private func seed(_ store: MemoryStore) async throws {
        try await store.upsert(
            MemoryRecord(
                text: "He is building an offline AI agent for macOS called Tesseract.",
                kind: .belief, provenance: .stated, bornAt: Date()))
        try await store.upsert(
            MemoryRecord(
                text: "The hard thing he keeps returning to is the Tesseract prefix cache.",
                kind: .pattern, provenance: .inferred, bornAt: Date()))
        try await store.append(
            Episode(
                source: .chat, occurredAt: Date(),
                text: "the prefix cache eviction is still the hard part"))
    }

    private func compose(_ engine: MemoryEngine, _ model: ScriptedModel) async -> String? {
        await MemoryCallback.compose(
            cue: "What is the one hard thing today?",
            engine: engine, arbiter: InMemoryInferenceArbiter(), complete: model.complete)
    }

    // MARK: - The gates

    @Test("A specific, grounded line the check keeps is the line he gets")
    func aGoodLineSurvives() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        model.line = "Morning — is the Tesseract prefix cache still the thing standing in your way?"

        let line = await compose(makeEngine(store), model)
        #expect(line == model.line)
        #expect(model.critiqued.count == 1, "a line must be checked before it is sent")
    }

    @Test("The hedge is refused — naming two things admits you know neither")
    func theHedgeIsRefused() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        // The shape of the real first morning beat — one real thing, one vague one,
        // joined by "or" — anchored in what this store actually holds so that it
        // clears the word-match and has to be killed by the check itself.
        model.line =
            "Morning. What's the hard thing today — the prefix cache, or that pending task?"
        model.verdict = "PASS"

        #expect(await compose(makeEngine(store), model) == nil)
        // It must die *at the check*, not at the word-match — "agent" is genuinely
        // his. If this ever passes with zero critiques, the test has stopped
        // testing the thing it was written for.
        #expect(model.critiqued.count == 1)
        let asked = try #require(model.critiqued.first)
        #expect(asked.contains(model.line), "the check has to see the actual line")
        #expect(asked.contains("Tesseract"), "and the evidence it is judged against")
    }

    @Test("An invented detail never reaches the check — nothing in it came from memory")
    func aninventedLineIsRefusedBeforeTheCheck() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        model.line = "Morning — how was the flight to Berlin yesterday?"

        #expect(await compose(makeEngine(store), model) == nil)
        // The cheap gate does its job first: no GPU spent checking a fabrication.
        #expect(model.critiqued.isEmpty)
    }

    @Test("A line that would land on a stranger is not a callback")
    func theGenericLineIsRefused() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        model.line = "Good morning — hope today goes well for you, what are you up to?"

        #expect(await compose(makeEngine(store), model) == nil)
        #expect(model.critiqued.isEmpty)
    }

    @Test("A confused check is a refusal — silence is never consent")
    func anUnparseableVerdictRefuses() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        model.line = "Morning — is the prefix cache eviction still the hard part?"
        model.verdict = "KEEP, but I would soften it — or maybe PASS, on reflection."

        #expect(await compose(makeEngine(store), model) == nil)
    }

    @Test("The model's own PASS is believed, and costs nothing to check")
    func theModelMayPass() async throws {
        let store = try makeStore()
        try await seed(store)
        let model = ScriptedModel()
        model.line = "PASS"

        #expect(await compose(makeEngine(store), model) == nil)
        #expect(model.critiqued.isEmpty)
    }

    @Test("Empty memory means the plain beat — nothing to be specific about")
    func emptyMemoryFallsBack() async throws {
        let store = try makeStore()
        let model = ScriptedModel()
        model.line = "Morning — how's the Tesseract agent coming along?"

        #expect(await compose(makeEngine(store), model) == nil)
    }

    // MARK: - Grounding, on its own

    @Test("Grounding matches the word, not the exact form of it")
    func groundingStems() {
        let evidence = ["He runs a hundred sit-ups most mornings.", "He is exercising daily."]
        #expect(MemoryCallback.grounded("Did the exercise happen today?", in: evidence))
        #expect(!MemoryCallback.grounded("Did anything happen today?", in: evidence))
    }

    @Test("Grounding needs a distinctive word — warm filler is not evidence")
    func groundingIgnoresFiller() {
        let evidence = ["He is building an offline AI agent."]
        // Every word this line shares with the evidence is one it would share with
        // any evidence at all.
        #expect(
            !MemoryCallback.grounded("Hope your morning is going well, what's next?", in: evidence))
        #expect(MemoryCallback.grounded("How's the offline agent going?", in: evidence))
    }

    @Test("With nothing recalled, nothing is grounded")
    func groundingOnEmptyEvidence() {
        #expect(!MemoryCallback.grounded("How's the Tesseract agent?", in: []))
    }

    // MARK: - Cleaning

    @Test("A banner is one sentence, unquoted, or it is nothing")
    func cleaning() {
        #expect(MemoryCallback.clean("PASS") == nil)
        #expect(
            MemoryCallback.clean("  \"Morning — how are the sit-ups?\"  ")
                == "Morning — how are the sit-ups?")
        #expect(
            MemoryCallback.clean(
                "Morning — how are the sit-ups?\n\nLet me know if you'd like another!")
                == "Morning — how are the sit-ups?")
        #expect(MemoryCallback.clean("Hi.") == nil)
        #expect(MemoryCallback.clean(String(repeating: "a", count: 300)) == nil)
    }
}
