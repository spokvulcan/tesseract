//
//  CompanionSleepTests.swift
//  tesseractTests
//
//  The entity's tail of the sleep pass (ADR-0046, #370): the standing-
//  instructions review, driven by a scripted model — no GPU, no weights —
//  so what matters is pinned: given what the model said, what happened to
//  the document, the day stamp, and the record.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct CompanionSleepTests {

    private final class ScriptedModel: @unchecked Sendable {
        var reply = "KEEP"
        private(set) var prompts: [String] = []
        var complete: @Sendable (String) async throws -> String {
            { [self] prompt in
                await MainActor.run { self.prompts.append(prompt) }
                return reply
            }
        }
    }

    private func makeSleep(
        _ store: MemoryStore, _ recorder: CompanionFlightRecorder, _ model: ScriptedModel,
        enabled: Bool = true
    ) -> CompanionSleep {
        CompanionSleep(
            store: store, recorder: recorder, arbiter: InMemoryInferenceArbiter(),
            complete: model.complete, isEnabled: { enabled })
    }

    private static let revisedDocument = """
        # IDENTITY

        You are Jarvis. Briefer than before.

        # LOOP POLICY

        Pulse at 14:00. Everything else as it was.
        """

    @Test func keepLeavesTheDocumentAndStampsTheDay() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        try await store.seedInstructionsIfNeeded(CompanionInstructions.seed)
        let model = ScriptedModel()

        await makeSleep(store, recorder, model).nightly()

        let current = try #require(try await store.currentInstructions())
        #expect(current.version == 1)
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(
            events.contains {
                $0.event == "instructions.sleep-review" && $0.snapshot?["verdict"] == "kept"
            })
        // The prompt carried both sections and the review runs once per day.
        #expect(model.prompts.count == 1)
        #expect(model.prompts[0].contains("# IDENTITY"))
        #expect(model.prompts[0].contains("# LOOP POLICY"))

        await makeSleep(store, recorder, model).nightly()
        #expect(model.prompts.count == 1)
    }

    @Test func aRevisionAppendsAnEntityVersion() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        try await store.seedInstructionsIfNeeded(CompanionInstructions.seed)
        let model = ScriptedModel()
        model.reply = "WHY: he asked for a later pulse\n\(Self.revisedDocument)"

        await makeSleep(store, recorder, model).nightly()

        let current = try #require(try await store.currentInstructions())
        #expect(current.version == 2)
        #expect(current.author == "entity")
        #expect(current.note == "sleep review: he asked for a later pulse")
        #expect(current.text.contains("Pulse at 14:00"))
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(
            events.contains {
                $0.event == "instructions.sleep-review"
                    && $0.snapshot?["verdict"] == "revised"
            })
    }

    @Test func garbledRepliesKeepTheDocument() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        try await store.seedInstructionsIfNeeded(CompanionInstructions.seed)
        let model = ScriptedModel()
        // No WHY, no markers — noise must never replace the document.
        model.reply = "Here is a much better document:\nBe helpful."

        await makeSleep(store, recorder, model).nightly()
        #expect(try #require(try await store.currentInstructions()).version == 1)
    }

    @Test func disabledOrUnseededDoesNothing() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        let model = ScriptedModel()

        // Unseeded: no document to review, no generation spent.
        await makeSleep(store, recorder, model).nightly()
        #expect(model.prompts.isEmpty)

        try await store.seedInstructionsIfNeeded(CompanionInstructions.seed)
        await makeSleep(store, recorder, model, enabled: false).nightly()
        #expect(model.prompts.isEmpty)
    }

    // MARK: - The parse seam

    @Test func parseReviewVerdicts() {
        #expect(CompanionSleep.parseReview("KEEP") == nil)
        #expect(CompanionSleep.parseReview("  keep.\n") == nil)
        #expect(CompanionSleep.parseReview("KEEP — it still serves.") == nil)

        let parsed = CompanionSleep.parseReview(
            "WHY: tightened the register\n\(Self.revisedDocument)")
        #expect(parsed?.why == "tightened the register")
        #expect(parsed?.text.contains("# LOOP POLICY") == true)

        // Missing either marker, or missing the why line: noise, keep.
        #expect(CompanionSleep.parseReview("WHY: x\n# IDENTITY\n\nonly half") == nil)
        #expect(CompanionSleep.parseReview(Self.revisedDocument) == nil)
        #expect(
            CompanionSleep.parseReview(
                "WHY: grew\n# IDENTITY\n\(String(repeating: "a", count: 13_000))\n# LOOP POLICY\nx"
            ) == nil)
    }
}
