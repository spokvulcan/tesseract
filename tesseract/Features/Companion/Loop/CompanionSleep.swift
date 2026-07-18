//
//  CompanionSleep.swift
//  tesseract
//
//  The entity's practice at the tail of the memory sleep pass (ADR-0046):
//  after consolidation, under the same cancellable run and the same
//  lease-per-generation discipline. #370's item: the standing-instructions
//  review — the document's own consolidation loop, promised by ADR-0040 §12.
//  Jarvis re-reads both sections against the week's verbatim feedback and
//  either keeps the document (the common, correct answer) or authors a full
//  revision, appended as an entity version on the owner-visible history.
//
//  Sleep passes fire on every idle; the review runs at most once per
//  calendar day (stamped in the loop's day state). #373 adds the Digest
//  beside it.
//

import Foundation

@MainActor
final class CompanionSleep {

    private let store: MemoryStore
    private let recorder: CompanionFlightRecorder
    private let arbiter: any InferenceArbitrating
    private let complete: @Sendable (String) async throws -> String
    private let isEnabled: () -> Bool

    init(
        store: MemoryStore,
        recorder: CompanionFlightRecorder,
        arbiter: any InferenceArbitrating,
        complete: @escaping @Sendable (String) async throws -> String,
        isEnabled: @escaping () -> Bool
    ) {
        self.store = store
        self.recorder = recorder
        self.arbiter = arbiter
        self.complete = complete
        self.isEnabled = isEnabled
    }

    /// The whole practice — called by the memory pass after its own items.
    func nightly(now: Date = Date()) async {
        guard isEnabled() else { return }
        await reviewInstructions(now: now)
    }

    // MARK: - Instruction review (#370)

    /// Once per day: read the document in force — both sections — beside the
    /// week's feedback, and keep or revise. Keeping is the expected outcome;
    /// a revision must come back whole, with both section markers, or it is
    /// discarded as noise (a document must never be lost to a garbled reply).
    func reviewInstructions(now: Date = Date()) async {
        guard let current = try? await store.currentInstructions() else { return }
        let todayKey = TrackingDay.key(for: now)
        guard let dayState = try? await store.loopDayState(todayKey),
            dayState.instructionsReviewedAt == nil
        else { return }

        let feedback = recorder.records(since: now.addingTimeInterval(-7 * 86_400))
            .filter { $0.event.hasPrefix("feedback.") }
            .compactMap(\.note)
        let feedbackBlock =
            feedback.isEmpty
            ? "None recorded." : feedback.suffix(20).map { "- \($0)" }.joined(separator: "\n")

        let prompt = """
            You are Jarvis, reviewing your standing instructions at the end of the \
            day — the document you live by. It has two sections: IDENTITY rides \
            every conversation you have; LOOP POLICY rides only your Mission \
            Control loop turns, where your delivery tools exist.

            The document in force (v\(current.version)):
            \(current.text)

            His feedback this week, verbatim:
            \(feedbackBlock)

            If the document still serves, answer with the single word KEEP — that \
            is the common, correct answer. Revise only for something durable you \
            can point at: a correction he gave, a rule that changed, guidance \
            sitting in the wrong section. If you revise, answer with:
            WHY: <one line>
            followed by the COMPLETE new document — both sections, keeping the \
            "\(CompanionInstructions.identityMarker)" and \
            "\(CompanionInstructions.loopPolicyMarker)" headers, no other text.
            """

        let revision: (text: String, why: String)?
        do {
            let response = try await generate(prompt)
            revision = Self.parseReview(response)
        } catch {
            // A failed generation is not a verdict — leave the day unstamped
            // so the next pass retries.
            Log.companion.error("Instruction review failed: \(error.localizedDescription)")
            return
        }

        if let revision {
            let version = try? await store.appendInstructions(
                text: revision.text, author: "entity", note: "sleep review: \(revision.why)")
            recorder.record(
                "instructions.sleep-review",
                snapshot: [
                    "verdict": "revised", "version": version.map(String.init) ?? "?",
                ],
                note: revision.why)
        } else {
            recorder.record("instructions.sleep-review", snapshot: ["verdict": "kept"])
        }

        var updated = (try? await store.loopDayState(todayKey)) ?? CompanionLoopDayState()
        updated.instructionsReviewedAt = now
        try? await store.setLoopDayState(todayKey, updated)
    }

    /// Nil means keep. A revision must carry the WHY line, both section
    /// markers, and fit the cap — anything else is noise, and noise keeps.
    static func parseReview(_ response: String) -> (text: String, why: String)? {
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine =
            trimmed.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? ""
        if firstLine.uppercased().contains("KEEP") { return nil }

        guard let whyRange = trimmed.range(of: "WHY:") else { return nil }
        let afterWhy = trimmed[whyRange.upperBound...]
        guard let lineBreak = afterWhy.firstIndex(of: "\n") else { return nil }
        let why = String(afterWhy[..<lineBreak]).trimmingCharacters(in: .whitespaces)
        let document = String(afterWhy[lineBreak...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !why.isEmpty,
            document.contains(CompanionInstructions.identityMarker),
            document.contains(CompanionInstructions.loopPolicyMarker),
            document.count <= CompanionInstructions.maxLength,
            !CompanionInstructions.split(document).identity.isEmpty
        else { return nil }
        return (document, why)
    }

    /// One generation under its own lease — the MemorySleep discipline: the
    /// worst a foreground turn waits is one generation.
    private func generate(_ prompt: String) async throws -> String {
        try await arbiter.withExclusiveGPU(.llm) { [complete] in
            try await complete(prompt)
        }
    }
}
