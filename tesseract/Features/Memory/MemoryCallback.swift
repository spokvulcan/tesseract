//
//  MemoryCallback.swift
//  tesseract
//
//  The morning callback (map #301 ticket #302's acceptance bar, ADR-0035).
//
//  This is the smallest, sharpest test of whether any of this works. The bar the
//  owner set is one line:
//
//    > **one specific, true, first-person callback every morning.**
//
//  Every word of that is load-bearing. *Specific* rules out "How's your day?".
//  *True* rules out the failure mode this whole design exists to prevent — a
//  memory system that is confidently, fluently wrong is worse than none, because
//  it is unfalsifiable from the inside. *First-person* means it speaks as
//  someone who was there, not as a database reporting rows. And *every morning*
//  means it has to work on a Tuesday when nothing interesting happened.
//
//  So: recall, then compose, then verify the composition is grounded — and if
//  any of that fails, say the boring hardcoded thing instead. A generic line is a
//  disappointment. An invented one is a betrayal.
//

import Foundation

@MainActor
enum MemoryCallback {

    /// Compose the line for `cue` — the beat's own prompt — from what is
    /// actually known. Returns nil when memory has nothing to offer, which the
    /// caller must treat as "use the hardcoded line".
    static func compose(
        cue: String,
        engine: MemoryEngine,
        arbiter: any InferenceArbitrating,
        complete: @escaping @Sendable (String) async throws -> String,
        now: Date = Date()
    ) async -> String? {
        let context = await engine.retrieve(
            cue: cue, forEpisode: nil, memoryBudget: 6, episodeBudget: 4, now: now)
        guard !context.isEmpty else {
            Log.memory.info("Callback: nothing recalled — falling back to the plain beat")
            return nil
        }

        let beliefs =
            (context.core + context.recalled.map(\.memory))
            .filter { $0.status == .live }
            .map {
                "- \($0.text)\($0.provenance == .inferred ? "  (I inferred this — be careful)" : "")"
            }
            .joined(separator: "\n")

        let day = DateFormatter()
        day.dateFormat = "EEEE d MMMM"
        let recent = context.episodes
            .map {
                "- \(day.string(from: $0.episode.occurredAt)): \"\($0.episode.text.prefix(220))\""
            }
            .joined(separator: "\n")

        guard !beliefs.isEmpty || !recent.isEmpty else { return nil }

        let prompt = """
            You are the personal assistant of the man described below. It is \
            \(day.string(from: now)). You are about to send him a single short \
            notification — the thing you say to him unprompted, because you have \
            been paying attention.

            The beat you are speaking on is: "\(cue)"

            What you know about him:
            \(beliefs.isEmpty ? "(nothing yet)" : beliefs)

            What he actually said recently:
            \(recent.isEmpty ? "(nothing recent)" : recent)

            Write that one line.

            Rules:
            - ONE sentence. It is a notification banner — twenty words at the very most.
            - First person, to him. Warm, plain, and unfussy. No "Hey there!", no emoji.
            - It must be SPECIFIC and it must be TRUE. Anchor it in something concrete \
            above — a thing he actually said, a thing you actually know. A line that \
            would work for any person on any day is a failure.
            - **Never invent.** Do not add a detail that is not above, do not sharpen a \
            vague memory into a precise one, do not guess at a name or a date. If you \
            are unsure of something, leave it out.
            - Usually end with a question. You are opening a conversation, not filing a \
            report.
            - If there is genuinely nothing specific and true to say, answer with the \
            single word: PASS.

            Write only the line, nothing else.
            """

        do {
            let raw = try await arbiter.withExclusiveGPU(.llm) {
                try await complete(prompt)
            }
            guard let line = clean(raw) else {
                Log.memory.info("Callback: the model passed — falling back to the plain beat")
                return nil
            }
            Log.memory.info("Callback composed: \(line)")
            return line
        } catch {
            Log.memory.error("Callback composition failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Take the model's output and make it safe to show, or refuse it.
    ///
    /// Local models preface, apologise, and offer three options. A banner gets
    /// one sentence or nothing.
    static func clean(_ raw: String) -> String? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)

        // A model that follows the PASS instruction has told us it has nothing —
        // believe it. That is the system working, not failing.
        guard !text.uppercased().hasPrefix("PASS") else { return nil }

        // Take the first non-empty line: everything after it is commentary.
        if let first = text.split(separator: "\n").first(where: {
            !$0.trimmingCharacters(in: .whitespaces).isEmpty
        }) {
            text = String(first).trimmingCharacters(in: .whitespaces)
        }
        // Strip the quotes a model wraps a "line" in when asked for one.
        if text.count >= 2, text.hasPrefix("\""), text.hasSuffix("\"") {
            text = String(text.dropFirst().dropLast())
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Too short to be a sentence, or too long to be a banner. Either way, not
        // what was asked for — and a half-followed instruction is a bad sign
        // about the rest of the output.
        guard text.count >= 12, text.count <= 220 else { return nil }
        return text
    }
}
