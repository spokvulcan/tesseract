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
//  So the line has to clear three gates before it is allowed out:
//
//    1. `clean`    — it is one sentence, not a model's essay about the sentence.
//    2. `grounded` — it re-uses a distinctive word from what was actually
//                    recalled. Cheap, deterministic, and it catches both halves
//                    of the failure: the generic line shares nothing but
//                    stopwords, the invented one shares nothing at all.
//    3. `critique` — a second pass, shown the same evidence, whose only job is to
//                    refuse. It catches what a word-match cannot: the hedge.
//                    "The AI agent or that pending task?" is grounded, fluent,
//                    and still a failure — a guess wearing a suit. Naming two
//                    things is admitting you know neither.
//
//  Fail any gate and the beat says its boring hardcoded thing instead. A generic
//  line is a disappointment. An invented one is a betrayal.
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

        let live = (context.core + context.recalled.map(\.memory)).filter { $0.status == .live }
        let beliefs =
            live
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

        // What the line is allowed to be about. The bulleted strings above are for
        // the model to read; this is for the code to check against.
        let evidence =
            live.map(\.text) + context.episodes.map { String($0.episode.text.prefix(220)) }

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
            - **Name ONE thing, and commit to it.** If you offer him two ("the X or \
            that Y?"), you are telling him you do not know which — and a guess in a \
            suit is still a guess. Do not write "that project", "your task", "the \
            thing you mentioned": if you cannot name it, you do not know it. Pick the \
            one thing you are surest of, or PASS.
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
            // One lease for both passes. Handing the GPU back between composing and
            // checking would just let something else take it and make the check wait.
            let line = try await arbiter.withExclusiveGPU(.llm) { () -> String? in
                guard let candidate = clean(try await complete(prompt)) else {
                    Log.memory.info("Callback: the model passed — falling back to the plain beat")
                    return nil
                }
                guard grounded(candidate, in: evidence) else {
                    Log.memory.info(
                        "Callback refused — nothing in it came from memory: \(candidate)")
                    return nil
                }
                guard try await critique(candidate, against: evidence, complete: complete) else {
                    Log.memory.info(
                        "Callback refused by the check — vague or unsupported: \(candidate)")
                    return nil
                }
                return candidate
            }
            if let line { Log.memory.info("Callback composed: \(line)") }
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

    /// Does this line actually come from `evidence` — or from the model's idea of
    /// a nice thing to say?
    ///
    /// The test is one distinctive word in common. It is a floor, not a filter:
    /// it cannot tell a good line from a mediocre one, but it does reliably
    /// catch the two lines that must never ship — the one that names something
    /// he never said, and the one that names nothing at all.
    static func grounded(_ line: String, in evidence: [String]) -> Bool {
        let known = Set(evidence.flatMap(stems(of:)))
        guard !known.isEmpty else { return false }
        return stems(of: line).contains(where: known.contains)
    }

    /// Words worth matching on, cut to a crude five-letter stem so "exercise"
    /// recognises "exercising" and "agent" recognises "agents". Stemming before
    /// the filler check is deliberate: it makes "morning" and "mornings" the same
    /// nothing-word. Short words carry no identity, and the fillers are how a
    /// generic line sounds warm without saying anything — neither grounds a claim.
    private static func stems(of text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .filter { $0.count >= 5 }
            .map { String($0.prefix(5)) }
            .filter { !Self.filler.contains($0) }
    }

    /// Five-letter stems, to match what `stems(of:)` produces.
    private static let filler: Set<String> = [
        "about", "after", "again", "alway", "anoth", "anyth", "becau", "befor", "could",
        "doing", "eveni", "every", "getti", "going", "great", "hello", "hopin", "later",
        "maybe", "might", "morni", "night", "other", "reall", "right", "shoul", "since",
        "somet", "still", "their", "there", "these", "thing", "think", "those", "today",
        "tomor", "tonig", "which", "while", "would", "yeste", "yours",
    ]

    /// The gate a word-match cannot be: a second look at the same evidence, from a
    /// model told its job is to refuse.
    ///
    /// Silence and confusion both mean PASS. The cost of a wrong refusal is a dull
    /// notification; the cost of a wrong approval is him learning he cannot trust
    /// the thing that claims to remember him. Those are not the same price.
    private static func critique(
        _ line: String,
        against evidence: [String],
        complete: @escaping @Sendable (String) async throws -> String
    ) async throws -> Bool {
        let prompt = """
            Here is everything I actually know about a man:

            \(evidence.map { "- \($0)" }.joined(separator: "\n"))

            I am about to send him this notification, unprompted:

            "\(line)"

            Your only job is to stop me from embarrassing myself. Answer KEEP only if \
            every one of these is true:

            1. TRUE — every concrete detail in it (a name, a project, an activity, a \
            number, a date) appears in the list above. Nothing invented, nothing \
            sharpened from vague to precise, nothing guessed.
            2. SPECIFIC — it is about *him*. If the same line would land fine on a \
            stranger, it fails.
            3. COMMITTED — it names one thing. If it offers alternatives ("the X or \
            the Y?"), or gestures at something without naming it ("that task", "your \
            project", "the thing you mentioned"), it fails.

            If any of them is false, or you are unsure, answer PASS.

            Answer with one word: KEEP or PASS.
            """

        let verdict = try await complete(prompt)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .uppercased()
        // Anything that is not an unambiguous KEEP is a PASS — including an empty
        // answer, an essay, or a KEEP with a "but" attached to it.
        return verdict.hasPrefix("KEEP") && !verdict.contains("PASS")
    }
}
