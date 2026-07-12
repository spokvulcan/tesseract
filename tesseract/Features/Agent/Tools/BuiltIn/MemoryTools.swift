//
//  MemoryTools.swift
//  tesseract
//
//  The agent's two hands on its own memory (ADR-0035 §6, §9).
//
//  These replace the `memories.md` era, in which "remembering" meant the model
//  reading a markdown file, appending a bullet, and writing it back — three tool
//  calls, no provenance, no lifecycle, and a file that only ever grew.
//
//  Deliberately only two, and deliberately asymmetric in power:
//
//  - `remember` writes **one atomic claim in the agent's voice** — the owner is
//    "he", the assistant is "I" — marked STATED. It is the single exception to
//    "no memory formation on the hot path", and it earns that exception by not
//    being a heuristic: the owner *asked*.
//  - `recall` reads, and is the agent's way into the part of memory that
//    automatic injection will not show it — the retired tail and the superseded.
//
//  There is no tool to *edit* or *delete* a memory, and that absence is the
//  design. Revision is sleep's job, under prediction-error gating, with the
//  episode kept verbatim; deletion is the owner's alone. An agent that could
//  rewrite its own beliefs inline could talk itself into anything, and nothing
//  in the store would remember that it had.
//

import Foundation

nonisolated struct MemoryToolError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - remember

nonisolated func createRememberTool(memory: MemoryEngine) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "remember",
        label: "remember",
        description: """
            Commit one thing to long-term memory about the person you are talking to. \
            Use it when they tell you something about themselves worth keeping — a \
            preference, a fact, a standing instruction, an event that matters — or when \
            they explicitly ask you to remember something.

            This is YOUR memory of HIM. Write ONE self-contained claim per call, about \
            him in the third person, as you would want to recall it months from now with \
            no other context: "He prefers terse answers when he is debugging", not \
            "prefers terse". Never store his words in his own voice: he says "I like to \
            eat apples in the morning", you store "He likes to eat apples in the \
            morning." In a memory, "I" always means you, the assistant — "He gave me the \
            nickname Pelican" — and a standing instruction is anchored on him: "He wants \
            me to answer briefly when he is debugging." If they told you three things, \
            call this three times. Do not use it for anything you merely inferred \
            from their behaviour — that is a job for consolidation, not for you, and a \
            guess recorded here would be indistinguishable from something they said.

            Everything said in this conversation is already being recorded; you do not \
            need this tool to make the conversation memorable. Use it only for what should \
            outlive the conversation.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "text": PropertySchema(
                    type: "string",
                    description:
                        "The claim, self-contained, about him in the third person "
                        + "(\"I\" means you, the assistant). One claim only."
                ),
                "kind": PropertySchema(
                    type: "string",
                    description:
                        "belief (a stable fact or preference), event (a thing that happened at a time), directive (a standing instruction to follow), or pattern (a recurring regularity). Default: belief."
                ),
            ],
            required: ["text"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let text = ToolArgExtractor.string(argsJSON, key: "text"),
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw MemoryToolError(message: "remember requires non-empty 'text'")
            }
            let kind =
                MemoryKind(rawValue: ToolArgExtractor.string(argsJSON, key: "kind") ?? "belief")
                ?? .belief

            guard let stored = await memory.remember(text, kind: kind) else {
                throw MemoryToolError(
                    message: "Memory is off, or the store could not be written.")
            }
            return .text("Remembered: \(stored.text)")
        }
    )
}

// MARK: - recall

nonisolated func createRecallTool(memory: MemoryEngine) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "recall",
        label: "recall",
        description: """
            Search your long-term memory of this person. Relevant memories are already \
            put in front of you automatically at the start of a turn — so reach for this \
            only when you need something that is NOT there: an old detail, something you \
            half-remember, or a direct question about what you know ("what do you know \
            about me?", "have I mentioned my sister?").

            Unlike the automatic recall, this searches everything: your distilled \
            beliefs — including ones that have gone quiet and ones that were later \
            replaced, plainly marked as what you used to think — and the raw record of \
            past conversations, so it also finds things said recently that have not yet \
            been distilled into a belief.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "query": PropertySchema(
                    type: "string",
                    description: "What you are trying to remember."
                ),
                "limit": PropertySchema(
                    type: "integer",
                    description: "How many memories to return (default 10)."
                ),
            ],
            required: ["query"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let query = ToolArgExtractor.string(argsJSON, key: "query"),
                !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw MemoryToolError(message: "recall requires a non-empty 'query'")
            }
            let limit = min(ToolArgExtractor.int(argsJSON, key: "limit") ?? 10, 30)

            let (memories, episodes) = await memory.searchEverything(query: query, limit: limit)
            guard !memories.isEmpty || !episodes.isEmpty else {
                return .text("Nothing in memory about that.")
            }
            var lines = memories.map(MemoryToolFormatter.line)
            if !episodes.isEmpty {
                lines.append("")
                lines.append("From past conversations (not yet distilled into beliefs):")
                lines.append(contentsOf: episodes.map(MemoryToolFormatter.line))
            }
            return .text(lines.joined(separator: "\n"))
        }
    )
}

// MARK: - Formatting

private nonisolated enum MemoryToolFormatter {

    /// One memory, one line. Provenance and status are never omitted: a model
    /// that cannot tell testimony from inference, or a live belief from a
    /// retracted one, will defend all four with equal confidence.
    static func line(_ hit: ScoredMemory) -> String {
        var flags: [String] = []
        if hit.memory.provenance == .inferred { flags.append("inferred") }
        switch hit.memory.status {
        case .live: break
        case .contested: flags.append("disputed by him — do not rely on it")
        case .superseded: flags.append("no longer true; I have replaced this")
        }
        if hit.memory.tier == .cold { flags.append("long dormant") }

        let suffix = flags.isEmpty ? "" : " [\(flags.joined(separator: "; "))]"
        return "- \(hit.memory.text)\(suffix)"
    }

    /// An episode line — raw testimony, so it is dated and quoted rather than
    /// asserted: what was *said*, not what I believe.
    static func line(_ hit: ScoredEpisode) -> String {
        let date = hit.episode.occurredAt.formatted(date: .abbreviated, time: .omitted)
        var text = hit.episode.text
        if text.count > 240 { text = String(text.prefix(240)) + "…" }
        return "- (\(date)) \"\(text)\""
    }
}
