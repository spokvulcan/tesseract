//
//  ProfileTools.swift
//  tesseract
//
//  Memory the owner controls: `remember` saves a Profile fact directly (an
//  explicit ask counts as approval), `forget` removes one, and `recall`
//  searches the Profile and past conversations and answers with dated
//  snippets. Nothing is ever injected into a chat on its own.
//

import Foundation

/// Reranks recall candidates by meaning, when the embedder is on disk.
typealias RecallEmbedding = @Sendable ([String]) async -> [[Float]]?

@MainActor
func createProfileTools(
    profile: ProfileStore, index: RecallIndex, embed: RecallEmbedding? = nil,
    now: @escaping @MainActor () -> Date = Date.init
) -> [AgentToolDefinition] {
    [
        AgentToolDefinition(
            name: "remember",
            label: "remember",
            description: """
                Save one lasting fact about the owner to their Profile when they ask you to \
                remember something ("remember that I…"). Write it as a short third-person fact: \
                "Goes to the gym on Mondays and Thursdays." One fact per call. Never save what \
                they didn't ask you to remember.
                """,
            parameterSchema: JSONSchema(
                type: "object",
                properties: [
                    "fact": PropertySchema(
                        type: "string", description: "The fact, third person, one sentence."),
                    "area": PropertySchema(
                        type: "string", description: "Optional Area it belongs to."),
                ],
                required: ["fact"]),
            execute: { _, args, _, _ in
                guard let fact = ToolArgExtractor.string(args, key: "fact") else {
                    throw AgendaError.invalid("remember needs the fact.")
                }
                let area = ToolArgExtractor.string(args, key: "area")
                return try await MainActor.run {
                    guard let saved = profile.add(fact, area: area, source: .chat) else {
                        throw AgendaError.invalid("There was nothing to remember.")
                    }
                    return .text("Remembered: \(saved.text)")
                }
            }),
        AgentToolDefinition(
            name: "forget",
            label: "forget",
            description: """
                Remove a fact from the owner's Profile when they say it's wrong or ask you to \
                forget it. Give the fact's id (from recall) or its words.
                """,
            parameterSchema: JSONSchema(
                type: "object",
                properties: [
                    "fact": PropertySchema(
                        type: "string", description: "The fact's id, or its words.")
                ],
                required: ["fact"]),
            execute: { _, args, _, _ in
                guard let fact = ToolArgExtractor.string(args, key: "fact") else {
                    throw AgendaError.invalid("forget needs the fact.")
                }
                return try await MainActor.run {
                    guard let target = profile.factToForget(fact) else {
                        // Not one fact for sure: name the candidates, delete nothing.
                        let near = profile.search(fact).prefix(3)
                        guard !near.isEmpty else {
                            throw AgendaError.notFound("a Profile fact matching “\(fact)”")
                        }
                        throw AgendaError.invalid(
                            "That could be more than one fact, or none exactly. Ask which, then forget it by its id: "
                                + near.map { "\($0.id): \($0.text)" }.joined(separator: "; "))
                    }
                    guard let removed = profile.delete(target.id) else {
                        throw AgendaError.notFound("a Profile fact matching “\(fact)”")
                    }
                    return .text("Forgotten: \(removed.text)")
                }
            }),
        AgentToolDefinition(
            name: "recall",
            label: "recall",
            description: """
                Search what the owner told you: their Profile and past conversations. Use it when \
                they ask about something from before ("what did I say about X last week?") or when \
                you need a detail you don't have. Answers with dated snippets.
                """,
            parameterSchema: JSONSchema(
                type: "object",
                properties: [
                    "query": PropertySchema(type: "string", description: "What to look for."),
                    "days": PropertySchema(type: "integer", description: "Only the last N days."),
                ],
                required: ["query"]),
            execute: { _, args, _, _ in
                guard let query = ToolArgExtractor.string(args, key: "query") else {
                    throw AgendaError.invalid("recall needs a query.")
                }
                let days = ToolArgExtractor.int(args, key: "days")
                let (facts, current): ([ProfileFact], Date) = await MainActor.run {
                    (profile.search(query), now())
                }
                var hits = await index.search(query, days: days, now: current)
                if let embed, hits.count > 1 {
                    hits = await Recall.rerank(hits, query: query, embed: embed)
                }
                return .text(
                    Recall.render(
                        query: query, facts: facts, hits: Array(hits.prefix(8)), now: current))
            }),
    ]
}

nonisolated enum Recall {

    /// Blend full-text rank with meaning: cosine to the query, then bm25 as
    /// the tiebreak.
    static func rerank(_ hits: [RecallHit], query: String, embed: RecallEmbedding) async
        -> [RecallHit]
    {
        guard let vectors = await embed([query] + hits.map(\.snippet)),
            vectors.count == hits.count + 1
        else { return hits }
        let q = vectors[0]
        return zip(hits, vectors.dropFirst())
            .map { hit, vector in (hit, cosine(q, vector)) }
            .sorted { ($0.1, $0.0.score) > ($1.1, $1.0.score) }
            .map(\.0)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return na > 0 && nb > 0 ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    static func render(query: String, facts: [ProfileFact], hits: [RecallHit], now: Date) -> String
    {
        var lines: [String] = []
        if !facts.isEmpty {
            lines.append("From their Profile:")
            lines += facts.prefix(8).map { "- \($0.text) [id \($0.id)]" }
        }
        if !hits.isEmpty {
            if !lines.isEmpty { lines.append("") }
            lines.append("From past conversations:")
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "EEE d MMM yyyy"
            for hit in hits {
                let who = hit.role == "owner" ? "they said" : "you said"
                let title = hit.title.isEmpty ? "" : " “\(hit.title.prefix(60))”"
                lines.append("- \(formatter.string(from: hit.at))\(title) — \(who): \(hit.snippet)")
            }
        }
        return lines.isEmpty ? "Nothing found for “\(query)”." : lines.joined(separator: "\n")
    }
}
