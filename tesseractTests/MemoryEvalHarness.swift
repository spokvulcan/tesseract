//
//  MemoryEvalHarness.swift
//  tesseractTests
//
//  The memory evaluation harness (ADR-0035 §10, #324).
//
//  The scoreboard exists before the mechanisms. In the continual-learning
//  record, brain-inspired methods keep LOSING to just keeping the data
//  (sleep-replay 48.5% vs plain rehearsal on 0.75% of the data 79.9%; EWC 0.087
//  vs an undefended MLP's 0.085). So the baseline — "store everything, retrieve
//  well" — is built FIRST, and every mechanism must beat or match it to stay.
//
//  The integrity rule for this file: **it never reimplements the thing it
//  measures.** Relevance comes from `MemoryEngine.relevance`, the score from
//  `MemoryLifecycle.retrievalScore`, the tiers from `MemoryLifecycle.sweepTier`,
//  the keyword half from `MemoryStore.keywordScores`. A harness that scores with
//  its own copy of the math grades a system nobody ships.
//
//  The two arms differ in EXACTLY one thing — the ranker:
//
//    baseline   score = relevance                                (no lifecycle)
//    lifecycle  score = relevance · need · entrenchment · interference
//
//  Same records, same relevance, same pool, same budget. Whatever separates
//  them is the mechanism, and nothing else.
//
//  Probes are derived MECHANICALLY from the corpus's own structure — no LLM, no
//  hand-written answer key, so the whole eval is reproducible and needs no
//  model to define what "correct" means.
//

import Foundation

@testable import Tesseract_Agent

// MARK: - Corpus value types

nonisolated enum EvalRole: String, Sendable {
    case user
    case assistant
}

/// One thing that was actually said. `episodeID` is assigned at load so probes
/// and the backfilled store agree on identity.
nonisolated struct EvalTurn: Sendable {
    let episodeID: UUID
    let index: Int
    let role: EvalRole
    let text: String
    let occurredAt: Date
}

nonisolated struct EvalConversation: Sendable {
    let id: String
    let title: String
    let createdAt: Date
    let turns: [EvalTurn]
}

// MARK: - Probes

/// A retrieval probe: a cue, the moment it was uttered, and the episodes that
/// SHOULD come back for it.
///
/// `at` is load-bearing. The candidate pool for a probe is every record born
/// strictly BEFORE `at` — a memory system at turn *j* cannot retrieve turn
/// *j+1*. Evaluating any other way silently grades a time machine, and it is
/// also what makes the lifecycle's need-probability term meaningful at all
/// (`now` is the cue's own timestamp, not wall-clock).
nonisolated struct EvalProbe: Sendable {
    nonisolated enum Family: String, Sendable, CaseIterable {
        /// Cue = a later user turn. Gold = the earlier turns of the same
        /// conversation. "Bring back the context this turn depends on."
        case sessionContinuation
        /// Cue = a later user turn that re-uses a term the conversation
        /// introduced earlier and that is RARE across the whole corpus. Gold =
        /// the earlier turn(s) that introduced it. This is the "rare-but-
        /// critical fact" family — the sharp, few-gold probe.
        case referenceBack
        /// Cue = the conversation's own title. Gold = that conversation's
        /// episodes. "Find the right conversation from a summary."
        case titleTopic
    }

    let family: Family
    let conversationID: String
    let cue: String
    let at: Date
    let goldEpisodeIDs: Set<UUID>
    /// The cue's own episode, which must never be retrievable for itself.
    let cueEpisodeID: UUID?
    /// Diagnostics: the rare terms that made a `referenceBack` probe.
    let anchors: [String]
}

// MARK: - Metrics

nonisolated struct RetrievalMetrics: Sendable {
    var probes = 0
    var recallAt1 = 0.0
    var recallAt3 = 0.0
    var recallAt5 = 0.0
    var recallAt10 = 0.0
    var mrr = 0.0
    /// Mean pool size — how hard the task actually was.
    var meanPool = 0.0

    mutating func add(rankedGoldHits ranks: [Int], goldCount: Int, poolSize: Int) {
        guard goldCount > 0 else { return }
        probes += 1
        let hits = Set(ranks)
        func recall(_ k: Int) -> Double {
            Double(hits.filter { $0 < k }.count) / Double(goldCount)
        }
        recallAt1 += recall(1)
        recallAt3 += recall(3)
        recallAt5 += recall(5)
        recallAt10 += recall(10)
        if let best = ranks.min() { mrr += 1.0 / Double(best + 1) }
        meanPool += Double(poolSize)
    }

    /// Turn the running sums into means.
    var averaged: RetrievalMetrics {
        guard probes > 0 else { return self }
        var m = self
        let n = Double(probes)
        m.recallAt1 /= n
        m.recallAt3 /= n
        m.recallAt5 /= n
        m.recallAt10 /= n
        m.mrr /= n
        m.meanPool /= n
        return m
    }

    func row(_ label: String) -> String {
        String(
            format: "  %-26@  n=%3d  R@1 %.3f  R@3 %.3f  R@5 %.3f  R@10 %.3f  MRR %.3f  pool %.0f",
            label as NSString, probes, recallAt1, recallAt3, recallAt5, recallAt10, mrr, meanPool)
    }
}

// MARK: - The corpus loader

/// Loads the owner's real conversations off disk.
///
/// Robust to every shape variant measured in the corpus (2026-07-12):
///   • `index.json` is a top-level ARRAY, not a conversation — it must be skipped.
///   • user `payload.content` is a String.
///   • assistant `payload.content` is an array of parts: `{type:"text"}`,
///     `{type:"thinking"}`, `{type:"toolCall"}`.
///   • `tool_result` content is an array too, and may carry `{data, mimeType}`
///     image parts.
///   • `payload.timestamp` is a Double in APPLE reference-date epoch
///     (`timeIntervalSinceReferenceDate`), NOT Unix epoch. Getting this wrong
///     shifts every episode by 31 years and silently destroys the temporal pool.
///   • `createdAt` / `updatedAt` are ISO-8601 strings.
nonisolated enum MemoryEvalCorpus: Sendable {

    /// Application Support, resolved the way the app itself resolves it
    /// (`AgentConversationStore.swift:47`, `DependencyContainer.swift:25`) —
    /// always through `.applicationSupportDirectory`, never a literal `~`, so it
    /// stays whatever the app would use.
    ///
    /// Post-#381 the agent is non-sandboxed, so this resolves to the real
    /// `~/Library/Application Support`, not the retired per-app container. That
    /// path starts near-empty on a fresh install, which is exactly why
    /// `minimumEvalConversations` gates the suites below rather than an
    /// existence check.
    static var applicationSupport: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
    }

    /// The owner's live corpus — the same directory the agent writes to.
    static var defaultDirectory: URL {
        applicationSupport.appendingPathComponent(
            "Tesseract Agent/agent/conversations", isDirectory: true)
    }

    static var directory: URL {
        if let override = ProcessInfo.processInfo.environment["TESSERACT_MEMORY_CORPUS"] {
            return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        }
        return defaultDirectory
    }

    /// The floor for an eval-scale corpus. These suites measure retrieval
    /// quality (recall@10, referenceBack probes, `episodes.count > 50`) and only
    /// mean anything against a real personal corpus. Post-#381 the agent left
    /// the sandbox, so this resolves to `~/Library/Application Support` — a
    /// fresh install with a handful of conversations, not the old container's
    /// full history. A ≥1 gate fails open on that fresh corpus and the evals run
    /// against nothing (the exact failure this replaces); requiring an eval-
    /// scale count skips cleanly until the corpus regrows, matching the `> 50`
    /// episode assertion the suites already make.
    static let minimumEvalConversations = 50

    /// The gate. CI and a fresh (post-#381) install have no eval-scale corpus,
    /// so every suite that needs one skips rather than fails. Existence is NOT
    /// the test: the test host boots the app, and `AgentConversationStore.init`
    /// creates the conversations directory (and, on a fresh install, one live
    /// conversation) — so an existence gate fails open and the suites run
    /// against nothing (four expectation failures and an `Index out of range`
    /// crash, run 29198624167; then the three #381 fresh-start failures).
    static var isAvailable: Bool {
        hasEvalCorpus(at: directory)
    }

    /// An eval-scale corpus: at least `minimumEvalConversations` conversation
    /// files, by the same filter `load` applies — `index.json` is the manifest,
    /// not a conversation.
    static func hasEvalCorpus(at directory: URL) -> Bool {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        let conversations = contents.filter {
            $0.pathExtension == "json" && $0.lastPathComponent != "index.json"
        }
        return conversations.count >= minimumEvalConversations
    }

    /// The embedder (`Qwen3-Embedding-0.6B-4bit-DWQ`, 1024-dim). Optional: with
    /// it, relevance is hybrid dense ⊕ BM25 exactly as ADR-0035 §5 specifies;
    /// without it, the harness degrades to keyword-only — the same fail-open
    /// path `MemoryEngine` itself takes — and still measures the arms fairly,
    /// because both arms share whatever relevance is available.
    /// Where `ModelDownloadManager` puts it: the HF repo id with `/` → `_`
    /// (`ModelDefinition.cacheSubdirectory`).
    static var embedderDirectory: URL? {
        let url: URL
        if let override = ProcessInfo.processInfo.environment["TESSERACT_MEMORY_EMBEDDER"] {
            url = URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        } else {
            url =
                applicationSupport
                .appendingPathComponent("models", isDirectory: true)
                .appendingPathComponent(
                    "mlx-community_Qwen3-Embedding-0.6B-4bit-DWQ", isDirectory: true)
        }
        return FileManager.default.fileExists(
            atPath: url.appendingPathComponent("config.json").path) ? url : nil
    }

    static var isEmbedderAvailable: Bool { embedderDirectory != nil }

    /// Parse every conversation in `directory`, chronologically.
    ///
    /// `JSONSerialization`, not `Codable`: the payload is heterogeneous by
    /// design (three message types, four part types, a stray top-level array),
    /// and a decoder that throws on the first surprise would make the loader
    /// brittle against exactly the variation this corpus has.
    static func load(from directory: URL = MemoryEvalCorpus.directory) throws -> [EvalConversation]
    {
        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            // The manifest, not a conversation: a top-level ARRAY. Parsing it as
            // one would throw; skipping it by name is the honest fix.
            .filter { $0.lastPathComponent != "index.json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let iso = ISO8601DateFormatter()
        var out: [EvalConversation] = []

        for file in files {
            guard let data = try? Data(contentsOf: file),
                let root = try? JSONSerialization.jsonObject(with: data),
                let object = root as? [String: Any],
                let id = object["id"] as? String,
                let messages = object["messages"] as? [[String: Any]]
            else { continue }

            var turns: [EvalTurn] = []
            for message in messages {
                guard let type = message["type"] as? String,
                    let payload = message["payload"] as? [String: Any]
                else { continue }

                // Apple reference-date epoch. Measured, not assumed.
                let occurredAt = (payload["timestamp"] as? Double)
                    .map { Date(timeIntervalSinceReferenceDate: $0) }

                let role: EvalRole
                let text: String
                switch type {
                case "user":
                    // The one shape where content is a bare String.
                    guard let content = payload["content"] as? String else { continue }
                    role = .user
                    // The same door the app's capture goes through: a skill fire
                    // puts the skill's whole body in the user message, and those
                    // are the app's words, not his. Grading a retriever on its
                    // ability to find its own boilerplate would be a fine way to
                    // score well and learn nothing.
                    guard let spoken = MemorySpeech.spoken(content) else { continue }
                    text = spoken
                case "assistant":
                    // Parts. `thinking` is the model's private monologue and
                    // `toolCall` is machine chatter — neither is "something that
                    // happened" in the owner's life, so only `text` survives.
                    guard let parts = payload["content"] as? [[String: Any]] else { continue }
                    role = .assistant
                    text =
                        parts
                        .filter { ($0["type"] as? String) == "text" }
                        .compactMap { $0["text"] as? String }
                        .joined(separator: " ")
                default:
                    // `tool_result` — file dumps and command output. This is the
                    // bulk of the corpus's 42 MB and none of its meaning; an
                    // Episode is a turn, not a tool's stdout (ADR-0035 §1).
                    continue
                }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, let occurredAt else { continue }
                turns.append(
                    EvalTurn(
                        episodeID: UUID(), index: turns.count, role: role,
                        text: trimmed, occurredAt: occurredAt))
            }

            guard !turns.isEmpty else { continue }
            let createdAt =
                (object["createdAt"] as? String).flatMap { iso.date(from: $0) }
                ?? turns[0].occurredAt
            out.append(
                EvalConversation(
                    id: id, title: (object["title"] as? String) ?? "", createdAt: createdAt,
                    turns: turns))
        }

        return out.sorted { $0.createdAt < $1.createdAt }
    }
}

// MARK: - Probe generation

nonisolated enum MemoryEvalProbes: Sendable {

    /// A term is *distinctive* when it appears in at most this many of the
    /// corpus's conversations. Purely corpus-derived — no hand-built stoplist,
    /// which is what keeps the probe set reproducible.
    static let rareDocumentFrequency = 3
    static let minimumTermLength = 4

    static func tokens(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= minimumTermLength }
    }

    /// Document frequency over CONVERSATIONS (not turns) — the unit a corpus
    /// term is rare or common *in*.
    static func documentFrequency(_ conversations: [EvalConversation]) -> [String: Int] {
        var df: [String: Int] = [:]
        for conversation in conversations {
            var seen = Set<String>()
            for turn in conversation.turns { seen.formUnion(tokens(turn.text)) }
            for term in seen { df[term, default: 0] += 1 }
        }
        return df
    }

    static func generate(_ conversations: [EvalConversation]) -> [EvalProbe] {
        let df = documentFrequency(conversations)
        var probes: [EvalProbe] = []

        for conversation in conversations {
            // Where each distinctive term first shows up in this conversation.
            var firstIndex: [String: Int] = [:]
            for turn in conversation.turns {
                for term in Set(tokens(turn.text))
                where df[term, default: 0] <= rareDocumentFrequency {
                    if firstIndex[term] == nil { firstIndex[term] = turn.index }
                }
            }

            for turn in conversation.turns where turn.role == .user && turn.index > 0 {
                let earlier = conversation.turns.filter { $0.index < turn.index }
                guard !earlier.isEmpty else { continue }

                // — sessionContinuation: gold = everything said earlier here.
                probes.append(
                    EvalProbe(
                        family: .sessionContinuation,
                        conversationID: conversation.id,
                        cue: turn.text,
                        at: turn.occurredAt,
                        goldEpisodeIDs: Set(earlier.map(\.episodeID)),
                        cueEpisodeID: turn.episodeID,
                        anchors: []))

                // — referenceBack: this turn re-uses a rare term introduced
                //   earlier. Gold = only the earlier turns that carry it.
                let anchors = Set(tokens(turn.text)).filter { term in
                    df[term, default: 0] <= rareDocumentFrequency
                        && (firstIndex[term] ?? Int.max) < turn.index
                }
                if !anchors.isEmpty {
                    let gold = earlier.filter { !Set(tokens($0.text)).isDisjoint(with: anchors) }
                    if !gold.isEmpty {
                        probes.append(
                            EvalProbe(
                                family: .referenceBack,
                                conversationID: conversation.id,
                                cue: turn.text,
                                at: turn.occurredAt,
                                goldEpisodeIDs: Set(gold.map(\.episodeID)),
                                cueEpisodeID: turn.episodeID,
                                anchors: anchors.sorted()))
                    }
                }
            }

            // — titleTopic: the title is a summary written independently of the
            //   retriever, which is what makes it a fair cue. Probed as of the
            //   conversation's END, so its own turns are all in the pool.
            let title = conversation.title.trimmingCharacters(in: .whitespacesAndNewlines)
            if !title.isEmpty, let last = conversation.turns.last {
                probes.append(
                    EvalProbe(
                        family: .titleTopic,
                        conversationID: conversation.id,
                        cue: title,
                        at: last.occurredAt.addingTimeInterval(1),
                        goldEpisodeIDs: Set(conversation.turns.map(\.episodeID)),
                        cueEpisodeID: nil,
                        anchors: []))
            }
        }

        return probes
    }
}

// MARK: - The harness

/// A backfilled store plus the machinery to rank against it.
///
/// One `MemoryRecord` per `Episode`, verbatim — that IS "store everything". The
/// baseline arm then ranks those records by relevance alone, and the lifecycle
/// arm ranks the SAME records with `MemoryLifecycle.retrievalScore`. Holding the
/// record set identical is what makes the difference attributable to the ranker.
@MainActor
final class MemoryEvalHarness {

    let store: MemoryStore
    let directory: URL
    let conversations: [EvalConversation]
    let probes: [EvalProbe]

    /// episode → the memory backfilled from it, and back.
    private(set) var memoryIDByEpisode: [UUID: UUID] = [:]
    private(set) var episodeIDByMemory: [UUID: UUID] = [:]
    /// The live record set, mutated by the usage simulation.
    private(set) var memories: [UUID: MemoryRecord] = [:]

    private var vectorByMemoryID: [UUID: [Float]] = [:]
    private var cueVectors: [String: [Float]] = [:]
    private(set) var isDense = false

    init(conversations: [EvalConversation], probes: [EvalProbe]) throws {
        self.conversations = conversations
        self.probes = probes
        self.directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-eval-\(UUID().uuidString)", isDirectory: true)
        self.store = try MemoryStore(directory: directory)
    }

    deinit {
        let directory = self.directory
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: Backfill

    /// Store everything: every turn becomes an episode AND a memory, with a
    /// dense vector when one is available.
    ///
    /// `vectors` is a text → embedding map owned by `MemoryEvalFixture`, so the
    /// 335 MB embedder is loaded and run exactly once across the whole suite
    /// while every test still gets a pristine store.
    func backfill(vectors: [String: [Float]], cueVectors: [String: [Float]] = [:]) async throws {
        let turns = conversations.flatMap { conversation in
            conversation.turns.map { (conversation.id, $0) }
        }.sorted { $0.1.occurredAt < $1.1.occurredAt }

        isDense = !vectors.isEmpty

        for entry in turns {
            let (conversationID, turn) = entry
            let vector: [Float]? = vectors[turn.text]

            let episode = Episode(
                id: turn.episodeID, source: .backfill, conversationID: conversationID,
                occurredAt: turn.occurredAt, text: turn.text,
                meta: ["role": turn.role.rawValue])
            try await store.append(episode, embedding: vector)

            let memory = MemoryRecord(
                text: turn.text,
                kind: .belief,
                // The owner said it, or the assistant did. That is exactly the
                // STATED/INFERRED distinction (ADR-0035 §2).
                provenance: turn.role == .user ? .stated : .inferred,
                specificity: .specific,
                tier: .hot,
                sourceEpisodeIDs: [turn.episodeID],
                bornAt: turn.occurredAt)
            try await store.upsert(memory, embedding: vector)

            memoryIDByEpisode[turn.episodeID] = memory.id
            episodeIDByMemory[memory.id] = turn.episodeID
            memories[memory.id] = memory
            if let vector { vectorByMemoryID[memory.id] = vector }
        }

        for probe in probes {
            if let vector = cueVectors[probe.cue] { self.cueVectors[probe.cue] = vector }
        }
    }

    // MARK: Ranking

    nonisolated enum Arm: String, Sendable, CaseIterable {
        /// "Store everything, retrieve well." The thing to beat.
        case baseline
        /// relevance · need · entrenchment · interference, with `.cold` out of
        /// the default pool and `.core` unconditionally present.
        case lifecycle
        /// **The null hypothesis, and the most important arm in this file.**
        ///
        /// Need-probability ALONE — no relevance, no cue, no embedding, no
        /// keyword. It ranks purely by "how recently was this last useful",
        /// which at cold start is exactly "how recently was this said".
        ///
        /// It exists because the lifecycle's decay term IS a recency prior, and
        /// on a corpus this young it is the only term with any dynamic range:
        /// `need ∈ [0.86, 1.0]` across the entire 5-day span, while entrenchment
        /// and interference sit pinned at 1.0 until usage is graded. If this arm
        /// scores anywhere near the lifecycle, then the lifecycle is not beating
        /// the baseline at *memory* — it is beating it at *recency*, and `ORDER
        /// BY timestamp DESC` would do the same for free. That is precisely the
        /// failure the continual-learning record keeps rediscovering, so it gets
        /// measured rather than assumed away.
        case recencyOnly
    }

    /// Gold memory IDs for a probe.
    func gold(_ probe: EvalProbe) -> Set<UUID> {
        Set(probe.goldEpisodeIDs.compactMap { memoryIDByEpisode[$0] })
    }

    /// The candidate pool: every memory born strictly before the cue, minus the
    /// cue's own record. A memory system at turn *j* cannot see turn *j+1*.
    func pool(for probe: EvalProbe) -> [MemoryRecord] {
        let cueMemoryID = probe.cueEpisodeID.flatMap { memoryIDByEpisode[$0] }
        return memories.values.filter { memory in
            memory.bornAt < probe.at && memory.id != cueMemoryID
        }
    }

    /// Rank a pool for a cue, by one arm. Returns memory IDs, best first.
    ///
    /// Both arms call `MemoryEngine.relevance` — the shipped hybrid — so the
    /// relevance signal is identical and only the ranker differs.
    func rank(probe: EvalProbe, arm: Arm, pool: [MemoryRecord], limit: Int = 10) async throws
        -> [UUID]
    {
        guard !pool.isEmpty else { return [] }
        let keyword = try await store.keywordScores(
            query: probe.cue, in: .memory, limit: 1_000)
        let cueVector = cueVectors[probe.cue]

        var scored: [(UUID, Double)] = []
        var core: [(UUID, Double)] = []

        for memory in pool {
            let relevance = MemoryEngine.relevance(
                cueVector: cueVector, id: memory.id, vectorByID: vectorByMemoryID,
                keyword: keyword)

            switch arm {
            case .baseline:
                // No lifecycle. No tiers. Relevance, and nothing else.
                guard relevance > 0 else { continue }
                scored.append((memory.id, relevance))

            case .lifecycle:
                if memory.tier == .core {
                    // Promotion's concrete grant: unconditional presence. Core
                    // memories occupy slots whether or not they scored.
                    core.append((memory.id, relevance))
                    continue
                }
                // `.cold` is genuinely out of the default pool — that is what
                // retirement MEANS here. Reachable only by `memory_search` and
                // the ε-slot, neither of which is a default read.
                guard memory.tier != .cold else { continue }
                guard relevance > 0 else { continue }
                scored.append(
                    (
                        memory.id,
                        MemoryLifecycle.retrievalScore(
                            memory: memory, relevance: relevance, now: probe.at)
                    ))

            case .recencyOnly:
                // Relevance is deliberately DISCARDED. This arm knows nothing
                // about the cue.
                scored.append(
                    (
                        memory.id,
                        MemoryLifecycle.needProbability(of: memory, now: probe.at)
                    ))
            }
        }

        scored.sort { $0.1 == $1.1 ? $0.0.uuidString < $1.0.uuidString : $0.1 > $1.1 }
        core.sort { $0.1 == $1.1 ? $0.0.uuidString < $1.0.uuidString : $0.1 > $1.1 }
        return (core.map(\.0) + scored.map(\.0)).prefix(limit).map { $0 }
    }

    /// Run every probe of a family through one arm.
    ///
    /// `from` restricts to a held-out window: the usage simulation trains on
    /// everything before it, so evaluating after it is the only honest way to
    /// ask whether what the lifecycle learned generalizes.
    func evaluate(
        family: EvalProbe.Family, arm: Arm, limit: Int = 10, from: Date? = nil
    ) async throws -> RetrievalMetrics {
        var metrics = RetrievalMetrics()
        for probe in probes where probe.family == family {
            if let from, probe.at < from { continue }
            let goldIDs = gold(probe)
            guard !goldIDs.isEmpty else { continue }
            let candidates = pool(for: probe)
            // A probe whose gold is not even in the pool is unanswerable and
            // would silently drag every arm's recall down by the same amount.
            let reachable = goldIDs.filter { id in candidates.contains { $0.id == id } }
            guard !reachable.isEmpty else { continue }

            let ranked = try await rank(probe: probe, arm: arm, pool: candidates, limit: limit)
            let ranks = ranked.enumerated()
                .filter { reachable.contains($0.element) }
                .map(\.offset)
            metrics.add(
                rankedGoldHits: ranks, goldCount: reachable.count, poolSize: candidates.count)
        }
        return metrics.averaged
    }

    // MARK: Usage simulation (the lifecycle's only source of signal)

    /// Give the lifecycle what it is DESIGNED to run on: graded outcomes.
    ///
    /// The lifecycle's discriminating terms — entrenchment (`SS`) and stability
    /// (`S`) — are both flat at backfill (`SS = 0`, `S = 3` for every record).
    /// Until something is graded useful, `retrievalScore` reduces to
    /// `relevance · need`, i.e. relevance times a recency prior. So a cold-start
    /// comparison measures the lifecycle with its main mechanism switched off.
    /// This replays the probes in chronological order and grades them the way
    /// sleep would, so the held-out evaluation sees a lifecycle that has
    /// actually learned something.
    ///
    /// The grader is mechanical, not an LLM: a retrieved record that is gold for
    /// the probe was genuinely useful; one that is not was `.ignored`. That is
    /// the strongest *honest* grade available without a judge, and `.ignored`
    /// is correctly not a lapse — it touches neither `S` nor `SS`.
    @discardableResult
    func primeWithUsage(upTo cutoff: Date, budget: Int = 8) async throws -> Int {
        var graded = 0
        let training =
            probes
            .filter { $0.at < cutoff && $0.family != .titleTopic }
            .sorted { $0.at < $1.at }

        for probe in training {
            let goldIDs = gold(probe)
            guard !goldIDs.isEmpty else { continue }
            let candidates = pool(for: probe)
            guard !candidates.isEmpty else { continue }

            let ranked = try await rank(
                probe: probe, arm: .lifecycle, pool: candidates, limit: budget)

            for (rank, memoryID) in ranked.enumerated() {
                guard var memory = memories[memoryID] else { continue }
                // Seen is diagnostic ONLY — it must never drive the lifecycle,
                // or the system trains on its own retriever's beliefs.
                memory = MemoryLifecycle.markSeen(memory, now: probe.at)

                let grade: UseGrade
                if goldIDs.contains(memoryID) {
                    grade = rank == 0 ? .decisive : .used
                } else {
                    grade = .ignored
                }
                memory = MemoryLifecycle.applyGrade(grade, to: memory, now: probe.at)
                memories[memoryID] = memory
                try await store.upsert(memory)
                try await store.log([
                    RetrievalEvent(
                        memoryID: memoryID, episodeID: probe.cueEpisodeID ?? UUID(),
                        retrievedAt: probe.at, cue: probe.cue, grade: grade)
                ])
                if grade.isUseful { graded += 1 }
            }
        }
        return graded
    }

    /// The lifecycle sweep, exactly as sleep runs it.
    @discardableResult
    func sweepTiers(now: Date) async throws -> [MemoryTier: Int] {
        var counts: [MemoryTier: Int] = [:]
        let daysByMemory = try await store.distinctUsefulDaysByMemory()
        for (id, memory) in memories {
            let swept = MemoryLifecycle.sweepTier(
                memory, distinctUsefulDays: daysByMemory[id] ?? 0, now: now)
            memories[id] = swept
            try await store.upsert(swept)
            counts[swept.tier, default: 0] += 1
        }
        return counts
    }

    // MARK: Retirement (ADR-0035 §4, and the policy it rejected)

    /// The three retirement policies worth contrasting.
    ///
    /// `disuse` is the one the ADR **rejects** — the Law of Disuse, demolished
    /// in 1932: "a memory never used retires." It is included precisely so the
    /// ADR's guard can be shown to buy something. If `adr` and `disuse` lose the
    /// same facts, the guard is decoration.
    nonisolated enum RetirementPolicy: String, Sendable, CaseIterable {
        /// `MemoryLifecycle.shouldRetireToCold` — superseded, or surfaced ≥8
        /// times and never once useful, or need-probability finally decayed.
        /// Always conjoined with `SS < θ`, which is the allergy guarantee.
        case adr
        /// Never usefully used ⇒ cold. **No storage-strength guard.** This is
        /// what almost every shipped agent memory does.
        case disuse
        /// Keep the K most recently born, retire the rest. Pure recency.
        case recency
    }

    /// Apply a retirement policy. Returns how many records went cold.
    @discardableResult
    func retire(policy: RetirementPolicy, now: Date, keep: Int = 100) async throws -> Int {
        var retired = 0
        switch policy {
        case .adr:
            for (id, memory) in memories {
                guard MemoryLifecycle.shouldRetireToCold(memory, now: now) else { continue }
                var cold = memory
                cold.tier = .cold
                memories[id] = cold
                retired += 1
            }
        case .disuse:
            for (id, memory) in memories where memory.usefulUseCount == 0 {
                var cold = memory
                cold.tier = .cold
                memories[id] = cold
                retired += 1
            }
        case .recency:
            let survivors = Set(
                memories.values.sorted { $0.bornAt > $1.bornAt }.prefix(keep).map(\.id))
            for (id, memory) in memories where !survivors.contains(id) {
                var cold = memory
                cold.tier = .cold
                memories[id] = cold
                retired += 1
            }
        }
        for memory in memories.values { try await store.upsert(memory) }
        return retired
    }

    /// **Retirement recall-regret** (ADR-0035 §10).
    ///
    /// Retire, then probe for the facts that should have survived. Regret is the
    /// share of genuinely-needed, genuinely-reachable gold that the policy has
    /// pushed out of the default retrieval pool. A memory that was ever useful
    /// must never be lost — so for the ADR policy this number must be 0 on the
    /// ever-useful set, forever.
    func recallRegret(from: Date) -> (regret: Double, probes: Int, lostEverUseful: Int) {
        var total = 0.0
        var counted = 0
        var lostEverUseful = 0

        for probe in probes where probe.at >= from {
            let goldIDs = gold(probe)
            guard !goldIDs.isEmpty else { continue }
            let candidates = Set(pool(for: probe).map(\.id))
            let reachable = goldIDs.intersection(candidates)
            guard !reachable.isEmpty else { continue }

            let lost = reachable.filter { memories[$0]?.tier == .cold }
            // The allergy guarantee: an ever-useful memory going cold is a
            // FAILURE of the design, not a tuning question.
            lostEverUseful += lost.filter { (memories[$0]?.usefulUseCount ?? 0) > 0 }.count

            total += Double(lost.count) / Double(reachable.count)
            counted += 1
        }
        guard counted > 0 else { return (0, 0, 0) }
        return (total / Double(counted), counted, lostEverUseful)
    }

    // MARK: Promotion forecasting (ADR-0035 §10)

    /// The memories that EXIST at `cutoff` and turn out to be needed after it.
    ///
    /// The forecast set is restricted to records already born — a promotion
    /// decision cannot forecast a memory that does not exist yet, and scoring it
    /// as if it could would be the eval grading a time machine.
    func trulyUsefulLater(cutoff: Date) -> Set<UUID> {
        var out = Set<UUID>()
        for probe in probes where probe.at >= cutoff {
            for id in gold(probe) where (memories[id]?.bornAt ?? .distantFuture) < cutoff {
                out.insert(id)
            }
        }
        return out
    }

    /// Candidates a forecaster is allowed to rank: everything alive at `cutoff`.
    func forecastCandidates(cutoff: Date) -> [MemoryRecord] {
        memories.values.filter { $0.bornAt < cutoff }
    }

    /// Overwrite the in-memory record set (used to restore a pristine copy
    /// between retirement policies without re-embedding the corpus).
    func override(_ records: [UUID: MemoryRecord]) { memories = records }

    var snapshot: [UUID: MemoryRecord] { memories }
}

// MARK: - The shared fixture

/// Loads the corpus and the embedder ONCE for the whole eval, then hands each
/// test a pristine, backfilled store.
///
/// The embedder is 335 MB of GPU-resident weights; loading it per test would be
/// slow and pointless. The vectors it produces are pure functions of the text,
/// so they cache perfectly. Safe as shared mutable state only because the eval
/// suites are serialized (`-parallel-testing-enabled NO`, per `docs/testing.md`
/// — the scheme otherwise runs twin runners that collide).
@MainActor
enum MemoryEvalFixture {

    private static var conversationsCache: [EvalConversation]?
    private static var probesCache: [EvalProbe]?
    private static var vectorCache: (documents: [String: [Float]], cues: [String: [Float]])?
    private static var embedder: MemoryEmbedder?

    static func conversations() throws -> [EvalConversation] {
        if let conversationsCache { return conversationsCache }
        let loaded = try MemoryEvalCorpus.load()
        conversationsCache = loaded
        return loaded
    }

    static func probes() throws -> [EvalProbe] {
        if let probesCache { return probesCache }
        let generated = MemoryEvalProbes.generate(try conversations())
        probesCache = generated
        return generated
    }

    /// Document vectors for every turn text, and query vectors for every probe
    /// cue — two maps because the two sides embed differently (#332): cues
    /// carry the instruct prefix the way `MemoryEngine` embeds them, documents
    /// never do. A cue that is textually identical to a turn must NOT share
    /// its vector: the turn is a document, the cue is a question about it.
    ///
    /// Both empty when the embedder is not downloaded — the harness then runs
    /// keyword-only, which is the same fail-open path `MemoryEngine` takes.
    static func vectors() async throws -> (
        documents: [String: [Float]], cues: [String: [Float]]
    ) {
        if let vectorCache { return vectorCache }
        guard let directory = MemoryEvalCorpus.embedderDirectory else {
            vectorCache = ([:], [:])
            return ([:], [:])
        }
        let model = embedder ?? MemoryEmbedder()
        embedder = model
        do {
            try await model.load(from: directory)
        } catch {
            Log.memory.error("[eval] embedder load failed: \(error.localizedDescription)")
            vectorCache = ([:], [:])
            return ([:], [:])
        }

        let conversations = try conversations()
        let documents = Array(Set(conversations.flatMap { $0.turns.map(\.text) })).sorted()
        let cues = Array(Set(try probes().map(\.cue))).sorted()

        var documentVectors: [String: [Float]] = [:]
        // Chunked: one 691-text batch would pad to the longest sequence in it.
        for chunk in stride(from: 0, to: documents.count, by: 32) {
            let slice = Array(documents[chunk..<min(chunk + 32, documents.count)])
            let embedded = await model.embed(slice)
            guard embedded.count == slice.count else { continue }
            for (text, vector) in zip(slice, embedded) { documentVectors[text] = vector }
        }
        var cueVectors: [String: [Float]] = [:]
        for chunk in stride(from: 0, to: cues.count, by: 32) {
            let slice = Array(cues[chunk..<min(chunk + 32, cues.count)])
            let embedded = await model.embedQueries(slice)
            guard embedded.count == slice.count else { continue }
            for (text, vector) in zip(slice, embedded) { cueVectors[text] = vector }
        }
        vectorCache = (documentVectors, cueVectors)
        return (documentVectors, cueVectors)
    }

    /// A fresh store, backfilled. Never shared — the usage simulation mutates it.
    static func makeHarness() async throws -> MemoryEvalHarness {
        let harness = try MemoryEvalHarness(
            conversations: try conversations(), probes: try probes())
        let (documents, cues) = try await vectors()
        try await harness.backfill(vectors: documents, cueVectors: cues)
        return harness
    }
}

// MARK: - The report

/// The eval is worthless if nobody can read it. `Log.memory` is the sanctioned
/// channel, but `os.Logger` does not reach xcodebuild's stdout — so the report
/// also lands on disk, under the app's own debug path.
@MainActor
final class MemoryEvalReport {

    private var lines: [String] = []

    func section(_ title: String) {
        add("")
        add("── \(title) " + String(repeating: "─", count: max(0, 58 - title.count)))
    }

    func add(_ line: String) {
        lines.append(line)
        Log.memory.info("[eval] \(line)")
    }

    /// Written to `DebugPaths.root/memory-eval/`, which lives inside the app
    /// container the test host already runs in.
    @discardableResult
    func flush(name: String) -> URL? {
        let directory = DebugPaths.root.appendingPathComponent("memory-eval", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(name).txt")
        try? lines.joined(separator: "\n").appending("\n").write(
            to: url, atomically: true, encoding: .utf8)
        Log.memory.info("[eval] report written to \(url.path)")
        return url
    }
}
