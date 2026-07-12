//
//  MemoryEngine.swift
//  tesseract
//
//  The memory system's facade (ADR-0035). Everything else — the chat hooks,
//  the Companion, the agent tools, the Memory window, sleep — talks to this.
//
//  Shape follows `ProofreadPass` (ADR-0034): a `@MainActor @Observable` policy
//  object over injected closures, with the model work behind actors. Every
//  disk or model failure is logged and swallowed — **memory may never take the
//  primary flow down.** A turn that cannot be remembered is still a turn that
//  must be answered.
//

import Foundation

/// What retrieval put in front of the model, and the events it logged so sleep
/// can grade them later.
nonisolated struct RetrievedContext: Sendable {
    var core: [MemoryRecord] = []
    var recalled: [ScoredMemory] = []
    var episodes: [ScoredEpisode] = []
    var events: [RetrievalEvent] = []

    var isEmpty: Bool { core.isEmpty && recalled.isEmpty && episodes.isEmpty }
}

nonisolated struct MemoryStats: Sendable, Equatable {
    var episodes: Int = 0
    var memories: Int = 0
    var byTier: [MemoryTier: Int] = [:]
    var ungradedRetrievals: Int = 0
}

@MainActor
@Observable
final class MemoryEngine {

    let store: MemoryStore
    private let embedder: MemoryEmbedder

    private let isEnabled: @MainActor () -> Bool
    private let isDictationCaptureEnabled: @MainActor () -> Bool
    private let embedderDirectory: @MainActor () -> URL?

    private(set) var isEmbedderLoaded = false
    private(set) var stats = MemoryStats()

    init(
        store: MemoryStore,
        embedder: MemoryEmbedder,
        isEnabled: @escaping @MainActor () -> Bool,
        isDictationCaptureEnabled: @escaping @MainActor () -> Bool,
        embedderDirectory: @escaping @MainActor () -> URL?
    ) {
        self.store = store
        self.embedder = embedder
        self.isEnabled = isEnabled
        self.isDictationCaptureEnabled = isDictationCaptureEnabled
        self.embedderDirectory = embedderDirectory
    }

    // MARK: - Lifecycle

    /// Load the embedder if it is downloaded. Fail-open: without it, retrieval
    /// degrades to keyword-only rather than breaking.
    func prewarm() async {
        guard isEnabled(), let directory = embedderDirectory() else { return }
        do {
            try await embedder.load(from: directory)
            isEmbedderLoaded = await embedder.isLoaded
            await refreshStats()
        } catch {
            Log.memory.error("Embedder load failed: \(error.localizedDescription)")
            isEmbedderLoaded = false
        }
    }

    func offload() async {
        await embedder.unload()
        isEmbedderLoaded = false
    }

    func refreshStats() async {
        do {
            let episodes = try await store.episodeCount()
            let memories = try await store.memoryCount()
            let byTier = try await store.countsByTier()
            let ungraded = try await store.ungradedRetrievals(limit: 10_000).count
            stats = MemoryStats(
                episodes: episodes, memories: memories, byTier: byTier,
                ungradedRetrievals: ungraded)
        } catch {
            Log.memory.error("Stats refresh failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Embedding

    func embed(_ text: String) async -> [Float]? {
        guard isEmbedderLoaded else { return nil }
        return await embedder.embed(text)
    }

    func embed(_ texts: [String]) async -> [[Float]] {
        guard isEmbedderLoaded, !texts.isEmpty else { return [] }
        return await embedder.embed(texts)
    }

    // MARK: - The write path (ADR-0035 §6)

    /// Append an episode. **This is the entire hot-path cost of memory**: an
    /// insert plus one embedding (~3 ms). No LLM, no importance judgment —
    /// salience is decided in sleep, because at this moment the information
    /// that determines it has not arrived yet.
    ///
    /// Detached and non-blocking by construction: callers `Task { }` this and
    /// never await it on a turn boundary.
    @discardableResult
    func record(
        source: MemorySource,
        text: String,
        conversationID: String? = nil,
        occurredAt: Date = Date(),
        meta: [String: String] = [:]
    ) async -> Episode? {
        guard isEnabled() else { return nil }
        guard source != .dictation || isDictationCaptureEnabled() else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let episode = Episode(
            source: source, conversationID: conversationID, occurredAt: occurredAt,
            text: trimmed, meta: meta)
        let vector = await embed(trimmed)
        do {
            try await store.append(episode, embedding: vector)
            return episode
        } catch {
            Log.memory.error("Episode append failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Append many episodes at once — the cold-start backfill's write path.
    ///
    /// Batched because the embedder is dramatically better used that way: 334
    /// texts/sec in a batch against roughly one per 3 ms one at a time. Chunked
    /// because a single batch pads every sequence to the longest one in it, and
    /// one pasted logfile in the corpus would otherwise pad all 207.
    ///
    /// Returns the number appended. `INSERT OR IGNORE` upstream means an episode
    /// whose id is already present is silently skipped — which is what makes the
    /// whole backfill re-runnable.
    @discardableResult
    func append(_ episodes: [Episode], chunk: Int = 32) async -> Int {
        guard isEnabled(), !episodes.isEmpty else { return 0 }
        var appended = 0
        for slice in stride(from: 0, to: episodes.count, by: chunk).map({
            Array(episodes[$0..<min($0 + chunk, episodes.count)])
        }) {
            let vectors = await embed(slice.map(\.text))
            for (index, episode) in slice.enumerated() {
                do {
                    try await store.append(
                        episode, embedding: index < vectors.count ? vectors[index] : nil)
                    appended += 1
                } catch {
                    Log.memory.error("Episode append failed: \(error.localizedDescription)")
                }
            }
        }
        return appended
    }

    /// The owner said "remember this". The one deliberate exception to
    /// "no memory formation on the hot path" — an explicit instruction is not
    /// a heuristic, and it should not have to wait for the next sleep.
    @discardableResult
    func remember(
        _ text: String,
        kind: MemoryKind = .belief,
        sourceEpisodeIDs: [UUID] = [],
        now: Date = Date()
    ) async -> MemoryRecord? {
        guard isEnabled() else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let memory = MemoryRecord(
            text: trimmed,
            kind: kind,
            // The owner said it. That is the definition of STATED.
            provenance: .stated,
            specificity: .general,
            tier: .hot,
            sourceEpisodeIDs: sourceEpisodeIDs,
            bornAt: now)
        let vector = await embed(trimmed)
        do {
            try await store.upsert(memory, embedding: vector)
            try await store.appendJournal(
                JournalEntry(
                    at: now, mutation: .added, memoryID: memory.id,
                    detail: "The owner asked me to remember this.", after: trimmed))
            await refreshStats()
            Log.memory.info("Remembered on request: \(trimmed)")
            return memory
        } catch {
            Log.memory.error("Remember failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - The read path (ADR-0035 §5)

    /// Retrieve for a cue.
    ///
    /// The score is multiplicative — relevance × need-probability ×
    /// entrenchment × interference — so a superseded or irrelevant memory
    /// cannot be rescued by sheer age. Core-tier memories bypass scoring
    /// entirely: unconditional presence is what promotion *grants*.
    ///
    /// This **logs** the retrieval and does not grade it. The read path is the
    /// lifecycle's sensor; the judge runs in sleep.
    func retrieve(
        cue: String,
        forEpisode episodeID: UUID?,
        memoryBudget: Int = 8,
        episodeBudget: Int = 3,
        now: Date = Date()
    ) async -> RetrievedContext {
        guard isEnabled() else { return RetrievedContext() }

        do {
            let core = try await store.memories(tier: .core, status: .live)
            let all = try await store.allLiveMemories()
            // `.cold` is genuinely out of the default pool — that is what
            // retirement *means* here. It is reachable two ways, and only two:
            // the ε-exploration draw below, and the agent's explicit
            // `memory_search` tool.
            let candidates = all.filter { $0.tier != .core && $0.tier != .cold }

            let cueVector = await embed(cue)
            let vectors = try await store.embeddings(kind: "memory")
            let vectorByID = Dictionary(vectors) { first, _ in first }
            let keyword = try await store.keywordScores(query: cue, table: "memories")
            let affinity = try await store.cueAffinities(cue: cue)

            var scored: [ScoredMemory] = []
            for memory in candidates {
                let relevance = Self.relevance(
                    cueVector: cueVector, id: memory.id, vectorByID: vectorByID,
                    keyword: keyword)
                guard relevance > 0 else { continue }
                var score = MemoryLifecycle.retrievalScore(
                    memory: memory, relevance: relevance, now: now)
                // Retrieved-and-ignored suppression: small, per-cue, reversible
                // — never a deletion driver.
                score *= (affinity[memory.id] ?? 1.0)
                scored.append(
                    ScoredMemory(
                        memory: memory, score: score, relevance: relevance, isExploration: false))
            }
            scored.sort { $0.score > $1.score }

            // The ε-exploration slot. Not optional: without it, the
            // counterfactual is unobservable and the cold tail never returns.
            let exploreCount = MemoryLifecycle.explorationSlots(of: memoryBudget)
            let exploitCount = max(0, memoryBudget - exploreCount)
            var recalled = Array(scored.prefix(exploitCount))

            if exploreCount > 0 {
                let chosen = Set(recalled.map(\.memory.id))
                // Drawn from `all`, not `candidates` — this is the one path by
                // which a cold memory can come back on its own.
                let coldPool = all.filter {
                    ($0.tier == .warm || $0.tier == .cold) && !chosen.contains($0.id)
                }
                for memory in coldPool.shuffled().prefix(exploreCount) {
                    let relevance = Self.relevance(
                        cueVector: cueVector, id: memory.id, vectorByID: vectorByID,
                        keyword: keyword)
                    recalled.append(
                        ScoredMemory(
                            memory: memory,
                            score: MemoryLifecycle.retrievalScore(
                                memory: memory, relevance: relevance, now: now),
                            relevance: relevance,
                            isExploration: true))
                }
            }

            let episodes = try await retrieveEpisodes(
                cue: cue, cueVector: cueVector, budget: episodeBudget, excluding: episodeID)

            // Mark seen — diagnostic only. This must never touch the lifecycle.
            for item in recalled {
                let seen = MemoryLifecycle.markSeen(item.memory, now: now)
                try? await store.upsert(seen)
            }
            for memory in core {
                let seen = MemoryLifecycle.markSeen(memory, now: now)
                try? await store.upsert(seen)
            }

            var events: [RetrievalEvent] = []
            if let episodeID {
                events =
                    (core.map {
                        RetrievalEvent(
                            memoryID: $0.id, episodeID: episodeID, retrievedAt: now, cue: cue)
                    }
                        + recalled.map {
                            RetrievalEvent(
                                memoryID: $0.memory.id, episodeID: episodeID, retrievedAt: now,
                                cue: cue, isExploration: $0.isExploration)
                        })
                try await store.log(events)
            }

            return RetrievedContext(
                core: core, recalled: recalled, episodes: episodes, events: events)
        } catch {
            Log.memory.error("Retrieval failed: \(error.localizedDescription)")
            return RetrievedContext()
        }
    }

    /// Episodes are retrievable too — which is exactly what makes same-day
    /// recall work with no hot-path extraction. Something said at 10am is
    /// findable at 2pm, long before sleep has distilled it into a belief.
    private func retrieveEpisodes(
        cue: String, cueVector: [Float]?, budget: Int, excluding: UUID?
    ) async throws -> [ScoredEpisode] {
        guard budget > 0 else { return [] }
        let keyword = try await store.keywordScores(query: cue, table: "episodes", limit: 30)
        let vectors = try await store.embeddings(kind: "episode")
        let vectorByID = Dictionary(vectors) { first, _ in first }

        var candidateIDs = Set(keyword.keys)
        if let cueVector {
            let top =
                vectors
                .map { ($0.0, MemoryStore.cosine(cueVector, $0.1)) }
                .sorted { $0.1 > $1.1 }
                .prefix(30)
                .map(\.0)
            candidateIDs.formUnion(top)
        }
        candidateIDs.subtract([excluding].compactMap { $0 })

        var out: [ScoredEpisode] = []
        for id in candidateIDs {
            guard let episode = try await store.episode(id: id) else { continue }
            let relevance = Self.relevance(
                cueVector: cueVector, id: id, vectorByID: vectorByID, keyword: keyword)
            guard relevance > 0.2 else { continue }
            out.append(ScoredEpisode(episode: episode, relevance: relevance))
        }
        out.sort { $0.relevance > $1.relevance }
        return Array(out.prefix(budget))
    }

    /// Hybrid relevance: dense cosine fused with FTS5/BM25.
    ///
    /// Both signals are model-free and cheap. The keyword half is not
    /// decoration — it catches the proper nouns and rare tokens that dense
    /// embeddings smear together, which in a *personal* memory (names, places,
    /// project codenames) is most of what matters.
    static func relevance(
        cueVector: [Float]?, id: UUID, vectorByID: [UUID: [Float]], keyword: [UUID: Double]
    ) -> Double {
        let dense: Double
        if let cueVector, let v = vectorByID[id] {
            dense = max(0, MemoryStore.cosine(cueVector, v))
        } else {
            dense = 0
        }
        let sparse = keyword[id] ?? 0
        // When the embedder is unavailable, keyword carries the whole load.
        guard cueVector != nil else { return sparse }
        return 0.75 * dense + 0.25 * sparse
    }

    /// Deliberate search — the agent asking, rather than the app injecting.
    ///
    /// This is the **second of the two paths back from the cold tier** (the
    /// other is the ε-exploration slot in `retrieve`). Automatic injection
    /// excludes retired memories on purpose; an agent that has been *asked*
    /// about something old should still be able to find it. So `search` scores
    /// by relevance alone and looks at everything, including what was retired
    /// and what was superseded — with the superseded plainly marked, because a
    /// belief that has been replaced is still evidence about the past.
    func search(query: String, limit: Int = 10, now: Date = Date()) async -> [ScoredMemory] {
        guard isEnabled() else { return [] }
        do {
            let all = try await store.memories(status: nil, limit: 5_000)
            let cueVector = await embed(query)
            let vectors = try await store.embeddings(kind: "memory")
            let vectorByID = Dictionary(vectors) { first, _ in first }
            let keyword = try await store.keywordScores(query: query, table: "memories", limit: 50)

            var scored: [ScoredMemory] = []
            for memory in all {
                let relevance = Self.relevance(
                    cueVector: cueVector, id: memory.id, vectorByID: vectorByID, keyword: keyword)
                guard relevance > 0.2 else { continue }
                scored.append(
                    ScoredMemory(
                        memory: memory, score: relevance, relevance: relevance, isExploration: false
                    )
                )
            }
            scored.sort { $0.score > $1.score }
            let hits = Array(scored.prefix(limit))

            // A deliberate search is a real retrieval: it marks the memory seen,
            // which is what makes "surfaced repeatedly and never once useful"
            // into evidence the lifecycle can act on.
            for hit in hits {
                try? await store.upsert(MemoryLifecycle.markSeen(hit.memory, now: now))
            }
            return hits
        } catch {
            Log.memory.error("Search failed: \(error.localizedDescription)")
            return []
        }
    }

    // MARK: - Inspection (the Memory window)

    func allMemories() async -> [MemoryRecord] {
        (try? await store.memories(status: nil, limit: 5_000)) ?? []
    }

    func episodes(for memory: MemoryRecord) async -> [Episode] {
        var out: [Episode] = []
        for id in memory.sourceEpisodeIDs {
            if let episode = try? await store.episode(id: id) { out.append(episode) }
        }
        return out.sorted { $0.occurredAt < $1.occurredAt }
    }

    func journal(limit: Int = 200) async -> [JournalEntry] {
        (try? await store.journal(limit: limit)) ?? []
    }

    /// The owner's hand. The only true deletion in the system.
    func delete(_ memory: MemoryRecord) async {
        do {
            try await store.deleteMemory(id: memory.id)
            await refreshStats()
            Log.memory.info("Owner deleted memory \(memory.id.uuidString)")
        } catch {
            Log.memory.error("Delete failed: \(error.localizedDescription)")
        }
    }

    /// "That's wrong." Marks the belief contested and queues it for
    /// reconciliation in the next sleep — never an inline overwrite, because
    /// one contradiction must not be able to rewrite a settled belief.
    func contest(_ memory: MemoryRecord, now: Date = Date()) async {
        var m = memory
        m.status = .contested
        do {
            try await store.upsert(m)
            try await store.appendJournal(
                JournalEntry(
                    at: now, mutation: .contested, memoryID: m.id,
                    detail: "The owner contested this. Queued for reconciliation in sleep.",
                    before: memory.text))
            await refreshStats()
        } catch {
            Log.memory.error("Contest failed: \(error.localizedDescription)")
        }
    }
}
