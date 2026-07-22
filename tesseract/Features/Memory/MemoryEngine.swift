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

/// What retrieval put in front of the model. The retrieval events it logged
/// live in the store, not here — sleep reads them from its own queue.
nonisolated struct RetrievedContext: Sendable {
    var core: [MemoryRecord] = []
    var recalled: [ScoredMemory] = []
    var episodes: [ScoredEpisode] = []

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

    private let store: MemoryStore
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
            await reconcileEmbeddingScheme()
            await refreshStats()
        } catch {
            Log.memory.error("Embedder load failed: \(error.localizedDescription)")
            isEmbedderLoaded = false
        }
    }

    /// Vectors are artifacts of an embedding scheme (#332). When the running
    /// scheme differs from the one the store's vectors were made under, every
    /// vector — and the cue affinities learned from ranking them — is stale
    /// evidence: wipe, regenerate, stamp. The stamp lands only on completion,
    /// so an interrupted pass re-runs from the wipe instead of leaving a store
    /// that claims vectors it does not have. Measured cost: ~2 s for the
    /// owner's 442 records at 334 texts/sec.
    private func reconcileEmbeddingScheme() async {
        guard isEmbedderLoaded else { return }
        do {
            let stored = try await store.embeddingScheme()
            guard stored != MemoryEmbedder.scheme else { return }
            Log.memory.info(
                "Embedding scheme \(stored) → \(MemoryEmbedder.scheme): regenerating every stored vector"
            )
            try await store.resetEmbeddings()
            var total = 0
            for owner in [MemoryOwner.memory, MemoryOwner.episode] {
                let rows = try await store.allTexts(of: owner)
                for start in stride(from: 0, to: rows.count, by: 32) {
                    let slice = Array(rows[start..<min(start + 32, rows.count)])
                    let vectors = await embedder.embed(slice.map(\.text))
                    guard vectors.count == slice.count else { continue }
                    try await store.setEmbeddings(
                        Array(zip(slice.map(\.id), vectors)), of: owner)
                    total += slice.count
                }
            }
            try await store.stampEmbeddingScheme(MemoryEmbedder.scheme)
            Log.memory.info("Re-embedded \(total) records under scheme \(MemoryEmbedder.scheme)")
        } catch {
            Log.memory.error(
                "Embedding-scheme reconcile failed: \(error.localizedDescription)")
        }
    }

    func refreshStats() async {
        do {
            let episodes = try await store.episodeCount()
            let memories = try await store.memoryCount()
            let byTier = try await store.countsByTier()
            let ungraded = try await store.ungradedRetrievalCount()
            stats = MemoryStats(
                episodes: episodes, memories: memories, byTier: byTier,
                ungradedRetrievals: ungraded)
        } catch {
            Log.memory.error("Stats refresh failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Embedding

    /// Documents — what gets stored. Cues go through `embedQuery`; the two
    /// sides are embedded differently on purpose (#332).
    func embed(_ text: String) async -> [Float]? {
        guard isEmbedderLoaded else { return nil }
        return await embedder.embed(text)
    }

    func embed(_ texts: [String]) async -> [[Float]] {
        guard isEmbedderLoaded, !texts.isEmpty else { return [] }
        return await embedder.embed(texts)
    }

    /// Queries — what does the asking. Carries the instruct prefix the
    /// embedder was trained to expect on the query side only.
    func embedQuery(_ text: String) async -> [Float]? {
        guard isEmbedderLoaded else { return nil }
        return await embedder.embedQuery(text)
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
        id: UUID = UUID(),
        source: MemorySource,
        text: String,
        conversationID: String? = nil,
        occurredAt: Date = Date(),
        meta: [String: String] = [:]
    ) async -> Episode? {
        guard isEnabled() else { return nil }
        guard source != .dictation || isDictationCaptureEnabled() else { return nil }
        // His words, not the app's: a skill fire puts the skill's entire body in
        // the user message (see `MemorySpeech`). An episode is testimony, and the
        // wrapper is not his testimony.
        guard let trimmed = MemorySpeech.spoken(text) else { return nil }

        // `id` is the caller's when the episode has a natural identity — chat
        // capture passes the user message's own id, which is what lets the
        // retrieval log point at this turn *before* it is written, and what
        // makes re-capture (and the backfill meeting live history) a no-op.
        let episode = Episode(
            id: id, source: source, conversationID: conversationID, occurredAt: occurredAt,
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
            // One transaction: the memory and its journal line land together.
            try await store.upsert(
                memory, embedding: vector,
                journal: JournalEntry(
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
            let all = try await store.allLiveMemories()
            // Core is a subset of the same scan — no second query, no second
            // decode. `status == .live` matters: a *contested* core belief has
            // lost its unconditional seat until sleep resolves the dispute.
            let core = all.filter { $0.tier == .core && $0.status == .live }
            // `.cold` is genuinely out of the default pool — that is what
            // retirement *means* here. It is reachable two ways, and only two:
            // the ε-exploration draw below, and the agent's explicit
            // `recall` tool.
            let candidates = all.filter { $0.tier != .core && $0.tier != .cold }

            let cueVector = await embedQuery(cue)
            let vectors = try await store.embeddings(of: .memory)
            let vectorByID = Dictionary(vectors) { first, _ in first }
            let keyword = try await store.keywordScores(query: cue, in: .memory)
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
            // A targeted increment, not a snapshot upsert: the records in hand
            // were read before several `await`s, and writing them back whole
            // would roll back anything that landed in between.
            try? await store.markSeen(recalled.map(\.memory.id) + core.map(\.id), at: now)

            if let episodeID {
                let events =
                    core.map {
                        RetrievalEvent(
                            memoryID: $0.id, episodeID: episodeID, retrievedAt: now, cue: cue)
                    }
                    + recalled.map {
                        RetrievalEvent(
                            memoryID: $0.memory.id, episodeID: episodeID, retrievedAt: now,
                            cue: cue, isExploration: $0.isExploration)
                    }
                try await store.log(events)
            }

            return RetrievedContext(core: core, recalled: recalled, episodes: episodes)
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
        let keyword = try await store.keywordScores(query: cue, in: .episode, limit: 30)
        let vectors = try await store.embeddings(of: .episode)
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

        // Score before fetching: most of the ~60 candidates fall to the
        // relevance floor, and the survivors come back in one batch query
        // rather than a store round-trip apiece.
        let scored =
            candidateIDs
            .map {
                (
                    id: $0,
                    relevance: Self.relevance(
                        cueVector: cueVector, id: $0, vectorByID: vectorByID, keyword: keyword)
                )
            }
            .filter { $0.relevance > 0.2 }
            .sorted { $0.relevance > $1.relevance }
            .prefix(budget)
        let byID = Dictionary(
            try await store.episodes(ids: scored.map(\.id)).map { ($0.id, $0) }
        ) { first, _ in first }
        return scored.compactMap { candidate in
            byID[candidate.id].map { ScoredEpisode(episode: $0, relevance: candidate.relevance) }
        }
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
    ///
    /// `marksSeen` is not a convenience. "Seen" means **surfaced to someone** —
    /// it is evidence, and it is the evidence the third retirement path acts on
    /// ("shown eight times and never once helped"). Sleep's own reconcile uses
    /// this method to find a claim's neighbours, and those lookups are not
    /// surfacings: nobody saw them. Left marking, sleep inflates the counter it
    /// then retires against, and the store cannibalises itself over successive
    /// nights — measured on the owner's real store, where "He runs an X/Twitter
    /// account: @spok_vulkan" reached `seenCount 8` and went cold without ever
    /// being shown to anyone. So sleep passes `false`; the agent's `recall` tool
    /// — which genuinely puts a memory in front of the model — does not.
    func search(query: String, limit: Int = 10, marksSeen: Bool = true) async
        -> [ScoredMemory]
    {
        guard isEnabled() else { return [] }
        do {
            let all = try await store.memories(status: nil, limit: 5_000)
            let cueVector = await embedQuery(query)
            let vectors = try await store.embeddings(of: .memory)
            let vectorByID = Dictionary(vectors) { first, _ in first }
            let keyword = try await store.keywordScores(query: query, in: .memory, limit: 50)

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
            // into evidence the lifecycle can act on. An *internal* lookup is not.
            if marksSeen {
                try? await store.markSeen(hits.map(\.memory.id), at: Date())
            }
            return hits
        } catch {
            Log.memory.error("Search failed: \(error.localizedDescription)")
            return []
        }
    }

    /// The recall tool's full sweep (#332): distilled beliefs AND the raw
    /// episodic record. The second half is not optional — a fact told this
    /// morning exists only as an episode until sleep distills it, so a
    /// beliefs-only recall has a same-day blind spot, and "searches
    /// everything" in the tool's contract would be a lie.
    func searchEverything(query: String, limit: Int = 10, episodeBudget: Int = 5) async
        -> (memories: [ScoredMemory], episodes: [ScoredEpisode])
    {
        guard isEnabled() else { return ([], []) }
        let memories = await search(query: query, limit: limit)
        do {
            let cueVector = await embedQuery(query)
            let episodes = try await retrieveEpisodes(
                cue: query, cueVector: cueVector, budget: episodeBudget, excluding: nil)
            return (memories, episodes)
        } catch {
            Log.memory.error("Episode search failed: \(error.localizedDescription)")
            return (memories, [])
        }
    }

    // MARK: - Inspection (the Memory window)

    func allMemories() async -> [MemoryRecord] {
        (try? await store.memories(status: nil, limit: 5_000)) ?? []
    }

    /// The backfill's "has anything ever been recorded" gate.
    func episodeCount() async -> Int {
        (try? await store.episodeCount()) ?? 0
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
    ///
    /// The store flips the status against the *fresh* row, guarded on it still
    /// being live — the window's snapshot may be hours old, and a contest must
    /// not resurrect a belief sleep has since superseded.
    func contest(_ memory: MemoryRecord, reason: String? = nil, now: Date = Date()) async {
        do {
            if try await store.contest(id: memory.id, at: now, reason: reason) == nil {
                Log.memory.info(
                    "Contest skipped — the belief is no longer live: \(memory.id.uuidString)")
            }
            await refreshStats()
        } catch {
            Log.memory.error("Contest failed: \(error.localizedDescription)")
        }
    }

    /// What the `contest` tool found when it tried to act — the agent relays
    /// the owner's veto by handle, and every miss needs its own answer, because
    /// "nothing happened" teaches the model nothing.
    enum ContestOutcome: Sendable, Equatable {
        case contested(MemoryRecord)
        case alreadyContested(MemoryRecord)
        case alreadySuperseded(MemoryRecord)
        case notFound
    }

    /// The agent's door to the owner's veto (ADR-0035 §9), addressed by the
    /// short handle `recall` prints — the first eight hex digits of the id.
    ///
    /// The handle must resolve to exactly one memory; anything else is
    /// `.notFound`, and the right move is to `recall` again for a fresh handle
    /// rather than guess. Only a live memory can be contested — a superseded
    /// one is already history, and a contested one is already queued.
    func contest(handle: String, reason: String, now: Date = Date()) async -> ContestOutcome {
        // `recall` prints the handle bracketed — "[a1b2c3d4]" — and models copy
        // it verbatim, brackets included; accept that form.
        let key = handle.lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]").union(.whitespacesAndNewlines))
        guard key.count >= 6, key.allSatisfy(\.isHexDigit) else { return .notFound }
        let all = (try? await store.memories(status: nil, limit: 5_000)) ?? []
        let matches = all.filter { $0.id.uuidString.lowercased().hasPrefix(key) }
        guard matches.count == 1, let match = matches.first else { return .notFound }

        switch match.status {
        case .contested: return .alreadyContested(match)
        case .superseded: return .alreadySuperseded(match)
        case .live:
            await contest(match, reason: reason, now: now)
            return .contested(match)
        }
    }
}
