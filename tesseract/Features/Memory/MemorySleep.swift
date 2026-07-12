//
//  MemorySleep.swift
//  tesseract
//
//  Consolidation (ADR-0035 §6, §7) — where the memory system actually thinks.
//
//  Everything on the hot path is deliberately stupid: an insert and an
//  embedding. Nothing is judged, nothing is extracted, nothing is decided. That
//  is not laziness, it is the central claim of the design — **salience is
//  retroactive**. At the moment something is said, the information that
//  determines whether it mattered has not arrived yet. A write-time importance
//  judge is guessing, and its guesses are unfalsifiable because it discards what
//  it decides is unimportant.
//
//  So the thinking happens here, later, offline, when the day is over and its
//  consequences are visible. Five work items, in this order, because each one
//  depends on the last:
//
//    1. **Grade** — what did the memories I recalled yesterday actually *do*?
//       This is the only place a grade is ever assigned. Retrieval logs; it does
//       not judge. "Retrieved" is not "useful", and conflating them is how these
//       systems come to believe their own priors.
//
//    2. **Re-examine** — what he told me I got wrong. A contested belief goes
//       back to the episodes it was drawn from, without the reading he rejected,
//       and either a corrected claim supersedes it or it goes cold. His word
//       about his own life outranks anything I concluded.
//
//    3. **Extract** — read the day's episodes and write down what is worth
//       keeping. In batches, by conversation, because a claim only makes sense
//       against its neighbours.
//
//    4. **Reconcile** — for each new claim, ask what I already believe that is
//       closest to it. **Prediction-error gated**: if the claim says nothing new,
//       the existing belief is *confirmed* and the rewriter is never invoked. A
//       memory system whose rewriter runs on every observation will drift into
//       fiction on rephrasing alone. Only a genuine contradiction supersedes, and
//       superseding never deletes.
//
//    5. **Sweep** — move tiers. Promote what has proved itself; retire what has
//       been superseded, or has been shown repeatedly and never once helped.
//       Storage strength never decreases, so this is a change of *reachability*,
//       not of existence.
//
//  Every item takes the GPU lease itself and releases it before the next, and the
//  whole run is one cancellable Task. The owner touching the keyboard cancels it
//  mid-generation: the lease drops, the foreground turn goes straight through, and
//  the abandoned item is simply redone next time — episodes stay unconsolidated
//  until an item that consumed them *completes*. Nothing here is half-written.
//

import Foundation

@MainActor
@Observable
final class MemorySleep {

    enum Phase: Equatable, Sendable {
        case idle
        case grading
        case reexamining
        case extracting
        case reconciling
        case sweeping
    }

    struct Summary: Sendable, Equatable {
        var graded = 0
        var reexamined = 0
        var episodesRead = 0
        var added = 0
        var confirmed = 0
        var superseded = 0
        var promoted = 0
        var retired = 0
        var yielded = false

        var didAnything: Bool {
            graded + reexamined + added + confirmed + superseded + promoted + retired > 0
        }
    }

    private(set) var phase: Phase = .idle
    private(set) var lastRun: Date?
    private(set) var lastSummary = Summary()
    var isRunning: Bool { runTask != nil }

    private let engine: MemoryEngine
    private let arbiter: any InferenceArbitrating
    private let complete: @Sendable (String) async throws -> String
    private let isEnabled: @MainActor () -> Bool

    private var runTask: Task<Void, Never>?
    /// Which run owns `runTask`. See `start()`.
    private var runGeneration = 0

    /// How many episodes one extraction call reads. Small enough that the model
    /// can hold them all, large enough that a claim can be seen recurring —
    /// which is most of what distinguishes a pattern from an accident.
    private let extractionBatch = 8

    /// How many batches one run will do — a runaway guard, not a budget.
    ///
    /// It was 12, on the assumption that a batch holds 8 episodes. It does not:
    /// batches are per-conversation, and the owner's conversations average three
    /// user turns, so 12 batches read 40 episodes of 207 and the first night got
    /// through a fifth of his history. Now that a batch is digested atomically,
    /// there is nothing to protect against by stopping early — the owner's return
    /// cancels the run mid-generation whatever the cap says, and the unread
    /// batches are simply still there tomorrow. So the cap sits above any real
    /// backlog and the *yield* does the work it was pretending to do.
    private let maxBatchesPerRun = 100

    init(
        engine: MemoryEngine,
        arbiter: any InferenceArbitrating,
        complete: @escaping @Sendable (String) async throws -> String,
        isEnabled: @escaping @MainActor () -> Bool = { true }
    ) {
        self.engine = engine
        self.arbiter = arbiter
        self.complete = complete
        self.isEnabled = isEnabled
    }

    // MARK: - The run

    /// Start a consolidation pass. Returns immediately; the work runs in a
    /// cancellable task.
    ///
    /// The generation counter is not ceremony. A cancelled run keeps executing
    /// until its `await` unwinds, so it can outlive the `yield()` that killed it
    /// — and by then, if the owner has stepped away again, a *new* run is already
    /// installed in `runTask`. Without the check, the dying run's cleanup would
    /// clear its successor's handle, `isRunning` would read false while a run was
    /// live, and the next idle tick would start a **second concurrent sleep** over
    /// the same episodes. Two consolidations racing to distil the same day is how
    /// a memory store fills with duplicates.
    func start() {
        guard isEnabled(), runTask == nil else { return }
        runGeneration += 1
        let generation = runGeneration
        runTask = Task { [weak self] in
            await self?.run()
            self?.finish(generation: generation)
        }
    }

    /// The owner is back. **This must be instant** — so it cancels rather than
    /// asking politely. An in-flight generation aborts, the lease drops within a
    /// token, and the foreground turn that is queued behind it goes through.
    /// The work item that died is redone next sleep; nothing was half-written.
    func yield() {
        guard let task = runTask else { return }
        // Retire this generation *before* cancelling: the task unwinds on its own
        // schedule, and it must not be able to clear a handle that by then belongs
        // to its successor.
        runGeneration += 1
        task.cancel()
        runTask = nil
        phase = .idle
        lastSummary.yielded = true
        Log.memory.info("Sleep yielded — the owner is back")
    }

    /// A run finished under its own power. Only the *current* run may release the
    /// handle; a cancelled predecessor unwinding late must not.
    private func finish(generation: Int) {
        guard generation == runGeneration else { return }
        runTask = nil
    }

    /// The whole work list. Public so a test — and the owner's "consolidate now"
    /// affordance — can drive one pass to completion synchronously.
    func run() async {
        guard isEnabled() else { return }
        let started = Date()
        var summary = Summary()
        Log.memory.info("Sleep: starting")

        do {
            phase = .grading
            summary.graded = try await gradeRetrievals()
            try Task.checkCancellation()

            // Before anything I concluded on my own: what he told me I got wrong.
            try await reexamineContested(into: &summary)
            try Task.checkCancellation()

            try await consolidate(into: &summary)
            try Task.checkCancellation()

            phase = .sweeping
            try await sweep(into: &summary)
        } catch is CancellationError {
            summary.yielded = true
        } catch {
            Log.memory.error("Sleep failed: \(error.localizedDescription)")
        }

        phase = .idle
        lastRun = started
        lastSummary = summary
        await engine.refreshStats()
        Log.memory.info(
            "Sleep done in \(Int(Date().timeIntervalSince(started)))s — "
                + "graded \(summary.graded), re-read \(summary.reexamined), "
                + "read \(summary.episodesRead), "
                + "added \(summary.added), confirmed \(summary.confirmed), "
                + "superseded \(summary.superseded), promoted \(summary.promoted), "
                + "retired \(summary.retired)\(summary.yielded ? " (yielded)" : "")")
    }

    // MARK: - 1. Grade

    /// What did the memories I recalled actually *do*?
    ///
    /// The judge re-reads the turn with the memories that were in front of it and
    /// says, for each, whether it decided the answer, shaped it, made no
    /// difference, or made it worse. This is the lifecycle's only source of
    /// truth, and it is deliberately not available at retrieval time — you cannot
    /// know whether a memory helped until you see what was said next.
    private func gradeRetrievals() async throws -> Int {
        let events = try await engine.store.ungradedRetrievals(limit: 200)
        guard !events.isEmpty else { return 0 }

        // Grade a turn at a time: the judge needs to see all the memories that
        // were competing for the same answer, not one in isolation.
        let byEpisode = Dictionary(grouping: events, by: \.episodeID)
        var graded = 0

        for (episodeID, group) in byEpisode {
            try Task.checkCancellation()
            guard let episode = try await engine.store.episode(id: episodeID) else {
                // The turn is gone; the events are unjudgeable. Retire them as
                // ignored so they don't clog the queue forever.
                for event in group { try await engine.store.setGrade(.ignored, for: event.id) }
                continue
            }

            var memories: [(RetrievalEvent, MemoryRecord)] = []
            for event in group {
                if let memory = try await engine.store.memory(id: event.memoryID) {
                    memories.append((event, memory))
                }
            }
            guard !memories.isEmpty else { continue }

            let verdicts = try await judge(episode: episode, memories: memories.map(\.1))
            for (index, pair) in memories.enumerated() {
                let grade = index < verdicts.count ? verdicts[index] : .ignored
                try await apply(grade: grade, to: pair.1, event: pair.0)
                graded += 1
            }
        }
        return graded
    }

    private func judge(
        episode: Episode, memories: [MemoryRecord]
    ) async throws -> [UseGrade] {
        let listed = memories.enumerated()
            .map { "[\($0.offset + 1)] \($0.element.text)" }
            .joined(separator: "\n")
        let reply = episode.meta["reply"].map { "\n\nWhat I answered:\n\($0.prefix(1_200))" } ?? ""

        let prompt = """
            You are grading your own memory. Below is something the person said to \
            you, what you answered, and the memories that were put in front of you \
            before you answered.

            What he said:
            \(episode.text.prefix(1_500))\(reply)

            The memories you were shown:
            \(listed)

            For each memory, say what it actually did for that answer:
              decisive — the answer would have been wrong or generic without it
              used     — it shaped the answer, but the answer would have survived without it
              ignored  — it made no difference to what you said
              harmful  — it pushed the answer in a wrong direction

            Be honest and be strict. Most memories are `ignored`; that is the normal \
            case and it costs them nothing. Reserve `decisive` for the ones the \
            answer genuinely turned on.

            Answer with exactly \(memories.count) line(s), in order, nothing else:
            1: <grade>
            2: <grade>
            """

        let response = try await generate(prompt)
        return Self.parseGrades(response, count: memories.count)
    }

    static func parseGrades(_ response: String, count: Int) -> [UseGrade] {
        var grades = [UseGrade](repeating: .ignored, count: count)
        for line in response.split(separator: "\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2, let index = Int(parts[0].trimmingCharacters(in: .whitespaces)),
                index >= 1, index <= count
            else { continue }
            let word = parts[1].lowercased().trimmingCharacters(
                in: .whitespaces.union(.punctuationCharacters))
            if let grade = UseGrade(rawValue: word) { grades[index - 1] = grade }
        }
        return grades
    }

    /// Apply the judge's verdict — the one place the lifecycle moves.
    private func apply(grade: UseGrade, to memory: MemoryRecord, event: RetrievalEvent) async throws
    {
        try await engine.store.setGrade(grade, for: event.id)
        let updated = MemoryLifecycle.applyGrade(grade, to: memory, now: Date())
        try await engine.store.upsert(updated)

        // `.ignored` is NOT a lapse. It touches nothing about the memory itself —
        // it only says this memory was a poor answer to *this cue*, which is a
        // fact about the pairing, not about the belief.
        if grade == .ignored {
            try await engine.store.decayCueAffinity(cue: event.cue, memoryID: memory.id)
        }
        if grade == .harmful {
            try await engine.store.appendJournal(
                JournalEntry(
                    at: Date(), mutation: .demoted, memoryID: memory.id,
                    detail: "Led me wrong when recalled for: \(event.cue.prefix(80))"))
        }
    }

    // MARK: - 2. The owner's veto

    /// He said "that's wrong." Now do something about it.
    ///
    /// Contest is not delete — deletion is his hand alone (ADR-0035 §9). It is a
    /// **request for a re-read**, and it can be honoured precisely because the
    /// store has two layers. The episodes are testimony: verbatim, immutable, not
    /// in doubt. The belief on top of them is *my inference*, and inference is
    /// exactly the layer that can be wrong. So sleep goes back to the episodes the
    /// belief was drawn from and asks what they actually support.
    ///
    /// Two outcomes, neither of them a deletion:
    ///
    ///   - the evidence supports a **corrected** claim — narrower, or the opposite
    ///     — and it supersedes the contested one, which remains as the record of
    ///     what I used to think;
    ///   - the evidence supports **nothing** once the rejected reading is taken
    ///     away, and the belief goes cold: still stored, still reachable if he asks
    ///     for it outright, never offered again.
    ///
    /// A contested belief never survives its contest intact. He is the authority on
    /// his own life, so a re-read that merely restates the rejected claim is not a
    /// correction — it is the model arguing with him, and it is dropped on the
    /// floor. The successor also inherits none of the rejected belief's strength or
    /// confirmations: a claim he threw out must not launder its credibility into
    /// the one that replaces it.
    private func reexamineContested(into summary: inout Summary) async throws {
        let contested = try await engine.store.memories(status: .contested, limit: 200)
        guard !contested.isEmpty else { return }
        phase = .reexamining

        for memory in contested {
            try Task.checkCancellation()

            let episodes = await engine.episodes(for: memory)
            let correction = episodes.isEmpty ? nil : try await reread(memory, from: episodes)

            guard let correction else {
                // Nothing survives. Note the tier move is all that happens: the
                // status stays `contested`, so if he ever does recall it explicitly
                // the dispute travels with it, and storage strength is untouched
                // because strength is monotone by construction (ADR-0035 §3).
                var retired = memory
                retired.tier = .cold
                try await engine.store.upsert(retired)
                try await engine.store.appendJournal(
                    JournalEntry(
                        at: Date(), mutation: .demoted, memoryID: memory.id,
                        detail: episodes.isEmpty
                            ? "He contested this, and there are no source episodes left to "
                                + "re-read. Retired."
                            : "He contested this, and re-reading what he actually said does not "
                                + "support it. Retired.",
                        before: memory.text))
                summary.reexamined += 1
                summary.retired += 1
                continue
            }

            try await add(correction, supersedes: memory, inheritStrength: false, into: &summary)
            summary.reexamined += 1
        }
    }

    /// Ask the episodes, not the belief.
    private func reread(_ memory: MemoryRecord, from episodes: [Episode]) async throws -> Claim? {
        let day = DateFormatter()
        day.dateFormat = "yyyy-MM-dd"
        let evidence =
            episodes
            .sorted { $0.occurredAt < $1.occurredAt }
            .map { "[\(day.string(from: $0.occurredAt))] He said: \($0.text.prefix(1_200))" }
            .joined(separator: "\n\n")

        let prompt = """
            The person I serve has told me that one of my memories is WRONG.

            The memory he rejected:
            \(memory.text)

            What he actually said — verbatim, and not in doubt. This is the evidence \
            I drew that memory from:

            \(evidence)

            He is the authority on his own life. His word overrules my conclusion; \
            the rejected memory is not on the table and must not be restated, \
            softened, or argued for.

            Re-read the evidence and answer with exactly one line:

            - If the evidence supports a genuinely different claim — narrower, more \
            careful, or the opposite of what I wrote — write that claim, in the \
            format below.
            - Otherwise answer with the single word NOTHING. That is a good answer, \
            and the common one: most rejected memories were over-readings of \
            evidence that will not carry any claim at all.

            Format — one line, three fields, pipe-separated:
            STATED|belief|He is allergic to shellfish.
            INFERRED|pattern|He works in long focused blocks late at night.

            STATED if he said it outright; INFERRED if I would be concluding it.
            """

        let response = try await generate(prompt)
        let claims = Self.parseClaims(response, sourceEpisodeIDs: episodes.map(\.id))
        // A "correction" that says the same thing again is the model arguing with
        // him. He wins.
        return claims.first { !Self.isSameClaim($0.text, as: memory.text) }
    }

    /// Same claim, allowing for punctuation and case — enough to catch a model that
    /// re-emits the rejected line with a full stop added.
    static func isSameClaim(_ lhs: String, as rhs: String) -> Bool {
        func normalise(_ text: String) -> String {
            text.lowercased().filter { $0.isLetter || $0.isNumber }
        }
        return normalise(lhs) == normalise(rhs)
    }

    // MARK: - 3 & 4. Extract, then reconcile — one batch at a time

    /// Read the unconsolidated episodes and write down what is worth keeping.
    ///
    /// **A batch is digested whole or not at all**, and that ordering is the
    /// point. Extraction used to run to completion across every batch, marking
    /// episodes consumed as it went, and only then reconcile the accumulated
    /// claims. Yield in the gap — which is to say, the owner touching his
    /// keyboard — and the claims were dropped while the episodes that produced
    /// them stayed marked consolidated. The knowledge was gone, and no later
    /// sleep would ever look at those turns again. Silent, permanent, and
    /// invisible in every summary line.
    ///
    /// So each batch now extracts, reconciles, and only *then* marks its episodes
    /// consumed. A cancel at any point leaves that batch untouched in the queue.
    /// Re-reading it next sleep is free: the prediction-error gate meets its own
    /// claims again and answers SAME, which confirms rather than duplicates.
    ///
    /// Batched *by conversation*: a claim is only legible against its neighbours
    /// ("the second one" means nothing alone), and a conversation is the natural
    /// unit in which that context holds.
    private func consolidate(into summary: inout Summary) async throws {
        let episodes = try await engine.store.unconsolidatedEpisodes(limit: 1_000)
        guard !episodes.isEmpty else { return }

        var batches: [[Episode]] = []
        for group in Dictionary(grouping: episodes, by: { $0.conversationID ?? $0.id.uuidString })
            .values
            .sorted(by: {
                ($0.first?.occurredAt ?? .distantPast) < ($1.first?.occurredAt ?? .distantPast)
            })
        {
            let ordered = group.sorted { $0.occurredAt < $1.occurredAt }
            for start in stride(from: 0, to: ordered.count, by: extractionBatch) {
                batches.append(Array(ordered[start..<min(start + extractionBatch, ordered.count)]))
            }
        }

        for batch in batches.prefix(maxBatchesPerRun) {
            try Task.checkCancellation()

            phase = .extracting
            let claims = try await extract(from: batch)

            phase = .reconciling
            try await reconcile(claims, into: &summary)

            summary.episodesRead += batch.count
            try await engine.store.markConsolidated(batch.map(\.id), at: Date())
        }

        if batches.count > maxBatchesPerRun {
            Log.memory.info(
                "Sleep: \(batches.count - maxBatchesPerRun) batches left for the next pass")
        }
    }

    private func extract(from episodes: [Episode]) async throws -> [Claim] {
        let day = DateFormatter()
        day.dateFormat = "yyyy-MM-dd"

        let transcript = episodes.map { episode -> String in
            var block =
                "[\(day.string(from: episode.occurredAt))] He said: \(episode.text.prefix(1_200))"
            if let reply = episode.meta["reply"] {
                block += "\n    I answered: \(reply.prefix(600))"
            }
            return block
        }.joined(separator: "\n\n")

        let prompt = """
            You are the long-term memory of a personal assistant, consolidating what \
            happened. Below is a stretch of conversation with the person you serve.

            \(transcript)

            Write down what is worth remembering about *him*, for months from now.

            Rules:
            - One claim per line, self-contained. It must still make sense in six \
            months with no other context: "He is allergic to shellfish", never \
            "he's allergic to it".
            - Write about him in the third person.
            - Mark each claim STATED or INFERRED. A guess recorded as testimony is \
            the one error you can never undo, so the test is strict: STATED means he \
            *asserted the claim himself, in words* — you could point at the sentence. \
            Everything you read off what he did, asked for, or chose is INFERRED, \
            however obvious it seems. **Asking about a thing is not stating an \
            interest in it.** "Research the Dota 2 tournament for me" supports \
            INFERRED|belief|He follows Dota 2 esports — never STATED. When in doubt, \
            INFERRED.
            - Keep only what is durable. Skip the task he was doing, the bug he was \
            chasing, the code you wrote — unless it reveals something lasting about \
            him. Ask of every line: would this still matter in six months?
            - Skip anything about you, or about this app's internals.
            - If nothing here is worth keeping, answer with the single word NOTHING. \
            That is a good and common answer.

            Format — exactly one claim per line, four fields, pipe-separated:
            STATED|belief|He is allergic to shellfish.
            INFERRED|pattern|He works in long focused blocks late at night.
            STATED|directive|He wants me to answer briefly when he is debugging.

            Kinds: belief (a stable fact or preference), event (a thing that \
            happened), pattern (a recurring regularity), directive (a standing \
            instruction to me).
            """

        let response = try await generate(prompt)
        let sourceIDs = episodes.map(\.id)
        return Self.parseClaims(response, sourceEpisodeIDs: sourceIDs)
    }

    struct Claim: Sendable, Equatable {
        let text: String
        let kind: MemoryKind
        let provenance: Provenance
        let sourceEpisodeIDs: [UUID]
    }

    static func parseClaims(_ response: String, sourceEpisodeIDs: [UUID]) -> [Claim] {
        var claims: [Claim] = []
        for rawLine in response.split(separator: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, line.uppercased() != "NOTHING" else { continue }
            let fields = line.split(separator: "|", maxSplits: 2).map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            guard fields.count == 3 else { continue }
            guard let provenance = Provenance(rawValue: fields[0].lowercased()) else { continue }
            let kind = MemoryKind(rawValue: fields[1].lowercased()) ?? .belief
            let text = fields[2]
            // A "claim" of three words is a fragment, not a memory.
            guard text.count > 10 else { continue }
            claims.append(
                Claim(
                    text: text, kind: kind, provenance: provenance,
                    sourceEpisodeIDs: sourceEpisodeIDs))
        }
        return claims
    }

    // MARK: - 4. Reconcile (the prediction-error gate)

    /// The gate that keeps the store from drifting into fiction.
    ///
    /// For each new claim, find what I already believe that is nearest to it and
    /// ask one question: is this *news*? If it is not — if the belief already
    /// covers it — the belief is confirmed and **the rewriter is never invoked**.
    /// This is the whole point. A system that regenerates its beliefs on every
    /// mention will, in a few hundred passes, be confidently telling you things
    /// no one ever said, each rewrite a plausible small step from the last.
    ///
    /// Only a real contradiction supersedes — and superseding is a status change,
    /// not a deletion. What I used to think is evidence about the past.
    private func reconcile(_ claims: [Claim], into summary: inout Summary) async throws {
        guard !claims.isEmpty else { return }

        for claim in claims {
            try Task.checkCancellation()

            // `marksSeen: false` — finding a claim's neighbours is bookkeeping, not
            // a surfacing. Nobody saw these. Marking them would inflate the very
            // counter the third retirement path acts on, and sleep would slowly
            // retire the store out from under itself.
            let neighbours = await engine.search(query: claim.text, limit: 4, marksSeen: false)
                .filter { $0.memory.status != .superseded }
                .map(\.memory)

            let verdict: Verdict
            if neighbours.isEmpty {
                verdict = .new
            } else {
                verdict = try await adjudicate(claim: claim, against: neighbours)
            }

            switch verdict {
            case .same(let index):
                guard index < neighbours.count else { break }
                // No prediction error. Confirm — cheap, and *not* a retrieval:
                // this raises confidence without touching stability, because
                // nothing was recalled and nothing proved useful.
                let confirmed = MemoryLifecycle.confirm(neighbours[index])
                try await engine.store.upsert(confirmed)
                summary.confirmed += 1

            case .new:
                try await add(claim, supersedes: nil, into: &summary)

            case .replaces(let index):
                guard index < neighbours.count else {
                    try await add(claim, supersedes: nil, into: &summary)
                    break
                }
                try await add(claim, supersedes: neighbours[index], into: &summary)
            }
        }
    }

    /// Internal, not private: `parseVerdict` is the seam the tests drive, and a
    /// private result type would make it unreachable from them.
    enum Verdict: Equatable {
        case same(Int)
        case new
        case replaces(Int)
    }

    private func adjudicate(claim: Claim, against neighbours: [MemoryRecord]) async throws
        -> Verdict
    {
        let listed = neighbours.enumerated()
            .map { "[\($0.offset + 1)] \($0.element.text)" }
            .joined(separator: "\n")

        let prompt = """
            I have just concluded something about the person I serve, and I need to \
            know whether it is news.

            What I have just concluded:
            \(claim.text)

            The closest things I already believe:
            \(listed)

            Answer with exactly one line, nothing else:

            SAME <n>      — belief <n> already says this. The new observation adds nothing.
            NEW           — this is genuinely new. Nothing I believe covers it.
            REPLACES <n>  — this CONTRADICTS belief <n>, or supersedes it. Belief <n> is no longer true.

            Be conservative. SAME is the most common answer and the safest: a \
            rephrasing, a narrower case, or the same fact said again is SAME, not \
            NEW. Use REPLACES only for a real contradiction — a changed fact, a \
            reversed preference — never for a mere elaboration.
            """

        let response = try await generate(prompt)
        return Self.parseVerdict(response, count: neighbours.count)
    }

    static func parseVerdict(_ response: String, count: Int) -> Verdict {
        let text = response.uppercased()
        // Scan for the first recognisable verdict token: local models like to
        // preface. Order matters — "REPLACES" must be tested before "SAME",
        // since a chatty answer may mention both.
        func index(after keyword: String) -> Int? {
            guard let range = text.range(of: keyword) else { return nil }
            let tail = text[range.upperBound...].prefix(6)
            guard let number = Int(tail.filter(\.isNumber).prefix(1)) else { return nil }
            return number - 1
        }
        if let i = index(after: "REPLACES"), i >= 0, i < count { return .replaces(i) }
        if let i = index(after: "SAME"), i >= 0, i < count { return .same(i) }
        if text.contains("NEW") { return .new }
        // Unparseable: treat as SAME-nothing, i.e. drop the claim. Silently
        // adding a memory the judge could not vouch for is the worse failure.
        return count > 0 ? .same(0) : .new
    }

    /// - Parameter inheritStrength: whether the successor takes on what the old
    ///   belief earned. True when a belief is superseded because the world moved on
    ///   — it was right for a while and the new claim is its continuation, not a
    ///   stranger. **False when the owner rejected it**: strength it accrued while
    ///   wrong is not credit the correction gets to spend.
    private func add(
        _ claim: Claim, supersedes old: MemoryRecord?, inheritStrength: Bool = true,
        into summary: inout Summary
    ) async throws {
        let now = Date()
        var memory = MemoryRecord(
            text: claim.text,
            kind: claim.kind,
            provenance: claim.provenance,
            specificity: .general,
            tier: .hot,
            sourceEpisodeIDs: claim.sourceEpisodeIDs,
            bornAt: now)

        if var old {
            // Superseded, never deleted. A belief that has been replaced is still
            // the truth about what was true, and about what I used to think.
            let wasContested = old.status == .contested
            old.status = .superseded
            old.supersededBy = memory.id
            try await engine.store.upsert(old)
            if inheritStrength {
                memory.storageStrength = max(memory.storageStrength, old.storageStrength)
                memory.confirmations = old.confirmations
            }
            try await engine.store.appendJournal(
                JournalEntry(
                    at: now, mutation: .superseded, memoryID: old.id,
                    detail: wasContested
                        ? "He said this was wrong. Corrected against what he actually said."
                        : "Replaced by a later observation.",
                    before: old.text, after: memory.text))
            summary.superseded += 1
        }

        let vector = await engine.embed(memory.text)
        try await engine.store.upsert(memory, embedding: vector)
        try await engine.store.appendJournal(
            JournalEntry(
                at: now, mutation: .added, memoryID: memory.id,
                detail: old == nil
                    ? "Learned this in consolidation." : "Learned this, replacing an older belief.",
                after: memory.text))
        summary.added += 1
    }

    // MARK: - 5. Sweep

    /// Move tiers. Nothing is deleted and no strength is ever taken away — this
    /// changes only what retrieval will *reach for* by default.
    private func sweep(into summary: inout Summary) async throws {
        let memories = try await engine.store.memories(status: nil, limit: 5_000)
        let now = Date()

        // Live beliefs only. A superseded one has no tier worth moving, and a
        // contested one was just placed by phase 2 — sweeping it here would look
        // at a freshly-born, never-used belief, conclude it belongs in `hot`, and
        // put the memory he rejected straight back in front of him.
        for memory in memories where memory.status == .live {
            try Task.checkCancellation()
            let usefulDays = try await engine.store.distinctUsefulDays(memoryID: memory.id)
            let updated = MemoryLifecycle.sweepTier(
                memory, distinctUsefulDays: usefulDays, now: now)
            guard updated.tier != memory.tier else { continue }
            try await engine.store.upsert(updated)

            let promoted = updated.tier > memory.tier
            if promoted { summary.promoted += 1 } else { summary.retired += 1 }
            try await engine.store.appendJournal(
                JournalEntry(
                    at: now, mutation: promoted ? .promoted : .demoted, memoryID: memory.id,
                    detail: promoted
                        ? "Proved itself \(usefulDays) days running — always present now."
                        : "Moved to \(updated.tier.rawValue): still reachable, "
                            + "no longer offered by default.",
                    after: memory.text))
        }

        // Contested memories are not swept: they were dealt with in phase 2, which
        // either corrected them or put them cold. Between the contest and that
        // sleep the read path still carries them — with the dispute attached, and
        // never silently.
    }

    // MARK: - The GPU

    /// One generation, under its own lease.
    ///
    /// Per *call*, not per run: the lease queue is FIFO with no preemption, so a
    /// run that held the lease for its whole duration would make the owner wait
    /// behind an hour of consolidation. Taking and dropping it around each call
    /// means the worst a foreground turn can wait is one sleep generation — and
    /// `yield()` cancels even that.
    private func generate(_ prompt: String) async throws -> String {
        try await arbiter.withExclusiveGPU(.llm) { [complete] in
            try await complete(prompt)
        }
    }
}
