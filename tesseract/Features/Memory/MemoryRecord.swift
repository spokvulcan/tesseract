//
//  MemoryRecord.swift
//  tesseract
//
//  The memory system's value types (ADR-0035, map #314).
//
//  Two layers, and the lower one is immutable:
//
//    Episode  — append-only, verbatim, never rewritten. The source of truth,
//               and the only thing that can ever correct a drifted belief.
//    MemoryRecord   — the derived, mutable, FIRST-PERSON layer ("I've noticed
//               Bohdan ..."), every record pointing back at the episodes it
//               came from via `sourceEpisodeIDs`.
//
//  A memory is ATOMIC: one memory = one claim = one `provenance`. Mixing
//  STATED and INFERRED inside one record is exactly what makes the
//  distinction unrecoverable later, and a reconstructive system cannot tell
//  confident-and-wrong from confident-and-right from the inside (ADR-0035 §2).
//

import Foundation

// MARK: - Episode (the immutable layer)

/// Where an episode came from. Chat, the Companion loop, and dictation are
/// the three write sources (owner's call, map #314 final grill).
nonisolated enum MemorySource: String, Codable, Sendable, CaseIterable {
    case chat
    case companion
    case dictation
    /// The cold-start backfill over the owner's existing corpus.
    case backfill
}

/// A verbatim record of something that happened. **Append-only.** No agent
/// may edit or delete one; the only deletion path in the system is the
/// owner's own hand in the Memory window.
nonisolated struct Episode: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let source: MemorySource
    /// The conversation this came from, when there is one.
    let conversationID: String?
    let occurredAt: Date
    /// The verbatim turn span — what was actually said.
    let text: String
    /// Free-form provenance detail (model, beat name, dictation duration…).
    let meta: [String: String]

    init(
        id: UUID = UUID(),
        source: MemorySource,
        conversationID: String? = nil,
        occurredAt: Date,
        text: String,
        meta: [String: String] = [:]
    ) {
        self.id = id
        self.source = source
        self.conversationID = conversationID
        self.occurredAt = occurredAt
        self.text = text
        self.meta = meta
    }
}

// MARK: - MemoryRecord (the derived layer)

/// What the user actually said versus what the assistant concluded.
///
/// A safety field, not a nicety: in the DRM paradigm 72% of false
/// recognitions of never-presented words received "remember" judgments —
/// *identical* to the rate for real words. The system cannot recover this
/// distinction later, so it must be recorded at the point of belief.
nonisolated enum Provenance: String, Codable, Sendable {
    /// The owner said this.
    case stated
    /// The assistant inferred this.
    case inferred
}

nonisolated enum MemoryKind: String, Codable, Sendable, CaseIterable {
    /// A stable trait or preference. The bulk of the semantic layer.
    case belief
    /// An assertion with a temporal argument ("his interview is Thursday").
    /// NOT a promise or a trigger — prospective memory is the Companion
    /// scheduler's business, not memory's (ADR-0035 §2).
    case event
    /// A distilled regularity: streaks, conscious switches, dismissal runs.
    /// A first-class product (#302), not an afterthought.
    case pattern
    /// How the owner wants the assistant to behave.
    case directive
}

/// Episodes want *separation*; generalizations want *completion*. The brain
/// does not dial between them — it runs two pathways and arbitrates. So a
/// record declares which it is, and **semanticization is a
/// `.specific → .general` migration** (ADR-0035 §2).
nonisolated enum Specificity: String, Codable, Sendable {
    case specific
    case general
}

nonisolated enum MemoryStatus: String, Codable, Sendable {
    case live
    /// Contradicted — by new evidence, or by the owner in the Memory window.
    /// Reconciled in the next sleep; never overwritten inline.
    case contested
    /// Replaced by a newer memory. **The interference path — this is how
    /// memories retire, not disuse** (ADR-0035 §4).
    case superseded
}

/// Retrieval priority tiers. Note what is absent: there is no `.deleted`.
/// Retirement is demotion, never deletion — storage strength never decreases,
/// and we can honour that literally because storage is cheap (ADR-0035 §3).
nonisolated enum MemoryTier: String, Codable, Sendable, CaseIterable, Comparable {
    /// Promoted to identity: injected unconditionally, every conversation.
    /// This is what promotion concretely grants.
    case core
    case hot
    case warm
    /// Out of the hot index and out of default retrieval — reachable by
    /// explicit search and by the ε-exploration slot. **Not deleted.**
    case cold

    private var rank: Int {
        switch self {
        case .core: 0
        case .hot: 1
        case .warm: 2
        case .cold: 3
        }
    }

    static func < (lhs: MemoryTier, rhs: MemoryTier) -> Bool { lhs.rank < rhs.rank }
}

/// The assistant's own belief about the owner. First-person by construction:
/// `text` reads "I've noticed Bohdan…", not "User prefers…".
nonisolated struct MemoryRecord: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    /// First-person, atomic, one claim.
    var text: String
    var kind: MemoryKind
    var provenance: Provenance
    var specificity: Specificity
    var status: MemoryStatus
    var tier: MemoryTier

    /// The episodes this belief was derived from. The rewriter always reads
    /// these — never only its own prior text — or it is Bartlett's
    /// serial-reproduction chain with a confabulation engine in it.
    var sourceEpisodeIDs: [UUID]

    let bornAt: Date

    // — the two strength numbers (ADR-0035 §3) —

    /// `S`, in days: the time-constant of the power-law need-probability
    /// curve. Grows only on a *useful* use.
    var stability: Double
    /// `SS`: **monotone non-decreasing, forever.** The deletion guard, and
    /// the answer to the rare-but-critical problem (the allergy mentioned
    /// once, needed in six months) that no shipped system solves.
    var storageStrength: Double
    /// `D ∈ [1,10]`: a slow-moving brittleness prior. Stored, but not fitted
    /// in v1 — with a noisy local judge it may be unidentifiable (ADR-0035
    /// Consequences).
    var difficulty: Double

    // — usage statistics: retrieved ≠ useful —

    /// Advances **only** on a graded-useful outcome. This is `t` in the decay
    /// term.
    var lastUsefulUseAt: Date
    var usefulUseCount: Int
    /// Retrieved into context, useful or not. **Diagnostic only** — it must
    /// never drive the lifecycle, or the system trains on its own retriever.
    var lastSeenAt: Date
    var seenCount: Int
    /// Times the store re-encountered this fact and found no surprise. The
    /// cheap path: `confirmations += 1` and the rewriter is never invoked.
    var confirmations: Int

    // — interference —

    var supersededBy: UUID?
    /// The near-duplicate neighbourhood. Cue overload — not disuse — is the
    /// real forgetting mechanism in a retrieval store.
    var cueClusterID: UUID?

    init(
        id: UUID = UUID(),
        text: String,
        kind: MemoryKind,
        provenance: Provenance,
        specificity: Specificity = .general,
        status: MemoryStatus = .live,
        tier: MemoryTier = .hot,
        sourceEpisodeIDs: [UUID] = [],
        bornAt: Date,
        stability: Double = MemoryLifecycle.initialStability,
        storageStrength: Double = 0,
        difficulty: Double = MemoryLifecycle.initialDifficulty,
        lastUsefulUseAt: Date? = nil,
        usefulUseCount: Int = 0,
        lastSeenAt: Date? = nil,
        seenCount: Int = 0,
        confirmations: Int = 0,
        supersededBy: UUID? = nil,
        cueClusterID: UUID? = nil
    ) {
        self.id = id
        self.text = text
        self.kind = kind
        self.provenance = provenance
        self.specificity = specificity
        self.status = status
        self.tier = tier
        self.sourceEpisodeIDs = sourceEpisodeIDs
        self.bornAt = bornAt
        self.stability = stability
        self.storageStrength = storageStrength
        self.difficulty = difficulty
        self.lastUsefulUseAt = lastUsefulUseAt ?? bornAt
        self.usefulUseCount = usefulUseCount
        self.lastSeenAt = lastSeenAt ?? bornAt
        self.seenCount = seenCount
        self.confirmations = confirmations
        self.supersededBy = supersededBy
        self.cueClusterID = cueClusterID
    }
}

// MARK: - Retrieval & grading

/// The outcome of having put a memory in front of the model.
///
/// The grade comes from the **sleep judge re-reading the turn** — never from
/// the hot path, and never from "was it retrieved". Retrieved-and-ignored is
/// *not* a lapse; it usually indicts the retriever (ADR-0035 §3).
nonisolated enum UseGrade: String, Codable, Sendable {
    /// The response depended on it.
    case decisive
    /// It informed the response.
    case used
    /// Retrieved and irrelevant. Not a lapse — a retriever miss.
    case ignored
    /// Stale or contradicted; it made the response worse.
    case harmful

    var isUseful: Bool { self == .decisive || self == .used }
}

/// One memory, placed in one turn's context, awaiting its grade.
nonisolated struct RetrievalEvent: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let memoryID: UUID
    /// The episode representing the turn this memory was retrieved *for* —
    /// what the sleep judge re-reads to grade it.
    let episodeID: UUID
    let retrievedAt: Date
    /// The cue that surfaced it, for per-cue affinity.
    let cue: String
    /// True when this slot came from the ε-exploration draw (warm/cold).
    /// Without these, the counterfactual is unobservable and the lifecycle
    /// trains on its own priors forever.
    let isExploration: Bool
    /// nil until sleep grades it.
    var grade: UseGrade?

    init(
        id: UUID = UUID(),
        memoryID: UUID,
        episodeID: UUID,
        retrievedAt: Date,
        cue: String,
        isExploration: Bool = false,
        grade: UseGrade? = nil
    ) {
        self.id = id
        self.memoryID = memoryID
        self.episodeID = episodeID
        self.retrievedAt = retrievedAt
        self.cue = cue
        self.isExploration = isExploration
        self.grade = grade
    }
}

// MARK: - The consolidation journal

/// Every mutation sleep makes, recorded. A bad consolidation must be
/// inspectable and revertable — and the episodic layer beneath is append-only,
/// so no consolidation can destroy the ground truth (ADR-0035 §7).
nonisolated enum MemoryMutation: String, Codable, Sendable {
    case added
    case confirmed
    case extended
    case contested
    case superseded
    case separated
    case promoted
    case demoted
    case graded
    /// The owner's hand. The only true deletion in the system.
    case deletedByOwner
}

nonisolated struct JournalEntry: Identifiable, Codable, Sendable {
    let id: UUID
    let at: Date
    let mutation: MemoryMutation
    let memoryID: UUID
    /// Human-readable: what changed, and why.
    let detail: String
    /// The memory's text before the change, when there was one.
    let before: String?
    let after: String?

    init(
        id: UUID = UUID(),
        at: Date,
        mutation: MemoryMutation,
        memoryID: UUID,
        detail: String,
        before: String? = nil,
        after: String? = nil
    ) {
        self.id = id
        self.at = at
        self.mutation = mutation
        self.memoryID = memoryID
        self.detail = detail
        self.before = before
        self.after = after
    }
}

// MARK: - Retrieval result

/// A memory, with the score that surfaced it. The score is multiplicative
/// (ADR-0035 §5) — the terms are probabilities of independent failure modes,
/// so a superseded or irrelevant memory cannot be rescued by sheer age.
nonisolated struct ScoredMemory: Sendable, Equatable {
    let memory: MemoryRecord
    let score: Double
    let relevance: Double
    let isExploration: Bool
}

/// An episode, with its relevance. Episodes are retrievable too — which is
/// what makes same-day recall work with no hot-path extraction at all.
nonisolated struct ScoredEpisode: Sendable, Equatable {
    let episode: Episode
    let relevance: Double
}
