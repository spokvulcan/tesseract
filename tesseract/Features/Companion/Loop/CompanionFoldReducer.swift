//
//  CompanionFoldReducer.swift
//  tesseract
//
//  The **Companion Fold Reducer** (ADR-0051): the Event Fold's write-side
//  sequencing as one pure decider. ADR-0046 extracted the fold's read side
//  (the Wake Evaluator) and its storage math into tested modules, but the
//  sequencing that binds them — consume only on a completed turn,
//  re-present on failure, the retry ladder where wakes fall back to
//  banners while Events wait for relaunch, the wake transitions scattered
//  across the loop's methods — lived inline, in code no test could reach.
//  Here it is a value machine: gather → decide → ordered effect values;
//  `CompanionLoop` performs them (store, notifier, recorder, agent calls)
//  and decides nothing.
//

import Foundation

/// Pure decider over one fold turn's writes and one reaction's writes.
/// Holds exactly one piece of state — the failed-attempt ledger — and
/// returns ordered `Effect` values naming every store/notifier mutation.
/// The correctness invariant (a wake or an Event is consumed only by a
/// completed turn; anything less re-presents) is decided here and only
/// performed by the loop.
nonisolated struct CompanionFoldReducer: Sendable, Equatable {

    /// A failing turn retries this many times before the generic fallback
    /// (ADR-0040 §13).
    static let maxTurnAttempts = 2

    /// Failed attempts per fold batch, keyed by the earliest wake id (or
    /// the earliest event id for an event-only batch).
    private var turnAttempts: [UUID: Int] = [:]

    /// One store/notifier write, as a value. The loop performs these in
    /// order; wakes ride along already transitioned (the reducer is the
    /// one home of wake-state flips), except where the performer must
    /// re-read to respect a concurrent move (`deliverFiredWake` — a
    /// `revise_wake` flip back to booked wins over delivery).
    enum Effect: Equatable, Sendable {
        /// Fired = presented to the entity, recorded before the turn so a
        /// crash between fire and completion is visible as
        /// fired-but-unconsumed. Carries the flipped wake.
        case fireWake(CompanionWake)
        /// The firing beat advances the ignored-promise ladder (#309); the
        /// performer collects the resurfaced wakes for the opening.
        case runResurfacingPass
        /// Events consumed by this completed turn — the invariant's
        /// success half.
        case consumeEvents(ids: [UUID], turnID: UUID)
        /// A wake this completed turn delivered. The performer re-reads
        /// and flips only a wake still `.fired` — a wake the turn itself
        /// moved (revise_wake back to booked) is respected.
        case deliverFiredWake(id: UUID, turnID: UUID, conversationID: UUID)
        /// Failure with retries remaining: the wake re-presents as booked
        /// (the pre-fire value — order and content untouched).
        case rebookWake(CompanionWake)
        /// Failure with retries remaining: the Events go back to pending
        /// in their original order.
        case representEvents(ids: [UUID])
        /// Retries exhausted — the brain is offline: the wake's own line
        /// delivers as a plain banner so never-silent-give-up holds
        /// (ADR-0040 §13). Carries the wake already flipped to delivered;
        /// the performer posts first, then upserts. The batch's Events
        /// deliberately stay presented — out of the retry path, recovered
        /// at next launch.
        case fallbackBanner(CompanionWake)
        /// Any reaction is proof the delivery reached him (#309).
        case stampWakeHeard(id: UUID)
        /// He engaged the banner: the wake's terminal upgrade.
        case engageWake(id: UUID)
        /// The beat that carried the resurfacing reached him — spare its
        /// agenda.
        case stampResurfacedHeard
        case openConversation(id: UUID)
        /// His reply becomes a followup wake due now: the next turn sees
        /// it with full situation context — one machinery, no side
        /// channel. The performer mints the wake (id/dates are its
        /// concern); the composed content is the decision.
        case bookReplyFollowup(content: String, conversationID: UUID?)
        /// Reactions accelerate the loop so replies feel instant.
        case accelerateEvaluation
    }

    /// What the granted turn does before running.
    enum TurnPlan: Equatable, Sendable {
        /// Drained nothing and no wake is due — no turn.
        case skip
        /// Present the batch: fire the due wakes (in order), then run the
        /// beat's resurfacing pass when the evaluator granted one.
        case present([Effect])
    }

    // MARK: - The fold turn

    /// The turn's opening writes. Pure — reads no clock but `now`.
    func begin(
        batch: [CompanionEvent], dueWakes: [CompanionWake], carriesBeat: Bool, now: Date
    ) -> TurnPlan {
        guard !batch.isEmpty || !dueWakes.isEmpty else { return .skip }
        var effects: [Effect] = dueWakes.map { wake in
            var fired = wake
            fired.state = .fired
            fired.firedAt = fired.firedAt ?? now
            return .fireWake(fired)
        }
        if carriesBeat {
            effects.append(.runResurfacingPass)
        }
        return .present(effects)
    }

    /// The turn's settlement — the invariant's home. A completed turn
    /// (`outcome` non-nil) consumes; a failed one re-presents everything
    /// while retries remain, then falls back to banners. The attempt
    /// ledger is keyed by the batch's earliest wake (or event) and cleared
    /// whole on any success — the shipped semantics.
    mutating func settle(
        batch: [CompanionEvent],
        wakes: [CompanionWake],
        outcome: (turnID: UUID, conversationID: UUID)?,
        now: Date
    ) -> [Effect] {
        if let outcome {
            turnAttempts.removeAll()
            var effects: [Effect] = []
            if !batch.isEmpty {
                effects.append(
                    .consumeEvents(ids: batch.map(\.id), turnID: outcome.turnID))
            }
            effects.append(
                contentsOf: wakes.map {
                    .deliverFiredWake(
                        id: $0.id, turnID: outcome.turnID,
                        conversationID: outcome.conversationID)
                })
            return effects
        }

        guard let key = wakes.first?.id ?? batch.first?.id else { return [] }
        let attempts = (turnAttempts[key] ?? 0) + 1
        turnAttempts[key] = attempts

        if attempts < Self.maxTurnAttempts {
            var effects: [Effect] = wakes.map { wake in
                var rebooked = wake
                rebooked.state = .booked
                return .rebookWake(rebooked)
            }
            if !batch.isEmpty {
                effects.append(.representEvents(ids: batch.map(\.id)))
            }
            return effects
        }

        turnAttempts[key] = nil
        return wakes.map { wake in
            var delivered = wake
            delivered.state = .delivered
            delivered.consumedAt = now
            return .fallbackBanner(delivered)
        }
    }

    // MARK: - Reactions

    /// What the owner's reaction to a posted banner writes. Heard is
    /// stamped first for every outcome — engage, reply, and wave-off alike
    /// are proof the delivery reached him (#309).
    func reaction(
        outcome: CompanionPingOutcome,
        wakeID: UUID?,
        conversationID: UUID?,
        note: String?
    ) -> [Effect] {
        var effects: [Effect] = []
        if let wakeID {
            effects.append(.stampWakeHeard(id: wakeID))
        }
        switch outcome {
        case .engaged:
            if let wakeID {
                effects.append(.engageWake(id: wakeID))
            }
            effects.append(.stampResurfacedHeard)
            if let conversationID {
                effects.append(.openConversation(id: conversationID))
            }
        case .replied:
            effects.append(.stampResurfacedHeard)
            guard let text = note, !text.isEmpty else { break }
            effects.append(
                .bookReplyFollowup(
                    content: "He replied to your notification: \"\(text)\" — respond.",
                    conversationID: conversationID))
            effects.append(.accelerateEvaluation)
        case .dismissed:
            break  // The heard stamp and the loop's dismissal record are the whole point.
        }
        return effects
    }
}
