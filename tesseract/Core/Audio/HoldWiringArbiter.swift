//
//  HoldWiringArbiter.swift
//  tesseract
//
//  The **Hold Wiring Arbiter** (ADR-0050): the voice hold's async
//  arbitration as a pure value machine. While a detached wiring owns the
//  engine, every competing intent — a re-wire request, the hold ending,
//  the wiring itself landing — resolves *here*, to a verdict the engine
//  performs. This is the discipline that kills the 2026-07-17 tap-rewire
//  crash class: two wirings never touch one engine at once, and a stale
//  wiring's work is discarded on the stopped engine, never raced.
//

/// Pure state machine over the hold's four arbitration facts: is the
/// session's hold active, does a detached wiring own the engine right
/// now, which generation is current, and what request folded in while
/// the wiring ran. Holds no engine state — the graph flags
/// (`voiceHoldWired`, `holdRenderWired`, the hosted player) and every
/// AVFoundation call stay on `AudioCaptureEngine`; the engine's drivers
/// call one transition each and perform its verdict.
nonisolated struct HoldWiringArbiter: Sendable, Equatable {

    /// The session-level intent: `beginVoiceHold` has run and
    /// `endVoiceHold` has not. Set synchronously — the wiring commits
    /// asynchronously, so "active" and "wired" are distinct facts.
    private(set) var isHoldActive = false

    /// A detached wiring owns the engine right now. Every MainActor path
    /// that would touch the engine defers while this is true — racing
    /// the wiring is the crash class the arbiter exists to kill. Cleared
    /// only by a landing's `commit` / `discardAndIdle` verdict, never by
    /// the hold ending (the in-flight work still has to land and be
    /// discarded in order).
    private(set) var isWiringInFlight = false

    /// Staleness guard for wiring landings — bumped on every schedule
    /// and on the hold ending, so a landing only commits when nothing
    /// changed underneath it.
    private(set) var currentGeneration = 0

    /// A wiring request folded in while another wiring ran; the landing
    /// discards its stale outcome and starts this next. Value =
    /// rebuild-first, OR-folded: once any queued request wants a fresh
    /// engine, the coalesced one does.
    private(set) var queuedRebuildFirst: Bool?

    /// What `scheduleHoldWiring` must do with a (re)wiring request.
    enum ScheduleVerdict: Sendable, Equatable {
        /// Start a wiring now, stamped with `generation` so its landing
        /// can prove it is still current.
        case start(rebuildFirst: Bool, generation: Int)
        /// A wiring is in flight — the request folded into the queue and
        /// the landing will start it. Do nothing.
        case folded
    }

    /// What `endVoiceHold` must do with the wiring machinery.
    enum EndVerdict: Sendable, Equatable {
        /// No hold was active — the end is a no-op.
        case alreadyIdle
        /// A wiring is in flight: leave the engine alone; the landing
        /// sees the bumped generation and discards on the stopped
        /// engine. Touching the engine here would race that work.
        case leaveDiscardToCommit
        /// No wiring in flight — unwire the hold's graph state now.
        case unwireNow
    }

    /// What the commit hop must do with a landed wiring's outcome.
    enum LandingVerdict: Sendable, Equatable {
        /// The landing is current: adopt the outcome.
        case commit
        /// Stale, but a request queued behind it (and the hold is still
        /// active): discard the outcome on its stopped engine, then
        /// start the queued wiring stamped with `generation`. The
        /// wiring stays in flight across the handoff.
        case discardAndStartNext(rebuildFirst: Bool, generation: Int)
        /// Stale with nothing queued (or the hold ended): discard the
        /// outcome and return the machinery to idle.
        case discardAndIdle
    }

    /// The hold begins. Returns `false` (unchanged) when already active.
    mutating func beginHold() -> Bool {
        guard !isHoldActive else { return false }
        isHoldActive = true
        currentGeneration += 1
        return true
    }

    /// The hold ends: any in-flight wiring is stale from here on, and
    /// whatever request queued behind it dies with the hold.
    mutating func endHold() -> EndVerdict {
        guard isHoldActive else { return .alreadyIdle }
        isHoldActive = false
        currentGeneration += 1
        queuedRebuildFirst = nil
        return isWiringInFlight ? .leaveDiscardToCommit : .unwireNow
    }

    /// A (re)wiring request. Serial by construction: while a wiring is
    /// in flight the request folds into the queue rather than racing it.
    /// Always bumps the generation — an in-flight wiring is stale the
    /// moment a newer request exists.
    mutating func schedule(rebuildFirst: Bool) -> ScheduleVerdict {
        currentGeneration += 1
        if isWiringInFlight {
            queuedRebuildFirst = (queuedRebuildFirst ?? false) || rebuildFirst
            return .folded
        }
        isWiringInFlight = true
        return .start(rebuildFirst: rebuildFirst, generation: currentGeneration)
    }

    /// A detached wiring landed carrying the generation it was stamped
    /// with. Commits only when the hold is still active and nothing
    /// (re-schedule, hold end) bumped the generation while it flew;
    /// otherwise the outcome is discarded and the queued request — if
    /// the hold still wants one — starts in the same breath.
    mutating func wiringLanded(generation: Int) -> LandingVerdict {
        if isHoldActive, generation == currentGeneration {
            isWiringInFlight = false
            return .commit
        }
        let queued = queuedRebuildFirst
        queuedRebuildFirst = nil
        if let queued, isHoldActive {
            return .discardAndStartNext(
                rebuildFirst: queued, generation: currentGeneration)
        }
        isWiringInFlight = false
        return .discardAndIdle
    }
}
