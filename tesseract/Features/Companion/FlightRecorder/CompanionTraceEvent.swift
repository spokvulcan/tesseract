//
//  CompanionTraceEvent.swift
//  tesseract
//
//  The flight recorder's closed vocabulary (#393): the dot-namespaced event
//  names that every producer writes and the weekly aggregator reads, named
//  once as a typed enum. The rawValue is the wire string — the JSONL corpus
//  and the v0 import stay byte-for-byte readable — so the persisted line still
//  stores a `String` (`CompanionTraceRecord.event`); this enum is the seam the
//  producer API and the reader share, so a typo is a compile error, not a
//  silently-zeroed weekly metric.
//
//  Names on disk that aren't in this set (legacy v0 `beat.*`, a future
//  version's events) decode to `nil` at the reader and are skipped by name —
//  the vocabulary is closed for producers, forward-compatible for readers.
//
//  Follows the typed `CompletionTraceLine` precedent (ADR-0031); extends the
//  flight recorder (#326). Reaction names are the #391 Reaction vocabulary.
//

import Foundation

nonisolated enum CompanionTraceEvent: String, Sendable, Equatable, CaseIterable {

    // MARK: - wake.* — the promise/follow-up lifecycle

    /// A wake was scheduled (by the model's tool or the loop's follow-up book).
    case wakeBooked = "wake.booked"
    /// An existing wake's time or content was changed.
    case wakeRevised = "wake.revised"
    /// A booked wake was cancelled before it fired.
    case wakeCancelled = "wake.cancelled"
    /// A wake came due and entered delivery.
    case wakeFired = "wake.fired"
    /// A fired wake was consumed by a completed turn.
    case wakeConsumed = "wake.consumed"
    /// A crash-orphaned wake was put back on the agenda to re-present.
    case wakeRepresented = "wake.represented"
    /// An unheard delivered promise was resurfaced onto this beat's agenda.
    case wakeResurfaced = "wake.resurfaced"
    /// A promise was delivered but never heard — the resurfacing terminal.
    case wakeDeliveredUnheard = "wake.delivered-unheard"
    /// A promise was dropped — the zero-silent-drops defect signal (#313).
    case wakeDropped = "wake.dropped"

    // MARK: - delivery.* — the escalation ladder's rungs

    /// The delivery fell back to a notification when the spoken rung couldn't run.
    case deliveryFallback = "delivery.fallback"
    /// Delivered as a Notification Center banner.
    case deliveryNotification = "delivery.notification"
    /// Delivered as spoken words through the notch overlay.
    case deliverySpoken = "delivery.spoken"
    /// Delivered as a glyph state change (the quietest rung).
    case deliveryGlyph = "delivery.glyph"
    /// Delivered as a voice-overlay summons.
    case deliverySummons = "delivery.summons"

    // MARK: - reaction.* — the owner's answer to a delivery (#391)

    /// He clicked through to the app.
    case reactionEngaged = "reaction.engaged"
    /// He answered inline from the notification.
    case reactionReplied = "reaction.replied"
    /// He explicitly waved the delivery off.
    case reactionDismissed = "reaction.dismissed"
    /// A summons lapsed unanswered — the not-a-Reaction summons record.
    case reactionUnheard = "reaction.unheard"
    /// He interrupted the spoken reply — a voice barge-in.
    case reactionBargeIn = "reaction.barge-in"

    // MARK: - loop.* — the slow loop's environment

    /// Notification authorization was denied.
    case loopAuthDenied = "loop.auth-denied"
    /// Calendar access was denied.
    case loopCalendarDenied = "loop.calendar-denied"

    // MARK: - event.* — Mission Control's admission queue

    /// An event was admitted to the fold's queue.
    case eventAdmitted = "event.admitted"
    /// Crash-orphaned events were put back for re-presentation.
    case eventRepresented = "event.represented"

    // MARK: - instructions.* — the standing-instructions ledger

    /// The standing instructions were seeded on first run.
    case instructionsSeeded = "instructions.seeded"
    /// The model revised the standing instructions.
    case instructionsRevised = "instructions.revised"
    /// The sleep pass reviewed the instructions (kept or revised).
    case instructionsSleepReview = "instructions.sleep-review"
    /// The owner edited the standing instructions by hand.
    case instructionsOwnerEdited = "instructions.owner-edited"

    // MARK: - turn.* — the model-slot turn lifecycle

    /// A turn began.
    case turnStarted = "turn.started"
    /// A turn was deferred because the model slot was busy.
    case turnDeferred = "turn.deferred"
    /// A turn failed or was cancelled.
    case turnFailed = "turn.failed"
    /// A turn completed.
    case turnCompleted = "turn.completed"

    // MARK: - dialogue.* — the standing-conversation nudge

    /// A dialogue began (banner engage or summons).
    case dialogueBegan = "dialogue.began"
    /// A newer dialogue superseded a pending one.
    case dialogueSuperseded = "dialogue.superseded"
    /// A follow-up nudge was delivered into the open dialogue.
    case dialogueNudged = "dialogue.nudged"
    /// A follow-up nudge could not be delivered (busy or moved on).
    case dialogueNudgeMissed = "dialogue.nudge-missed"

    // MARK: - digest.* — the fold-down compaction

    /// A proposed digest was rejected (empty or too long).
    case digestRejected = "digest.rejected"
    /// A digest was folded into Mission Control.
    case digestFolded = "digest.folded"
    /// The digest generation failed.
    case digestFailed = "digest.failed"

    // MARK: - glyph.* — the presence glyph

    /// The presence glyph changed state.
    case glyphChanged = "glyph.changed"
    /// The entity's glyph notice was cleared.
    case glyphNoticeCleared = "glyph.notice-cleared"

    // MARK: - feedback.* — the owner's testimony (model-reported + incidents)

    /// A solicited reaction (the bookend answer).
    case feedbackSolicited = "feedback.solicited"
    /// An unprompted reaction.
    case feedbackSpontaneous = "feedback.spontaneous"
    /// He flagged a claim that never happened — a defect.
    case feedbackFabricationFlag = "feedback.fabrication-flag"
    /// He called the pinging noise annoying.
    case feedbackAnnoyance = "feedback.annoyance"
    /// He changed how firm he wants the entity.
    case feedbackDialChange = "feedback.dial-change"
    /// The Companion toggle was turned off — an incident, reviewed weekly.
    case feedbackToggleOff = "feedback.toggle-off"

    // MARK: - report-back.* — the agent's deposit door

    /// A report-back was deposited into Mission Control's queue.
    case reportBackDeposited = "report-back.deposited"

    // MARK: - hold.* — the tracking triage

    /// A perceived item was tracked as important-but-held.
    case holdTracked = "hold.tracked"

    // MARK: - voice.* — the voice session machine (#354, ADR-0042)

    /// A voice session was entered.
    case voiceSessionEntered = "voice.session-entered"
    /// A voice session exited.
    case voiceSessionExited = "voice.session-exited"
    /// The entity's spoken reply began.
    case voiceReplySpoken = "voice.reply-spoken"
    /// The owner's transcribed turn landed.
    case voiceOwnerTurn = "voice.owner-turn"
    /// The watchdog forced the utterance closed (missing done-callback).
    case voiceWatchdogExit = "voice.watchdog-exit"
    /// Stage one of a soft barge — an energy onset ducked the reply.
    case voiceBargeSoftOnset = "voice.barge-soft-onset"
    /// A soft barge was disproven and the reply resumed — a false onset.
    case voiceBargeFalseResume = "voice.barge-false-resume"
    /// A periodic energy sample while speaking (field-tuning evidence).
    case voiceEnergySample = "voice.energy-sample"
    /// A barge candidate was suppressed under the echo floor.
    case voiceBargeSuppressed = "voice.barge-suppressed"

    // MARK: - Derived families

    /// The reaction family derives from the ping outcome — never interpolated,
    /// so a new outcome forces a decision here (#391).
    init(reaction outcome: CompanionPingOutcome) {
        switch outcome {
        case .engaged: self = .reactionEngaged
        case .replied: self = .reactionReplied
        case .dismissed: self = .reactionDismissed
        }
    }

    /// The `log_feedback` tool's validated `kind` → its typed event. `nil` for
    /// any kind outside the tool's schema, so the mapping can't drift silently.
    init?(feedbackKind kind: String) {
        switch kind {
        case "solicited": self = .feedbackSolicited
        case "spontaneous": self = .feedbackSpontaneous
        case "fabrication-flag": self = .feedbackFabricationFlag
        case "annoyance": self = .feedbackAnnoyance
        case "dial-change": self = .feedbackDialChange
        default: return nil
        }
    }
}
