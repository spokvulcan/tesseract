//
//  CompanionTraceEvent.swift
//  tesseract
//
//  The Companion Trace's closed vocabulary: every event name a producer may
//  write, named once as a typed enum so a typo is a compile error rather than
//  a silently empty query. The rawValue is the wire string in the JSONL files.
//  Names on disk that are not in this set decode to nil at the reader.
//
//  Follows the Trace Vocabulary pattern of the completion trace (ADR-0031).
//

import Foundation

nonisolated enum CompanionTraceEvent: String, Sendable, Equatable, CaseIterable {

    // MARK: - moment.*: one model call at a moment

    /// A moment's model call began. Carries the trigger and the model.
    case momentStarted = "moment.started"
    /// A moment produced a valid card. Carries tokens, latency and the
    /// thermal and power state.
    case momentFinished = "moment.finished"
    /// A moment's call failed or produced no valid card; the fallback card
    /// was used instead.
    case momentFailed = "moment.failed"

    // MARK: - card.*: what reached the owner, and what he did with it

    /// A card was put on a delivery rung (glyph, panel, banner, voice).
    case cardPresented = "card.presented"
    /// The owner acted on a card or one of its items. Carries the time to react.
    case cardReaction = "card.reaction"

    // MARK: - cue.*: the plan (and the body) keeping time

    /// A planned step's slot started and code put it on the panel (a Step
    /// Cue). Carries how late it went up and the slot's length — or, with
    /// phase "break", two hours at the Mac put a Break Cue up: how long.
    case cuePresented = "cue.presented"
    /// The owner answered a Step Cue (start, later, resume after a meeting,
    /// tomorrow, done, undo — Done taken back — or dismiss), or a Break Cue (phase "break": taking,
    /// later, dismiss, or away — up from the Mac with it on the panel).
    /// Carries the time to react.
    case cueReaction = "cue.reaction"
    /// A meeting landed on a planned step not yet begun: code moved the step
    /// to the first free time after it. Carries how far, and its length.
    case cueMoved = "cue.moved"
    /// A step the owner started whole was seen done in time: the minutes it
    /// had and the minutes it took, what the Morning Plan's pace is made of.
    case cueTimed = "cue.timed"

    // MARK: - nudge.*: OS-scheduled notifications for events and reminders

    /// A nudge was scheduled with the OS.
    case nudgeScheduled = "nudge.scheduled"
    /// A scheduled nudge was withdrawn after the agenda changed.
    case nudgeCancelled = "nudge.cancelled"
    /// A nudge was delivered while Tesseract was running.
    case nudgeFired = "nudge.fired"

    // MARK: - notification.*: other apps' banners and the seen ledger

    /// Another app's banner was observed.
    case notificationArrived = "notification.arrived"
    /// The owner saw a notification (its app came to the front, or he clicked it).
    case notificationSeen = "notification.seen"
    /// A batch of unresolved notifications was triaged.
    case notificationTriaged = "notification.triaged"

    // MARK: - agent.*: coding agents reporting in

    /// A coding agent's hook posted a signal (waiting or finished).
    case agentSignal = "agent.signal"

    // MARK: - agenda.*: Reminders and Calendar

    /// A reminder or event was created, completed, moved or deleted.
    case agendaChanged = "agenda.changed"

    // MARK: - fact.* and profile.*: the owner's Profile

    /// Jarvis proposed a fact ("Should I remember this?").
    case factProposed = "fact.proposed"
    /// The owner kept, edited or dropped a proposal.
    case factDecided = "fact.decided"
    /// A Profile fact was added, edited or deleted.
    case profileChanged = "profile.changed"

    // MARK: - task.*: tasks the day showed, proposed for Reminders

    /// The Night Reflection proposed a task the day showed. Carries the count.
    case taskProposed = "task.proposed"
    /// The owner added a proposed task or let it go.
    case taskDecided = "task.decided"

    // MARK: - thread.*: the Day Thread

    /// A new Day Thread opened with its Day Opening.
    case threadOpened = "thread.opened"
    /// A Day Thread passed its ceiling and was compacted.
    case threadCompacted = "thread.compacted"

    // MARK: - presence.*

    /// The owner came back from five minutes away or more — not the night,
    /// nor the app starting: how long they had sat at the Mac before it
    /// (minutes; from a Taking 5 if that came later) and how long they were
    /// away. What the Break Cue's two hours are tuned by.
    case breakTaken = "presence.break"

    // MARK: - night.*

    /// Quiet hours began with the owner at the Mac: the night's one banner
    /// said when tomorrow starts. Carries the minutes until then.
    case windDown = "night.wind-down"
    /// The day's first sit-down after the night: how long the Mac was left
    /// (minutes) and whether the owner was at it past midnight — the
    /// measure of whether the wind-down and a light day change the nights.
    case nightEnded = "night.ended"

    // MARK: - governor.*

    /// Non-urgent model work was deferred (thermal state, battery).
    case governorDeferred = "governor.deferred"

    // MARK: - migration.*

    /// The retired memory store and Mission Control conversation were deleted.
    case migrationWiped = "migration.wiped"

    // MARK: - voice.*: the voice session machine (ADR-0042)

    /// A voice session opened.
    case voiceSessionEntered = "voice.session-entered"
    /// A voice session closed. Carries the reason and the exchange count.
    case voiceSessionExited = "voice.session-exited"
    /// A spoken reply was played.
    case voiceReplySpoken = "voice.reply-spoken"
    /// The owner finished a spoken turn.
    case voiceOwnerTurn = "voice.owner-turn"
    /// The speech watchdog ended a stuck utterance.
    case voiceWatchdogExit = "voice.watchdog-exit"
    /// The owner interrupted a spoken reply (a key or a click). Carries the
    /// source and how far into the reply it landed.
    case voiceBargeIn = "voice.barge-in"
    /// The capture engine found the open capture's input dead; the session
    /// closed it and reopens on the backoff.
    case voiceCaptureDead = "voice.capture-dead"
}
