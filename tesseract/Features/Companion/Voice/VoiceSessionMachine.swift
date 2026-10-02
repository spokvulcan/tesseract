//
//  VoiceSessionMachine.swift
//  tesseract
//
//  The **Voice Session Machine**: the pure reducer that owns every judgment
//  of the voice session's auto-listen loop — phases, the half-duplex turn
//  order (the mic is closed whenever speech plays, ADR-0082), **Barge-In**
//  by a key press or a click, the dead-capture recovery, the deaf window
//  after speech, the speaking watchdog, and the capture retry backoff. One
//  `handle(event, at:)` call folds an event into state and returns the
//  effects the performer must execute, in order — so a whole session
//  (listen → turn → reply → interrupt → listen → mutual silence) replays as a
//  decision table with no ticker, no CoreAudio, and no wall clock.
//
//  The same policy/performer split as the **Capture Engine Lifecycle**
//  (ADR-0025) and the seam ADR-0042 records: the machine decides,
//  `CompanionVoiceSessionController` performs. The `VoiceEndpointer` is
//  machine sub-state.
//
//  Time is an input: every `handle` takes `now` (reference-date seconds), and
//  the mic level, the capture engine's dead-input flag and the speech
//  engine's activity ride in on `.tick`. Effects carry no live reads.
//

import Foundation

nonisolated struct VoiceSessionMachine {

    // MARK: - Phase

    enum Phase: Equatable {
        case idle
        /// Mic open, waiting for the owner to start speaking.
        case listening
        /// The owner is speaking; trailing silence ends the turn.
        case capturing
        /// ASR + proofread on the closed take.
        case transcribing
        /// The turn is with the agent.
        case awaitingReply
        /// Jarvis is speaking and the mic is closed. A key press or a click
        /// stops him and opens it.
        case speaking
    }

    private(set) var phase: Phase = .idle
    var isActive: Bool { phase != .idle }

    // MARK: - Inputs

    /// The taste-ledger tunables (Settings), read live by the performer and
    /// carried on `.enter` and `.tick`; the machine keeps the last copy for
    /// the rare non-tick decision (≤ one tick stale).
    struct Tunables: Equatable, Sendable {
        var trailingSilence: TimeInterval
        var sessionTimeout: TimeInterval
        var autoSend: Bool
    }

    /// One 20 Hz sample of the outside world.
    struct Tick: Equatable, Sendable {
        /// The mic meter level (0–1, −60 dB-floor normalized).
        var level: Float
        /// The capture engine gave up on the open capture's input (its
        /// live-input check); the capture stays open until it is closed.
        var inputDead = false
        /// Whether the speech engine reads as active — the watchdog's input,
        /// and while listening, other speech the mic must stay closed for.
        var speechActive: Bool
        /// The engine state's description, for the watchdog-exit record.
        var speechDescription: String = ""
    }

    enum Event: Equatable, Sendable {
        case enter(via: String, tunables: Tunables)
        case exit(reason: String)
        case tick(Tick, tunables: Tunables)
        /// The owner interrupts: a key press or a click on the speaking
        /// line. Acts only while speech plays — the session's reply, or other
        /// speech holding the mic closed.
        case bargeIn(source: String)
        /// ChatSession's reply hook; `nil`/empty is a silent (pure tool) turn.
        case replyArrived(String?)
        /// The speak effect's success callback fired.
        case speechDone
        /// Feedback from `.openCapture`: the mic is live.
        case captureOpened
        /// Feedback from `.openCapture`: mic busy or start failure — resolves
        /// on a later tick at backoff cadence, never at tick cadence.
        case captureUnavailable
        /// The take died before it became a turn (no audio, empty, failed).
        case takeUnusable(reason: String)
        /// The take transcribed — committed text, or a rejected take's raw
        /// text ("a rejected proofread is still his words").
        case turnTranscribed(String)
    }

    // MARK: - Effects

    enum FeedState: Equatable, Sendable { case listening, thinking, speaking }

    /// What the performer must do, in order. Values only — every live read
    /// happens performer-side, every decision machine-side.
    enum Effect: Equatable, Sendable {
        case overlayBeginSession
        case overlayEndSession
        case feedState(FeedState)
        /// Settle the companion line, begin the spoken line, reveal it, and
        /// show `.speaking` — the one intent behind four feed calls.
        case presentSpokenReply(String)
        case settleOwnerLine(String)
        /// Start the capture; the performer answers with `.captureOpened` or
        /// `.captureUnavailable`.
        case openCapture
        /// Cancel the open capture (discard, no transcription).
        case closeCapture
        /// Stop the capture and run ASR + proofread on the take; outcomes
        /// come back as `.turnTranscribed` / `.takeUnusable`.
        case finishTake
        case speak(String)
        case stopSpeaking
        case send(String)
        case stageToComposer(String)
        /// A flight-recorder record; the performer stamps session and
        /// conversation IDs.
        case record(event: CompanionTraceEvent, snapshot: [String: String])
    }

    // MARK: - Session state

    private var tunables = Tunables(trailingSilence: 1.8, sessionTimeout: 30, autoSend: true)
    private var endpointer = VoiceEndpointer(config: .listening())
    private var captureOpen = false
    private var listeningSince: TimeInterval?
    private var speakingSince: TimeInterval?
    private var exchanges = 0
    /// Post-utterance grace: endpointer events are ignored until this
    /// deadline so the reply's room tail can't seed a turn.
    private var deafUntil: TimeInterval?
    /// Consecutive ticks the speech engine read as settled — the watchdog
    /// only exits `.speaking` on a sustained reading, never a single sample.
    private var settledTicks = 0
    /// A failed capture start retries no sooner than the backoff — persists
    /// across sessions (a busy mic stays busy through a toggle).
    private var lastCaptureAttemptFailedAt: TimeInterval?
    /// Captures the engine declared dead since the owner was last heard.
    private var deadCaptures = 0
    /// The owner's take is closed and still being transcribed. Opening a
    /// capture now would supersede it, so the mic waits for its outcome.
    private var takeInFlight = false
    /// Speech the session didn't start (the chat reading a reply aloud, a
    /// Companion line) is playing while it listens: the mic is held closed.
    private var heldForOtherSpeech = false

    // MARK: - Constants

    /// Ignore endpointer events for this long after an utterance ends —
    /// output-device latency plus room tail.
    static let postUtteranceGrace: TimeInterval = 0.3
    /// Settled ticks (× 50 ms) required before the watchdog exits `.speaking`.
    static let watchdogSettledTicks = 6
    /// A failed start (mic busy, engine refusing) retries no sooner than
    /// this. Without the backoff the 20 Hz ticker retried every 50 ms, and a
    /// failing `startCapture` can cost hundreds of ms of CoreAudio work per
    /// attempt on the main thread — the app-wide freeze in the 2026-07-17
    /// crash report.
    static let captureRetryBackoff: TimeInterval = 1.0
    /// Dead captures in a row (the owner not heard in between) that end the
    /// session: the microphone is gone, and listening to nothing helps no one.
    static let deadCapturesBeforeExit = 2

    // MARK: - The fold

    /// Fold one event into state; returns the effects to perform, in order.
    /// Total: events that don't fit the current phase fall out as `[]` —
    /// including *any* event in `.idle` except `.enter`, which is what makes
    /// a late transcription outcome after an exit provably inert.
    mutating func handle(_ event: Event, at now: TimeInterval) -> [Effect] {
        switch event {
        case .enter(let via, let tunables):
            return enter(via: via, tunables: tunables, now: now)
        case .exit(let reason):
            guard isActive else { return [] }
            return exitSession(reason: reason)
        case .tick(let tick, let tunables):
            guard isActive else { return [] }
            self.tunables = tunables
            return handleTick(tick, now: now)
        case .bargeIn(let source):
            guard phase == .speaking || heldForOtherSpeech else { return [] }
            return bargeIn(source: source, now: now)
        case .replyArrived(let text):
            guard phase == .awaitingReply || phase == .transcribing else { return [] }
            return replyArrived(text, now: now)
        case .speechDone:
            guard phase == .speaking else { return [] }
            return utteranceFinished(now: now)
        case .captureOpened:
            // A capture that opens after the session ended has no owner —
            // close it rather than leave the mic running.
            guard isActive else { return [.closeCapture] }
            captureOpen = true
            lastCaptureAttemptFailedAt = nil
            return []
        case .captureUnavailable:
            guard isActive else { return [] }
            // micBusy (dictation mid-take) or a start failure resolves on a
            // later tick — the session keeps listening state without a live
            // mic, retrying at backoff cadence, never at tick cadence.
            lastCaptureAttemptFailedAt = now
            return []
        case .takeUnusable:
            guard isActive else { return [] }
            return takeDied(now: now)
        case .turnTranscribed(let text):
            guard isActive else { return [] }
            return turnTranscribed(text, now: now)
        }
    }

    // MARK: - Entry / exit

    private mutating func enter(
        via: String, tunables: Tunables, now: TimeInterval
    ) -> [Effect] {
        guard phase == .idle else { return [] }
        self.tunables = tunables
        exchanges = 0
        deadCaptures = 0
        takeInFlight = false
        var effects: [Effect] = [
            .record(event: .voiceSessionEntered, snapshot: ["via": via]),
            .overlayBeginSession,
        ]
        effects += beginListening(now: now)
        return effects
    }

    private mutating func exitSession(reason: String) -> [Effect] {
        var effects: [Effect] = []
        if phase == .speaking { effects.append(.stopSpeaking) }
        if captureOpen { effects.append(.closeCapture) }
        captureOpen = false
        deafUntil = nil
        takeInFlight = false
        heldForOtherSpeech = false
        phase = .idle
        effects.append(
            .record(
                event: .voiceSessionExited,
                snapshot: ["reason": reason, "exchanges": String(exchanges)]))
        effects.append(.overlayEndSession)
        return effects
    }

    // MARK: - The reply

    private mutating func replyArrived(_ text: String?, now: TimeInterval) -> [Effect] {
        guard let text, !text.isEmpty else {
            // A silent reply (pure tool turn) — reopen the mic and move on,
            // unless the owner's take is still transcribing: its outcome
            // decides what comes next.
            return takeInFlight ? [] : beginListening(now: now)
        }
        // Half-duplex: the mic is closed before he speaks and stays closed
        // until he stops, so nothing he says can come back as the owner's turn.
        var effects: [Effect] = []
        if captureOpen {
            effects.append(.closeCapture)
            captureOpen = false
        }
        phase = .speaking
        speakingSince = now
        deafUntil = nil
        settledTicks = 0
        effects.append(.presentSpokenReply(text))
        effects.append(.speak(text))
        effects.append(
            .record(event: .voiceReplySpoken, snapshot: ["chars": String(text.count)]))
        return effects
    }

    /// The owner interrupts — by a key or a click, never by voice: the mic is
    /// closed while anything speaks. The speech stops at once and the mic
    /// opens for his turn; nothing resumes it.
    private mutating func bargeIn(source: String, now: TimeInterval) -> [Effect] {
        var snapshot = ["source": source]
        if phase == .speaking {
            snapshot["offsetSeconds"] = speakingOffsetSeconds(now: now)
        } else {
            snapshot["speech"] = "other"
        }
        return utteranceFinished(
            now: now, record: .record(event: .voiceBargeIn, snapshot: snapshot))
    }

    /// However the speech ended — drained, interrupted, or caught by the
    /// watchdog — TTS is stopped before the mic opens, and the endpointer
    /// stays deaf through the room tail. A take still transcribing keeps the
    /// mic closed: its outcome opens it.
    private mutating func utteranceFinished(
        now: TimeInterval, record: Effect? = nil
    ) -> [Effect] {
        var effects: [Effect] = [.stopSpeaking]
        if let record { effects.append(record) }
        guard !takeInFlight else {
            phase = .transcribing
            effects.append(.feedState(.thinking))
            return effects
        }
        effects += beginListening(gracePeriod: Self.postUtteranceGrace, now: now)
        return effects
    }

    // MARK: - The tick

    private mutating func handleTick(_ tick: Tick, now: TimeInterval) -> [Effect] {
        switch phase {
        case .listening:
            if captureOpen, tick.inputDead {
                return captureDied(now: now)
            }
            if tick.speechActive {
                // Speech the session didn't start: half-duplex holds for it
                // too, and the session timeout waits.
                heldForOtherSpeech = true
                guard captureOpen else { return [] }
                captureOpen = false
                return [.closeCapture]
            }
            if heldForOtherSpeech {
                // That speech ended: listen again, deaf through its tail.
                return beginListening(gracePeriod: Self.postUtteranceGrace, now: now)
            }
            if listen(tick, now: now) == .speechStarted {
                // He was heard, so the input is alive.
                deadCaptures = 0
                phase = .capturing
                return []
            }
            if let since = listeningSince, now - since > tunables.sessionTimeout {
                // Checked before any reopen: a capture opened on the tick
                // that ends the session would be left running.
                return exitSession(reason: "mutual-silence")
            }
            return captureOpen ? [] : attemptOpenCapture(now: now)

        case .capturing:
            if tick.speechActive {
                // Other speech started over his turn: the take closes now on
                // what he said, before the mic can hear it.
                return finishOwnerTurn(now: now)
            }
            // An input that dies mid-turn needs no rule here: the engine
            // zeroes the meter, so trailing silence closes the turn on what
            // arrived before it died.
            if listen(tick, now: now) == .endOfSpeech {
                return finishOwnerTurn(now: now)
            }
            return []

        case .speaking:
            // The mic is closed; only the watchdog runs.
            return watchdog(tick, now: now)

        case .idle, .transcribing, .awaitingReply:
            return []
        }
    }

    /// The endpointer hears the open capture, except through a deaf window.
    private mutating func listen(_ tick: Tick, now: TimeInterval) -> VoiceEndpointer.Event? {
        let deaf = deafUntil.map { now < $0 } ?? false
        guard captureOpen, !deaf else { return nil }
        return endpointer.ingest(level: tick.level, at: now)
    }

    /// The engine stopped without the success callback (an error, or an
    /// external stop). Exit only on a *sustained* settled reading — a single
    /// transient sample would cut a live reply short (the 2026-07-16 trace).
    private mutating func watchdog(_ tick: Tick, now: TimeInterval) -> [Effect] {
        settledTicks = isSpeechEngineSettled(tick: tick, now: now) ? settledTicks + 1 : 0
        guard settledTicks >= Self.watchdogSettledTicks else { return [] }
        settledTicks = 0
        return utteranceFinished(
            now: now,
            record: .record(
                event: .voiceWatchdogExit, snapshot: ["speechState": tick.speechDescription]))
    }

    /// After ~a second of grace, a settled engine means the utterance is over.
    private func isSpeechEngineSettled(tick: Tick, now: TimeInterval) -> Bool {
        guard let since = speakingSince, now - since > 1.0 else { return false }
        return !tick.speechActive
    }

    /// The capture engine gave up on the open capture's input: close it,
    /// record it, and reopen on the backoff — the next capture starts on a
    /// fresh engine. Two in a row without the owner heard in between end the
    /// session.
    private mutating func captureDied(now: TimeInterval) -> [Effect] {
        captureOpen = false
        deadCaptures += 1
        lastCaptureAttemptFailedAt = now
        var effects: [Effect] = [
            .closeCapture,
            .record(event: .voiceCaptureDead, snapshot: ["count": String(deadCaptures)]),
        ]
        if deadCaptures >= Self.deadCapturesBeforeExit {
            effects += exitSession(reason: "capture-dead")
        }
        return effects
    }

    // MARK: - Turn plumbing

    private mutating func beginListening(
        gracePeriod: TimeInterval = 0, now: TimeInterval
    ) -> [Effect] {
        var effects = reopenCapture(now: now)
        endpointer.reset(config: .listening(trailingSilence: tunables.trailingSilence))
        deafUntil = gracePeriod > 0 ? now + gracePeriod : nil
        heldForOtherSpeech = false
        phase = .listening
        listeningSince = now
        effects.append(.feedState(.listening))
        return effects
    }

    private mutating func finishOwnerTurn(now: TimeInterval) -> [Effect] {
        phase = .transcribing
        var effects: [Effect] = [.feedState(.thinking)]
        guard captureOpen else {
            effects += beginListening(now: now)
            return effects
        }
        captureOpen = false
        takeInFlight = true
        effects.append(.finishTake)
        return effects
    }

    /// The take in flight produced nothing usable: listen again, deaf for a
    /// beat in case a reply only just stopped — or, while a reply still
    /// speaks, leave the mic closed until it ends. Any other unusable outcome
    /// is stale.
    private mutating func takeDied(now: TimeInterval) -> [Effect] {
        guard takeInFlight else { return [] }
        takeInFlight = false
        guard phase == .transcribing else { return [] }
        return beginListening(gracePeriod: Self.postUtteranceGrace, now: now)
    }

    private mutating func turnTranscribed(_ text: String, now: TimeInterval) -> [Effect] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return takeDied(now: now) }
        takeInFlight = false
        var effects: [Effect] = []
        if phase == .speaking {
            // A reply that landed while this take transcribed: his words are
            // newer, so it stops before his turn goes out.
            effects.append(.stopSpeaking)
            phase = .transcribing
        }
        exchanges += 1
        effects.append(
            .record(
                event: .voiceOwnerTurn, snapshot: ["chars": String(trimmed.count)]))
        guard tunables.autoSend else {
            // The escape hatch (#310 taste ledger): stage, never send.
            effects.append(.stageToComposer(trimmed))
            effects += exitSession(reason: "staged-to-composer")
            return effects
        }
        effects.append(.settleOwnerLine(trimmed))
        effects.append(.feedState(.thinking))
        phase = .awaitingReply
        effects.append(.send(trimmed))
        return effects
    }

    // MARK: - Capture plumbing

    /// Try to open the capture unless the backoff is still cooling; the
    /// performer answers with `.captureOpened` / `.captureUnavailable`.
    private mutating func attemptOpenCapture(now: TimeInterval) -> [Effect] {
        guard !captureOpen else { return [] }
        if let failedAt = lastCaptureAttemptFailedAt,
            now - failedAt < Self.captureRetryBackoff
        {
            return []
        }
        return [.openCapture]
    }

    private mutating func reopenCapture(now: TimeInterval) -> [Effect] {
        var effects: [Effect] = []
        if captureOpen {
            effects.append(.closeCapture)
            captureOpen = false
        }
        effects += attemptOpenCapture(now: now)
        return effects
    }

    /// How far into the spoken reply the event landed, for barge records.
    private func speakingOffsetSeconds(now: TimeInterval) -> String {
        String(format: "%.1f", speakingSince.map { now - $0 } ?? 0)
    }
}
