//
//  VoiceSessionMachine.swift
//  tesseract
//
//  The **Voice Session Machine**: the pure reducer that owns every judgment
//  of the voice session's auto-listen loop — phases, the two-stage
//  **Barge-In** (Soft Barge → hard pause), the **Echo Floor** consultation,
//  the false-barge escalation ladder (#354), the deaf windows, the speaking
//  watchdog, and the capture retry backoff. One `handle(event, at:)` call
//  folds an event into state and returns the effects the performer must
//  execute, in order — so a whole session (listen → speak → false barge →
//  flap → confirm → turn → mutual silence) replays as a decision table with
//  no ticker, no CoreAudio, and no wall clock.
//
//  The same policy/performer split as the **Capture Engine Lifecycle**
//  (ADR-0025) and the seam ADR-0042 records: the machine decides,
//  `CompanionVoiceSessionController` performs. The machine owns the real
//  `VoiceEndpointer` and `EchoResidualFloor` as sub-state, so the calibration
//  lock (`VoiceBargeReplayTests`) replays hardware traces through the exact
//  shipped path — floor ingest → threshold → endpointer → reaction — instead
//  of a hand-copied mirror of it.
//
//  Time is an input: every `handle` takes `now` (reference-date seconds), and
//  meter/playback levels ride in on `.tick`. Energy-event snapshots on
//  non-tick events use the last tick's levels (≤ one tick stale, diagnostic
//  only). Effects carry no live reads.
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
        /// Jarvis is speaking; the mic is open underneath for barge-in.
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
        var bargeInLevel: Float
        var autoSend: Bool
    }

    /// One 20 Hz sample of the outside world.
    struct Tick: Equatable, Sendable {
        /// The mic meter level (0–1, −60 dB-floor normalized).
        var level: Float
        /// The reply's loudness at the playback head — the Echo Floor's
        /// far-end signal.
        var playbackLevel: Float
        /// Whether the speech engine reads as active — the watchdog's input.
        var speechActive: Bool
        /// The engine state's description, for the watchdog-exit record.
        var speechDescription: String = ""
    }

    enum Event: Equatable, Sendable {
        case enter(via: String, tunables: Tunables)
        case exit(reason: String)
        case tick(Tick, tunables: Tunables)
        /// The overlay's deliberate barge (or any non-energy detector).
        case clickBarge(source: String)
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
        case beginVoiceHold
        case endVoiceHold
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
        case pauseSpeaking
        case resumeSpeaking
        case fadeSpeech(target: Float, duration: TimeInterval)
        case send(String)
        case stageToComposer(String)
        /// A flight-recorder record; the performer stamps session and
        /// conversation IDs.
        case record(event: CompanionTraceEvent, snapshot: [String: String])
    }

    // MARK: - Soft Barge (two-stage barge-in, ADR-0041)

    /// How an energy onset during playback resolves: an instant duck opens a
    /// confirm window; only sustained voicing turns the duck into a pause.
    enum SoftBargeVerdict: Equatable {
        /// Real voicing — commit the hard pause and capture the take.
        case confirm
        /// The window closed without voicing — restore the reply's volume.
        case fadeBack
        /// The window is still open and voicing hasn't accumulated yet.
        case keepWaiting
    }

    /// Pure resolution of a soft barge — pinned by tests. Confirms *early*
    /// once voicing accumulates (a real interruption should not wait out the
    /// window); `voicedSeconds` is loud-time since the onset reset, never
    /// `isInSpeech` (which holds through 1.8 s of trailing silence and would
    /// confirm every false fire).
    static func resolveSoftBarge(
        voicedSeconds: TimeInterval, elapsed: TimeInterval,
        confirmWindow: TimeInterval, confirmVoiced: TimeInterval
    ) -> SoftBargeVerdict {
        if voicedSeconds >= confirmVoiced { return .confirm }
        return elapsed >= confirmWindow ? .fadeBack : .keepWaiting
    }

    // MARK: - Session state

    /// Where the current barge stands — the single source of truth for a
    /// barged reply awaiting resolution: `.soft` = reply ducked, the confirm
    /// window open since `startedAt`; `.hard` = reply paused (confirmed
    /// voicing or a click), false unless a speech onset lands within the
    /// verify window of `verifyStartedAt`.
    private enum BargeMode: Equatable {
        case none
        case soft(startedAt: TimeInterval)
        case hard(verifyStartedAt: TimeInterval)

        var isHard: Bool { if case .hard = self { true } else { false } }
    }

    private var tunables = Tunables(
        trailingSilence: 1.8, sessionTimeout: 30, bargeInLevel: 0.25, autoSend: true)
    private var endpointer = VoiceEndpointer(config: .listening())
    private var captureOpen = false
    private var listeningSince: TimeInterval?
    private var speakingSince: TimeInterval?
    private var speechDoneCallbackSeen = false
    private var exchanges = 0
    private var bargeMode: BargeMode = .none
    /// A barged (ducked or paused) reply awaits resolution: resume or turn.
    private var bargedUtterance: Bool { bargeMode != .none }
    /// The Echo Floor (ADR-0041): tracked self-echo residual while the reply
    /// plays; the energy barge threshold always sits `margin` above it.
    private var echoFloor = EchoResidualFloor()
    /// False energy barges on the *current* utterance — the escalation
    /// ladder's input (#354): ≥2 widens the floor margin, ≥4 mutes the
    /// energy detector for the rest of the utterance (the click keeps
    /// working).
    private var falseBargeCount = 0
    /// Tick counter for the 1 Hz energy-sample cadence while speaking.
    private var tickCount = 0
    /// Post-utterance grace: endpointer events are ignored until this
    /// deadline so the reply's room tail can't seed a turn.
    private var deafUntil: TimeInterval?
    /// Consecutive ticks the speech engine read as settled — the watchdog
    /// only exits `.speaking` on a sustained reading, never a single sample.
    private var settledTicks = 0
    /// A failed capture start retries no sooner than the backoff — persists
    /// across sessions (a busy mic stays busy through a toggle).
    private var lastCaptureAttemptFailedAt: TimeInterval?
    /// The last tick's meter levels — the energy snapshot's inputs when a
    /// non-tick event records (≤ one tick stale, diagnostic only).
    private var lastLevel: Float = 0
    private var lastPlaybackLevel: Float = 0

    /// The tracked Echo Floor level — a read-only diagnostic for the replay
    /// tests' floor assertions.
    var echoFloorLevel: Float { echoFloor.floor }

    // MARK: - Constants

    /// Ignore endpointer events for this long after an utterance ends —
    /// output-device latency plus room tail.
    static let postUtteranceGrace: TimeInterval = 0.3
    /// A barge with no speech onset within this window is false — resume.
    static let bargeVerifyWindow: TimeInterval = 2.0
    /// Settled ticks (× 50 ms) required before the watchdog exits `.speaking`.
    static let watchdogSettledTicks = 6

    // Soft Barge constants (ADR-0041; calibrated by the voice-hold lab).
    /// The duck target — quiet enough that residual falls well under the
    /// listening threshold, loud enough that a false fire stays a murmur.
    static let softDuckLevel: Float = 0.25
    static let softDuckRampDown: TimeInterval = 0.1
    static let softDuckRampUp: TimeInterval = 0.2
    /// The confirm window an energy onset opens, and the voiced time inside
    /// it that commits the hard pause.
    static let softBargeConfirmWindow: TimeInterval = 0.8
    static let softBargeConfirmVoiced: TimeInterval = 0.3
    /// Endpointer deafness after a false-barge resume/fade-back — the fade-up
    /// and AEC re-settling transient must not re-trigger the detector
    /// (observed re-barge onsets ~0.85 s post-resume, flight 2026-07-17).
    static let postResumeGrace: TimeInterval = 1.0
    // The escalation ladder (#354 item 2), per utterance.
    static let escalateMarginAfter = 2
    static let escalatedMarginScale: Float = 1.5
    static let energyBargeMuteAfter = 4
    /// Every 20th tick (1 Hz) records an energy sample while speaking.
    static let energySampleEveryTicks = 20
    /// A failed start (mic busy, engine refusing) retries no sooner than
    /// this. Without the backoff the 20 Hz ticker retried every 50 ms, and a
    /// failing `startCapture` can cost hundreds of ms of CoreAudio work per
    /// attempt on the main thread — the app-wide freeze in the 2026-07-17
    /// crash report.
    static let captureRetryBackoff: TimeInterval = 1.0

    /// The listening-mode start threshold — the endpointer's static default,
    /// deliberately not a Setting.
    private var speechLevel: Float { VoiceEndpointer.Config.listening().speechLevel }

    // The escalation ladder (#354), both rungs derived from `falseBargeCount`.
    private var escalationMarginScale: Float {
        falseBargeCount >= Self.escalateMarginAfter ? Self.escalatedMarginScale : 1.0
    }

    private var energyBargeMuted: Bool {
        falseBargeCount >= Self.energyBargeMuteAfter
    }

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
        case .clickBarge(let source):
            guard isActive else { return [] }
            return clickBarge(source: source, now: now)
        case .replyArrived(let text):
            guard phase == .awaitingReply || phase == .transcribing else { return [] }
            return replyArrived(text, now: now)
        case .speechDone:
            guard isActive else { return [] }
            speechDoneCallbackSeen = true
            return utteranceFinished(now: now)
        case .captureOpened:
            guard isActive else { return [] }
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
        case .takeUnusable(let reason):
            guard isActive else { return [] }
            return abandonTake(reason: reason, now: now)
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
        tickCount = 0
        // The hold's detached wiring starts now — it has ~2.3 s on this
        // hardware (lab E6) to land while the overlay appears; captures
        // fast-fail into the 1 s backoff until the mic is live.
        var effects: [Effect] = [
            .beginVoiceHold,
            .record(event: .voiceSessionEntered, snapshot: ["via": via]),
            .overlayBeginSession,
        ]
        effects += beginListening(now: now)
        return effects
    }

    private mutating func exitSession(reason: String) -> [Effect] {
        var effects: [Effect] = []
        if phase == .speaking || bargedUtterance { effects.append(.stopSpeaking) }
        bargeMode = .none
        deafUntil = nil
        if captureOpen { effects.append(.closeCapture) }
        captureOpen = false
        // The capture is closed first, so the hold ends on a free engine.
        effects.append(.endVoiceHold)
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
            // A silent reply (pure tool turn) — reopen the mic and move on.
            return beginListening(now: now)
        }
        phase = .speaking
        speakingSince = now
        speechDoneCallbackSeen = false
        bargeMode = .none
        deafUntil = nil
        settledTicks = 0
        echoFloor.reset()
        falseBargeCount = 0
        endpointer.reset(config: .bargeIn(speechLevel: tunables.bargeInLevel))
        var effects: [Effect] = [.presentSpokenReply(text)]
        // The mic opens *under* the utterance: VPIO's AEC keeps his voice out
        // of the input, and the Echo Floor rides whatever residual remains,
        // so speech energy over both is the owner (#310 §4, ADR-0041).
        effects += attemptOpenCapture(now: now)
        effects.append(.speak(text))
        effects.append(
            .record(event: .voiceReplySpoken, snapshot: ["chars": String(text.count)]))
        return effects
    }

    // MARK: - Barge-in

    /// The hard (immediate-pause) barge — the overlay's click, which is
    /// deliberate and has zero false positives in the field data. A click
    /// while a Soft Barge is verifying commits it instead.
    private mutating func clickBarge(source: String, now: TimeInterval) -> [Effect] {
        if case .soft = bargeMode {
            return hardenSoftBarge(detector: source, now: now)
        }
        guard phase == .speaking else { return [] }
        // Pause, don't stop (pause-on-barge): a false barge resumes the
        // reply where it left off; only a committed turn makes the
        // interruption permanent.
        bargeMode = .hard(verifyStartedAt: now)
        settledTicks = 0
        var effects: [Effect] = [
            .pauseSpeaking,
            energyRecord(
                .reactionBargeIn,
                extra: ["offsetSeconds": speakingOffsetSeconds(now: now), "detector": source]),
        ]
        // Fresh take from the interruption on — the playback-period audio
        // (echo-cancelled silence) is dropped, not transcribed.
        effects += reopenCapture(now: now)
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: tunables.trailingSilence))
        if source != "click" {
            // Energy barge: he is already mid-word — seed the candidate so
            // the start debounce measures from the interruption itself.
            _ = endpointer.ingest(level: 1.0, at: now)
        }
        phase = .capturing
        effects.append(.feedState(.listening))
        return effects
    }

    /// Stage one of the Soft Barge: an energy onset ducks the reply and
    /// opens the confirm window — capture starts now so no owner words are
    /// lost, but the reply is not paused until voicing confirms.
    private mutating func softBargeIn(now: TimeInterval) -> [Effect] {
        guard phase == .speaking else { return [] }
        bargeMode = .soft(startedAt: now)
        settledTicks = 0
        var effects: [Effect] = [
            energyRecord(
                .voiceBargeSoftOnset,
                extra: ["offsetSeconds": speakingOffsetSeconds(now: now)]),
            .fadeSpeech(target: Self.softDuckLevel, duration: Self.softDuckRampDown),
        ]
        // Fresh take from the onset — mirrors the hard barge; the ducked
        // reply (−12 dB) reads well under the listening threshold, so
        // voicing accumulated from here on is the owner.
        effects += reopenCapture(now: now)
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: tunables.trailingSilence))
        _ = endpointer.ingest(level: 1.0, at: now)
        phase = .capturing
        effects.append(.feedState(.listening))
        return effects
    }

    /// Stage two, confirmed: voicing sustained through the confirm window —
    /// the duck becomes the real pause and the take proceeds as a hard barge.
    private mutating func hardenSoftBarge(detector: String, now: TimeInterval) -> [Effect] {
        guard case .soft = bargeMode else { return [] }
        bargeMode = .hard(verifyStartedAt: now)
        return [
            .pauseSpeaking,
            energyRecord(
                .reactionBargeIn,
                extra: [
                    "offsetSeconds": speakingOffsetSeconds(now: now), "detector": detector,
                ]),
        ]
    }

    /// Drives an open confirm window each tick: accumulated voicing (or a
    /// closed take) hardens the duck into the real pause; an expired window
    /// fades the reply back.
    private mutating func tickSoftBarge(
        startedAt: TimeInterval, event: VoiceEndpointer.Event?, now: TimeInterval
    ) -> [Effect] {
        if event == .endOfSpeech {
            // A closed take must resolve through the hard path — reachable
            // when the trailing-silence Setting undercuts the confirm window.
            var effects = hardenSoftBarge(detector: "energy-soft", now: now)
            effects += finishOwnerTurn(now: now)
            return effects
        }
        switch Self.resolveSoftBarge(
            voicedSeconds: endpointer.voicedSeconds,
            elapsed: now - startedAt,
            confirmWindow: Self.softBargeConfirmWindow,
            confirmVoiced: Self.softBargeConfirmVoiced)
        {
        case .keepWaiting:
            return []
        case .confirm:
            return hardenSoftBarge(detector: "energy-soft", now: now)
        case .fadeBack:
            // Stage two, unconfirmed: the window closed without voicing —
            // restore the reply's volume and re-arm. The false fire cost a
            // ~1 s murmur.
            return restoreReplyAfterFalseBarge(
                reason: "soft-fadeback", resumePlayback: false, now: now)
        }
    }

    /// The barge produced nothing that counts as a turn — resume the paused
    /// reply and rearm barge-in detection.
    private mutating func resumeAfterFalseBarge(
        reason: String, now: TimeInterval
    ) -> [Effect] {
        guard bargedUtterance else { return [] }
        return restoreReplyAfterFalseBarge(
            reason: reason, resumePlayback: bargeMode.isHard, now: now)
    }

    /// The false-barge restore both stages share: count the fire toward the
    /// escalation ladder (#354), record it, and re-arm the barge watch deaf
    /// through the restore transient. `resumePlayback` un-pauses a
    /// hard-barged reply; a soft one only ducked and just fades back up.
    private mutating func restoreReplyAfterFalseBarge(
        reason: String, resumePlayback: Bool, now: TimeInterval
    ) -> [Effect] {
        bargeMode = .none
        falseBargeCount += 1
        var effects: [Effect] = []
        if captureOpen {
            effects.append(.closeCapture)
            captureOpen = false
        }
        effects.append(energyRecord(.voiceBargeFalseResume, extra: ["reason": reason]))
        if speechDoneCallbackSeen {
            // The utterance drained while barged — nothing to restore.
            effects.append(.fadeSpeech(target: 1.0, duration: 0))
            effects += beginListening(now: now)
            return effects
        }
        endpointer.reset(config: .bargeIn(speechLevel: tunables.bargeInLevel))
        settledTicks = 0
        effects += attemptOpenCapture(now: now)
        // Deaf through the restore: the volume ramp / resume and AEC
        // re-settling must not re-trigger the detector (the 2026-07-17
        // flap cycle).
        deafUntil = now + Self.postResumeGrace
        phase = .speaking
        effects.append(.feedState(.speaking))
        if resumePlayback { effects.append(.resumeSpeaking) }
        effects.append(.fadeSpeech(target: 1.0, duration: Self.softDuckRampUp))
        return effects
    }

    /// The take produced nothing usable — resume a paused reply if one is
    /// waiting, otherwise reopen the mic and listen.
    private mutating func abandonTake(reason: String, now: TimeInterval) -> [Effect] {
        if bargedUtterance {
            return resumeAfterFalseBarge(reason: reason, now: now)
        }
        return beginListening(now: now)
    }

    // MARK: - The tick

    private mutating func handleTick(_ tick: Tick, now: TimeInterval) -> [Effect] {
        tickCount += 1
        lastLevel = tick.level
        lastPlaybackLevel = tick.playbackLevel
        let deaf = deafUntil.map { now < $0 } ?? false
        // The Echo Floor tracks through deafness — residual keeps arriving
        // whether or not the endpointer is allowed to react to it.
        if phase == .speaking {
            echoFloor.ingest(
                micLevel: tick.level, playbackLevel: tick.playbackLevel, at: now)
        }
        let speechFloor: Float? =
            phase == .speaking
            ? echoFloor.threshold(
                atLeast: tunables.bargeInLevel, marginScale: escalationMarginScale)
            : nil
        let event =
            captureOpen && !deaf
            ? endpointer.ingest(level: tick.level, at: now, speechFloor: speechFloor)
            : nil

        switch phase {
        case .listening:
            var effects: [Effect] = []
            if !captureOpen { effects += attemptOpenCapture(now: now) }
            if event == .speechStarted {
                phase = .capturing
            } else if let since = listeningSince, now - since > tunables.sessionTimeout {
                effects += exitSession(reason: "mutual-silence")
            }
            return effects

        case .capturing:
            if case .soft(let startedAt) = bargeMode {
                return tickSoftBarge(startedAt: startedAt, event: event, now: now)
            } else if event == .endOfSpeech {
                return finishOwnerTurn(now: now)
            } else if case .hard(let since) = bargeMode, !endpointer.isInSpeech,
                now - since > Self.bargeVerifyWindow
            {
                return resumeAfterFalseBarge(reason: "no-speech", now: now)
            }
            return []

        case .speaking:
            var effects: [Effect] = []
            if tickCount % Self.energySampleEveryTicks == 0 {
                effects.append(energyRecord(.voiceEnergySample, extra: [:]))
            }
            if event == .speechStarted {
                if energyBargeMuted {
                    // The escalation ladder's top: the detector cried wolf
                    // ≥4 times this utterance — log, never react.
                    effects.append(energyRecord(.voiceBargeSuppressed, extra: [:]))
                } else {
                    effects += softBargeIn(now: now)
                }
            } else if !speechDoneCallbackSeen {
                // The engine stopped without the success callback (error, or
                // an external stop). Exit only on a *sustained* settled
                // reading — a single transient sample reopened the mic under
                // live TTS (the 2026-07-16 self-echo trace, ADR-0041).
                if isSpeechEngineSettled(tick: tick, now: now) {
                    settledTicks += 1
                } else {
                    settledTicks = 0
                }
                if settledTicks >= Self.watchdogSettledTicks {
                    settledTicks = 0
                    effects.append(
                        .record(
                            event: .voiceWatchdogExit,
                            snapshot: ["speechState": tick.speechDescription]))
                    effects += utteranceFinished(now: now)
                }
            }
            return effects

        case .idle, .transcribing, .awaitingReply:
            return []
        }
    }

    /// After ~a second of grace, a settled engine means the utterance is over.
    private func isSpeechEngineSettled(tick: Tick, now: TimeInterval) -> Bool {
        guard let since = speakingSince, now - since > 1.0 else { return false }
        return !tick.speechActive
    }

    private mutating func utteranceFinished(now: TimeInterval) -> [Effect] {
        guard phase == .speaking else { return [] }
        // Force-stop: however the utterance ended, TTS must be provably
        // silent before the mic reopens in listening config. On the normal
        // path this is a no-op sweep; on a watchdog exit it is the fix.
        var effects: [Effect] = [.stopSpeaking]
        bargeMode = .none
        effects += beginListening(gracePeriod: Self.postUtteranceGrace, now: now)
        return effects
    }

    // MARK: - Turn plumbing

    private mutating func beginListening(
        gracePeriod: TimeInterval = 0, now: TimeInterval
    ) -> [Effect] {
        var effects = reopenCapture(now: now)
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: tunables.trailingSilence))
        deafUntil = gracePeriod > 0 ? now + gracePeriod : nil
        phase = .listening
        listeningSince = now
        effects.append(.feedState(.listening))
        return effects
    }

    private mutating func finishOwnerTurn(now: TimeInterval) -> [Effect] {
        phase = .transcribing
        var effects: [Effect] = [.feedState(.thinking)]
        guard captureOpen else {
            effects += abandonTake(reason: "no-capture", now: now)
            return effects
        }
        captureOpen = false
        effects.append(.finishTake)
        return effects
    }

    private mutating func turnTranscribed(_ text: String, now: TimeInterval) -> [Effect] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return abandonTake(reason: "empty", now: now)
        }
        var effects: [Effect] = []
        if bargedUtterance {
            // A confirmed barge that transcribed to anything is a real
            // interruption — the pause becomes a stop for good. No word
            // gate: the barge decision is purely acoustic; the empty take
            // already resumed above.
            bargeMode = .none
            effects.append(.stopSpeaking)
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

    // MARK: - Record plumbing

    /// A barge-family record with the detector's inputs at this instant
    /// stamped alongside the event's own fields, so field tuning is never
    /// blind again (the 2026-07-17 storms shipped no numbers at all).
    private func energyRecord(_ event: CompanionTraceEvent, extra: [String: String]) -> Effect {
        let energy: [String: String] = [
            "level": String(format: "%.3f", lastLevel),
            "threshold": String(
                format: "%.3f",
                echoFloor.threshold(
                    atLeast: tunables.bargeInLevel, marginScale: escalationMarginScale)),
            "floor": String(format: "%.3f", echoFloor.floor),
            "playbackLevel": String(format: "%.3f", lastPlaybackLevel),
            "falseBargeCount": String(falseBargeCount),
        ]
        return .record(
            event: event, snapshot: energy.merging(extra) { _, new in new })
    }

    /// How far into the spoken reply the event landed, for barge records.
    private func speakingOffsetSeconds(now: TimeInterval) -> String {
        String(format: "%.1f", speakingSince.map { now - $0 } ?? 0)
    }
}
