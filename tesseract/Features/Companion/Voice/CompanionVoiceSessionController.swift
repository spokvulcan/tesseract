//
//  CompanionVoiceSessionController.swift
//  tesseract
//
//  The voice session (#310): voice as a mode of the one conversation, never a
//  separate surface. A session binds the #328 overlay concept, the speech
//  engine, and an auto-listen loop to the interactive chat — spoken and typed
//  turns are the same persisted message stream.
//
//  The loop: listen (mic open, endpointer armed) → owner speaks → trailing
//  silence auto-sends the transcription → reply arrives → Jarvis speaks it
//  (mic open underneath; VPIO's echo cancellation keeps his voice out of the
//  input, and the Echo Floor tracks whatever residual leaks past it so the
//  barge threshold always sits above the reply's own read-back, ADR-0041).
//  Sustained speech energy is a **Soft Barge**: the reply ducks instantly
//  and keeps murmuring while a short window confirms real voicing — only a
//  confirmed interruption *pauses* him (a false fire costs a dip, never a
//  pause). From the pause, a take with substance commits and stops the reply
//  for good, a Session Directive stops it without reaching the agent, and a
//  false barge resumes him where he paused. The overlay's click barge pauses
//  immediately — a click is deliberate. Exit: overlay ✕, the chat toggle, or
//  mutual silence.
//
//  Every tunable the taste ledger named is a Setting: trailing silence,
//  session timeout, barge-in sensitivity, auto-send (the escape hatch stages
//  to the composer instead).
//

import Foundation
import Observation

@Observable @MainActor
final class CompanionVoiceSessionController {

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

    // MARK: - Barge resolution (Substance Gate + Session Directives)

    nonisolated enum BargeResolution: Equatable {
        /// An allowlisted control word — acts on the session, never sent.
        case directive(String)
        /// A real turn: stop the reply for good and commit.
        case turn
        /// Not enough substance to be a turn — resume the paused reply.
        case falseBarge
    }

    /// English playback-control words. Content-ambiguous words ("no", "yes",
    /// "okay") are deliberately absent — those are answers, not directives.
    nonisolated static let sessionDirectives: Set<String> = ["stop", "wait", "pause", "quiet"]

    /// The **Substance Gate**: what a take captured under a barged reply must
    /// show to count as a real turn rather than echo residual or a thump.
    nonisolated static let substanceMinVoicedSeconds: TimeInterval = 0.6
    nonisolated static let substanceMinWords = 2

    /// Pure resolution of a barge take — pinned by tests.
    nonisolated static func resolveBargeTake(
        text: String, voicedSeconds: TimeInterval
    ) -> BargeResolution {
        let words = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
        if words.count == 1, let word = words.first, sessionDirectives.contains(word) {
            return .directive(word)
        }
        if voicedSeconds >= substanceMinVoicedSeconds, words.count >= substanceMinWords {
            return .turn
        }
        return .falseBarge
    }

    // MARK: - Soft Barge (two-stage barge-in, ADR-0041)

    /// How an energy onset during playback resolves: an instant duck opens a
    /// confirm window; only sustained voicing turns the duck into a pause.
    nonisolated enum SoftBargeVerdict: Equatable {
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
    nonisolated static func resolveSoftBarge(
        voicedSeconds: TimeInterval, elapsed: TimeInterval,
        confirmWindow: TimeInterval, confirmVoiced: TimeInterval
    ) -> SoftBargeVerdict {
        if voicedSeconds >= confirmVoiced { return .confirm }
        return elapsed >= confirmWindow ? .fadeBack : .keepWaiting
    }

    // MARK: - Dependencies

    private let capture: VoiceCaptureSession
    private let meterLevel: @MainActor () -> Float
    private let meterSpectrum: @MainActor () -> [Float]
    /// The reply's loudness at the playback head (the coordinator's active
    /// sink) — the Echo Floor's far-end signal.
    private let playbackLevel: @MainActor () -> Float
    /// Ramps the reply's volume (target, duration) — the Soft Barge duck.
    private let fadeSpeech: @MainActor (Float, TimeInterval) -> Void
    private let sendMessage: @MainActor (String) -> Void
    private let stageToComposer: @MainActor (String) -> Void
    private let speak: @MainActor (String, @escaping @MainActor @Sendable () -> Void) -> Void
    private let stopSpeaking: @MainActor () -> Void
    private let pauseSpeaking: @MainActor () -> Void
    private let resumeSpeaking: @MainActor () -> Void
    private let speechState: @MainActor () -> SpeechState
    private let currentConversationID: @MainActor () -> UUID?
    private let overlay: CompanionVoicePrototype
    private let recorder: CompanionFlightRecorder
    private let settings: SettingsManager
    private let proofreadPass: ProofreadPass?

    // MARK: - Session state

    private var ticker: Task<Void, Never>?
    private var endpointer = VoiceEndpointer(config: .listening())
    private var captureOpen = false
    private var listeningSince: Date?
    private var speakingSince: Date?
    private var speechDoneCallbackSeen = false
    private var exchanges = 0
    /// A barged (ducked or paused) reply awaits resolution: resume,
    /// directive, or turn.
    private var bargedUtterance = false
    private var bargeVerifyStartedAt: Date?
    /// Where the current barge stands: `.soft` = reply ducked, confirm
    /// window open; `.hard` = reply paused (confirmed voicing or a click).
    private var bargeMode: BargeMode = .none
    private enum BargeMode: Equatable {
        case none
        case soft(startedAt: Date)
        case hard
    }
    /// The Echo Floor (ADR-0041): tracked self-echo residual while the reply
    /// plays; the energy barge threshold always sits `margin` above it.
    private var echoFloor = EchoResidualFloor()
    /// False energy barges on the *current* utterance — the escalation
    /// ladder's input (#354): ≥2 widens the floor margin, ≥4 mutes the
    /// energy detector for the rest of the utterance (click and directives
    /// keep working).
    private var falseBargeCount = 0
    private var energyBargeMuted = false
    /// Stable ID stamped on every voice.* record of one session (#354).
    private var sessionID = UUID()
    /// Tick counter for the 1 Hz energy-sample cadence while speaking.
    private var tickCount = 0
    /// The endpointer's voiced-time reading at turn close — the Substance
    /// Gate's energy input, snapshotted before the endpointer resets.
    private var lastTakeVoicedSeconds: TimeInterval = 0
    /// Post-utterance grace: endpointer events are ignored until this
    /// deadline so the reply's room tail can't seed a turn.
    private var deafUntil: Date?
    /// Consecutive ticks the speech engine read as settled — the watchdog
    /// only exits `.speaking` on a sustained reading, never a single sample.
    private var settledTicks = 0

    /// Ticker cadence — 20 Hz gives the endpointer 50 ms resolution, an order
    /// under the shortest debounce.
    private static let tickInterval: Duration = .milliseconds(50)
    /// Ignore endpointer events for this long after an utterance ends —
    /// output-device latency plus room tail.
    private static let postUtteranceGrace: TimeInterval = 0.3
    /// A barge with no speech onset within this window is false — resume.
    private static let bargeVerifyWindow: TimeInterval = 2.0
    /// Settled ticks (× 50 ms) required before the watchdog exits `.speaking`.
    private static let watchdogSettledTicks = 6

    // Soft Barge constants (ADR-0041; calibrated by the voice-hold lab).
    /// The duck target — quiet enough that residual falls well under the
    /// listening threshold, loud enough that a false fire stays a murmur.
    private static let softDuckLevel: Float = 0.25
    private static let softDuckRampDown: TimeInterval = 0.1
    private static let softDuckRampUp: TimeInterval = 0.2
    /// The confirm window an energy onset opens, and the voiced time inside
    /// it that commits the hard pause.
    private static let softBargeConfirmWindow: TimeInterval = 0.8
    private static let softBargeConfirmVoiced: TimeInterval = 0.3
    /// Endpointer deafness after a false-barge resume/fade-back — the fade-up
    /// and AEC re-settling transient must not re-trigger the detector
    /// (observed re-barge onsets ~0.85 s post-resume, flight 2026-07-17).
    private static let postResumeGrace: TimeInterval = 1.0
    // The escalation ladder (#354 item 2), per utterance.
    private static let escalateMarginAfter = 2
    private static let escalatedMarginScale: Float = 1.5
    private static let energyBargeMuteAfter = 4
    /// Every 20th tick (1 Hz) records an energy sample while speaking.
    private static let energySampleEveryTicks = 20

    init(
        capture: VoiceCaptureSession,
        meterLevel: @escaping @MainActor () -> Float,
        meterSpectrum: @escaping @MainActor () -> [Float],
        sendMessage: @escaping @MainActor (String) -> Void,
        stageToComposer: @escaping @MainActor (String) -> Void,
        speak: @escaping @MainActor (String, @escaping @MainActor @Sendable () -> Void) -> Void,
        stopSpeaking: @escaping @MainActor () -> Void,
        pauseSpeaking: @escaping @MainActor () -> Void,
        resumeSpeaking: @escaping @MainActor () -> Void,
        speechState: @escaping @MainActor () -> SpeechState,
        currentConversationID: @escaping @MainActor () -> UUID?,
        overlay: CompanionVoicePrototype,
        recorder: CompanionFlightRecorder,
        settings: SettingsManager,
        proofreadPass: ProofreadPass?,
        playbackLevel: @escaping @MainActor () -> Float = { 0 },
        fadeSpeech: @escaping @MainActor (Float, TimeInterval) -> Void = { _, _ in }
    ) {
        self.capture = capture
        self.meterLevel = meterLevel
        self.meterSpectrum = meterSpectrum
        self.playbackLevel = playbackLevel
        self.fadeSpeech = fadeSpeech
        self.sendMessage = sendMessage
        self.stageToComposer = stageToComposer
        self.speak = speak
        self.stopSpeaking = stopSpeaking
        self.pauseSpeaking = pauseSpeaking
        self.resumeSpeaking = resumeSpeaking
        self.speechState = speechState
        self.currentConversationID = currentConversationID
        self.overlay = overlay
        self.recorder = recorder
        self.settings = settings
        self.proofreadPass = proofreadPass
    }

    // MARK: - Entry / exit

    func toggle() {
        if isActive { exit(reason: "toggled-off") } else { enter(via: "chat-toggle") }
    }

    func enter(via: String) {
        guard !isActive else { return }
        exchanges = 0
        sessionID = UUID()
        tickCount = 0
        recordVoice("voice.session-entered", snapshot: ["via": via])
        overlay.beginLiveSession(
            actions: CompanionVoiceActions(
                engage: {},
                bargeIn: { [weak self] in self?.bargeIn(source: "click") },
                dismiss: { [weak self] in self?.exit(reason: "dismissed") },
                openChat: { [weak self] in self?.overlay.openChatFromLiveSession() }
            ))
        startTicker()
        beginListening()
    }

    func exit(reason: String) {
        guard isActive else { return }
        if phase == .speaking || bargedUtterance { stopSpeaking() }
        bargedUtterance = false
        bargeVerifyStartedAt = nil
        bargeMode = .none
        deafUntil = nil
        closeCapture()
        ticker?.cancel()
        ticker = nil
        phase = .idle
        recordVoice(
            "voice.session-exited",
            snapshot: ["reason": reason, "exchanges": String(exchanges)])
        overlay.endLiveSession()
    }

    // MARK: - The reply hook (ChatSession's agentEnd calls this)

    /// Returns true when the session consumed the reply (suppresses the
    /// chat's own autoSpeak).
    func replyCompleted(_ text: String?) -> Bool {
        guard isActive else { return false }
        guard phase == .awaitingReply || phase == .transcribing else { return false }
        guard let text, !text.isEmpty else {
            // A silent reply (pure tool turn) — reopen the mic and move on.
            beginListening()
            return true
        }
        overlay.feed.settle(role: .companion, text: text)
        overlay.feed.beginSpokenLine(text)
        overlay.feed.revealAll()
        overlay.feed.setState(.speaking)
        phase = .speaking
        speakingSince = Date()
        speechDoneCallbackSeen = false
        bargedUtterance = false
        bargeVerifyStartedAt = nil
        bargeMode = .none
        deafUntil = nil
        settledTicks = 0
        echoFloor.reset()
        falseBargeCount = 0
        energyBargeMuted = false
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        // The mic opens *under* the utterance: VPIO's AEC keeps his voice out
        // of the input, and the Echo Floor rides whatever residual remains,
        // so speech energy over both is the owner (#310 §4, ADR-0041).
        openCapture()
        speak(text) { [weak self] in
            self?.speechDoneCallbackSeen = true
            self?.utteranceFinished()
        }
        recordVoice("voice.reply-spoken", snapshot: ["chars": String(text.count)])
        return true
    }

    // MARK: - Barge-in

    /// The hard (immediate-pause) barge — the overlay's click, which is
    /// deliberate and has zero false positives in the field data. A click
    /// while a Soft Barge is verifying commits it instead.
    func bargeIn(source: String) {
        if case .soft = bargeMode {
            hardenSoftBarge(detector: source)
            return
        }
        guard phase == .speaking else { return }
        let offset = speakingSince.map { Date().timeIntervalSince($0) } ?? 0
        // Pause, don't stop (pause-on-barge): a false barge resumes the
        // reply where it left off; only a committed turn or a Session
        // Directive makes the interruption permanent.
        pauseSpeaking()
        bargedUtterance = true
        bargeMode = .hard
        bargeVerifyStartedAt = Date()
        settledTicks = 0
        recordVoice(
            "reaction.barge-in",
            snapshot: energyFields(level: meterLevel()).merging([
                "offsetSeconds": String(format: "%.1f", offset), "detector": source,
            ]) { _, new in new })
        // Fresh take from the interruption on — the playback-period audio
        // (echo-cancelled silence) is dropped, not transcribed.
        reopenCapture()
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: trailingSilence))
        if source != "click" {
            // Energy barge: he is already mid-word — seed the candidate so
            // the start debounce measures from the interruption itself.
            _ = endpointer.ingest(level: 1.0, at: Date().timeIntervalSinceReferenceDate)
        }
        phase = .capturing
        overlay.feed.setState(.listening)
    }

    /// Stage one of the Soft Barge: an energy onset ducks the reply and
    /// opens the confirm window — capture starts now so no owner words are
    /// lost, but the reply is not paused until voicing confirms.
    private func softBargeIn() {
        guard phase == .speaking else { return }
        let offset = speakingSince.map { Date().timeIntervalSince($0) } ?? 0
        bargeMode = .soft(startedAt: Date())
        bargedUtterance = true
        settledTicks = 0
        recordVoice(
            "voice.barge-soft-onset",
            snapshot: energyFields(level: meterLevel()).merging([
                "offsetSeconds": String(format: "%.1f", offset)
            ]) { _, new in new })
        fadeSpeech(Self.softDuckLevel, Self.softDuckRampDown)
        // Fresh take from the onset — mirrors the hard barge; the ducked
        // reply (−12 dB) reads well under the listening threshold, so
        // voicing accumulated from here on is the owner.
        reopenCapture()
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: trailingSilence))
        _ = endpointer.ingest(level: 1.0, at: Date().timeIntervalSinceReferenceDate)
        phase = .capturing
        overlay.feed.setState(.listening)
    }

    /// Stage two, confirmed: voicing sustained through the confirm window —
    /// the duck becomes the real pause and the take proceeds as a hard barge.
    private func hardenSoftBarge(detector: String) {
        guard case .soft = bargeMode else { return }
        bargeMode = .hard
        pauseSpeaking()
        bargeVerifyStartedAt = Date()
        let offset = speakingSince.map { Date().timeIntervalSince($0) } ?? 0
        recordVoice(
            "reaction.barge-in",
            snapshot: energyFields(level: meterLevel()).merging([
                "offsetSeconds": String(format: "%.1f", offset), "detector": detector,
            ]) { _, new in new })
    }

    /// Stage two, unconfirmed: the window closed without voicing — restore
    /// the reply's volume and re-arm. The false fire cost a ~1 s murmur.
    private func softFadeBack() {
        guard case .soft = bargeMode else { return }
        bargeMode = .none
        bargedUtterance = false
        registerFalseBarge()
        if captureOpen { closeCapture() }
        recordVoice(
            "voice.barge-false-resume",
            snapshot: energyFields(level: meterLevel()).merging([
                "reason": "soft-fadeback"
            ]) { _, new in new })
        if speechDoneCallbackSeen || !isActive {
            // The utterance drained while ducked (or the session died).
            fadeSpeech(1.0, 0)
            if isActive { beginListening() }
            return
        }
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        settledTicks = 0
        openCapture()
        // Deaf through the fade-up: the volume ramp and AEC re-settling
        // must not re-trigger the detector (the 2026-07-17 flap cycle).
        deafUntil = Date().addingTimeInterval(Self.postResumeGrace)
        phase = .speaking
        overlay.feed.setState(.speaking)
        fadeSpeech(1.0, Self.softDuckRampUp)
    }

    /// The barge produced nothing that counts as a turn — resume the paused
    /// reply and rearm barge-in detection.
    private func resumeAfterFalseBarge(reason: String) {
        guard bargedUtterance else { return }
        bargedUtterance = false
        bargeMode = .none
        bargeVerifyStartedAt = nil
        registerFalseBarge()
        if captureOpen { closeCapture() }
        recordVoice(
            "voice.barge-false-resume",
            snapshot: energyFields(level: meterLevel()).merging([
                "reason": reason
            ]) { _, new in new })
        if speechDoneCallbackSeen || !isActive {
            // The utterance drained while paused (or the session died) —
            // nothing to resume.
            fadeSpeech(1.0, 0)
            if isActive { beginListening() }
            return
        }
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        settledTicks = 0
        openCapture()
        // Deaf through the resume transient — see `softFadeBack`.
        deafUntil = Date().addingTimeInterval(Self.postResumeGrace)
        phase = .speaking
        overlay.feed.setState(.speaking)
        resumeSpeaking()
        fadeSpeech(1.0, Self.softDuckRampUp)
    }

    /// One more false energy barge on this utterance — climb the escalation
    /// ladder (#354): a wider floor margin first, then a muted energy
    /// detector for the utterance's remainder.
    private func registerFalseBarge() {
        falseBargeCount += 1
        if falseBargeCount >= Self.energyBargeMuteAfter {
            energyBargeMuted = true
        }
    }

    private var escalationMarginScale: Float {
        falseBargeCount >= Self.escalateMarginAfter ? Self.escalatedMarginScale : 1.0
    }

    /// The take produced nothing usable — resume a paused reply if one is
    /// waiting, otherwise reopen the mic and listen.
    private func abandonTake(reason: String) {
        if bargedUtterance {
            resumeAfterFalseBarge(reason: reason)
        } else {
            beginListening()
        }
    }

    // MARK: - The ticker

    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: CompanionVoiceSessionController.tickInterval)
                self?.tick()
            }
        }
    }

    private func tick() {
        guard isActive else { return }
        tickCount += 1
        let now = Date()
        let level = meterLevel()
        overlay.feed.setMeter(level: level, spectrum: meterSpectrum())
        let deaf = deafUntil.map { now < $0 } ?? false
        // The Echo Floor tracks through deafness — residual keeps arriving
        // whether or not the endpointer is allowed to react to it.
        if phase == .speaking {
            echoFloor.ingest(
                micLevel: level, playbackLevel: playbackLevel(),
                at: now.timeIntervalSinceReferenceDate)
        }
        let speechFloor: Float? =
            phase == .speaking
            ? echoFloor.threshold(atLeast: bargeInLevel, marginScale: escalationMarginScale)
            : nil
        let event =
            captureOpen && !deaf
            ? endpointer.ingest(
                level: level, at: now.timeIntervalSinceReferenceDate,
                speechFloor: speechFloor)
            : nil

        switch phase {
        case .listening:
            if !captureOpen { openCapture() }
            if event == .speechStarted {
                phase = .capturing
            } else if let since = listeningSince,
                now.timeIntervalSince(since) > sessionTimeout
            {
                exit(reason: "mutual-silence")
            }

        case .capturing:
            if case .soft(let startedAt) = bargeMode {
                if event == .endOfSpeech {
                    // Defensive: trailing silence (1.8 s) cannot beat the
                    // confirm window (0.8 s) with current constants, but a
                    // closed take must resolve through the hard path.
                    hardenSoftBarge(detector: "energy-soft")
                    finishOwnerTurn()
                } else {
                    switch Self.resolveSoftBarge(
                        voicedSeconds: endpointer.voicedSeconds,
                        elapsed: now.timeIntervalSince(startedAt),
                        confirmWindow: Self.softBargeConfirmWindow,
                        confirmVoiced: Self.softBargeConfirmVoiced)
                    {
                    case .keepWaiting:
                        break
                    case .confirm:
                        hardenSoftBarge(detector: "energy-soft")
                    case .fadeBack:
                        softFadeBack()
                    }
                }
            } else if event == .endOfSpeech {
                finishOwnerTurn()
            } else if bargedUtterance, !endpointer.isInSpeech,
                let since = bargeVerifyStartedAt,
                now.timeIntervalSince(since) > Self.bargeVerifyWindow
            {
                resumeAfterFalseBarge(reason: "no-speech")
            }

        case .speaking:
            if tickCount % Self.energySampleEveryTicks == 0 {
                recordVoice("voice.energy-sample", snapshot: energyFields(level: level))
            }
            if event == .speechStarted {
                if energyBargeMuted {
                    // The escalation ladder's top: the detector cried wolf
                    // ≥4 times this utterance — log, never react.
                    recordVoice(
                        "voice.barge-suppressed",
                        snapshot: energyFields(level: level))
                } else {
                    softBargeIn()
                }
            } else if !speechDoneCallbackSeen {
                // The engine stopped without the success callback (error, or
                // an external stop). Exit only on a *sustained* settled
                // reading — a single transient sample reopened the mic under
                // live TTS (the 2026-07-16 self-echo trace, ADR-0041).
                if isSpeechEngineSettled { settledTicks += 1 } else { settledTicks = 0 }
                if settledTicks >= Self.watchdogSettledTicks {
                    settledTicks = 0
                    recordVoice(
                        "voice.watchdog-exit",
                        snapshot: ["speechState": String(describing: speechState())])
                    utteranceFinished()
                }
            }

        case .idle, .transcribing, .awaitingReply:
            break
        }
    }

    /// After ~a second of grace, a settled engine means the utterance is over.
    private var isSpeechEngineSettled: Bool {
        guard let since = speakingSince, Date().timeIntervalSince(since) > 1.0 else {
            return false
        }
        return !speechState().isActive
    }

    private func utteranceFinished() {
        guard phase == .speaking else { return }
        // Force-stop: however the utterance ended, TTS must be provably
        // silent before the mic reopens in listening config. On the normal
        // path this is a no-op sweep; on a watchdog exit it is the fix.
        stopSpeaking()
        bargedUtterance = false
        bargeMode = .none
        beginListening(gracePeriod: Self.postUtteranceGrace)
    }

    // MARK: - Turn plumbing

    private func beginListening(gracePeriod: TimeInterval = 0) {
        reopenCapture()
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: trailingSilence))
        deafUntil = gracePeriod > 0 ? Date().addingTimeInterval(gracePeriod) : nil
        phase = .listening
        listeningSince = Date()
        overlay.feed.setState(.listening)
    }

    private func finishOwnerTurn() {
        // Snapshot before the endpointer resets — the Substance Gate's input.
        lastTakeVoicedSeconds = endpointer.voicedSeconds
        phase = .transcribing
        overlay.feed.setState(.thinking)
        guard captureOpen else {
            abandonTake(reason: "no-capture")
            return
        }
        captureOpen = false
        switch capture.stop() {
        case .noAudio, .tooShort:
            abandonTake(reason: "no-audio")
        case .audio(let audio, _):
            let language = settings.language
            var proofread: (@MainActor (String) async -> ProofreadVerdict?)?
            if let pass = proofreadPass {
                proofread = { text in await pass.proofread(text) }
            }
            Task { [weak self] in
                guard let self else { return }
                let outcome = await self.capture.transcribeAndCommit(
                    audio, language: language, proofread: proofread
                ) { [weak self] text, _ in
                    self?.ownerTurnTranscribed(text)
                }
                switch outcome {
                case .committed, .superseded:
                    break
                case .rejected(let raw, _):
                    // A rejected proofread is still his words — voice flows on.
                    self.ownerTurnTranscribed(raw)
                case .empty:
                    self.abandonTake(reason: "empty")
                case .failed, .cancelled:
                    guard self.isActive else { return }
                    self.abandonTake(reason: "failed")
                }
            }
        }
    }

    private func ownerTurnTranscribed(_ text: String) {
        guard isActive else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            abandonTake(reason: "empty")
            return
        }
        if bargedUtterance {
            switch Self.resolveBargeTake(
                text: trimmed, voicedSeconds: lastTakeVoicedSeconds)
            {
            case .directive(let word):
                bargedUtterance = false
                bargeVerifyStartedAt = nil
                bargeMode = .none
                stopSpeaking()
                recordVoice("voice.session-directive", snapshot: ["word": word])
                beginListening()
                return
            case .falseBarge:
                resumeAfterFalseBarge(reason: "below-gate")
                return
            case .turn:
                // A real interruption — the pause becomes a stop for good.
                bargedUtterance = false
                bargeVerifyStartedAt = nil
                bargeMode = .none
                stopSpeaking()
            }
        }
        exchanges += 1
        recordVoice("voice.owner-turn", snapshot: ["chars": String(trimmed.count)])
        guard settings.companionVoiceAutoSend else {
            // The escape hatch (#310 taste ledger): stage, never send.
            stageToComposer(trimmed)
            exit(reason: "staged-to-composer")
            return
        }
        overlay.feed.settle(role: .owner, text: trimmed)
        overlay.feed.setState(.thinking)
        phase = .awaitingReply
        sendMessage(trimmed)
    }

    // MARK: - Flight-recorder plumbing

    /// Every voice.* record carries the session's stable ID (#354) — one
    /// session's events group without timestamp heuristics.
    private func recordVoice(_ event: String, snapshot: [String: String]) {
        var snapshot = snapshot
        snapshot["sessionID"] = sessionID.uuidString
        recorder.record(
            event, conversationID: currentConversationID(), snapshot: snapshot)
    }

    /// The detector's inputs at this instant — stamped on every barge event
    /// and the 1 Hz energy sample, so field tuning is never blind again
    /// (the 2026-07-17 storms shipped no numbers at all).
    private func energyFields(level: Float) -> [String: String] {
        [
            "level": String(format: "%.3f", level),
            "threshold": String(
                format: "%.3f",
                echoFloor.threshold(
                    atLeast: bargeInLevel, marginScale: escalationMarginScale)),
            "floor": String(format: "%.3f", echoFloor.floor),
            "playbackLevel": String(format: "%.3f", playbackLevel()),
            "falseBargeCount": String(falseBargeCount),
        ]
    }

    // MARK: - Capture plumbing

    /// A failed start (mic busy, engine refusing) retries no sooner than
    /// this. Without the backoff the 20 Hz ticker retried every 50 ms, and a
    /// failing `startCapture` can cost hundreds of ms of CoreAudio work per
    /// attempt on the main thread — the app-wide freeze in the 2026-07-17
    /// crash report.
    private static let captureRetryBackoff: TimeInterval = 1.0
    private var lastCaptureAttemptFailedAt: Date?

    private func openCapture() {
        guard !captureOpen else { return }
        if let failedAt = lastCaptureAttemptFailedAt,
            Date().timeIntervalSince(failedAt) < Self.captureRetryBackoff
        {
            return
        }
        if case .started = capture.start() {
            captureOpen = true
            lastCaptureAttemptFailedAt = nil
        } else {
            // micBusy (dictation mid-take) or a start failure resolves on a
            // later tick — the session keeps listening state without a live
            // mic, retrying at backoff cadence, never at tick cadence.
            lastCaptureAttemptFailedAt = Date()
        }
    }

    private func reopenCapture() {
        if captureOpen { capture.cancel() }
        captureOpen = false
        openCapture()
    }

    private func closeCapture() {
        if captureOpen { capture.cancel() }
        captureOpen = false
    }

    // MARK: - Tunables (the taste ledger, as Settings)

    private var trailingSilence: TimeInterval { settings.companionVoiceTrailingSilence }
    private var sessionTimeout: TimeInterval { settings.companionVoiceSessionTimeout }
    private var bargeInLevel: Float { Float(settings.companionVoiceBargeInLevel) }
    private var speechLevel: Float { VoiceEndpointer.Config.listening().speechLevel }
}
