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
//  (mic open underneath; VPIO's loopback echo cancellation keeps his voice
//  out of the input — the ADR-0041 dual-path reference routing is deferred
//  until the voice hold can be implemented without taps on a running engine).
//  Sustained speech energy is a barge-in that *pauses* him: a take with
//  substance commits and stops the reply for good, a Session Directive stops
//  it without reaching the agent, and a false barge resumes him where he
//  paused. Exit: overlay ✕, the chat toggle, or mutual silence.
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

    // MARK: - Dependencies

    private let capture: VoiceCaptureSession
    private let meterLevel: @MainActor () -> Float
    private let meterSpectrum: @MainActor () -> [Float]
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
    /// A barged (paused) reply awaits resolution: resume, directive, or turn.
    private var bargedUtterance = false
    private var bargeVerifyStartedAt: Date?
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
        proofreadPass: ProofreadPass?
    ) {
        self.capture = capture
        self.meterLevel = meterLevel
        self.meterSpectrum = meterSpectrum
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
        recorder.record(
            "voice.session-entered",
            conversationID: currentConversationID(),
            snapshot: ["via": via])
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
        deafUntil = nil
        closeCapture()
        ticker?.cancel()
        ticker = nil
        phase = .idle
        recorder.record(
            "voice.session-exited",
            conversationID: currentConversationID(),
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
        deafUntil = nil
        settledTicks = 0
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        // The mic opens *under* the utterance: VPIO's AEC keeps his voice out
        // of the input, so speech energy here is the owner (#310 §4).
        openCapture()
        speak(text) { [weak self] in
            self?.speechDoneCallbackSeen = true
            self?.utteranceFinished()
        }
        recorder.record(
            "voice.reply-spoken",
            conversationID: currentConversationID(),
            snapshot: ["chars": String(text.count)])
        return true
    }

    // MARK: - Barge-in

    func bargeIn(source: String) {
        guard phase == .speaking else { return }
        let offset = speakingSince.map { Date().timeIntervalSince($0) } ?? 0
        // Pause, don't stop (pause-on-barge): a false barge resumes the
        // reply where it left off; only a committed turn or a Session
        // Directive makes the interruption permanent.
        pauseSpeaking()
        bargedUtterance = true
        bargeVerifyStartedAt = Date()
        settledTicks = 0
        recorder.record(
            "reaction.barge-in",
            source: .appObserved,
            conversationID: currentConversationID(),
            snapshot: [
                "offsetSeconds": String(format: "%.1f", offset), "detector": source,
            ])
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

    /// The barge produced nothing that counts as a turn — resume the paused
    /// reply and rearm barge-in detection.
    private func resumeAfterFalseBarge(reason: String) {
        guard bargedUtterance else { return }
        bargedUtterance = false
        bargeVerifyStartedAt = nil
        if captureOpen { closeCapture() }
        recorder.record(
            "voice.barge-false-resume",
            conversationID: currentConversationID(),
            snapshot: ["reason": reason])
        if speechDoneCallbackSeen || !isActive {
            // The utterance drained while paused (or the session died) —
            // nothing to resume.
            if isActive { beginListening() }
            return
        }
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        settledTicks = 0
        openCapture()
        phase = .speaking
        overlay.feed.setState(.speaking)
        resumeSpeaking()
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
        let now = Date()
        let level = meterLevel()
        overlay.feed.setMeter(level: level, spectrum: meterSpectrum())
        let deaf = deafUntil.map { now < $0 } ?? false
        let event =
            captureOpen && !deaf
            ? endpointer.ingest(level: level, at: now.timeIntervalSinceReferenceDate)
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
            if event == .endOfSpeech {
                finishOwnerTurn()
            } else if bargedUtterance, !endpointer.isInSpeech,
                let since = bargeVerifyStartedAt,
                now.timeIntervalSince(since) > Self.bargeVerifyWindow
            {
                resumeAfterFalseBarge(reason: "no-speech")
            }

        case .speaking:
            if event == .speechStarted {
                bargeIn(source: "energy")
            } else if !speechDoneCallbackSeen {
                // The engine stopped without the success callback (error, or
                // an external stop). Exit only on a *sustained* settled
                // reading — a single transient sample reopened the mic under
                // live TTS (the 2026-07-16 self-echo trace, ADR-0041).
                if isSpeechEngineSettled { settledTicks += 1 } else { settledTicks = 0 }
                if settledTicks >= Self.watchdogSettledTicks {
                    settledTicks = 0
                    recorder.record(
                        "voice.watchdog-exit",
                        conversationID: currentConversationID(),
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
                stopSpeaking()
                recorder.record(
                    "voice.session-directive",
                    conversationID: currentConversationID(),
                    snapshot: ["word": word])
                beginListening()
                return
            case .falseBarge:
                resumeAfterFalseBarge(reason: "below-gate")
                return
            case .turn:
                // A real interruption — the pause becomes a stop for good.
                bargedUtterance = false
                bargeVerifyStartedAt = nil
                stopSpeaking()
            }
        }
        exchanges += 1
        recorder.record(
            "voice.owner-turn",
            conversationID: currentConversationID(),
            snapshot: ["chars": String(trimmed.count)])
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
