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
//  (mic stays open under VPIO echo-cancellation; sustained speech energy is a
//  barge-in that stops him mid-word, app-observed in the flight recorder) →
//  auto-listen reopens. Exit: overlay ✕, the chat toggle, or mutual silence.
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

    // MARK: - Dependencies

    private let capture: VoiceCaptureSession
    private let meterLevel: @MainActor () -> Float
    private let meterSpectrum: @MainActor () -> [Float]
    private let sendMessage: @MainActor (String) -> Void
    private let stageToComposer: @MainActor (String) -> Void
    private let speak: @MainActor (String, @escaping @MainActor @Sendable () -> Void) -> Void
    private let stopSpeaking: @MainActor () -> Void
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

    /// Ticker cadence — 20 Hz gives the endpointer 50 ms resolution, an order
    /// under the shortest debounce.
    private static let tickInterval: Duration = .milliseconds(50)

    init(
        capture: VoiceCaptureSession,
        meterLevel: @escaping @MainActor () -> Float,
        meterSpectrum: @escaping @MainActor () -> [Float],
        sendMessage: @escaping @MainActor (String) -> Void,
        stageToComposer: @escaping @MainActor (String) -> Void,
        speak: @escaping @MainActor (String, @escaping @MainActor @Sendable () -> Void) -> Void,
        stopSpeaking: @escaping @MainActor () -> Void,
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
        if phase == .speaking { stopSpeaking() }
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
        endpointer.reset(config: .bargeIn(speechLevel: bargeInLevel))
        // The mic opens *under* the utterance: VPIO's AEC keeps his voice out
        // of the input, so speech energy here is the owner (#310 §4).
        openCapture()
        speak(text) { [weak self] in
            self?.speechDoneCallbackSeen = true
            self?.utteranceFinished(interrupted: false)
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
        stopSpeaking()
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
        // He is already mid-sentence: the endpointer starts in-speech.
        _ = endpointer.ingest(level: 1.0, at: Date().timeIntervalSinceReferenceDate)
        phase = .capturing
        overlay.feed.setState(.listening)
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
        let event =
            captureOpen
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
            if event == .endOfSpeech { finishOwnerTurn() }

        case .speaking:
            if event == .speechStarted {
                bargeIn(source: "energy")
            } else if !speechDoneCallbackSeen, isSpeechEngineSettled {
                // The engine stopped without the success callback (error, or
                // an external stop) — treat it as the utterance ending.
                utteranceFinished(interrupted: true)
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
        switch speechState() {
        case .idle, .error: return true
        default: return false
        }
    }

    private func utteranceFinished(interrupted: Bool) {
        guard phase == .speaking else { return }
        _ = interrupted
        beginListening()
    }

    // MARK: - Turn plumbing

    private func beginListening() {
        reopenCapture()
        endpointer.reset(
            config: .listening(
                speechLevel: speechLevel, trailingSilence: trailingSilence))
        phase = .listening
        listeningSince = Date()
        overlay.feed.setState(.listening)
    }

    private func finishOwnerTurn() {
        phase = .transcribing
        overlay.feed.setState(.thinking)
        guard captureOpen else {
            beginListening()
            return
        }
        captureOpen = false
        switch capture.stop() {
        case .noAudio, .tooShort:
            beginListening()
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
                    self.beginListening()
                case .failed, .cancelled:
                    if self.isActive { self.beginListening() }
                }
            }
        }
    }

    private func ownerTurnTranscribed(_ text: String) {
        guard isActive else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            beginListening()
            return
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

    private func openCapture() {
        guard !captureOpen else { return }
        if case .started = capture.start() { captureOpen = true }
        // micBusy (dictation mid-take) resolves on a later tick — the session
        // just keeps listening state without a live mic until it frees.
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
