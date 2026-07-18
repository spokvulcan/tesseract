//
//  CompanionVoiceSessionController.swift
//  tesseract
//
//  The voice session (#310): voice as a mode of the one conversation, never a
//  separate surface. A session binds the #328 overlay concept, the speech
//  engine, and an auto-listen loop to the interactive chat — spoken and typed
//  turns are the same persisted message stream.
//
//  This controller is the **performer** half of the ADR-0042 split: every
//  judgment of the loop — phases, the two-stage Soft Barge, the Echo Floor
//  gate, escalation, deaf windows, the watchdog, capture backoff — lives in
//  the pure **Voice Session Machine**; this class pumps events in (the 20 Hz
//  ticker, the overlay's click, ChatSession's reply hook, transcription
//  outcomes) and executes the effects that come back, in order. Capture
//  start results feed back into the machine as events, so even the retry
//  backoff is machine judgment.
//
//  Every tunable the taste ledger named is a Setting: trailing silence,
//  session timeout, barge-in sensitivity, auto-send (the escape hatch stages
//  to the composer instead).
//

import Foundation
import Observation

@Observable @MainActor
final class CompanionVoiceSessionController {

    typealias Phase = VoiceSessionMachine.Phase

    /// Observable mirror of the machine's phase, refreshed after every
    /// dispatch — the only render state views read.
    private(set) var phase: Phase = .idle
    var isActive: Bool { phase != .idle }

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
    /// The ADR-0041 voice hold: the capture engine is held (and hosts the
    /// reply's playback) for the session's lifetime. Injected so tests keep
    /// their fakes; the composition root binds `AudioCaptureEngine`.
    private let beginVoiceHold: @MainActor () -> Void
    private let endVoiceHold: @MainActor () -> Void

    // MARK: - Adapter state

    @ObservationIgnored private var machine = VoiceSessionMachine()
    @ObservationIgnored private var ticker: Task<Void, Never>?
    /// Stable ID stamped on every voice.* record of one session (#354).
    @ObservationIgnored private var sessionID = UUID()

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
        pauseSpeaking: @escaping @MainActor () -> Void,
        resumeSpeaking: @escaping @MainActor () -> Void,
        speechState: @escaping @MainActor () -> SpeechState,
        currentConversationID: @escaping @MainActor () -> UUID?,
        overlay: CompanionVoicePrototype,
        recorder: CompanionFlightRecorder,
        settings: SettingsManager,
        proofreadPass: ProofreadPass?,
        playbackLevel: @escaping @MainActor () -> Float = { 0 },
        fadeSpeech: @escaping @MainActor (Float, TimeInterval) -> Void = { _, _ in },
        beginVoiceHold: @escaping @MainActor () -> Void = {},
        endVoiceHold: @escaping @MainActor () -> Void = {}
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
        self.beginVoiceHold = beginVoiceHold
        self.endVoiceHold = endVoiceHold
    }

    // MARK: - Entry / exit

    func toggle() {
        if isActive { exit(reason: "toggled-off") } else { enter(via: "chat-toggle") }
    }

    func enter(via: String) {
        guard !isActive else { return }
        sessionID = UUID()
        dispatch(.enter(via: via, tunables: currentTunables))
    }

    func exit(reason: String) {
        dispatch(.exit(reason: reason))
    }

    // MARK: - The reply hook (ChatSession's agentEnd calls this)

    /// Returns true when the session consumed the reply (suppresses the
    /// chat's own autoSpeak).
    func replyCompleted(_ text: String?) -> Bool {
        guard isActive, phase == .awaitingReply || phase == .transcribing else {
            return false
        }
        dispatch(.replyArrived(text))
        return true
    }

    // MARK: - Barge-in

    /// The hard (immediate-pause) barge — the overlay's click, which is
    /// deliberate and has zero false positives in the field data. A click
    /// while a Soft Barge is verifying commits it instead.
    func bargeIn(source: String) {
        dispatch(.clickBarge(source: source))
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
        let level = meterLevel()
        overlay.feed.setMeter(level: level, spectrum: meterSpectrum())
        let engineState = speechState()
        dispatch(
            .tick(
                VoiceSessionMachine.Tick(
                    level: level,
                    playbackLevel: playbackLevel(),
                    speechActive: engineState.isActive,
                    speechDescription: String(describing: engineState)),
                tunables: currentTunables))
    }

    // MARK: - Dispatch

    /// Fold an event through the machine and perform its effects, in order.
    /// Effects with results (`openCapture`, `finishTake`) feed back as
    /// events in the same loop, so machine state is settled before the next
    /// external event arrives.
    private func dispatch(_ event: VoiceSessionMachine.Event) {
        var pending: [VoiceSessionMachine.Event] = [event]
        while !pending.isEmpty {
            let next = pending.removeFirst()
            let effects = machine.handle(next, at: Date().timeIntervalSinceReferenceDate)
            phase = machine.phase
            for effect in effects {
                if let feedback = perform(effect) { pending.append(feedback) }
            }
        }
        syncTicker()
    }

    /// The ticker runs exactly while a session is live — derived from the
    /// machine's phase after every dispatch, not tracked separately.
    private func syncTicker() {
        if machine.isActive {
            if ticker == nil { startTicker() }
        } else {
            ticker?.cancel()
            ticker = nil
        }
    }

    private func perform(_ effect: VoiceSessionMachine.Effect) -> VoiceSessionMachine.Event? {
        switch effect {
        case .beginVoiceHold:
            beginVoiceHold()
        case .endVoiceHold:
            endVoiceHold()
        case .overlayBeginSession:
            overlay.beginLiveSession(
                actions: CompanionVoiceActions(
                    engage: {},
                    bargeIn: { [weak self] in self?.bargeIn(source: "click") },
                    dismiss: { [weak self] in self?.exit(reason: "dismissed") },
                    openChat: { [weak self] in self?.overlay.openChatFromLiveSession() }
                ))
        case .overlayEndSession:
            overlay.endLiveSession()
        case .feedState(let state):
            overlay.feed.setState(feedState(state))
        case .presentSpokenReply(let text):
            overlay.feed.settle(role: .companion, text: text)
            overlay.feed.beginSpokenLine(text)
            overlay.feed.revealAll()
            overlay.feed.setState(.speaking)
        case .settleOwnerLine(let text):
            overlay.feed.settle(role: .owner, text: text)
        case .openCapture:
            if case .started = capture.start() {
                return .captureOpened
            }
            return .captureUnavailable
        case .closeCapture:
            capture.cancel()
        case .finishTake:
            return finishTake()
        case .speak(let text):
            speak(text) { [weak self] in self?.dispatch(.speechDone) }
        case .stopSpeaking:
            stopSpeaking()
        case .pauseSpeaking:
            pauseSpeaking()
        case .resumeSpeaking:
            resumeSpeaking()
        case .fadeSpeech(let target, let duration):
            fadeSpeech(target, duration)
        case .send(let text):
            sendMessage(text)
        case .stageToComposer(let text):
            stageToComposer(text)
        case .record(let event, let snapshot):
            recordVoice(event, snapshot: snapshot)
        }
        return nil
    }

    /// Close the take and run ASR + proofread on it; outcomes come back as
    /// machine events — from a later main-actor turn for the async pipeline,
    /// synchronously for a dead take.
    private func finishTake() -> VoiceSessionMachine.Event? {
        switch capture.stop() {
        case .noAudio, .tooShort:
            return .takeUnusable(reason: "no-audio")
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
                    self?.dispatch(.turnTranscribed(text))
                }
                switch outcome {
                case .committed, .superseded:
                    break
                case .rejected(let raw, _):
                    // A rejected proofread is still his words — voice flows on.
                    self.dispatch(.turnTranscribed(raw))
                case .empty:
                    self.dispatch(.takeUnusable(reason: "empty"))
                case .failed, .cancelled:
                    self.dispatch(.takeUnusable(reason: "failed"))
                }
            }
            return nil
        }
    }

    private func feedState(_ state: VoiceSessionMachine.FeedState) -> CompanionVoiceFeed.State {
        switch state {
        case .listening: .listening
        case .thinking: .thinking
        case .speaking: .speaking
        }
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

    // MARK: - Tunables (the taste ledger, as Settings)

    private var currentTunables: VoiceSessionMachine.Tunables {
        VoiceSessionMachine.Tunables(
            trailingSilence: settings.companionVoiceTrailingSilence,
            sessionTimeout: settings.companionVoiceSessionTimeout,
            bargeInLevel: Float(settings.companionVoiceBargeInLevel),
            autoSend: settings.companionVoiceAutoSend)
    }
}
