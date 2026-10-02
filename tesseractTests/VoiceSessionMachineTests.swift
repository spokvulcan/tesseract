//
//  VoiceSessionMachineTests.swift
//  tesseractTests
//
//  The Voice Session Machine's decision tables (ADR-0042, half-duplex since
//  ADR-0082): whole sessions — listen → turn → reply → interrupt → listen →
//  mutual silence — replayed as event sequences against the pure machine,
//  with no ticker, no CoreAudio, and no wall clock. Every scenario that
//  previously required hardware (the watchdog's 2026-07-16 trace, the
//  capture-retry freeze, a dead microphone) is a table here.
//

import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Harness

/// Drives the machine the way the controller does: performs the dispatch
/// loop's effect feedback (`openCapture` → opened/unavailable per
/// `micAvailable`) and advances a virtual 20 Hz clock.
struct VoiceSessionMachineHarness {
    var machine = VoiceSessionMachine()
    var now: TimeInterval = 0
    var micAvailable = true
    var tunables = VoiceSessionMachine.Tunables(
        trailingSilence: 1.8, sessionTimeout: 30, autoSend: true)

    @discardableResult
    mutating func send(_ event: VoiceSessionMachine.Event) -> [VoiceSessionMachine.Effect] {
        var batch: [VoiceSessionMachine.Effect] = []
        var pending = [event]
        while !pending.isEmpty {
            let effects = machine.handle(pending.removeFirst(), at: now)
            batch += effects
            for effect in effects where effect == .openCapture {
                pending.append(micAvailable ? .captureOpened : .captureUnavailable)
            }
        }
        return batch
    }

    /// One 50 ms tick — the virtual clock advances first, like the real
    /// ticker's sleep-then-tick.
    @discardableResult
    mutating func tick(
        level: Float = 0.02, inputDead: Bool = false, speechActive: Bool = false
    ) -> [VoiceSessionMachine.Effect] {
        now += 0.05
        return send(
            .tick(
                VoiceSessionMachine.Tick(
                    level: level, inputDead: inputDead, speechActive: speechActive),
                tunables: tunables))
    }

    @discardableResult
    mutating func ticks(
        _ count: Int, level: Float = 0.02, speechActive: Bool = false
    ) -> [VoiceSessionMachine.Effect] {
        var all: [VoiceSessionMachine.Effect] = []
        for _ in 0..<count {
            all += tick(level: level, speechActive: speechActive)
        }
        return all
    }

    mutating func enterListening() {
        send(.enter(via: "test", tunables: tunables))
    }

    /// Reach `.speaking` through real events: enter → a transcribed turn
    /// (auto-send) → the reply. Leans on `turnTranscribed` having no phase
    /// guard — staleness is the upstream Operation Guard's job.
    mutating func startSpeaking(reply: String = "the reply") {
        enterListening()
        send(.turnTranscribed("hi"))
        send(.replyArrived(reply))
    }

    /// One whole spoken turn from listening: onset, speech, trailing
    /// silence — the take closes and the machine is transcribing. Long
    /// enough to ride out a post-utterance grace before the onset.
    mutating func speakATurn() {
        ticks(15, level: 0.6)
        ticks(40, level: 0.02)
    }
}

extension [VoiceSessionMachine.Effect] {
    var recordNames: [String] {
        compactMap {
            if case .record(let event, _) = $0 { event.rawValue } else { nil }
        }
    }

    func contains(record name: String) -> Bool { recordNames.contains(name) }

    func snapshot(of name: String) -> [String: String]? {
        for effect in self {
            if case .record(let event, let snapshot) = effect, event.rawValue == name {
                return snapshot
            }
        }
        return nil
    }
}

// MARK: - The loop's decision tables

@Suite struct VoiceSessionMachineTests {

    private typealias Effect = VoiceSessionMachine.Effect

    // MARK: Entry / exit

    @Test func enterOpensTheOverlayAndListens() {
        var h = VoiceSessionMachineHarness()
        let effects = h.send(.enter(via: "test", tunables: h.tunables))
        #expect(
            effects == [
                .record(event: .voiceSessionEntered, snapshot: ["via": "test"]),
                .overlayBeginSession,
                .openCapture,
                .feedState(.listening),
            ])
        #expect(h.machine.phase == .listening)
    }

    @Test func enterWhileActiveIsIgnored() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        #expect(h.send(.enter(via: "again", tunables: h.tunables)).isEmpty)
    }

    @Test func mutualSilenceTimeoutExitsTheSession() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.now = 30.0
        let effects = h.tick()
        #expect(effects.contains(.closeCapture))
        #expect(effects.contains(.overlayEndSession))
        #expect(effects.snapshot(of: "voice.session-exited")?["reason"] == "mutual-silence")
        #expect(h.machine.phase == .idle)
    }

    @Test func exitStopsSpeakingOnlyWhenSpeaking() {
        var quiet = VoiceSessionMachineHarness()
        quiet.enterListening()
        #expect(!quiet.send(.exit(reason: "test")).contains(.stopSpeaking))

        var speaking = VoiceSessionMachineHarness()
        speaking.startSpeaking()
        let effects = speaking.send(.exit(reason: "test"))
        #expect(effects.first == .stopSpeaking)
        // The mic was already closed for the reply.
        #expect(!effects.contains(.closeCapture))
    }

    @Test func exitClosesTheCaptureAndEndsTheOverlay() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        let effects = h.send(.exit(reason: "dismissed"))
        #expect(
            effects == [
                .closeCapture,
                .record(
                    event: .voiceSessionExited,
                    snapshot: ["reason": "dismissed", "exchanges": "0"]),
                .overlayEndSession,
            ])
        #expect(h.machine.phase == .idle)
    }

    // MARK: Listening → turn

    @Test func speechOnsetStartsTheCapture() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.ticks(7, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func trailingSilenceFinishesTheTurn() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.ticks(6, level: 0.6)
        let effects = h.ticks(38, level: 0.02)
        #expect(effects.contains(.finishTake))
        #expect(effects.contains(.feedState(.thinking)))
        #expect(h.machine.phase == .transcribing)
    }

    @Test func transcribedTurnSendsAndAwaitsTheReply() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        let effects = h.send(.turnTranscribed("  hello  "))
        #expect(effects.contains(.settleOwnerLine("hello")))
        #expect(effects.contains(.send("hello")))
        #expect(effects.snapshot(of: "voice.owner-turn")?["chars"] == "5")
        #expect(h.machine.phase == .awaitingReply)
    }

    @Test func autoSendOffStagesToComposerAndExits() {
        var h = VoiceSessionMachineHarness()
        h.tunables.autoSend = false
        h.enterListening()
        let effects = h.send(.turnTranscribed("a note"))
        #expect(effects.contains(.stageToComposer("a note")))
        #expect(!effects.contains(.send("a note")))
        #expect(
            effects.snapshot(of: "voice.session-exited")?["reason"] == "staged-to-composer")
        #expect(h.machine.phase == .idle)
    }

    @Test func unusableTakeReturnsToListening() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        let effects = h.send(.takeUnusable(reason: "empty"))
        #expect(effects.contains(.openCapture))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
    }

    // MARK: The reply (half-duplex)

    @Test func replyArrivesAndSpeaks() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.send(.turnTranscribed("hi"))
        let effects = h.send(.replyArrived("hello there"))
        #expect(
            effects == [
                .closeCapture,
                .presentSpokenReply("hello there"),
                .speak("hello there"),
                .record(event: .voiceReplySpoken, snapshot: ["chars": "11"]),
            ])
        #expect(h.machine.phase == .speaking)
    }

    @Test func aReplyAfterAClosedTakeSpeaksWithoutTouchingTheMic() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        h.send(.turnTranscribed("hi"))
        let effects = h.send(.replyArrived("hello"))
        #expect(!effects.contains(.closeCapture))
        #expect(!effects.contains(.openCapture))
        #expect(effects.contains(.speak("hello")))
    }

    @Test func micStaysClosedWhileSpeakingAndLoudTicksDoNotReact() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        // Two seconds of loud input — his own voice in the room, or the
        // owner talking over him: nothing opens, nothing captures.
        let effects = h.ticks(40, level: 0.9, speechActive: true)
        #expect(effects.isEmpty)
        #expect(h.machine.phase == .speaking)
    }

    @Test func silentReplyReopensTheMic() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        h.send(.turnTranscribed("hi"))
        let effects = h.send(.replyArrived(nil))
        #expect(effects.contains(.openCapture))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
    }

    @Test func replyOutsideItsPhasesIsIgnored() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        #expect(h.send(.replyArrived("stray")).isEmpty)
        #expect(h.machine.phase == .listening)
    }

    @Test func aTurnThatLandsWhileSpeakingStopsTheReplyFirst() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.send(.turnTranscribed("wait"))
        #expect(effects.first == .stopSpeaking)
        #expect(effects.contains(.send("wait")))
        #expect(!effects.contains(.openCapture))
        #expect(h.machine.phase == .awaitingReply)
    }

    @Test func aLateUnusableTakeNeverOpensTheMicUnderTheReply() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        #expect(h.send(.takeUnusable(reason: "empty")).isEmpty)
        #expect(h.send(.turnTranscribed("   ")).isEmpty)
        #expect(h.machine.phase == .speaking)
    }

    // MARK: Barge-in (a key or a click)

    @Test func bargeInStopsTheReplyAndListens() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.now += 2.0
        let effects = h.send(.bargeIn(source: "key"))
        #expect(effects.first == .stopSpeaking)
        let barge = effects.snapshot(of: "voice.barge-in")
        #expect(barge?["source"] == "key")
        #expect(barge?["offsetSeconds"] == "2.0")
        // Speech stops before the mic opens.
        let stop = effects.firstIndex(of: .stopSpeaking)
        let open = effects.firstIndex(of: .openCapture)
        #expect(stop != nil && open != nil)
        if let stop, let open { #expect(stop < open) }
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
    }

    @Test func aClickIsRecordedAsItsSource() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.send(.bargeIn(source: "click"))
        #expect(effects.snapshot(of: "voice.barge-in")?["source"] == "click")
        #expect(h.machine.phase == .listening)
    }

    @Test func bargeInOutsideSpeakingDoesNothing() {
        var idle = VoiceSessionMachineHarness()
        #expect(idle.send(.bargeIn(source: "key")).isEmpty)
        #expect(idle.machine.phase == .idle)

        var listening = VoiceSessionMachineHarness()
        listening.enterListening()
        #expect(listening.send(.bargeIn(source: "key")).isEmpty)
        #expect(listening.machine.phase == .listening)

        var capturing = VoiceSessionMachineHarness()
        capturing.enterListening()
        capturing.ticks(7, level: 0.6)
        #expect(capturing.send(.bargeIn(source: "click")).isEmpty)
        #expect(capturing.machine.phase == .capturing)

        var awaiting = VoiceSessionMachineHarness()
        awaiting.enterListening()
        awaiting.send(.turnTranscribed("hi"))
        #expect(awaiting.send(.bargeIn(source: "key")).isEmpty)
        #expect(awaiting.machine.phase == .awaitingReply)
    }

    @Test func theOwnerIsHeardAfterTheGraceThatFollowsAnInterrupt() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.send(.bargeIn(source: "key"))
        // The reply's room tail inside the grace cannot seed a turn…
        h.ticks(5, level: 0.6)
        #expect(h.machine.phase == .listening)
        // …but the owner speaking after it can.
        h.ticks(7, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func nothingResumesAnInterruptedReply() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking(reply: "a long answer")
        h.send(.bargeIn(source: "click"))
        // A stray success callback from the stopped reply is inert.
        #expect(h.send(.speechDone).isEmpty)
        // An empty take after the interrupt just listens again.
        h.speakATurn()
        let effects = h.send(.takeUnusable(reason: "empty"))
        #expect(!effects.contains(.speak("a long answer")))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
        // A second interrupt has nothing to stop.
        #expect(h.send(.bargeIn(source: "key")).isEmpty)
    }

    // MARK: Other speech (half-duplex holds for it too)

    @Test func otherSpeechHoldsTheMicClosedWhileListening() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        // The chat starts reading a reply aloud: the mic closes…
        #expect(h.tick(speechActive: true) == [.closeCapture])
        // …and stays closed through it, loud or not, past the session timeout.
        h.now += 40
        let held = h.ticks(20, level: 0.9, speechActive: true)
        #expect(!held.contains(.openCapture))
        #expect(h.machine.phase == .listening)
        // It ends: the mic reopens, deaf through the room tail…
        #expect(h.tick().contains(.openCapture))
        h.ticks(5, level: 0.6)
        #expect(h.machine.phase == .listening)
        // …then hears the owner (a few spare ticks: at this clock the grace
        // boundary rounds either way).
        h.ticks(10, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func aKeyStopsOtherSpeechAndListens() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.tick(speechActive: true)
        let effects = h.send(.bargeIn(source: "key"))
        #expect(effects.first == .stopSpeaking)
        #expect(effects.snapshot(of: "voice.barge-in")?["speech"] == "other")
        #expect(effects.contains(.openCapture))
        #expect(h.machine.phase == .listening)
    }

    @Test func otherSpeechOverATurnClosesTheTakeOnWhatWasSaid() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.ticks(7, level: 0.6)
        #expect(h.machine.phase == .capturing)
        let effects = h.tick(level: 0.6, speechActive: true)
        #expect(effects.contains(.finishTake))
        #expect(h.machine.phase == .transcribing)
    }

    // MARK: A take still transcribing (the mic waits for it)

    @Test func aReplyThatLandsDuringATakeNeverSupersedesIt() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        #expect(h.machine.phase == .transcribing)
        // A reply to something typed earlier speaks while his take
        // transcribes…
        h.send(.replyArrived("about the typed question"))
        #expect(h.machine.phase == .speaking)
        // …and an interrupt stops it without opening the mic: a new capture
        // would supersede the take still in flight.
        let interrupted = h.send(.bargeIn(source: "key"))
        #expect(interrupted.first == .stopSpeaking)
        #expect(!interrupted.contains(.openCapture))
        #expect(h.machine.phase == .transcribing)
        // The take lands and goes out as his turn.
        let landed = h.send(.turnTranscribed("my words"))
        #expect(landed.contains(.send("my words")))
        #expect(h.machine.phase == .awaitingReply)
    }

    @Test func aReplyThatEndsBeforeTheTakeWaitsForItsOutcome() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        h.send(.replyArrived("short"))
        let done = h.send(.speechDone)
        #expect(!done.contains(.openCapture))
        #expect(h.machine.phase == .transcribing)
        let died = h.send(.takeUnusable(reason: "empty"))
        #expect(died.contains(.openCapture))
        #expect(h.machine.phase == .listening)
    }

    @Test func aSilentReplyDuringATakeLeavesTheMicClosed() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        #expect(h.send(.replyArrived(nil)).isEmpty)
        #expect(h.machine.phase == .transcribing)
        #expect(h.send(.turnTranscribed("still mine")).contains(.send("still mine")))
    }

    @Test func anUnusableTakeUnderAReplyLetsTheReplyFinish() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        h.send(.replyArrived("the reply"))
        #expect(h.send(.takeUnusable(reason: "empty")).isEmpty)
        #expect(h.machine.phase == .speaking)
        // The reply's end opens the mic as usual.
        #expect(h.send(.speechDone).contains(.openCapture))
        #expect(h.machine.phase == .listening)
    }

    // MARK: Utterance end

    @Test func speechDoneReturnsToListeningDeafThroughTheGrace() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.send(.speechDone)
        #expect(effects.first == .stopSpeaking)
        #expect(effects.contains(.openCapture))
        #expect(!effects.contains(record: "voice.barge-in"))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)

        // The room tail inside the post-utterance grace cannot seed a turn…
        h.ticks(5, level: 0.6)
        #expect(h.machine.phase == .listening)
        // …but the owner speaking after it can.
        h.ticks(7, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func watchdogExitsOnlyOnASustainedSettledReading() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        // Inside the 1 s grace nothing counts as settled (0.90 s of ticks —
        // clear of the boundary, which float-accumulated ticks would graze).
        h.ticks(18, level: 0.02, speechActive: false)
        #expect(h.machine.phase == .speaking)
        // Clearly past the grace: a settled run broken by one active sample
        // resets the counter, so five-and-five never exits.
        h.now = 2.0
        h.ticks(5, level: 0.02, speechActive: false)
        h.ticks(1, level: 0.02, speechActive: true)
        h.ticks(5, level: 0.02, speechActive: false)
        #expect(h.machine.phase == .speaking)
        // A sustained settled reading exits.
        let effects = h.ticks(2, level: 0.02, speechActive: false)
        #expect(effects.contains(record: "voice.watchdog-exit"))
        #expect(effects.contains(.stopSpeaking))
        #expect(effects.contains(.openCapture))
        #expect(h.machine.phase == .listening)
    }

    // MARK: Dead input (the capture engine's live-input check)

    @Test func deadInputWhileListeningClosesAndReopensOnTheBackoff() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        let effects = h.tick(inputDead: true)
        #expect(
            effects == [
                .closeCapture,
                .record(event: .voiceCaptureDead, snapshot: ["count": "1"]),
            ])
        #expect(h.machine.phase == .listening)
        // No reopen at tick cadence…
        #expect(!h.ticks(10).contains(.openCapture))
        // …only past the backoff, on a fresh capture.
        h.now += 1.0
        #expect(h.tick().contains(.openCapture))
        h.ticks(8, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func twoDeadCapturesInARowExitTheSession() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.tick(inputDead: true)
        h.now += 1.0
        #expect(h.tick().contains(.openCapture))
        let effects = h.tick(inputDead: true)
        #expect(effects.snapshot(of: "voice.capture-dead")?["count"] == "2")
        #expect(effects.snapshot(of: "voice.session-exited")?["reason"] == "capture-dead")
        #expect(effects.contains(.overlayEndSession))
        #expect(h.machine.phase == .idle)
    }

    @Test func hearingTheOwnerResetsTheDeadCount() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.tick(inputDead: true)
        h.now += 1.0
        h.tick()
        // The reopened input works: he is heard, and his take ends.
        h.speakATurn()
        h.send(.takeUnusable(reason: "empty"))
        let effects = h.tick(inputDead: true)
        #expect(effects.snapshot(of: "voice.capture-dead")?["count"] == "1")
        #expect(h.machine.phase == .listening)
    }

    @Test func aDeadFlagWithoutAnOpenCaptureIsIgnored() {
        // The mic is busy elsewhere (a dictation take): its dead input is
        // that capture's owner's business.
        var h = VoiceSessionMachineHarness()
        h.micAvailable = false
        h.enterListening()
        let effects = h.tick(inputDead: true)
        #expect(!effects.contains(.closeCapture))
        #expect(!effects.contains(record: "voice.capture-dead"))
        #expect(h.machine.phase == .listening)
    }

    // MARK: Capture backoff

    @Test func failedCaptureStartsRetryAtBackoffCadenceNeverTickCadence() {
        var h = VoiceSessionMachineHarness()
        h.micAvailable = false
        let entered = h.send(.enter(via: "test", tunables: h.tunables))
        #expect(entered.contains(.openCapture))
        // The next tick must not retry — the 2026-07-17 freeze was a 20 Hz
        // retry of a failing CoreAudio start.
        #expect(!h.tick().contains(.openCapture))
        // Past the backoff it retries…
        h.now += 1.0
        #expect(h.tick().contains(.openCapture))
        // …and a recovered mic opens.
        h.micAvailable = true
        h.now += 1.0
        #expect(h.tick().contains(.openCapture))
        h.ticks(8, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    // MARK: Late outcomes (the post-exit zombie fix)

    @Test func takeOutcomesAfterExitAreInert() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.speakATurn()
        #expect(h.machine.phase == .transcribing)
        h.send(.exit(reason: "dismissed"))
        #expect(h.send(.turnTranscribed("hello")).isEmpty)
        #expect(h.send(.takeUnusable(reason: "empty")).isEmpty)
        #expect(h.send(.bargeIn(source: "key")).isEmpty)
        // A capture that opens after the end has no owner: closed at once.
        #expect(h.send(.captureOpened) == [.closeCapture])
        #expect(h.machine.phase == .idle)
    }

    @Test func theTimeoutNeverOpensACaptureItWouldLeaveRunning() {
        var h = VoiceSessionMachineHarness()
        h.micAvailable = false
        h.enterListening()  // the start fails; the retry waits out the backoff
        h.micAvailable = true
        h.now = 30.0  // the retry and the timeout fall on one tick
        let effects = h.tick()
        #expect(!effects.contains(.openCapture))
        #expect(effects.snapshot(of: "voice.session-exited")?["reason"] == "mutual-silence")
        #expect(h.machine.phase == .idle)
    }
}
