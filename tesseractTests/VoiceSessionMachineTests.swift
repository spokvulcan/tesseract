//
//  VoiceSessionMachineTests.swift
//  tesseractTests
//
//  The Voice Session Machine's decision tables (ADR-0042): whole sessions —
//  listen → speak → false barge → escalation → confirm → turn → mutual
//  silence — replayed as event sequences against the pure machine, with no
//  ticker, no CoreAudio, and no wall clock. Every scenario that previously
//  required hardware (the 2026-07-17 flap storm, the watchdog's self-echo
//  trace, the capture-retry freeze) is a table here.
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
        trailingSilence: 1.8, sessionTimeout: 30, bargeInLevel: 0.25, autoSend: true)

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
        level: Float = 0.02, playback: Float = 0, speechActive: Bool = true
    ) -> [VoiceSessionMachine.Effect] {
        now += 0.05
        return send(
            .tick(
                VoiceSessionMachine.Tick(
                    level: level, playbackLevel: playback, speechActive: speechActive),
                tunables: tunables))
    }

    @discardableResult
    mutating func ticks(
        _ count: Int, level: Float = 0.02, playback: Float = 0, speechActive: Bool = true
    ) -> [VoiceSessionMachine.Effect] {
        var all: [VoiceSessionMachine.Effect] = []
        for _ in 0..<count {
            all += tick(level: level, playback: playback, speechActive: speechActive)
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

    @Test func enterOpensHoldOverlayAndListens() {
        var h = VoiceSessionMachineHarness()
        let effects = h.send(.enter(via: "test", tunables: h.tunables))
        #expect(
            effects == [
                .beginVoiceHold,
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
        #expect(effects.contains(.endVoiceHold))
        #expect(effects.contains(.overlayEndSession))
        #expect(effects.snapshot(of: "voice.session-exited")?["reason"] == "mutual-silence")
        #expect(h.machine.phase == .idle)
    }

    @Test func exitStopsSpeakingOnlyWhenSpeakingOrBarged() {
        var quiet = VoiceSessionMachineHarness()
        quiet.enterListening()
        #expect(!quiet.send(.exit(reason: "test")).contains(.stopSpeaking))

        var speaking = VoiceSessionMachineHarness()
        speaking.startSpeaking()
        let effects = speaking.send(.exit(reason: "test"))
        #expect(effects.first == .stopSpeaking)
    }

    @Test func exitClosesCaptureBeforeEndingTheHold() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        let effects = h.send(.exit(reason: "dismissed"))
        let close = effects.firstIndex(of: .closeCapture)
        let hold = effects.firstIndex(of: .endVoiceHold)
        #expect(close != nil && hold != nil)
        if let close, let hold { #expect(close < hold) }
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
        h.ticks(6, level: 0.6)
        h.ticks(38, level: 0.02)
        let effects = h.send(.takeUnusable(reason: "empty"))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
    }

    // MARK: The reply

    @Test func replyArrivesAndSpeaks() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.send(.turnTranscribed("hi"))
        let effects = h.send(.replyArrived("hello there"))
        #expect(
            effects == [
                .presentSpokenReply("hello there"),
                .speak("hello there"),
                .record(event: .voiceReplySpoken, snapshot: ["chars": "11"]),
            ])
        #expect(h.machine.phase == .speaking)
    }

    @Test func silentReplyReopensTheMic() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        h.send(.turnTranscribed("hi"))
        let effects = h.send(.replyArrived(nil))
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)
    }

    @Test func replyOutsideItsPhasesIsIgnored() {
        var h = VoiceSessionMachineHarness()
        h.enterListening()
        #expect(h.send(.replyArrived("stray")).isEmpty)
        #expect(h.machine.phase == .listening)
    }

    // MARK: Soft Barge (ADR-0041)

    @Test func energyOnsetDucksAndCapturesWithoutPausing() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.ticks(12, level: 0.6)
        #expect(effects.contains(record: "voice.barge-soft-onset"))
        #expect(
            effects.contains(
                .fadeSpeech(
                    target: VoiceSessionMachine.softDuckLevel,
                    duration: VoiceSessionMachine.softDuckRampDown)))
        #expect(effects.contains(.feedState(.listening)))
        #expect(!effects.contains(.pauseSpeaking))
        #expect(h.machine.phase == .capturing)
    }

    @Test func sustainedVoicingHardensTheDuckIntoThePause() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.ticks(12, level: 0.6)
        let effects = h.ticks(8, level: 0.6)
        #expect(effects.contains(.pauseSpeaking))
        #expect(effects.snapshot(of: "reaction.barge-in")?["detector"] == "energy-soft")
    }

    @Test func aSilentConfirmWindowFadesTheReplyBack() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.ticks(12, level: 0.6)
        let effects = h.ticks(17, level: 0.02)
        #expect(effects.snapshot(of: "voice.barge-false-resume")?["reason"] == "soft-fadeback")
        #expect(
            effects.contains(
                .fadeSpeech(target: 1.0, duration: VoiceSessionMachine.softDuckRampUp)))
        #expect(!effects.contains(.resumeSpeaking))
        #expect(effects.contains(.feedState(.speaking)))
        #expect(h.machine.phase == .speaking)

        // Deaf through the restore transient (the 2026-07-17 flap cycle):
        // loud input inside the post-resume grace must not re-onset.
        let deaf = h.ticks(12, level: 0.6)
        #expect(!deaf.contains(record: "voice.barge-soft-onset"))
    }

    @Test func escalationMutesTheEnergyDetectorAfterFourFalseBarges() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        for cycle in 1...4 {
            h.ticks(12, level: 0.6)  // onset → duck
            let fadeback = h.ticks(17, level: 0.02)  // window expires
            #expect(
                fadeback.snapshot(of: "voice.barge-false-resume")?["falseBargeCount"]
                    == String(cycle))
            h.ticks(22, level: 0.02)  // wait out the post-resume deafness
        }
        let suppressed = h.ticks(12, level: 0.6)
        #expect(suppressed.contains(record: "voice.barge-suppressed"))
        #expect(!suppressed.contains(record: "voice.barge-soft-onset"))
        #expect(h.machine.phase == .speaking)
    }

    @Test func escalationWidensTheFloorMarginAfterTwoFalseBarges() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        for _ in 1...2 {
            h.ticks(12, level: 0.6)
            h.ticks(17, level: 0.02)
            h.ticks(22, level: 0.02)
        }
        // Converge the floor on residual: believable = min(0.4, 0.8 − 0.2).
        let effects = h.ticks(20, level: 0.4, playback: 0.8)
        guard let sample = effects.snapshot(of: "voice.energy-sample"),
            let threshold = Float(sample["threshold"] ?? "")
        else {
            Issue.record("no energy sample with a parseable threshold")
            return
        }
        // margin 0.08 × 1.5 over the converged floor (~0.4) — the unscaled
        // margin would sit at ~0.48.
        #expect(threshold > 0.5)
    }

    // MARK: Click barge

    @Test func clickBargePausesImmediately() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.send(.clickBarge(source: "click"))
        #expect(effects.first == .pauseSpeaking)
        #expect(effects.snapshot(of: "reaction.barge-in")?["detector"] == "click")
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .capturing)
    }

    @Test func clickDuringTheSoftWindowCommitsThePause() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.ticks(12, level: 0.6)
        let effects = h.send(.clickBarge(source: "click"))
        #expect(effects.contains(.pauseSpeaking))
        #expect(effects.snapshot(of: "reaction.barge-in")?["detector"] == "click")
        #expect(h.machine.phase == .capturing)
    }

    @Test func aBargeWithNoSpeechResumesTheReply() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.send(.clickBarge(source: "click"))
        let effects = h.ticks(42, level: 0.02)
        #expect(effects.snapshot(of: "voice.barge-false-resume")?["reason"] == "no-speech")
        #expect(effects.contains(.resumeSpeaking))
        #expect(
            effects.contains(
                .fadeSpeech(target: 1.0, duration: VoiceSessionMachine.softDuckRampUp)))
        #expect(h.machine.phase == .speaking)
    }

    @Test func aBargedTurnThatTranscribesStopsTheReplyForGood() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.send(.clickBarge(source: "click"))
        h.ticks(6, level: 0.6)
        h.ticks(38, level: 0.02)
        #expect(h.machine.phase == .transcribing)
        let effects = h.send(.turnTranscribed("stop that"))
        #expect(effects.contains(.stopSpeaking))
        #expect(effects.contains(.send("stop that")))
        #expect(h.machine.phase == .awaitingReply)
    }

    @Test func anEmptyBargedTakeResumesTheReply() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.send(.clickBarge(source: "click"))
        h.ticks(6, level: 0.6)
        h.ticks(38, level: 0.02)
        let effects = h.send(.takeUnusable(reason: "empty"))
        #expect(effects.contains(.resumeSpeaking))
        #expect(h.machine.phase == .speaking)
    }

    // MARK: Utterance end

    @Test func speechDoneReturnsToListeningDeafThroughTheGrace() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        let effects = h.send(.speechDone)
        #expect(effects.first == .stopSpeaking)
        #expect(effects.contains(.feedState(.listening)))
        #expect(h.machine.phase == .listening)

        // The room tail inside the post-utterance grace cannot seed a turn…
        h.ticks(5, level: 0.6)
        #expect(h.machine.phase == .listening)
        // …but the owner speaking after it can.
        h.ticks(7, level: 0.6)
        #expect(h.machine.phase == .capturing)
    }

    @Test func speechDoneWhileBargedRestoresWithoutResuming() {
        var h = VoiceSessionMachineHarness()
        h.startSpeaking()
        h.send(.clickBarge(source: "click"))
        #expect(h.send(.speechDone).isEmpty)
        let effects = h.ticks(42, level: 0.02)
        #expect(effects.contains(.fadeSpeech(target: 1.0, duration: 0)))
        #expect(!effects.contains(.resumeSpeaking))
        #expect(h.machine.phase == .listening)
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
        h.ticks(6, level: 0.6)
        h.ticks(38, level: 0.02)
        #expect(h.machine.phase == .transcribing)
        h.send(.exit(reason: "dismissed"))
        #expect(h.send(.turnTranscribed("hello")).isEmpty)
        #expect(h.send(.takeUnusable(reason: "empty")).isEmpty)
        #expect(h.machine.phase == .idle)
    }
}
