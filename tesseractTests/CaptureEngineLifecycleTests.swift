//
//  CaptureEngineLifecycleTests.swift
//  tesseractTests
//
//  The **Capture Engine Lifecycle** decision table at its own seam — the
//  keep-vs-rebuild verdicts that previously lived as inline conditionals in
//  the capture engine, testable only with real audio hardware. No
//  AVAudioEngine anywhere.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct CaptureEngineLifecycleTests {

    private let alwaysArmed = CaptureEngineLifecycle(voiceProcessing: .alwaysArmed)
    private let fallback = CaptureEngineLifecycle(voiceProcessing: .disarmAfterGrace)

    // MARK: - Press

    @Test
    func pressRebuildsWhenNoEngineOrDirty() {
        for policy in [alwaysArmed, fallback] {
            #expect(
                policy.pressAction(engineExists: false, needsRebuild: false) == .rebuildArmed)
            #expect(
                policy.pressAction(engineExists: true, needsRebuild: true) == .rebuildArmed)
            #expect(
                policy.pressAction(engineExists: false, needsRebuild: true) == .rebuildArmed)
        }
    }

    @Test
    func pressReusesTheKeptEngineAndOnlyTheFallbackReconcilesTheArm() {
        #expect(
            alwaysArmed.pressAction(engineExists: true, needsRebuild: false)
                == .reuse(reconcileArm: false))
        #expect(
            fallback.pressAction(engineExists: true, needsRebuild: false)
                == .reuse(reconcileArm: true))
    }

    // MARK: - Prewarm

    @Test
    func onlyTheAlwaysArmedLifecyclePrewarmsArmed() {
        #expect(alwaysArmed.prewarmBuildsArmed)
        #expect(!fallback.prewarmBuildsArmed)
    }

    // MARK: - Configuration-change echo window

    @Test
    func configChangeInsideTheEchoWindowIsOurOwnDoing() {
        #expect(!alwaysArmed.isExternalConfigChange(sinceLastIntentionalReconfigure: 0))
        #expect(!alwaysArmed.isExternalConfigChange(sinceLastIntentionalReconfigure: 0.99))
        #expect(alwaysArmed.isExternalConfigChange(sinceLastIntentionalReconfigure: 1.0))
        #expect(alwaysArmed.isExternalConfigChange(sinceLastIntentionalReconfigure: 60))
    }

    // MARK: - Empty capture

    @Test
    func emptyCaptureIsWedgedOnlyAtOrPastTheGrace() {
        #expect(alwaysArmed.emptyCaptureVerdict(duration: 0.1) == .tapBeatFirstBuffer)
        #expect(alwaysArmed.emptyCaptureVerdict(duration: 0.49) == .tapBeatFirstBuffer)
        #expect(alwaysArmed.emptyCaptureVerdict(duration: 0.5) == .wedgedInput)
        #expect(alwaysArmed.emptyCaptureVerdict(duration: 12) == .wedgedInput)
    }

    // MARK: - Background work gating

    @Test
    func onlyTheFallbackDisarmsAfterCapture() {
        #expect(!alwaysArmed.disarmsAfterCapture)
        #expect(fallback.disarmsAfterCapture)
    }

    @Test
    func onlyTheAlwaysArmedLifecycleRebuildsWhileIdle() {
        #expect(alwaysArmed.rebuildsWhileIdle)
        #expect(!fallback.rebuildsWhileIdle)
    }

    @Test
    func armRetryFiresOnlyForABuiltButUnarmedEngine() {
        #expect(alwaysArmed.idleRebuildNeedsArmRetry(engineExists: true, armed: false))
        #expect(!alwaysArmed.idleRebuildNeedsArmRetry(engineExists: true, armed: true))
        #expect(!alwaysArmed.idleRebuildNeedsArmRetry(engineExists: false, armed: false))
    }

    // MARK: - Live-input check

    /// One check's verdict on a long-lived engine, past the first-buffer grace
    /// unless the row says otherwise.
    private func verdict(
        _ policy: CaptureEngineLifecycle, since: Int, total: Int, rebuilt: Bool = false,
        secondsSinceStart: TimeInterval = 2, engineAge: TimeInterval = 3600
    ) -> CaptureEngineLifecycle.LiveInputVerdict {
        policy.liveInputVerdict(
            buffersSinceLastCheck: since, buffersThisCapture: total, rebuiltThisCapture: rebuilt,
            secondsSinceStart: secondsSinceStart, engineAge: engineAge)
    }

    @Test
    func anyBufferSinceTheLastCheckIsAlive() {
        for policy in [alwaysArmed, fallback] {
            #expect(verdict(policy, since: 1, total: 1) == .alive)
            #expect(verdict(policy, since: 5, total: 40, rebuilt: true) == .alive)
            #expect(verdict(policy, since: 1, total: 1, secondsSinceStart: 0.5) == .alive)
        }
    }

    @Test
    func aSlowInputGetsItsFirstSecond() {
        // A Bluetooth headset switching to its microphone profile takes about
        // a second to deliver its first buffer: no rebuild before the grace.
        #expect(verdict(alwaysArmed, since: 0, total: 0, secondsSinceStart: 0.5) == .waiting)
        #expect(verdict(alwaysArmed, since: 0, total: 0, secondsSinceStart: 1.0) == .waiting)
        #expect(alwaysArmed.firstBufferGrace == 1.5)
    }

    @Test
    func anInputThatNeverDeliveredIsRebuiltOnce() {
        // Nothing recorded yet, so the restart is lossless — and it is tried
        // only once per capture.
        #expect(verdict(alwaysArmed, since: 0, total: 0) == .rebuildAndRestart)
        #expect(verdict(alwaysArmed, since: 0, total: 0, rebuilt: true) == .dead)
    }

    @Test
    func anEngineBuiltMomentsAgoIsNotRebuiltAgain() {
        // Back-to-back voice-processing engines are what wedge CoreAudio
        // input: a silent input on a young engine is reported dead instead.
        #expect(verdict(alwaysArmed, since: 0, total: 0, engineAge: 2) == .dead)
        #expect(verdict(alwaysArmed, since: 0, total: 0, engineAge: 6) == .rebuildAndRestart)
        #expect(alwaysArmed.freshEngineAge == 5)
    }

    @Test
    func anInputThatWentQuietMidCaptureIsDead() {
        // Audio arrived, then stopped: a restart would cut the take, so the
        // owner hears about it instead.
        #expect(verdict(alwaysArmed, since: 0, total: 12) == .dead)
        #expect(verdict(alwaysArmed, since: 0, total: 12, rebuilt: true) == .dead)
        #expect(verdict(alwaysArmed, since: 0, total: 12, secondsSinceStart: 0.5) == .dead)
    }

    @Test
    func theCheckRunsAtTheEmptyCaptureGrace() {
        // A live input delivers a buffer every ~100 ms, through silence too.
        #expect(alwaysArmed.liveInputInterval == .milliseconds(500))
        #expect(alwaysArmed.emptyCaptureGrace == 0.5)
    }
}
