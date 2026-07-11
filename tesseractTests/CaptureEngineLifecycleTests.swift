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
}
