//
//  VoiceProcessingDuckPolicyTests.swift
//  tesseractTests
//
//  Pins the **System Audio Duck** policy (PRD #188 / ADR-0025) against a fake
//  port — the duck/un-duck sequencing that regressed twice in one week when it
//  lived inline in the capture engine. The port carries the platform effects
//  (VPIO ducking level, `AudioDeviceDuck` un-duck, default-output watcher);
//  these tests assert only the *sequence of treatments* the policy requests,
//  never how the adapter performs them. The audible result is hardware
//  territory (manual acceptance checklist, PRD #188).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
final class FakeSystemAudioDucking: SystemAudioDucking {
    enum Call: Equatable {
        case restoreIdle
        case duckForRecording
    }

    var isUnduckAvailable = true
    private(set) var calls: [Call] = []
    private(set) var defaultOutputChangeHandler: (@MainActor () -> Void)?

    func duckForRecording() { calls.append(.duckForRecording) }
    func restoreIdleTreatment() { calls.append(.restoreIdle) }
    func setDefaultOutputChangeHandler(_ handler: (@MainActor () -> Void)?) {
        defaultOutputChangeHandler = handler
    }

    func fireDefaultOutputChange() { defaultOutputChangeHandler?() }
}

@MainActor
struct VoiceProcessingDuckPolicyTests {

    @Test
    func unduckAvailabilitySelectsTheLifecycle() {
        // ADR-0025's fallback ladder: the SPI present means the engine can stay
        // armed forever (idle is un-ducked); absent, the only full duck release
        // is disarming, so the disarm-after-grace lifecycle returns.
        let available = FakeSystemAudioDucking()
        available.isUnduckAvailable = true
        #expect(VoiceProcessingDuckPolicy(port: available).lifecycle == .alwaysArmed)

        let absent = FakeSystemAudioDucking()
        absent.isUnduckAvailable = false
        #expect(VoiceProcessingDuckPolicy(port: absent).lifecycle == .disarmAfterGrace)
    }

    @Test
    func armingAppliesTheIdleTreatment() {
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()

        #expect(port.calls == [.restoreIdle])
    }

    @Test
    func recordingRaisesTheDuckAndStopRestoresIdle() {
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()
        policy.captureDidStart(meteringOnly: false)
        policy.captureDidStop()

        #expect(port.calls == [.restoreIdle, .duckForRecording, .restoreIdle])
    }

    @Test
    func meteringOnlyCaptureNeverGetsTheRecordingLevel() {
        // The settings level meter must not punish background audio (PRD #188):
        // a metering-only capture keeps the idle treatment end to end.
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()
        policy.captureDidStart(meteringOnly: true)
        policy.captureDidStop()

        #expect(!port.calls.contains(.duckForRecording))
    }

    @Test
    func defaultOutputChangeWhileIdleRefiresTheIdleTreatment() {
        // The un-duck targets a device *by ID* — plugging in headphones makes
        // the previous un-duck a silent no-op on the wrong device, so the idle
        // treatment re-fires at the new default output (ADR-0025).
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()
        port.fireDefaultOutputChange()

        #expect(port.calls == [.restoreIdle, .restoreIdle])
    }

    @Test
    func defaultOutputChangeWhileRecordingDefersToTheStop() {
        // Mid-recording the duck is *desired* — a device change must not lift
        // it; the stop restores idle on the new device anyway.
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()
        policy.captureDidStart(meteringOnly: false)
        port.fireDefaultOutputChange()
        #expect(port.calls == [.restoreIdle, .duckForRecording])

        policy.captureDidStop()
        #expect(port.calls == [.restoreIdle, .duckForRecording, .restoreIdle])
    }

    @Test
    func unarmedCaptureGetsNoTreatment() {
        // Raw fallback (the platform refused Voice Processing): no VPIO exists,
        // so there is no duck to manage — the port must stay untouched.
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.captureDidStart(meteringOnly: false)
        port.fireDefaultOutputChange()
        policy.captureDidStop()

        #expect(port.calls.isEmpty)
    }

    @Test
    func disarmingStopsTreatmentUntilTheNextArm() {
        // The disarm-after-grace fallback (and the wedge teardown) leave no
        // armed VPIO behind; treatments resume only when an engine re-arms.
        let port = FakeSystemAudioDucking()
        let policy = VoiceProcessingDuckPolicy(port: port)

        policy.engineDidArm()
        policy.engineDidDisarm()
        port.fireDefaultOutputChange()
        policy.captureDidStart(meteringOnly: false)

        #expect(port.calls == [.restoreIdle])

        policy.engineDidArm()
        #expect(port.calls == [.restoreIdle, .restoreIdle])
    }
}
