//
//  VoiceProcessingDuckPolicy.swift
//  tesseract
//
//  The **System Audio Duck** policy (PRD #188 / ADR-0025): *when* each duck
//  treatment applies, expressed against the `SystemAudioDucking` port so the
//  sequencing — which regressed twice while it lived inline in the capture
//  engine — is pinned by `VoiceProcessingDuckPolicyTests`. The capture engine
//  reports its lifecycle moments (arm, capture start/stop, disarm); this
//  object turns them into treatments and re-fires the idle treatment when the
//  default output device changes underneath a stale un-duck target.
//

import Foundation

@MainActor
final class VoiceProcessingDuckPolicy {
    /// Which Voice Processing lifecycle the engine should run (ADR-0025's
    /// fallback ladder, selected once per process by the un-duck probe).
    enum Lifecycle {
        /// The design: armed once at prewarm, never disarmed — idle is
        /// un-ducked at full volume, so staying armed costs nothing audible.
        case alwaysArmed
        /// The SPI is absent: only disarming fully releases the duck, so the
        /// engine arms per capture and disarms after a grace (the pre-#188
        /// lifecycle — latency returns, correctness doesn't).
        case disarmAfterGrace
    }

    let lifecycle: Lifecycle

    private let port: any SystemAudioDucking
    private var isArmed = false
    private var isRecording = false

    init(port: any SystemAudioDucking) {
        self.port = port
        self.lifecycle = port.isUnduckAvailable ? .alwaysArmed : .disarmAfterGrace
        port.setDefaultOutputChangeHandler { [weak self] in
            self?.defaultOutputDidChange()
        }
    }

    /// Voice Processing was armed (at prewarm, a rebuild, or the fallback's
    /// per-capture arm): the armed VPIO ducks from this moment, so the idle
    /// treatment applies immediately.
    func engineDidArm() {
        isArmed = true
        isRecording = false
        port.restoreIdleTreatment()
    }

    /// Voice Processing went away — the fallback's disarm or a wedge teardown.
    /// No VPIO, no duck: treatments pause until the next arm.
    func engineDidDisarm() {
        isArmed = false
        isRecording = false
    }

    /// A capture is about to start. A real recording raises the duck to the
    /// standard level; the settings level meter keeps the idle treatment —
    /// checking the microphone must not punish background audio.
    func captureDidStart(meteringOnly: Bool) {
        guard isArmed else { return }
        isRecording = !meteringOnly
        if !meteringOnly {
            port.duckForRecording()
        }
    }

    /// The capture ended: back to the idle treatment (full volume) at once.
    func captureDidStop() {
        guard isArmed else { return }
        isRecording = false
        port.restoreIdleTreatment()
    }

    // MARK: - Private

    private func defaultOutputDidChange() {
        // Mid-recording the duck is desired — the stop re-fires idle on the
        // new device anyway. Idle armed, the un-duck must chase the device.
        guard isArmed, !isRecording else { return }
        port.restoreIdleTreatment()
    }
}
