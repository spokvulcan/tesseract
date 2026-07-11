//
//  CaptureEngineLifecycle.swift
//  tesseract
//
//  The **Capture Engine Lifecycle** policy: every keep-vs-rebuild decision
//  the capture engine makes about its kept `AVAudioEngine`, as one pure
//  decision table — the same policy/controller split as the **System Audio
//  Duck** (ADR-0025): the policy decides, the engine performs. These
//  decisions previously lived as inline conditionals at four sites in the
//  engine, tangled with AVFoundation effects and untestable without real
//  audio hardware.
//

import Foundation

/// Which Voice Processing lifecycle the engine runs (ADR-0025's fallback
/// ladder, selected once per process by the un-duck probe).
nonisolated enum VoiceProcessingLifecycle: Sendable, Equatable {
    /// The design: armed once at prewarm, never disarmed — idle is un-ducked
    /// at full volume, so staying armed costs nothing audible.
    case alwaysArmed
    /// The SPI is absent: only disarming fully releases the duck, so the
    /// engine arms per capture and disarms after a grace (the pre-#188
    /// lifecycle — latency returns, correctness doesn't).
    case disarmAfterGrace
}

nonisolated struct CaptureEngineLifecycle: Sendable {

    let voiceProcessing: VoiceProcessingLifecycle

    // MARK: - Tuning

    /// A capture this long with zero tap buffers is a wedged input, not a
    /// quick tap — one 1024-frame buffer arrives within ~25 ms even with
    /// Voice Processing ramp-up. Matches the session's minimum recording
    /// duration, below which the capture is discarded as "too short" anyway.
    let emptyCaptureGrace: TimeInterval = 0.5
    /// Fallback lifecycle only (un-duck unavailable): how long the kept
    /// engine stays armed after a capture ends. Within the grace a re-record
    /// skips the VPIO arm cost; when it lapses, VP is disarmed — with no
    /// un-duck, disarming is the only full duck release.
    let voiceProcessingDisarmGrace: TimeInterval = 10
    /// Arm/disarm and start/stop reconfigure the engine's own graph and fire
    /// `AVAudioEngineConfigurationChange` for our own doing; a notification
    /// landing within this window of an intentional reconfiguration is an
    /// echo, not an external change.
    let selfInflictedConfigChangeWindow: TimeInterval = 1.0
    /// How long after an external configuration change (device swap, format
    /// change) the idle rebuild fires — coalesces the notification burst a
    /// device switch produces into one rebuild, so the arm cost is paid
    /// while idle instead of on the next press.
    let idleRebuildDelay: Duration = .milliseconds(500)
    /// Back-to-back VPIO arming can flake with an undocumented error (WebRTC
    /// retries the same way); the idle rebuild retries once after this beat
    /// before settling for raw capture.
    let armRetryDelay: Duration = .milliseconds(150)

    // MARK: - Decisions

    /// What a press does with the kept engine.
    enum PressAction: Equatable {
        /// No engine, or a dirty one: build fresh with Voice Processing
        /// requested — a press always wants VP, whichever lifecycle runs.
        case rebuildArmed
        /// Keep the engine; `reconcileArm` says the fallback lifecycle must
        /// arm Voice Processing in place first (the once-per-burst cost the
        /// disarm grace amortizes; the always-armed lifecycle never pays it).
        case reuse(reconcileArm: Bool)
    }

    func pressAction(engineExists: Bool, needsRebuild: Bool) -> PressAction {
        guard engineExists, !needsRebuild else { return .rebuildArmed }
        return .reuse(reconcileArm: voiceProcessing == .disarmAfterGrace)
    }

    /// Whether prewarm builds the engine armed: the arm cost (170–600 ms
    /// measured) is paid at launch only under the always-armed lifecycle —
    /// the fallback idles plain and arms per burst.
    var prewarmBuildsArmed: Bool { voiceProcessing == .alwaysArmed }

    /// A configuration-change notification is trusted as a real device or
    /// format change only outside the echo window of our own last
    /// reconfigure (build, arm/disarm, start, stop).
    func isExternalConfigChange(
        sinceLastIntentionalReconfigure elapsed: TimeInterval
    ) -> Bool {
        elapsed >= selfInflictedConfigChangeWindow
    }

    /// What a capture that ended with zero samples means.
    enum EmptyCaptureVerdict: Equatable {
        /// The engine ran long enough that the tap must have fired, yet
        /// delivered nothing: discard the engine and report "no audio" —
        /// the truth — rather than letting an empty transcription claim
        /// "no speech detected".
        case wedgedInput
        /// A tap that beat the first buffer; the session's minimum-duration
        /// guard ("too short") handles it.
        case tapBeatFirstBuffer
    }

    func emptyCaptureVerdict(duration: TimeInterval) -> EmptyCaptureVerdict {
        duration >= emptyCaptureGrace ? .wedgedInput : .tapBeatFirstBuffer
    }

    /// The fallback lifecycle schedules the post-capture disarm grace — with
    /// no un-duck available, the disarm is what finally releases the duck.
    var disarmsAfterCapture: Bool { voiceProcessing == .disarmAfterGrace }

    /// Only the always-armed lifecycle re-arms in the background after an
    /// external change or a wedge teardown; the fallback would just re-duck
    /// the system while idle.
    var rebuildsWhileIdle: Bool { voiceProcessing == .alwaysArmed }

    /// After an idle rebuild, an engine that exists but did not arm gets one
    /// retry — a refusal on rebuild would otherwise silently downgrade every
    /// following capture to raw.
    func idleRebuildNeedsArmRetry(engineExists: Bool, armed: Bool) -> Bool {
        engineExists && !armed
    }
}
