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

    // MARK: - Voice hold (Dual-Path Playback, ADR-0041)

    /// What `beginVoiceHold` does about the wiring. Every wire-up happens on
    /// a stopped engine — tap install/remove and the render-side connection
    /// on a running VP engine are the 2026-07-17 crash class
    /// (`CreateRecordingTap` → `SetOutputFormat` → `SetFormat`).
    enum HoldBeginAction: Equatable {
        /// Engine missing or dirty: rebuild (armed per the lifecycle), then wire.
        case rebuildThenWire
        /// Engine healthy and stopped: wire the hold now.
        case wireNow
        /// A capture is mid-take: mark the wiring pending — that capture's
        /// `stopCapture` stops the engine and the wiring runs there. Until
        /// then the hold is capture-only and playback falls back.
        case deferToCaptureStop
    }

    func holdBeginAction(
        engineExists: Bool, needsRebuild: Bool, isCapturing: Bool, engineArmed: Bool
    ) -> HoldBeginAction {
        guard !isCapturing else { return .deferToCaptureStop }
        // The hold wants VP armed for the whole session — a kept plain
        // engine (the fallback lifecycle's idle) is rebuilt armed, never
        // wired plain: a hold without the AEC is the bug the ADR exists to
        // fix.
        let usable = engineExists && !needsRebuild && engineArmed
        return usable ? .wireNow : .rebuildThenWire
    }

    /// A capture stop keeps the engine running only under a fully wired hold
    /// — every other stop lands on a stopped engine: the non-hold path, and
    /// the pending-hold stop that must free the engine for wiring.
    func captureStopKeepsEngineRunning(holdWired: Bool) -> Bool { holdWired }

    /// The pending hold's wiring runs on the stopped engine right after the
    /// in-progress capture's stop — never by installing anything on the
    /// running engine.
    func shouldWireHoldAfterCaptureStop(holdActive: Bool, holdWired: Bool) -> Bool {
        holdActive && !holdWired
    }

    /// A rebuild (device change, wedge teardown) under an active hold
    /// re-wires the hold on the fresh engine so the session's next reply can
    /// attach — the reply that was playing was invalidated by the teardown.
    func shouldRewireAfterRebuild(holdActive: Bool) -> Bool { holdActive }

    /// Hosted playback requires the AEC — an engine that refused to arm (or
    /// whose render side failed verification) hosting playback buys nothing
    /// acoustically, so the reply falls back to the dedicated engine.
    func hostsPlayback(armed: Bool, renderVerified: Bool) -> Bool {
        armed && renderVerified
    }

    /// The fallback lifecycle's post-capture disarm grace never fires under a
    /// hold: the held engine is running (disarm requires stopped), and the
    /// session wants VP for the whole conversation anyway.
    func shouldDisarmAfterCapture(holdActive: Bool) -> Bool {
        disarmsAfterCapture && !holdActive
    }
}
