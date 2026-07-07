//
//  SystemAudioDucking.swift
//  tesseract
//
//  The **System Audio Duck** port (PRD #188 / ADR-0025): what happens to all
//  *other* system audio while Voice Processing is armed. An armed VPIO ducks
//  everything else for as long as it exists — engine running or stopped — and
//  no public ducking level reaches zero. The port carries the two treatments
//  the capture engine can request plus the levers behind them: the private
//  `AudioDeviceDuck` un-duck (the same weak-linked call Chromium and WebKit
//  ship) and the default-output-device watcher that keeps the un-duck aimed
//  at the right device. `VoiceProcessingDuckPolicy` decides *when*; this
//  adapter only performs.
//

import AVFoundation
import CoreAudio
import Foundation

@MainActor
protocol SystemAudioDucking: AnyObject {
    /// Whether the un-duck lever exists in this process — the probe that
    /// selects between the always-armed and disarm-after-grace lifecycles.
    var isUnduckAvailable: Bool { get }
    /// The recording treatment: the standard VPIO ducking level, so other
    /// audio dips exactly while the user dictates (source-level noise
    /// reduction, PRD #175).
    func duckForRecording()
    /// The idle treatment: the minimum VPIO ducking level as the floor, plus
    /// the un-duck at full volume — by ear, indistinguishable from the app
    /// not running (owner-verified 2026-07-07).
    func restoreIdleTreatment()
    /// Installs the handler re-fired when the default output device changes —
    /// the un-duck targets a device by ID, so a stale target is a silent
    /// no-op on the wrong device.
    func setDefaultOutputChangeHandler(_ handler: (@MainActor () -> Void)?)
}

/// The production adapter: VPIO ducking configuration (applied through a
/// capture-engine-owned closure — only the engine has the input node),
/// the `AudioDeviceDuck` SPI, and the CoreAudio default-output listener.
@MainActor
final class SystemAudioDuckController: SystemAudioDucking {
    /// `AudioDeviceDuck(device, level, when, ramp)` — private CoreAudio SPI,
    /// resolved at runtime and nil-guarded: a future macOS removing it must
    /// degrade to the disarm-after-grace lifecycle, never crash (ADR-0025).
    private typealias AudioDeviceDuckFn =
        @convention(c) (
            AudioObjectID, Float32, UnsafePointer<AudioTimeStamp>?, Float32
        ) -> OSStatus

    private nonisolated static let audioDeviceDuck: AudioDeviceDuckFn? = {
        guard let symbol = dlsym(dlopen(nil, RTLD_NOW), "AudioDeviceDuck") else { return nil }
        return unsafeBitCast(symbol, to: AudioDeviceDuckFn.self)
    }()

    /// Seconds the un-duck ramps back to full volume — long enough to not pop,
    /// short enough that idle never sounds ducked (prototype value).
    private static let unduckRamp: Float32 = 0.5

    private enum Ducking {
        static let idle = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .min)
        static let recording = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .default)
    }

    /// Applies a VPIO ducking configuration to the currently armed input node.
    /// Assigned by `AudioCaptureEngine`, which owns the node; it no-ops while
    /// no engine is armed.
    var duckingConfigurator: ((AVAudioVoiceProcessingOtherAudioDuckingConfiguration) -> Void)?

    private var outputChangeHandler: (@MainActor () -> Void)?
    private var outputListener: AudioObjectPropertyListenerBlock?

    deinit {
        MainActor.assumeIsolated {
            removeOutputListener()
        }
    }

    var isUnduckAvailable: Bool { Self.audioDeviceDuck != nil }

    func duckForRecording() {
        duckingConfigurator?(Ducking.recording)
    }

    func restoreIdleTreatment() {
        duckingConfigurator?(Ducking.idle)
        unduckDefaultOutput()
    }

    func setDefaultOutputChangeHandler(_ handler: (@MainActor () -> Void)?) {
        outputChangeHandler = handler
        if handler != nil {
            installOutputListenerIfNeeded()
        } else {
            removeOutputListener()
        }
    }

    // MARK: - Private

    private func unduckDefaultOutput() {
        guard let unduck = Self.audioDeviceDuck else { return }
        guard let device = Self.defaultOutputDevice() else {
            Log.audio.error("Un-duck skipped: no default output device")
            return
        }
        let status = unduck(device, 1.0, nil, Self.unduckRamp)
        if status == noErr {
            Log.audio.info("Un-ducked output device \(device) to full volume")
        } else {
            Log.audio.error("AudioDeviceDuck(\(device)) failed: \(status)")
        }
    }

    private nonisolated static func defaultOutputAddress() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private nonisolated static func defaultOutputDevice() -> AudioObjectID? {
        var address = defaultOutputAddress()
        var deviceID = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        guard status == noErr, deviceID != kAudioObjectUnknown else { return nil }
        return deviceID
    }

    private func installOutputListenerIfNeeded() {
        guard outputListener == nil else { return }

        var address = Self.defaultOutputAddress()
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            Task { @MainActor in
                self?.outputChangeHandler?()
            }
        }
        outputListener = listener
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &address, DispatchQueue.main, listener)
    }

    private func removeOutputListener() {
        guard let listener = outputListener else { return }
        var address = Self.defaultOutputAddress()
        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &address, DispatchQueue.main, listener)
        outputListener = nil
    }
}
