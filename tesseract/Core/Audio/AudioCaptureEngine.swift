//
//  AudioCaptureEngine.swift
//  tesseract
//

import Foundation
import Observation
import AVFoundation
import Accelerate
import Combine

// Thread-safe sample storage - nonisolated for real-time audio thread access
nonisolated final class SampleBuffer: @unchecked Sendable {
    private var samples: [Float] = []
    private let lock = NSLock()

    func append(_ newSamples: [Float]) {
        lock.lock()
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }

    /// Append straight from the tap's channel pointer — no intermediate `[Float]`
    /// allocation on the real-time audio thread (~47 buffers/s at 48 kHz).
    func append(_ newSamples: UnsafeBufferPointer<Float>) {
        lock.lock()
        samples.append(contentsOf: newSamples)
        lock.unlock()
    }

    func getAndClear() -> [Float] {
        lock.lock()
        let result = samples
        samples.removeAll()
        lock.unlock()
        return result
    }

    func reserveCapacity(_ capacity: Int) {
        lock.lock()
        samples.reserveCapacity(capacity)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }
}

// Thread-safe audio level storage for real-time callback - nonisolated for audio thread access
nonisolated final class AudioLevelRelay: @unchecked Sendable {
    private var _level: Float = 0
    private let lock = NSLock()

    var level: Float {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _level
        }
        set {
            lock.lock()
            _level = newValue
            lock.unlock()
        }
    }
}

@MainActor
protocol AudioCapturing: AnyObject {
    var isCapturing: Bool { get }
    func startCapture() throws
    func stopCapture() -> AudioData?
}

@MainActor
@Observable
final class AudioCaptureEngine: AudioCapturing {
    private enum Defaults {
        static let defaultInputSampleRate: Double = 48_000
        static let bufferSize: AVAudioFrameCount = 1024
        static let meterInterval: TimeInterval = 0.05
        static let reserveSeconds: Int = 60
        /// A capture this long with zero tap buffers is a wedged input, not a
        /// quick tap — one 1024-frame buffer arrives within ~25 ms even with
        /// Voice Processing ramp-up. Matches the session's minimum recording
        /// duration, below which the capture is discarded as "too short" anyway.
        static let emptyCaptureGrace: TimeInterval = 0.5
        /// How long the kept engine stays *armed* (Voice Processing enabled)
        /// after a capture ends. Within the grace, a re-record skips the VPIO
        /// arm cost entirely; when it lapses, VP is disarmed to fully lift the
        /// system-audio duck — an armed VPIO ducks for as long as it exists,
        /// stopped or not, and no ducking level reaches zero.
        static let voiceProcessingDisarmGrace: TimeInterval = 10
        /// Arm/disarm and start/stop reconfigure the engine's own graph and
        /// fire `AVAudioEngineConfigurationChange` for our own doing; a
        /// notification landing within this window of an intentional
        /// reconfiguration is ignored instead of marking the engine dirty.
        static let selfInflictedConfigChangeWindow: TimeInterval = 1.0
    }

    /// The Voice Processing IO unit ducks all other system audio for as long
    /// as it is armed — engine running *or stopped* — and `.min` only softens
    /// that duck (measured elsewhere; it cannot reach zero). So the duck is
    /// scoped two ways: while armed, `.min` outside a real dictation capture
    /// and the standard level during one; and the unit itself is disarmed
    /// after ``Defaults/voiceProcessingDisarmGrace`` so idle leaves other
    /// audio completely untouched.
    private enum Ducking {
        static let idle = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .min)
        static let recording = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .default)
    }

    private(set) var isCapturing = false
    private(set) var audioLevel: Float = 0

    /// Kept alive across captures — engine create/destroy cycles are the
    /// pattern that wedges CoreAudio input. Voice Processing, however, is
    /// **armed only around captures**: an armed VPIO ducks all other system
    /// audio for its whole lifetime, so the engine idles disarmed (plain
    /// input node, zero duck) and ``startCapture()`` arms it in place — a
    /// stopped engine accepts `setVoiceProcessingEnabled` without a rebuild.
    /// The arm cost is paid once per burst of dictations, not per press:
    /// ``scheduleVoiceProcessingDisarm()`` keeps it armed for a short grace.
    private var audioEngine: AVAudioEngine?
    /// Whether Voice Processing is currently enabled on the kept engine —
    /// requested AND accepted by the platform. Tags the `RawCapture`.
    private var voiceProcessingArmed = false
    /// Lifts the duck once the post-capture grace lapses; cancelled by the
    /// next capture (which reuses the still-armed engine at no cost).
    private var disarmTask: Task<Void, Never>?
    /// When we last reconfigured the engine ourselves (build, arm/disarm,
    /// start, stop) — used to tell our own `AVAudioEngineConfigurationChange`
    /// echoes apart from real device/format changes.
    private var lastIntentionalReconfigure: Date = .distantPast
    /// Raised by `AVAudioEngineConfigurationChange` (default input device or
    /// format changed): the kept engine is not trusted, the next capture rebuilds.
    private var engineNeedsRebuild = false
    private var configChangeCancellable: AnyCancellable?

    private var inputTapInstalled = false
    private let sampleBuffer = SampleBuffer()
    private var captureStartTime: Date?
    private let levelRelay = AudioLevelRelay()
    private var levelUpdateTimer: Timer?

    /// The current capture records levels only (the settings meter): the tap
    /// appends no samples, and `stopCapture()` discards the empty recording —
    /// a minutes-long meter session costs no memory and its stop is instant.
    private var meteringOnly = false

    private var inputSampleRate: Double = Defaults.defaultInputSampleRate
    private let bufferSize: AVAudioFrameCount = Defaults.bufferSize

    /// Whether **Voice Processing** (PRD #175) should be requested at capture
    /// start. Queried per capture, so a settings flip applies to the next
    /// push-to-talk without restart.
    private let isVoiceProcessingEnabled: () -> Bool

    init(isVoiceProcessingEnabled: @escaping () -> Bool = { false }) {
        self.isVoiceProcessingEnabled = isVoiceProcessingEnabled
    }

    /// Builds the engine ahead of the first press. Deliberately built *plain*
    /// (no Voice Processing): an armed VPIO ducks all other system audio even
    /// while the engine merely sits stopped, so arming waits for an actual
    /// capture. A no-op while capturing and without microphone permission —
    /// prewarming must never surface a permission prompt on its own.
    func prewarm() {
        guard !isCapturing else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }

        if audioEngine == nil || engineNeedsRebuild {
            rebuildEngine(voiceProcessing: false)
        }
    }

    func startCapture() throws {
        guard !isCapturing else { return }

        // Check microphone permission first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        guard authStatus == .authorized else {
            throw DictationError.microphonePermissionDenied
        }

        // The capture reuses the still-armed engine; the pending disarm (if
        // any) must not fire mid-recording.
        disarmTask?.cancel()
        disarmTask = nil

        let wantsVoiceProcessing = isVoiceProcessingEnabled()
        if audioEngine == nil || engineNeedsRebuild {
            rebuildEngine(voiceProcessing: wantsVoiceProcessing)
            try beginCapture()
        } else {
            do {
                try reconcileVoiceProcessing(wantsVoiceProcessing)
                try beginCapture()
            } catch {
                // A kept-alive engine can go stale in ways no configuration-change
                // notification reported — rebuild once and retry before failing
                // the press.
                Log.audio.error(
                    "Capture start failed on the kept engine, rebuilding: \(error.localizedDescription)"
                )
                rebuildEngine(voiceProcessing: wantsVoiceProcessing)
                try beginCapture()
            }
        }
    }

    /// Arms or disarms Voice Processing in place on the kept engine — legal
    /// only while the engine is stopped, which is always the case between
    /// captures. This is the once-per-burst cost the disarm grace amortizes.
    private func reconcileVoiceProcessing(_ wanted: Bool) throws {
        guard let audioEngine, voiceProcessingArmed != wanted else { return }

        let clock = Date()
        lastIntentionalReconfigure = clock
        try audioEngine.inputNode.setVoiceProcessingEnabled(wanted)
        voiceProcessingArmed = wanted
        if wanted {
            audioEngine.inputNode.voiceProcessingOtherAudioDuckingConfiguration = Ducking.idle
        }
        Log.audio.info(
            """
            Voice processing \(wanted ? "armed" : "disarmed") in \
            \(String(format: "%.0f", Date().timeIntervalSince(clock) * 1000)) ms
            """)
    }

    /// Starts a metering-only capture for the settings level meter. Shares the
    /// microphone-busy semantics of `startCapture` (a running dictation wins).
    func startLevelMetering() throws {
        guard !isCapturing else { return }
        meteringOnly = true
        do {
            try startCapture()
        } catch {
            meteringOnly = false
            throw error
        }
    }

    /// Builds a fresh engine configured for `voiceProcessing`, replacing any kept
    /// one. Voice Processing must be requested before the engine starts; a
    /// platform refusal falls back to raw capture — dictation must never be
    /// blocked by it (PRD #175).
    private func rebuildEngine(voiceProcessing: Bool) {
        tearDownEngine()

        let buildStart = Date()
        lastIntentionalReconfigure = buildStart
        let engine = AVAudioEngine()
        voiceProcessingArmed = false
        if voiceProcessing {
            do {
                try engine.inputNode.setVoiceProcessingEnabled(true)
                engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration = Ducking.idle
                voiceProcessingArmed = true
            } catch {
                Log.audio.error(
                    "Voice processing unavailable, capturing raw: \(error.localizedDescription)")
            }
        }
        Log.audio.info(
            """
            Capture engine built in \
            \(String(format: "%.0f", Date().timeIntervalSince(buildStart) * 1000)) ms \
            (voice processing: \(voiceProcessingArmed))
            """)

        audioEngine = engine
        engineNeedsRebuild = false

        configChangeCancellable = NotificationCenter.default
            .publisher(for: .AVAudioEngineConfigurationChange, object: engine)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self else { return }
                // Our own arm/disarm and start/stop reconfigure the graph and
                // echo this notification; only an outside change (device swap,
                // format change) marks the kept engine dirty.
                guard
                    Date().timeIntervalSince(self.lastIntentionalReconfigure)
                        >= Defaults.selfInflictedConfigChangeWindow
                else { return }
                self.engineNeedsRebuild = true
            }
    }

    private func tearDownEngine() {
        disarmTask?.cancel()
        disarmTask = nil
        configChangeCancellable = nil
        tearDownAudioEngine(audioEngine)
        audioEngine = nil
        voiceProcessingArmed = false
    }

    /// Starts one capture on the current engine: tap, level timer, engine start.
    /// Fails leaving the engine allocated but idle — the caller decides whether
    /// to rebuild and retry.
    private func beginCapture() throws {
        guard let audioEngine else {
            throw DictationError.audioCaptureFailed("Failed to create audio engine")
        }
        let inputNode = audioEngine.inputNode

        // Read the format only after voice processing may have changed it. A
        // kept engine whose device went away reports a zero format — treat it
        // as a start failure so the rebuild-and-retry path runs.
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 else {
            throw DictationError.audioCaptureFailed("Input device reports no format")
        }
        inputSampleRate = inputFormat.sampleRate

        // Create format for our tap
        guard
            let recordingFormat = AudioConverter.monoFloat32Format(
                sampleRate: inputFormat.sampleRate)
        else {
            throw DictationError.audioCaptureFailed("Failed to create recording format")
        }

        sampleBuffer.clear()
        if !meteringOnly {
            // Reserve at the rate the tap actually delivers — reserving at the
            // 16 kHz target rate covered only a third of the capture and let
            // the array re-grow (multi-MB realloc) on the audio thread.
            sampleBuffer.reserveCapacity(Int(inputSampleRate) * Defaults.reserveSeconds)
        }

        captureStartTime = Date()

        // Install tap with nonisolated handler to avoid MainActor inheritance
        let buffer = meteringOnly ? nil : sampleBuffer
        let relay = levelRelay
        inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: recordingFormat,
            block: Self.makeAudioTapHandler(buffer: buffer, relay: relay)
        )
        inputTapInstalled = true

        // Poll the level relay on the main thread. `.common` mode so the meter
        // doesn't freeze during menu tracking; the epsilon skip keeps silence
        // from invalidating observers 20×/s for identical values.
        let timer = Timer(timeInterval: Defaults.meterInterval, repeats: true) {
            [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let level = self.levelRelay.level
                if abs(level - self.audioLevel) > 0.001 {
                    self.audioLevel = level
                }
            }
        }
        timer.tolerance = 0.01
        RunLoop.main.add(timer, forMode: .common)
        levelUpdateTimer = timer

        // Duck other system audio only for a real dictation capture — the
        // settings meter keeps the softest level. Set before start so the
        // level is baked into this run of the IO unit.
        if voiceProcessingArmed {
            inputNode.voiceProcessingOtherAudioDuckingConfiguration =
                meteringOnly ? Ducking.idle : Ducking.recording
        }

        let startClock = Date()
        lastIntentionalReconfigure = startClock
        do {
            audioEngine.prepare()
            try audioEngine.start()
            isCapturing = true
            Log.audio.info(
                """
                Capture started in \
                \(String(format: "%.0f", Date().timeIntervalSince(startClock) * 1000)) ms
                """)
        } catch {
            levelUpdateTimer?.invalidate()
            levelUpdateTimer = nil
            inputNode.removeTap(onBus: 0)
            inputTapInstalled = false
            captureStartTime = nil
            throw DictationError.audioCaptureFailed(error.localizedDescription)
        }
    }

    func stopCapture() -> AudioData? {
        guard isCapturing else { return nil }

        isCapturing = false
        levelUpdateTimer?.invalidate()
        levelUpdateTimer = nil
        audioLevel = 0
        levelRelay.level = 0

        // Stop IO but keep the engine (and its Voice Processing configuration)
        // for the next press. Same order as teardown: stop before removing the
        // tap, so AudioOutputUnitStop never races a nil tap callback.
        let stopClock = Date()
        lastIntentionalReconfigure = stopClock
        if let audioEngine {
            audioEngine.stop()
            if inputTapInstalled {
                audioEngine.inputNode.removeTap(onBus: 0)
                inputTapInstalled = false
            }
            if voiceProcessingArmed {
                // Soften the duck the moment the capture ends, and schedule
                // the disarm that lifts it entirely once the re-record grace
                // lapses.
                audioEngine.inputNode.voiceProcessingOtherAudioDuckingConfiguration = Ducking.idle
                scheduleVoiceProcessingDisarm()
            }
        }
        Log.audio.info(
            """
            Capture stopped in \
            \(String(format: "%.0f", Date().timeIntervalSince(stopClock) * 1000)) ms
            """)

        let duration = captureStartTime.map { Date().timeIntervalSince($0) } ?? 0
        captureStartTime = nil

        if meteringOnly {
            meteringOnly = false
            return nil
        }

        let samples = sampleBuffer.getAndClear()

        if samples.isEmpty && duration >= Defaults.emptyCaptureGrace {
            // The engine ran long enough that the tap must have fired, yet it
            // delivered nothing — a wedged input. Discard the engine (the next
            // press builds fresh) and report "no audio", which is the truth,
            // rather than letting an empty transcription claim "no speech
            // detected". Below the grace, an empty capture is just a tap that
            // beat the first buffer; it falls through to the session's
            // minimum-duration guard ("too short").
            Log.audio.error(
                """
                Capture delivered no samples over \
                \(String(format: "%.1f", duration)) s — discarding engine
                """)
            tearDownEngine()
            return nil
        }

        // The recognizer resamples to 16 kHz on its own actor — returning the
        // native-rate samples keeps MB-scale conversion off the key-release
        // path, which runs on the main thread under the app's system-wide
        // event tap. `samples` and `raw` share one copy-on-write storage.
        return AudioData(
            samples: samples,
            sampleRate: inputSampleRate,
            duration: duration,
            raw: RawCapture(
                samples: samples,
                sampleRate: inputSampleRate,
                voiceProcessed: voiceProcessingArmed
            )
        )
    }

    // MARK: - Private

    /// Starts the post-capture grace after which Voice Processing is disarmed.
    /// Within the grace a new capture reuses the armed engine at no cost; after
    /// it, the disarm fully lifts the VPIO's system-audio duck (deallocation
    /// or disarm are the only things that do — no ducking *level* reaches zero).
    private func scheduleVoiceProcessingDisarm() {
        disarmTask?.cancel()
        disarmTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Defaults.voiceProcessingDisarmGrace))
            guard !Task.isCancelled, let self, !self.isCapturing else { return }
            do {
                try self.reconcileVoiceProcessing(false)
            } catch {
                // A lingering duck is the one outcome this exists to prevent —
                // discard the engine instead (deallocation provably lifts it);
                // the next press rebuilds.
                Log.audio.error(
                    "Voice processing disarm failed, discarding engine: \(error.localizedDescription)"
                )
                self.tearDownEngine()
            }
        }
    }

    private func tearDownAudioEngine(_ engine: AVAudioEngine?) {
        guard let engine else {
            inputTapInstalled = false
            return
        }

        // Keep the tap callback alive while CoreAudio stops its IO thread. Removing
        // the tap first can leave AudioOutputUnitStop racing a nil tap callback.
        engine.stop()

        if inputTapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            inputTapInstalled = false
        }

        engine.reset()
    }

    /// Creates an audio tap handler that runs on the real-time audio thread.
    /// This is nonisolated to prevent MainActor isolation inheritance.
    nonisolated private static func makeAudioTapHandler(
        buffer: SampleBuffer?,
        relay: AudioLevelRelay
    ) -> AVAudioNodeTapBlock {
        return { audioBuffer, _ in
            guard let channelData = audioBuffer.floatChannelData?[0] else { return }
            let frameCount = Int(audioBuffer.frameLength)

            // Calculate RMS for level metering
            var rms: Float = 0
            vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(frameCount))

            // Convert to dB scale (with floor at -60dB)
            let db = 20 * log10(max(rms, 0.001))
            let normalizedLevel = (db + 60) / 60  // Normalize -60dB to 0dB -> 0 to 1

            // Copy samples to the thread-safe buffer (nil for a metering-only
            // capture — the settings meter wants the level, not the audio)
            buffer?.append(UnsafeBufferPointer(start: channelData, count: frameCount))

            // Store level in thread-safe relay (polled by timer on main thread)
            relay.level = max(0, min(1, normalizedLevel))
        }
    }

}
