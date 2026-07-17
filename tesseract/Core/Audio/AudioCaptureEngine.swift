//
//  AudioCaptureEngine.swift
//  tesseract
//

import Foundation
import Observation
import AVFoundation
import Accelerate
import Combine

// Thread-safe sample storage - nonisolated for real-time audio thread access.
//
// Chunked (audit #285 item 8): `reserveCapacity` fixes the chunk size, and a
// capture that outgrows it seals the full chunk and starts a fresh one —
// growth never reallocates-and-copies the whole capture on the real-time
// audio thread. The old single-array reserve covered 60 s while recordings
// run to 1800 s, so every long capture paid multi-MB memmoves at tap cadence
// past the first minute; now the worst case per append is one zero-copy
// chunk allocation. The one coalescing copy happens in `getAndClear()`, off
// the tap (the tap is removed before stop reads the capture) — and the
// common short dictation never fills a second chunk, keeping its
// `getAndClear()` an O(1) buffer steal.
nonisolated final class SampleBuffer: @unchecked Sendable {
    /// Samples per chunk; 0 = unchunked (plain growth) until `reserveCapacity`.
    private var chunkCapacity = 0
    private var sealedChunks: [[Float]] = []
    private var currentChunk: [Float] = []
    private let lock = NSLock()

    func append(_ newSamples: [Float]) {
        newSamples.withUnsafeBufferPointer { append($0) }
    }

    /// Append straight from the tap's channel pointer — no intermediate `[Float]`
    /// allocation on the real-time audio thread (~47 buffers/s at 48 kHz).
    func append(_ newSamples: UnsafeBufferPointer<Float>) {
        lock.lock()
        if chunkCapacity > 0, currentChunk.count + newSamples.count > chunkCapacity {
            sealedChunks.append(currentChunk)
            currentChunk = []
            currentChunk.reserveCapacity(chunkCapacity)
        }
        currentChunk.append(contentsOf: newSamples)
        lock.unlock()
    }

    func getAndClear() -> [Float] {
        lock.lock()
        defer {
            sealedChunks = []
            currentChunk = []
            lock.unlock()
        }
        if sealedChunks.isEmpty {
            return currentChunk
        }
        var result: [Float] = []
        result.reserveCapacity(
            sealedChunks.reduce(0) { $0 + $1.count } + currentChunk.count)
        for chunk in sealedChunks {
            result.append(contentsOf: chunk)
        }
        result.append(contentsOf: currentChunk)
        return result
    }

    func reserveCapacity(_ capacity: Int) {
        lock.lock()
        chunkCapacity = capacity
        currentChunk.reserveCapacity(capacity)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        sealedChunks = []
        currentChunk = []
        lock.unlock()
    }

    /// A non-destructive copy of everything captured so far — the **Live
    /// Partial** lane's mid-capture read (ticket #291). The tap keeps
    /// appending; the coalescing copy happens on the caller's thread, never
    /// the real-time audio thread.
    func snapshot() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        if sealedChunks.isEmpty {
            return currentChunk
        }
        var result: [Float] = []
        result.reserveCapacity(
            sealedChunks.reduce(0) { $0 + $1.count } + currentChunk.count)
        for chunk in sealedChunks {
            result.append(contentsOf: chunk)
        }
        result.append(contentsOf: currentChunk)
        return result
    }
}

/// The voice hold's **Capture Gate** (ADR-0041): under a hold the tap runs
/// continuously on the held engine, and a capture "start"/"stop" is just this
/// flag — buffer discipline, never tap install/remove on a running VP engine
/// (the 2026-07-17 crash class). Written on the main actor, read on the
/// real-time audio thread. A torn read lands a buffer in the adjacent take at
/// worst — bounded by the tap's ~100 ms delivery grain and covered by the
/// session's deaf windows — so a plain Bool suffices; no lock on the RT
/// thread. Outside a hold the gate simply tracks `isCapturing`, keeping one
/// tap-handler code path for both lifecycles.
nonisolated final class CaptureGate: @unchecked Sendable {
    var isActive = false
}

@MainActor
protocol AudioCapturing: AnyObject {
    var isCapturing: Bool { get }
    func startCapture() throws
    func stopCapture() -> AudioData?
    /// A mid-capture snapshot of the audio so far — the **Live Partial**
    /// lane's read (ticket #291). `nil` when not capturing or when the
    /// implementation has nothing to offer (the default) — the partial track
    /// degrades to silence, never an error.
    func captureSnapshot() -> AudioData?
}

extension AudioCapturing {
    func captureSnapshot() -> AudioData? { nil }
}

@MainActor
@Observable
final class AudioCaptureEngine: AudioCapturing {
    private enum Defaults {
        static let defaultInputSampleRate: Double = 48_000
        static let bufferSize: AVAudioFrameCount = 1024
        /// The `SampleBuffer` chunk size in seconds — one chunk covers the
        /// common dictation entirely; longer captures grow chunk-at-a-time.
        static let reserveSeconds: Int = 60
    }

    private(set) var isCapturing = false

    /// Meter frames (level + spectrum) at tap cadence, straight from the
    /// real-time tap — the `DictationFeed` pumps this onto the main actor.
    /// Replaces the retired 20 Hz Timer poll + `@Observable audioLevel` ferry
    /// (audit #285 item 2). Buffered-newest: a slow consumer sees the latest
    /// frame, never a backlog.
    var meters: AsyncStream<MeterFrame> { meterStream.stream }
    private let meterStream = AsyncStream.makeStream(
        of: MeterFrame.self, bufferingPolicy: .bufferingNewest(1))

    /// Kept alive across captures — engine create/destroy cycles are the
    /// pattern that wedges CoreAudio input. Voice Processing is the standard
    /// mode (PRD #188): under the always-armed lifecycle the engine is built
    /// *armed* at prewarm and never disarmed — the arm cost (170–600 ms
    /// measured) is paid at launch, never on a press, and the idle duck is
    /// reversed through the **System Audio Duck** port (ADR-0025). Under the
    /// disarm-after-grace fallback (un-duck unavailable) the engine idles
    /// plain and arms per burst, exactly the pre-#188 behavior.
    private var audioEngine: AVAudioEngine?
    /// Whether Voice Processing is currently enabled on the kept engine —
    /// requested AND accepted by the platform. Tags the `RawCapture`.
    private var voiceProcessingArmed = false
    /// Fallback lifecycle only: lifts the duck once the post-capture grace
    /// lapses; cancelled by the next capture.
    private var disarmTask: Task<Void, Never>?
    /// The pending idle rebuild after an external configuration change or a
    /// wedge teardown — re-arms in the background so the next press stays at
    /// engine-start cost. Cancelled by a press, which rebuilds inline anyway.
    private var idleRebuildTask: Task<Void, Never>?
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

    // MARK: Voice hold (Dual-Path Playback, ADR-0041)

    /// While a voice session runs the engine is *held*: it keeps running
    /// between captures (capture start/stop degrade to gate flips) and hosts
    /// the session's TTS player nodes, so VPIO's echo canceller hears the
    /// reply as its own render-stream reference instead of reconstructed
    /// device loopback. Set synchronously at `beginVoiceHold` — session
    /// semantics; the wiring commits asynchronously (measured ~860–900 ms for
    /// tap + render wire + start with the render side, `research/voice-hold-lab`
    /// E6 — far too long for the main actor, so it runs detached, E7).
    private(set) var voiceHoldActive = false
    /// The hold's tap and render side are installed and the held engine is
    /// (or should be) running. False while the wiring is pending/queued/in
    /// flight, after a teardown, and after `endVoiceHold`.
    private(set) var voiceHoldWired = false
    /// Whether the held engine's render side is verified and playback nodes
    /// can attach. Requires armed + render-verified + running — an unarmed
    /// engine hosting playback buys nothing acoustically (the reply falls
    /// back to the dedicated engine).
    private(set) var voicePlaybackHosted = false
    /// A detached wiring owns the engine right now. Every MainActor path that
    /// would touch the engine (press, prewarm, hold end) defers to the
    /// wiring's commit hop — racing it is the crash class the redo exists to
    /// kill. Cleared only by the *current* generation's commit.
    private var holdWiringInProgress = false
    /// The in-flight/last wiring task. Never cancelled mid-flight to
    /// reschedule — a newer request folds into `holdWireQueued` instead, so
    /// two wirings never touch one engine at once.
    private var holdWireTask: Task<HoldWireOutcome, Never>?
    /// A wiring request folded in while another wiring ran — the commit hop
    /// discards its stale outcome and runs this next. Value = rebuildFirst.
    private var holdWireQueued: Bool?
    /// Staleness guard for wiring commits — bumped on every (re)schedule and
    /// on `endVoiceHold`.
    private var holdGeneration = 0
    /// Coalesced rebuild-under-hold (device change while held) — the
    /// config-change sink's hold-aware counterpart to `idleRebuildTask`.
    private var holdRebuildTask: Task<Void, Never>?
    /// Under a hold the tap runs continuously; a capture is this flag.
    private let captureGate = CaptureGate()
    /// The hold's render-side connection (mainMixer→output) is physically
    /// present on the engine. Tracked separately from `voiceHoldWired` so a
    /// hold that ends mid-take still gets its render side unwired — at the
    /// in-progress capture's own stop, on the then-stopped engine.
    private var holdRenderWired = false
    /// Fired when the engine is torn down or rebuilt underneath attached
    /// playback nodes — the playback adapter treats it as end-of-utterance.
    var onVoicePlaybackInvalidated: (@MainActor () -> Void)?
    private var attachedPlaybackNodes: [AVAudioPlayerNode] = []

    /// The current capture records levels only (the settings meter): the tap
    /// appends no samples, and `stopCapture()` discards the empty recording —
    /// a minutes-long meter session costs no memory and its stop is instant.
    private var meteringOnly = false

    private var inputSampleRate: Double = Defaults.defaultInputSampleRate
    private let bufferSize: AVAudioFrameCount = Defaults.bufferSize

    /// The **System Audio Duck** seam (PRD #188): the policy decides when each
    /// duck treatment applies; the controller performs it (VPIO ducking level
    /// through `duckingConfigurator`, `AudioDeviceDuck` un-duck, output-device
    /// watcher).
    private let duckPolicy: VoiceProcessingDuckPolicy

    /// The **Capture Engine Lifecycle** policy: every keep-vs-rebuild
    /// decision about the kept engine, pinned by its own decision-table
    /// tests. This engine is the performing adapter.
    private let lifecycle: CaptureEngineLifecycle

    init(duckController: SystemAudioDuckController = SystemAudioDuckController()) {
        self.duckPolicy = VoiceProcessingDuckPolicy(port: duckController)
        self.lifecycle = CaptureEngineLifecycle(voiceProcessing: duckPolicy.lifecycle)
        duckController.duckingConfigurator = { [weak self] configuration in
            guard let self, let engine = self.audioEngine, self.voiceProcessingArmed else { return }
            engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration = configuration
        }
    }

    /// Builds the engine ahead of the first press — under the always-armed
    /// lifecycle, *armed*: the VPIO arm is the expensive step (170–600 ms
    /// measured) and this is where the user cannot feel it. Idle stays
    /// silent-cost because the idle treatment un-ducks other audio to full
    /// volume (ADR-0025). A no-op while capturing and without microphone
    /// permission — prewarming must never surface a permission prompt.
    func prewarm() {
        guard !isCapturing else { return }
        // A hold (or a wiring in flight) owns the engine — rebuilding here
        // would race the detached wiring or kill the session's render side.
        guard !voiceHoldActive, !holdWiringInProgress else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }

        if audioEngine == nil || engineNeedsRebuild {
            rebuildEngine(voiceProcessing: lifecycle.prewarmBuildsArmed)
        }
    }

    func startCapture() throws {
        guard !isCapturing else { return }

        // Check microphone permission first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        guard authStatus == .authorized else {
            throw DictationError.microphonePermissionDenied
        }

        // The press wins over any pending background work: the fallback's
        // disarm must not fire mid-recording, and an in-flight idle rebuild
        // is superseded by the inline rebuild below if one is still needed.
        disarmTask?.cancel()
        disarmTask = nil
        idleRebuildTask?.cancel()
        idleRebuildTask = nil

        // Voice hold: the held engine keeps running and the tap stays — a
        // capture start is buffer discipline (ADR-0041), never engine work.
        if voiceHoldActive {
            guard voiceHoldWired, !holdWiringInProgress,
                let audioEngine, audioEngine.isRunning
            else {
                // The wiring (or a rebuild) hasn't committed — fail fast so
                // the session's 1 s backoff retries into a wired hold instead
                // of attempting engine work that would race the wiring.
                if voiceHoldWired, !holdWiringInProgress,
                    let audioEngine, !audioEngine.isRunning
                {
                    // The held engine silently stopped (no config change
                    // reached us): mark unwired and re-wire off this press.
                    voiceHoldWired = false
                    voicePlaybackHosted = false
                    scheduleHoldWiring(rebuildFirst: false)
                }
                throw DictationError.audioCaptureFailed("Voice hold is still wiring")
            }
            if meteringOnly {
                // The hold's tap meters continuously — a metering capture is
                // already running; only the mic-busy semantics apply.
                isCapturing = true
                return
            }
            sampleBuffer.clear()
            sampleBuffer.reserveCapacity(Int(inputSampleRate) * Defaults.reserveSeconds)
            captureStartTime = Date()
            duckPolicy.captureDidStart(meteringOnly: false)
            captureGate.isActive = true
            isCapturing = true
            return
        }

        // A hold ended mid-wiring and its commit hop is discarding the
        // outcome right now — engine work here would race that teardown.
        guard !holdWiringInProgress else {
            throw DictationError.audioCaptureFailed("Voice hold wiring in progress")
        }

        switch lifecycle.pressAction(
            engineExists: audioEngine != nil, needsRebuild: engineNeedsRebuild)
        {
        case .rebuildArmed:
            rebuildEngine(voiceProcessing: true)
            try beginCapture()
        case .reuse(let reconcileArm):
            do {
                if reconcileArm {
                    try reconcileVoiceProcessing(true)
                }
                try beginCapture()
            } catch {
                // A kept-alive engine can go stale in ways no configuration-change
                // notification reported — rebuild once and retry before failing
                // the press.
                Log.audio.error(
                    "Capture start failed on the kept engine, rebuilding: \(error.localizedDescription)"
                )
                rebuildEngine(voiceProcessing: true)
                try beginCapture()
            }
        }
    }

    /// Fallback lifecycle only: arms or disarms Voice Processing in place on
    /// the kept engine — legal only while the engine is stopped, which is
    /// always the case between captures. This is the once-per-burst cost the
    /// disarm grace amortizes; the always-armed lifecycle never pays it.
    private func reconcileVoiceProcessing(_ wanted: Bool) throws {
        guard let audioEngine, voiceProcessingArmed != wanted else { return }

        let clock = Date()
        lastIntentionalReconfigure = clock
        try audioEngine.inputNode.setVoiceProcessingEnabled(wanted)
        voiceProcessingArmed = wanted
        if wanted {
            duckPolicy.engineDidArm()
        } else {
            duckPolicy.engineDidDisarm()
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

    /// Builds a fresh engine configured for `voiceProcessing`. Nonisolated so
    /// the voice hold's detached wiring can build off the main actor — the
    /// arm cost (170–600 ms measured) is the expensive step either way. A
    /// platform refusal falls back to raw capture — dictation must never be
    /// blocked by it (PRD #175).
    nonisolated private static func buildEngine(
        voiceProcessing: Bool
    ) -> (engine: AVAudioEngine, armed: Bool) {
        let buildStart = Date()
        let engine = AVAudioEngine()
        var armed = false
        if voiceProcessing {
            do {
                try engine.inputNode.setVoiceProcessingEnabled(true)
                armed = true
            } catch {
                Log.audio.error(
                    "Voice processing unavailable, capturing raw: \(error.localizedDescription)")
            }
        }
        Log.audio.info(
            """
            Capture engine built in \
            \(String(format: "%.0f", Date().timeIntervalSince(buildStart) * 1000)) ms \
            (voice processing: \(armed))
            """)
        return (engine, armed)
    }

    /// Adopts a built engine as the kept one: state, duck bookkeeping, and
    /// the configuration-change sink. Shared by the press/idle rebuild path
    /// and the voice hold's detached-wiring commit.
    private func adoptBuiltEngine(_ built: (engine: AVAudioEngine, armed: Bool)) {
        lastIntentionalReconfigure = Date()
        audioEngine = built.engine
        voiceProcessingArmed = built.armed
        engineNeedsRebuild = false
        if built.armed {
            duckPolicy.engineDidArm()
        }

        configChangeCancellable = NotificationCenter.default
            .publisher(for: .AVAudioEngineConfigurationChange, object: built.engine)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self else { return }
                // Our own arm/disarm and start/stop reconfigure the graph and
                // echo this notification; only an outside change (device swap,
                // format change) marks the kept engine dirty.
                guard
                    self.lifecycle.isExternalConfigChange(
                        sinceLastIntentionalReconfigure:
                            Date().timeIntervalSince(self.lastIntentionalReconfigure))
                else { return }
                self.engineNeedsRebuild = true
                if self.voiceHoldActive {
                    // Under a hold a rebuild must re-wire, not just re-arm —
                    // the hold's tap and render side die with the old engine.
                    self.scheduleHoldRebuild()
                } else {
                    // Re-arm while idle so the press after a device change
                    // stays at engine-start cost instead of paying the
                    // rebuild + arm.
                    self.scheduleIdleRebuild()
                }
            }
    }

    /// Replaces the kept engine with a fresh one configured for
    /// `voiceProcessing`. Voice Processing must be requested before the
    /// engine starts.
    private func rebuildEngine(voiceProcessing: Bool) {
        tearDownEngine()
        adoptBuiltEngine(Self.buildEngine(voiceProcessing: voiceProcessing))
    }

    private func tearDownEngine() {
        disarmTask?.cancel()
        disarmTask = nil
        configChangeCancellable = nil
        // The engine underneath attached playback nodes is going away — the
        // playback adapter hears it as end-of-utterance, not as buffer
        // callbacks that will never fire.
        invalidateVoicePlayback()
        voiceHoldWired = false
        holdRenderWired = false
        captureGate.isActive = false
        tearDownAudioEngine(audioEngine)
        audioEngine = nil
        voiceProcessingArmed = false
        duckPolicy.engineDidDisarm()
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
            // 16 kHz target rate covered only a third of the capture. The
            // reserve is the buffer's *chunk* size: a capture that outlives it
            // (max duration runs to 1800 s) seals the chunk and starts a new
            // one, so growth never copies the capture on the audio thread.
            sampleBuffer.reserveCapacity(Int(inputSampleRate) * Defaults.reserveSeconds)
        }

        captureStartTime = Date()
        // The gate drives the handler's appends (hold or not — one code
        // path); open it before start so the first buffers land.
        captureGate.isActive = true

        // Install tap with nonisolated handler to avoid MainActor inheritance.
        // The meter tap computes level + spectrum on the audio thread and
        // yields straight into `meters` — no timer, no main-thread poll.
        let buffer = meteringOnly ? nil : sampleBuffer
        let meterTap = AudioMeterTap(
            sampleRate: recordingFormat.sampleRate,
            continuation: meterStream.continuation)
        inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: recordingFormat,
            block: Self.makeAudioTapHandler(gate: captureGate, buffer: buffer, meter: meterTap)
        )
        inputTapInstalled = true

        // Duck other system audio only for a real dictation capture — the
        // settings meter keeps the idle treatment. Set before start so the
        // level is baked into this run of the IO unit.
        duckPolicy.captureDidStart(meteringOnly: meteringOnly)

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
            captureGate.isActive = false
            inputNode.removeTap(onBus: 0)
            inputTapInstalled = false
            captureStartTime = nil
            duckPolicy.captureDidStop()
            meterStream.continuation.yield(.zero)
            throw DictationError.audioCaptureFailed(error.localizedDescription)
        }
    }

    func stopCapture() -> AudioData? {
        guard isCapturing else { return nil }

        isCapturing = false
        meterStream.continuation.yield(.zero)
        captureGate.isActive = false

        // Voice hold: the engine keeps running and the tap stays installed —
        // a capture stop is buffer discipline (ADR-0041), never engine
        // discipline.
        if voiceHoldWired {
            let duration = captureStartTime.map { Date().timeIntervalSince($0) } ?? 0
            captureStartTime = nil
            if meteringOnly {
                meteringOnly = false
                return nil
            }
            duckPolicy.captureDidStop()
            let samples = sampleBuffer.getAndClear()
            let wasVoiceProcessed = voiceProcessingArmed
            if samples.isEmpty,
                lifecycle.emptyCaptureVerdict(duration: duration) == .wedgedInput
            {
                // A wedged input under a hold: the engine is suspect — tear
                // it down (attached playback is invalidated) and re-wire the
                // hold on a fresh one.
                Log.audio.error(
                    """
                    Capture delivered no samples over \
                    \(String(format: "%.1f", duration)) s — discarding engine
                    """)
                tearDownEngine()
                scheduleHoldWiring(rebuildFirst: true)
                return nil
            }
            return AudioData(
                samples: samples,
                sampleRate: inputSampleRate,
                duration: duration,
                raw: RawCapture(
                    samples: samples,
                    sampleRate: inputSampleRate,
                    voiceProcessed: wasVoiceProcessed
                )
            )
        }

        // The non-hold stop — also the deferred hold wiring's runway: a
        // capture that was mid-take when the hold began (or when a rebuild
        // came due) lands here, and its stop is what frees the engine for
        // the hold's stopped-engine wiring.
        let wireHoldAfterStop = lifecycle.shouldWireHoldAfterCaptureStop(
            holdActive: voiceHoldActive, holdWired: voiceHoldWired)
        let wireHoldIfPending = {
            if wireHoldAfterStop {
                self.scheduleHoldWiring(
                    rebuildFirst: self.audioEngine == nil || self.engineNeedsRebuild)
            }
        }

        // Stop IO but keep the engine (and its Voice Processing arm) for the
        // next press. Same order as teardown: stop before removing the tap,
        // so AudioOutputUnitStop never races a nil tap callback.
        let stopClock = Date()
        lastIntentionalReconfigure = stopClock
        if let audioEngine {
            audioEngine.stop()
            if inputTapInstalled {
                audioEngine.inputNode.removeTap(onBus: 0)
                inputTapInstalled = false
            }
            // A hold that ended mid-take leaves its render side on this
            // engine — restore the pristine input-only dictation graph while
            // the engine is stopped.
            if holdRenderWired {
                audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
                holdRenderWired = false
            }
            // Back to the idle treatment (full volume) the moment the capture
            // ends; the fallback lifecycle additionally schedules the disarm
            // that releases its duck for good — never under a hold, whose
            // engine is armed for the session.
            duckPolicy.captureDidStop()
            if voiceProcessingArmed,
                lifecycle.shouldDisarmAfterCapture(holdActive: voiceHoldActive)
            {
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
            wireHoldIfPending()
            return nil
        }

        let samples = sampleBuffer.getAndClear()
        let wasVoiceProcessed = voiceProcessingArmed

        if samples.isEmpty,
            lifecycle.emptyCaptureVerdict(duration: duration) == .wedgedInput
        {
            // The idle rebuild re-arms in the background so the next press
            // starts fresh at engine-start cost — under a hold, the rebuild
            // re-wires the hold instead. A `tapBeatFirstBuffer` verdict falls
            // through to the session's minimum-duration guard ("too short").
            Log.audio.error(
                """
                Capture delivered no samples over \
                \(String(format: "%.1f", duration)) s — discarding engine
                """)
            tearDownEngine()
            if voiceHoldActive {
                scheduleHoldWiring(rebuildFirst: true)
            } else {
                scheduleIdleRebuild()
            }
            return nil
        }

        // The recognizer resamples to 16 kHz on its own actor — returning the
        // native-rate samples keeps MB-scale conversion off the key-release
        // path, which runs on the main thread under the app's system-wide
        // event tap. `samples` and `raw` share one copy-on-write storage.
        wireHoldIfPending()
        return AudioData(
            samples: samples,
            sampleRate: inputSampleRate,
            duration: duration,
            raw: RawCapture(
                samples: samples,
                sampleRate: inputSampleRate,
                voiceProcessed: wasVoiceProcessed
            )
        )
    }

    func captureSnapshot() -> AudioData? {
        guard isCapturing, !meteringOnly else { return nil }
        let samples = sampleBuffer.snapshot()
        guard !samples.isEmpty, inputSampleRate > 0 else { return nil }
        return AudioData(
            samples: samples,
            sampleRate: inputSampleRate,
            duration: Double(samples.count) / inputSampleRate,
            raw: nil
        )
    }

    // MARK: - Private

    /// Re-arms in the background after an external configuration change or a
    /// wedge teardown (always-armed lifecycle only). Coalesces the
    /// notification burst behind a short delay, defers to any capture in
    /// progress, and retries a flaky arm once — a refusal on rebuild would
    /// otherwise silently downgrade every following capture to raw.
    private func scheduleIdleRebuild() {
        guard lifecycle.rebuildsWhileIdle else { return }
        idleRebuildTask?.cancel()
        idleRebuildTask = Task { [weak self] in
            guard let lifecycle = self?.lifecycle else { return }
            try? await Task.sleep(for: lifecycle.idleRebuildDelay)
            guard !Task.isCancelled, let self, !self.isCapturing else { return }
            self.prewarm()
            if lifecycle.idleRebuildNeedsArmRetry(
                engineExists: self.audioEngine != nil, armed: self.voiceProcessingArmed)
            {
                try? await Task.sleep(for: lifecycle.armRetryDelay)
                guard !Task.isCancelled, !self.isCapturing else { return }
                self.engineNeedsRebuild = true
                self.prewarm()
            }
        }
    }

    /// Fallback lifecycle only: starts the post-capture grace after which
    /// Voice Processing is disarmed. Within the grace a new capture reuses the
    /// armed engine at no cost; after it, the disarm fully lifts the VPIO's
    /// system-audio duck — with no un-duck available, disarm or deallocation
    /// are the only things that do.
    private func scheduleVoiceProcessingDisarm() {
        disarmTask?.cancel()
        disarmTask = Task { [weak self] in
            guard let grace = self?.lifecycle.voiceProcessingDisarmGrace else { return }
            try? await Task.sleep(for: .seconds(grace))
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
        gate: CaptureGate,
        buffer: SampleBuffer?,
        meter: AudioMeterTap?
    ) -> AVAudioNodeTapBlock {
        return { audioBuffer, _ in
            guard let channelData = audioBuffer.floatChannelData?[0] else { return }
            let frameCount = Int(audioBuffer.frameLength)

            // Calculate RMS for level metering
            var rms: Float = 0
            vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(frameCount))

            // Convert to dB scale (with floor at -60dB)
            let db = 20 * log10(max(rms, 0.001))
            let normalizedLevel = max(0, min(1, (db + 60) / 60))

            // Copy samples to the thread-safe buffer while the gate is open
            // (nil for a metering-only capture — the settings meter wants the
            // level, not the audio)
            if gate.isActive {
                buffer?.append(UnsafeBufferPointer(start: channelData, count: frameCount))
            }

            // Level + spectrum straight into the meter stream.
            meter?.process(channelData, frameCount: frameCount, level: normalizedLevel)
        }
    }

    // MARK: - Voice hold (Dual-Path Playback, ADR-0041)

    /// Begins the voice-session hold: the engine will keep running until
    /// `endVoiceHold`, hosting the session's TTS player nodes so VPIO's echo
    /// canceller hears the reply as its own render-stream reference. The
    /// wiring is detached (measured ~860–900 ms — `research/voice-hold-lab`
    /// E6/E7) and commits asynchronously; captures fast-fail into the
    /// session's backoff until it lands, and playback attaches only once
    /// `voicePlaybackHosted` flips true. A no-op without microphone
    /// permission (the session's own capture start surfaces that error).
    func beginVoiceHold() {
        guard !voiceHoldActive else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }
        voiceHoldActive = true
        holdGeneration += 1
        // A pending disarm (fallback lifecycle) or idle rebuild must not fire
        // underneath the hold's wiring.
        disarmTask?.cancel()
        disarmTask = nil
        idleRebuildTask?.cancel()
        idleRebuildTask = nil
        switch lifecycle.holdBeginAction(
            engineExists: audioEngine != nil, needsRebuild: engineNeedsRebuild,
            isCapturing: isCapturing, engineArmed: voiceProcessingArmed)
        {
        case .deferToCaptureStop:
            // A capture is mid-take — its stop frees the engine and wires the
            // hold there (stopCapture's wireHoldAfterStop). Until then the
            // hold is capture-less and playback falls back.
            Log.audio.info("Voice hold began (wiring deferred to the capture's stop)")
        case .rebuildThenWire:
            scheduleHoldWiring(rebuildFirst: true)
        case .wireNow:
            scheduleHoldWiring(rebuildFirst: false)
        }
    }

    /// Ends the hold: playback nodes detach, then — all on the stopped
    /// engine — the tap comes off and the render side unwires. The engine
    /// returns to the armed-stopped idle the dictation lifecycle expects.
    func endVoiceHold() {
        guard voiceHoldActive else { return }
        voiceHoldActive = false
        holdGeneration += 1
        holdRebuildTask?.cancel()
        holdRebuildTask = nil
        holdWireTask?.cancel()
        holdWireTask = nil
        holdWireQueued = nil
        if !isCapturing {
            captureGate.isActive = false
        }
        if holdWiringInProgress {
            // The detached wiring's commit hop sees the bumped generation and
            // discards whatever it built (stop → remove tap → unwire, all on
            // the stopped engine). Touching the engine here would race that
            // work — the crash class the redo exists to kill.
            voiceHoldWired = false
            voicePlaybackHosted = false
            Log.audio.info("Voice hold ended (wiring in flight — discard left to its commit)")
            return
        }
        for node in attachedPlaybackNodes {
            node.stop()
            if let audioEngine, audioEngine.attachedNodes.contains(node) {
                audioEngine.detach(node)
            }
        }
        attachedPlaybackNodes.removeAll()
        voicePlaybackHosted = false
        if isCapturing {
            // A capture the session doesn't own (a dictation take) is
            // mid-flight: it keeps the gate and the engine, and its own stop
            // unwires the hold's render side (stopCapture's holdRenderWired
            // sweep) on the then-stopped engine.
            Log.audio.error(
                "Voice hold ended mid-capture — the take keeps the engine; render unwires at its stop"
            )
        } else if let audioEngine, voiceHoldWired {
            lastIntentionalReconfigure = Date()
            if audioEngine.isRunning {
                audioEngine.stop()
            }
            if inputTapInstalled {
                audioEngine.inputNode.removeTap(onBus: 0)
                inputTapInstalled = false
            }
            if holdRenderWired {
                audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
                holdRenderWired = false
            }
        }
        voiceHoldWired = false
        // The fallback lifecycle expects the engine plain at idle; the hold
        // built it armed for the AEC.
        if !isCapturing, voiceProcessingArmed, lifecycle.voiceProcessing == .disarmAfterGrace {
            try? reconcileVoiceProcessing(false)
        }
        duckPolicy.captureDidStop()
        Log.audio.info("Voice hold ended")
    }

    /// Attaches a voice-session player node to the held engine's mixer —
    /// Apple-documented dynamic reconfiguration on a running engine (all
    /// reconnections stay upstream of the mixer, AVAudioEngine.h). Returns
    /// false when the engine cannot host playback right now — the caller
    /// falls back to its dedicated engine (weaker cancellation, but the
    /// reply still plays).
    func attachVoicePlayback(node: AVAudioPlayerNode, format: AVAudioFormat) -> Bool {
        guard voiceHoldActive, voicePlaybackHosted, !holdWiringInProgress,
            let audioEngine, audioEngine.isRunning
        else { return false }
        audioEngine.attach(node)
        audioEngine.connect(node, to: audioEngine.mainMixerNode, format: format)
        attachedPlaybackNodes.append(node)
        return true
    }

    func detachVoicePlayback(node: AVAudioPlayerNode) {
        attachedPlaybackNodes.removeAll { $0 === node }
        guard let audioEngine, audioEngine.attachedNodes.contains(node) else { return }
        node.stop()
        audioEngine.detach(node)
    }

    /// The engine underneath attached playback nodes is going away — tell
    /// the adapter so it can end its utterance instead of waiting on buffer
    /// callbacks that will never fire.
    private func invalidateVoicePlayback() {
        guard !attachedPlaybackNodes.isEmpty || voicePlaybackHosted else { return }
        attachedPlaybackNodes.removeAll()
        voicePlaybackHosted = false
        onVoicePlaybackInvalidated?()
    }

    /// Coalesced rebuild-under-hold: a device change while held means the
    /// fresh engine must be re-wired, not just re-armed. A mid-take rebuild
    /// is deferred to the capture's own stop — never torn down mid-take.
    private func scheduleHoldRebuild() {
        holdRebuildTask?.cancel()
        holdRebuildTask = Task { [weak self] in
            guard let lifecycle = self?.lifecycle else { return }
            try? await Task.sleep(for: lifecycle.idleRebuildDelay)
            guard !Task.isCancelled, let self, self.voiceHoldActive else { return }
            if self.isCapturing {
                // The take rides out on the dirty engine (an empty one hits
                // the wedge path); its stop wires the fresh hold.
                self.voiceHoldWired = false
                self.voicePlaybackHosted = false
                return
            }
            self.scheduleHoldWiring(rebuildFirst: true)
        }
    }

    /// Single entry for every hold (re)wiring. Serial by construction: while
    /// a wiring is in flight a new request folds into `holdWireQueued` rather
    /// than racing it — two wirings never touch one engine at once.
    private func scheduleHoldWiring(rebuildFirst: Bool) {
        holdGeneration += 1
        lastIntentionalReconfigure = Date()
        if holdWiringInProgress {
            holdWireQueued = (holdWireQueued ?? false) || rebuildFirst
            return
        }
        startHoldWiring(rebuildFirst: rebuildFirst)
    }

    /// Serial-handoff box for the kept engine into the detached wiring.
    /// `@unchecked` because the safety is the hold's own discipline: while a
    /// wiring is in flight, every MainActor path that would touch the engine
    /// defers to `holdWiringInProgress` — the engine changes hands exactly
    /// once, never concurrently.
    nonisolated private struct HoldEngineBox: @unchecked Sendable {
        let engine: AVAudioEngine
        let armed: Bool
    }

    private func startHoldWiring(rebuildFirst: Bool) {
        holdRebuildTask?.cancel()
        holdRebuildTask = nil
        holdWiringInProgress = true
        if rebuildFirst { tearDownEngine() }
        let kept =
            rebuildFirst
            ? nil
            : audioEngine.map { HoldEngineBox(engine: $0, armed: voiceProcessingArmed) }
        let generation = holdGeneration
        let gate = captureGate
        let buffer = sampleBuffer
        let bufferSize = self.bufferSize
        let meterContinuation = meterStream.continuation
        Log.audio.info("Voice hold wiring started (rebuild: \(rebuildFirst))")
        let wiring = Task.detached { () -> HoldWireOutcome in
            // The hold always wants the AEC — a voice session without echo
            // cancellation is the bug this ADR exists to fix — so a fresh
            // build arms regardless of the idle lifecycle.
            let built =
                kept.map { (engine: $0.engine, armed: $0.armed) }
                ?? Self.buildEngine(voiceProcessing: true)
            return Self.performHoldWiring(
                built: built, bufferSize: bufferSize, gate: gate, buffer: buffer,
                meterContinuation: meterContinuation)
        }
        holdWireTask = wiring
        Task { [weak self] in
            let outcome = await wiring.value
            self?.commitHoldWiring(
                outcome, generation: generation, freshEngine: rebuildFirst)
        }
    }

    /// The wiring's MainActor landing strip. The current generation commits;
    /// a stale one discards its outcome on the stopped engine and runs
    /// whatever queued behind it.
    private func commitHoldWiring(
        _ outcome: HoldWireOutcome, generation: Int, freshEngine: Bool
    ) {
        guard voiceHoldActive, holdGeneration == generation else {
            discardHoldWiring(outcome)
            let queued = holdWireQueued
            holdWireQueued = nil
            if let queued, voiceHoldActive {
                startHoldWiring(rebuildFirst: queued)
                return
            }
            holdWiringInProgress = false
            holdWireTask = nil
            return
        }
        holdWiringInProgress = false
        holdWireTask = nil
        if freshEngine {
            adoptBuiltEngine((engine: outcome.engine, armed: outcome.armed))
        }
        inputTapInstalled = outcome.tapInstalled
        if outcome.tapInstalled {
            inputSampleRate = outcome.inputSampleRate
        }
        holdRenderWired = outcome.renderVerified
        voiceHoldWired = outcome.tapInstalled && outcome.engineRunning
        voicePlaybackHosted =
            outcome.engineRunning
            && lifecycle.hostsPlayback(
                armed: outcome.armed, renderVerified: outcome.renderVerified)
        lastIntentionalReconfigure = outcome.completedAt
        Log.audio.info(
            """
            Voice hold wired (running: \(outcome.engineRunning), \
            render verified: \(outcome.renderVerified), hosted: \(self.voicePlaybackHosted))
            """)
    }

    /// A stale wiring's outcome, undone in the only legal order: stop first,
    /// then remove the tap, then unwire the render side.
    private func discardHoldWiring(_ outcome: HoldWireOutcome) {
        if outcome.engineRunning {
            outcome.engine.stop()
        }
        if outcome.tapInstalled {
            outcome.engine.inputNode.removeTap(onBus: 0)
        }
        if outcome.renderVerified {
            outcome.engine.disconnectNodeOutput(outcome.engine.mainMixerNode)
        }
        Log.audio.info("Discarded stale voice hold wiring")
    }

    /// The outcome of one detached wiring — everything the commit hop needs
    /// to adopt or discard the work without re-querying the engine.
    /// `@unchecked Sendable` for the serial detached→MainActor handoff (the
    /// engine inside is only ever touched by one side at a time — the hold's
    /// `holdWiringInProgress` discipline).
    nonisolated private struct HoldWireOutcome: @unchecked Sendable {
        let engine: AVAudioEngine
        let armed: Bool
        var inputSampleRate: Double = 0
        var tapInstalled = false
        var renderVerified = false
        var engineRunning = false
        var completedAt = Date.distantPast
    }

    /// The hold's whole stopped-engine discipline, run detached (E7-verified):
    /// install the tap once, wire the render side with format verification,
    /// start. Every format-touching call happens here and only here — never
    /// on a running engine (the 2026-07-17 crash class: installTap →
    /// CreateRecordingTap → SetOutputFormat → SetFormat under running VP).
    nonisolated private static func performHoldWiring(
        built: (engine: AVAudioEngine, armed: Bool),
        bufferSize: AVAudioFrameCount,
        gate: CaptureGate,
        buffer: SampleBuffer,
        meterContinuation: AsyncStream<MeterFrame>.Continuation
    ) -> HoldWireOutcome {
        var outcome = HoldWireOutcome(engine: built.engine, armed: built.armed)
        let engine = built.engine
        let inputNode = engine.inputNode

        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0,
            let tapFormat = AudioConverter.monoFloat32Format(
                sampleRate: inputFormat.sampleRate)
        else {
            Log.audio.error(
                "Voice hold wiring: input reports no format — hold stays unwired")
            outcome.completedAt = Date()
            return outcome
        }
        outcome.inputSampleRate = inputFormat.sampleRate

        // Invariant 1: the tap goes on once per hold, only on a stopped engine.
        let meter = AudioMeterTap(
            sampleRate: tapFormat.sampleRate, continuation: meterContinuation)
        inputNode.installTap(
            onBus: 0, bufferSize: bufferSize, format: tapFormat,
            block: makeAudioTapHandler(gate: gate, buffer: buffer, meter: meter))
        outcome.tapInstalled = true

        // Invariant 2: the render side is wired stopped, and only with the
        // pin verified (AVAudioIONode.h: the input node's output format and
        // the output node's input format must match, changeable only while
        // stopped). The tap just set the input side; the render side follows.
        let ioFormat = inputNode.outputFormat(forBus: 0)
        var renderConnected = false
        if ioFormat.sampleRate > 0, ioFormat.channelCount > 0 {
            engine.connect(engine.mainMixerNode, to: engine.outputNode, format: ioFormat)
            renderConnected = true
            let readBack = engine.outputNode.inputFormat(forBus: 0)
            outcome.renderVerified =
                readBack.sampleRate == ioFormat.sampleRate
                && readBack.channelCount == ioFormat.channelCount
            if !outcome.renderVerified {
                Log.audio.error(
                    """
                    Voice hold wiring: render format read-back mismatch \
                    (\(readBack.sampleRate) Hz/\(readBack.channelCount) ch vs \
                    \(ioFormat.sampleRate) Hz/\(ioFormat.channelCount) ch) — input-only
                    """)
                engine.disconnectNodeOutput(engine.mainMixerNode)
                renderConnected = false
            }
        }

        engine.prepare()
        do {
            try engine.start()
            outcome.engineRunning = true
        } catch {
            if renderConnected {
                // A render side the device refuses must not cost the session
                // its microphone: retry input-only and let playback fall back.
                Log.audio.error(
                    """
                    Voice hold start failed with render side, retrying input-only: \
                    \(error.localizedDescription)
                    """)
                engine.disconnectNodeOutput(engine.mainMixerNode)
                outcome.renderVerified = false
                engine.prepare()
                do {
                    try engine.start()
                    outcome.engineRunning = true
                } catch {
                    Log.audio.error(
                        "Voice hold input-only start also failed: \(error.localizedDescription)")
                }
            } else {
                Log.audio.error("Voice hold start failed: \(error.localizedDescription)")
            }
        }
        outcome.completedAt = Date()
        return outcome
    }

}
