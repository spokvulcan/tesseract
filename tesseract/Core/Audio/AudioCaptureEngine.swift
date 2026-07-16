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

    /// The current capture records levels only (the settings meter): the tap
    /// appends no samples, and `stopCapture()` discards the empty recording —
    /// a minutes-long meter session costs no memory and its stop is instant.
    private var meteringOnly = false

    // MARK: Voice hold (Dual-Path Playback, ADR-0041)

    /// While a voice session runs the engine is *held*: it keeps running
    /// between captures (start/stop degrade to tap install/remove) and hosts
    /// the session's TTS player nodes, so VPIO's echo canceller hears the
    /// reply as its own far-end reference instead of "other audio".
    private(set) var voiceHoldActive = false
    /// Whether the held engine's render side is wired and playback nodes can
    /// attach. False when the render wiring failed or the hold began while a
    /// capture was mid-take — callers fall back to their dedicated engine.
    private(set) var voicePlaybackHosted = false
    /// Fired when the engine is torn down or rebuilt underneath attached
    /// playback nodes — the playback adapter treats it as end-of-utterance.
    var onVoicePlaybackInvalidated: (@MainActor () -> Void)?
    private var attachedPlaybackNodes: [AVAudioPlayerNode] = []

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
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }

        if audioEngine == nil || engineNeedsRebuild {
            rebuildEngine(voiceProcessing: lifecycle.prewarmBuildsArmed)
        }
    }

    // MARK: - Voice hold (Dual-Path Playback, ADR-0041)

    /// Begins the voice-session hold: the engine starts now and keeps running
    /// until `endVoiceHold()`, with its render side wired so the session's
    /// TTS plays through the VPIO unit. A no-op without microphone permission
    /// (the session's own capture start surfaces that error).
    func beginVoiceHold() {
        guard !voiceHoldActive else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }
        voiceHoldActive = true
        if audioEngine == nil || engineNeedsRebuild {
            // rebuildEngine restarts the held engine on its way out.
            rebuildEngine(voiceProcessing: lifecycle.prewarmBuildsArmed)
        } else {
            startHeldEngine()
        }
        Log.audio.info("Voice hold began (playback hosted: \(self.voicePlaybackHosted))")
    }

    func endVoiceHold() {
        guard voiceHoldActive else { return }
        voiceHoldActive = false
        for node in attachedPlaybackNodes {
            node.stop()
            if let audioEngine, audioEngine.attachedNodes.contains(node) {
                audioEngine.detach(node)
            }
        }
        attachedPlaybackNodes.removeAll()
        voicePlaybackHosted = false
        if let audioEngine {
            lastIntentionalReconfigure = Date()
            if !isCapturing, audioEngine.isRunning {
                audioEngine.stop()
            }
            // Back to the input-only graph — the render side is the hold's.
            audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
        }
        Log.audio.info("Voice hold ended")
    }

    /// Attaches a voice-session player node to the held engine's mixer.
    /// Returns false when the engine cannot host playback right now — the
    /// caller falls back to its dedicated engine (weaker echo cancellation,
    /// but the reply still plays).
    func attachVoicePlayback(node: AVAudioPlayerNode, format: AVAudioFormat) -> Bool {
        guard voiceHoldActive, voicePlaybackHosted,
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

    /// Starts (or restarts) the held engine with its render side wired. The
    /// mixer→output connection is made while stopped — under Voice
    /// Processing the IO formats are pinned to each other and only mutable
    /// on a stopped engine (`AVAudioIONode.h`).
    private func startHeldEngine() {
        guard voiceHoldActive, let audioEngine, !audioEngine.isRunning else { return }
        let ioFormat = audioEngine.inputNode.outputFormat(forBus: 0)
        if ioFormat.sampleRate > 0, ioFormat.channelCount > 0 {
            audioEngine.connect(
                audioEngine.mainMixerNode, to: audioEngine.outputNode, format: ioFormat)
            voicePlaybackHosted = true
        } else {
            voicePlaybackHosted = false
        }
        lastIntentionalReconfigure = Date()
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            // A render side the device refuses must not cost the session its
            // microphone: retry input-only and let playback fall back.
            Log.audio.error(
                """
                Voice hold start failed with render side, retrying input-only: \
                \(error.localizedDescription)
                """)
            audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
            voicePlaybackHosted = false
            lastIntentionalReconfigure = Date()
            do { try audioEngine.start() } catch {
                Log.audio.error("Voice hold start failed: \(error.localizedDescription)")
            }
        }
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
        if voiceProcessingArmed {
            duckPolicy.engineDidArm()
        }

        configChangeCancellable = NotificationCenter.default
            .publisher(for: .AVAudioEngineConfigurationChange, object: engine)
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
                // Re-arm while idle so the press after a device change stays
                // at engine-start cost instead of paying the rebuild + arm.
                self.scheduleIdleRebuild()
            }

        // A rebuild under a voice hold restarts the held engine (render side
        // wired) so the session's next reply can attach — the reply that was
        // playing was invalidated by the teardown above.
        if voiceHoldActive { startHeldEngine() }
    }

    private func tearDownEngine() {
        disarmTask?.cancel()
        disarmTask = nil
        configChangeCancellable = nil
        invalidateVoicePlayback()
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
            block: Self.makeAudioTapHandler(buffer: buffer, meter: meterTap)
        )
        inputTapInstalled = true

        // Duck other system audio only for a real dictation capture — the
        // settings meter keeps the idle treatment. Set before start so the
        // level is baked into this run of the IO unit.
        duckPolicy.captureDidStart(meteringOnly: meteringOnly)

        let startClock = Date()
        lastIntentionalReconfigure = startClock
        do {
            // Under a voice hold the engine is already running (it hosts the
            // session's TTS playback) — the capture is just the tap.
            if !audioEngine.isRunning {
                audioEngine.prepare()
                try audioEngine.start()
            }
            isCapturing = true
            Log.audio.info(
                """
                Capture started in \
                \(String(format: "%.0f", Date().timeIntervalSince(startClock) * 1000)) ms
                """)
        } catch {
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

        // Stop IO but keep the engine (and its Voice Processing arm) for the
        // next press. Same order as teardown: stop before removing the tap,
        // so AudioOutputUnitStop never races a nil tap callback.
        let stopClock = Date()
        lastIntentionalReconfigure = stopClock
        if let audioEngine {
            // Under a voice hold the engine keeps running between captures —
            // stopping it would cut the session's TTS mid-word (ADR-0041).
            if !voiceHoldActive {
                audioEngine.stop()
            }
            if inputTapInstalled {
                audioEngine.inputNode.removeTap(onBus: 0)
                inputTapInstalled = false
            }
            // Back to the idle treatment (full volume) the moment the capture
            // ends; the fallback lifecycle additionally schedules the disarm
            // that releases its duck for good.
            duckPolicy.captureDidStop()
            if voiceProcessingArmed, lifecycle.disarmsAfterCapture {
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
        let wasVoiceProcessed = voiceProcessingArmed

        if samples.isEmpty,
            lifecycle.emptyCaptureVerdict(duration: duration) == .wedgedInput
        {
            // The idle rebuild re-arms in the background so the next press
            // starts fresh at engine-start cost. A `tapBeatFirstBuffer`
            // verdict falls through to the session's minimum-duration guard
            // ("too short").
            Log.audio.error(
                """
                Capture delivered no samples over \
                \(String(format: "%.1f", duration)) s — discarding engine
                """)
            tearDownEngine()
            scheduleIdleRebuild()
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

            // Copy samples to the thread-safe buffer (nil for a metering-only
            // capture — the settings meter wants the level, not the audio)
            buffer?.append(UnsafeBufferPointer(start: channelData, count: frameCount))

            // Level + spectrum straight into the meter stream.
            meter?.process(channelData, frameCount: frameCount, level: normalizedLevel)
        }
    }

}
