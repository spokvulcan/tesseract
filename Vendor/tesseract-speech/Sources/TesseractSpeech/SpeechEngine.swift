// TesseractSpeech — engine v2 (ADR-0038). The facade actor: lifecycle,
// sessions, utterance admission with deterministic supersession, and the
// per-segment burst driver that owns GPU-lease and memory discipline.

import Foundation

public actor SpeechEngine {
    private let model: TTSModelSpec
    private let synthesizer: any SpeechSynthesizing
    private let gpu: any GPULeasing
    private let memory: MemoryPolicy
    private let diagnostics: (any SpeechDiagnosticsTap)?

    public private(set) var readiness: Readiness = .unloaded
    private var inFlightPrepare: Task<Void, Error>?

    private struct SessionState {
        var profile: SessionProfile
        var voice: Voice
        var anchor: AnchorHandle?
        var closed = false
    }

    private var sessions: [UUID: SessionState] = [:]
    private var active: (utteranceID: UUID, task: Task<Void, Never>, channel: UtteranceChannel)?

    public init(
        model: TTSModelSpec,
        synthesizer: any SpeechSynthesizing,
        gpu: any GPULeasing,
        memory: MemoryPolicy = .default,
        diagnostics: (any SpeechDiagnosticsTap)? = nil
    ) {
        self.model = model
        self.synthesizer = synthesizer
        self.gpu = gpu
        self.memory = memory
        self.diagnostics = diagnostics
    }

    // MARK: - Lifecycle (ADR-0039)

    /// Drive the engine to `target` readiness. Idempotent; concurrent calls
    /// coalesce onto one transition. `.warm` additionally primes `priming`
    /// voices' instruct prefixes off the hot path.
    public func prepare(
        _ target: Readiness,
        priming: [Voice] = [],
        onPhase: (@Sendable (EnginePhase) -> Void)? = nil
    ) async throws {
        if target == .unloaded {
            await unload()
            return
        }
        try await ensureLoaded(onPhase: onPhase)
        if target == .warm {
            try await withLeaseMapped {
                onPhase?(.warmingKernels)
                try await self.synthesizer.warmUp()
                for voice in priming {
                    onPhase?(.primingVoice)
                    try await self.synthesizer.primeVoice(
                        description: voice.description, language: voice.language)
                }
            }
            if readiness < .warm { readiness = .warm }
            onPhase?(.ready)
        }
    }

    /// Deterministic teardown: cancels the active utterance (its stream
    /// terminates before this returns), releases weights/KV/caches, syncs the
    /// GPU stream. Sessions survive as ingredient values (ADR-0038/0039).
    public func unload() async {
        await cancelActiveAndWait()
        inFlightPrepare?.cancel()
        inFlightPrepare = nil
        await synthesizer.unload()
        readiness = .unloaded
    }

    private func ensureLoaded(onPhase: (@Sendable (EnginePhase) -> Void)? = nil) async throws {
        if readiness >= .loaded { return }
        if let inFlight = inFlightPrepare {
            try await inFlight.value
            return
        }
        let task = Task { [model, synthesizer, gpu] in
            try await gpu.withLease {
                try await synthesizer.load(model, onPhase: onPhase)
                try await synthesizer.warmUp()
            }
        }
        inFlightPrepare = task
        defer { inFlightPrepare = nil }
        do {
            try await task.value
            readiness = .warm
        } catch {
            readiness = .unloaded
            throw mapLoadError(error)
        }
    }

    // MARK: - Sessions

    /// Open a voice session. Ensures the model is resident and the voice's
    /// instruct prefix is primed — off the utterance hot path (ADR-0039).
    public func session(_ profile: SessionProfile, voice: Voice) async throws -> SpeechSession {
        if case .pinned(let pinned) = voice {
            guard pinned.modelFingerprint == model.fingerprint else {
                throw SpeechEngineError.voiceIncompatible(
                    expected: model.fingerprint, found: pinned.modelFingerprint)
            }
        }
        try await ensureLoaded()
        try await withLeaseMapped {
            try await self.synthesizer.primeVoice(
                description: voice.description, language: voice.language)
        }

        let id = UUID()
        var state = SessionState(profile: profile, voice: voice)
        if case .pinned(let pinned) = voice, !pinned.codeFrames.isEmpty {
            state.anchor = AnchorHandle(
                codeFrames: pinned.codeFrames,
                voiceDescription: pinned.voiceDescription,
                language: pinned.language)
        }
        sessions[id] = state
        return SpeechSession(engine: self, id: id, voice: voice)
    }

    func closeSession(_ id: UUID) async {
        guard sessions[id] != nil else { return }
        sessions[id]?.closed = true
        // An active utterance from this session stops with the session.
        await cancelActiveAndWait()
        sessions[id] = nil
    }

    func exportPinnedVoice(_ id: UUID) -> PinnedVoice? {
        guard let state = sessions[id], let anchor = state.anchor else { return nil }
        return PinnedVoice(
            modelFingerprint: model.fingerprint,
            voiceDescription: anchor.voiceDescription,
            language: anchor.language,
            codeFrames: anchor.codeFrames)
    }

    // MARK: - Admission (deterministic supersession, ADR-0038)

    func admit(sessionID: UUID, text: String, options: SpeechOptions) async throws -> Utterance {
        guard let state = sessions[sessionID], !state.closed else {
            throw SpeechEngineError.sessionClosed
        }
        try await ensureLoaded()
        guard let format = await synthesizer.audioFormat() else {
            throw SpeechEngineError.modelUnavailable("audio format unavailable after load")
        }

        // Supersede: the previous utterance's stream has terminated before we return.
        await cancelActiveAndWait()

        let segments = Segmenter.segment(text)
        let parameters = options.parameters ?? state.profile.defaults
        let seed: UInt64
        switch options.seed {
        case .entropy: seed = UInt64.random(in: UInt64.min...UInt64.max)
        case .fixed(let value): seed = value
        }

        let channel = UtteranceChannel()
        let utteranceID = UUID()
        let driver = Task { [weak self] in
            guard let self else { return }
            await self.runUtterance(
                utteranceID: utteranceID, sessionID: sessionID, segments: segments,
                profile: state.profile, voice: state.voice, parameters: parameters,
                seed: seed, startingAnchor: state.anchor, format: format, channel: channel)
        }
        await channel.setOnConsumerGone { driver.cancel() }
        active = (utteranceID, driver, channel)

        let token = DropToken { driver.cancel() }
        return Utterance(
            sampleRate: format.sampleRate,
            framesPerSecond: format.framesPerSecond,
            segmentCount: segments.count,
            channel: channel,
            dropToken: token)
    }

    private func cancelActiveAndWait() async {
        guard let current = active else { return }
        active = nil
        current.task.cancel()
        await current.task.value
    }

    // MARK: - The utterance driver

    private struct BurstOutcome: Sendable {
        var frameCount: Int
        var captured: AnchorHandle?
    }

    private func runUtterance(
        utteranceID: UUID, sessionID: UUID, segments: [TextSegment],
        profile: SessionProfile, voice: Voice, parameters: TTSParameters,
        seed: UInt64, startingAnchor: AnchorHandle?, format: AudioFormat,
        channel: UtteranceChannel
    ) async {
        var cumulativeFrames = 0
        var segmentFrameCounts: [Int] = []
        var utteranceAnchor = startingAnchor

        do {
            for segment in segments {
                if case .lookahead(let limit) = profile.pacing {
                    await channel.waitForDemand(limit: limit)
                }
                try Task.checkCancellation()

                let captureSteps = anchorCaptureSteps(
                    profile: profile, segmentIndex: segment.index, anchor: utteranceAnchor)
                let request = SegmentRequest(
                    text: segment.text,
                    voiceDescription: voice.description,
                    language: voice.language,
                    parameters: parameters,
                    seed: seed,
                    anchor: utteranceAnchor,
                    captureAnchorSteps: captureSteps)

                let startFrame = cumulativeFrames
                let segmentIndex = segment.index
                let synthesizer = self.synthesizer
                let samplesPerFrame = format.samplesPerFrame

                diagnostics?.event("burst.begin", "segment \(segmentIndex)")
                let outcome: BurstOutcome = try await gpu.withLease {
                    let offsets = try await synthesizer.alignmentOffsets(for: segment.text)
                    await channel.send(.segment(SegmentScript(
                        index: segmentIndex, text: segment.text,
                        tokenCharOffsets: offsets, startFrame: startFrame)))

                    var segmentSamples = 0
                    var emittedFrames = 0
                    var captured: AnchorHandle?

                    let stream = await synthesizer.synthesizeSegment(request)
                    for try await event in stream {
                        switch event {
                        case .chunk(let samples):
                            guard !samples.isEmpty else { break }
                            segmentSamples += samples.count
                            let totalFrames =
                                (segmentSamples + samplesPerFrame - 1) / samplesPerFrame
                            let range = (startFrame + emittedFrames)..<(startFrame + totalFrames)
                            emittedFrames = totalFrames
                            await channel.send(.audio(AudioChunk(
                                samples: samples, frames: range, segmentIndex: segmentIndex)))
                        case .done(let anchor):
                            captured = anchor
                        }
                        try Task.checkCancellation()
                    }
                    try Task.checkCancellation()
                    return BurstOutcome(frameCount: emittedFrames, captured: captured)
                }
                diagnostics?.event("burst.end", "segment \(segmentIndex), \(outcome.frameCount) frames")

                cumulativeFrames += outcome.frameCount
                segmentFrameCounts.append(outcome.frameCount)
                if let captured = outcome.captured, utteranceAnchor == nil {
                    utteranceAnchor = captured
                    if case .pinned = profile.anchor {
                        sessions[sessionID]?.anchor = captured
                    }
                }
                await channel.send(.segmentDone(index: segmentIndex))
            }

            await channel.send(.finished(SessionSummary(
                totalFrames: cumulativeFrames, segmentFrameCounts: segmentFrameCounts)))
            await channel.finish(throwing: nil)
        } catch is CancellationError {
            await channel.finish(throwing: CancellationError())
        } catch {
            await channel.finish(throwing: SpeechEngineError.generationFailed(
                String(describing: error)))
        }

        if memory.clearsCacheAtUtteranceEnd {
            await synthesizer.trimCaches()
        }
        if active?.utteranceID == utteranceID {
            active = nil
        }
    }

    private func anchorCaptureSteps(
        profile: SessionProfile, segmentIndex: Int, anchor: AnchorHandle?
    ) -> Int? {
        guard anchor == nil, segmentIndex == 0 else { return nil }
        switch profile.anchor {
        case .none: return nil
        case .perUtterance(let steps), .pinned(let steps): return steps
        }
    }

    // MARK: - Helpers

    private func withLeaseMapped(_ body: @escaping @Sendable () async throws -> Void) async throws {
        do {
            try await gpu.withLease(body)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as SpeechEngineError {
            throw error
        } catch {
            throw SpeechEngineError.generationFailed(String(describing: error))
        }
    }

    private func mapLoadError(_ error: Error) -> Error {
        if error is CancellationError { return error }
        if let known = error as? SpeechEngineError { return known }
        return SpeechEngineError.modelUnavailable(String(describing: error))
    }
}

// MARK: - SpeechSession

/// A voice bound to cached model state, with a defined lifetime (ADR-0038).
public final class SpeechSession: Sendable {
    private let engine: SpeechEngine
    private let id: UUID
    public let voice: Voice

    init(engine: SpeechEngine, id: UUID, voice: Voice) {
        self.engine = engine
        self.id = id
        self.voice = voice
    }

    /// Admit one utterance. Supersedes any active utterance engine-wide —
    /// deterministically: the superseded stream has terminated before this
    /// returns. Returns before audio generation begins.
    public func speak(_ text: String, options: SpeechOptions = .default) async throws -> Utterance {
        try await engine.admit(sessionID: id, text: text, options: options)
    }

    /// The session's voice realization, exportable once an anchor has formed
    /// (nil before). Survives relaunch via `PinnedVoice.serialized()`.
    public func exportPinnedVoice() async -> PinnedVoice? {
        await engine.exportPinnedVoice(id)
    }

    /// Deterministic release of the session's voice state; idempotent.
    public func close() async {
        await engine.closeSession(id)
    }
}
