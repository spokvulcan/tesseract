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
        var reference: ReferenceTake?
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
            // Before the lease: a checkpoint that isn't on disk fails here
            // rather than queueing behind, or holding up, LLM work.
            try await synthesizer.checkAvailable(model)
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
            state.reference = pinned.referenceTake
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
        guard let state = sessions[id], let take = state.reference else { return nil }
        return PinnedVoice(
            modelFingerprint: model.fingerprint,
            voiceDescription: state.voice.description,
            language: state.voice.language,
            referenceText: take.text,
            codeFrames: take.codeFrames)
    }

    // MARK: - Admission (deterministic supersession, ADR-0038)

    /// `retake`: speak only the lead segment, rendered from the description
    /// alone, and make it the session's new Reference Take once it finishes
    /// (a cancelled retake keeps the old one).
    func admit(
        sessionID: UUID, text: String, options: SpeechOptions, retake: Bool = false
    ) async throws -> Utterance {
        guard let state = sessions[sessionID], !state.closed else {
            throw SpeechEngineError.sessionClosed
        }
        try await ensureLoaded()
        guard let format = await synthesizer.audioFormat() else {
            throw SpeechEngineError.modelUnavailable("audio format unavailable after load")
        }

        // Supersede: the previous utterance's stream has terminated before we return.
        await cancelActiveAndWait()

        // A session that has no take yet (or is retaking) keeps this
        // utterance's first segment short: it becomes the Reference Take.
        let capturesReference =
            state.profile.reference == .pinned && (retake || state.reference == nil)
        var segments = Segmenter.segment(
            text, leadTokens: capturesReference ? Segmenter.referenceLeadTokens : nil)
        if retake { segments = Array(segments.prefix(1)) }
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
                voice: state.voice, parameters: parameters, seed: seed,
                pacing: state.profile.pacing,
                startingReference: retake ? nil : state.reference,
                capturesReference: capturesReference, format: format, channel: channel)
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
        var captured: ReferenceTake?
    }

    private func runUtterance(
        utteranceID: UUID, sessionID: UUID, segments: [TextSegment],
        voice: Voice, parameters: TTSParameters, seed: UInt64, pacing: PacingPolicy,
        startingReference: ReferenceTake?, capturesReference: Bool,
        format: AudioFormat, channel: UtteranceChannel
    ) async {
        var cumulativeFrames = 0
        var segmentFrameCounts: [Int] = []
        var reference = startingReference

        do {
            for segment in segments {
                if case .lookahead(let limit) = pacing {
                    await channel.waitForDemand(limit: limit)
                }
                try Task.checkCancellation()

                // The lead segment of a capturing utterance is rendered from
                // the description alone and becomes the take; every other
                // segment continues the take.
                let captures = capturesReference && segment.index == 0
                let request = SegmentRequest(
                    text: segment.text,
                    voiceDescription: voice.description,
                    language: voice.language,
                    parameters: parameters,
                    seed: seed,
                    reference: captures ? nil : reference,
                    capturesReference: captures)

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
                    var captured: ReferenceTake?

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
                        case .done(let take):
                            captured = take
                        }
                        try Task.checkCancellation()
                    }
                    try Task.checkCancellation()
                    return BurstOutcome(frameCount: emittedFrames, captured: captured)
                }
                diagnostics?.event("burst.end", "segment \(segmentIndex), \(outcome.frameCount) frames")

                cumulativeFrames += outcome.frameCount
                segmentFrameCounts.append(outcome.frameCount)
                if let captured = outcome.captured {
                    reference = captured
                    sessions[sessionID]?.reference = captured
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

    /// Re-roll the voice: speak the opening of `text` (its first sentence or
    /// two) from the description alone, and keep it as the session's new
    /// Reference Take once it finishes. Pass a fresh seed (`.entropy`), or
    /// the same seed renders the same take again. A cancelled retake leaves
    /// the old take in place.
    public func retake(_ text: String, options: SpeechOptions) async throws -> Utterance {
        try await engine.admit(sessionID: id, text: text, options: options, retake: true)
    }

    /// The session's voice with its Reference Take, once one exists (nil
    /// before the first segment finishes). Survives relaunch via
    /// `PinnedVoice.serialized()`.
    public func exportPinnedVoice() async -> PinnedVoice? {
        await engine.exportPinnedVoice(id)
    }

    /// Deterministic release of the session's voice state; idempotent.
    public func close() async {
        await engine.closeSession(id)
    }
}
