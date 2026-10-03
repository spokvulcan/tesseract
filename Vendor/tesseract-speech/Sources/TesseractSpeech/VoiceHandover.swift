// TesseractSpeech — the phone's two voices behind one Speech Synthesizer
// port (ADR-0084): the neural voice, and the System Voice that reads while
// the neural voice downloads or prepares, where it can't keep up, and when
// the phone is too warm for it.

import Foundation

/// Hands each segment to one of two synthesizers, chosen as the segment
/// starts: so a change of voice lands on a segment boundary, never inside
/// one, and the engine's frames stay gapless across it. Both must speak the
/// same audio format.
///
/// Loading, warming up, priming and unloading go to the fallback only. The
/// primary is the app's to prepare and release (the phone's Voice
/// Preparation), and `choose` names it only while it is ready.
public struct VoiceHandover: SpeechSynthesizing {
    public enum Choice: Sendable, Equatable {
        case primary
        case fallback
    }

    private let primary: any SpeechSynthesizing
    private let fallback: any SpeechSynthesizing
    private let choose: @Sendable () async -> Choice
    private let speakers: [String]

    /// `speakers`: the Preset Voices sessions may open with, whichever voice
    /// reads them.
    public init(
        primary: any SpeechSynthesizing, fallback: any SpeechSynthesizing, speakers: [String],
        choose: @escaping @Sendable () async -> Choice
    ) {
        self.primary = primary
        self.fallback = fallback
        self.speakers = speakers
        self.choose = choose
    }

    public func checkAvailable(_ spec: TTSModelSpec) async throws {
        try await fallback.checkAvailable(spec)
    }

    public func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws {
        try await fallback.load(spec, onPhase: onPhase)
    }

    public func warmUp() async throws {
        try await fallback.warmUp()
    }

    public func primeVoice(description: String?, language: String?) async throws {
        try await fallback.primeVoice(description: description, language: language)
    }

    public func presetSpeakers() async -> [String] { speakers }

    public func unload() async {
        await fallback.unload()
    }

    public func audioFormat() async -> AudioFormat? {
        await fallback.audioFormat()
    }

    public func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    {
        switch await choose() {
        case .primary: await primary.synthesizeSegment(request)
        case .fallback: await fallback.synthesizeSegment(request)
        }
    }

    public func trimCaches() async {
        await primary.trimCaches()
        await fallback.trimCaches()
    }
}
