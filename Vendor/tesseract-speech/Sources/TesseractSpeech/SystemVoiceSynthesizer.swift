// TesseractSpeech — the system voice behind the Speech Synthesizer port.

@preconcurrency import AVFoundation
import Foundation
import Synchronization

/// The system's own voice (`AVSpeechSynthesizer`) behind the Speech
/// Synthesizer port. It reads where the neural voice can't: while that voice
/// downloads or is prepared, on a phone too slow for it, and while the phone
/// cools down. It is never a voice identity: it ignores the description and
/// any Reference Take, and follows only the language.
///
/// Its audio comes in the neural voice's format, 24 kHz in frames of 1,920
/// samples, so the engine, the Read-Along and playback can't tell the two
/// apart. Each segment is resampled to it and padded with silence to whole
/// frames, so frame counts stay exact from segment to segment. Its words are
/// timed from the synthesizer's own word marks.
public struct SystemVoiceSynthesizer: SpeechSynthesizing {
    public static let format = AudioFormat(sampleRate: 24_000, samplesPerFrame: 1_920)

    private let renderer: any SystemSpeechRendering

    public init(renderer: any SystemSpeechRendering = AVSpeechRenderer()) {
        self.renderer = renderer
    }

    // The system voice is always there: nothing to check, load or warm.
    public func checkAvailable(_ spec: TTSModelSpec) async throws {}
    public func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws {
        onPhase?(.ready)
    }
    public func warmUp() async throws {}
    public func primeVoice(description: String?, language: String?) async throws {}
    public func unload() async {}
    public func trimCaches() async {}
    public func audioFormat() async -> AudioFormat? { Self.format }

    public func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    {
        let pieces = renderer.render(request.text, language: request.language)
        let words = WordOffsets(request.text)
        return AsyncThrowingStream { continuation in
            let task = Task {
                var segment = SystemSegment(words: words)
                do {
                    for try await piece in pieces {
                        try Task.checkCancellation()
                        for event in segment.add(piece) { continuation.yield(event) }
                    }
                    try Task.checkCancellation()
                    for event in segment.finish() { continuation.yield(event) }
                    continuation.yield(.done(capturedReference: nil))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

// MARK: - Rendering

/// One piece of a system rendering, in the order it comes.
public enum SystemSpeechPiece: Sendable, Equatable {
    /// Mono samples at `sampleRate`.
    case audio([Float], sampleRate: Double)
    /// A word of the rendered text, by its UTF-16 range, starts `sample`
    /// samples into the rendering, counted at its audio's sample rate.
    case word(NSRange, sample: Int)
}

/// Renders text with a system voice. `AVSpeechRenderer` in the apps; tests
/// script their own. Audio comes before the first word mark, and a mark
/// counts samples at the rate of the audio before it.
public protocol SystemSpeechRendering: Sendable {
    /// The rendering of `text` in `language` (the model's language names:
    /// "English", "German"…; nil for the device's own), piece by piece.
    /// Cancelling the consumer stops it.
    func render(_ text: String, language: String?) -> AsyncThrowingStream<SystemSpeechPiece, Error>
}

/// `AVSpeechSynthesizer` writing to buffers instead of the speaker, with
/// word marks. The best installed voice for the language reads.
public struct AVSpeechRenderer: SystemSpeechRendering {
    public init() {}

    public func render(_ text: String, language: String?)
        -> AsyncThrowingStream<SystemSpeechPiece, Error>
    {
        AsyncThrowingStream { continuation in
            let handle = SynthesizerHandle()
            let synthesizer = handle.synthesizer
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = Self.voice(for: language)
            let marks = MarkQueue()
            synthesizer.write(
                utterance,
                toBufferCallback: { buffer in
                    guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                    guard pcm.frameLength > 0 else {
                        for piece in marks.drain() { continuation.yield(piece) }
                        continuation.finish()
                        return
                    }
                    let bytesPerFrame = Int(pcm.format.streamDescription.pointee.mBytesPerFrame)
                    continuation.yield(.audio(Self.samples(pcm), sampleRate: pcm.format.sampleRate))
                    for piece in marks.setBytesPerFrame(bytesPerFrame) { continuation.yield(piece) }
                },
                toMarkerCallback: { markers in
                    for marker in markers where marker.mark == .word {
                        for piece in marks.add(marker.textRange, byte: marker.byteSampleOffset) {
                            continuation.yield(piece)
                        }
                    }
                })
            continuation.onTermination = { _ in handle.stop() }
        }
    }

    /// The model's language names, as the system's voices name them.
    static let locales = [
        "English": "en-US", "Chinese": "zh-CN", "Japanese": "ja-JP", "Korean": "ko-KR",
        "German": "de-DE", "French": "fr-FR", "Russian": "ru-RU", "Portuguese": "pt-BR",
        "Spanish": "es-ES", "Italian": "it-IT",
    ]

    /// The best installed voice for `language`: its usual locale first, then
    /// any of the language's, premium before enhanced before default.
    /// Novelty voices are left out.
    static func voice(for language: String?) -> AVSpeechSynthesisVoice? {
        let locale = language.flatMap { locales[$0] } ?? AVSpeechSynthesisVoice.currentLanguageCode()
        let code = String(locale.prefix(2))
        let candidates = AVSpeechSynthesisVoice.speechVoices().filter {
            $0.language.hasPrefix(code) && !$0.voiceTraits.contains(.isNoveltyVoice)
        }
        let best = candidates.max { a, b in
            (a.language == locale ? 1 : 0, a.quality.rawValue)
                < (b.language == locale ? 1 : 0, b.quality.rawValue)
        }
        return best ?? AVSpeechSynthesisVoice(language: locale)
    }

    /// The buffer's first channel as floats, whatever its sample format.
    static func samples(_ pcm: AVAudioPCMBuffer) -> [Float] {
        let count = Int(pcm.frameLength)
        if let floats = pcm.floatChannelData {
            return Array(UnsafeBufferPointer(start: floats[0], count: count))
        }
        if let ints = pcm.int16ChannelData {
            return UnsafeBufferPointer(start: ints[0], count: count).map { Float($0) / 32_768 }
        }
        if let ints = pcm.int32ChannelData {
            return UnsafeBufferPointer(start: ints[0], count: count).map { Float($0) / 2_147_483_648 }
        }
        return []
    }
}

/// The rendering's synthesizer, which the stream's end stops. A box because
/// the synthesizer isn't `Sendable`; it is only written to before the
/// rendering starts and only stopped after.
private final class SynthesizerHandle: @unchecked Sendable {
    let synthesizer = AVSpeechSynthesizer()

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}

/// Word marks waiting for the audio's sample size: a mark's offset counts
/// bytes, and the first buffer says how many make a sample.
private final class MarkQueue: Sendable {
    private struct State {
        var bytesPerFrame: Int?
        var pending: [(NSRange, Int)] = []
    }
    private let state = Mutex(State())

    func add(_ range: NSRange, byte: Int) -> [SystemSpeechPiece] {
        state.withLock { state in
            guard let size = state.bytesPerFrame, size > 0 else {
                state.pending.append((range, byte))
                return []
            }
            return [.word(range, sample: byte / size)]
        }
    }

    func setBytesPerFrame(_ size: Int) -> [SystemSpeechPiece] {
        state.withLock { state in
            guard state.bytesPerFrame == nil else { return [] }
            state.bytesPerFrame = size
            defer { state.pending = [] }
            return state.pending.map { .word($0.0, sample: $0.1 / max(size, 1)) }
        }
    }

    /// Marks still waiting when the rendering ends, read as float samples.
    func drain() -> [SystemSpeechPiece] {
        state.withLock { state in
            defer { state.pending = [] }
            return state.pending.map { .word($0.0, sample: $0.1 / (state.bytesPerFrame ?? 4)) }
        }
    }
}

// MARK: - One segment

/// Where each word of a segment's text starts, in UTF-16 offsets, with words
/// split as the engine splits them (`Character.separatesWords`).
struct WordOffsets: Sendable {
    private(set) var starts: [Int] = []

    init(_ text: String) {
        var offset = 0
        var inWord = false
        for character in text {
            let separates = character.separatesWords
            if !separates, !inWord { starts.append(offset) }
            inWord = !separates
            offset += character.utf16.count
        }
    }

    /// The word holding UTF-16 offset `location`: the last one starting at
    /// or before it.
    func word(at location: Int) -> Int? {
        var low = 0
        var high = starts.count - 1
        var found: Int?
        while low <= high {
            let mid = (low + high) / 2
            if starts[mid] <= location {
                found = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return found
    }
}

/// A segment's rendering turned into the port's events: audio resampled to
/// the engine's rate, padded to whole frames at the end, and word starts
/// sent once the audio they start in has been.
struct SystemSegment {
    let words: WordOffsets
    private var resampler: Resampler?
    private var samplesSent = 0
    /// Timed words not yet sent: word index and its first sample at 24 kHz.
    private var pending: [(word: Int, sample: Int)] = []
    /// The last word timed, so marks only move forward.
    private var lastWord = -1

    init(words: WordOffsets) {
        self.words = words
    }

    private var framesSent: Int { samplesSent / SystemVoiceSynthesizer.format.samplesPerFrame }

    mutating func add(_ piece: SystemSpeechPiece) -> [SynthesisEvent] {
        let target = Double(SystemVoiceSynthesizer.format.sampleRate)
        switch piece {
        case .audio(let samples, let rate):
            if resampler == nil || resampler?.inputRate != rate {
                resampler = Resampler(from: rate, to: target)
            }
            let out = resampler?.convert(samples) ?? []
            var events: [SynthesisEvent] = []
            if !out.isEmpty {
                samplesSent += out.count
                events.append(.chunk(out))
            }
            return events + timedWords(upTo: framesSent)
        case .word(let range, let sample):
            guard let word = words.word(at: range.location), word > lastWord else { return [] }
            lastWord = word
            let rate = resampler?.inputRate ?? target
            pending.append((word, Int((Double(sample) * target / rate).rounded())))
            return timedWords(upTo: framesSent)
        }
    }

    /// The resampler's tail and the silence that completes the last frame,
    /// then every word still waiting.
    mutating func finish() -> [SynthesisEvent] {
        var tail = resampler?.finish() ?? []
        let frame = SystemVoiceSynthesizer.format.samplesPerFrame
        let total = samplesSent + tail.count
        tail += [Float](repeating: 0, count: (frame - total % frame) % frame)
        var events: [SynthesisEvent] = []
        if !tail.isEmpty {
            samplesSent += tail.count
            events.append(.chunk(tail))
        }
        return events + timedWords(upTo: Int.max)
    }

    /// Words whose first frame is below `frames`, the frames already sent.
    private mutating func timedWords(upTo frames: Int) -> [SynthesisEvent] {
        let frame = SystemVoiceSynthesizer.format.samplesPerFrame
        let ready = pending.prefix { $0.sample / frame < frames }
        guard !ready.isEmpty else { return [] }
        pending.removeFirst(ready.count)
        return [.words(ready.map { WordStart(word: $0.word, frame: $0.sample / frame) })]
    }
}

/// Mono float samples from one rate to another, a stream at a time.
final class Resampler {
    let inputRate: Double
    private let outputRate: Double
    private let converter: AVAudioConverter?
    private let input: AVAudioFormat?
    private let output: AVAudioFormat?

    init(from inputRate: Double, to outputRate: Double) {
        self.inputRate = inputRate
        self.outputRate = outputRate
        guard inputRate != outputRate,
            let input = AVAudioFormat(standardFormatWithSampleRate: inputRate, channels: 1),
            let output = AVAudioFormat(standardFormatWithSampleRate: outputRate, channels: 1)
        else {
            converter = nil
            self.input = nil
            self.output = nil
            return
        }
        self.input = input
        self.output = output
        converter = AVAudioConverter(from: input, to: output)
    }

    func convert(_ samples: [Float]) -> [Float] {
        guard converter != nil else { return samples }
        return run(samples, end: false)
    }

    /// What the converter still holds, at the end of the stream.
    func finish() -> [Float] {
        guard converter != nil else { return [] }
        return run([], end: true)
    }

    private func run(_ samples: [Float], end: Bool) -> [Float] {
        guard let converter, let input, let output else { return samples }
        var buffer: AVAudioPCMBuffer?
        if !samples.isEmpty {
            buffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(samples.count))
            buffer?.frameLength = AVAudioFrameCount(samples.count)
            samples.withUnsafeBufferPointer { source in
                buffer?.floatChannelData?[0].update(from: source.baseAddress!, count: samples.count)
            }
        }
        var fed = buffer == nil
        var result: [Float] = []
        let capacity = AVAudioFrameCount(Double(samples.count) * outputRate / inputRate) + 1_024
        while true {
            guard let out = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: capacity) else { break }
            var error: NSError?
            let status = converter.convert(to: out, error: &error) { _, status in
                if !fed, let buffer {
                    fed = true
                    status.pointee = .haveData
                    return buffer
                }
                status.pointee = end ? .endOfStream : .noDataNow
                return nil
            }
            if out.frameLength > 0, let channel = out.floatChannelData?[0] {
                result.append(contentsOf: UnsafeBufferPointer(start: channel, count: Int(out.frameLength)))
            }
            guard status == .haveData, out.frameLength > 0 else { break }
        }
        return result
    }
}
