// v2-listen — engine v2 listening-artifact + measurement harness.
// NOT part of the app. Drives the production stack (SpeechEngine actor →
// Qwen3Synthesizer → Qwen3TTS) against real weights and writes:
//   pinned    — 6 utterances in one pinned-voice session + a 7th from a
//               serialized/restored PinnedVoice (voice-consistency listen)
//   longform  — one long read-aloud utterance (multi-segment); reports
//               per-segment TTFA, wall, RTF, and peak RSS (the ADR-0037 gate)
//   ab        — #339-matched settings (seed 42, t=0.9 for both models,
//               p=1.0, rp=1.05) for same-seed A/B against
//               research/model-bench-339/audio WAVs
//
// Usage: v2-listen --mode pinned|longform|ab [--precision 8bit|6bit|bf16]
//          [--checkpoint DIR] [--out-dir DIR] [--text-file PATH] [--seed N]
//          [--temperature T] [--detail-temperature T] [--voice DESCRIPTION]
//          [--reference pinned|none] [--neural-engine on|off]
//
// Loads the checkpoint the app downloaded (Application Support/models) unless
// --checkpoint names another directory; it never downloads. Build with
// xcodebuild (scheme v2-listen) so MLX's metallib lands next to the binary.

import AVFoundation
import Foundation
import TesseractSpeech

// MARK: - Args

struct Args {
    var mode = "pinned"
    var precision = "8bit"
    var outDir = "."
    var textFile: String?
    var checkpoint: String?
    var seed: UInt64 = 42
    var temperature: Float?
    var detailTemperature: Float?
    var voice: String?
    var reference = "pinned"
    var timing = false
    var neuralEngine = true
}

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    func next(_ flag: String) -> String {
        guard let v = it.next() else { fatalError("missing value for \(flag)") }
        return v
    }
    while let flag = it.next() {
        switch flag {
        case "--mode": a.mode = next(flag)
        case "--precision": a.precision = next(flag)
        case "--out-dir": a.outDir = next(flag)
        case "--text-file": a.textFile = next(flag)
        case "--checkpoint": a.checkpoint = next(flag)
        case "--temperature": a.temperature = Float(next(flag))!
        case "--detail-temperature": a.detailTemperature = Float(next(flag))!
        case "--voice": a.voice = next(flag)
        case "--reference": a.reference = next(flag)
        case "--seed": a.seed = UInt64(next(flag))!
        case "--timing": a.timing = true
        case "--neural-engine": a.neuralEngine = next(flag) != "off"
        default: fatalError("unknown flag \(flag)")
        }
    }
    return a
}

// MARK: - Helpers

struct ImmediateLease: GPULeasing {
    func withLease<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        try await body()
    }
}

/// Prints engine burst events with a monotonic timestamp so per-segment burst
/// wall and inter-segment gaps can be attributed (perf pass, spec §6).
struct StderrTimingTap: SpeechDiagnosticsTap {
    let epoch = DispatchTime.now()
    func event(_ name: StaticString, _ detail: @autoclosure @Sendable () -> String) {
        let t = Double(DispatchTime.now().uptimeNanoseconds - epoch.uptimeNanoseconds) / 1e3
        fputs(String(format: "[timing] %10.0fµs %@ — %@\n", t, "\(name)", detail()), stderr)
    }
}

func peakRSSGB() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_maxrss) / 1e9
}

func spec(for precision: String) -> TTSModelSpec {
    switch precision {
    case "8bit": return .voiceDesign17B(.q8)
    case "6bit": return .voiceDesign17B(.q6)
    case "bf16": return .voiceDesign17B(.bf16)
    default: fatalError("unknown precision \(precision)")
    }
}

struct UtteranceCapture {
    var samples: [Float] = []
    var sampleRate = 24_000
    var ttfaMs: Double = -1
    var wallSec: Double = 0
    var segmentTTFAsMs: [Double] = []
    var segmentCount = 0
    /// Where each segment starts in `samples`, for per-segment listening and
    /// scoring.
    var segmentStarts: [Int] = []
    /// Each segment's text and first frame, and its words' starts (ADR-0077),
    /// for scoring the word timing against a transcript.
    var scripts: [SegmentScript] = []
    var wordStarts: [Int: [WordStart]] = [:]
}

func drain(_ utterance: Utterance) async throws -> UtteranceCapture {
    var capture = UtteranceCapture()
    capture.sampleRate = utterance.sampleRate
    capture.segmentCount = utterance.segmentCount
    let t0 = DispatchTime.now()
    var segmentStart = t0
    var sawAudioForSegment = false
    for try await event in utterance.events {
        switch event {
        case .segment(let script):
            segmentStart = DispatchTime.now()
            sawAudioForSegment = false
            capture.scripts.append(script)
        case .words(let timing):
            capture.wordStarts[timing.segmentIndex, default: []] += timing.starts
        case .audio(let chunk):
            let now = DispatchTime.now()
            if capture.ttfaMs < 0 {
                capture.ttfaMs =
                    Double(now.uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            }
            if !sawAudioForSegment {
                capture.segmentTTFAsMs.append(
                    Double(now.uptimeNanoseconds - segmentStart.uptimeNanoseconds) / 1e6)
                sawAudioForSegment = true
            }
            while capture.segmentStarts.count <= chunk.segmentIndex {
                capture.segmentStarts.append(capture.samples.count)
            }
            capture.samples.append(contentsOf: chunk.samples)
        case .segmentDone, .finished:
            break
        }
    }
    capture.wallSec =
        Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
    return capture
}

/// Mono float32 WAV through AVAudioFile.
func writeWav(samples: ArraySlice<Float>, sampleRate: Int, to url: URL) throws {
    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
            interleaved: false),
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
    else {
        throw CocoaError(.fileWriteUnknown)
    }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { source in
        buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
    }
    let file = try AVAudioFile(
        forWriting: url, settings: format.settings, commonFormat: format.commonFormat,
        interleaved: format.isInterleaved)
    try file.write(from: buffer)
}

func write(_ capture: UtteranceCapture, to url: URL, label: String) throws {
    try writeWav(samples: capture.samples[...], sampleRate: capture.sampleRate, to: url)
    if capture.segmentStarts.count > 1 {
        let stem = url.deletingPathExtension().lastPathComponent
        let ends = capture.segmentStarts.dropFirst() + [capture.samples.count]
        for (index, (start, end)) in zip(capture.segmentStarts, ends).enumerated() {
            try writeWav(
                samples: capture.samples[start..<end], sampleRate: capture.sampleRate,
                to: url.deletingLastPathComponent().appendingPathComponent(
                    String(format: "%@_seg%02d.wav", stem, index + 1)))
        }
    }
    if !capture.wordStarts.isEmpty {
        // Per segment: its text, and each word's start in frames from the
        // segment's own first frame (its WAV's time 0).
        let segments: [[String: Any]] = capture.scripts.map { script in
            [
                "index": script.index, "text": script.text, "startFrame": script.startFrame,
                "words": (capture.wordStarts[script.index] ?? []).map {
                    ["word": $0.word, "frame": $0.frame - script.startFrame]
                },
            ]
        }
        let stem = url.deletingPathExtension().lastPathComponent
        try JSONSerialization.data(withJSONObject: ["segments": segments], options: [.prettyPrinted])
            .write(to: url.deletingLastPathComponent().appendingPathComponent("\(stem)_words.json"))
    }
    let audioSec = Double(capture.samples.count) / Double(capture.sampleRate)
    let rtf = audioSec > 0 ? capture.wallSec / audioSec : -1
    let segTTFAs = capture.segmentTTFAsMs.map { String(format: "%.0f", $0) }
        .joined(separator: ",")
    print(
        "\(label): \(url.lastPathComponent) segments=\(capture.segmentCount) "
            + "audio=\(String(format: "%.1f", audioSec))s "
            + "wall=\(String(format: "%.1f", capture.wallSec))s "
            + "rtf=\(String(format: "%.3f", rtf)) "
            + "ttfa=\(String(format: "%.0f", capture.ttfaMs))ms "
            + "segTTFAs=[\(segTTFAs)]ms "
            + "peakRSS=\(String(format: "%.2f", peakRSSGB()))GB")
}

// MARK: - Main

let args = parseArgs()
let outDir = URL(fileURLWithPath: args.outDir, isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

// The engine only loads from disk: the app's model store, or --checkpoint.
let modelSpec = spec(for: args.precision)
let checkpoint =
    args.checkpoint.map { URL(fileURLWithPath: $0, isDirectory: true) }
    ?? URL.applicationSupportDirectory
    .appendingPathComponent("models")
    .appendingPathComponent(modelSpec.repo.replacingOccurrences(of: "/", with: "_"))

let synthesizer = Qwen3Synthesizer(
    checkpointDirectory: { [checkpoint] _ in checkpoint },
    neuralEngineCache: args.neuralEngine ? Qwen3Synthesizer.defaultNeuralEngineCache : nil)
let engine = SpeechEngine(
    model: modelSpec,
    synthesizer: synthesizer,
    gpu: ImmediateLease(),
    diagnostics: args.timing ? StderrTimingTap() : nil
)
// Load and warm up, then wait for the Neural Engine codec (built in the
// background on first use), so every rendering below uses one decoder.
try await engine.prepare(.warm)
print("decoder: \(await synthesizer.neuralEngineReady() ?? "MLX")")

let narrator = args.voice ?? "A calm, warm female narrator with a clear, steady tone."

/// The engine defaults, with any sampler overrides from the command line.
var listenParameters: TTSParameters {
    var parameters = TTSParameters()
    if let t = args.temperature { parameters.temperature = t }
    if let t = args.detailTemperature { parameters.detailTemperature = t }
    return parameters
}
let referencePolicy: ReferencePolicy = args.reference == "none" ? .none : .pinned

do {
    switch args.mode {
    case "pinned":
        // Six lines, one pinned session: the voice-consistency listen.
        let lines = [
            "Good morning. Here is the first thing worth knowing today.",
            "The build finished overnight, and every test came back green.",
            "Two of the papers you saved yesterday turned out to be related.",
            "Rain is expected after four, so the afternoon walk should come first.",
            "The long-form read you queued is ready whenever you want it.",
            "That is everything for now; I will speak up if anything changes.",
        ]
        let session = try await engine.session(
            SessionProfile(reference: referencePolicy, pacing: .eager),
            voice: .designed(description: narrator, language: nil))
        for (index, line) in lines.enumerated() {
            let utterance = try await session.speak(
                line, options: SpeechOptions(seed: .fixed(args.seed), parameters: listenParameters))
            let capture = try await drain(utterance)
            try write(
                capture,
                to: outDir.appendingPathComponent(
                    String(format: "pinned_%02d.wav", index + 1)),
                label: "pinned \(index + 1)/\(lines.count)")
        }

        // Serialize → restore → speak: the PinnedVoice relaunch guarantee.
        guard let pinned = await session.exportPinnedVoice() else {
            fatalError("no PinnedVoice exported after six utterances")
        }
        let data = try pinned.serialized()
        print("exported PinnedVoice: \(data.count) bytes")
        await session.close()

        let restored = try PinnedVoice(validating: data)
        let restoredSession = try await engine.session(.companion, voice: .pinned(restored))
        let utterance = try await restoredSession.speak(
            "And this line comes from the restored voice, after a relaunch.",
            options: SpeechOptions(seed: .fixed(args.seed), parameters: listenParameters))
        let capture = try await drain(utterance)
        try write(
            capture,
            to: outDir.appendingPathComponent("pinned_07_restored.wav"),
            label: "pinned restored")
        await restoredSession.close()

    case "longform":
        guard let textFile = args.textFile else {
            fatalError("longform needs --text-file")
        }
        let text = try String(contentsOfFile: textFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let session = try await engine.session(
            SessionProfile(reference: referencePolicy, pacing: .lookahead(segments: 1)),
            voice: .designed(description: narrator, language: nil))
        let utterance = try await session.speak(
            text, options: SpeechOptions(seed: .fixed(args.seed), parameters: listenParameters))
        let capture = try await drain(utterance)
        try write(
            capture,
            to: outDir.appendingPathComponent("longform_\(args.precision).wav"),
            label: "longform \(args.precision)")
        await session.close()

    case "ab":
        // Match #339 bench settings exactly (seed 42, t=0.9/p=1.0/rp=1.05)
        // so the WAV is same-seed comparable to vd_<precision>_short/long.
        guard let textFile = args.textFile else {
            fatalError("ab needs --text-file (research/model-bench-339/texts/…)")
        }
        let text = try String(contentsOfFile: textFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // #339 ran both models at 0.9, before ADR-0072 split the temperature;
        // the rest are the engine defaults.
        let benchParams = TTSParameters(detailTemperature: 0.9)
        let session = try await engine.session(
            SessionProfile(reference: referencePolicy, pacing: .eager),
            voice: .designed(description: narrator, language: nil))
        let utterance = try await session.speak(
            text,
            options: SpeechOptions(seed: .fixed(args.seed), parameters: benchParams))
        let capture = try await drain(utterance)
        let stem = URL(fileURLWithPath: textFile).deletingPathExtension().lastPathComponent
        try write(
            capture,
            to: outDir.appendingPathComponent("ab_v2_\(args.precision)_\(stem).wav"),
            label: "ab \(args.precision) \(stem)")
        await session.close()

    default:
        fatalError("unknown mode \(args.mode)")
    }

    await engine.unload()
    print("DONE peakRSS=\(String(format: "%.2f", peakRSSGB()))GB")
    exit(0)
} catch {
    fputs("v2-listen FAIL: \(error)\n", stderr)
    exit(1)
}
