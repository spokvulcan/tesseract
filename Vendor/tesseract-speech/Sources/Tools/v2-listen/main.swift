// v2-listen — engine v2 listening-artifact + measurement harness.
// NOT part of the app. Drives the production stack (SpeechEngine actor →
// Qwen3Synthesizer → re-vendored MLXAudioTTS) against real weights and writes:
//   pinned    — 6 utterances in one pinned-voice session + a 7th from a
//               serialized/restored PinnedVoice (anchored-consistency listen)
//   longform  — one long read-aloud utterance (multi-segment); reports
//               per-segment TTFA, wall, RTF, and peak RSS (the ADR-0037 gate)
//   ab        — #339-matched settings (seed 42, t=0.9/p=1.0/rp=1.05) for
//               same-seed A/B against research/model-bench-339/audio WAVs
//
// Usage: swift run -c release v2-listen --mode pinned|longform|ab \
//          [--precision 8bit|6bit|bf16] [--out-dir DIR] [--text-file PATH] [--seed N]

import Foundation
import MLXAudioCore
import TesseractSpeech

// MARK: - Args

struct Args {
    var mode = "pinned"
    var precision = "8bit"
    var outDir = "."
    var textFile: String?
    var seed: UInt64 = 42
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
        case "--seed": a.seed = UInt64(next(flag))!
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
        case .segment:
            segmentStart = DispatchTime.now()
            sawAudioForSegment = false
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
            capture.samples.append(contentsOf: chunk.samples)
        case .segmentDone, .finished:
            break
        }
    }
    capture.wallSec =
        Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
    return capture
}

func write(_ capture: UtteranceCapture, to url: URL, label: String) throws {
    try AudioUtils.writeWavFile(
        samples: capture.samples, sampleRate: capture.sampleRate, fileURL: url)
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

let engine = SpeechEngine(
    model: spec(for: args.precision),
    synthesizer: Qwen3Synthesizer(),
    gpu: ImmediateLease()
)

let narrator = "A calm, warm female narrator with a clear, steady tone."

do {
    switch args.mode {
    case "pinned":
        // Six lines, one pinned session: the anchored-consistency listen.
        let lines = [
            "Good morning. Here is the first thing worth knowing today.",
            "The build finished overnight, and every test came back green.",
            "Two of the papers you saved yesterday turned out to be related.",
            "Rain is expected after four, so the afternoon walk should come first.",
            "The long-form read you queued is ready whenever you want it.",
            "That is everything for now; I will speak up if anything changes.",
        ]
        let session = try await engine.session(
            .companion, voice: .designed(description: narrator, language: nil))
        for (index, line) in lines.enumerated() {
            let utterance = try await session.speak(
                line, options: SpeechOptions(seed: .fixed(args.seed)))
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
            options: SpeechOptions(seed: .fixed(args.seed)))
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
            .readAloud, voice: .designed(description: narrator, language: nil))
        let utterance = try await session.speak(
            text, options: SpeechOptions(seed: .fixed(args.seed)))
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
        let benchParams = TTSParameters(
            temperature: 0.9, topP: 1.0, repetitionPenalty: 1.05, maxTokens: 4096)
        let session = try await engine.session(
            SessionProfile(anchor: .perUtterance(), pacing: .eager),
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
