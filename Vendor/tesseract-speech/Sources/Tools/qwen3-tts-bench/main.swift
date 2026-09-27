// qwen3-tts-bench — Qwen3-TTS model-level measurement harness (NOT part of
// the app). Talks to the Qwen3TTS target directly, below the engine, so each
// component can be timed and the process's memory read at every phase.
//
// Usage: qwen3-tts-bench --mode MODE --checkpoint DIR [--seed N] [--out PATH]
//          [--wav-dir DIR] [--repeat N] [--chunk N] [--golden FILE.json]
//          [--hold SECONDS]
//
//   memory   — physical footprint, RSS and MLX active/cache/peak after load,
//              warm-up, a plain generation and a reference-take generation.
//   profile  — talker prefill/step, code-predictor frame and decoder chunk
//              timings with eval barriers, and graph build against run.
//   generate — renders the fixed prompts with a fixed seed and writes their
//              code frames (and per-prompt timings) as JSON: the golden
//              files a refactor is checked against.
//   trace    — teacher-forced talker and code-predictor logits on the golden
//              frames, float32 files in --out, for comparing against Qwen's
//              PyTorch model; trace-unfused does it with the fused kernels off.
//   decode   — decodes the golden frames at several chunk sizes (--wav-dir).
//   neural   — builds the Neural Engine codec and renders with it.
//   memtrace — MLX memory through one generation, plain and with a take.
//   kernels  — each fused kernel on and off: codes compared, time per frame.
//   audit    — every fused kernel call checked against the MLX ops it
//              replaces, teacher-forced and sampled; mismatches to --out.
//   prefill  — prompt prefills as one graph and a layer at a time: time,
//              and what each new length leaves in MLX's pool.
//
// Build with xcodebuild (scheme qwen3-tts-bench) so MLX's metallib lands next
// to the binary.

import AVFoundation
import Darwin
import Foundation
import MLX
import Qwen3TTS

// MARK: - Args

struct Args {
    var mode = "memory"
    var checkpoint = ""
    var seed: UInt64 = 42
    var out: String?
    var wavDir: String?
    var repeatCount = 1
    var chunk = 5
    var hold: Double = 0
    var golden: String?
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
        case "--checkpoint": a.checkpoint = next(flag)
        case "--seed": a.seed = UInt64(next(flag))!
        case "--out": a.out = next(flag)
        case "--wav-dir": a.wavDir = next(flag)
        case "--repeat": a.repeatCount = Int(next(flag))!
        case "--chunk": a.chunk = Int(next(flag))!
        case "--hold": a.hold = Double(next(flag))!
        case "--golden": a.golden = next(flag)
        default: fatalError("unknown flag \(flag)")
        }
    }
    precondition(!a.checkpoint.isEmpty, "--checkpoint DIR is required")
    return a
}

// MARK: - Memory probes

struct MemorySample: Codable {
    var label: String
    var footprintMB: Double
    var lifetimePeakFootprintMB: Double
    var rssMB: Double
    var mlxActiveMB: Double
    var mlxCacheMB: Double
    var mlxPeakMB: Double
}

func physFootprint() -> (current: UInt64, lifetimePeak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    var usage = rusage_info_v4()
    let rr = withUnsafeMutablePointer(to: &usage) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
        }
    }
    return (
        kr == KERN_SUCCESS ? info.phys_footprint : 0,
        rr == 0 ? usage.ri_lifetime_max_phys_footprint : 0
    )
}

func residentSize() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : 0
}

nonisolated(unsafe) var memorySamples: [MemorySample] = []

@MainActor @discardableResult
func probe(_ label: String) -> MemorySample {
    func mb<T: BinaryInteger>(_ b: T) -> Double { Double(b) / 1_048_576 }
    let (footprint, peak) = physFootprint()
    let sample = MemorySample(
        label: label,
        footprintMB: mb(footprint),
        lifetimePeakFootprintMB: mb(peak),
        rssMB: mb(residentSize()),
        mlxActiveMB: mb(Memory.activeMemory),
        mlxCacheMB: mb(Memory.cacheMemory),
        mlxPeakMB: mb(Memory.peakMemory))
    memorySamples.append(sample)
    print(
        String(
            format: "[mem] %-34@ footprint %7.0f MB (peak %7.0f)  rss %7.0f  mlx active %7.0f cache %7.0f peak %7.0f",
            label as NSString, sample.footprintMB, sample.lifetimePeakFootprintMB, sample.rssMB,
            sample.mlxActiveMB, sample.mlxCacheMB, sample.mlxPeakMB))
    return sample
}

// MARK: - Fixtures

let voice = "A calm, warm female narrator with a clear, steady tone."

/// Fixed prompts for golden files: a short companion line, a medium
/// sentence pair, and a long paragraph that runs past the decoder's
/// 72-frame attention window.
let prompts: [(name: String, text: String)] = [
    ("short", "Good morning. Here is the first thing worth knowing today."),
    ("medium",
     "The build finished overnight, and every test came back green. Two of the papers you saved yesterday turned out to be related."),
    ("long",
     "Rain is expected after four, so the afternoon walk should come first. The long-form read you queued is ready whenever you want it, and the notes from yesterday's meeting are waiting in the shared folder. That is everything for now; I will speak up if anything changes, but otherwise the rest of the day is yours."),
]

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func seconds(since t0: UInt64) -> Double { Double(now() - t0) / 1e9 }

struct GenerationRecord: Codable {
    var name: String
    var reference: Bool
    var seconds: Double
    var firstAudioSeconds: Double
    var frames: Int
    var audioSeconds: Double
    var codeFrames: [[Int32]]
}

func render(
    _ model: Qwen3TTSModel, text: String, reference: Qwen3TTSReference?, seed: UInt64
) async throws -> (record: GenerationRecord, samples: [Float]) {
    let t0 = now()
    var firstAudio = -1.0
    var samples: [Float] = []
    var frames: [[Int32]] = []
    let stream = model.generateStream(
        text: text, voice: voice, language: nil, reference: reference,
        sampling: Qwen3TTSSampling(), seed: seed, streamingInterval: 0.4)
    for try await event in stream {
        switch event {
        case .audio(let audio):
            if firstAudio < 0 { firstAudio = seconds(since: t0) }
            samples.append(contentsOf: audio)
        case .codeFrames(let f):
            frames = f
        }
    }
    let elapsed = seconds(since: t0)
    let audioSeconds = Double(samples.count) / Double(model.sampleRate)
    return (
        GenerationRecord(
            name: "", reference: reference != nil, seconds: elapsed,
            firstAudioSeconds: firstAudio, frames: frames.count, audioSeconds: audioSeconds,
            codeFrames: frames),
        samples
    )
}

func writeWav(_ samples: [Float], sampleRate: Int, to url: URL) throws {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
        interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer {
        buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
    }
    let file = try AVAudioFile(
        forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32,
        interleaved: false)
    try file.write(from: buffer)
}

func stats(_ xs: [Double]) -> String {
    guard !xs.isEmpty else { return "n/a" }
    let s = xs.sorted()
    let mean = xs.reduce(0, +) / Double(xs.count)
    return String(
        format: "mean %.2f ms  p50 %.2f  p90 %.2f  min %.2f (n=%d)",
        mean * 1e3, s[s.count / 2] * 1e3, s[Int(Double(s.count) * 0.9)] * 1e3, s[0] * 1e3,
        xs.count)
}

// MARK: - Main

let args = parseArgs()
let checkpoint = URL(fileURLWithPath: args.checkpoint, isDirectory: true)

probe("start")

if args.mode == "decode" {
    // The codec decoder alone on the golden frames: fp32 and fp16, several
    // chunk sizes. Writes <out>/<name>.<dtype>.c<chunk>.f32 for parity checks.
    guard let golden = args.golden, let out = args.out else {
        fatalError("decode needs --golden FILE.json and --out DIR")
    }
    struct Golden: Codable { var records: [GenerationRecord] }
    let records = try JSONDecoder().decode(
        Golden.self, from: Data(contentsOf: URL(fileURLWithPath: golden))
    ).records
    let outDir = URL(fileURLWithPath: out, isDirectory: true)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let directory = checkpoint.appendingPathComponent("speech_tokenizer")
    for dtype in [DType.float32, .float16] {
        let t0 = now()
        let decoder = try Qwen3TTSCodecDecoder(directory: directory, dtype: dtype)
        print(String(format: "decoder %@ loaded in %.2f s", "\(dtype)", seconds(since: t0)))
        probe("decoder \(dtype) loaded")
        for name in ["short", "medium", "long", "medium+take"] {
            guard let record = records.first(where: { $0.name == name }) else { continue }
            let groups = record.codeFrames[0].count
            let all = MLXArray(record.codeFrames.flatMap { $0 }).reshaped(
                1, record.codeFrames.count, groups)
            for chunk in [1, 5, 25, 1000] {
                var stream = decoder.makeStream()
                var samples: [Float] = []
                var times: [Double] = []
                var start = 0
                while start < record.frames {
                    let end = min(start + chunk, record.frames)
                    let codes = all[0..., start ..< end, 0...]
                    eval(codes)
                    let t = now()
                    let audio = try decoder.decode(codes, stream: &stream)
                    eval([audio] + stream.arrays)
                    times.append(seconds(since: t) / Double(end - start))
                    samples.append(contentsOf: audio.asArray(Float.self))
                    start = end
                }
                try samples.withUnsafeBufferPointer { Data(buffer: $0) }.write(
                    to: outDir.appendingPathComponent("\(name).\(dtype).c\(chunk).f32"))
                if name == "long" {
                    print("  \(dtype) chunk \(chunk) per frame: \(stats(Array(times.dropFirst(min(2, times.count - 1)))))")
                }
            }
        }
        probe("decoder \(dtype) done")
    }
    exit(0)
}

let loadStart = now()
let model = try await Qwen3TTSModel.fromModelDirectory(checkpoint)
print(String(format: "load %.2f s", seconds(since: loadStart)))
probe("after load")

_ = try await model.generate(
    text: ".", voice: nil, language: "English", sampling: Qwen3TTSSampling(maxTokens: 3))
probe("after warm-up")

switch args.mode {
case "memory":
    Memory.peakMemory = 0  // resets the MLX peak
    let plain = try await render(model, text: prompts[2].text, reference: nil, seed: args.seed)
    print(
        String(
            format: "plain: %d frames, %.1f s audio in %.2f s (RTF %.3f), first audio %.0f ms",
            plain.record.frames, plain.record.audioSeconds, plain.record.seconds,
            plain.record.seconds / plain.record.audioSeconds, plain.record.firstAudioSeconds * 1e3))
    probe("after plain generation")

    Memory.peakMemory = 0  // resets the MLX peak
    let take = Qwen3TTSReference(codeFrames: plain.record.codeFrames, text: prompts[2].text)
    let icl = try await render(model, text: prompts[1].text, reference: take, seed: args.seed)
    print(
        String(
            format: "reference: %d frames, %.1f s audio in %.2f s (RTF %.3f), first audio %.0f ms",
            icl.record.frames, icl.record.audioSeconds, icl.record.seconds,
            icl.record.seconds / icl.record.audioSeconds, icl.record.firstAudioSeconds * 1e3))
    probe("after reference generation")

    // Four more segments continuing the take, as a read-aloud does.
    for i in 0 ..< 4 {
        Memory.peakMemory = 0  // resets the MLX peak
        let segment = try await render(
            model, text: prompts[i % 3].text, reference: take, seed: args.seed + UInt64(i))
        print(String(format: "segment %d: %d frames, first audio %.0f ms", i, segment.record.frames, segment.record.firstAudioSeconds * 1e3))
        probe("after segment \(i)")
    }

    Memory.clearCache()
    probe("after clearCache")

case "profile":
    for length in [60, 150, 400] {
        let talker = model.benchTalker(promptLength: length, steps: 40)
        print("talker prefill (\(length) positions): \(stats(talker.prefill))")
        if length == 400 {
            print("talker step (cache ~400): \(stats(Array(talker.talkerStep.dropFirst(3))))")
            print("code predictor frame:     \(stats(Array(talker.codePredictorFrame.dropFirst(3))))")
        }
    }

    let split = model.benchBuildVersusRun(steps: 40)
    print("talker step build: \(stats(Array(split.talkerBuild.dropFirst(3))))")
    print("talker step run:   \(stats(Array(split.talkerRun.dropFirst(3))))")
    print("cp frame build:    \(stats(Array(split.cpBuild.dropFirst(3))))")
    print("cp frame run:      \(stats(Array(split.cpRun.dropFirst(3))))")

    let plain = try await render(model, text: prompts[2].text, reference: nil, seed: args.seed)
    for chunk in [1, 5, 25] {
        let decoded = try model.benchStreamingDecode(codeFrames: plain.record.codeFrames, chunk: chunk)
        let perFrame = decoded.stepSeconds.map { $0 / Double(chunk) }
        print("decoder chunk \(chunk) (per frame): \(stats(Array(perFrame.dropFirst(2))))")
    }
    probe("after profile")

case "generate":
    var records: [GenerationRecord] = []
    for repeatIndex in 0 ..< args.repeatCount {
        var previous: GenerationRecord?
        for (name, text) in prompts {
            let (record, samples) = try await render(
                model, text: text, reference: nil, seed: args.seed)
            var named = record
            named.name = name
            records.append(named)
            print(
                String(
                    format: "%@ #%d: %d frames, %.1f s audio, %.2f s (RTF %.3f), first audio %.0f ms",
                    name as NSString, repeatIndex, record.frames, record.audioSeconds,
                    record.seconds, record.seconds / max(record.audioSeconds, 1e-9),
                    record.firstAudioSeconds * 1e3))
            if let wavDir = args.wavDir, repeatIndex == 0 {
                try writeWav(
                    samples, sampleRate: model.sampleRate,
                    to: URL(fileURLWithPath: wavDir).appendingPathComponent("\(name).wav"))
            }
            previous = named
        }
        // One reference-take continuation: the long render conditions the
        // medium text.
        if let previous {
            let take = Qwen3TTSReference(codeFrames: previous.codeFrames, text: prompts[2].text)
            let (record, samples) = try await render(
                model, text: prompts[1].text, reference: take, seed: args.seed)
            var named = record
            named.name = "medium+take"
            records.append(named)
            print(
                String(
                    format: "medium+take #%d: %d frames, %.1f s audio, %.2f s (RTF %.3f), first audio %.0f ms",
                    repeatIndex, record.frames, record.audioSeconds, record.seconds,
                    record.seconds / max(record.audioSeconds, 1e-9), record.firstAudioSeconds * 1e3))
            if let wavDir = args.wavDir, repeatIndex == 0 {
                try writeWav(
                    samples, sampleRate: model.sampleRate,
                    to: URL(fileURLWithPath: wavDir).appendingPathComponent("medium+take.wav"))
            }
        }
    }
    if let out = args.out {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(["records": records]).write(to: URL(fileURLWithPath: out))
    }
    probe("after generate")

case "neural":
    // The Neural Engine codec: build or load it, decode the golden frames
    // (parity against Qwen's decoder), then time generations with it.
    let cache = URL(fileURLWithPath: args.out ?? NSTemporaryDirectory())
        .appendingPathComponent("ane-cache", isDirectory: true)
    let t0 = now()
    let report = try await model.prepareNeuralEngine(cacheDirectory: cache, frames: args.chunk)
    print(String(format: "neural codec ready in %.2f s: %@", seconds(since: t0), report))
    probe("neural codec loaded")
    if let golden = args.golden {
        struct Golden: Codable { var records: [GenerationRecord] }
        let records = try JSONDecoder().decode(
            Golden.self, from: Data(contentsOf: URL(fileURLWithPath: golden))
        ).records
        let outDir = URL(fileURLWithPath: args.out!).appendingPathComponent("decode-ane")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        for name in ["short", "medium", "long", "medium+take"] {
            guard let record = records.first(where: { $0.name == name }) else { continue }
            let (samples, times) = try model.benchNeuralDecode(codeFrames: record.codeFrames)
            try samples.withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: outDir.appendingPathComponent("\(name).ane.f32"))
            print("  \(name): \(record.frames) frames, ANE call \(stats(Array(times.dropFirst())))")
        }
    }
    for repeatIndex in 0 ..< 2 {
        for (name, text) in prompts {
            let (record, samples) = try await render(model, text: text, reference: nil, seed: args.seed)
            print(
                String(
                    format: "ANE %@ #%d: %d frames, %.1f s audio, %.2f s (RTF %.3f), first audio %.0f ms",
                    name as NSString, repeatIndex, record.frames, record.audioSeconds, record.seconds,
                    record.seconds / max(record.audioSeconds, 1e-9), record.firstAudioSeconds * 1e3))
            if let wavDir = args.wavDir, repeatIndex == 0 {
                try writeWav(
                    samples, sampleRate: model.sampleRate,
                    to: URL(fileURLWithPath: wavDir).appendingPathComponent("ane-\(name).wav"))
            }
        }
    }
    probe("after neural generations")

case "kernels":
    // The fused kernels against the MLX ops they replace: same seed, codes
    // compared, time per frame.
    let configurations: [(String, Bool, Bool, Bool, Bool)] = [
        ("off", false, false, false, false), ("sampler", true, true, false, false),
        ("normRoPE", true, false, true, false), ("addNorm", true, false, false, true),
        ("all", true, true, true, true),
    ]
    var reference: [[[Int32]]]?
    for (label, enabled, sampler, normRoPE, addNorm) in configurations {
        model.setFusedKernels(enabled, sampler: sampler, normRoPE: normRoPE, addNorm: addNorm)
        var frames = 0
        var seconds = 0.0
        var codes: [[[Int32]]] = []
        for (_, text) in prompts {
            let (record, _) = try await render(model, text: text, reference: nil, seed: args.seed)
            frames += record.frames
            seconds += record.seconds
            codes.append(record.codeFrames)
        }
        if reference == nil { reference = codes }
        print(String(format: "fused %@: %d frames, %.2f ms per frame (RTF %.3f), codes %@",
                     label as NSString, frames, seconds / Double(frames) * 1e3,
                     seconds / (Double(frames) * 0.08),
                     codes == reference! ? "identical" : "DIFFER"))
    }

case "memtrace":
    let plain = try await render(model, text: prompts[2].text, reference: nil, seed: args.seed)
    Memory.clearCache()
    Memory.peakMemory = 0  // resets the MLX peak
    let take = Qwen3TTSReference(codeFrames: plain.record.codeFrames, text: prompts[2].text)
    print("-- plain")
    try model.benchMemoryTrace(text: prompts[1].text, voice: voice, reference: nil, frames: 100, every: 25) { print("  \($0)") }
    print("-- reference")
    try model.benchMemoryTrace(text: prompts[1].text, voice: voice, reference: take, frames: 100, every: 25) { print("  \($0)") }

case "prefill":
    // Prefills of new lengths as one graph and a layer at a time: time, and
    // what they leave in MLX's pool.
    let lengths = [40, 150, 310, 365, 420, 505, 590, 640]
    for layerByLayer in [false, true] {
        print("-- \(layerByLayer ? "a layer at a time" : "one graph")")
        model.benchPrefill(lengths: lengths, layerByLayer: layerByLayer).forEach { print("  \($0)") }
    }

case "audit":
    // Every fused kernel call against the MLX ops it replaces: teacher-forced
    // runs on the golden frames, then sampled generations.
    guard let golden = args.golden else { fatalError("audit needs --golden FILE.json") }
    struct Golden: Codable { var records: [GenerationRecord] }
    let records = try JSONDecoder().decode(
        Golden.self, from: Data(contentsOf: URL(fileURLWithPath: golden))
    ).records
    let dump = args.out.map { URL(fileURLWithPath: $0, isDirectory: true) }
    if let dump { try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true) }
    for name in ["short", "medium", "medium+take"] {
        guard let record = records.first(where: { $0.name == name }) else { continue }
        let text = prompts.first { name.hasPrefix($0.name) }!.text
        let long = records.first { $0.name == "long" }!
        let reference = name == "medium+take"
            ? Qwen3TTSReference(codeFrames: long.codeFrames, text: prompts[2].text) : nil
        let report = try await model.benchAudit(dump: name == "short" ? dump : nil) {
            _ = try model.benchTeacherForced(
                text: text, voice: voice, language: nil, reference: reference,
                codeFrames: record.codeFrames)
        }
        print("audit teacher-forced \(name): \(report)")
    }
    for (name, text) in prompts.prefix(2) {
        let report = try await model.benchAudit(dump: nil) {
            _ = try await render(model, text: text, reference: nil, seed: args.seed)
        }
        print("audit generated \(name): \(report)")
    }

case "trace", "trace-unfused":
    if args.mode == "trace-unfused" { model.setFusedKernels(false) }
    // Teacher-forced logits on the golden frames, for numerical comparison
    // across implementations: <out>/<name>.talker.f32 and .detail.f32.
    guard let golden = args.golden, let out = args.out else {
        fatalError("trace needs --golden FILE.json and --out DIR")
    }
    struct Golden: Codable { var records: [GenerationRecord] }
    let records = try JSONDecoder().decode(
        Golden.self, from: Data(contentsOf: URL(fileURLWithPath: golden))
    ).records
    let outDir = URL(fileURLWithPath: out, isDirectory: true)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let long = records.first { $0.name == "long" }!
    for name in ["short", "medium", "long", "medium+take"] {
        guard let record = records.first(where: { $0.name == name }) else { continue }
        let text = prompts.first { name.hasPrefix($0.name) }!.text
        let reference = name == "medium+take"
            ? Qwen3TTSReference(codeFrames: long.codeFrames, text: prompts[2].text) : nil
        let t0 = now()
        let (talkerRows, detailRows) = try model.benchTeacherForced(
            text: text, voice: voice, language: nil, reference: reference,
            codeFrames: record.codeFrames)
        func write(_ rows: [Float], _ suffix: String) throws {
            try rows.withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: outDir.appendingPathComponent("\(name).\(suffix).f32"))
        }
        try write(talkerRows, "talker")
        try write(detailRows, "detail")
        print(String(format: "trace %@: %d frames in %.2f s", name as NSString, record.frames, seconds(since: t0)))
    }

default:
    fatalError("unknown mode \(args.mode)")
}

if args.hold > 0 {
    print("holding \(args.hold) s, pid \(getpid())")
    fflush(stdout)
    try await Task.sleep(nanoseconds: UInt64(args.hold * 1e9))
}

if let out = args.out, !["generate", "trace", "trace-unfused", "neural", "decode", "audit", "prefill"].contains(args.mode) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(["memory": memorySamples]).write(to: URL(fileURLWithPath: out))
}
exit(0)
