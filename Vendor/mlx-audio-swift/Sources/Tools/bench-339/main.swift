// bench(339) — candidate-model benchmark harness. NOT part of the port.
// One model per process. Measures load time, TTFA (first streamed audio chunk),
// steps/s, RTF, MLX peak GPU memory, and peak RSS; writes WAV artifacts and
// appends one JSON line per rep to a results file.
//
// Modes:
//   design       VoiceDesign / instruct prompt (--voice = description)
//   clone-icl    Base full in-context cloning (--ref-audio + --ref-text)
//   clone-xvec   Base x-vector-only cloning (--ref-audio)
//   customvoice  CustomVoice named speaker (--voice = "speaker, instruction")

import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

struct Args {
    var model = ""
    var mode = "design"
    var text = ""
    var voice: String?
    var refAudioPath: String?
    var compareAudioPath: String?
    var refText: String?
    var seed: UInt64 = 42
    var reps = 1
    var interval = 2.0
    var maxTokens = 4096
    var out: String?
    var json: String?
    var label = ""
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
        case "--model": a.model = next(flag)
        case "--mode": a.mode = next(flag)
        case "--text": a.text = next(flag)
        case "--text-file": a.text = try! String(contentsOfFile: next(flag), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        case "--voice": a.voice = next(flag)
        case "--ref-audio": a.refAudioPath = next(flag)
        case "--compare-audio": a.compareAudioPath = next(flag)
        case "--ref-text": a.refText = next(flag)
        case "--seed": a.seed = UInt64(next(flag))!
        case "--reps": a.reps = Int(next(flag))!
        case "--interval": a.interval = Double(next(flag))!
        case "--max-tokens": a.maxTokens = Int(next(flag))!
        case "--out": a.out = next(flag)
        case "--json": a.json = next(flag)
        case "--label": a.label = next(flag)
        default: fatalError("unknown flag \(flag)")
        }
    }
    precondition(!a.model.isEmpty && !a.text.isEmpty, "--model and --text are required")
    return a
}

func now() -> DispatchTime { DispatchTime.now() }
func secs(since t: DispatchTime) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e9
}

func peakRSSGB() -> Double {
    var ru = rusage()
    getrusage(RUSAGE_SELF, &ru)
    return Double(ru.ru_maxrss) / 1e9 // bytes on macOS
}

// Waveform sanity stats for machine-level quality checks.
struct AudioStats {
    let seconds: Double
    let rms: Double
    let peak: Double
    let silentWindowFraction: Double // fraction of 100 ms windows under -55 dBFS RMS
    let longestSilenceMs: Double // longest internal silent run (excludes tail)
    let clippedFraction: Double // fraction of samples at |x| >= 0.999
}

func analyze(_ s: [Float], sampleRate: Int) -> AudioStats {
    guard !s.isEmpty else {
        return AudioStats(seconds: 0, rms: 0, peak: 0, silentWindowFraction: 1, longestSilenceMs: 0, clippedFraction: 0)
    }
    let n = s.count
    var sumSq = 0.0, peak = 0.0
    var clipped = 0
    for x in s {
        let d = Double(x)
        sumSq += d * d
        peak = max(peak, abs(d))
        if abs(d) >= 0.999 { clipped += 1 }
    }
    let win = sampleRate / 10 // 100 ms
    var silentWindows = 0, totalWindows = 0
    var windowIsSilent = [Bool]()
    var i = 0
    while i < n {
        let end = min(i + win, n)
        var wSum = 0.0
        for j in i ..< end { wSum += Double(s[j]) * Double(s[j]) }
        let wRms = (wSum / Double(end - i)).squareRoot()
        let silent = wRms < 0.00178 // -55 dBFS
        windowIsSilent.append(silent)
        if silent { silentWindows += 1 }
        totalWindows += 1
        i = end
    }
    // longest internal silent run: strip trailing silence first
    var trimmed = windowIsSilent
    while trimmed.last == true { trimmed.removeLast() }
    var longest = 0, run = 0
    for silent in trimmed {
        run = silent ? run + 1 : 0
        longest = max(longest, run)
    }
    return AudioStats(
        seconds: Double(n) / Double(sampleRate),
        rms: (sumSq / Double(n)).squareRoot(),
        peak: peak,
        silentWindowFraction: totalWindows > 0 ? Double(silentWindows) / Double(totalWindows) : 1,
        longestSilenceMs: Double(longest) * 100.0,
        clippedFraction: Double(clipped) / Double(n)
    )
}

func writeWAV(_ samples: [Float], sampleRate: Int, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true
    )
    try? FileManager.default.removeItem(at: url)
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
        channels: 1, interleaved: false
    ), let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
        fatalError("audio buffer alloc failed")
    }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { src in
        buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
    }
    let file = try AVAudioFile(
        forWriting: url, settings: format.settings,
        commonFormat: .pcmFormatFloat32, interleaved: false
    )
    try file.write(from: buffer)
}

func appendJSON(_ obj: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    let line = String(data: data, encoding: .utf8)! + "\n"
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile()
        fh.write(line.data(using: .utf8)!)
        fh.closeFile()
    } else {
        try line.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

let args = parseArgs()

do {
    let modelDir = URL(fileURLWithPath: args.model)
    Memory.peakMemory = 0
    let tLoad = now()
    let model = try await Qwen3TTSModel.fromModelDirectory(modelDir)
    let loadSec = secs(since: tLoad)

    // simcheck: cosine similarity between the speaker embeddings (x-vectors) of two
    // clips, using this (Base) model's speaker encoder as the judge. A repeatable
    // proxy for cloning fidelity; scores are comparative, not absolute.
    if args.mode == "simcheck" {
        guard let refPath = args.refAudioPath, let cmpPath = args.compareAudioPath else {
            fatalError("simcheck needs --ref-audio and --compare-audio")
        }
        let (_, refArr) = try loadAudioArray(from: URL(fileURLWithPath: refPath), sampleRate: 24000)
        let (_, cmpArr) = try loadAudioArray(from: URL(fileURLWithPath: cmpPath), sampleRate: 24000)
        guard let ea = model.extractSpeakerEmbedding(from: refArr),
              let eb = model.extractSpeakerEmbedding(from: cmpArr) else {
            fatalError("simcheck requires a checkpoint with a speaker encoder (Base models)")
        }
        let a = ea.flattened().asType(.float32)
        let b = eb.flattened().asType(.float32)
        let cos = (a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))
        eval(cos)
        let cosVal = Double(cos.item(Float.self))
        print("simcheck cos=\(String(format: "%.4f", cosVal)) ref=\(refPath) cmp=\(cmpPath)")
        if let jsonPath = args.json {
            try appendJSON([
                "label": args.label, "mode": "simcheck",
                "judge": modelDir.lastPathComponent,
                "ref": (refPath as NSString).lastPathComponent,
                "cmp": (cmpPath as NSString).lastPathComponent,
                "cos": (cosVal * 10000).rounded() / 10000,
            ], to: jsonPath)
        }
        exit(0)
    }

    // simcheck-set: mean-centered cosine — raw x-vector cosines saturate (~0.99 for
    // unrelated voices), so subtract the set mean before comparing. --compare-audio
    // takes a comma-separated list; the centering mean is computed over ref + all clips.
    if args.mode == "simcheck-set" {
        guard let refPath = args.refAudioPath, let cmpList = args.compareAudioPath else {
            fatalError("simcheck-set needs --ref-audio and --compare-audio (comma-separated)")
        }
        let cmpPaths = cmpList.split(separator: ",").map(String.init)
        func embed(_ path: String) throws -> MLXArray {
            let (_, arr) = try loadAudioArray(from: URL(fileURLWithPath: path), sampleRate: 24000)
            guard let e = model.extractSpeakerEmbedding(from: arr) else {
                fatalError("no speaker encoder in this checkpoint")
            }
            return e.flattened().asType(.float32)
        }
        let refE = try embed(refPath)
        let cmpEs = try cmpPaths.map(embed)
        var mean = refE
        for e in cmpEs { mean = mean + e }
        mean = mean / Float(cmpEs.count + 1)
        func cosine(_ x: MLXArray, _ y: MLXArray) -> Double {
            let a = x - mean, b = y - mean
            let c = (a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))
            eval(c)
            return Double(c.item(Float.self))
        }
        for (p, e) in zip(cmpPaths, cmpEs) {
            let c = cosine(refE, e)
            print("simcheck-set cos=\(String(format: "%.4f", c)) cmp=\((p as NSString).lastPathComponent)")
            if let jsonPath = args.json {
                try appendJSON([
                    "label": args.label, "mode": "simcheck-set",
                    "judge": modelDir.lastPathComponent,
                    "ref": (refPath as NSString).lastPathComponent,
                    "cmp": (p as NSString).lastPathComponent,
                    "cos": (c * 10000).rounded() / 10000,
                ], to: jsonPath)
            }
        }
        exit(0)
    }
    let loadPeakGB = Double(Memory.peakMemory) / 1e9
    FileHandle.standardError.write("model loaded in \(String(format: "%.2f", loadSec))s\n".data(using: .utf8)!)

    var refAudio: MLXArray?
    if let p = args.refAudioPath {
        let (_, arr) = try loadAudioArray(from: URL(fileURLWithPath: p), sampleRate: 24000)
        refAudio = arr
        eval(arr)
    }

    let params = GenerateParameters(
        maxTokens: args.maxTokens, temperature: 0.9, topP: 1.0, repetitionPenalty: 1.05
    )

    for rep in 0 ..< args.reps {
        Memory.peakMemory = 0
        model.seed = args.seed

        let stream: AsyncThrowingStream<AudioGeneration, Error>
        switch args.mode {
        case "design":
            stream = model.generateStream(
                text: args.text, voice: args.voice, refAudio: nil, refText: nil,
                language: nil, generationParameters: params,
                streamingInterval: args.interval
            )
        case "clone-icl":
            guard let refAudio, let refText = args.refText else { fatalError("clone-icl needs --ref-audio and --ref-text") }
            stream = model.generateStream(
                text: args.text, voice: nil, refAudio: refAudio, refText: refText,
                language: nil, generationParameters: params,
                streamingInterval: args.interval
            )
        case "clone-xvec":
            guard let refAudio else { fatalError("clone-xvec needs --ref-audio") }
            stream = model.generateStreamXVectorOnly(
                text: args.text, refAudio: refAudio, language: nil,
                generationParameters: params, streamingInterval: args.interval
            )
        case "customvoice":
            stream = model.generateStream(
                text: args.text, voice: args.voice, refAudio: nil, refText: nil,
                language: nil, generationParameters: params,
                streamingInterval: args.interval
            )
        default:
            fatalError("unknown mode \(args.mode)")
        }

        var samples = [Float]()
        var ttfa: Double?
        var tokenCount = 0
        var infoTokPerSec = 0.0
        var infoGenerateTime = 0.0
        let t0 = now()
        for try await event in stream {
            switch event {
            case .token:
                tokenCount += 1
            case .audio(let chunk):
                if ttfa == nil { ttfa = secs(since: t0) }
                samples.append(contentsOf: chunk.asArray(Float.self))
            case .info(let info):
                infoTokPerSec = info.tokensPerSecond
                infoGenerateTime = info.generateTime
            }
        }
        let wall = secs(since: t0)
        let stats = analyze(samples, sampleRate: model.sampleRate)
        let steps = max(tokenCount - 1, 0) // last onToken is EOS
        let result: [String: Any] = [
            "label": args.label,
            "model": modelDir.lastPathComponent,
            "mode": args.mode,
            "rep": rep,
            "seed": args.seed,
            "textChars": args.text.count,
            "loadSec": (loadSec * 1000).rounded() / 1000,
            "loadPeakGB": (loadPeakGB * 100).rounded() / 100,
            "ttfaMs": ttfa.map { ($0 * 1000).rounded() } ?? -1,
            "wallSec": (wall * 1000).rounded() / 1000,
            "steps": steps,
            "stepsPerSec": wall > 0 ? ((Double(steps) / wall) * 100).rounded() / 100 : 0,
            "infoTokPerSec": (infoTokPerSec * 100).rounded() / 100,
            "infoGenerateSec": (infoGenerateTime * 1000).rounded() / 1000,
            "audioSec": (stats.seconds * 100).rounded() / 100,
            "rtf": stats.seconds > 0 ? ((wall / stats.seconds) * 1000).rounded() / 1000 : -1,
            "genPeakGB": (Double(Memory.peakMemory) / 1e9 * 100).rounded() / 100,
            "peakRssGB": (peakRSSGB() * 100).rounded() / 100,
            "rms": (stats.rms * 10000).rounded() / 10000,
            "peakAmp": (stats.peak * 1000).rounded() / 1000,
            "silentWindowFrac": (stats.silentWindowFraction * 1000).rounded() / 1000,
            "longestSilenceMs": stats.longestSilenceMs,
            "clippedFrac": (stats.clippedFraction * 100000).rounded() / 100000,
        ]
        if let jsonPath = args.json { try appendJSON(result, to: jsonPath) }
        let summary = "rep \(rep): ttfa=\(result["ttfaMs"]!)ms steps=\(steps) steps/s=\(result["stepsPerSec"]!) audio=\(result["audioSec"]!)s rtf=\(result["rtf"]!) peakGPU=\(result["genPeakGB"]!)GB"
        print(summary)

        if rep == args.reps - 1, let out = args.out {
            try writeWAV(samples, sampleRate: model.sampleRate, to: out)
            print("wrote \(out) (\(samples.count) samples)")
        }
    }
    print("BENCH OK")
} catch {
    FileHandle.standardError.write("BENCH FAIL: \(error)\n".data(using: .utf8)!)
    exit(1)
}
