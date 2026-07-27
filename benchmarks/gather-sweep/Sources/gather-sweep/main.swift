import Foundation
import MLX
import MLXFast
import MLXNN

setbuf(stdout, nil)

// === qmmtiles: dense qmm_t tile-geometry sweep (MLX_QMM_TILES probe) =======
// One process = one tile config (the probe hook in Cmlx's quantized.cpp reads
// MLX_QMM_TILES once per process; unset = stock 32,32,32). Each launch walks
// ALL shapes: writes a fixed-seed bitwise dump per shape, then times ONE lazy
// graph of R quantizedMM reps cycling 8 disjoint weight sets. The driver
// (qmmtiles-driver.sh) interleaves launches stock,cand,cand,stock,... per
// candidate and cmps the dumps across processes for the bitwise gate.

struct QTShape {
    let m: Int, n: Int, k: Int
    var name: String { "m\(m)_n\(n)_k\(k)" }
}

func runQmmTiles() {
    let qtShapes: [QTShape] = [
        .init(m: 1024, n: 248320, k: 2048),  // lm_head (K=2048 model)
        .init(m: 1024, n: 248320, k: 2560),  // lm_head (K=2560 model)
        .init(m: 1024, n: 8192, k: 2048),    // MoE attn/GDN
        .init(m: 1024, n: 512, k: 2048),
        .init(m: 1024, n: 2048, k: 4096),
        .init(m: 1024, n: 4096, k: 2048),
        .init(m: 1024, n: 8192, k: 2560),    // dense attn/GDN/MLP
        .init(m: 1024, n: 1024, k: 2560),
        .init(m: 1024, n: 2560, k: 4096),
        .init(m: 1024, n: 9216, k: 2560),
        .init(m: 1024, n: 2560, k: 9216),
        .init(m: 128, n: 8192, k: 2048),     // ctx-128 single chunk
        .init(m: 128, n: 2048, k: 4096),
        .init(m: 128, n: 8192, k: 2560),
        .init(m: 512, n: 8192, k: 2048),     // mid-chunk sanity
        .init(m: 512, n: 9216, k: 2560),
    ]
    let cfg = ProcessInfo.processInfo.environment["MLX_QMM_TILES"] ?? "stock"
    let cfgSlug = cfg.replacingOccurrences(of: ",", with: "x")
    let dumpDir = "/tmp/gather-sweep/qmmtiles-dumps"
    try? FileManager.default.createDirectory(
        atPath: dumpDir, withIntermediateDirectories: true)
    let gs = 128, bits = 4
    for shape in qtShapes {
        let (M, N, K) = (shape.m, shape.n, shape.k)
        autoreleasepool {
            // 8 disjoint weight sets; fixed keys -> byte-identical inputs in
            // every launch regardless of config.
            var wq: [MLXArray] = [], sc: [MLXArray] = [], bi: [MLXArray?] = []
            for i in 0 ..< 8 {
                let wFull = MLXRandom.uniform(
                    low: -0.5, high: 0.5, [N, K], key: MLXRandom.key(UInt64(1000 + i))
                ).asType(.float16)
                let (wqI, scI, biI) = quantized(wFull, groupSize: gs, bits: bits)
                wq.append(wqI); sc.append(scI); bi.append(biI)
            }
            let x = MLXRandom.uniform(
                low: -1, high: 1, [M, K], key: MLXRandom.key(7)
            ).asType(.float16)
            // Materialize weights/x now so generation+quantize stay OUT of the
            // timed graph.
            for i in 0 ..< 8 {
                eval(wq[i], sc[i])
                if let b = bi[i] { eval(b) }
            }
            eval(x)
            // Bitwise dump: single call, weight set 0, fixed seeds.
            let d = quantizedMM(
                x, wq[0], scales: sc[0], biases: bi[0],
                transpose: true, groupSize: gs, bits: bits)
            let data = d.asData(access: .copy).data
            let path = "\(dumpDir)/\(shape.name)__\(cfgSlug).bin"
            try! data.write(to: URL(fileURLWithPath: path), options: .atomic)
            print("QTDUMP \(cfgSlug) \(shape.name) \(data.count)")
            // Warm the big-graph path (kernel pipeline already compiled by the
            // dump eval above).
            do {
                var warm: [MLXArray] = []
                for i in 0 ..< 3 {
                    warm.append(quantizedMM(
                        x, wq[i], scales: sc[i], biases: bi[i],
                        transpose: true, groupSize: gs, bits: bits))
                }
                eval(warm)
                _ = warm[2][0, 0].item(Float.self)
            }
            // Timed: ONE lazy graph, R reps cycling the 8 sets, single eval.
            let flops = 2.0 * Double(M) * Double(N) * Double(K)
            let R = max(8, min(512, Int((0.08 * 8e12 / flops).rounded(.up))))
            var outs: [MLXArray] = []
            outs.reserveCapacity(R)
            for r in 0 ..< R {
                let i = r % 8
                outs.append(quantizedMM(
                    x, wq[i], scales: sc[i], biases: bi[i],
                    transpose: true, groupSize: gs, bits: bits))
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            eval(outs)
            _ = outs[R - 1].sum().item(Float.self)  // completion sync
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print(String(
                format: "QT %@ %@ R=%d %.3f ms %.3f TFLOPs",
                cfgSlug, shape.name, R, dt * 1e3, flops * Double(R) / dt / 1e12))
            fflush(stdout)
        }
        Memory.clearCache()
    }
}

if CommandLine.arguments.contains("qmmtiles") {
    runQmmTiles()
    exit(0)
}

if CommandLine.arguments.contains("tokprofile") {
    try await runTokProfile()
    exit(0)
}

if CommandLine.arguments.contains("tokdiff") {
    try await runTokDiff()
    exit(0)
}

// C14 mechanism: concat+SDPA unfused (milestone-A pattern) vs inside a
// shapeless-compiled block (whole-step pattern), at 8K with production
// cache behavior (cap buffer, slice_update chain across tokens).
let KVH = 4, H = 16, D = 256, HID = 2560
let scale = pow(Float(D), -0.5)
nonisolated(unsafe) let wq = MLXRandom.uniform(low: -0.02, high: 0.02, [H * D * 2, HID], key: MLXRandom.key(1)).asType(.float16)
nonisolated(unsafe) let wk = MLXRandom.uniform(low: -0.02, high: 0.02, [KVH * D, HID], key: MLXRandom.key(2)).asType(.float16)
nonisolated(unsafe) let wv = MLXRandom.uniform(low: -0.02, high: 0.02, [KVH * D, HID], key: MLXRandom.key(3)).asType(.float16)
nonisolated(unsafe) let wo = MLXRandom.uniform(low: -0.02, high: 0.02, [HID, H * D], key: MLXRandom.key(4)).asType(.float16)
nonisolated(unsafe) let n1 = MLXArray.ones([HID], dtype: .float16)
nonisolated(unsafe) let n2 = MLXArray.ones([D], dtype: .float16)
nonisolated(unsafe) let n3 = MLXArray.ones([D], dtype: .float16)

func faBody(x: MLXArray, cachedK: MLXArray, cachedV: MLXArray, offset: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let xn = MLXFast.rmsNorm(x, weight: n1, eps: 1e-6)
    let qg = xn.matmul(wq.transposed()).reshaped(1, 1, H, 2, -1)
    var queries = take(qg, MLXArray([0]), axis: 3).squeezed(axis: 3)
    let gate = take(qg, MLXArray([1]), axis: 3).squeezed(axis: 3).reshaped(1, 1, -1)
    var keys = xn.matmul(wk.transposed())
    var values = xn.matmul(wv.transposed())
    queries = MLXFast.rmsNorm(queries, weight: n2, eps: 1e-6).transposed(0, 2, 1, 3)
    keys = MLXFast.rmsNorm(keys.reshaped(1, 1, KVH, -1), weight: n3, eps: 1e-6).transposed(0, 2, 1, 3)
    values = values.reshaped(1, 1, KVH, -1).transposed(0, 2, 1, 3)
    queries = MLXFast.RoPE(queries, dimensions: 64, traditional: false, base: 100000.0, scale: 1, offset: offset)
    keys = MLXFast.RoPE(keys, dimensions: 64, traditional: false, base: 100000.0, scale: 1, offset: offset)
    let fullK = MLX.concatenated([cachedK, keys], axis: 2)
    let fullV = MLX.concatenated([cachedV, values], axis: 2)
    let output = MLXFast.scaledDotProductAttention(
        queries: queries, keys: fullK, values: fullV, scale: scale, mask: .none
    ).transposed(0, 2, 1, 3).reshaped(1, 1, -1)
    return ((output * sigmoid(gate)).matmul(wo.transposed()), keys, values)
}

let compiled = compile(shapeless: true) { (args: [MLXArray]) -> [MLXArray] in
    let (o, k, v) = faBody(x: args[0], cachedK: args[1], cachedV: args[2], offset: args[3])
    return [o, k, v]
}

let N0 = 8192, cap = 8448, tokens = 64
let key = MLXRandom.key(42)
var bufK = MLXArray.zeros([1, KVH, cap, D], dtype: .float16)
var bufV = MLXArray.zeros([1, KVH, cap, D], dtype: .float16)
bufK[.ellipsis, ..<N0, 0...] = MLXRandom.uniform(low: -1, high: 1, [1, KVH, N0, D], key: key).asType(.float16)
bufV[.ellipsis, ..<N0, 0...] = MLXRandom.uniform(low: -1, high: 1, [1, KVH, N0, D], key: key).asType(.float16)
let x = MLXRandom.uniform(low: -1, high: 1, [1, 1, HID], key: key).asType(.float16)
let newK = MLXRandom.uniform(low: -1, high: 1, [1, KVH, 1, D], key: key).asType(.float16)
let newV = MLXRandom.uniform(low: -1, high: 1, [1, KVH, 1, D], key: key).asType(.float16)
eval(wq, wk, wv, wo, x, newK, newV)

// warm both
for t in 0..<4 {
    let off = MLXArray(Int32(N0 + t))
    let sk = bufK[.ellipsis, ..<(N0 + t), 0...]
    let sv = bufV[.ellipsis, ..<(N0 + t), 0...]
    let (o, k, v) = faBody(x: x, cachedK: sk, cachedV: sv, offset: off)
    bufK[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = k
    bufV[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = v
    let r = compiled([x, sk, sv, off])
    eval(o, r[0])
}

func timePattern(_ label: String, _ body: (Int) -> Void) {
    let t0 = ContinuousClock.now
    for t in 0..<tokens { body(t) }
    let d = t0.duration(to: .now).components
    let ms = Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15
    print("\(label): \(String(format: "%.3f", ms / Double(tokens))) ms/token")
}

timePattern("A unfused (concat+SDPA outside)") { t in
    let off = MLXArray(Int32(N0 + t))
    let sk = bufK[.ellipsis, ..<(N0 + t), 0...]
    let sv = bufV[.ellipsis, ..<(N0 + t), 0...]
    let (o, k, v) = faBody(x: x, cachedK: sk, cachedV: sv, offset: off)
    bufK[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = k
    bufV[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = v
    eval(o, k, v)
}
timePattern("C compiled (concat+SDPA inside)") { t in
    let off = MLXArray(Int32(N0 + t))
    let sk = bufK[.ellipsis, ..<(N0 + t), 0...]
    let sv = bufV[.ellipsis, ..<(N0 + t), 0...]
    let r = compiled([x, sk, sv, off])
    bufK[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = r[1]
    bufV[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = r[2]
    eval(r[0], r[1], r[2])
}
timePattern("A again") { t in
    let off = MLXArray(Int32(N0 + t))
    let sk = bufK[.ellipsis, ..<(N0 + t), 0...]
    let sv = bufV[.ellipsis, ..<(N0 + t), 0...]
    let (o, k, v) = faBody(x: x, cachedK: sk, cachedV: sv, offset: off)
    bufK[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = k
    bufV[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = v
    eval(o, k, v)
}
timePattern("C again") { t in
    let off = MLXArray(Int32(N0 + t))
    let sk = bufK[.ellipsis, ..<(N0 + t), 0...]
    let sv = bufV[.ellipsis, ..<(N0 + t), 0...]
    let r = compiled([x, sk, sv, off])
    bufK[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = r[1]
    bufV[.ellipsis, (N0 + t)..<(N0 + t + 1), 0...] = r[2]
    eval(r[0], r[1], r[2])
}

// === tokprofile: tokenizer-path profiler for the ParoParityBench "tokenize"
// phase =====================================================================
// Reproduces the production call exactly:
//   app: ParoQuantInputProcessor.prepare (tesseract .../ParoQuant/ParoQuantLoader.swift:67)
//     -> DefaultMessageGenerator: [["role":"user","content":userText]]
//     -> MLXLMCommon.Tokenizer (macro bridge) -> Tokenizers.PreTrainedTokenizer
//        .applyChatTemplate(messages:tools:nil,additionalContext:nil)
//        (swift-transformers Tokenizer.swift:739-836): jinja render (cached
//        Template, lstrip/trimBlocks) + encode(rendered, addSpecialTokens:false)
//   bench prompt: ParoParityBenchRunner.buildPromptText (lines 349-368).
// Tokenizer loaded via Tokenizers.AutoTokenizer.from(modelFolder:) — the same
// entry the app's #huggingFaceTokenizerLoader() macro uses.
// Sub-modes: `phase` (128/8K/32K render-vs-encode split), `scale` (encode
// throughput 1K..32K), `encodeloop [seconds]` (32K encode loop for `sample`).

import Tokenizers
import Hub
import Jinja

// NOTE: these live on an enum because top-level `let` initializers in
// main.swift only execute when control flow reaches them — the tokprofile
// dispatch sits earlier in the file, so plain file-level lets would be
// uninitialized garbage. Static lets are lazily initialized on first access.
private enum TPConst {
    static let modelDir = NSHomeDirectory()
        + "/Library/Application Support/models/z-lab_Qwen3.5-4B-PARO"

    // Verbatim from ParoParityBenchRunner.swift (fillerSentence / question).
    static let fillerSentence =
        "The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits, and the modern era of parallel accelerators that execute trillions of operations per second. "
    static let question = "\n\nSummarize the text above in as much detail as you can."

    static let specialTokenAttributes: [String] = [
        "bos_token", "eos_token", "unk_token", "sep_token",
        "pad_token", "cls_token", "mask_token", "additional_special_tokens",
    ]
}

/// Exact port of ParoParityBenchRunner.buildPromptText (lines 349-368), with
/// the processor.prepare overhead call spelled out as its applyChatTemplate.
private func tpBuildPromptText(tokenizer: any Tokenizers.Tokenizer, targetTokens: Int) throws -> String {
    let overhead = try tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": TPConst.question]],
        tools: nil, additionalContext: nil
    ).count
    let fillerBudget = max(targetTokens - overhead, 0)
    guard fillerBudget > 0 else { return TPConst.question }
    let fillerIds = tokenizer.encode(text: TPConst.fillerSentence, addSpecialTokens: false)
    let reps = (fillerBudget / max(fillerIds.count, 1)) + 1
    let fillerText = String(repeating: TPConst.fillerSentence, count: reps)
    let trimmedIds = Array(
        tokenizer.encode(text: fillerText, addSpecialTokens: false)
            .prefix(fillerBudget)
    )
    return tokenizer.decode(tokens: trimmedIds, skipSpecialTokens: false) + TPConst.question
}

// --- render-only replication (swift-transformers Tokenizer.swift:752-825) ---

/// Copy of internal `addedTokenAsString` (Tokenizer.swift:118-126).
private func tpAddedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() { return stringValue }
    return addedToken.content.string()
}

/// Stand-in for PreTrainedTokenizer.applyChatTemplate up to (but excluding)
/// the encode call: same template selection, same compiled Template options,
/// same context construction. Template compiled once (production caches it in
/// compiledChatTemplateCache after the first prepare).
private struct TPRenderer {
    let template: Jinja.Template
    let tokenizerConfig: Config

    init(tokenizerConfig: Config) throws {
        guard let templateString = tokenizerConfig.chatTemplate.string() else {
            fatalError("tokprofile: no chat_template in tokenizerConfig")
        }
        self.template = try Jinja.Template(
            templateString, with: .init(lstripBlocks: true, trimBlocks: true))
        self.tokenizerConfig = tokenizerConfig
    }

    /// The special-token-attributes context block (Tokenizer.swift:809-823),
    /// shared by both render entry points.
    private func specialTokenContext() throws -> [String: Jinja.Value] {
        var context: [String: Jinja.Value] = [:]
        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            if TPConst.specialTokenAttributes.contains(key.string), !value.isNull() {
                if let stringValue = value.string() {
                    context[key.string] = .string(stringValue)
                } else if let dictionary = value.dictionary() {
                    if let addedTokenString = tpAddedTokenAsString(Config(dictionary)) {
                        context[key.string] = .string(addedTokenString)
                    }
                } else if let array: [String] = value.get() {
                    context[key.string] = .array(array.map { .string($0) })
                } else {
                    context[key.string] = try Jinja.Value(any: value)
                }
            }
        }
        return context
    }

    func render(messages: [Tokenizers.Message]) throws -> String {
        var context: [String: Jinja.Value] = try [
            "messages": .array(messages.map { try Jinja.Value(any: $0) }),
            "add_generation_prompt": .boolean(true),
        ]
        for (k, v) in try specialTokenContext() { context[k] = v }
        return try template.render(context)
    }

    /// Full replication of PreTrainedTokenizer.applyChatTemplate's context
    /// construction (Tokenizer.swift:792-825), incl. tools.
    func render(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        addGenerationPrompt: Bool
    ) throws -> String {
        var context: [String: Jinja.Value] = try [
            "messages": .array(messages.map { try Jinja.Value(any: $0) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if let tools {
            context["tools"] = try .array(tools.map { try Jinja.Value(any: $0) })
        }
        for (k, v) in try specialTokenContext() { context[k] = v }
        return try template.render(context)
    }
}

// --- harness ----------------------------------------------------------------

private func tpNow() -> ContinuousClock.Instant { ContinuousClock.now }
private func tpMs(_ since: ContinuousClock.Instant) -> Double {
    let c = (ContinuousClock.now - since).components
    return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
}

private func tpMedian(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s.isEmpty ? 0 : s[s.count / 2]
}

private struct TPPhaseCell {
    var full: [Double] = []
    var render: [Double] = []
    var encode: [Double] = []
}

private func runTokProfile() async throws {
    let args = CommandLine.arguments
    let mode = args.count > 2 ? args[2] : "phase"

    let modelURL = URL(fileURLWithPath: TPConst.modelDir)
    let t0 = tpNow()
    let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: modelURL)
    let configLoader = LanguageModelConfigurationFromHub(modelFolder: modelURL)
    guard let tokenizerConfig = try await configLoader.tokenizerConfig else {
        fatalError("tokprofile: no tokenizerConfig")
    }
    let renderer = try TPRenderer(tokenizerConfig: tokenizerConfig)
    print(String(format: "LOAD tokenizer+config %.1f ms", tpMs(t0)))
    print("tokenizer type: \(type(of: tokenizer))")

    // Warm the compiled-template cache inside the tokenizer (production state
    // after the first prepare) so `full` measurements see cache hits.
    _ = try tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": "warm"]], tools: nil,
        additionalContext: nil)

    switch mode {
    case "phase":
        try tpPhaseMode(tokenizer: tokenizer, renderer: renderer)
    case "scale":
        try tpScaleMode(tokenizer: tokenizer, renderer: renderer)
    case "encodeloop":
        let seconds = args.count > 3 ? Double(args[3]) ?? 120 : 120
        try tpEncodeLoopMode(tokenizer: tokenizer, renderer: renderer, seconds: seconds)
    case "regexdiff":
        try tpRegexDiffMode(tokenizer: tokenizer, renderer: renderer)
    case "enginediff":
        try tpEngineDiffMode(tokenizer: tokenizer, renderer: renderer)
    case "slowcheck":
        try tpSlowCheckMode(tokenizer: tokenizer, renderer: renderer)
    case "pieces":
        try tpPiecesMode(tokenizer: tokenizer)
    default:
        fatalError("tokprofile: unknown mode \(mode)")
    }
}

private func tpMessages(_ userText: String) -> [Tokenizers.Message] {
    [["role": "user", "content": userText]]
}

private func tpPhaseMode(tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer) throws {
    let sizes = [128, 8192, 32768]
    let rounds = 8  // round 0 discarded as warmup
    var cells: [Int: TPPhaseCell] = [:]

    for target in sizes {
        let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: target)
        let messages = tpMessages(userText)
        let rendered = try renderer.render(messages: messages)

        // Parity gate for the render replication: render+encode must equal the
        // one-shot applyChatTemplate ids exactly.
        let fullIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let splitIds = tokenizer.encode(text: rendered, addSpecialTokens: false)
        print(String(
            format: "PARITY target=%d fullIds=%d splitIds=%d %@",
            target, fullIds.count, splitIds.count,
            fullIds == splitIds ? "IDENTICAL" : "MISMATCH"))
        print(String(
            format: "PROMPT target=%d userTextBytes=%d renderedBytes=%d",
            target, userText.utf8.count, rendered.utf8.count))

        var cell = TPPhaseCell()
        for round in 0 ..< rounds {
            let order = (0 ..< 3).map { ($0 + round) % 3 }  // rotate full/render/encode
            for phase in order {
                autoreleasepool {
                    switch phase {
                    case 0:
                        let s = tpNow()
                        _ = try! tokenizer.applyChatTemplate(
                            messages: messages, tools: nil, additionalContext: nil)
                        let dt = tpMs(s)
                        if round > 0 { cell.full.append(dt) }
                    case 1:
                        let s = tpNow()
                        _ = try! renderer.render(messages: messages)
                        let dt = tpMs(s)
                        if round > 0 { cell.render.append(dt) }
                    default:
                        let s = tpNow()
                        _ = tokenizer.encode(text: rendered, addSpecialTokens: false)
                        let dt = tpMs(s)
                        if round > 0 { cell.encode.append(dt) }
                    }
                }
            }
        }
        cells[target] = cell
    }

    print("\nPHASE-SPLIT (median ms [min..max], 7 timed reps, round-robin interleaved)")
    print("target | full(prepare) | render-only | encode-only | full-render-encode | implied-render=full-encode")
    for target in sizes {
        guard let c = cells[target] else { continue }
        func fmt(_ xs: [Double]) -> String {
            String(format: "%8.2f [%7.2f..%7.2f]", tpMedian(xs), xs.min() ?? 0, xs.max() ?? 0)
        }
        let resid = tpMedian(c.full) - tpMedian(c.render) - tpMedian(c.encode)
        let implied = tpMedian(c.full) - tpMedian(c.encode)
        print(String(
            format: "%6d | %@ | %@ | %@ | %+7.2f | %7.2f",
            target, fmt(c.full), fmt(c.render), fmt(c.encode), resid, implied))
    }
}

private func tpScaleMode(tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer) throws {
    let sizes = [1024, 4096, 16384, 32768]
    print("\nENCODE-SCALE (encode-only on rendered prompts, median of 5 after 1 warmup)")
    print("targetTokens | renderedBytes | medianMs | tok/s | us/KB")
    for target in sizes {
        let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: target)
        let rendered = try renderer.render(messages: tpMessages(userText))
        var dts: [Double] = []
        var nTokens = 0
        for rep in 0 ..< 6 {
            autoreleasepool {
                let s = tpNow()
                let ids = tokenizer.encode(text: rendered, addSpecialTokens: false)
                let dt = tpMs(s)
                nTokens = ids.count
                if rep > 0 { dts.append(dt) }
            }
        }
        let med = tpMedian(dts)
        print(String(
            format: "%12d | %13d | %8.2f | %9.0f | %6.2f",
            target, rendered.utf8.count, med,
            Double(nTokens) / (med / 1000.0),
            med * 1000.0 / Double(rendered.utf8.count) * 1024.0))
    }
}

private func tpEncodeLoopMode(
    tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer, seconds: Double
) throws {
    let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: 32768)
    let rendered = try renderer.render(messages: tpMessages(userText))
    print(String(
        format: "ENCODELOOP pid=%d renderedBytes=%d duration=%.0fs",
        getpid(), rendered.utf8.count, seconds))
    fflush(stdout)
    // warmup
    _ = tokenizer.encode(text: rendered, addSpecialTokens: false)
    let start = tpNow()
    var iter = 0
    var totalTokens = 0
    while tpMs(start) < seconds * 1000 {
        autoreleasepool {
            let ids = tokenizer.encode(text: rendered, addSpecialTokens: false)
            totalTokens += ids.count
        }
        iter += 1
        if iter % 10 == 0 {
            let el = tpMs(start)
            print(String(
                format: "iter=%d elapsed=%.1fs avgTok/s=%.0f",
                iter, el / 1000, Double(totalTokens) / (el / 1000)))
            fflush(stdout)
        }
    }
    print(String(format: "ENCODELOOP-DONE iters=%d", iter))
}

// --- regexdiff: Split-pretokenizer strategy comparison ----------------------
// Replicates String.split(by:options:includeSeparators:) from swift-transformers
// String+PreTokenization.swift:47-65 (the SplitPreTokenizer regexp path used by
// this model's pre_tokenizer Sequence) and compares it — for exact-equal
// output and for speed — against a single precompiled NSRegularExpression
// pass. Evidence for the "single-pass regex" optimization candidate.

/// Verbatim port of String+PreTokenization.swift:47-65 (isolated behavior).
private func tpSplitLoop(_ text: String, pattern: String) -> [String] {
    var result: [String] = []
    var start = text.startIndex
    while let range = text.range(
        of: pattern, options: .regularExpression, range: start..<text.endIndex)
    {
        if start < range.lowerBound {
            result.append(String(text[start..<range.lowerBound]))
        }
        result.append(String(text[range]))
        start = range.upperBound
    }
    if start < text.endIndex {
        result.append(String(text[start...]))
    }
    return result
}

/// Single-pass alternative: one precompiled NSRegularExpression over the whole
/// text, emitting gap substrings and match substrings in order (same
/// "isolated" semantics).
private func tpSplitSinglePass(_ text: String, regex: NSRegularExpression) -> [String] {
    let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
    let matches = regex.matches(in: text, options: [], range: fullRange)
    var result: [String] = []
    var start = text.startIndex
    for m in matches {
        guard let r = Range(m.range, in: text) else { continue }
        if start < r.lowerBound {
            result.append(String(text[start..<r.lowerBound]))
        }
        result.append(String(text[r]))
        start = r.upperBound
    }
    if start < text.endIndex {
        result.append(String(text[start...]))
    }
    return result
}

private func tpRegexDiffMode(tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer) throws {
    // Pull the Split pattern straight out of tokenizer.json (no transcription).
    let tjURL = URL(fileURLWithPath: TPConst.modelDir).appending(path: "tokenizer.json")
    let tj = try JSONSerialization.jsonObject(with: Data(contentsOf: tjURL)) as! [String: Any]
    let pt = tj["pre_tokenizer"] as! [String: Any]
    let subs = pt["pretokenizers"] as! [[String: Any]]
    let splitSub = subs.first { ($0["type"] as? String) == "Split" }!
    // Force native storage: JSONSerialization strings are NSString-bridged, and
    // String.range(of:.regularExpression) runs ~3-4x slower with a bridged
    // pattern (measured); the production Config/yyjson path holds native
    // strings (production split ~= 76ms of the 126ms encode, per sampling).
    let pattern = String(decoding: ((splitSub["pattern"] as! [String: Any])["Regex"] as! String).utf8, as: UTF8.self)
    print("PATTERN \(pattern)")

    let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: 32768)
    let rendered = try renderer.render(messages: tpMessages(userText))
    let regex = try NSRegularExpression(pattern: pattern, options: [])

    var a: [String] = []
    var b: [String] = []
    // warmup + timed reps
    var dtsA: [Double] = []
    var dtsB: [Double] = []
    for rep in 0 ..< 6 {
        autoreleasepool {
            let s = tpNow()
            a = tpSplitLoop(rendered, pattern: pattern)
            let dt = tpMs(s)
            if rep > 0 { dtsA.append(dt) }
        }
        autoreleasepool {
            let s = tpNow()
            b = tpSplitSinglePass(rendered, regex: regex)
            let dt = tpMs(s)
            if rep > 0 { dtsB.append(dt) }
        }
    }
    print(String(format: "PRETokens loop=%d singlePass=%d equal=%@", a.count, b.count,
        a == b ? "IDENTICAL" : "MISMATCH"))
    if a != b {
        let n = min(a.count, b.count)
        for i in 0 ..< n where a[i] != b[i] {
            print("first divergence at \(i): loop=\(a[i].debugDescription) single=\(b[i].debugDescription)")
            break
        }
    }
    print(String(format: "SPLIT-TIME loopPerMatch=%.2f ms  singlePass=%.2f ms  speedup=%.1fx",
        tpMedian(dtsA), tpMedian(dtsB), tpMedian(dtsA) / max(tpMedian(dtsB), 0.001)))
}

// --- enginediff: same-engine single-pass Swift Regex + ICU-world id impact --
// (a) Production drives the pattern through String.range(of:.regularExpression)
//     once per match. A precompiled `Regex` (SAME engine, SAME syntax parser)
//     driven once via `ranges(of:)` must yield identical boundaries — verify,
//     then time both.
// (b) Estimate the final-ID impact if splits were ICU-style instead: replicate
//     the production tokenize() structure (addedTokens split -> NFC -> split ->
//     per-pretoken encode) with the NSRegularExpression split from regexdiff,
//     and diff the id streams.

private func tpPretokensFromRanges(_ text: String, ranges: [Range<String.Index>]) -> [String] {
    var result: [String] = []
    var start = text.startIndex
    for r in ranges {
        if start < r.lowerBound { result.append(String(text[start..<r.lowerBound])) }
        result.append(String(text[r]))
        start = r.upperBound
    }
    if start < text.endIndex { result.append(String(text[start...])) }
    return result
}

private func tpEngineDiffMode(tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer) throws {
    let tjURL = URL(fileURLWithPath: TPConst.modelDir).appending(path: "tokenizer.json")
    let tj = try JSONSerialization.jsonObject(with: Data(contentsOf: tjURL)) as! [String: Any]
    let pt = tj["pre_tokenizer"] as! [String: Any]
    let subs = pt["pretokenizers"] as! [[String: Any]]
    let splitSub = subs.first { ($0["type"] as? String) == "Split" }!
    let pattern = String(decoding: ((splitSub["pattern"] as! [String: Any])["Regex"] as! String).utf8, as: UTF8.self)

    let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: 32768)
    let rendered = try renderer.render(messages: tpMessages(userText))

    // (a) same-engine precompiled Regex
    let swiftRegex = try Regex(pattern)
    var dtsLoop: [Double] = []
    var dtsRanges: [Double] = []
    var loopTokens: [String] = []
    var rangesTokens: [String] = []
    for rep in 0 ..< 6 {
        autoreleasepool {
            let s = tpNow()
            loopTokens = tpSplitLoop(rendered, pattern: pattern)
            let dt = tpMs(s)
            if rep > 0 { dtsLoop.append(dt) }
        }
        autoreleasepool {
            let s = tpNow()
            rangesTokens = tpPretokensFromRanges(rendered, ranges: rendered.ranges(of: swiftRegex))
            let dt = tpMs(s)
            if rep > 0 { dtsRanges.append(dt) }
        }
    }
    print(String(format: "SWIFTREGEX loop=%d ranges=%d equal=%@",
        loopTokens.count, rangesTokens.count,
        loopTokens == rangesTokens ? "IDENTICAL" : "MISMATCH"))
    print(String(format: "SWIFTREGEX-TIME loopPerMatch=%.2f ms  precompiledRanges=%.2f ms  speedup=%.1fx",
        tpMedian(dtsLoop), tpMedian(dtsRanges), tpMedian(dtsLoop) / max(tpMedian(dtsRanges), 0.001)))

    // (b) ICU-world final ids. Added-token section split replicated from
    // tokenizer.json (same construction as PreTrainedTokenizer init:
    // length-desc sort, escaped, lstrip/rstrip \s* wrappers).
    let added = (tj["added_tokens"] as! [[String: Any]]).map {
        (content: $0["content"] as! String, id: $0["id"] as! Int,
         lstrip: $0["lstrip"] as? Bool ?? false, rstrip: $0["rstrip"] as? Bool ?? false)
    }.sorted { $0.content.count > $1.content.count }
    let addedPattern = added.map {
        let esc = NSRegularExpression.escapedPattern(for: $0.content)
        return "\($0.lstrip ? #"\s*"# : "")(\(esc))\($0.rstrip ? #"\s*"# : "")"
    }.joined(separator: "|")
    let addedRe = try NSRegularExpression(pattern: addedPattern)
    let addedContents = Set(added.map { $0.content })
    let addedIds = Dictionary(uniqueKeysWithValues: added.map { ($0.content, $0.id) })

    // production tokenize()'s added-token section split (String.split(by:) —
    // capture-group variant, String+PreTokenization.swift:68-107)
    func addedSections(_ text: String) -> [String] {
        let selfRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = addedRe.matches(in: text, options: [], range: selfRange)
        if matches.isEmpty { return [text] }
        var result: [String] = []
        var start = text.startIndex
        for match in matches {
            guard let matchRange = Range(match.range, in: text) else { continue }
            if start < matchRange.lowerBound {
                result.append(String(text[start..<matchRange.lowerBound]))
            }
            start = matchRange.upperBound
            for r in (0 ..< match.numberOfRanges).reversed() {
                if let sepRange = Range(match.range(at: r), in: text) {
                    result.append(String(text[sepRange]))
                    break
                }
            }
        }
        if start < text.endIndex { result.append(String(text[start...])) }
        return result
    }

    let icuRe = try NSRegularExpression(pattern: pattern)
    var icuIds: [Int] = []
    icuIds.reserveCapacity(40000)
    for section in addedSections(rendered) {
        if addedContents.contains(section) {
            icuIds.append(addedIds[section]!)
            continue
        }
        let normalized = section.precomposedStringWithCanonicalMapping
        for pretoken in tpSplitSinglePass(normalized, regex: icuRe) {
            icuIds.append(contentsOf: tokenizer.encode(text: pretoken, addSpecialTokens: false))
        }
    }

    let prodIds = tokenizer.encode(text: rendered, addSpecialTokens: false)
    print(String(format: "ICUWORLD prod=%d icu=%d equal=%@",
        prodIds.count, icuIds.count, prodIds == icuIds ? "IDENTICAL" : "MISMATCH"))
    if prodIds != icuIds {
        let n = min(prodIds.count, icuIds.count)
        var diffs = 0
        var firstAt = -1
        for i in 0 ..< n where prodIds[i] != icuIds[i] {
            diffs += 1
            if firstAt < 0 { firstAt = i }
        }
        print("ICUWORLD diffs=\(diffs) firstAt=\(firstAt) lenDelta=\(prodIds.count - icuIds.count)")
        if firstAt >= 0 {
            let lo = max(0, firstAt - 3), hi = min(n, firstAt + 4)
            print("prod[\(lo)..<\(hi)] = \(Array(prodIds[lo..<hi]))")
            print("icu [\(lo)..<\(hi)] = \(Array(icuIds[lo..<hi]))")
            print("prod context: \(tokenizer.decode(tokens: Array(prodIds[lo..<hi]), skipSpecialTokens: false).debugDescription)")
            print("icu  context: \(tokenizer.decode(tokens: Array(icuIds[lo..<hi]), skipSpecialTokens: false).debugDescription)")
        }
    }
}

// --- slowcheck: why does tpSplitLoop measure ~300ms on the rendered bench
// text while sampling attributes only ~76ms/encode to the same code? --------
private func tpSlowCheckMode(tokenizer: any Tokenizers.Tokenizer, renderer: TPRenderer) throws {
    let tjURL = URL(fileURLWithPath: TPConst.modelDir).appending(path: "tokenizer.json")
    let tj = try JSONSerialization.jsonObject(with: Data(contentsOf: tjURL)) as! [String: Any]
    let pt = tj["pre_tokenizer"] as! [String: Any]
    let subs = pt["pretokenizers"] as! [[String: Any]]
    let splitSub = subs.first { ($0["type"] as? String) == "Split" }!
    let pattern = String(decoding: ((splitSub["pattern"] as! [String: Any])["Regex"] as! String).utf8, as: UTF8.self)

    let userText = try tpBuildPromptText(tokenizer: tokenizer, targetTokens: 32768)
    let rendered = try renderer.render(messages: tpMessages(userText))
    let freshCopy = String(decoding: rendered.utf8, as: UTF8.self)
    let benchLike = String(repeating: TPConst.fillerSentence, count: 800)

    func timeSplit(_ label: String, _ text: String) {
        var dts: [Double] = []
        var n = 0
        for rep in 0 ..< 4 {
            autoreleasepool {
                let s = tpNow()
                n = tpSplitLoop(text, pattern: pattern).count
                let dt = tpMs(s)
                if rep > 0 { dts.append(dt) }
            }
        }
        print(String(format: "%@ pretokens=%d median=%.2f ms", label, n, tpMedian(dts)))
    }
    timeSplit("rendered      ", rendered)
    timeSplit("freshCopy     ", freshCopy)
    timeSplit("benchLike     ", benchLike)

    // reference: full production encode on the same rendered text
    var dts: [Double] = []
    for rep in 0 ..< 4 {
        autoreleasepool {
            let s = tpNow()
            _ = tokenizer.encode(text: rendered, addSpecialTokens: false)
            let dt = tpMs(s)
            if rep > 0 { dts.append(dt) }
        }
    }
    print(String(format: "production encode median=%.2f ms", tpMedian(dts)))
}

// --- pieces: why do Swift-Regex "\nThe" and ICU "\n"|"The" converge to the
// same final ids? Show BPE pieces for the divergent pretoken shapes. --------
private func tpPiecesMode(tokenizer: any Tokenizers.Tokenizer) throws {
    for probe in ["\nThe", "\n", "The", "user\nThe history", ".\n\nSummarize"] {
        let pieces = tokenizer.tokenize(text: probe)
        let ids = tokenizer.encode(text: probe, addSpecialTokens: false)
        print("PROBE \(probe.debugDescription) pieces=\(pieces.map { $0.debugDescription }) ids=\(ids)")
    }
}

// === tokdiff: differential-equivalence harness for the Split-pretokenizer
// engine swap (per-match Swift Regex loop -> single-pass NSRegularExpression)
// ==========================================================================
// Arms:
//   P  production: PreTrainedTokenizer.encode — ground truth.
//   P' replication gate: exact port of the production tokenize() pipeline
//      (added-tokens capture split -> NFC -> Split via Swift-Regex loop,
//      isolated -> ByteLevel use_regex=false byteEncode -> per-pretoken BPE
//      via a pre_tokenizer/normalizer-stripped PreTrainedTokenizer, loaded
//      through the same AutoTokenizer path). MUST equal P everywhere; if it
//      doesn't, the harness is broken, not the candidate.
//   A  ICU original: P' with the Split loop replaced by ONE precompiled
//      NSRegularExpression pass over the ORIGINAL pattern.
//   B  ICU quirk-mutated: same, with [^\r\n\p{L}\p{N}] -> [^\p{L}\p{N}].
//      Characterization proven by tokquirk.swift: in a negated Swift-Regex
//      class, \r and \n individually escape the negation ONLY when the class
//      source contains the adjacent pair "\r\n" ([^\n\p{L}] and [^\r\p{L}]
//      behave correctly; [^\r\n] misbehaves even without \p{...}; \s-based
//      negated classes are unaffected). Dropping \r\n from the class is
//      therefore the exact ICU fold-in of the Swift behavior.

private enum TDArm: String {
    case pPrime = "P'"
    case icuA = "A"
    case icuB = "B"
}

private struct TDAdded {
    let regex: NSRegularExpression
    let contents: Set<String>
    let ids: [String: Int]
}

private final class TDCtx {
    let tokenizer: any Tokenizers.Tokenizer  // production (arm P)
    let bpeOnly: any Tokenizers.Tokenizer  // stripped pipeline: identity normalize/pretokenize
    let renderer: TPRenderer
    let pattern: String  // original Split pattern (native storage)
    let patternMut: String  // quirk-mutated pattern
    let icuA: NSRegularExpression
    let icuB: NSRegularExpression
    let added: TDAdded

    init(
        tokenizer: any Tokenizers.Tokenizer, bpeOnly: any Tokenizers.Tokenizer,
        renderer: TPRenderer, pattern: String, patternMut: String,
        icuA: NSRegularExpression, icuB: NSRegularExpression, added: TDAdded
    ) {
        self.tokenizer = tokenizer
        self.bpeOnly = bpeOnly
        self.renderer = renderer
        self.pattern = pattern
        self.patternMut = patternMut
        self.icuA = icuA
        self.icuB = icuB
        self.added = added
    }
}

/// Port of ByteEncoder.swift's byteEncoderTable (GPT-2 bytes_to_unicode):
/// printable bytes map to themselves, the rest to U+0100+n in byte order.
/// NOTE: static-let inside an enum — top-level `let` in main.swift only
/// initializes when control flow reaches it, which never happens for the
/// early `tokdiff` dispatch (same trap TPConst documents above).
private enum TDConst {
    static let byteEncoderTable: [String] = {
    var table = [String](repeating: "", count: 256)
    var printable = Set(33 ... 126)
    printable.formUnion(161 ... 172)
    printable.formUnion(174 ... 255)
    for b in 0 ... 255 where printable.contains(b) {
        table[b] = String(Unicode.Scalar(b)!)
    }
    var n = 0
    for b in 0 ... 255 where !printable.contains(b) {
        table[b] = String(Unicode.Scalar(0x100 + n)!)
        n += 1
    }
    precondition(
        table[0] == "\u{0100}" && table[10] == "\u{010A}" && table[13] == "\u{010D}"
            && table[32] == "\u{0120}" && table[65] == "A" && table[127] == "\u{0121}"
            && table[173] == "\u{0143}" && table[255] == "\u{00FF}",
        "TDConst.byteEncoderTable mismatch with ByteEncoder.swift")
    return table
}()
}

/// ByteLevelPreTokenizer.byteEncodeToken (use_regex=false, add_prefix_space=false).
private func tdByteEncode(_ token: String) -> String {
    var encoded = ""
    encoded.reserveCapacity(token.utf8.count * 2)
    for byte in token.utf8 {
        encoded.append(TDConst.byteEncoderTable[Int(byte)])
    }
    return encoded
}

/// Verbatim port of String.split(by captureRegex:) — String+PreTokenization.swift:68-107.
private func tdAddedSections(_ text: String, regex: NSRegularExpression) -> [String] {
    let selfRange = NSRange(text.startIndex..<text.endIndex, in: text)
    let matches = regex.matches(in: text, options: [], range: selfRange)
    if matches.isEmpty { return [text] }
    var result: [String] = []
    var start = text.startIndex
    for match in matches {
        guard let matchRange = Range(match.range, in: text) else { continue }
        if start < matchRange.lowerBound {
            result.append(String(text[start..<matchRange.lowerBound]))
        }
        start = matchRange.upperBound
        for r in (0 ..< match.numberOfRanges).reversed() {
            if let sepRange = Range(match.range(at: r), in: text) {
                result.append(String(text[sepRange]))
                break
            }
        }
    }
    if start < text.endIndex {
        result.append(String(text[start...]))
    }
    return result
}

/// The candidate pipeline: production tokenize() with the Split step swapped
/// per arm. Mirrors PreTrainedTokenizer.tokenize/encode (Tokenizer.swift:620-644):
/// sections -> added-token passthrough -> NFC -> split -> byteEncode -> BPE.
/// Returns final ids. `splitNs` accumulates Split-phase nanoseconds.
private func tdNs(_ since: ContinuousClock.Instant) -> UInt64 {
    let c = (ContinuousClock.now - since).components
    return UInt64(c.seconds) * 1_000_000_000 + UInt64(c.attoseconds / 1_000_000_000)
}

private func tdEncodeArm(_ text: String, arm: TDArm, ctx: TDCtx, splitNs: inout UInt64) -> [Int] {
    var ids: [Int] = []
    ids.reserveCapacity(text.utf8.count / 3)
    for section in tdAddedSections(text, regex: ctx.added.regex) {
        if ctx.added.contents.contains(section) {
            ids.append(ctx.added.ids[section]!)
            continue
        }
        let normalized = section.precomposedStringWithCanonicalMapping
        let t0 = ContinuousClock.now
        let pretokens: [String]
        switch arm {
        case .pPrime: pretokens = tpSplitLoop(normalized, pattern: ctx.pattern)
        case .icuA: pretokens = tpSplitSinglePass(normalized, regex: ctx.icuA)
        case .icuB: pretokens = tpSplitSinglePass(normalized, regex: ctx.icuB)
        }
        splitNs &+= tdNs(t0)
        for pretoken in pretokens {
            ids.append(contentsOf: ctx.bpeOnly.encode(text: tdByteEncode(pretoken), addSpecialTokens: false))
        }
    }
    return ids
}

/// Pretoken stream of an arm (for diagnosis): (sectionIndex, pretoken) pairs.
private func tdPretokenStream(_ text: String, arm: TDArm, ctx: TDCtx) -> [(Int, String)] {
    var stream: [(Int, String)] = []
    for (si, section) in tdAddedSections(text, regex: ctx.added.regex).enumerated() {
        if ctx.added.contents.contains(section) {
            stream.append((si, "<ADDED:\(section)>"))
            continue
        }
        let normalized = section.precomposedStringWithCanonicalMapping
        let pretokens: [String]
        switch arm {
        case .pPrime: pretokens = tpSplitLoop(normalized, pattern: ctx.pattern)
        case .icuA: pretokens = tpSplitSinglePass(normalized, regex: ctx.icuA)
        case .icuB: pretokens = tpSplitSinglePass(normalized, regex: ctx.icuB)
        }
        for p in pretokens { stream.append((si, p)) }
    }
    return stream
}

// --- deterministic corpus generation ----------------------------------------

private struct TDRng {
    var s: UInt64
    init(_ seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func int(_ n: Int) -> Int { Int(next() % UInt64(max(n, 1))) }
    mutating func pick<T>(_ a: [T]) -> T { a[int(a.count)] }
    mutating func chance(_ num: Int, _ den: Int) -> Bool { int(den) < num }
}

private struct TDItem {
    let cls: String
    let name: String
    let text: String
    var chatIds: [Int]? = nil  // production applyChatTemplate ids (chat items only)
}

private extension TDConst {
    static let words = [
        "history", "computing", "spans", "mechanical", "calculators", "vacuum", "tubes",
        "transistors", "integrated", "circuits", "modern", "parallel", "accelerators",
        "execute", "trillions", "operations", "second", "The", "memory", "bandwidth",
        "kernel", "compiler", "silicon", "cache", "branch", "vector", "tensor", "pipeline",
        "latency", "throughput", "scheduler", "register", "allocator", "optimizer",
    ]
    static let digitChars = Array("0123456789")
}

private func tdSentence(_ rng: inout TDRng, words: Int) -> String {
    var parts: [String] = []
    for _ in 0 ..< words { parts.append(rng.pick(TDConst.words)) }
    return parts.joined(separator: " ")
}

private func tdPad(_ s: String, toBytes target: Int, _ rng: inout TDRng) -> String {
    var out = s
    while out.utf8.count < target {
        out += tdSentence(&rng, words: 8) + ".\n"
    }
    return out
}

private func tdBuildCorpus(ctx: TDCtx, scale: Double) throws -> [TDItem] {
    let advTarget = Int(300_000 * scale)
    var items: [TDItem] = []
    var nextSeed: UInt64 = 1000
    func adv(_ cls: String, _ name: String, target: Int? = nil, body: (inout TDRng, Int) -> String) {
        var rng = TDRng(nextSeed)
        nextSeed += 1
        items.append(TDItem(cls: cls, name: name, text: body(&rng, target ?? advTarget)))
    }

    // -- class: newline-before-letter (the known quirk class) --
    adv("newline-letter", "nl-runs") { rng, target in
        var out = ""
        let openers = ["The", "A", "Summarize", "history", "a", "ab", "é", "Ж", "かな", "x"]
        while out.utf8.count < target {
            out += tdSentence(&rng, words: 3 + rng.int(8))
            let nl = rng.pick(["\n", "\n\n", "\n", "\r", "\r\n", " \n"])
            out += nl + rng.pick(openers) + tdSentence(&rng, words: rng.int(4)) + " "
        }
        return out
    }

    // -- class: \r\n line endings --
    adv("crlf", "prose-crlf") { rng, target in
        var out = ""
        while out.utf8.count < target {
            out += tdSentence(&rng, words: 5 + rng.int(10)) + "."
            out += rng.chance(1, 6) ? "\r\n\r\n" : "\r\n"
        }
        return out
    }

    // -- class: lone \r --
    adv("lone-cr", "prose-cr") { rng, target in
        var out = ""
        while out.utf8.count < target {
            out += tdSentence(&rng, words: 5 + rng.int(10)) + "."
            out += rng.chance(1, 8) ? "\r\r" : "\r"
        }
        return out
    }

    // -- class: tabs --
    adv("tabs", "tab-indent") { rng, target in
        var out = ""
        while out.utf8.count < target {
            let indent = String(repeating: "\t", count: rng.int(4))
            out += indent + tdSentence(&rng, words: 2 + rng.int(6))
            out += rng.chance(1, 5) ? "\t\t" : "\n"
        }
        return out
    }

    // -- class: multi-space runs --
    adv("multispace", "space-runs") { rng, target in
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(TDConst.words)
            out += String(repeating: " ", count: 1 + rng.int(8))
        }
        return out
    }

    // -- class: trailing-space-before-newline (\s+(?!\S) territory) --
    adv("trailing-space", "trail-sp-nl") { rng, target in
        var out = ""
        while out.utf8.count < target {
            out += tdSentence(&rng, words: 3 + rng.int(7))
            out += String(repeating: " ", count: 1 + rng.int(4))
            out += rng.chance(1, 7) ? " \n \n" : "\n"
        }
        out += "   "  // trailing spaces at EOF: \s+(?!\S) match
        return out
    }

    // -- class: digits --
    adv("digits", "num-forms") { rng, target in
        var out = ""
        while out.utf8.count < target {
            switch rng.int(6) {
            case 0: out += String((1 ... (1 + rng.int(6))).map { _ in TDConst.digitChars[rng.int(10)] })
            case 1: out += "\(rng.int(1000)).\(100 + rng.int(900))"
            case 2: out += "\(rng.int(10)),\(100 + rng.int(900)),\(100 + rng.int(900))"
            case 3: out += "v\(rng.int(10)).\(rng.int(20)).\(rng.int(99)) "
            case 4: out += "+1-\(100 + rng.int(900))-\(100 + rng.int(900))-\(1000 + rng.int(9000)) "
            default: out += rng.pick(TDConst.words) + "\(rng.int(10000))" + rng.pick(TDConst.words) + " "
            }
            out += rng.chance(1, 12) ? "\n" : " "
        }
        return out
    }

    // -- class: ASCII punctuation runs --
    adv("punct", "ascii-punct") { rng, target in
        let puncts = ["!", "!!", "?!", "...", "--", "===", "***", "###", "@", "#$%", "&&", "|", "~", "`", ";;", "::", ">>", "<<", "()", "[]", "{}", "<>", "/", "\\", "\"", "'"]
        var out = ""
        while out.utf8.count < target {
            out += rng.chance(1, 3) ? rng.pick(puncts) : rng.pick(TDConst.words)
            out += rng.chance(1, 8) ? "\n" : " "
        }
        return out
    }

    // -- class: contractions --
    adv("contractions", "english-contr") { rng, target in
        let contrs = ["I'll", "don't", "we're", "'tis", "he's", "they'd", "can't", "won't", "o'clock", "'90s", "y'all", "isn't", "she'll", "we've", "I'm", "you're", "it'll", "wouldn't've", "ma'am", "'em", "n'", "rock'n'roll", "DON'T", "Don't"]
        var out = ""
        while out.utf8.count < target {
            out += rng.chance(2, 3) ? rng.pick(contrs) : rng.pick(TDConst.words)
            out += rng.chance(1, 9) ? "\n" : " "
        }
        return out
    }

    // -- class: emoji incl ZWJ --
    adv("emoji", "emoji-zwj") { rng, target in
        let emojis = ["😀", "🎉", "👍", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "❤️", "©️", "*️⃣", "🇺🇸", "🇯🇵", "🧑‍💻", "🚀", "🔥", "✨", "🤖", "👩‍👩‍👦", "🐍", "🍎", "⚙️", "🛠️", "💾", "🖥️", "⌨️"]
        var out = ""
        while out.utf8.count < target {
            out += rng.chance(3, 5) ? rng.pick(emojis) : rng.pick(TDConst.words)
            out += rng.pick([" ", "", " ", "\n", "! "])
        }
        return out
    }

    // -- class: CJK --
    adv("cjk", "zh-ja-ko") { rng, target in
        let chunks = ["计算的历史", "横跨了机械计算器", "真空管", "晶体管", "集成电路", "以及现代并行加速器", "コンピュータの歴史", "ひらがなとカタカナ", "漢字を混ぜた文章", "コンピュータ의 역사", "한국어 텍스트", "。、！？", "「引用」", "『二重引用』", "第一、第二、第三"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(chunks)
            out += rng.pick(["", "。", "、", "\n", " ", "！"])
        }
        return out
    }

    // -- class: Cyrillic --
    adv("cyrillic", "russian") { rng, target in
        let chunks = ["История", "вычислительной", "техники", "охватывает", "механические", "калькуляторы", "электронные", "лампы", "транзисторы", "интегральные", "схемы", "и", "современную", "эпоху", "параллельных", "ускорителей", "Ёлка", "ъезд", "подъём"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(chunks)
            out += rng.pick([" ", " ", ". ", ", ", "\n", " — "])
        }
        return out
    }

    // -- class: accented Latin (precomposed) --
    adv("accented-latin", "precomposed") { rng, target in
        let chunks = ["café", "naïve", "Zürich", "Ångström", "résumé", "piñata", "São", "Łódź", "Œuvre", "æther", "smörgåsbord", "Müller", "français", "español", "português", "Đà", "Nẵng", "İstanbul", "święto"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(chunks)
            out += rng.pick([" ", " ", ", ", ".\n", " et "])
        }
        return out
    }

    // -- class: combining marks (incl \n + mark: quirk-adjacent) --
    adv("combining-marks", "decomposed") { rng, target in
        let bases = ["a", "e", "o", "n", "c", "s", "u", "i", "y", "ka", "し", "ก", "क"]
        let marks = ["\u{0301}", "\u{0300}", "\u{0302}", "\u{0303}", "\u{0308}", "\u{0327}", "\u{3099}", "\u{093C}"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(bases) + rng.pick(marks)
            if rng.chance(1, 4) { out += rng.pick(marks) }
            switch rng.int(8) {
            case 0: out += "\n" + rng.pick(marks)  // \n immediately before \p{M}
            case 1: out += "\n"
            case 2: out += " "
            default: break
            }
            if rng.chance(1, 10) { out += " " }
        }
        return out
    }

    // -- class: Arabic (RTL) --
    adv("arabic", "rtl-mixed") { rng, target in
        let chunks = ["تاريخ", "الحوسبة", "يمتد", "عبر", "الآلات", "الحاسبة", "الميكانيكية", "والأنابيب", "المفرغة", "والترانزستورات", "١٢٣٤٥", "٦٧٨٩٠", "\u{200E}", "\u{200F}", "العَرَبِيَّة"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(chunks)
            out += rng.pick([" ", " ", "، ", ". ", "\n", " Latin "])
        }
        return out
    }

    // -- class: mixed script --
    adv("mixed-script", "polyglot") { rng, target in
        let chunks = ["Hello", "世界", "Привет", "مرحبا", "こんにちは", "안녕하세요", "café", "😀", "123", "नमस्ते", "สวัสดี", "γειά", "שלום", "हिन्दी"]
        var out = ""
        while out.utf8.count < target {
            out += rng.pick(chunks)
            out += rng.pick([" ", " ", "\n", " — ", "、", "! "])
        }
        return out
    }

    // -- class: code with heavy \n    indentation --
    adv("code", "swift-py-json") { rng, target in
        var out = ""
        while out.utf8.count < target {
            let fn = rng.pick(TDConst.words)
            out += "func \(fn)\(rng.int(100))(x: Int) -> Int {\n"
            let depth = 1 + rng.int(3)
            for d in 0 ..< depth {
                let ind = String(repeating: "    ", count: d + 1)
                out += "\(ind)if x > \(rng.int(1000)) {\n"
                out += "\(ind)    let v\(rng.int(100)) = x * \(rng.int(100))\n"
            }
            for d in (0 ..< depth).reversed() {
                let ind = String(repeating: "    ", count: d + 1)
                out += "\(ind)}\n"
            }
            out += "}\n\n"
            out += "def \(fn)_py(data):\n    return [x ** 2 for x in data if x % \(2 + rng.int(7))]\n\n"
            out += "{ \"key\(rng.int(100))\": [\(rng.int(1000)), \(rng.int(1000)), { \"nested\": true }] }\n"
        }
        return out
    }

    // -- class: JSON tool specs (synthetic 40-tool schema, raw JSON) --
    let tools = tdSyntheticTools()
    adv("json-tools", "tool-schema-raw", target: max(advTarget / 4, 50_000)) { rng, target in
        var out = "# Tool schema dump\n\n"
        var i = 0
        while out.utf8.count < target {
            let d = try! JSONSerialization.data(
                withJSONObject: tools[i % tools.count], options: [.prettyPrinted, .sortedKeys])
            out += String(decoding: d, as: UTF8.self) + "\n"
            i += 1
        }
        _ = rng
        return out
    }

    // -- class: markdown --
    adv("markdown", "md-full") { rng, target in
        var out = ""
        while out.utf8.count < target {
            out += String(repeating: "#", count: 1 + rng.int(4)) + " " + tdSentence(&rng, words: 3) + "\n\n"
            out += tdSentence(&rng, words: 12) + ".\n\n"
            for _ in 0 ..< (2 + rng.int(4)) {
                out += String(repeating: "  ", count: rng.int(3)) + "- \(rng.pick(TDConst.words)) \(rng.int(100))\n"
            }
            out += "\n| col1 | col2 |\n| --- | --- |\n| \(rng.int(100)) | \(rng.pick(TDConst.words)) |\n\n"
            out += "```swift\nlet x\(rng.int(100)) = \(rng.int(1000))\nprint(x)\n```\n\n"
            out += "> \(tdSentence(&rng, words: 6))\n\n[link](https://example.com/\(rng.int(1000))) **bold** *italic* `code`\n\n"
        }
        return out
    }

    // -- class: whitespace-only runs + Unicode whitespace zoo --
    adv("whitespace-runs", "ws-zoo") { rng, target in
        let zoo = [" ", "\t", "\n", "\r", "\u{0B}", "\u{0C}", "\u{85}", "\u{A0}", "\u{1680}",
                   "\u{2000}", "\u{2001}", "\u{2002}", "\u{2003}", "\u{2004}", "\u{2005}",
                   "\u{2006}", "\u{2007}", "\u{2008}", "\u{2009}", "\u{200A}", "\u{2028}",
                   "\u{2029}", "\u{205F}", "\u{3000}", "\u{FEFF}"]
        var out = ""
        while out.utf8.count < target {
            if rng.chance(1, 20) {
                out += rng.pick(TDConst.words)  // rare anchor
            } else {
                out += String(repeating: rng.pick(zoo), count: 1 + rng.int(6))
            }
        }
        return out
    }

    // -- bench prompts at 128 / 8192 / 32768 (production buildPromptText port) --
    for target in [128, 8192, 32768] {
        let userText = try tpBuildPromptText(tokenizer: ctx.tokenizer, targetTokens: target)
        let rendered = try ctx.renderer.render(messages: tpMessages(userText))
        items.append(TDItem(cls: "bench", name: "bench-\(target)", text: rendered))
    }

    // -- realistic chat renders: system + 40 tools + multi-turn, ~8K and ~32K --
    for target in [8192, 32768] {
        let item = try tdBuildChatItem(ctx: ctx, tools: tools, target: target, seed: 4242 + UInt64(target))
        items.append(item)
    }

    // -- random Unicode soup U+0020..U+2FFF --
    adv("unicode-soup", "soup-0020-2fff", target: Int(100_000 * scale)) { rng, target in
        var scalars = String.UnicodeScalarView()
        var bytes = 0
        while bytes < target {
            let scalar = Unicode.Scalar(0x20 + rng.int(0x2FE0))!
            scalars.append(scalar)
            bytes += scalar.utf8.count
        }
        return String(scalars)
    }

    return items
}

/// 40 synthetic Qwen-style function-tool JSON schemas.
private func tdSyntheticTools() -> [Tokenizers.ToolSpec] {
    let domains = [
        ("calendar", "event", ["title", "date", "duration_minutes", "attendees"]),
        ("email", "message", ["recipient", "subject", "body", "cc"]),
        ("filesystem", "file", ["path", "content", "mode", "encoding"]),
        ("web", "search", ["query", "max_results", "region", "safe"]),
        ("code", "execution", ["language", "source", "timeout_seconds", "stdin"]),
        ("database", "query", ["table", "columns", "where_clause", "limit"]),
        ("weather", "forecast", ["location", "days", "units", "hourly"]),
        ("reminder", "task", ["title", "due_date", "priority", "notes"]),
    ]
    let actions = ["get", "create", "update", "delete", "list"]
    var tools: [Tokenizers.ToolSpec] = []
    for (domain, noun, props) in domains {
        for action in actions {
            var properties: [String: any Sendable] = [:]
            for prop in props {
                let isInt = prop.contains("count") || prop.contains("minutes") || prop.contains("limit")
                    || prop.contains("days") || prop.contains("seconds") || prop.contains("results")
                let type = isInt ? "integer" : (prop == "hourly" || prop == "safe" ? "boolean" : "string")
                properties[prop] = ["type": type, "description": "The \(prop) of the \(noun) to \(action)."] as [String: any Sendable]
            }
            tools.append([
                "type": "function",
                "function": [
                    "name": "\(domain)_\(action)_\(noun)",
                    "description": "\(action.capitalized) a \(noun) in the \(domain) system. Use this when the user asks to \(action) \(noun) information.",
                    "parameters": [
                        "type": "object",
                        "properties": properties,
                        "required": Array(props.prefix(2)),
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ])
        }
    }
    return tools  // 8 domains × 5 actions = 40
}

/// Build a chat-render item at ~target tokens: system prompt + tools +
/// multi-turn user/assistant history, rendered via the production jinja path.
private func tdBuildChatItem(ctx: TDCtx, tools: [Tokenizers.ToolSpec], target: Int, seed: UInt64) throws -> TDItem {
    var rng = TDRng(seed)
    let systemPrompt = """
        You are Tesseract, a fully offline AI assistant running locally on Apple Silicon. \
        You know the user's goals, habits, and preferences from persistent memory. \
        Today is 2026-07-27. The user prefers concise, evidence-based answers. \
        When calling tools, follow the schema exactly and never invent parameters.
        """
    var messages: [Tokenizers.Message] = [["role": "system", "content": systemPrompt]]
    // Grows turns until the production applyChatTemplate reaches the target.
    var chatIds: [Int] = []
    for turn in 0 ..< 200 {
        let user = "Turn \(turn): " + tdSentence(&rng, words: 30) + "?\n\nSome context:\n" + tdPad("", toBytes: 400, &rng)
        let assistant = "Answer \(turn): " + tdSentence(&rng, words: 40) + ".\n\n- point \(rng.int(100))\n- point \(rng.int(100))\n\n```\ncode_\(rng.int(1000))\n```"
        messages.append(["role": "user", "content": user])
        messages.append(["role": "assistant", "content": assistant])
        if turn % 4 == 3 {
            chatIds = try ctx.tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil, addGenerationPrompt: true,
                truncation: false, maxLength: nil, tools: tools)
            if chatIds.count >= target { break }
        }
    }
    let rendered = try ctx.renderer.render(messages: messages, tools: tools, addGenerationPrompt: true)
    return TDItem(cls: "chat", name: "chat-\(target)", text: rendered, chatIds: chatIds)
}

// --- harness ------------------------------------------------------------------

private struct TDItemResult {
    var bytes = 0
    var tokensP = 0
    var mismatchItems: [TDArm: Int] = [:]
    var mismatchIds: [TDArm: Int] = [:]
    var splitMsLoop = 0.0
    var splitMsA = 0.0
    var splitMsB = 0.0
    var encodeMsP = 0.0
}

private func tdFirstDiff(_ a: [Int], _ b: [Int]) -> Int? {
    let n = min(a.count, b.count)
    for i in 0 ..< n where a[i] != b[i] { return i }
    return a.count == b.count ? nil : n
}

private func tdDiagnose(item: TDItem, arm: TDArm, ctx: TDCtx, idsP: [Int], idsX: [Int]) {
    guard let first = tdFirstDiff(idsP, idsX) else { return }
    let nDiff = zip(idsP, idsX).filter { $0 != $1 }.count
    print("  MISMATCH arm=\(arm.rawValue) firstAt=\(first) lenP=\(idsP.count) lenX=\(idsX.count) idDiffs=\(nDiff)")
    let lo = max(0, first - 3), hiP = min(idsP.count, first + 4), hiX = min(idsX.count, first + 4)
    print("  P  ids[\(lo)..<\(hiP)] = \(Array(idsP[lo..<hiP]))")
    print("  \(arm.rawValue)  ids[\(lo)..<\(hiX)] = \(Array(idsX[lo..<hiX]))")
    print("  P  text: \(ctx.tokenizer.decode(tokens: Array(idsP[lo..<hiP]), skipSpecialTokens: false).debugDescription)")
    print("  \(arm.rawValue)  text: \(ctx.tokenizer.decode(tokens: Array(idsX[lo..<hiX]), skipSpecialTokens: false).debugDescription)")

    // pretoken-level first divergence
    let sP = tdPretokenStream(item.text, arm: .pPrime, ctx: ctx)
    let sX = tdPretokenStream(item.text, arm: arm, ctx: ctx)
    var pIdx = -1
    for i in 0 ..< min(sP.count, sX.count) where sP[i] != sX[i] { pIdx = i; break }
    if pIdx < 0, sP.count != sX.count { pIdx = min(sP.count, sX.count) }
    if pIdx >= 0 {
        let wlo = max(0, pIdx - 2), whiP = min(sP.count, pIdx + 3), whiX = min(sX.count, pIdx + 3)
        print("  PRETOK first divergence at stream[\(pIdx)] (section \(pIdx < sP.count ? sP[pIdx].0 : -1)):")
        print("    P': \(sP[wlo..<whiP].map { $0.1.debugDescription })")
        print("    \(arm.rawValue) : \(sX[wlo..<whiX].map { $0.1.debugDescription })")
        // minimal standalone repro: window of P' pretokens around divergence
        let repro = sP[max(0, pIdx - 1) ..< min(sP.count, pIdx + 2)].map { $0.1 }.joined()
        var sink: UInt64 = 0
        let rP = ctx.tokenizer.encode(text: repro, addSpecialTokens: false)
        let rX = tdEncodeArm(repro, arm: arm, ctx: ctx, splitNs: &sink)
        print("    repro \(repro.debugDescription) standalone: \(rP == rX ? "converges" : "STILL DIVERGES") P=\(rP) \(arm.rawValue)=\(rX)")
    }
}

private func tdRunMode(ctx: TDCtx) throws {
    print("PATTERN-ORIG \(ctx.pattern)")
    print("PATTERN-MUT  \(ctx.patternMut)")

    var scale = 1.0
    var grandTokens = 0
    var finalResults: [(TDItem, TDItemResult)] = []
    while true {
        let corpus = try tdBuildCorpus(ctx: ctx, scale: scale)
        finalResults = []
        grandTokens = 0
        for item in corpus {
            var res = TDItemResult()
            res.bytes = item.text.utf8.count
            autoreleasepool {
                var splitNs: [TDArm: UInt64] = [:]

                let tP0 = ContinuousClock.now
                let idsP = item.chatIds ?? ctx.tokenizer.encode(text: item.text, addSpecialTokens: false)
                res.encodeMsP = tpMs(tP0)
                res.tokensP = idsP.count
                if let chatIds = item.chatIds {
                    let reIds = ctx.tokenizer.encode(text: item.text, addSpecialTokens: false)
                    print("CHAT-GATE \(item.name) applyChatTemplate==encode(rendered): \(chatIds == reIds ? "IDENTICAL" : "MISMATCH")")
                }

                var ns: UInt64 = 0
                let idsPp = tdEncodeArm(item.text, arm: .pPrime, ctx: ctx, splitNs: &ns)
                splitNs[.pPrime] = ns
                ns = 0
                let idsA = tdEncodeArm(item.text, arm: .icuA, ctx: ctx, splitNs: &ns)
                splitNs[.icuA] = ns
                ns = 0
                let idsB = tdEncodeArm(item.text, arm: .icuB, ctx: ctx, splitNs: &ns)
                splitNs[.icuB] = ns

                res.splitMsLoop = Double(splitNs[.pPrime] ?? 0) / 1e6
                res.splitMsA = Double(splitNs[.icuA] ?? 0) / 1e6
                res.splitMsB = Double(splitNs[.icuB] ?? 0) / 1e6

                let arms: [(TDArm, [Int])] = [(.pPrime, idsPp), (.icuA, idsA), (.icuB, idsB)]
                var flags: [String] = []
                for (arm, idsX) in arms {
                    let equal = idsP == idsX
                    flags.append("\(arm.rawValue)=\(equal ? "=" : "!")")
                    if !equal {
                        res.mismatchItems[arm] = 1
                        res.mismatchIds[arm] = zip(idsP, idsX).filter { $0 != $1 }.count
                            + abs(idsP.count - idsX.count)
                    }
                }
                print(String(
                    format: "ITEM %@/%@ bytes=%d tokP=%d %@ | splitMs loop=%.2f A=%.2f B=%.2f | encP=%.2fms",
                    item.cls, item.name, res.bytes, res.tokensP, flags.joined(separator: " "),
                    res.splitMsLoop, res.splitMsA, res.splitMsB, res.encodeMsP))
                if res.splitMsA > res.splitMsLoop * 1.05 || res.splitMsB > res.splitMsLoop * 1.05 {
                    print("  SLOWER-THAN-P: \(item.cls)/\(item.name) A=\(res.splitMsA)ms B=\(res.splitMsB)ms loop=\(res.splitMsLoop)ms")
                }
                for (arm, idsX) in arms where idsP != idsX {
                    tdDiagnose(item: item, arm: arm, ctx: ctx, idsP: idsP, idsX: idsX)
                }
            }
            finalResults.append((item, res))
            grandTokens += res.tokensP
        }
        print("CORPUS-PASS scale=\(String(format: "%.2f", scale)) items=\(corpus.count) totalTokensP=\(grandTokens)")
        if grandTokens >= 2_200_000 || scale >= 3.0 { break }
        scale *= 1.6
        print("CORPUS-REBUILD: under 2.2M tokens, scaling up")
    }

    // coverage table
    print("\nCOVERAGE (class x items x tokens x mismatches)")
    print("class | items | tokensP | mismItems P'/A/B | mismIds P'/A/B")
    var classes: [String] = []
    for (item, _) in finalResults where !classes.contains(item.cls) { classes.append(item.cls) }
    var grandMism: [TDArm: Int] = [:]
    for cls in classes {
        let rows = finalResults.filter { $0.0.cls == cls }
        var toks = 0
        var mi: [TDArm: Int] = [:]
        var mids: [TDArm: Int] = [:]
        for (_, r) in rows {
            toks += r.tokensP
            for arm in [TDArm.pPrime, .icuA, .icuB] {
                mi[arm, default: 0] += r.mismatchItems[arm] ?? 0
                mids[arm, default: 0] += r.mismatchIds[arm] ?? 0
            }
        }
        for arm in [TDArm.pPrime, .icuA, .icuB] { grandMism[arm, default: 0] += mi[arm] ?? 0 }
        print(String(
            format: "%@ | %d | %d | %d/%d/%d | %d/%d/%d",
            cls, rows.count, toks,
            mi[.pPrime] ?? 0, mi[.icuA] ?? 0, mi[.icuB] ?? 0,
            mids[.pPrime] ?? 0, mids[.icuA] ?? 0, mids[.icuB] ?? 0))
    }
    print(String(format: "TOTAL | %d | %d | %d/%d/%d",
        finalResults.count, grandTokens,
        grandMism[.pPrime] ?? 0, grandMism[.icuA] ?? 0, grandMism[.icuB] ?? 0))

    // split-phase speed per class (loop vs ICU)
    print("\nSPLIT-PHASE TIME per class (ms, loop / A / B)")
    for cls in classes {
        let rows = finalResults.filter { $0.0.cls == cls }
        let l = rows.reduce(0.0) { $0 + $1.1.splitMsLoop }
        let a = rows.reduce(0.0) { $0 + $1.1.splitMsA }
        let b = rows.reduce(0.0) { $0 + $1.1.splitMsB }
        print(String(format: "%@ loop=%.2f A=%.2f B=%.2f ratioA=%.1fx ratioB=%.1fx",
            cls, l, a, b, l / max(a, 0.001), l / max(b, 0.001)))
    }

    let gateOK = (grandMism[.pPrime] ?? 0) == 0
    let aOK = (grandMism[.icuA] ?? 0) == 0
    let bOK = (grandMism[.icuB] ?? 0) == 0
    print("\nVERDICT-INPUT gate(P'==P)=\(gateOK) A==P=\(aOK) B==P=\(bOK)")
}

private func tdTimingMode(ctx: TDCtx) throws {
    let userText = try tpBuildPromptText(tokenizer: ctx.tokenizer, targetTokens: 32768)
    let rendered = try ctx.renderer.render(messages: tpMessages(userText))
    print("TIMING-32K renderedBytes=\(rendered.utf8.count)")
    // warmup
    _ = tpSplitLoop(rendered, pattern: ctx.pattern)
    _ = tpSplitSinglePass(rendered, regex: ctx.icuA)
    var dtsLoop: [Double] = []
    var dtsA: [Double] = []
    for _ in 0 ..< 7 {  // ABBA per rep: loop, A, A, loop
        autoreleasepool {
            let s = tpNow()
            _ = tpSplitLoop(rendered, pattern: ctx.pattern)
            dtsLoop.append(tpMs(s))
        }
        autoreleasepool {
            let s = tpNow()
            _ = tpSplitSinglePass(rendered, regex: ctx.icuA)
            dtsA.append(tpMs(s))
        }
        autoreleasepool {
            let s = tpNow()
            _ = tpSplitSinglePass(rendered, regex: ctx.icuA)
            dtsA.append(tpMs(s))
        }
        autoreleasepool {
            let s = tpNow()
            _ = tpSplitLoop(rendered, pattern: ctx.pattern)
            dtsLoop.append(tpMs(s))
        }
    }
    let medLoop = tpMedian(dtsLoop), medA = tpMedian(dtsA)
    print(String(format: "SPLIT-ABBA loop median=%.2f ms [%.2f..%.2f]  singlePass median=%.2f ms [%.2f..%.2f]  speedup=%.1fx",
        medLoop, dtsLoop.min() ?? 0, dtsLoop.max() ?? 0,
        medA, dtsA.min() ?? 0, dtsA.max() ?? 0, medLoop / max(medA, 0.001)))
}

private func runTokDiff() async throws {
    let args = CommandLine.arguments
    let sub = args.count > 2 ? args[2] : "run"

    let modelURL = URL(fileURLWithPath: TPConst.modelDir)
    let t0 = tpNow()
    let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: modelURL)
    let configLoader = LanguageModelConfigurationFromHub(modelFolder: modelURL)
    guard let tokenizerConfig = try await configLoader.tokenizerConfig else {
        fatalError("tokdiff: no tokenizerConfig")
    }
    let renderer = try TPRenderer(tokenizerConfig: tokenizerConfig)
    print(String(format: "LOAD tokenizer+config %.1f ms", tpMs(t0)))

    // Split pattern straight from tokenizer.json, forced to native storage.
    let tjURL = modelURL.appending(path: "tokenizer.json")
    let tj = try JSONSerialization.jsonObject(with: Data(contentsOf: tjURL)) as! [String: Any]
    let pt = tj["pre_tokenizer"] as! [String: Any]
    let subs = pt["pretokenizers"] as! [[String: Any]]
    let splitSub = subs.first { ($0["type"] as? String) == "Split" }!
    let pattern = String(decoding: ((splitSub["pattern"] as! [String: Any])["Regex"] as! String).utf8, as: UTF8.self)

    // Quirk mutation: [^\r\n\p{L}\p{N}] -> [^\p{L}\p{N}] (exactly one site).
    let classOrig = #"[^\r\n\p{L}\p{N}]"#
    let classMut = #"[^\p{L}\p{N}]"#
    precondition(pattern.contains(classOrig), "tokdiff: original class not found in pattern")
    let patternMut = pattern.replacingOccurrences(of: classOrig, with: classMut)
    precondition(!patternMut.contains(classOrig) && patternMut.contains(classMut),
        "tokdiff: mutation failed")

    // Added-token regex: same construction as PreTrainedTokenizer.init
    // (Tokenizer.swift:507-523): length-desc sort, escaped, \s* wrappers.
    let added = (tj["added_tokens"] as! [[String: Any]]).map {
        (content: $0["content"] as! String, id: $0["id"] as! Int,
         lstrip: $0["lstrip"] as? Bool ?? false, rstrip: $0["rstrip"] as? Bool ?? false)
    }.sorted { $0.content.count > $1.content.count }
    let addedPattern = added.map {
        let esc = NSRegularExpression.escapedPattern(for: $0.content)
        return "\($0.lstrip ? #"\s*"# : "")(\(esc))\($0.rstrip ? #"\s*"# : "")"
    }.joined(separator: "|")
    let tdAdded = TDAdded(
        regex: try NSRegularExpression(pattern: addedPattern),
        contents: Set(added.map { $0.content }),
        ids: Dictionary(uniqueKeysWithValues: added.map { ($0.content, $0.id) }))

    // BPE-only tokenizer: tokenizer.json minus pre_tokenizer/normalizer, loaded
    // via the SAME AutoTokenizer/modelFolder path as production.
    let stripDir = "/tmp/gather-sweep/tokdiff-model"
    try FileManager.default.createDirectory(atPath: stripDir, withIntermediateDirectories: true)
    var tjMod = tj
    tjMod.removeValue(forKey: "pre_tokenizer")
    tjMod.removeValue(forKey: "normalizer")
    let tjModData = try JSONSerialization.data(withJSONObject: tjMod, options: [])
    try tjModData.write(to: URL(fileURLWithPath: stripDir + "/tokenizer.json"), options: .atomic)
    for f in ["tokenizer_config.json", "config.json"] {
        let dst = stripDir + "/" + f
        try? FileManager.default.removeItem(atPath: dst)
        try FileManager.default.copyItem(
            atPath: TPConst.modelDir + "/" + f, toPath: dst)
    }
    let bpeOnly = try await Tokenizers.AutoTokenizer.from(
        modelFolder: URL(fileURLWithPath: stripDir))

    let ctx = TDCtx(
        tokenizer: tokenizer, bpeOnly: bpeOnly, renderer: renderer,
        pattern: pattern, patternMut: patternMut,
        icuA: try NSRegularExpression(pattern: pattern),
        icuB: try NSRegularExpression(pattern: patternMut),
        added: tdAdded)

    // Smoke test: bpeOnly must encode a byte-encoded pretoken non-trivially;
    // the global P' gate is the authoritative harness validation.
    let smokeIds = ctx.bpeOnly.encode(text: "ĠTheĊ", addSpecialTokens: false)
    precondition(!smokeIds.isEmpty, "tokdiff: bpeOnly smoke encode failed")

    switch sub {
    case "timing":
        try tdTimingMode(ctx: ctx)
    case "repro":
        // Minimal standalone final-id repros for the two divergence classes.
        let probes = [
            ".\r\n\r\n", "x\r\n\r\n", "!\r\n\r\n!", "\r\r\r\n", "a\r\n\r\nb",
            "1️⃣", "2️⃣3️⃣", " 👩‍👩‍👦*️⃣history", "a*️⃣b", "©️x",
            "❤️", "a❤️b", "🤖❤️!", "❤️x", "👍🏽x", "🇺🇸x", "🏳️‍🌈x", "✨y",
        ]
        var sink: UInt64 = 0
        for p in probes {
            let idsP = ctx.tokenizer.encode(text: p, addSpecialTokens: false)
            let idsA = tdEncodeArm(p, arm: .icuA, ctx: ctx, splitNs: &sink)
            let idsB = tdEncodeArm(p, arm: .icuB, ctx: ctx, splitNs: &sink)
            print("REPRO \(p.debugDescription)")
            print("  P=\(idsP)")
            print("  A=\(idsA) \(idsP == idsA ? "=" : "!")  B=\(idsB) \(idsP == idsB ? "=" : "!")")
        }
    default:
        try tdRunMode(ctx: ctx)
    }
}
