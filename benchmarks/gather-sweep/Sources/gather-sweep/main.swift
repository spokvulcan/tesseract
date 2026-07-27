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
