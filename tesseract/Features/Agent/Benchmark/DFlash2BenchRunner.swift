import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// The DFlash2 perf ruler: autoregressive baseline vs the DFlash2 speculative
/// arm on the same long-context prompt, ABBA-interleaved against thermal
/// drift (the experiments-ledger discipline), decode-only timing (iterator
/// construction/prefill happens outside the timed region).
///
/// Driven via `scripts/bench.sh quick --model qwen3.8-27b --dflash2-bench`.
nonisolated struct DFlash2BenchRunner {
    let runner: BenchmarkRunner

    private struct RoundTiming: Sendable, Encodable {
        let width: Int
        let accepted: Int
        let milliseconds: Double
    }

    private struct ArmResult: Sendable, Encodable {
        let arm: String
        let runIndex: Int
        let decodeSeconds: Double
        let tokens: Int
        let accepted: Int
        let proposed: Int
        let rounds: Int
        let prefillSeconds: Double
        let roundTimings: [RoundTiming]
        /// Full stream for saved reports or --bench-check; first eight otherwise.
        let fingerprint: [Int]
        var tokPerSec: Double { Double(tokens) / decodeSeconds }
    }

    private static var arguments: [String] { ProcessInfo.processInfo.arguments }
    private static var fast: Bool { arguments.contains("--bench-fast") }
    private static var check: Bool { arguments.contains("--bench-check") }
    private static var captureFullStream: Bool { check || option("--bench-json") != nil }
    private static var draftPolicy: String { option("--bench-draft-policy") ?? "4bit" }

    private static func option(_ name: String) -> String? {
        guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
        return arguments[i + 1]
    }

    private static func positiveOption(_ name: String, default fallback: Int) -> Int {
        guard let raw = option(name), let value = Int(raw), value > 0 else { return fallback }
        return value
    }

    private struct LogitProbe: LogitProcessor {
        let label: String
        let position: Int
        let path: String
        var count = 0
        var history: [MLXArray] = []

        mutating func prompt(_ prompt: MLXArray) {}
        mutating func didSample(token: MLXArray) {
            count += 1
            history.append(token.reshaped([1]))
            if history.count > 8 { history.removeFirst() }
        }

        func process(logits: MLXArray) -> MLXArray {
            if count == position {
                let row = logits.flattened()
                let ids = argPartition(row, kth: row.size - 5)[(row.size - 5)...]
                let scores = take(row, ids).asArray(Float.self)
                let record: [String: Any] = [
                    "arm": label, "position": position,
                    "history": history.flatMap { $0.asArray(Int32.self) },
                    "ids": ids.asArray(Int32.self), "logits": scores,
                ]
                if let data = try? JSONSerialization.data(
                    withJSONObject: record, options: [.sortedKeys]),
                    let handle = FileHandle(forWritingAtPath: path)
                {
                    defer { try? handle.close() }
                    _ = try? handle.seekToEnd()
                    try? handle.write(contentsOf: data + Data([10]))
                }
            }
            return logits
        }
    }

    private static func components(label: String) -> GenerationComponents {
        guard let raw = option("--bench-logits-at"), let position = Int(raw),
            let path = option("--bench-logits-file")
        else { return .init() }
        return GenerationComponents(logitProcessorFactory: {
            LogitProbe(label: label, position: position, path: path)
        })
    }

    /// Kernel-loop ruler for the small-M quantized matmul on the verify
    /// shapes: wall time per synced launch, GB/s, and the max error against a
    /// dequantized f32 matmul. `DFLASH2_QMM_MICROBENCH=1` runs it instead of
    /// the model bench; `DFLASH2_QMM_SHAPES="M,N,K;M,N,K"` and
    /// `DFLASH2_QMM_ITERATIONS=N` override the defaults. With
    /// `MLX_KERNEL_PROFILE=1` the mlx fork also reports exact GPU time per
    /// shape under the `qmm-MxNxK` window keys.
    @MainActor
    private func qmmMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        var shapes: [(m: Int, n: Int, k: Int)] = [
            (8, 34816, 5120), (8, 5120, 17408), (8, 16480, 5120), (8, 5120, 6144),
            (8, 248320, 5120), (7, 248320, 5120), (1, 34816, 5120),
        ]
        if let raw = env["DFLASH2_QMM_SHAPES"] {
            shapes = raw.split(separator: ";").compactMap { spec in
                let parts = spec.split(separator: ",").compactMap { Int($0) }
                return parts.count == 3 ? (parts[0], parts[1], parts[2]) : nil
            }
        }
        let iterations = Int(env["DFLASH2_QMM_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        emit("[qmm-microbench] \(shapes.count) shapes, \(iterations) synced launches each")
        for shape in shapes {
            let (m, n, k) = (shape.m, shape.n, shape.k)
            MLXRandom.seed(7)
            let x = MLXRandom.normal([1, m, k], dtype: .bfloat16)
            let (wq, scales, biases) = quantized(
                MLXRandom.normal([n, k], dtype: .bfloat16), groupSize: 64, bits: 4)
            guard let biases else { throw DFlash2BenchError.draftMissing }
            eval(x, wq, scales, biases)
            let launch = {
                quantizedMatmul(
                    x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64,
                    bits: 4)
            }
            let reference = matmul(
                x.asType(.float32),
                dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
                    .asType(.float32).transposed())
            let out = launch()
            let error = (out.asType(.float32) - reference).abs().max().item(Float.self)
            let magnitude = reference.abs().max().item(Float.self)
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", "qmm-\(m)x\(n)x\(k)", 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            let perLaunch = seconds / Double(iterations)
            let bytes =
                Double(n * k) / 2 + Double(n * (k / 64) * 4) + Double(m * k * 2)
                + Double(m * n * 2)
            emit(
                String(
                    format: "[qmm-microbench] M=%d N=%d K=%d  %.1f us/launch  %.0f GB/s  "
                        + "maxerr %.4f (of %.2f)",
                    m, n, k, perLaunch * 1e6, bytes / perLaunch / 1e9, error, magnitude))
            Memory.clearCache()
        }
    }

    /// Fused residual + RMS norm ruler: `rmsNormResidual` against `x + r`
    /// then `MLXFast.rmsNorm` on the decode row shape, bitwise, then both
    /// timed under the profiler windows `rmsres-split` / `rmsres-fused`.
    /// `DFLASH2_RMSRES_MICROBENCH=1`.
    @MainActor
    private func rmsResidualMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_RMSRES_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        var mismatches = 0
        for (seed, rows, axis, dtype) in [
            (1, 8, 5120, DType.bfloat16), (2, 1, 5120, .bfloat16), (3, 5, 2048, .float16),
            (4, 3, 100, .bfloat16), (5, 8, 5120, .float32), (6, 8, 5120, .bfloat16),
        ] as [(UInt64, Int, Int, DType)] {
            MLXRandom.seed(seed)
            let x = (MLXRandom.normal([1, rows, axis]) * 3).asType(dtype)
            let r = (MLXRandom.normal([1, rows, axis]) * 0.5).asType(dtype)
            let weight = (MLXRandom.normal([axis]) + 1).asType(dtype)
            let eps: Float = 1e-6
            let h = x + r
            let reference = MLXFast.rmsNorm(h, weight: weight, eps: eps)
            let fused = rmsNormResidual(x, r, weight: weight, eps: eps)
            let sumMatch = (fused.h .== h).all().item(Bool.self)
            let normMatch = (fused.out .== reference).all().item(Bool.self)
            let normDiff = (fused.out.asType(.float32) - reference.asType(.float32)).abs().max()
                .item(Float.self)
            if !sumMatch || !normMatch { mismatches += 1 }
            emit(
                "[rmsres-microbench] rows=\(rows) axis=\(axis) \(dtype): sum "
                    + "\(sumMatch ? "MATCH" : "DIFFERS") norm \(normMatch ? "MATCH" : "DIFFERS") "
                    + "maxdiff \(normDiff)")
        }
        MLXRandom.seed(1)
        let x = (MLXRandom.normal([1, 8, 5120]) * 3).asType(.bfloat16)
        let r = (MLXRandom.normal([1, 8, 5120]) * 0.5).asType(.bfloat16)
        let weight = (MLXRandom.normal([5120]) + 1).asType(.bfloat16)
        eval(x, r, weight)
        let arms: [(String, () -> MLXArray)] = [
            ("rmsres-split", { MLXFast.rmsNorm(x + r, weight: weight, eps: 1e-6) }),
            ("rmsres-fused", { rmsNormResidual(x, r, weight: weight, eps: 1e-6).out }),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[rmsres-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
        emit("[rmsres-microbench] mismatching cases: \(mismatches)")
    }

    /// Fused GDN conv → silu → q/k norm → head scale ruler:
    /// `gatedDeltaConvNormQKV` against the separate ops on the target's
    /// shapes, bitwise, then the ops chain and the kernel timed under the
    /// profiler windows `gdnconv-ops` / `gdnconv-fused`.
    /// `DFLASH2_GDNCONV_MICROBENCH=1`.
    @MainActor
    private func gdnConvNormMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_GDNCONV_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (hk, hv, hd, taps) = (16, 48, 128, 4)
        let d = (2 * hk + hv) * hd
        let keyDim = hk * hd
        let invScale = pow(Float(hd), -0.5)
        let qScale = MLXArray(pow(invScale, 2)).asType(.bfloat16)
        let kScale = MLXArray(invScale).asType(.bfloat16)
        let scales = concatenated([qScale.reshaped([1]), kScale.reshaped([1])])
        eval(qScale, kScale, scales)
        MLXRandom.seed(7)
        let weight = (MLXRandom.normal([d, taps, 1]) * 0.5).asType(.bfloat16)
        eval(weight)
        func reference(_ convInput: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            let b = convInput.dim(0)
            let s = convInput.dim(1) - taps + 1
            let convOut = silu(conv1d(convInput, weight, groups: d))
            let qk = MLXFast.rmsNorm(
                convOut[.ellipsis, ..<(2 * keyDim)].reshaped(b, s, 2 * hk, hd),
                weight: MLXArray.mlxNone, eps: 1e-6)
            return (
                qScale * qk[.ellipsis, ..<hk, 0...],
                kScale * qk[.ellipsis, hk..., 0...],
                convOut[.ellipsis, (2 * keyDim)...].reshaped(b, s, hv, hd)
            )
        }
        func bits(_ x: MLXArray) -> MLXArray { contiguous(x).view(dtype: .uint16) }
        // The kernel reads the state rows and the qkv columns of a projection
        // row separately; the reference sees their concat.
        let projRow = 16480
        var inputs: [(MLXArray, MLXArray, MLXArray, (q: MLXArray, k: MLXArray, v: MLXArray))] = []
        for (seed, s) in [(1, 8), (2, 1), (3, 5)] as [(UInt64, Int)] {
            MLXRandom.seed(seed)
            let state = (MLXRandom.normal([1, taps - 1, d]) * 2).asType(.bfloat16)
            let proj = (MLXRandom.normal([1, s, projRow]) * 2).asType(.bfloat16)
            let x = concatenated([state, proj[0..., 0..., ..<d]], axis: 1)
            let ref = reference(x)
            eval(state, proj, x, ref.q, ref.k, ref.v)
            inputs.append((state, proj, x, ref))
        }
        var parts: [String] = []
        var allMatch = true
        for (state, proj, x, ref) in inputs {
            guard
                let fused = gatedDeltaConvNormQKV(
                    convState: state, rows: proj, rowOffset: 0, weight: weight,
                    numKHeads: hk, numVHeads: hv, headDim: hd, scales: scales, eps: 1e-6)
            else {
                parts.append("unsupported")
                allMatch = false
                continue
            }
            let qm = (bits(fused.q) .== bits(ref.q)).all().item(Bool.self)
            let km = (bits(fused.k) .== bits(ref.k)).all().item(Bool.self)
            let vm = (bits(fused.v) .== bits(ref.v)).all().item(Bool.self)
            let cm = (bits(fused.convInput) .== bits(x)).all().item(Bool.self)
            let nm = (bits(fused.nextConvState) .== bits(x[0..., (x.dim(1) - taps + 1)..., 0...]))
                .all().item(Bool.self)
            if !(cm && nm) {
                parts.append("concat\(cm ? "=" : "!=") next\(nm ? "=" : "!=")")
                allMatch = false
            }
            let qd = (fused.q.asType(.float32) - ref.q.asType(.float32)).abs().max()
                .item(Float.self)
            let vd = (fused.v.asType(.float32) - ref.v.asType(.float32)).abs().max()
                .item(Float.self)
            parts.append(
                "S=\(x.dim(1) - taps + 1) q\(qm ? "=" : "!=")(\(qd)) k\(km ? "=" : "!=") "
                    + "v\(vm ? "=" : "!=")(\(vd))")
            if !(qm && km && vm) { allMatch = false }
        }
        emit(
            "[gdnconv-microbench] fused kernel: \(allMatch ? "MATCH" : "DIFFERS") "
                + parts.joined(separator: " | "))
        let (state0, proj0, _, _) = inputs[0]
        let arms: [(String, () -> [MLXArray])] = [
            (
                "gdnconv-ops",
                {
                    let x = concatenated([state0, proj0[0..., 0..., ..<d]], axis: 1)
                    let r = reference(x)
                    return [r.q, r.k, r.v, x, contiguous(x[0..., (x.dim(1) - taps + 1)..., 0...])]
                }
            ),
            (
                "gdnconv-fused",
                {
                    let r = gatedDeltaConvNormQKV(
                        convState: state0, rows: proj0, rowOffset: 0, weight: weight,
                        numKHeads: hk, numVHeads: hv, headDim: hd, scales: scales, eps: 1e-6)!
                    return [r.q, r.k, r.v, r.convInput, r.nextConvState]
                }
            ),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[gdnconv-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
    }

    /// Drafter dynamic conv ruler: `dflash2DynamicConv` against the separate
    /// ops as `DFlash2DynamicConv.convolve` writes them (eager, and under
    /// `compile` as the segment traces run them) on the drafter's shapes,
    /// bitwise per slot, then the compiled ops and the kernel timed under the
    /// profiler windows `dconv-ops` / `dconv-fused`. `DFLASH2_DCONV_MICROBENCH=1`.
    @MainActor
    private func dynamicConvMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_DCONV_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (h, k, c) = (5120, 2, 16)
        let g = h / c
        let row = 2 * k * g
        MLXRandom.seed(11)
        let base = (MLXRandom.normal([2, k, h]) * 0.5).asType(.bfloat16)
        eval(base)
        let referenceGraph: @Sendable (MLXArray, MLXArray, MLXArray, Int) -> MLXArray = {
            hidden, dynamic, base, slot in
            let (b, l) = (hidden.dim(0), hidden.dim(1))
            let blocks = hidden.reshaped(b, l, g, c)
            let dyn = dynamic.reshaped(b, l, 2, k, g)[0..., 0..., slot, 0..., 0...]
            var output: MLXArray?
            for tap in 0..<k {
                let taps = base[slot, tap].reshaped(1, 1, g, c)
                let values =
                    tap == 0
                    ? blocks
                    : padded(
                        blocks[0..., ..<(l - tap)],
                        widths: [IntOrPair(0), IntOrPair((tap, 0)), IntOrPair(0), IntOrPair(0)])
                let term = taps * values
                output = output.map { $0 + term } ?? term
                output = output! + dyn[0..., 0..., tap, 0..., .newAxis] * values
            }
            return output!.reshaped(hidden.shape)
        }
        func reference(_ hidden: MLXArray, dynamic: MLXArray, slot: Int) -> MLXArray {
            referenceGraph(hidden, dynamic, base, slot)
        }
        func fused(_ hidden: MLXArray, dynamic: MLXArray, slot: Int) -> MLXArray? {
            dflash2DynamicConv(
                hidden, dynamic: dynamic, dynamicOffset: slot * k * g, dynamicRowLength: row,
                base: base[slot], kernelSize: k, groupSize: c)
        }
        func bits(_ x: MLXArray) -> MLXArray { contiguous(x).view(dtype: .uint16) }
        var compiled: [Int: ([MLXArray]) -> [MLXArray]] = [:]
        for slot in 0..<2 {
            compiled[slot] = compile { (args: [MLXArray]) -> [MLXArray] in
                [referenceGraph(args[0], args[1], args[2], slot)]
            }
        }
        var first: (MLXArray, MLXArray)?
        for (seed, s) in [(1, 8), (2, 1), (3, 5), (4, 7)] as [(UInt64, Int)] {
            MLXRandom.seed(seed)
            let hidden = (MLXRandom.normal([1, s, h]) * 2).asType(.bfloat16)
            let dynamic = (MLXRandom.normal([1, s, row]) * 2).asType(.bfloat16)
            eval(hidden, dynamic)
            if first == nil { first = (hidden, dynamic) }
            var parts: [String] = []
            for slot in 0..<2 {
                guard let out = fused(hidden, dynamic: dynamic, slot: slot) else {
                    parts.append("slot\(slot) unsupported")
                    continue
                }
                let eager = reference(hidden, dynamic: dynamic, slot: slot)
                let traced = compiled[slot]!([hidden, dynamic, base])[0]
                let em = (bits(out) .== bits(eager)).all().item(Bool.self)
                let tm = (bits(out) .== bits(traced)).all().item(Bool.self)
                let ed = (out.asType(.float32) - eager.asType(.float32)).abs().max().item(
                    Float.self)
                parts.append(
                    "slot\(slot) eager\(em ? "=" : "!=")(\(ed)) compiled\(tm ? "=" : "!=")")
            }
            emit("[dconv-microbench] S=\(s): " + parts.joined(separator: " | "))
        }
        let (hidden0, dynamic0) = first!
        let arms: [(String, () -> [MLXArray])] = [
            ("dconv-ops", { compiled[0]!([hidden0, dynamic0, base]) }),
            ("dconv-fused", { [fused(hidden0, dynamic: dynamic0, slot: 0)!] }),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[dconv-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
    }

    /// Fused q/k norm + RoPE ruler: `attentionNormRope` against
    /// `MLXFast.rmsNorm` → transpose → `MLXFast.RoPE` on the target's
    /// (q|gate interleave, 64 rotary dims of 256) and the drafter's (plain,
    /// full rotation) stacked-row layouts, bitwise over S = 8/1/5 and three
    /// offsets, then the ops chain and the kernel timed under the profiler
    /// windows `normrope-ops` / `normrope-fused`.
    /// `DFLASH2_NORMROPE_MICROBENCH=1`.
    @MainActor
    private func normRopeMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_NORMROPE_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        struct Layout {
            var name: String
            var hq: Int
            var hk: Int
            var hd: Int
            var rd: Int
            var gated: Bool
            var qEnd: Int { hq * hd * (gated ? 2 : 1) }
            var kEnd: Int { qEnd + hk * hd }
            var row: Int { kEnd + hk * hd }
        }
        let layouts = [
            Layout(name: "target", hq: 24, hk: 4, hd: 256, rd: 64, gated: true),
            Layout(name: "drafter", hq: 32, hk: 8, hd: 128, rd: 128, gated: false),
        ]
        let base: Float = 10_000_000
        func bits(_ x: MLXArray) -> MLXArray { contiguous(x).view(dtype: .uint16) }
        func reference(
            _ rows: MLXArray, _ lay: Layout, qw: MLXArray, kw: MLXArray, offset: MLXArray
        )
            -> (MLXArray, MLXArray)
        {
            let (b, l) = (rows.dim(0), rows.dim(1))
            var q: MLXArray
            if lay.gated {
                q =
                    rows[.ellipsis, ..<lay.qEnd].reshaped(b, l, lay.hq, -1).split(
                        parts: 2, axis: -1)[0]
            } else {
                q = rows[.ellipsis, ..<lay.qEnd].reshaped(b, l, lay.hq, lay.hd)
            }
            q = MLXFast.rmsNorm(q, weight: qw, eps: 1e-6).transposed(0, 2, 1, 3)
            var k = rows[.ellipsis, lay.qEnd..<lay.kEnd].reshaped(b, l, lay.hk, lay.hd)
            k = MLXFast.rmsNorm(k, weight: kw, eps: 1e-6).transposed(0, 2, 1, 3)
            return (
                MLXFast.RoPE(
                    q, dimensions: lay.rd, traditional: false, base: base, scale: 1, offset: offset),
                MLXFast.RoPE(
                    k, dimensions: lay.rd, traditional: false, base: base, scale: 1, offset: offset)
            )
        }
        struct Timing {
            let layout: Layout
            let rows: MLXArray
            let offset: MLXArray
            let qw: MLXArray
            let kw: MLXArray
        }
        var timing: Timing?
        for lay in layouts {
            let rope = RoPE(dimensions: lay.rd, traditional: false, base: base, scale: 1)
            MLXRandom.seed(31)
            let qw =
                (MLXRandom.normal([lay.hd]) * 0.1 + 1).asType(.bfloat16)
                * MLXArray(Float(lay.gated ? 0.0625 : 1)).asType(.bfloat16)
            let kw = (MLXRandom.normal([lay.hd]) * 0.1 + 1).asType(.bfloat16)
            eval(qw, kw)
            var cases: [(MLXArray, MLXArray, (MLXArray, MLXArray))] = []
            for (seed, s) in [(1, 8), (2, 1), (3, 5)] as [(UInt64, Int)] {
                for off in [0, 1234, 77777] as [Int32] {
                    MLXRandom.seed(seed)
                    let rows = (MLXRandom.normal([1, s, lay.row]) * 2).asType(.bfloat16)
                    let offset = MLXArray([off])
                    let ref = reference(rows, lay, qw: qw, kw: kw, offset: offset)
                    eval(rows, offset, ref.0, ref.1)
                    cases.append((rows, offset, ref))
                }
            }
            var allMatch = true
            var parts: [String] = []
            for (rows, offset, ref) in cases {
                guard
                    let fused = attentionNormRope(
                        rows: rows, queryOffset: 0,
                        queryHeadStride: lay.gated ? 2 * lay.hd : lay.hd,
                        queryHeads: lay.hq, keyOffset: lay.qEnd, keyHeadStride: lay.hd,
                        keyHeads: lay.hk, headDim: lay.hd, queryWeight: qw, keyWeight: kw,
                        eps: 1e-6,
                        rope: PlainRoPEParameters(dimensions: lay.rd, base: base, scale: 1),
                        offset: offset)
                else {
                    parts.append("unsupported")
                    allMatch = false
                    continue
                }
                let qm = (bits(fused.queries) .== bits(ref.0)).all().item(Bool.self)
                let km = (bits(fused.keys) .== bits(ref.1)).all().item(Bool.self)
                let (refQ, refK) = ref
                let qd = (fused.queries.asType(DType.float32) - refQ.asType(DType.float32))
                    .abs().max().item(Float.self)
                let kd = (fused.keys.asType(DType.float32) - refK.asType(DType.float32))
                    .abs().max().item(Float.self)
                if !(qm && km) {
                    allMatch = false
                    parts.append(
                        "S=\(rows.dim(1)) off=\(offset.item(Int32.self)) q\(qm ? "=" : "!=")(\(qd)) k\(km ? "=" : "!=")(\(kd))"
                    )
                }
            }
            emit(
                "[normrope-microbench] \(lay.name) fused kernel: "
                    + (allMatch ? "MATCH" : "DIFFERS " + parts.prefix(3).joined(separator: " | ")))
            if lay.name == "target" {
                let (rows, offset, _) = cases[0]
                timing = Timing(layout: lay, rows: rows, offset: offset, qw: qw, kw: kw)
            }
        }
        guard let timing else { return }
        let (lay, rows, offset, qw, kw) = (
            timing.layout, timing.rows, timing.offset, timing.qw, timing.kw
        )
        let rope = RoPE(dimensions: lay.rd, traditional: false, base: base, scale: 1)
        let arms: [(String, () -> [MLXArray])] = [
            (
                "normrope-ops",
                {
                    let r = reference(rows, lay, qw: qw, kw: kw, offset: offset)
                    return [r.0, r.1]
                }
            ),
            (
                "normrope-fused",
                {
                    let r = attentionNormRope(
                        rows: rows, queryOffset: 0, queryHeadStride: 2 * lay.hd, queryHeads: lay.hq,
                        keyOffset: lay.qEnd, keyHeadStride: lay.hd, keyHeads: lay.hk,
                        headDim: lay.hd,
                        queryWeight: qw, keyWeight: kw, eps: 1e-6,
                        rope: PlainRoPEParameters(dimensions: lay.rd, base: base, scale: 1),
                        offset: offset)!
                    return [r.queries, r.keys]
                }
            ),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[normrope-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
    }

    /// Selector greedy-walk ruler: `dflash2GreedyWalk` against the per-
    /// position gather → add → argmax → gather loop on the selector's shape
    /// (7 positions, 16 candidates), over random and heavily tied scores,
    /// then the compiled loop and the kernel timed under the profiler
    /// windows `walk-ops` / `walk-fused`. `DFLASH2_WALK_MICROBENCH=1`.
    @MainActor
    private func greedyWalkMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_WALK_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (l, k) = (7, 16)
        let referenceGraph: @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray = {
            unary, edges, anchorEdges, candidates in
            let length = unary.dim(1)
            var path: [MLXArray] = []
            var previous: MLXArray?
            for position in 0..<length {
                var scores = unary[0..., position]
                if let previous {
                    scores +=
                        takeAlong(
                            edges[0..., position - 1, 0..., 0...],
                            previous[0..., .newAxis, .newAxis], axis: 1
                        )[0..., 0, 0...]
                } else {
                    scores += anchorEdges
                }
                let chosen = argMax(scores, axis: -1)
                previous = chosen
                path.append(
                    takeAlong(candidates[0..., position], chosen[.newAxis, 0...], axis: -1)[
                        0..., 0])
            }
            return stacked(path, axis: 1)
        }
        let compiled = compile { (args: [MLXArray]) -> [MLXArray] in
            [referenceGraph(args[0], args[1], args[2], args[3])]
        }
        var mismatches = 0
        var first: [MLXArray]?
        for seed in 0..<40 as Range<UInt64> {
            MLXRandom.seed(seed)
            let tied = seed % 2 == 1
            func scores(_ shape: [Int]) -> MLXArray {
                tied
                    ? MLXRandom.randInt(0..<4, shape).asType(.bfloat16)
                    : (MLXRandom.normal(shape) * 4).asType(.bfloat16)
            }
            let unary = scores([1, l, k])
            let edges = scores([1, l - 1, k, k])
            let anchor = scores([1, k])
            let candidates = MLXRandom.randInt(0..<250_000, [1, l, k]).asType(.uint32)
            eval(unary, edges, anchor, candidates)
            guard
                let fused = dflash2GreedyWalk(
                    unary: unary, edges: edges, anchorEdges: anchor, candidates: candidates)
            else {
                emit("[walk-microbench] seed \(seed): unsupported")
                mismatches += 1
                continue
            }
            let eager = referenceGraph(unary, edges, anchor, candidates)
            let traced = compiled([unary, edges, anchor, candidates])[0]
            let em = (fused .== eager).all().item(Bool.self)
            let tm = (fused .== traced).all().item(Bool.self)
            if !(em && tm) {
                mismatches += 1
                emit(
                    "[walk-microbench] seed \(seed) (\(tied ? "tied" : "random")): "
                        + "eager\(em ? "=" : "!=") compiled\(tm ? "=" : "!=") fused \(fused) ref \(eager)"
                )
            }
            if first == nil { first = [unary, edges, anchor, candidates] }
        }
        emit("[walk-microbench] 40 seeds: \(mismatches == 0 ? "MATCH" : "\(mismatches) DIFFER")")
        guard let inputs = first else { return }
        let arms: [(String, () -> [MLXArray])] = [
            ("walk-ops", { compiled(inputs) }),
            (
                "walk-fused",
                {
                    [
                        dflash2GreedyWalk(
                            unary: inputs[0], edges: inputs[1], anchorEdges: inputs[2],
                            candidates: inputs[3])!
                    ]
                }
            ),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[walk-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
    }

    /// GDN scan geometry ruler: the y-only and replay kernels with 1, 2 and
    /// 4 value rows per thread on the target's verify shapes, bitwise
    /// against the one-row geometry, then each timed under the profiler
    /// windows `gdn-y-rptN` / `gdn-st-rptN`. `DFLASH2_GDN_MICROBENCH=1`.
    @MainActor
    private func gdnScanMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_GDN_ITERATIONS"] ?? "") ?? 40
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (t, hk, dk, hv, dv) = (8, 16, 128, 48, 128)
        MLXRandom.seed(11)
        let q = (MLXRandom.normal([1, t, hk, dk]) * 0.09).asType(.bfloat16)
        let k = (MLXRandom.normal([1, t, hk, dk]) * 0.09).asType(.bfloat16)
        let v = MLXRandom.normal([1, t, hv, dv]).asType(.bfloat16)
        let g = MLXRandom.uniform(low: 0.5, high: 1.0, [1, t, hv]).asType(.float32)
        let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [1, t, hv]).asType(.float32)
        let state = (MLXRandom.normal([1, hv, dv, dk]) * 0.5).asType(.float32)
        let valid = MLXArray([Int32(5)])
        eval(q, k, v, g, beta, state, valid)
        let refY = gatedDeltaOutputVariant(
            q: q, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state, rowsPerThread: 1)
        let refS = gatedDeltaStateAfterVariant(
            validCount: valid, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state,
            rowsPerThread: 1)
        eval(refY, refS)
        func bits(_ x: MLXArray) -> MLXArray {
            contiguous(x).view(dtype: x.dtype == .float32 ? .uint32 : .uint16)
        }
        for rpt in [1, 2, 4] {
            let y = gatedDeltaOutputVariant(
                q: q, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state,
                rowsPerThread: rpt)
            let st = gatedDeltaStateAfterVariant(
                validCount: valid, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state,
                rowsPerThread: rpt)
            let ym = (bits(y) .== bits(refY)).all().item(Bool.self)
            let sm = (bits(st) .== bits(refS)).all().item(Bool.self)
            emit(
                "[gdn-microbench] rpt=\(rpt): y \(ym ? "MATCH" : "DIFFERS") state \(sm ? "MATCH" : "DIFFERS")"
            )
            let arms: [(String, () -> MLXArray)] = [
                (
                    "gdn-y-rpt\(rpt)",
                    {
                        gatedDeltaOutputVariant(
                            q: q, k: k, v: v, gates: .precomputed(g: g, beta: beta), state: state,
                            rowsPerThread: rpt)
                    }
                ),
                (
                    "gdn-st-rpt\(rpt)",
                    {
                        gatedDeltaStateAfterVariant(
                            validCount: valid, k: k, v: v, gates: .precomputed(g: g, beta: beta),
                            state: state, rowsPerThread: rpt)
                    }
                ),
            ]
            for (window, launch) in arms {
                for _ in 0..<5 { eval(launch()) }
                setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
                let start = ContinuousClock.now
                for _ in 0..<iterations { eval(launch()) }
                let seconds = Self.elapsedSeconds(since: start)
                unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
                emit(
                    String(
                        format: "[gdn-microbench] %@  %.1f us/launch (wall, synced)", window,
                        seconds / Double(iterations) * 1e6))
            }
        }

        // Fused gates: the kernel computing g/beta from the projection's
        // a/b columns against the ops' precomputed pair, bitwise on y and
        // the replayed state, for the production geometry and the split
        // shapes; then timed.
        let (row, bOff, aOff) = (16480, 10240 + 6144, 10240 + 6144 + hv)
        for (seed, steps) in [(31, t), (32, 1), (33, 5)] as [(UInt64, Int)] {
            MLXRandom.seed(seed)
            let proj = (MLXRandom.normal([1, steps, row]) * 1.5).asType(.bfloat16)
            let aLog = MLX.log(MLXRandom.uniform(low: 1.0, high: 16.0, [hv])).asType(.bfloat16)
            let dtBias = (MLXRandom.normal([hv]) * 0.5).asType(.bfloat16)
            let qs = (MLXRandom.normal([1, steps, hk, dk]) * 0.09).asType(.bfloat16)
            let ks = (MLXRandom.normal([1, steps, hk, dk]) * 0.09).asType(.bfloat16)
            let vs = MLXRandom.normal([1, steps, hv, dv]).asType(.bfloat16)
            let validSteps = MLXArray([Int32(min(steps, 5))])
            eval(proj, aLog, dtBias, qs, ks, vs, validSteps)
            let source = GatedDeltaGateSource(
                aSource: proj, aOffset: aOff, bSource: proj, bOffset: bOff, aLog: aLog,
                dtBias: dtBias)
            let (g2, beta2) = gatedDeltaGates(a: source.a, b: source.b, aLog: aLog, dtBias: dtBias)
            let pre = GatedDeltaGates.precomputed(g: g2, beta: beta2)
            let fused = GatedDeltaGates.source(source)
            let yPre = gatedDeltaOutputVariant(
                q: qs, k: ks, v: vs, gates: pre, state: state, rowsPerThread: 2)
            let yFused = gatedDeltaOutputVariant(
                q: qs, k: ks, v: vs, gates: fused, state: state, rowsPerThread: 2)
            let sPre = gatedDeltaStateAfterVariant(
                validCount: validSteps, k: ks, v: vs, gates: pre, state: state, rowsPerThread: 2)
            let sFused = gatedDeltaStateAfterVariant(
                validCount: validSteps, k: ks, v: vs, gates: fused, state: state, rowsPerThread: 2)
            let ym = (bits(yFused) .== bits(yPre)).all().item(Bool.self)
            let sm = (bits(sFused) .== bits(sPre)).all().item(Bool.self)
            let yd = (yFused.asType(.float32) - yPre.asType(.float32)).abs().max().item(Float.self)
            let sd = (sFused - sPre).abs().max().item(Float.self)
            emit(
                "[gdn-microbench] fused gates S=\(steps): y \(ym ? "MATCH" : "DIFFERS") (maxdiff \(yd)) state \(sm ? "MATCH" : "DIFFERS") (maxdiff \(sd))"
            )
            if steps != t { continue }
            let arms: [(String, () -> MLXArray)] = [
                (
                    "gdn-y-fg",
                    {
                        gatedDeltaOutputVariant(
                            q: qs, k: ks, v: vs, gates: fused, state: state, rowsPerThread: 2)
                    }
                ),
                (
                    "gdn-st-fg",
                    {
                        gatedDeltaStateAfterVariant(
                            validCount: validSteps, k: ks, v: vs, gates: fused, state: state,
                            rowsPerThread: 2)
                    }
                ),
                (
                    "gdn-gates-ops",
                    { gatedDeltaGates(a: source.a, b: source.b, aLog: aLog, dtBias: dtBias).g }
                ),
            ]
            for (window, launch) in arms {
                for _ in 0..<5 { eval(launch()) }
                setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
                let start = ContinuousClock.now
                for _ in 0..<iterations { eval(launch()) }
                let seconds = Self.elapsedSeconds(since: start)
                unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
                emit(
                    String(
                        format: "[gdn-microbench] %@  %.1f us/launch (wall, synced)", window,
                        seconds / Double(iterations) * 1e6))
            }
        }
    }

    /// Memory-bandwidth ruler: a 1 GiB bf16 reduce (read) and add (read +
    /// write) against the production 4-bit QMM shapes at M = 1 and M = 8,
    /// each under a profiler window `bw-*`; the log line carries the bytes
    /// each launch moves so the per-kernel GPU time converts to GB/s.
    /// `DFLASH2_BW_MICROBENCH=1`.
    @MainActor
    private func bandwidthMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_BW_ITERATIONS"] ?? "") ?? 20
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        MLXRandom.seed(5)
        let big = MLXRandom.normal([512, 1024, 1024]).asType(.bfloat16)
        eval(big)
        let gib = Double(big.nbytes)
        var arms: [(String, Double, () -> MLXArray)] = [
            ("bw-sum-1g", gib, { big.sum() }),
            ("bw-add-1g", 2 * gib, { big + 1 }),
        ]
        for (n, k) in [(34816, 5120), (5120, 17408), (16480, 5120), (248320, 5120)] {
            let w = MLXRandom.normal([n, k]).asType(.bfloat16)
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            eval(wq, scales, biases!)
            let bytes = Double(wq.nbytes + scales.nbytes + biases!.nbytes)
            for m in [1, 8] {
                let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
                eval(x)
                arms.append(
                    (
                        "bw-qmm-m\(m)-n\(n)-k\(k)", bytes,
                        {
                            quantizedMM(
                                x, wq, scales: scales, biases: biases, transpose: true,
                                groupSize: 64,
                                bits: 4)
                        }
                    ))
            }
        }
        for (window, bytes, launch) in arms {
            for _ in 0..<3 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            let perLaunch = seconds / Double(iterations)
            emit(
                String(
                    format: "[bw-microbench] %@ bytes=%.0f  %.1f us/launch wall  %.0f GB/s wall",
                    window, bytes, perLaunch * 1e6, bytes / perLaunch / 1e9))
        }
    }

    /// Gated output norm ruler: `gatedDeltaNormGate` against the compiled
    /// `rmsNorm` + `silu(z.f32) * x.f32` chain on the GDN output shapes,
    /// bitwise, then both timed under the profiler windows `gdngate-ops` /
    /// `gdngate-fused`. `DFLASH2_GDNGATE_MICROBENCH=1`.
    @MainActor
    private func gdnGateMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_GDNGATE_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (hv, hd, row, zOff) = (48, 128, 16480, 10240)
        MLXRandom.seed(21)
        let weight = (MLXRandom.normal([hd]) * 0.1 + 1).asType(.bfloat16)
        eval(weight)
        let compiledChain = compile { (inputs: [MLXArray]) -> [MLXArray] in
            let (x, proj, w) = (inputs[0], inputs[1], inputs[2])
            let b = x.dim(0)
            let s = x.dim(1)
            let z = proj[0..., 0..., zOff..<(zOff + hv * hd)].reshaped(b, s, hv, hd)
            let n = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
            return [(silu(z.asType(.float32)) * n.asType(.float32)).asType(.bfloat16)]
        }
        func compiledReference(_ x: MLXArray, _ proj: MLXArray) -> MLXArray {
            compiledChain([x, proj, weight])[0]
        }
        func plainReference(_ x: MLXArray, _ proj: MLXArray) -> MLXArray {
            let z = proj[0..., 0..., zOff..<(zOff + hv * hd)].reshaped(x.dim(0), x.dim(1), hv, hd)
            let n = MLXFast.rmsNorm(x, weight: weight, eps: 1e-6)
            return (silu(z.asType(.float32)) * n.asType(.float32)).asType(.bfloat16)
        }
        func bits(_ x: MLXArray) -> MLXArray { contiguous(x).view(dtype: .uint16) }
        var timing: (MLXArray, MLXArray)?
        for (seed, s) in [(1, 8), (2, 1), (3, 5)] as [(UInt64, Int)] {
            MLXRandom.seed(seed)
            let x = (MLXRandom.normal([1, s, hv, hd]) * 2).asType(.bfloat16)
            let proj = (MLXRandom.normal([1, s, row]) * 2).asType(.bfloat16)
            eval(x, proj)
            let ref = compiledReference(x, proj)
            let plain = plainReference(x, proj)
            guard
                let fused = gatedDeltaNormGate(
                    x, gateSource: proj, gateOffset: zOff, gateRowLength: row, weight: weight,
                    eps: 1e-6)
            else {
                emit("[gdngate-microbench] S=\(s): unsupported")
                continue
            }
            let m = (bits(fused) .== bits(ref)).all().item(Bool.self)
            let mp = (bits(fused) .== bits(plain)).all().item(Bool.self)
            let cp = (bits(ref) .== bits(plain)).all().item(Bool.self)
            let d = (fused.asType(.float32) - ref.asType(.float32)).abs().max().item(Float.self)
            emit(
                "[gdngate-microbench] S=\(s): fused vs compiled \(m ? "MATCH" : "DIFFERS") "
                    + "(maxdiff \(d)); fused vs plain ops \(mp ? "MATCH" : "DIFFERS"); "
                    + "compiled vs plain ops \(cp ? "MATCH" : "DIFFERS")")
            if s == 8 { timing = (x, proj) }
        }
        guard let timing else { return }
        let (x, proj) = timing
        let arms: [(String, () -> MLXArray)] = [
            ("gdngate-ops", { compiledReference(x, proj) }),
            (
                "gdngate-fused",
                {
                    gatedDeltaNormGate(
                        x, gateSource: proj, gateOffset: zOff, gateRowLength: row, weight: weight,
                        eps: 1e-6)!
                }
            ),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[gdngate-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
    }

    /// Selector top-k ruler: `topKIndices` against `argPartition` on the
    /// drafter's `[1, 7, V]` bf16 logits, exact-order parity over seeds with
    /// and without heavy ties, then both timed under the profiler windows
    /// `topk-argp` / `topk-kernel`. `DFLASH2_TOPK_MICROBENCH=1`;
    /// `DFLASH2_TOPK_ITERATIONS=N` overrides the default.
    @MainActor
    private func topKMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let iterations = Int(env["DFLASH2_TOPK_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        let (rows, vocab, k) = (7, 248320, 16)
        let cases: [(String, (MLXArray) -> MLXArray)] = [
            ("normal", { $0 * 4 }),
            ("ties", { ($0 * 3).round() }),
            ("wide", { $0 * 40 }),
        ]
        var mismatches = 0
        for (seed, (label, shape)) in cases.enumerated() {
            MLXRandom.seed(UInt64(seed + 3))
            let logits = shape(MLXRandom.normal([1, rows, vocab])).asType(.bfloat16)
            eval(logits)
            let reference = argPartition(logits, kth: vocab - k, axis: -1)[
                .ellipsis, (vocab - k)...]
            let candidate = topKIndices(logits, k: k)
            let equal = (reference .== candidate).all().item(Bool.self)
            let sameSet = sorted(reference, axis: -1) .== sorted(candidate, axis: -1)
            let sameSetAll = sameSet.all().item(Bool.self)
            if !equal { mismatches += 1 }
            emit(
                "[topk-microbench] \(label): order \(equal ? "MATCH" : "DIFFERS") "
                    + "set \(sameSetAll ? "MATCH" : "DIFFERS") "
                    + "shape \(candidate.shape) dtype \(candidate.dtype)")
        }
        MLXRandom.seed(3)
        let logits = (MLXRandom.normal([1, rows, vocab]) * 4).asType(.bfloat16)
        eval(logits)
        let arms: [(String, () -> MLXArray)] = [
            (
                "topk-argp",
                { argPartition(logits, kth: vocab - k, axis: -1)[.ellipsis, (vocab - k)...] }
            ),
            ("topk-kernel", { topKIndices(logits, k: k) }),
        ]
        for (window, launch) in arms {
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", window, 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            emit(
                String(
                    format: "[topk-microbench] %@  %.1f us/launch (wall, synced)",
                    window, seconds / Double(iterations) * 1e6))
        }
        emit("[topk-microbench] mismatching cases: \(mismatches)")
    }

    /// Attention ruler for the verify shape at long context: `q [1, 24, 8,
    /// 256]` against `k/v [1, 4, N, 256]` with the verify's bool mask (every
    /// row sees the whole context and the block causally), checked against
    /// an unfused f32 reference and timed under the profiler window
    /// `sdpa-N`. `DFLASH2_SDPA_MICROBENCH=1`; `DFLASH2_SDPA_LENGTHS=N,N,...`
    /// and `DFLASH2_SDPA_ITERATIONS=N` override the defaults.
    @MainActor
    private func sdpaMicrobench() throws {
        let env = ProcessInfo.processInfo.environment
        let lengths = (env["DFLASH2_SDPA_LENGTHS"] ?? "1536,4096,8192,16384")
            .split(separator: ",").compactMap { Int($0) }
        let iterations = Int(env["DFLASH2_SDPA_ITERATIONS"] ?? "") ?? 50
        let (emit, handle) = microbenchEmitter()
        defer { try? handle?.close() }
        emit("[sdpa-microbench] \(lengths.count) lengths, \(iterations) synced launches each")
        let (heads, kvHeads, rows, dim) = (24, 4, 8, 256)
        let scale = pow(Float(dim), -0.5)
        for n in lengths {
            MLXRandom.seed(11)
            let q = MLXRandom.normal([1, heads, rows, dim], dtype: .bfloat16)
            let k = MLXRandom.normal([1, kvHeads, n, dim], dtype: .bfloat16)
            let v = MLXRandom.normal([1, kvHeads, n, dim], dtype: .bfloat16)
            let columns = MLXArray(Int32(0)..<Int32(n)).expandedDimensions(axis: 0)
            let limits = (MLXArray(Int32(0)..<Int32(rows)) + Int32(n - rows))
                .expandedDimensions(axis: 1)
            let mask = columns .<= limits
            eval(q, k, v, mask)
            let launch = {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
            }
            let gqa = heads / kvHeads
            let kFull = repeated(k, count: gqa, axis: 1).asType(.float32)
            let vFull = repeated(v, count: gqa, axis: 1).asType(.float32)
            let scores = matmul(q.asType(.float32) * scale, kFull.transposed(0, 1, 3, 2))
            let probabilities = softmax(which(mask, scores, MLXArray(Float(-1e30))), axis: -1)
            let reference = matmul(probabilities, vFull)
            let out = launch()
            let error = (out.asType(.float32) - reference).abs().max().item(Float.self)
            let magnitude = reference.abs().max().item(Float.self)
            for _ in 0..<10 { eval(launch()) }
            setenv("MLX_KERNEL_PROFILE_ACTIVE", "sdpa-\(n)", 1)
            let start = ContinuousClock.now
            for _ in 0..<iterations { eval(launch()) }
            let seconds = Self.elapsedSeconds(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            let perLaunch = seconds / Double(iterations)
            let bytes = Double(2 * kvHeads * n * dim * 2)
            emit(
                String(
                    format: "[sdpa-microbench] N=%d  %.1f us/launch  %.0f GB/s (K+V)  "
                        + "maxerr %.5f (of %.3f)",
                    n, perLaunch * 1e6, bytes / perLaunch / 1e9, error, magnitude))
            Memory.clearCache()
        }
    }

    /// The microbench line sink: the bench log file plus the process log.
    @MainActor
    private func microbenchEmitter() -> ((String) -> Void, FileHandle?) {
        let outputDir = runner.activeConfig.outputDir
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let logURL = outputDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let handle = FileHandle(forWritingAtPath: logURL.path)
        let emit: (String) -> Void = { line in
            handle?.write(Data((line + "\n").utf8))
            Self.log(line)
        }
        return (emit, handle)
    }

    @MainActor
    func run() async throws {
        // Pipelined-round default (ledger R36). Must precede the first MLX
        // eval: the command-buffer cap is latched on first use. `overwrite: 0`
        // keeps explicit overrides from the command line. The 10-buffer cap
        // re-throttles the round seam once the next verify is scheduled a
        // round ahead.
        setenv("MLX_MAX_ACTIVE_TASKS", "40", 0)
        // In-place verify-row writes into the attention caches (mlx fork C9):
        // the rows are written by a dynamic slice update whose readers all
        // precede it on the stream, so the whole-store copy per pass goes.
        setenv("MLX_DYNSLICE_INPLACE", "1", 0)
        if ProcessInfo.processInfo.environment["DFLASH2_QMM_MICROBENCH"] != nil {
            try qmmMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_RMSRES_MICROBENCH"] != nil {
            try rmsResidualMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_GDNCONV_MICROBENCH"] != nil {
            try gdnConvNormMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_GDN_MICROBENCH"] != nil {
            try gdnScanMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_BW_MICROBENCH"] != nil {
            try bandwidthMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_DCONV_MICROBENCH"] != nil {
            try dynamicConvMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_NORMROPE_MICROBENCH"] != nil {
            try normRopeMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_WALK_MICROBENCH"] != nil {
            try greedyWalkMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_GDNGATE_MICROBENCH"] != nil {
            try gdnGateMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_TOPK_MICROBENCH"] != nil {
            try topKMicrobench()
            return
        }
        if ProcessInfo.processInfo.environment["DFLASH2_SDPA_MICROBENCH"] != nil {
            try sdpaMicrobench()
            return
        }
        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        Self.log("[dflash2-bench] loading model: \(modelDir.path)")
        try await engine.llmActor.loadModel(from: modelDir, visionMode: false, speculation: .off)
        Self.log("[dflash2-bench] model loaded")

        guard
            let draftDir = DFlash2Support.draftDirectory(
                storageRoot: ModelDownloadManager.modelStorageURL)
        else {
            Self.log("[dflash2-bench] FATAL: draft not downloaded")
            throw DFlash2BenchError.draftMissing
        }

        // The harness runs via `open -W` (nice 0 — the ledger discipline), so
        // the report lands in the bench log file, not stdout. Write every
        // line as it is produced: a mid-bench crash must not lose the runs
        // that already finished (last time a String(format:) crash left an
        // empty log).
        let outputDir = runner.activeConfig.outputDir
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let logURL = outputDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let handle = FileHandle(forWritingAtPath: logURL.path)
        let emit: @Sendable (String) -> Void = { line in
            handle?.write(Data((line + "\n").utf8))
            Self.log(line)
        }
        defer { try? handle?.close() }

        let results = try await engine.llmActor.withModelContainer { container in
            try await container.perform { context in
                try await Self.benchAll(context: context, draftDir: draftDir, emit: emit)
            }
        }

        // Summary: median tok/s per arm.
        emit("[dflash2-bench] === summary ===")
        let arms =
            ["ar"]
            + Self.blockSizes().map { "dflash2-bs\($0)" }
        var medians: [String: Double] = [:]
        for arm in arms {
            let rates = results.filter { $0.arm == arm }.map(\.tokPerSec).sorted()
            guard !rates.isEmpty else { continue }
            let median = (rates[(rates.count - 1) / 2] + rates[rates.count / 2]) / 2
            medians[arm] = median
            let accepted = results.filter { $0.arm == arm }.reduce(0) { $0 + $1.accepted }
            let proposed = results.filter { $0.arm == arm }.reduce(0) { $0 + $1.proposed }
            let accStr =
                proposed > 0
                ? String(
                    format: "  acceptance %.1f%% (%d/%d)",
                    100.0 * Double(accepted) / Double(proposed), accepted, proposed)
                : ""
            emit(
                "[dflash2-bench] \(arm.paddedToColumn) median \(String(format: "%6.1f", median)) tok/s\(accStr)"
            )
        }
        // Check every measured stream, including repeated runs.
        if let arFingerprint = results.first(where: { $0.arm == "ar" })?.fingerprint {
            var mismatched = false
            for result in results {
                let fp = result.fingerprint
                let diverge =
                    zip(fp, arFingerprint).enumerated().first(where: {
                        $0.element.0 != $0.element.1
                    })?.offset
                    ?? (fp.count == arFingerprint.count ? nil : min(fp.count, arFingerprint.count))
                mismatched = mismatched || fp != arFingerprint
                emit(
                    "[dflash2-bench] \(result.arm) run\(result.runIndex) identity \(Self.captureFullStream ? "full stream" : "first 8"): \(fp == arFingerprint ? "MATCH" : "DIVERGED at +\(diverge.map(String.init) ?? "?")")"
                )
            }
            if Self.check, mismatched { throw DFlash2BenchError.outputMismatch }
        }
        if let base = medians["ar"] {
            for arm in arms where arm != "ar" {
                if let m = medians[arm] {
                    emit(
                        "[dflash2-bench] \(arm.paddedToColumn) speedup \(String(format: "%.2f", m / base))x"
                    )
                }
            }
        }
    }

    /// The whole bench inside one Metal-affine batch: prepare the prompt,
    /// load the draft, then ABBA-interleave the arms (decode-only timing —
    /// iterator construction/prefill happens before the clock starts).
    private static func benchAll(
        context: ModelContext, draftDir: URL, emit: (String) -> Void
    ) async throws -> [ArmResult] {
        let maxNewTokens = positiveOption("--bench-max-tokens", default: 192)
        let fingerprintLimit = captureFullStream ? maxNewTokens : 8

        // Stack same-input projections into one QMM each (bitwise-neutral,
        // applies to both arms via the shared model).
        let stacked = stackSameInputProjections(in: context.model)
        // The stacking transiently duplicated the weights; the freed
        // originals sit in the buffer cache and crowd GPU residency —
        // release them before the runs.
        GPU.clearCache()
        emit("[dflash2-bench] same-input projections stacked in \(stacked) blocks")

        let promptText = try buildPromptText()
        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(promptText)]))
        let promptTokens = prepared.text.tokens.dim(-1)
        // Acceptance references are only comparable across identical prompt
        // bytes (ledger R55): every bank records the prompt hash.
        emit("[dflash2-bench] prompt sha256 \(promptSHA256(promptText))")

        let draft: any DFlash2DrafterModel
        if draftPolicy == "4bit" {
            draft = try DFlash2Support.loadDrafter(directory: draftDir)
        } else {
            let model = DFlash2DraftModel(
                try DFlash2Support.draftConfiguration(directory: draftDir))
            try await loadWeights(modelDirectory: draftDir, model: model)
            switch draftPolicy {
            case "8bit": quantize(model: model, groupSize: 64, bits: 8)
            case "unquantized": break
            case "fc8", "selector8", "fc-selector8":
                quantize(model: model) { path, _ in
                    let higher =
                        (path == "fc" && draftPolicy != "selector8")
                        || (path.hasPrefix("candidate_selector.") && draftPolicy != "fc8")
                    return (groupSize: 64, bits: higher ? 8 : 4, mode: .affine)
                }
            default: throw DFlash2BenchError.unsupportedDraftPolicy
            }
            eval(model)
            draft = model
        }
        let stackedDraft = stackSameInputProjections(in: draft)
        emit("[dflash2-bench] drafter projections stacked in \(stackedDraft) blocks")

        func runAR(_ runIndex: Int, prepared: LMInput = prepared) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            let prefillStart = ContinuousClock.now
            var iterator = try TokenIterator(
                input: prepared, model: context.model, cache: nil,
                parameters: parameters, components: components(label: "ar"))
            let prefillSeconds = elapsedSeconds(since: prefillStart)
            // Diagnostic window for the mlx fork's MLX_KERNEL_PROFILE probe.
            setenv("MLX_KERNEL_PROFILE_ACTIVE", "ar", 1)
            defer { unsetenv("MLX_KERNEL_PROFILE_ACTIVE") }
            let start = ContinuousClock.now
            var tokens = 0
            var fingerprint: [Int] = []
            var allTokens: [Int] = []
            while let token = iterator.next() {
                if fingerprint.count < fingerprintLimit { fingerprint.append(token) }
                allTokens.append(token)
                tokens += 1
            }
            let seconds = elapsedSeconds(since: start)
            // DFLASH2_DUMP_TEXT=1: decode the AR output for quality
            // eyeballing (a trajectory shift is only acceptable if the
            // content stays coherent).
            if runIndex == 0,
                ProcessInfo.processInfo.environment["DFLASH2_DUMP_TEXT"] == "1"
            {
                emit(
                    "[dflash2-bench] ar-text run0: \(context.tokenizer.decode(tokenIds: allTokens))"
                )
            }
            return ArmResult(
                arm: "ar", runIndex: runIndex, decodeSeconds: seconds,
                tokens: tokens, accepted: 0, proposed: 0, rounds: 0,
                prefillSeconds: prefillSeconds, roundTimings: [], fingerprint: fingerprint)
        }

        func runDFlash2(
            _ runIndex: Int, blockSize: Int, prepared: LMInput = prepared
        ) throws -> ArmResult {
            var parameters = GenerateParameters(maxTokens: maxNewTokens)
            parameters.temperature = 0
            let cache = try context.model.newCache(parameters: parameters)
            let prefillStart = ContinuousClock.now
            var iterator = try DFlash2SpeculativeTokenIterator(
                input: prepared, mainModel: context.model, drafter: draft,
                mainCache: cache, parameters: parameters, blockSize: blockSize,
                components: components(label: "dflash2-bs\(blockSize)"))
            let prefillSeconds = elapsedSeconds(since: prefillStart)
            // Diagnostic window for the mlx fork's MLX_KERNEL_PROFILE probe.
            setenv("MLX_KERNEL_PROFILE_ACTIVE", "spec", 1)
            defer { unsetenv("MLX_KERNEL_PROFILE_ACTIVE") }
            let start = ContinuousClock.now
            var tokens = 0
            var fingerprint: [Int] = []
            var timings: [RoundTiming] = []
            let timeRounds = arguments.contains("--bench-round-timings")
            var lastRound = start
            var lastProposed = 0
            var lastAccepted = 0
            while let token = iterator.next() {
                if fingerprint.count < fingerprintLimit { fingerprint.append(token) }
                tokens += 1
                if timeRounds, iterator.proposedCount != lastProposed {
                    let now = ContinuousClock.now
                    timings.append(
                        RoundTiming(
                            width: iterator.proposedCount - lastProposed + 1,
                            accepted: iterator.acceptedCount - lastAccepted,
                            milliseconds: elapsedSeconds(since: lastRound) * 1000))
                    lastRound = now
                    lastProposed = iterator.proposedCount
                    lastAccepted = iterator.acceptedCount
                }
            }
            let seconds = elapsedSeconds(since: start)
            let rounds = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            iterator.finalizeGeneration()
            // Drain speculative lookahead before the next arm starts its clock.
            Stream.gpu.synchronize()
            return ArmResult(
                arm: "dflash2-bs\(blockSize)", runIndex: runIndex,
                decodeSeconds: seconds, tokens: tokens,
                accepted: iterator.acceptedCount, proposed: iterator.proposedCount,
                rounds: rounds, prefillSeconds: prefillSeconds,
                roundTimings: timings, fingerprint: fingerprint)
        }

        func report(_ result: ArmResult) {
            let acc =
                result.proposed > 0
                ? " accepted=\(result.accepted)/\(result.proposed) rounds=\(result.rounds)"
                    + String(
                        format: " tokens/round=%.2f ms/round=%.2f",
                        Double(result.accepted + result.rounds) / Double(max(1, result.rounds)),
                        result.decodeSeconds * 1000 / Double(max(1, result.rounds))) : ""
            emit(
                "[dflash2-bench] \(result.arm.paddedToColumn) run\(result.runIndex): "
                    + String(
                        format: "%6.1f tok/s (%d tokens in %.2fs)", result.tokPerSec, result.tokens,
                        result.decodeSeconds)
                    + acc)
            emit(String(format: "[dflash2-bench] prefill %.2fs", result.prefillSeconds))
        }

        var results: [ArmResult] = []
        emit(
            String(
                format: "[dflash2-bench] prompt: %d tokens; %d new per run", promptTokens,
                maxNewTokens))
        emit("[dflash2-bench] draft loaded (\(draftPolicy))")
        let blocks = Self.blockSizes()

        // `--bench-prompt-variants N`: the acceptance-spread ruler. Draft
        // acceptance is a property of one exact greedy trajectory (ledger
        // R44/R54/R55: the same content class rolled 45.7%, 33.6%, 26.7%,
        // 21.6% across one-line prompt edits), so a single prompt cannot
        // price a tokens/round lever. This mode decodes N near-identical
        // prompts (the docs body shifted by 97 characters per variant, same
        // question) once each per arm and reports the spread; the canonical
        // ABBA run below is untouched when the flag is absent.
        if let variants = Self.promptVariants(), variants > 0 {
            try await benchVariants(
                variants, context: context, blocks: blocks, emit: emit,
                runAR: runAR, runDFlash2: runDFlash2)
            return []
        }

        if fast {
            if check {
                let ar = try runAR(0)
                results.append(ar)
                report(ar)
            }
            for run in 0..<positiveOption("--bench-runs", default: 1) {
                for blockSize in blocks {
                    let result = try runDFlash2(run, blockSize: blockSize)
                    results.append(result)
                    report(result)
                }
            }
            if !check {
                emit("[dflash2-bench] identity NOT CHECKED (--bench-check enables full comparison)")
            }
        } else {
            for round in 0..<positiveOption("--bench-runs", default: 2) {
                let a = try runAR(round * 2)
                results.append(a)
                report(a)
                for blockSize in blocks {
                    let b = try runDFlash2(round, blockSize: blockSize)
                    results.append(b)
                    report(b)
                }
                let a2 = try runAR(round * 2 + 1)
                results.append(a2)
                report(a2)
            }
        }
        if let path = option("--bench-json") {
            struct Report: Encodable {
                let promptSHA256: String
                let promptTokens: Int
                let maxNewTokens: Int
                let model: String
                let draftPolicy: String
                let fullIdentityCheck: Bool
                let capturedFullStream: Bool
                let identityMatch: Bool?
                let sourceRevision: String?
                let inputTokenIds: [Int32]
                let results: [ArmResult]
            }
            let report = Report(
                promptSHA256: promptSHA256(promptText), promptTokens: promptTokens,
                maxNewTokens: maxNewTokens, model: context.configuration.name,
                draftPolicy: draftPolicy,
                fullIdentityCheck: check,
                capturedFullStream: captureFullStream,
                identityMatch: check
                    ? results.allSatisfy { $0.fingerprint == results.first?.fingerprint } : nil,
                sourceRevision: option("--bench-source-revision"),
                inputTokenIds: prepared.text.tokens.asArray(Int32.self), results: results)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path), options: .atomic)
        }
        return results
    }

    /// `--bench-blocks 3,4,5,8` overrides the default [8] arm. A trailing
    /// `f` (the retired fixed-width marker, e.g. `8f`) is accepted and
    /// ignored so older bench recipes keep working.
    private static func blockSizes() -> [Int] {
        func parse(_ raw: String) -> [Int] {
            raw.split(separator: ",").compactMap { token in
                Int(token.hasSuffix("f") ? token.dropLast() : token)
            }
        }
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--bench-blocks"), i + 1 < args.count else {
            return parse("8")
        }
        let parsed = parse(args[i + 1])
        return parsed.isEmpty ? parse("8") : parsed
    }

    private struct VariantRow {
        let arm: String
        let tokPerSec: Double
        let acceptance: Double
        let match: Bool
    }

    /// `--bench-prompt-variants N`: the acceptance-spread ruler. Draft
    /// acceptance is a property of one exact greedy trajectory (ledger
    /// R44/R54/R55/R56: the same content class rolled 45.7%, 33.6%, 26.7%,
    /// 21.6% across one-line prompt edits), so a single prompt cannot price a
    /// tokens/round lever. Decodes N near-identical prompts (the docs body
    /// shifted by 97 characters per variant, same question) once per arm and
    /// reports the spread; the canonical ABBA run is untouched without the flag.
    private static func benchVariants(
        _ variants: Int, context: ModelContext, blocks: [Int],
        emit: (String) -> Void,
        runAR: (Int, LMInput) throws -> ArmResult,
        runDFlash2: (Int, Int, LMInput) throws -> ArmResult
    ) async throws {
        var rows: [VariantRow] = []
        for variant in 0..<variants {
            let text = try buildPromptText(variant: variant)
            let prepared = try await context.processor.prepare(
                input: UserInput(chat: [.user(text)]))
            emit(
                "[dflash2-bench] variant \(variant): prompt sha256 \(promptSHA256(text)) "
                    + "(\(prepared.text.tokens.dim(-1)) tokens)")
            let ar = try runAR(variant, prepared)
            for blockSize in blocks {
                let spec = try runDFlash2(variant, blockSize, prepared)
                let rounds = (spec.proposed + blockSize - 2) / (blockSize - 1)
                let perRound =
                    rounds > 0
                    ? String(format: " tok/round %.2f", Double(spec.tokens) / Double(rounds))
                    : ""
                let match = spec.fingerprint == ar.fingerprint
                emit(
                    "[dflash2-bench] variant \(variant) \(spec.arm.paddedToColumn) "
                        + String(format: "%6.1f tok/s", spec.tokPerSec)
                        + " accepted=\(spec.accepted)/\(spec.proposed)" + perRound
                        + String(format: " ar %.1f", ar.tokPerSec)
                        + " identity \(match ? "MATCH" : "DIVERGED")")
                rows.append(
                    VariantRow(
                        arm: spec.arm, tokPerSec: spec.tokPerSec,
                        acceptance: Double(spec.accepted) / Double(max(1, spec.proposed)),
                        match: match))
            }
        }
        emit("[dflash2-bench] === variant summary (n=\(variants)) ===")
        for blockSize in blocks {
            let arm = "dflash2-bs\(blockSize)"
            let armRows = rows.filter { $0.arm == arm }
            guard !armRows.isEmpty else { continue }
            let acc = armRows.map(\.acceptance)
            let rates = armRows.map(\.tokPerSec)
            emit(
                "[dflash2-bench] \(arm.paddedToColumn) acceptance mean "
                    + String(
                        format: "%.1f%% min %.1f%% max %.1f%%",
                        100 * acc.reduce(0, +) / Double(acc.count), 100 * (acc.min() ?? 0),
                        100 * (acc.max() ?? 0))
                    + String(
                        format: " | tok/s mean %.1f min %.1f max %.1f",
                        rates.reduce(0, +) / Double(rates.count), rates.min() ?? 0,
                        rates.max() ?? 0)
                    + " | identity \(armRows.filter(\.match).count)/\(armRows.count) MATCH")
        }
    }

    /// `--bench-prompt-variants N` (nil when absent).
    private static func promptVariants() -> Int? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--bench-prompt-variants"), i + 1 < args.count else {
            return nil
        }
        return Int(args[i + 1])
    }

    private static func promptSHA256(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }

    private static func log(_ line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }

    /// Long-context workload: the tesseract repo's own docs plus a question
    /// (mirrors research/bench_dflash.py for cross-stack comparability).
    /// `DFLASH2_BENCH_PROMPT=repeat` swaps in a tiled predictable paragraph —
    /// the agent-typical high-acceptance regime (the docs prompt is the
    /// adversarial low-acceptance one).
    /// `variant` > 0 shifts the docs body start by 97 characters per step
    /// (the `--bench-prompt-variants` acceptance-spread ruler); 0 is the
    /// canonical prompt.
    private static func buildPromptText(variant: Int = 0) throws -> String {
        if let path = option("--bench-prompt-file") {
            // Explicit fixtures must never silently fall back to another workload.
            return try String(contentsOfFile: path, encoding: .utf8)
        }
        // `DFLASH2_BENCH_PROMPT_FILE=<path>`: the file's contents become the
        // user message verbatim (content-class probes: math, code, chat).
        if let path = ProcessInfo.processInfo.environment["DFLASH2_BENCH_PROMPT_FILE"],
            let text = try? String(contentsOfFile: path, encoding: .utf8)
        {
            return text
        }
        if ProcessInfo.processInfo.environment["DFLASH2_BENCH_PROMPT"] == "repeat" {
            let sentence =
                "func fibonacci(_ n: Int) -> Int { n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2) }\n"
            let tiled = String(repeating: sentence, count: 400)
            return String(tiled.prefix(24_000))
                + "\n\nQuestion: rewrite the function with an iterative loop.\nAnswer:"
        }
        var parts: [String] = []
        let repo = "/Users/owl/projects/tesseract"
        for rel in ["ARCHITECTURE.md", "CONTEXT.md", "AGENTS.md"] {
            if let text = try? String(contentsOfFile: "\(repo)/\(rel)", encoding: .utf8) {
                parts.append(text)
            }
        }
        if let adrs = try? FileManager.default.contentsOfDirectory(
            atPath: "\(repo)/docs/adr")
        {
            for name in adrs.filter({ $0.hasSuffix(".md") }).sorted().prefix(12) {
                if let text = try? String(
                    contentsOfFile: "\(repo)/docs/adr/\(name)", encoding: .utf8)
                {
                    parts.append(text)
                }
            }
        }
        var joined = parts.joined(separator: "\n\n")
        if variant > 0 {
            joined = String(joined.dropFirst(97 * variant))
        }
        // `--bench-context-mult N` tiles the prompt body N times before
        // truncation, scaling the context for the long-KV regime (decode-only
        // timing makes the repeated prefill cost irrelevant).
        var body = String(joined.prefix(24_000))
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--bench-context-mult"), i + 1 < args.count,
            let mult = Int(args[i + 1]), mult > 1
        {
            body = String(
                (0..<mult).map { _ in body }.joined(separator: "\n\n").prefix(24_000 * mult))
        }
        return body
            + "\n\nQuestion: summarize the key architectural decisions in one paragraph.\nAnswer:"
    }
}

private enum DFlash2BenchError: Error {
    case draftMissing
    case outputMismatch
    case unsupportedDraftPolicy
}

extension String {
    /// Fixed-width arm column for bench lines. Replaces `String(format: "%-12s")` —
    /// `%s` on a Swift String bridges to an object pointer, not a `char *`, and
    /// segfaulted the first bench run (DiagnosticReports 2026-08-19 22:19).
    nonisolated fileprivate var paddedToColumn: String {
        padding(toLength: 12, withPad: " ", startingAt: 0)
    }
}
