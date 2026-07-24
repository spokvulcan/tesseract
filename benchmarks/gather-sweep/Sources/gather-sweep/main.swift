import Foundation
import Metal
import MLX

// M5 probe: fused causal-mask + softmax for the SDPA ops-fallback (prefill).
//
// Production chain (fast.cpp `fallback` lambda, causal case), per layer-chunk:
//   1. mask = greater_equal(arange(offset, qL+offset)[qL,1], arange(0, kL)[1,kL])
//   2. scores = where(mask, scores, bf16(finfo.min))   // exact select
//   3. scores = softmax(scores, axis: -1, precise: true)
//      -> looped_softmax_precise_bfloat16 (softmax_looped<bfloat16_t, float, 4>)
//         for axis_size > 4096, threadgroup = maxTotalThreadsPerThreadgroup.
//
// Fused kernel: ONE MLXFast kernel per scores row: causal select in-register
// at BOTH load sites of the verbatim softmax_looped body (it reads in[] once
// for the max/sum pass and once for the write pass).
//
// Verified facts (this session):
//   - MLXFast JIT: fastMathEnabled(false) (device.cpp:619); prebuilt metallib:
//     MTL_FAST_MATH=YES (project.pbxproj). M4 proved verbatim arithmetic in
//     MLXFast JIT reproduces the metallib bitwise for qmv; softmax adds two
//     new risk sites: fast::exp internals and `1 / normalizer`.
//   - The bf16 `finfo.min` constant only ever feeds exp(min - max) -> 0 (every
//     row has >= 1 valid causal element), so its exact bits cannot change the
//     output; we still use the exact constant -1.9921875 * 2^127 (0xFF7F).

let QL = 1024
let N_HEADS = 8

// MARK: - Reference chain (mirrors the fallback lambda exactly)

/// finfo(bfloat16).min as a bf16 scalar, matching
/// `array(finfo(scores.dtype()).min, scores.dtype())` in the fallback.
nonisolated(unsafe) let bf16Min = MLXArray(bfloat16: Float(-3.3895313892515355e38))

func causalMask(qL: Int, kL: Int, offset: Int) -> MLXArray {
    let qIdx = MLXArray.arange(offset, qL + offset)  // [qL] int32
    let kIdx = MLXArray.arange(0, kL)  // [kL] int32
    let qE = qIdx.expandedDimensions(axis: 1)  // [qL, 1]
    let kE = kIdx.expandedDimensions(axis: 0)  // [1, kL]
    return qE .>= kE  // [qL, kL] bool
}

func chainY(_ scores: MLXArray, mask: MLXArray) -> MLXArray {
    let masked = MLX.where(mask, scores, bf16Min)
    return MLX.softmax(masked, axis: -1, precise: true)
}

func maskWhereY(_ scores: MLXArray, mask: MLXArray) -> MLXArray {
    MLX.where(mask, scores, bf16Min)
}

func softmaxY(_ scores: MLXArray) -> MLXArray {
    MLX.softmax(scores, axis: -1, precise: true)
}

// MARK: - Fused kernel source
//
// Body = softmax_looped<bfloat16_t, float, N_READS=4> from
// mlx/backend/metal/kernels/softmax.h, verbatim except:
//   - kernel-signature attributes rebound to locals (MLXFast style),
//   - axis_size/cofs/qL read from params,
//   - causal select injected at the two in[] load sites,
//   - rowlim = cofs + (gid % qL): max valid column for this row.
// Variant switches (bitwise-risk sites, used only if v1 mismatches):
//   rv=1: `1 / normalizer` (verbatim)   rv=2: `1.0f / normalizer` (same)
//   rv=3: precise::divide               rv=4: fast::divide
//   ev=1: fast::exp (verbatim)          ev=2: precise::exp

func fusedHeader(ev: Int) -> String {
    let expFn = ev == 2 ? "precise::exp(x)" : "fast::exp(x)"
    return """
        constant constexpr float BF16_MINF = -0x1.FEp+127f;  // bf16 0xFF7F as f32 (exact)
        template <typename T>
        inline T softmax_exp(T x) {
          return \(expFn);
        }
        """
}

func fusedBody(rv: Int) -> String {
    let recip: String
    switch rv {
    case 3: recip = "normalizer = precise::divide(1.0f, normalizer);"
    case 4: recip = "normalizer = fast::divide(1.0f, normalizer);"
    default: recip = "normalizer = 1 / normalizer;"
    }
    return """
        const int axis_size = params[0];
        const int cofs = params[1];
        const int qL = params[2];

        uint gid = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint lsize = threads_per_threadgroup.x;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;

        const device bfloat16_t* in = scores;
        device bfloat16_t* out = y;
        in += gid * size_t(axis_size);
        const int rowlim = cofs + int(gid) % qL;

        constexpr int SIMD_SIZE = 32;
        constexpr int N_READS = 4;

        threadgroup float local_max[SIMD_SIZE];
        threadgroup float local_normalizer[SIMD_SIZE];

        // Get the max and the normalizer in one go
        float prevmax;
        float maxval = Limits<float>::finite_min;
        float normalizer = 0;
        for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
             r++) {
          int offset = r * lsize * N_READS + lid * N_READS;
          float vals[N_READS];
          if (offset + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              vals[i] = (offset + i <= rowlim)
                  ? float(in[offset + i]) : BF16_MINF;
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              vals[i] =
                  (offset + i < axis_size)
                  ? ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
                  : Limits<float>::min;
            }
          }
          prevmax = maxval;
          for (int i = 0; i < N_READS; i++) {
            maxval = (maxval < vals[i]) ? vals[i] : maxval;
          }
          normalizer *= softmax_exp(prevmax - maxval);
          for (int i = 0; i < N_READS; i++) {
            normalizer += softmax_exp(vals[i] - maxval);
          }
        }
        // Now we got partial normalizer of N_READS * ceildiv(axis_size, N_READS *
        // lsize) parts. We need to combine them.
        //    1. We start by finding the max across simd groups
        //    2. We then change the partial normalizers to account for a possible
        //       change in max
        //    3. We sum all normalizers
        prevmax = maxval;
        maxval = simd_max(maxval);
        normalizer *= softmax_exp(prevmax - maxval);
        normalizer = simd_sum(normalizer);

        // Now the normalizer and max value is correct for each simdgroup. We write
        // them shared memory and combine them.
        prevmax = maxval;
        if (simd_lane_id == 0) {
          local_max[simd_group_id] = maxval;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maxval = simd_max(local_max[simd_lane_id]);
        normalizer *= softmax_exp(prevmax - maxval);
        if (simd_lane_id == 0) {
          local_normalizer[simd_group_id] = normalizer;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer = simd_sum(local_normalizer[simd_lane_id]);
        \(recip)

        // Finally given the normalizer and max value we can directly write the
        // softmax output
        out += gid * size_t(axis_size);
        for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
             r++) {
          int offset = r * lsize * N_READS + lid * N_READS;
          if (offset + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              out[offset + i] = bfloat16_t(softmax_exp(
                  ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
                  - maxval) * normalizer);
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              if (offset + i < axis_size) {
                out[offset + i] = bfloat16_t(softmax_exp(
                    ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
                    - maxval) * normalizer);
              }
            }
          }
        }
        """
}

func makeFusedKernel(rv: Int, ev: Int) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
        name: "m5_fused_rv\(rv)_ev\(ev)",
        inputNames: ["scores", "params"],
        outputNames: ["y"],
        source: fusedBody(rv: rv),
        header: fusedHeader(ev: ev))
}

// MARK: - Data

func genScores(seed: UInt64, S: Int, count: Int) -> [MLXArray] {
    MLXRandom.seed(seed)
    var sets: [MLXArray] = []
    for _ in 0 ..< count {
        let s = MLXRandom.normal([1, N_HEADS, QL, S], dtype: .bfloat16, scale: 2)
        eval(s)
        sets.append(s)
    }
    return sets
}

// MARK: - Bitwise comparison

func bitwiseGate(_ name: String, _ ref: MLXArray, _ out: MLXArray, quiet: Bool = false) -> Bool {
    eval(ref, out)
    let eq = (ref .== out).all().item(Bool.self)
    if eq {
        if !quiet { print("\(name): IDENT (\(ref.size) outputs)") }
        return true
    }
    let nDiff = (ref .!= out).asType(.int32).sum().item(Int32.self)
    let refF = ref.asType(.float32)
    let outF = out.asType(.float32)
    let mad = abs(refF - outF).max().item(Float.self)
    print("\(name): DIFF count=\(nDiff)/\(ref.size) maxAbs=\(mad)")
    let mask = (ref .!= out).asType(.int32).asArray(Int32.self)
    let rf = refF.asArray(Float.self)
    let of = outF.asArray(Float.self)
    var shown = 0
    for i in 0 ..< mask.count where mask[i] != 0 && shown < 8 {
        let v = abs(rf[i])
        let ulp = v > 0 ? powf(2, floor(log2f(v)) - 7) : powf(2, -133)
        let row = i / (mask.count > 0 ? 1 : 1)
        print(String(
            format: "  [%d] ref=%.9g out=%.9g  delta=%.3g (~%.1f ulp)",
            i, rf[i], of[i], of[i] - rf[i], (of[i] - rf[i]) / ulp))
        shown += 1
        _ = row
    }
    return false
}

// MARK: - Timing

@discardableResult
func timeBlock(reps: Int, body: (Int) -> MLXArray) -> (wall: Double, build: Double, tail: Double) {
    var outs: [MLXArray] = []
    outs.reserveCapacity(reps)
    let t0 = CFAbsoluteTimeGetCurrent()
    for j in 0 ..< reps { outs.append(body(j)) }
    let t1 = CFAbsoluteTimeGetCurrent()
    eval(outs)
    let t2 = CFAbsoluteTimeGetCurrent()
    _ = outs[outs.count - 1].sum().item(Float.self)
    let t3 = CFAbsoluteTimeGetCurrent()
    return (t3 - t0, t1 - t0, t3 - t2)
}

func warm(reps: Int, body: (Int) -> MLXArray) {
    var outs: [MLXArray] = []
    for j in 0 ..< reps { outs.append(body(j)) }
    eval(outs)
    _ = outs.last!.sum().item(Float.self)
}

struct ArmStat {
    var wall: Double = 0
    var blocks: Int = 0
    var perCall: Double { wall / Double(blocks) }
}

// MARK: - Step 1: ceiling

func runCeiling() {
    let S = 32768
    let nSets = 6
    let reps = 6
    print("== M5 ceiling at [1,\(N_HEADS),\(QL),\(S)] bf16 ==")
    let scores = genScores(seed: 42, S: S, count: nSets)
    let offset = S - QL  // standard last-chunk case
    let mask = causalMask(qL: QL, kL: S, offset: offset)
    eval(mask)

    let chain = { (i: Int) -> MLXArray in chainY(scores[i % nSets], mask: mask) }
    let sm = { (i: Int) -> MLXArray in softmaxY(scores[i % nSets]) }
    let mw = { (i: Int) -> MLXArray in maskWhereY(scores[i % nSets], mask: mask) }

    warm(reps: 2, body: chain)
    warm(reps: 2, body: sm)
    warm(reps: 2, body: mw)

    var acc: [String: ArmStat] = ["chain": ArmStat(), "softmax": ArmStat(), "mask+where": ArmStat()]
    let order = ["chain", "softmax", "softmax", "chain", "chain", "softmax", "softmax", "chain"]
    for arm in order {
        let body = arm == "chain" ? chain : sm
        let (wall, build, tail) = timeBlock(reps: reps, body: body)
        acc[arm]!.wall += wall / Double(reps)
        acc[arm]!.blocks += 1
        print(String(format: "  block %@ wall/call %.3f ms (build %.3f, tail %.3f)",
                     arm, wall / Double(reps) * 1e3, build / Double(reps) * 1e3,
                     tail / Double(reps) * 1e3))
    }
    for _ in 0 ..< 2 {
        let (wall, _, _) = timeBlock(reps: reps, body: mw)
        acc["mask+where"]!.wall += wall / Double(reps)
        acc["mask+where"]!.blocks += 1
    }

    let chainMs = acc["chain"]!.perCall * 1e3
    let smMs = acc["softmax"]!.perCall * 1e3
    let mwMs = acc["mask+where"]!.perCall * 1e3
    let gb = Double(QL * N_HEADS * S * 2) / 1e9
    print(String(format: "chain %.3f ms | softmax-only %.3f ms | mask+where %.3f ms",
                 chainMs, smMs, mwMs))
    print(String(format: "scores bytes/call %.2f GB; softmax effective BW %.0f GB/s",
                 gb, gb * 2 / (smMs / 1e3)))
    // prefill model: 28 s total, 320 layer-chunks at this width
    let prefillUs = 28e6
    let layerChunks = 320.0
    print(String(format: "prefill share (x%.0f of 28s): chain %.2f%%  non-softmax part %.2f%%  fused-ideal (==softmax) %.2f%%",
                 layerChunks,
                 chainMs * 1e3 * layerChunks / prefillUs * 100,
                 (chainMs - smMs) * 1e3 * layerChunks / prefillUs * 100,
                 smMs * 1e3 * layerChunks / prefillUs * 100))
    let nonSoftmaxPct = (chainMs - smMs) * 1e3 * layerChunks / prefillUs * 100
    print(nonSoftmaxPct < 1.0
        ? "CEILING: BELOW 1% BAR — fused not worth it, stop."
        : "CEILING: above 1% bar — proceed to step 2 (fused kernel).")
}

// MARK: - Step 2: fused gates + timing

func fusedY(_ scores: MLXArray, S: Int, offset: Int, kernel: MLXFast.MLXFastKernel) -> MLXArray {
    let nRows = N_HEADS * QL
    let params = MLXArray([Int32(S), Int32(offset), Int32(QL)])
    return kernel(
        [scores, params],
        grid: (nRows * 1024, 1, 1), threadGroup: (1024, 1, 1),
        outputShapes: [[1, N_HEADS, QL, S]], outputDTypes: [.bfloat16])[0]
}

func runFused() {
    // introspect the production kernel's threadgroup size for the record
    if let device = MTLCreateSystemDefaultDevice(),
       let lib = try? device.makeLibrary(filepath: "/tmp/gather-sweep/.build/release/mlx.metallib") {
        let names = lib.functionNames.filter { $0.contains("softmax") }
        print("metallib softmax functions: \(names)")
        for n in names where n.contains("looped") && n.contains("bfloat16") {
            if let f = lib.makeFunction(name: n),
               let pso = try? device.makeComputePipelineState(function: f) {
                print("\(n): maxTotalThreadsPerThreadgroup = \(pso.maxTotalThreadsPerThreadgroup)")
            }
        }
    }

    print("== fused bitwise gates ==")
    let seeds: [UInt64] = [42, 1234, 98765]
    var allOk = true
    for S in [8192, 32768] {
        let offsets = [S - QL, 0, (S - QL) / 2]
        for offset in offsets {
            let mask = causalMask(qL: QL, kL: S, offset: offset)
            eval(mask)
            for seed in (S == 32768 || offset == S - QL ? seeds : [seeds[0]]) {
                let scores = genScores(seed: seed ^ UInt64(S), S: S, count: 1)[0]
                let ref = chainY(scores, mask: mask)
                let k = makeFusedKernel(rv: 1, ev: 1)
                let out = fusedY(scores, S: S, offset: offset, kernel: k)
                let ok = bitwiseGate("S=\(S) off=\(offset) seed=\(seed)", ref, out)
                allOk = allOk && ok
            }
        }
    }
    print(allOk ? "GATES: all IDENT" : "GATES: mismatches — see above")

    // ABBA timing at both S
    for S in [8192, 32768] {
        print("== timing S=\(S) ==")
        let nSets = S == 32768 ? 6 : 16
        let reps = S == 32768 ? 6 : 12
        let scores = genScores(seed: 777, S: S, count: nSets)
        let offset = S - QL
        let mask = causalMask(qL: QL, kL: S, offset: offset)
        eval(mask)
        let k = makeFusedKernel(rv: 1, ev: 1)
        let chain = { (i: Int) -> MLXArray in chainY(scores[i % nSets], mask: mask) }
        let fus = { (i: Int) -> MLXArray in fusedY(scores[i % nSets], S: S, offset: offset, kernel: k) }
        let sm = { (i: Int) -> MLXArray in softmaxY(scores[i % nSets]) }
        _ = bitwiseGate("timing-data fused vs chain", chainY(scores[0], mask: mask),
                        fusedY(scores[0], S: S, offset: offset, kernel: k))
        warm(reps: 2, body: chain)
        warm(reps: 2, body: fus)
        warm(reps: 2, body: sm)
        var acc: [String: ArmStat] = ["chain": ArmStat(), "fused": ArmStat(), "softmax": ArmStat()]
        let order = ["chain", "fused", "fused", "chain", "chain", "fused", "fused", "chain"]
        for arm in order {
            let body = arm == "chain" ? chain : fus
            let (wall, _, _) = timeBlock(reps: reps, body: body)
            acc[arm]!.wall += wall / Double(reps)
            acc[arm]!.blocks += 1
        }
        for _ in 0 ..< 2 {
            let (wall, _, _) = timeBlock(reps: reps, body: sm)
            acc["softmax"]!.wall += wall / Double(reps)
            acc["softmax"]!.blocks += 1
        }
        let c = acc["chain"]!.perCall * 1e3
        let f = acc["fused"]!.perCall * 1e3
        let s = acc["softmax"]!.perCall * 1e3
        print(String(format: "S=%d: chain %.3f ms | fused %.3f ms | softmax-only %.3f ms | fused saves %.1f%% of chain",
                     S, c, f, s, (c - f) / c * 100))
    }
}

// MARK: - Driver

setvbuf(stdout, nil, _IONBF, 0)

let args = CommandLine.arguments
if args.contains("dump") {
    let text = """
        // M5 fused causal-mask + softmax kernel (winning variant rv=1, ev=1 — verbatim
        // softmax_looped<bfloat16_t, float, 4> with the causal select injected at both
        // in[] load sites). Bitwise-identical to the 3-op fallback chain
        // (greater_equal -> where(bf16 finfo.min) -> softmax precise) on all tested
        // seeds / S / offsets. Kernel name: m5_fused_rv1_ev1.
        // Generated signature (MLXFast wraps this):
        //   inputs:  scores (device bfloat16_t*, [B,H,qL,S] contiguous),
        //            params (constant int*, [axis_size=S, cofs=kL-qL, qL])
        //   output:  y (device bfloat16_t*, same shape)
        //   grid (threads): (B*H*qL * 1024, 1, 1), threadgroup: (1024, 1, 1)
        //   NOTE: threadgroup size MUST equal the production softmax dispatch
        //   (maxTotalThreadsPerThreadgroup, 1024 here) — the f32 accumulation order
        //   depends on lsize; a different lsize changes the bits.
        // ---- header (prepended by MLXFast after metal::utils()) ----
        \(fusedHeader(ev: 1))
        // ---- body ----
        \(fusedBody(rv: 1))
        """
    try! text.write(toFile: "/tmp/gather-sweep/m5-fused-kernel.metal",
                    atomically: true, encoding: .utf8)
    print("wrote /tmp/gather-sweep/m5-fused-kernel.metal")
    exit(0)
}
if args.contains("ceiling") {
    runCeiling()
} else if args.contains("fused") {
    runFused()
} else {
    runCeiling()
    runFused()
}
