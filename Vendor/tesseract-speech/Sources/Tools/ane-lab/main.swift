// ane-lab: which MIL ops Core ML places on this machine's Neural Engine.
// Builds tiny ML programs with MLProgramBuilder, compiles them, and prints
// the compute plan's device for every op. Not linked by the app.

import CoreML
import Foundation
@preconcurrency import MLX
import Qwen3TTS

@main
struct ANELab {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        let only = args.first { !$0.hasPrefix("--") }
        let target: MLProgramPackage.Target = args.contains("--coreml7") ? .coreML7 : .coreML8
        if args.contains("--prefill-placement") {
            for tokens in [64, 128, 256, 1024] {
                for (name, build) in prefillProbes(tokens: tokens) {
                    let report = try await placement(name: name, build, target: target)
                    print("\(name) T=\(tokens): \(report)")
                }
            }
            return
        }
        if args.contains("--prefill") {
            // Prefill-shaped throughput: T tokens through 5120 -> 17408 -> 5120.
            for tokens in [32, 64, 96, 128, 160, 192] {
                for (name, build) in prefillProbes(tokens: tokens) where only == nil || name.contains(only!) {
                    let ms = try await latency(
                        name: name, build, inputShape: [1, 5120, 1, tokens].map { NSNumber(value: $0) })
                    let flops = 2.0 * 2 * 5120 * 17408 * Double(tokens) * Double(prefillLayers)
                    print("\(name) T=\(tokens): \(String(format: "%.2f", ms)) ms median, "
                        + String(format: "%.2f TFLOP/s", flops / ms / 1e9))
                }
            }
            return
        }
        if args.contains("--time") {
            for (name, build) in timed where only == nil || name.contains(only!) {
                let ms = try await latency(name: name, build)
                print("\(name): \(String(format: "%.2f", ms)) ms median")
            }
            return
        }
        for (name, build) in probes where only == nil || name.contains(only!) {
            do {
                let report = try await placement(name: name, build, target: target)
                print("\(name): \(report)")
            } catch {
                print("\(name): FAILED \(error)")
            }
        }
    }

    typealias Build = @Sendable (MILFunctionBuilder) -> Void

    static let probes: [(String, Build)] = [
        ("conv-int8-1x1", { f in
            let x = f.input("x", .fp16([1, 1024, 1, 1]))
            let w = blockwise(f, out: 3072, in: 1024)
            let y = f.op("conv", convInputs(f, x, w), .fp16([1, 3072, 1, 1]))
            f.output(f.op("add", [("x", [y]), ("y", [f.half(0)])], y.type, name: "y"))
        }),
        ("linear-int8", { f in
            let x = f.input("x", .fp16([1, 1024]))
            let w = blockwise2D(f, out: 3072, in: 1024)
            let y = f.op("linear", [("x", [x]), ("weight", [w])], .fp16([1, 3072]))
            f.output(f.op("add", [("x", [y]), ("y", [f.half(0)])], y.type, name: "y"))
        }),
        ("conv-fp16-1x1", { f in
            let x = f.input("x", .fp16([1, 1024, 1, 1]))
            let w = f.weight(MLXArray.zeros([3072, 1024, 1, 1], dtype: .float16), shape: [3072, 1024, 1, 1])
            let y = f.op("conv", convInputs(f, x, w), .fp16([1, 3072, 1, 1]))
            f.output(f.op("add", [("x", [y]), ("y", [f.half(0)])], y.type, name: "y"))
        }),
        ("attention-matmul-softmax", { f in
            let q = f.input("q", .fp16([1, 8, 2, 128]))
            let k = f.input("k", .fp16([1, 8, 128, 512]))
            let v = f.input("v", .fp16([1, 8, 128, 512]))
            let mask = f.input("mask", .fp16([1, 1, 1, 512]))
            let s = f.op("matmul", [("x", [q]), ("y", [k]), ("transpose_x", [f.bool(false)]), ("transpose_y", [f.bool(false)])], .fp16([1, 8, 2, 512]))
            let scaled = f.op("mul", [("x", [s]), ("y", [f.half(0.088)])], s.type)
            let masked = f.op("add", [("x", [scaled]), ("y", [mask])], s.type)
            let p = f.op("softmax", [("x", [masked]), ("axis", [f.int(-1)])], s.type)
            let o = f.op("matmul", [("x", [p]), ("y", [v]), ("transpose_x", [f.bool(false)]), ("transpose_y", [f.bool(true)])], .fp16([1, 8, 2, 128]), name: "o")
            f.output(o)
        }),
        ("state-read-slice", { f in
            let cache = f.input("cache", .state([1, 28672, 1, 512]))
            let x = f.input("x", .fp16([1, 1024, 1, 1]))
            let all = f.op("read_state", [("input", [cache])], .fp16([1, 28672, 1, 512]))
            let layer = f.op("slice_by_index", [("x", [all]), ("begin", [f.ints([0, 6144, 0, 0])]), ("end", [f.ints([1, 7168, 1, 512])])], .fp16([1, 1024, 1, 512]))
            let y = f.op("mul", [("x", [layer]), ("y", [x])], layer.type)
            f.output(f.op("reduce_sum", [("x", [y]), ("axes", [f.ints([3])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1024, 1, 1]), name: "y"))
        }),
        ("state-write", { f in
            let cache = f.input("cache", .state([1, 1024, 1, 512]))
            let x = f.input("x", .fp16([1, 1024, 1, 512]))
            let old = f.op("read_state", [("input", [cache])], .fp16([1, 1024, 1, 512]))
            let new = f.op("add", [("x", [old]), ("y", [x])], old.type)
            f.ops("write_state", [("input", [cache]), ("data", [new])], [])
            f.output(f.op("reduce_sum", [("x", [new]), ("axes", [f.ints([3])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1024, 1, 1]), name: "y"))
        }),
        ("rmsnorm-reduce-mean-rsqrt", { f in
            let x = f.input("x", .fp16([1, 1024, 1, 1]))
            let sq = f.op("mul", [("x", [x]), ("y", [x])], x.type)
            let m = f.op("reduce_mean", [("x", [sq]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
            let e = f.op("add", [("x", [m]), ("y", [f.half(1e-6)])], m.type)
            let r = f.op("rsqrt", [("x", [e])], m.type)
            f.output(f.op("mul", [("x", [x]), ("y", [r])], x.type, name: "y"))
        }),
        ("layer-norm", { f in
            let x = f.input("x", .fp16([1, 2048, 1, 1]))
            f.output(f.op("layer_norm", [("x", [x]), ("axes", [f.ints([1])]), ("epsilon", [f.half(1e-6)])], x.type, name: "y"))
        }),
        ("gumbel-onehot-matmul", { f in
            let logits = f.input("logits", .fp16([1, 2048]))
            let noise = f.input("noise", .fp16([1, 2048]))
            let t = f.op("mul", [("x", [logits]), ("y", [f.half(2)])], logits.type)
            let z = f.op("add", [("x", [t]), ("y", [noise])], logits.type)
            let mx = f.op("reduce_max", [("x", [z]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1]))
            let eq = f.op("equal", [("x", [z]), ("y", [mx])], .bool([1, 2048]))
            let onehot = f.op("cast", [("x", [eq]), ("dtype", [f.string("fp16")])], .fp16([1, 2048]))
            let table = blockwise2D(f, out: 2048, in: 1024)
            let e = f.op("matmul", [("x", [onehot]), ("y", [table]), ("transpose_x", [f.bool(false)]), ("transpose_y", [f.bool(false)])], .fp16([1, 1024]), name: "e")
            f.output(e)
        }),
        ("greater-equal-select", { f in
            let z = f.input("z", .fp16([1, 2048]))
            let th = f.input("th", .fp16([1, 1]))
            let keep = f.op("greater_equal", [("x", [z]), ("y", [th])], .bool([1, 2048]))
            f.output(f.op("select", [("cond", [keep]), ("a", [z]), ("b", [f.half(-60000)])], z.type, name: "y"))
        }),
        ("topk", { f in
            let z = f.input("z", .fp16([1, 2048]))
            let out = f.ops("topk", [("x", [z]), ("k", [f.int(50)]), ("axis", [f.int(-1)])], [.fp16([1, 50]), .int32([1, 50])], names: ["v", "i"])
            f.output(out[0])
        }),
        ("reduce-argmax", { f in
            let z = f.input("z", .fp16([1, 2048]))
            f.output(f.op("reduce_argmax", [("x", [z]), ("axis", [f.int(1)]), ("keep_dims", [f.bool(true)])], .int32([1, 1]), name: "y"))
        }),
        ("gather-const-table", { f in
            let idx = f.input("idx", .int32([1]))
            let table = f.weight(MLXArray.zeros([2048, 1024], dtype: .float16), shape: [2048, 1024])
            f.output(f.op("gather", [("x", [table]), ("indices", [idx]), ("axis", [f.int(0)])], .fp16([1, 1024]), name: "y"))
        }),
        ("conv-fp16-w8", { f in
            let x = f.input("x", .fp16([1, 1024, 1, 8]))
            let w = f.weight(MLXArray.zeros([1024, 1024, 1, 1], dtype: .float16), shape: [1024, 1024, 1, 1])
            let y = f.op("conv", convInputs(f, x, w), .fp16([1, 1024, 1, 8]), name: "y")
            f.output(y)
        }),
        ("conv-chain-w1", { f in
            var x = f.input("x", .fp16([1, 1024, 1, 1]))
            for _ in 0 ..< 12 {
                let w = f.weight(MLXArray.zeros([1024, 1024, 1, 1], dtype: .float16), shape: [1024, 1024, 1, 1])
                x = f.op("conv", convInputs(f, x, w), .fp16([1, 1024, 1, 1]))
            }
            f.output(f.op("add", [("x", [x]), ("y", [f.half(0)])], x.type, name: "y"))
        }),
        ("conv-chain-int8-w1", { f in
            var x = f.input("x", .fp16([1, 1024, 1, 1]))
            for _ in 0 ..< 12 {
                let w = blockwise(f, out: 1024, in: 1024)
                x = f.op("conv", convInputs(f, x, w), .fp16([1, 1024, 1, 1]))
            }
            f.output(f.op("add", [("x", [x]), ("y", [f.half(0)])], x.type, name: "y"))
        }),
        ("chain-lut-tensor", { f in
            chain(f) { f in
                f.palettizedWeight(indices: Data(count: 1024 * 1024), lut: half(0.01, count: 256),
                    shape: [1024, 1024, 1, 1], lutShape: [1, 1, 1, 1, 256, 1])
            }
        }),
        ("chain-lut-channel", { f in
            chain(f) { f in
                f.palettizedWeight(indices: Data(count: 1024 * 1024), lut: half(0.01, count: 1024 * 256),
                    shape: [1024, 1024, 1, 1], lutShape: [1024, 1, 1, 1, 256, 1])
            }
        }),
        ("chain-lut-group16", { f in
            chain(f) { f in
                f.palettizedWeight(indices: Data(count: 1024 * 1024), lut: half(0.01, count: 64 * 256),
                    shape: [1024, 1024, 1, 1], lutShape: [64, 1, 1, 1, 256, 1])
            }
        }),
        ("chain-int8-channel", { f in
            chain(f) { f in
                f.symmetricWeight(data: Data(count: 1024 * 1024), scale: half(0.01, count: 1024),
                    shape: [1024, 1024, 1, 1], blockShape: [1024, 1, 1, 1])
            }
        }),
        ("chain-int8-block32", { f in
            chain(f) { f in
                f.symmetricWeight(data: Data(count: 1024 * 1024), scale: half(0.01, count: 1024 * 32),
                    shape: [1024, 1024, 1, 1], blockShape: [1024, 32, 1, 1])
            }
        }),
        ("chain-uint8-channel-offset", { f in
            chain(f) { f in
                f.blockwiseWeight(data: Data(count: 1024 * 1024), scale: half(0.01, count: 1024),
                    offset: half(128, count: 1024), shape: [1024, 1024, 1, 1], blockShape: [1024, 1, 1, 1])
            }
        }),
        ("sampler-in-heavy", { f in
            // A decode-sized graph, then sampling and the next code's
            // embedding, as the fused code predictor runs per pass.
            var x = f.input("x", .fp16([1, 1024, 1, 1]))
            let noise = f.input("noise", .fp16([1, 2048, 1, 1]))
            let temperature = f.input("inv_temperature", .fp16([1, 1, 1, 1]))
            for _ in 0 ..< 4 {
                let w = f.symmetricWeight(data: random(1024 * 1024), scale: half(0.001, count: 1024), shape: [1024, 1024, 1, 1], blockShape: [1024, 1, 1, 1])
                x = f.op("conv", convInputs(f, x, w), .fp16([1, 1024, 1, 1]))
            }
            let head = f.symmetricWeight(data: random(2048 * 1024), scale: half(0.001, count: 2048), shape: [2048, 1024, 1, 1], blockShape: [2048, 1, 1, 1])
            let logits = f.op("conv", convInputs(f, x, head), .fp16([1, 2048, 1, 1]))
            let scaled = f.op("mul", [("x", [logits]), ("y", [temperature])], logits.type)
            let flat = f.op("reshape", [("x", [scaled]), ("shape", [f.ints([1, 2048])])], .fp16([1, 2048]))
            let top = f.ops("topk", [("x", [flat]), ("k", [f.int(50)]), ("axis", [f.int(-1)]), ("ascending", [f.bool(false)])], [.fp16([1, 50]), .int32([1, 50])])
            let kth = f.op("slice_by_index", [("x", [top[0]]), ("begin", [f.ints([0, 49])]), ("end", [f.ints([1, 50])])], .fp16([1, 1]))
            let kth4 = f.op("reshape", [("x", [kth]), ("shape", [f.ints([1, 1, 1, 1])])], .fp16([1, 1, 1, 1]))
            let keep = f.op("greater_equal", [("x", [scaled]), ("y", [kth4])], .bool([1, 2048, 1, 1]))
            let z = f.op("add", [("x", [scaled]), ("y", [noise])], logits.type)
            let masked = f.op("select", [("cond", [keep]), ("a", [z]), ("b", [f.half(-30000)])], logits.type)
            let mx = f.op("reduce_max", [("x", [masked]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
            let hit = f.op("equal", [("x", [masked]), ("y", [mx])], .bool([1, 2048, 1, 1]))
            let onehot = f.op("cast", [("x", [hit]), ("dtype", [f.string("fp16")])], .fp16([1, 2048, 1, 1]))
            let table = f.symmetricWeight(data: random(1024 * 2048), scale: half(0.001, count: 1024), shape: [1024, 2048, 1, 1], blockShape: [1024, 1, 1, 1])
            let embed = f.op("conv", convInputs(f, onehot, table), .fp16([1, 1024, 1, 1]), name: "embed")
            f.output(embed)
            f.output(f.op("add", [("x", [onehot]), ("y", [f.half(0)])], onehot.type, name: "onehot"))
        }),
        ("sampler-bisect-in-heavy", { f in
            var x = f.input("x", .fp16([1, 1024, 1, 1]))
            let noise = f.input("noise", .fp16([1, 2048, 1, 1]))
            let temperature = f.input("inv_temperature", .fp16([1, 1, 1, 1]))
            for _ in 0 ..< 4 {
                let w = f.symmetricWeight(data: random(1024 * 1024), scale: half(0.001, count: 1024), shape: [1024, 1024, 1, 1], blockShape: [1024, 1, 1, 1])
                x = f.op("conv", convInputs(f, x, w), .fp16([1, 1024, 1, 1]))
            }
            let head = f.symmetricWeight(data: random(2048 * 1024), scale: half(0.001, count: 2048), shape: [2048, 1024, 1, 1], blockShape: [2048, 1, 1, 1])
            let logits = f.op("conv", convInputs(f, x, head), .fp16([1, 2048, 1, 1]))
            let scaled = f.op("mul", [("x", [logits]), ("y", [temperature])], logits.type)
            // Bisection for the 50th largest: lo..hi brackets it.
            var lo = f.op("reduce_min", [("x", [scaled]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
            var hi = f.op("reduce_max", [("x", [scaled]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
            for _ in 0 ..< 12 {
                let sum = f.op("add", [("x", [lo]), ("y", [hi])], lo.type)
                let mid = f.op("mul", [("x", [sum]), ("y", [f.half(0.5)])], lo.type)
                let ge = f.op("greater_equal", [("x", [scaled]), ("y", [mid])], .bool([1, 2048, 1, 1]))
                let gef = f.op("cast", [("x", [ge]), ("dtype", [f.string("fp16")])], .fp16([1, 2048, 1, 1]))
                let count = f.op("reduce_sum", [("x", [gef]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
                let enough = f.op("greater_equal", [("x", [count]), ("y", [f.half(50)])], .bool([1, 1, 1, 1]))
                lo = f.op("select", [("cond", [enough]), ("a", [mid]), ("b", [lo])], lo.type)
                hi = f.op("select", [("cond", [enough]), ("a", [hi]), ("b", [mid])], hi.type)
            }
            let keep = f.op("greater_equal", [("x", [scaled]), ("y", [lo])], .bool([1, 2048, 1, 1]))
            let z = f.op("add", [("x", [scaled]), ("y", [noise])], logits.type)
            let masked = f.op("select", [("cond", [keep]), ("a", [z]), ("b", [f.half(-30000)])], logits.type)
            let mx = f.op("reduce_max", [("x", [masked]), ("axes", [f.ints([1])]), ("keep_dims", [f.bool(true)])], .fp16([1, 1, 1, 1]))
            let hit = f.op("equal", [("x", [masked]), ("y", [mx])], .bool([1, 2048, 1, 1]))
            let onehot = f.op("cast", [("x", [hit]), ("dtype", [f.string("fp16")])], .fp16([1, 2048, 1, 1]))
            let table = f.symmetricWeight(data: random(1024 * 2048), scale: half(0.001, count: 1024), shape: [1024, 2048, 1, 1], blockShape: [1024, 1, 1, 1])
            let embed = f.op("conv", convInputs(f, onehot, table), .fp16([1, 1024, 1, 1]), name: "embed")
            f.output(embed)
            f.output(f.op("add", [("x", [onehot]), ("y", [f.half(0)])], onehot.type, name: "onehot"))
        }),
        ("silu-sigmoid", { f in
            let x = f.input("x", .fp16([1, 3072, 1, 1]))
            f.output(f.op("silu", [("x", [x])], x.type, name: "y"))
        }),
        ("rope-slices-concat", { f in
            let x = f.input("x", .fp16([1, 16, 128, 1]))
            let cos = f.input("cos", .fp16([1, 1, 128, 1]))
            let sin = f.input("sin", .fp16([1, 1, 128, 1]))
            let a = f.op("slice_by_index", [("x", [x]), ("begin", [f.ints([0, 0, 0, 0])]), ("end", [f.ints([1, 16, 64, 1])])], .fp16([1, 16, 64, 1]))
            let b = f.op("slice_by_index", [("x", [x]), ("begin", [f.ints([0, 0, 64, 0])]), ("end", [f.ints([1, 16, 128, 1])])], .fp16([1, 16, 64, 1]))
            let nb = f.op("mul", [("x", [b]), ("y", [f.half(-1)])], b.type)
            let rot = f.op("concat", [("values", [nb, a]), ("axis", [f.int(2)]), ("interleave", [f.bool(false)])], x.type)
            let xc = f.op("mul", [("x", [x]), ("y", [cos])], x.type)
            let rs = f.op("mul", [("x", [rot]), ("y", [sin])], x.type)
            f.output(f.op("add", [("x", [xc]), ("y", [rs])], x.type, name: "y"))
        }),
    ]

    /// Twelve width-1 convs with weights from `weight`, as a decode step runs.
    static func chain(_ f: MILFunctionBuilder, weight: (MILFunctionBuilder) -> MILVar) {
        var x = f.input("x", .fp16([1, 1024, 1, 1]))
        for _ in 0 ..< 12 {
            x = f.op("conv", convInputs(f, x, weight(f)), .fp16([1, 1024, 1, 1]))
        }
        f.output(f.op("add", [("x", [x]), ("y", [f.half(0)])], x.type, name: "y"))
    }

    /// Weight-heavy stacks: 16 layers of 1024→3072→1024 (100M weights), one
    /// position per call, as the talker's decode step streams its weights.
    static let timed: [(String, Build)] = [
        ("time-fp16", { f in heavy(f) { f, o, i in f.weight(MLXRandom.normal([o, i, 1, 1]).asType(.float16) * 0.01, shape: [o, i, 1, 1]) } }),
        ("time-lut-tensor", { f in heavy(f) { f, o, i in
            f.palettizedWeight(indices: random(o * i), lut: half(0.001, count: 256), shape: [o, i, 1, 1], lutShape: [1, 1, 1, 1, 256, 1]) } }),
        ("time-lut-channel", { f in heavy(f) { f, o, i in
            f.palettizedWeight(indices: random(o * i), lut: half(0.001, count: o * 256), shape: [o, i, 1, 1], lutShape: [o, 1, 1, 1, 256, 1]) } }),
        ("time-int8-channel", { f in heavy(f) { f, o, i in
            f.symmetricWeight(data: random(o * i), scale: half(0.001, count: o), shape: [o, i, 1, 1], blockShape: [o, 1, 1, 1]) } }),
    ]

    static let prefillLayers = 2

    static func prefillStack(
        _ f: MILFunctionBuilder, tokens: Int, weight: (MILFunctionBuilder, Int, Int) -> MILVar
    ) {
        var x = f.input("x", .fp16([1, 5120, 1, tokens]))
        for _ in 0 ..< prefillLayers {
            let up = f.op("conv", convInputs(f, x, weight(f, 17408, 5120)), .fp16([1, 17408, 1, tokens]))
            let act = f.op("silu", [("x", [up])], up.type)
            let down = f.op("conv", convInputs(f, act, weight(f, 5120, 17408)), .fp16([1, 5120, 1, tokens]))
            x = f.op("add", [("x", [x]), ("y", [down])], x.type)
        }
        f.output(f.op("add", [("x", [x]), ("y", [f.half(0)])], x.type, name: "y"))
    }

    static func prefillProbes(tokens: Int) -> [(String, Build)] {
        [
            ("prefill-fp16", { f in prefillStack(f, tokens: tokens) { f, o, i in
                f.weight(MLXRandom.normal([o, i, 1, 1]).asType(.float16) * 0.01, shape: [o, i, 1, 1]) } }),
            ("prefill-int8-channel", { f in prefillStack(f, tokens: tokens) { f, o, i in
                f.symmetricWeight(data: random(o * i), scale: half(0.001, count: o), shape: [o, i, 1, 1], blockShape: [o, 1, 1, 1]) } }),
        ]
    }

    static func heavy(_ f: MILFunctionBuilder, weight: (MILFunctionBuilder, Int, Int) -> MILVar) {
        var x = f.input("x", .fp16([1, 1024, 1, 1]))
        for _ in 0 ..< 16 {
            let up = f.op("conv", convInputs(f, x, weight(f, 3072, 1024)), .fp16([1, 3072, 1, 1]))
            let act = f.op("silu", [("x", [up])], up.type)
            let down = f.op("conv", convInputs(f, act, weight(f, 1024, 3072)), .fp16([1, 1024, 1, 1]))
            x = f.op("add", [("x", [x]), ("y", [down])], x.type)
        }
        f.output(f.op("add", [("x", [x]), ("y", [f.half(0)])], x.type, name: "y"))
    }

    /// Median latency of one prediction on the Neural Engine, after warm-up.
    static func latency(name: String, _ build: Build, inputShape: [NSNumber] = [1, 1024, 1, 1]) async throws -> Double {
        let fm = FileManager.default
        let package = fm.temporaryDirectory.appendingPathComponent("ane-lab-\(name)-\(UUID().uuidString).mlpackage")
        defer { try? fm.removeItem(at: package) }
        let blobs = try MILBlobWriter(url: MLProgramPackage.weightsURL(in: package))
        let f = MILFunctionBuilder(blobs: blobs)
        build(f)
        try blobs.finish()
        try MLProgramPackage.write(
            specification: MLProgramPackage.specification(f, metadata: [:], target: .coreML8), to: package)
        let compiled = try await MLModel.compileModel(at: package)
        defer { try? fm.removeItem(at: compiled) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        let x = try MLMultiArray(shape: inputShape, dataType: .float16)
        let input = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
        for _ in 0 ..< 10 { _ = try await model.prediction(from: input) }
        var times: [Double] = []
        for _ in 0 ..< 50 {
            let start = ContinuousClock.now
            _ = try await model.prediction(from: input)
            let d = start.duration(to: .now)
            times.append(Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1000)
        }
        return times.sorted()[times.count / 2]
    }

    static func convInputs(_ f: MILFunctionBuilder, _ x: MILVar, _ w: MILVar) -> [(String, [MILVar])] {
        [("x", [x]), ("weight", [w]), ("strides", [f.ints([1, 1])]), ("pad_type", [f.string("valid")]),
         ("pad", [f.ints([0, 0, 0, 0])]), ("dilations", [f.ints([1, 1])]), ("groups", [f.int(1)])]
    }

    /// A zero 8-bit weight `[out, in, 1, 1]` in blocks of 32 along `in`.
    static func blockwise(_ f: MILFunctionBuilder, out: Int, in inDim: Int) -> MILVar {
        let blocks = inDim / 32
        return f.blockwiseWeight(
            data: Data(count: out * inDim), scale: half(1, count: out * blocks),
            offset: half(0, count: out * blocks), shape: [out, inDim, 1, 1], blockShape: [out, blocks, 1, 1])
    }

    static func blockwise2D(_ f: MILFunctionBuilder, out: Int, in inDim: Int) -> MILVar {
        let blocks = inDim / 32
        return f.blockwiseWeight(
            data: Data(count: out * inDim), scale: half(1, count: out * blocks),
            offset: half(0, count: out * blocks), shape: [out, inDim], blockShape: [out, blocks])
    }

    static func random(_ count: Int) -> Data {
        var g = SystemRandomNumberGenerator()
        return Data((0 ..< count).map { _ in UInt8.random(in: 0 ... 255, using: &g) })
    }

    static func half(_ value: Float, count: Int) -> Data {
        let bits = Float16(value).bitPattern
        var d = Data(capacity: count * 2)
        for _ in 0 ..< count { d.append(UInt8(bits & 0xFF)); d.append(UInt8(bits >> 8)) }
        return d
    }

    /// Builds, compiles and plans one probe; reports each op's device.
    static func placement(name: String, _ build: Build, target: MLProgramPackage.Target) async throws -> String {
        let fm = FileManager.default
        let package = fm.temporaryDirectory.appendingPathComponent("ane-lab-\(name)-\(UUID().uuidString).mlpackage")
        defer { try? fm.removeItem(at: package) }
        let blobs = try MILBlobWriter(url: MLProgramPackage.weightsURL(in: package))
        let f = MILFunctionBuilder(blobs: blobs)
        build(f)
        try blobs.finish()
        try MLProgramPackage.write(
            specification: MLProgramPackage.specification(f, metadata: [:], target: target), to: package)
        let compiled = try await MLModel.compileModel(at: package)
        defer { try? fm.removeItem(at: compiled) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
            return "no program"
        }
        var parts: [String] = []
        for operation in main.block.operations where !operation.operatorName.hasPrefix("const") {
            let device: String
            switch plan.deviceUsage(for: operation)?.preferred {
            case .neuralEngine: device = "ANE"
            case .gpu: device = "GPU"
            case .cpu: device = "CPU"
            default: device = "?"
            }
            parts.append("\(operation.operatorName)→\(device)")
        }
        return parts.joined(separator: ", ")
    }
}
