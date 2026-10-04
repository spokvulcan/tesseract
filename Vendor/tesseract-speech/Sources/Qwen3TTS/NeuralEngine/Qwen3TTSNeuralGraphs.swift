import Foundation
@preconcurrency import MLX
import MLXNN

// The talker and the code predictor as Neural Engine graphs (ADR-0084, 0088).
//
// Both are Qwen3 decoder stacks, written in the Neural Engine's layout:
// activations `[1, channels, 1, positions]`, every projection a 1×1 conv,
// matrices int8 with one fp16 scale per output row (the format this engine
// streams at 8 bits; finer blocks or offsets send the convs to the CPU),
// everything else fp16.
//
// - The talker runs one position per call. Its KV cache is two Core ML
//   states, every layer's keys (and values) stacked along the channels. The
//   graph reads the cache in place and attends to the new position's key
//   and value beside it; it returns them, and the host writes them into the
//   states at the position. No call copies or rewrites the cache.
// - The code predictor runs a whole frame per call: its fifteen passes
//   unrolled, each sampling its code in the graph and looking up the code's
//   embedding as a matmul with a one-hot vector. On the Neural Engine
//   `topk`, `gather` and `argmax` run on the CPU; the ops used here don't.
//
// The norms run as the Neural Engine's layer norm over `[x, -x]`: its mean
// is zero, so its variance is x's mean square, and the engine's own kernel
// never squares x in fp16.
//
// Some layers' MLP products grow huge in one channel (the "massive
// activations" of LLMs: about 150,000 in the 0.6B code predictor's third
// layer, past fp16's 65,504). Through that channel, int8's rounding of the
// down projection swamps the residual stream (the code predictor's logits
// fell to 10 dB against MLX's), so such a layer keeps its down projection in
// fp16 and its product scaled down to fit (`NeuralPrecision`, measured with
// MLX). The residual stream itself (about 50,000 there) runs unscaled: scaled
// down, its small values fall below fp16's normal range and the Neural Engine
// loses them.

// MARK: - Constants from MLX weights

/// MLX parameters as graph constants. Conversion runs on MLX's CPU device,
/// so building a graph never uses the GPU.
struct NeuralConstants {
    let f: MILFunctionBuilder

    /// A linear layer's weight, dense, `[out, in]` float32.
    static func dense(_ linear: Linear) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            if let q = linear as? QuantizedLinear {
                return dequantized(
                    q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize,
                    bits: q.bits, mode: q.mode, dtype: .float32)
            }
            return linear.weight.asType(.float32)
        }
    }

    /// An embedding table, dense, `[count, dimensions]` float32.
    static func dense(_ embedding: Embedding) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            if let q = embedding as? QuantizedEmbedding {
                return dequantized(
                    q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize,
                    bits: q.bits, mode: q.mode, dtype: .float32)
            }
            return embedding.weight.asType(.float32)
        }
    }

    /// A `[out, in]` matrix as a 1×1 conv weight `[out, in, 1, 1]`.
    func matrix(_ w: MLXArray) -> MILVar {
        let (data, scale) = Self.int8Rows(w)
        return f.symmetricWeight(
            data: data, scale: scale, shape: [w.dim(0), w.dim(1), 1, 1],
            blockShape: [w.dim(0), 1, 1, 1])
    }

    /// `w` `[rows, columns]` as int8 with one fp16 scale per row: the row's
    /// largest magnitude over 127.
    static func int8Rows(_ w: MLXArray) -> (data: Data, scale: Data) {
        Device.withDefaultDevice(.cpu) {
            let x = w.asType(.float32)
            let scale = maximum(abs(x).max(axis: 1, keepDims: true) / 127, MLXArray(Float(1e-12)))
            let q = clip(round(x / scale), min: -127, max: 127).asType(.int8)
            let halves = scale.asType(.float16)
            eval(q, halves)
            return (q.asData(access: .copy).data, halves.asData(access: .copy).data)
        }
    }

    /// `w` `[out, in]` as an fp16 1×1 conv weight.
    func halfMatrix(_ w: MLXArray) -> MILVar {
        vector(w, shape: [w.dim(0), w.dim(1), 1, 1])
    }

    /// `v` as an fp16 constant of `shape`.
    func vector(_ v: MLXArray, shape: [Int]) -> MILVar {
        let halves = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let h = contiguous(v.asType(.float16))
            eval(h)
            return h
        }
        return f.weight(halves, shape: shape)
    }

    /// A norm's weight as the layer norm's gamma over `[x, -x]`: twice over,
    /// times `scale`.
    func gamma(_ weight: MLXArray, scale: Float = 1) -> MILVar {
        let w = Device.withDefaultDevice(.cpu) { weight.asType(.float32) * scale }
        return vector(
            Device.withDefaultDevice(.cpu) { concatenated([w, w], axis: 0) },
            shape: [2 * weight.dim(0)])
    }
}

// MARK: - The decoder layer

/// A Qwen3 decoder stack's sizes.
struct NeuralGeometry: Sendable, Equatable {
    let hidden: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let intermediate: Int
    let eps: Float
    let ropeBase: Float

    var groups: Int { heads / kvHeads }
    var kvWidth: Int { kvHeads * headDim }

    init(_ layer: Qwen3TTSDecoderLayer) {
        let a = layer.attention
        hidden = layer.inputNorm.weight.dim(0)
        heads = a.heads
        kvHeads = a.kvHeads
        headDim = a.headDim
        intermediate = layer.mlp.downProj.shape.1
        eps = layer.inputNorm.eps
        ropeBase = a.ropeBase
    }
}

/// Which layers of a stack can't take int8 throughout: those whose MLP
/// product grows past `int8ProductLimit`. Each keeps its down projection in
/// fp16, with the product scaled by a power of two to stay inside fp16 (the
/// up projection carries the scale, down undoes it).
struct NeuralPrecision: Sendable, Equatable, Codable, CustomStringConvertible {
    /// Layer index to its product's scale.
    var outliers: [Int: Float]

    static let int8 = NeuralPrecision(outliers: [:])

    /// The largest product a layer may reach and keep int8 down weights. The
    /// 0.6B talker's reach about 1,400 and match MLX at 33 dB.
    static let int8ProductLimit: Float = 8_192

    /// From each layer's largest MLP product.
    init(largestProducts: [Float]) {
        var outliers: [Int: Float] = [:]
        for (layer, largest) in largestProducts.enumerated() where largest > Self.int8ProductLimit {
            outliers[layer] = pow(2, -(log2(largest / Self.int8ProductLimit)).rounded(.up))
        }
        self.outliers = outliers
    }

    init(outliers: [Int: Float]) {
        self.outliers = outliers
    }

    var description: String {
        outliers.isEmpty
            ? "int8"
            : "int8, fp16 down in layer "
                + outliers.keys.sorted().map { "\($0) (product ×\(outliers[$0]!))" }
                .joined(separator: ", ")
    }
}

/// One decoder layer's constants.
struct NeuralLayer {
    let inputNorm: MILVar
    let postNorm: MILVar
    /// q, k and v stacked: `[(heads + 2 kv) · hd, hidden, 1, 1]`.
    let qkv: MILVar
    /// The query norm carries attention's 1/√hd, so scores need no scaling.
    let qNorm: MILVar
    let kNorm: MILVar
    let o: MILVar
    /// gate and up stacked: `[2 · intermediate, hidden, 1, 1]`.
    let gateUp: MILVar
    let down: MILVar

    /// `productScale`, for an outlier layer: fp16 down, the product scaled.
    init(_ layer: Qwen3TTSDecoderLayer, constants c: NeuralConstants, productScale: Float? = nil) {
        let a = layer.attention
        let mlp = layer.mlp
        let qkvDense =
            a.qkvProj.map(NeuralConstants.dense)
            ?? Device.withDefaultDevice(.cpu) {
                concatenated([a.qProj!, a.kProj!, a.vProj!].map(NeuralConstants.dense), axis: 0)
            }
        let gateUpDense =
            mlp.gateUpProj.map(NeuralConstants.dense)
            ?? Device.withDefaultDevice(.cpu) {
                concatenated([mlp.gateProj!, mlp.upProj!].map(NeuralConstants.dense), axis: 0)
            }
        let intermediate = gateUpDense.dim(0) / 2
        inputNorm = c.gamma(layer.inputNorm.weight)
        postNorm = c.gamma(layer.postNorm.weight)
        qkv = c.matrix(qkvDense)
        qNorm = c.gamma(a.qNorm.weight, scale: a.scale)
        kNorm = c.gamma(a.kNorm.weight)
        o = c.matrix(NeuralConstants.dense(a.oProj))
        if let productScale {
            gateUp = c.matrix(
                Device.withDefaultDevice(.cpu) {
                    concatenated(
                        [gateUpDense[..<intermediate], gateUpDense[intermediate...] * productScale],
                        axis: 0)
                })
            down = c.halfMatrix(
                Device.withDefaultDevice(.cpu) { NeuralConstants.dense(mlp.downProj) / productScale })
        } else {
            gateUp = c.matrix(gateUpDense)
            down = c.matrix(NeuralConstants.dense(mlp.downProj))
        }
    }
}

/// What stands for minus infinity in fp16: far below any score, and its
/// exponential is zero.
let neuralExcluded: Float = -30_000

/// MIL emission for the decoder, in the `[1, C, 1, n]` layout.
struct NeuralOps {
    let f: MILFunctionBuilder
    let g: NeuralGeometry

    func conv(_ x: MILVar, _ w: MILVar, out: Int) -> MILVar {
        f.op(
            "conv",
            [
                ("x", [x]), ("weight", [w]), ("strides", [f.ints([1, 1])]),
                ("pad_type", [f.string("valid")]), ("pad", [f.ints([0, 0, 0, 0])]),
                ("dilations", [f.ints([1, 1])]), ("groups", [f.int(1)]),
            ],
            .fp16([1, out, 1, x.type.shape[3]]))
    }

    func mul(_ x: MILVar, _ y: MILVar) -> MILVar { binary("mul", x, y) }
    func add(_ x: MILVar, _ y: MILVar) -> MILVar { binary("add", x, y) }
    func sub(_ x: MILVar, _ y: MILVar) -> MILVar { binary("sub", x, y) }

    /// An elementwise op on broadcast shapes; a comparison's result is bool.
    func binary(_ name: String, _ x: MILVar, _ y: MILVar, bool: Bool = false) -> MILVar {
        f.op(
            name, [("x", [x]), ("y", [y])],
            MILType(
                dataType: bool ? .bool : x.type.dataType,
                shape: Self.broadcast(x.type.shape, y.type.shape)))
    }

    static func broadcast(_ a: [Int], _ b: [Int]) -> [Int] {
        guard a.count == b.count else { return a.count >= b.count ? a : b }
        return zip(a, b).map { max($0, $1) }
    }

    func reshape(_ x: MILVar, _ shape: [Int]) -> MILVar {
        f.op(
            "reshape", [("x", [x]), ("shape", [f.ints(shape)])],
            MILType(dataType: x.type.dataType, shape: shape))
    }

    func transpose(_ x: MILVar, _ perm: [Int]) -> MILVar {
        f.op(
            "transpose", [("x", [x]), ("perm", [f.ints(perm)])],
            MILType(dataType: x.type.dataType, shape: perm.map { x.type.shape[$0] }))
    }

    /// `x` cut to `range` along `axis`.
    func slice(_ x: MILVar, axis: Int, _ range: Range<Int>) -> MILVar {
        var begin = [Int](repeating: 0, count: x.type.shape.count)
        var end = x.type.shape
        begin[axis] = range.lowerBound
        end[axis] = range.upperBound
        var shape = x.type.shape
        shape[axis] = range.count
        return f.op(
            "slice_by_index", [("x", [x]), ("begin", [f.ints(begin)]), ("end", [f.ints(end)])],
            MILType(dataType: x.type.dataType, shape: shape))
    }

    func concat(_ values: [MILVar], axis: Int, name: String? = nil) -> MILVar {
        var shape = values[0].type.shape
        shape[axis] = values.map { $0.type.shape[axis] }.reduce(0, +)
        return f.op(
            "concat",
            [("values", values), ("axis", [f.int(axis)]), ("interleave", [f.bool(false)])],
            MILType(dataType: values[0].type.dataType, shape: shape), name: name)
    }

    func reduce(_ name: String, _ x: MILVar, axis: Int) -> MILVar {
        var shape = x.type.shape
        shape[axis] = 1
        return f.op(
            name, [("x", [x]), ("axes", [f.ints([axis])]), ("keep_dims", [f.bool(true)])],
            .fp16(shape))
    }

    func cast(_ x: MILVar) -> MILVar {
        f.op("cast", [("x", [x]), ("dtype", [f.string("fp16")])], .fp16(x.type.shape))
    }

    func select(_ condition: MILVar, _ a: MILVar, _ b: MILVar) -> MILVar {
        let shape = Self.broadcast(condition.type.shape, Self.broadcast(a.type.shape, b.type.shape))
        return f.op("select", [("cond", [condition]), ("a", [a]), ("b", [b])], .fp16(shape))
    }

    /// `x` under the output name `name`.
    func named(_ x: MILVar, _ name: String) -> MILVar {
        f.op("identity", [("x", [x])], x.type, name: name)
    }

    /// RMSNorm over `axis`: the layer norm over `[x, -x]` with `gamma` (the
    /// weight twice over), then the first half.
    func rmsNorm(_ x: MILVar, axis: Int, gamma: MILVar) -> MILVar {
        let doubled = concat([x, mul(x, f.half(-1))], axis: axis)
        let normed = f.op(
            "layer_norm",
            [
                ("x", [doubled]), ("axes", [f.ints([axis])]), ("gamma", [gamma]),
                ("epsilon", [f.half(g.eps)]),
            ],
            doubled.type)
        return slice(normed, axis: axis, 0 ..< x.type.shape[axis])
    }

    /// Rotary positions on `[1, h, hd, n]`, rotating halves (Qwen's
    /// non-interleaved form), with `cos` and `sin` `[1, 1, hd, n]`.
    func rope(_ x: MILVar, cos: MILVar, sin: MILVar) -> MILVar {
        let half = g.headDim / 2
        let first = slice(x, axis: 2, 0 ..< half)
        let second = slice(x, axis: 2, half ..< g.headDim)
        let rotated = concat([mul(second, f.half(-1)), first], axis: 2)
        return add(mul(x, cos), mul(rotated, sin))
    }

    /// The layer's queries, keys and values for `x` `[1, hidden, 1, n]`:
    /// q `[1, heads, hd, n]` (scaled), k and v `[1, kv, hd, n]`.
    func projections(_ x: MILVar, layer: NeuralLayer, cos: MILVar, sin: MILVar)
        -> (q: MILVar, k: MILVar, v: MILVar)
    {
        let n = x.type.shape[3]
        let normed = rmsNorm(x, axis: 1, gamma: layer.inputNorm)
        let qWidth = g.heads * g.headDim
        let qkv = conv(normed, layer.qkv, out: qWidth + 2 * g.kvWidth)
        let q = reshape(slice(qkv, axis: 1, 0 ..< qWidth), [1, g.heads, g.headDim, n])
        let k = reshape(
            slice(qkv, axis: 1, qWidth ..< (qWidth + g.kvWidth)), [1, g.kvHeads, g.headDim, n])
        let v = reshape(
            slice(qkv, axis: 1, (qWidth + g.kvWidth) ..< (qWidth + 2 * g.kvWidth)),
            [1, g.kvHeads, g.headDim, n])
        return (
            rope(rmsNorm(q, axis: 2, gamma: layer.qNorm), cos: cos, sin: sin),
            rope(rmsNorm(k, axis: 2, gamma: layer.kNorm), cos: cos, sin: sin),
            v
        )
    }

    /// Queries `[1, heads, hd, n]` grouped by kv head: `[1, kv, groups · n,
    /// hd]`, row `group · n + position`.
    func grouped(_ q: MILVar) -> MILVar {
        let n = q.type.shape[3]
        return n == 1
            ? reshape(q, [1, g.kvHeads, g.groups, g.headDim])
            : reshape(transpose(q, [0, 1, 3, 2]), [1, g.kvHeads, g.groups * n, g.headDim])
    }

    /// Grouped attention outputs `[1, kv, groups · n, hd]` back to `[1,
    /// heads · hd, 1, n]`.
    func ungrouped(_ out: MILVar, n: Int) -> MILVar {
        n == 1
            ? reshape(out, [1, g.heads * g.headDim, 1, 1])
            : reshape(
                transpose(reshape(out, [1, g.heads, n, g.headDim]), [0, 1, 3, 2]),
                [1, g.heads * g.headDim, 1, n])
    }

    func matmul(_ x: MILVar, _ y: MILVar, transposeY: Bool, shape: [Int]) -> MILVar {
        f.op(
            "matmul",
            [
                ("x", [x]), ("y", [y]), ("transpose_x", [f.bool(false)]),
                ("transpose_y", [f.bool(transposeY)]),
            ],
            .fp16(shape))
    }

    /// Attention over keys and values `[1, kv, hd, m]` held in the graph;
    /// `mask` `[1, 1, groups · n, m]` adds to the scores.
    func attention(q: MILVar, k: MILVar, v: MILVar, mask: MILVar?) -> MILVar {
        let n = q.type.shape[3]
        let m = k.type.shape[3]
        let rows = g.groups * n
        let scores = matmul(grouped(q), k, transposeY: false, shape: [1, g.kvHeads, rows, m])
        let masked = mask.map { add(scores, $0) } ?? scores
        let p = f.op("softmax", [("x", [masked]), ("axis", [f.int(-1)])], masked.type)
        let out = matmul(p, v, transposeY: true, shape: [1, g.kvHeads, rows, g.headDim])
        return ungrouped(out, n: n)
    }

    /// One query position over a cache `[1, kv, hd, L]` (`mask` `[1, 1, 1,
    /// L]` keeps the positions written so far) and the position's own key
    /// and value `[1, kv, hd, 1]`. Returns the output and the scores `[1,
    /// kv, groups, L + 1]`, the position's own last.
    func attention(
        q: MILVar, cachedK: MILVar, cachedV: MILVar, mask: MILVar, k: MILVar, v: MILVar
    ) -> (out: MILVar, scores: MILVar) {
        let length = cachedK.type.shape[3]
        let queries = grouped(q)
        let cached = add(
            matmul(queries, cachedK, transposeY: false, shape: [1, g.kvHeads, g.groups, length]),
            mask)
        let own = matmul(queries, k, transposeY: false, shape: [1, g.kvHeads, g.groups, 1])
        let scores = concat([cached, own], axis: 3)
        let p = f.op("softmax", [("x", [scores]), ("axis", [f.int(-1)])], scores.type)
        let fromCache = matmul(
            slice(p, axis: 3, 0 ..< length), cachedV, transposeY: true,
            shape: [1, g.kvHeads, g.groups, g.headDim])
        let fromOwn = mul(
            slice(p, axis: 3, length ..< (length + 1)), reshape(v, [1, g.kvHeads, 1, g.headDim]))
        return (ungrouped(add(fromCache, fromOwn), n: 1), scores)
    }

    /// The rest of the layer after attention: o, the residual, the MLP.
    func finish(_ x: MILVar, attention: MILVar, layer: NeuralLayer) -> MILVar {
        let n = x.type.shape[3]
        let h = add(x, conv(attention, layer.o, out: g.hidden))
        let normed = rmsNorm(h, axis: 1, gamma: layer.postNorm)
        let gateUp = conv(normed, layer.gateUp, out: 2 * g.intermediate)
        let gate = f.op(
            "silu", [("x", [slice(gateUp, axis: 1, 0 ..< g.intermediate)])],
            .fp16([1, g.intermediate, 1, n]))
        let up = slice(gateUp, axis: 1, g.intermediate ..< (2 * g.intermediate))
        return add(h, conv(mul(gate, up), layer.down, out: g.hidden))
    }
}

/// Qwen's rotary angles at `positions`, as `[headDim, n]` tables
/// (position-minor), both halves at the same frequencies.
func neuralRopeTables(positions: Range<Int>, headDim: Int, base: Float)
    -> (cos: [Float], sin: [Float])
{
    let half = headDim / 2
    let n = positions.count
    var cos = [Float](repeating: 0, count: headDim * n)
    var sin = cos
    for d in 0 ..< headDim {
        let frequency = 1 / pow(base, Float(2 * (d % half)) / Float(headDim))
        for (i, p) in positions.enumerated() {
            let angle = Float(p) * frequency
            cos[d * n + i] = Foundation.cos(angle)
            sin[d * n + i] = Foundation.sin(angle)
        }
    }
    return (cos, sin)
}

// MARK: - The talker

enum Qwen3TTSNeuralTalkerGraph {
    /// Bump when the graph changes: part of the cache key.
    static let version = 2

    /// Emits the talker's step: one position in; its codec logits, its final
    /// hidden state, its keys and values for every layer (for the host to
    /// store at the position) and, with an alignment head, that head's
    /// scaled scores over the cache and then the position itself.
    ///
    /// Inputs: `x` `[1, hidden, 1, 1]`; `cos`, `sin` `[1, 1, hd, 1]` for the
    /// position; `mask` `[1, 1, 1, L]`, 0 before the position and excluded
    /// from it on; the states `keys` and `values` `[1, layers · kv · hd, 1,
    /// L]`.
    static func build(
        _ f: MILFunctionBuilder, talker: Qwen3TTSTalker, contextLength length: Int,
        alignment: Qwen3TTSAlignmentHead?, precision: NeuralPrecision
    ) {
        let layers = talker.model.layers
        let g = NeuralGeometry(layers[0])
        let c = NeuralConstants(f: f)
        let ops = NeuralOps(f: f, g: g)
        let cacheWidth = layers.count * g.kvWidth
        let vocabulary = talker.codecHead.shape.0

        var x = f.input("x", .fp16([1, g.hidden, 1, 1]))
        let cos = f.input("cos", .fp16([1, 1, g.headDim, 1]))
        let sin = f.input("sin", .fp16([1, 1, g.headDim, 1]))
        let mask = f.input("mask", .fp16([1, 1, 1, length]))
        let keysState = f.input("keys", .state([1, cacheWidth, 1, length]))
        let valuesState = f.input("values", .state([1, cacheWidth, 1, length]))
        let keys = f.op("read_state", [("input", [keysState])], .fp16([1, cacheWidth, 1, length]))
        let values = f.op(
            "read_state", [("input", [valuesState])], .fp16([1, cacheWidth, 1, length]))

        var newKeys: [MILVar] = []
        var newValues: [MILVar] = []
        var alignmentRow: MILVar?
        for (i, module) in layers.enumerated() {
            let layer = NeuralLayer(module, constants: c, productScale: precision.outliers[i])
            let (q, k, v) = ops.projections(x, layer: layer, cos: cos, sin: sin)
            newKeys.append(ops.reshape(k, [1, g.kvWidth, 1, 1]))
            newValues.append(ops.reshape(v, [1, g.kvWidth, 1, 1]))
            let span = (i * g.kvWidth) ..< ((i + 1) * g.kvWidth)
            let cacheShape = [1, g.kvHeads, g.headDim, length]
            let (heard, scores) = ops.attention(
                q: q, cachedK: ops.reshape(ops.slice(keys, axis: 1, span), cacheShape),
                cachedV: ops.reshape(ops.slice(values, axis: 1, span), cacheShape),
                mask: mask, k: k, v: v)
            if let alignment, alignment.layer == i {
                let kvHead = alignment.head / g.groups
                let row = alignment.head % g.groups
                alignmentRow = ops.slice(
                    ops.slice(scores, axis: 1, kvHead ..< (kvHead + 1)), axis: 2,
                    row ..< (row + 1))
            }
            x = ops.finish(x, attention: heard, layer: layer)
        }
        let hidden = ops.rmsNorm(x, axis: 1, gamma: c.gamma(talker.model.norm.weight))
        let logits = ops.conv(
            hidden, c.matrix(NeuralConstants.dense(talker.codecHead)), out: vocabulary)

        f.output(ops.named(logits, "logits"))
        f.output(ops.named(hidden, "hidden"))
        f.output(ops.concat(newKeys, axis: 1, name: "new_keys"))
        f.output(ops.concat(newValues, axis: 1, name: "new_values"))
        if let alignmentRow { f.output(ops.named(alignmentRow, "alignment")) }
    }
}

// MARK: - The code predictor

enum Qwen3TTSNeuralCodePredictorGraph {
    /// Bump when the graph changes: part of the cache key.
    static let version = 2

    /// The top-k cut is read off two grids of this many thresholds: one unit
    /// apart below the best logit, then 1/32 of a unit apart.
    static let gridSize = 32

    /// Emits one frame of the code predictor: `numCodeGroups - 1` passes,
    /// each sampling the next code among the top `topK` and adding its
    /// embedding.
    ///
    /// Inputs: `hidden` and `code0_embed` `[1, talkerHidden, 1, 1]`; `noise`
    /// `[1, vocabulary, 1, passes]`, Gumbel noise (zeros for greedy);
    /// `inverse_temperature` `[1, 1, 1, 1]`. Outputs: `codes` `[1, 1, 1,
    /// passes]` (fp16 integers) and `embed_sum` `[1, talkerHidden, 1, 1]`, the
    /// first code's embedding plus every sampled code's.
    ///
    /// `forced` (the bench's parity check) takes the codes instead, one-hot
    /// `codes_onehot` `[1, vocabulary, 1, passes]` in place of the noise and
    /// the temperature, and returns every pass's `logits` `[1, vocabulary, 1,
    /// passes]` in place of the codes.
    static func build(
        _ f: MILFunctionBuilder, predictor: Qwen3TTSCodePredictor, topK: Int,
        precision: NeuralPrecision, forced: Bool = false
    ) {
        let modules = predictor.model.layers
        let g = NeuralGeometry(modules[0])
        let c = NeuralConstants(f: f)
        let ops = NeuralOps(f: f, g: g)
        let passes = predictor.numCodeGroups - 1
        let vocabulary = predictor.lmHead[0].shape.0
        let talkerHidden = predictor.codecEmbedding[0].shape.1
        precondition(vocabulary <= 2048, "fp16 holds every code index exactly up to 2048")

        let hidden = f.input("hidden", .fp16([1, talkerHidden, 1, 1]))
        let code0 = f.input("code0_embed", .fp16([1, talkerHidden, 1, 1]))
        let noise = f.input(forced ? "codes_onehot" : "noise", .fp16([1, vocabulary, 1, passes]))
        let inverseTemperature = forced ? nil : f.input("inverse_temperature", .fp16([1, 1, 1, 1]))

        let layers = modules.enumerated().map {
            NeuralLayer($1, constants: c, productScale: precision.outliers[$0])
        }
        let finalNorm = c.gamma(predictor.model.norm.weight)
        let projection = predictor.projection.map { linear in
            (
                weight: c.matrix(NeuralConstants.dense(linear)),
                bias: linear.bias.map { c.vector($0, shape: [1, g.hidden, 1, 1]) }
            )
        }
        let heads = predictor.lmHead.map { c.matrix(NeuralConstants.dense($0)) }
        // Embedding tables transposed, `[talkerHidden, vocabulary]`: one
        // times a one-hot code is the code's row.
        let tables = predictor.codecEmbedding.map { embedding in
            c.matrix(Device.withDefaultDevice(.cpu) { NeuralConstants.dense(embedding).transposed() })
        }
        let sampler = Sampler(ops: ops, vocabulary: vocabulary, topK: topK)

        let rope = neuralRopeTables(
            positions: 0 ..< (passes + 1), headDim: g.headDim, base: g.ropeBase)
        func rotary(_ positions: Range<Int>) -> (cos: MILVar, sin: MILVar) {
            func pick(_ table: [Float]) -> [Float] {
                (0 ..< g.headDim).flatMap { d in positions.map { table[d * (passes + 1) + $0] } }
            }
            let shape = [1, 1, g.headDim, positions.count]
            return (f.halves(pick(rope.cos), shape: shape), f.halves(pick(rope.sin), shape: shape))
        }
        // The first pass's two positions: the first sees only itself.
        let firstMask = f.halves(
            (0 ..< g.groups).flatMap { _ in [0, neuralExcluded, 0, 0] },
            shape: [1, 1, g.groups * 2, 2])

        var keys = [MILVar?](repeating: nil, count: layers.count)
        var values = [MILVar?](repeating: nil, count: layers.count)
        var embedSum = code0
        var codes: [MILVar] = []
        var allLogits: [MILVar] = []
        var input = ops.concat([hidden, code0], axis: 3)
        var position = 0
        for pass in 0 ..< passes {
            let n = input.type.shape[3]
            var x = input
            if let projection {
                x = ops.conv(x, projection.weight, out: g.hidden)
                if let bias = projection.bias { x = ops.add(x, bias) }
            }
            let (cos, sin) = rotary(position ..< (position + n))
            for (i, layer) in layers.enumerated() {
                let (q, k, v) = ops.projections(x, layer: layer, cos: cos, sin: sin)
                let allK = keys[i].map { ops.concat([$0, k], axis: 3) } ?? k
                let allV = values[i].map { ops.concat([$0, v], axis: 3) } ?? v
                keys[i] = allK
                values[i] = allV
                let heard = ops.attention(
                    q: q, k: allK, v: allV, mask: pass == 0 ? firstMask : nil)
                x = ops.finish(x, attention: heard, layer: layer)
            }
            position += n
            let last = n == 1 ? x : ops.slice(x, axis: 3, (n - 1) ..< n)
            let logits = ops.conv(
                ops.rmsNorm(last, axis: 1, gamma: finalNorm), heads[pass], out: vocabulary)
            let onehot: MILVar
            if let inverseTemperature {
                let drawn = sampler.draw(
                    logits: logits, noise: ops.slice(noise, axis: 3, pass ..< (pass + 1)),
                    inverseTemperature: inverseTemperature)
                codes.append(drawn.code)
                onehot = drawn.onehot
            } else {
                allLogits.append(logits)
                onehot = ops.slice(noise, axis: 3, pass ..< (pass + 1))
            }
            let embedding = ops.conv(onehot, tables[pass], out: talkerHidden)
            embedSum = ops.add(embedSum, embedding)
            input = embedding
        }
        f.output(
            forced
                ? ops.concat(allLogits, axis: 3, name: "logits")
                : ops.concat(codes, axis: 3, name: "codes"))
        f.output(ops.named(embedSum, "embed_sum"))
    }

    /// Draws a code from `softmax(logits / T)` over its top k, as
    /// Gumbel-max: the best `z + noise` among the kept codes, `z` being the
    /// logits over T less their maximum (fp16 is finest near zero).
    ///
    /// The cut is the highest threshold that keeps at least k codes, read off
    /// two grids of thresholds at once rather than searched for. So ties at
    /// the cut stay, as in the MLX sampler, and so do codes within 1/32 of
    /// it, whose chance is about the cut's own, the smallest kept. Codes
    /// more than `gridSize` below the best are never drawn: their chance is
    /// under e⁻³². A tie for the best goes to the lowest code, so the one-hot
    /// has a single one.
    struct Sampler {
        let ops: NeuralOps
        let iota: MILVar
        let notACode: MILVar
        let coarse: MILVar
        let fine: MILVar
        let k: MILVar
        let excluded: MILVar

        init(ops: NeuralOps, vocabulary: Int, topK: Int) {
            let f = ops.f
            let size = Qwen3TTSNeuralCodePredictorGraph.gridSize
            self.ops = ops
            iota = f.halves((0 ..< vocabulary).map(Float.init), shape: [1, vocabulary, 1, 1])
            notACode = f.half(Float(2 * vocabulary))
            coarse = f.halves((1 ... size).map { -Float($0) }, shape: [1, 1, 1, size])
            fine = f.halves((0 ..< size).map { Float($0) / Float(size) }, shape: [1, 1, 1, size])
            k = f.half(Float(topK))
            excluded = f.half(neuralExcluded)
        }

        /// The code `[1, 1, 1, 1]` and its one-hot `[1, vocabulary, 1, 1]`.
        func draw(logits: MILVar, noise: MILVar, inverseTemperature: MILVar)
            -> (code: MILVar, onehot: MILVar)
        {
            let scaled = ops.mul(logits, inverseTemperature)
            let z = ops.sub(scaled, ops.reduce("reduce_max", scaled, axis: 1))
            let low = highest(of: coarse, keeping: z)
            let cut = highest(of: ops.add(low, fine), keeping: z)
            let kept = ops.binary("greater_equal", z, cut, bool: true)
            let candidates = ops.select(kept, ops.add(z, noise), excluded)
            let best = ops.reduce("reduce_max", candidates, axis: 1)
            let winners = ops.binary("equal", candidates, best, bool: true)
            let code = ops.reduce("reduce_min", ops.select(winners, iota, notACode), axis: 1)
            return (code, ops.cast(ops.binary("equal", iota, code, bool: true)))
        }

        /// The highest of `thresholds` `[1, 1, 1, m]` that keeps at least k
        /// of `z`'s codes, or the excluded value when none does.
        private func highest(of thresholds: MILVar, keeping z: MILVar) -> MILVar {
            let above = ops.cast(ops.binary("greater_equal", z, thresholds, bool: true))
            let enough = ops.binary(
                "greater_equal", ops.reduce("reduce_sum", above, axis: 1), k, bool: true)
            return ops.reduce("reduce_max", ops.select(enough, thresholds, excluded), axis: 3)
        }
    }
}
