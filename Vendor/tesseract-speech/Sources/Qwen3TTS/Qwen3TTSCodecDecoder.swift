import Foundation
@preconcurrency import MLX
import MLXNN

/// The Qwen3-TTS-Tokenizer-12Hz decoder: codec frames to 24 kHz audio.
///
/// It follows Qwen's `Qwen3TTSTokenizerV2Decoder`: residual codebooks, a
/// causal conv, an 8-layer transformer that attends over a 72-frame sliding
/// window, two ×2 upsamplers with ConvNeXt blocks, then four upsampling blocks
/// (×8, ×5, ×4, ×3) of SnakeBeta, a transposed conv and three dilated residual
/// units. Every layer is causal, so decoding a stream a chunk at a time with
/// the right carried state gives exactly the audio of decoding it in one pass.
/// `Stream` carries that state; the chunk size never changes the samples.
///
/// Weights are converted once at load: MLX's channels-last layout, contiguous,
/// in `dtype` (fp16 by default: 55–59 dB SNR against Qwen's fp32 decoder, half
/// the memory). The codebooks are folded through their output projections
/// into one lookup table. Each transposed conv becomes its polyphase form: a
/// stride-1 conv with two taps (x[t-1], x[t]) to `stride × out` channels,
/// whose channels-last output reshapes into the upsampled sequence. That is
/// exact for Qwen's kernels (stride or twice the stride), skips the zeros a
/// transposed conv would multiply, and leaves one input frame as its state.
/// The encoder half, which only voice cloning from reference audio uses, is
/// never read.
package final class Qwen3TTSCodecDecoder: @unchecked Sendable {
    package let config: Qwen3TTSTokenizerDecoderConfig
    /// Audio samples per codec frame: 1,920 at 24 kHz, 12.5 frames a second.
    package let samplesPerFrame: Int
    package let dtype: DType
    let front: FrontEnd

    /// The conv stack's weights (two thirds of the decoder's memory). While
    /// the Neural Engine runs the conv stack they can be released; the next
    /// MLX synthesis reads them back from the checkpoint.
    private var loadedStack: ConvStack?
    private let reloadStack: (@Sendable () throws -> ConvStack)?
    private let stackLock = NSLock()

    var convStack: ConvStack {
        get throws {
            try stackLock.withLock {
                if let loadedStack { return loadedStack }
                guard let reloadStack else {
                    throw AudioGenerationError.modelNotInitialized("The conv stack was released.")
                }
                let stack = try reloadStack()
                loadedStack = stack
                return stack
            }
        }
    }

    /// Frees the conv stack's MLX weights, when they can be read back.
    @discardableResult
    package func releaseConvStack() -> Bool {
        stackLock.withLock {
            guard reloadStack != nil else { return false }
            loadedStack = nil
            return true
        }
    }

    // MARK: - Streaming state

    /// What decoding a stream carries from one chunk to the next: each causal
    /// conv's left context (for an upsampler, its last input frame), and the
    /// transformer's last `slidingWindow - 1` keys and values per layer.
    package struct Stream {
        fileprivate var position = 0
        fileprivate var preConv: MLXArray?
        fileprivate var keys: [MLXArray?]
        fileprivate var values: [MLXArray?]
        fileprivate var convNeXt: [MLXArray?]
        fileprivate var initialConv: MLXArray?
        fileprivate var blockUpsample: [MLXArray?]
        fileprivate var residual: [[MLXArray?]]
        fileprivate var finalConv: MLXArray?

        /// Frames decoded so far.
        package var framesDecoded: Int { position }

        fileprivate init(layers: Int, upsamplers: Int, blocks: Int, unitsPerBlock: Int) {
            keys = Array(repeating: nil, count: layers)
            values = Array(repeating: nil, count: layers)
            convNeXt = Array(repeating: nil, count: upsamplers)
            blockUpsample = Array(repeating: nil, count: blocks)
            residual = Array(
                repeating: Array(repeating: nil, count: unitsPerBlock), count: blocks)
        }

        /// The arrays the state holds, to evaluate with a chunk's audio so no
        /// lazy graph stays chained between chunks.
        package var arrays: [MLXArray] {
            ([preConv, initialConv, finalConv] + keys + values + convNeXt + blockUpsample
                + residual.flatMap { $0 }).compactMap { $0 }
        }
    }

    package func makeStream() -> Stream {
        Stream(
            layers: front.layers.count, upsamplers: config.upsamplingRatios.count,
            blocks: config.upsampleRates.count, unitsPerBlock: 3)
    }

    // MARK: - Decoding

    /// Decodes the next `codes` of a stream, `[1, frames, numQuantizers]`
    /// int32, to `[frames × samplesPerFrame]` float32 samples in [-1, 1].
    /// Lazy: evaluate the result (and `stream.arrays`) to run it.
    package func decode(_ codes: MLXArray, stream: inout Stream) throws -> MLXArray {
        try synthesize(latent(codes, stream: &stream), stream: &stream)
    }

    /// The front end: codes through the codebooks, the causal pre-conv and
    /// the windowed transformer, to the `[1, frames, latentDim]` latent the
    /// conv stack upsamples. About 2% of the decoder's work.
    package func latent(_ codes: MLXArray, stream: inout Stream) -> MLXArray {
        let w = front
        let frames = codes.dim(1)
        // Codebooks, already projected: one gather over the stacked tables.
        let offsets = MLXArray(
            (0 ..< config.numQuantizers).map { Int32($0 * config.codebookSize) })
        var x = w.codebooks[codes + offsets].sum(axis: 2)  // [1, T, codebookDim]
        x = causalConv(x, w.preConv, context: &stream.preConv)
        x = transformer(x, stream: &stream)
        stream.position += frames
        return x
    }

    /// The conv stack: the latent through the two ×2 upsamplers, the initial
    /// conv, the four upsampling blocks and the output conv, to samples.
    package func synthesize(_ latent: MLXArray, stream: inout Stream) throws -> MLXArray {
        let w = try convStack
        var x = latent
        for (index, upsampler) in w.upsamplers.enumerated() {
            var none: MLXArray?  // kernel == stride: one tap, no context
            x = upsample(x, upsampler.upsample, context: &none)
            x = convNeXt(x, upsampler.block, context: &stream.convNeXt[index])
        }
        x = causalConv(x, w.initialConv, context: &stream.initialConv)
        for (b, block) in w.blocks.enumerated() {
            x = snake(x, block.snake)
            x = upsample(x, block.upsample, context: &stream.blockUpsample[b])
            for (u, unit) in block.units.enumerated() {
                var h = snake(x, unit.snake1)
                h = causalConv(h, unit.conv1, context: &stream.residual[b][u])
                h = snake(h, unit.snake2)
                var pointwise: MLXArray?  // kernel 1: no context
                x = x + causalConv(h, unit.conv2, context: &pointwise)
            }
        }
        x = snake(x, w.finalSnake)
        x = causalConv(x, w.finalConv, context: &stream.finalConv)
        return clip(x, min: -1, max: 1).reshaped(-1).asType(.float32)
    }

    /// Decodes a whole sequence in one pass, `[1, frames, numQuantizers]`.
    package func decodeAll(_ codes: MLXArray) throws -> MLXArray {
        var stream = makeStream()
        return try decode(codes, stream: &stream)
    }

    // MARK: - Layers

    /// A causal conv: the previous chunk's last inputs (zeros at the start)
    /// in front of this chunk, then a valid convolution.
    private func causalConv(_ x: MLXArray, _ conv: Conv, context: inout MLXArray?) -> MLXArray {
        var input = x
        let needed = (conv.kernel - 1) * conv.dilation
        if needed > 0 {
            let previous = context ?? MLXArray.zeros([x.dim(0), needed, x.dim(2)], dtype: x.dtype)
            input = concatenated([previous, x], axis: 1)
            context = input[0..., (input.dim(1) - needed)..., 0...]
        }
        let y = conv1d(input, conv.weight, dilation: conv.dilation, groups: conv.groups)
        return conv.bias.map { y + $0 } ?? y
    }

    /// A causal transposed conv in polyphase form: output `t·s + j` is
    /// phase `j` of the two-tap conv at input `t`, so the conv's
    /// `[1, T, s·out]` output is already `[1, T·s, out]` in memory.
    private func upsample(_ x: MLXArray, _ conv: Upsample, context: inout MLXArray?) -> MLXArray {
        let y = causalConv(x, conv.conv, context: &context)
        return y.reshaped(y.dim(0), y.dim(1) * conv.stride, -1)
    }

    private func snake(_ x: MLXArray, _ s: Snake) -> MLXArray {
        compiledSnake(x, s.alpha, s.inverseBeta)
    }

    private func convNeXt(_ x: MLXArray, _ block: ConvNeXt, context: inout MLXArray?) -> MLXArray {
        var h = causalConv(x, block.depthwise, context: &context)
        h = MLXFast.layerNorm(h, weight: block.normWeight, bias: block.normBias, eps: 1e-6)
        h = block.pointwise2(gelu(block.pointwise1(h)))
        return x + block.gamma * h
    }

    /// The transformer between the codebooks and the upsamplers. Each query
    /// sees itself and the 71 frames before it (Qwen's sliding window of 72),
    /// so the carried keys and values stay bounded however long the stream.
    private func transformer(_ input: MLXArray, stream: inout Stream) -> MLXArray {
        let w = front
        let frames = input.dim(1)
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let window = config.slidingWindow
        let start = stream.position

        var x = w.inputProjection(input)
        let cachedCount = stream.keys.first.flatMap { $0?.dim(2) } ?? 0
        let mask = Self.slidingWindowMask(
            queries: frames, cached: cachedCount, window: window)

        for (i, layer) in w.layers.enumerated() {
            let h = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: config.rmsNormEps)
            let qkv = layer.qkv(h).reshaped(1, frames, heads + 2 * kvHeads, headDim)
            var q = qkv[0..., 0..., ..<heads, 0...].transposed(0, 2, 1, 3)
            var k = qkv[0..., 0..., heads ..< (heads + kvHeads), 0...].transposed(0, 2, 1, 3)
            var v = qkv[0..., 0..., (heads + kvHeads)..., 0...].transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(
                q, dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1,
                offset: start)
            k = MLXFast.RoPE(
                k, dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1,
                offset: start)
            if let cachedK = stream.keys[i], let cachedV = stream.values[i] {
                k = concatenated([cachedK, k], axis: 2)
                v = concatenated([cachedV, v], axis: 2)
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(),
                mask: mask)
            let keep = min(k.dim(2), window - 1)
            stream.keys[i] = k[0..., 0..., (k.dim(2) - keep)..., 0...]
            stream.values[i] = v[0..., 0..., (v.dim(2) - keep)..., 0...]

            let attention = layer.output(
                attended.transposed(0, 2, 1, 3).reshaped(1, frames, heads * headDim))
            x = x + layer.attentionScale * attention
            let m = MLXFast.rmsNorm(x, weight: layer.postNorm, eps: config.rmsNormEps)
            let (gate, up) = layer.gateUp(m).split(axis: -1)
            x = x + layer.mlpScale * layer.down(silu(gate) * up)
        }
        x = MLXFast.rmsNorm(x, weight: w.finalNorm, eps: config.rmsNormEps)
        return w.outputProjection(x)
    }

    /// The attention mask for `queries` new frames after `cached` carried
    /// ones: causal, and no key more than `window - 1` frames back.
    static func slidingWindowMask(queries: Int, cached: Int, window: Int)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        let keys = cached + queries
        // One query that sees every carried key needs no mask.
        if queries == 1, keys <= window { return .none }
        let q = MLXArray(Int32(cached) ..< Int32(keys))[0..., .newAxis]
        let k = MLXArray(Int32(0) ..< Int32(keys))[.newAxis]
        return .array((q .>= k) .&& (q .< k + window))
    }

    // MARK: - Weights

    struct Linear {
        let weight: MLXArray  // [out, in]
        let bias: MLXArray?

        var arrays: [MLXArray] { [weight] + [bias].compactMap { $0 } }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            if let bias { return addMM(bias, x, weight.T) }
            return matmul(x, weight.T)
        }
    }

    struct Conv {
        let weight: MLXArray  // [out, kernel, in / groups]
        let bias: MLXArray?
        let kernel: Int
        let dilation: Int
        let groups: Int

        var arrays: [MLXArray] { [weight] + [bias].compactMap { $0 } }
    }

    /// A transposed conv as its polyphase conv: `[stride·out, taps, in]`,
    /// output channel `j·out + c` being phase `j` of channel `c`.
    struct Upsample {
        let conv: Conv
        let stride: Int
    }

    /// SnakeBeta, x + sin²(αx)/β, with α and 1/(β + 1e-9) precomputed.
    struct Snake {
        let alpha: MLXArray
        let inverseBeta: MLXArray

        var arrays: [MLXArray] { [alpha, inverseBeta] }
    }

    struct ConvNeXt {
        let depthwise: Conv
        let normWeight: MLXArray
        let normBias: MLXArray
        let pointwise1: Linear
        let pointwise2: Linear
        let gamma: MLXArray
    }

    struct TransformerLayer {
        let inputNorm: MLXArray
        let qkv: Linear
        let output: Linear
        let attentionScale: MLXArray
        let postNorm: MLXArray
        let gateUp: Linear
        let down: Linear
        let mlpScale: MLXArray
    }

    struct Upsampler {
        let upsample: Upsample
        let block: ConvNeXt
    }

    struct ResidualUnit {
        let snake1: Snake
        let conv1: Conv
        let snake2: Snake
        let conv2: Conv
    }

    struct Block {
        let snake: Snake
        let upsample: Upsample
        let units: [ResidualUnit]
    }

    /// Codes to latent: the codebooks, the pre-conv and the transformer.
    struct FrontEnd {
        /// All codebooks through their output projections, stacked:
        /// `[numQuantizers × codebookSize, codebookDim]`.
        let codebooks: MLXArray
        let preConv: Conv
        let inputProjection: Linear
        let layers: [TransformerLayer]
        let finalNorm: MLXArray
        let outputProjection: Linear

        var arrays: [MLXArray] {
            var all = [codebooks, finalNorm] + preConv.arrays + inputProjection.arrays
                + outputProjection.arrays
            for l in layers {
                all += [l.inputNorm, l.attentionScale, l.postNorm, l.mlpScale]
                all += l.qkv.arrays + l.output.arrays + l.gateUp.arrays + l.down.arrays
            }
            return all
        }
    }

    /// Latent to samples.
    struct ConvStack {
        let upsamplers: [Upsampler]
        let initialConv: Conv
        let blocks: [Block]
        let finalSnake: Snake
        let finalConv: Conv

        var arrays: [MLXArray] {
            var all = initialConv.arrays + finalSnake.arrays + finalConv.arrays
            for u in upsamplers {
                all += u.upsample.conv.arrays + u.block.depthwise.arrays
                all += u.block.pointwise1.arrays + u.block.pointwise2.arrays
                all += [u.block.normWeight, u.block.normBias, u.block.gamma]
            }
            for b in blocks {
                all += b.snake.arrays + b.upsample.conv.arrays
                for unit in b.units {
                    all += unit.snake1.arrays + unit.conv1.arrays + unit.snake2.arrays
                        + unit.conv2.arrays
                }
            }
            return all
        }
    }

    // MARK: - Loading

    /// Loads the decoder from a checkpoint's `speech_tokenizer` directory
    /// (its `config.json` and safetensors). A released conv stack is read
    /// back from there.
    package convenience init(directory: URL, dtype: DType = .float16) throws {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)
        let decoderConfig = config.decoderConfig ?? .defaults
        try self.init(
            config: decoderConfig, samplesPerFrame: config.decodeUpsampleRate,
            tensors: Self.readTensors(directory), dtype: dtype,
            reloadStack: {
                let reader = WeightReader(tensors: try Self.readTensors(directory), dtype: dtype)
                let stack = try Self.readConvStack(config: decoderConfig, reader: reader)
                eval(stack.arrays)
                return stack
            })
    }

    package convenience init(
        config: Qwen3TTSTokenizerDecoderConfig, samplesPerFrame: Int,
        tensors: [String: MLXArray], dtype: DType = .float16
    ) throws {
        try self.init(
            config: config, samplesPerFrame: samplesPerFrame, tensors: tensors, dtype: dtype,
            reloadStack: nil)
    }

    init(
        config: Qwen3TTSTokenizerDecoderConfig, samplesPerFrame: Int,
        tensors: [String: MLXArray], dtype: DType,
        reloadStack: (@Sendable () throws -> ConvStack)?
    ) throws {
        self.config = config
        self.samplesPerFrame = samplesPerFrame
        self.dtype = dtype
        self.reloadStack = reloadStack
        let reader = WeightReader(tensors: tensors, dtype: dtype)
        self.front = try Self.readFrontEnd(config: config, reader: reader)
        self.loadedStack = try Self.readConvStack(config: config, reader: reader)
        eval(front.arrays + loadedStack!.arrays)
    }

    /// The decoder half of a speech tokenizer directory's tensors, keyed
    /// without their `decoder.` prefix. Lazy: nothing is read until used, and
    /// the encoder never is.
    private static func readTensors(_ directory: URL) throws -> [String: MLXArray] {
        var tensors: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            for (key, value) in try MLX.loadArrays(url: file) {
                let key = Self.stripPrefixes(key)
                if key.hasPrefix("decoder.") { tensors[String(key.dropFirst(8))] = value }
            }
        }
        return tensors
    }

    private static func stripPrefixes(_ key: String) -> String {
        var key = key
        for prefix in ["speech_tokenizer.", "decoder_model."] where key.hasPrefix(prefix) {
            key = String(key.dropFirst(prefix.count))
        }
        return key
    }

    private static func readFrontEnd(
        config c: Qwen3TTSTokenizerDecoderConfig, reader r: WeightReader
    ) throws -> FrontEnd {
        // Codebooks: embedding = sum / max(usage, 1e-5), then the 1×1 output
        // projection of its group (the first codebook has its own).
        let half = c.codebookDim / 2
        var projections: [String: MLXArray] = [:]
        for group in ["rvq_first", "rvq_rest"] {
            projections[group] = try r.raw("quantizer.\(group).output_proj.weight")
                .asType(.float32).reshaped(c.codebookDim, half)
        }
        var tables: [MLXArray] = []
        for q in 0 ..< c.numQuantizers {
            let group = q < c.numSemanticQuantizers ? "rvq_first" : "rvq_rest"
            let index = q < c.numSemanticQuantizers ? q : q - c.numSemanticQuantizers
            let base = "quantizer.\(group).vq.layers.\(index)._codebook"
            let sum = try r.raw("\(base).embedding_sum").asType(.float32)
            let usage = try r.raw("\(base).cluster_usage").asType(.float32)
            let embedding = sum / maximum(usage, 1e-5).reshaped(-1, 1)
            let table = matmul(embedding, projections[group]!.T).asType(r.dtype)
            eval(table)
            tables.append(table)
        }
        let codebooks = concatenated(tables, axis: 0)
        eval(codebooks)
        tables.removeAll()

        let preConv = try r.conv(
            "pre_conv.conv", in: c.codebookDim, out: c.latentDim, kernel: 3)

        let hidden = c.hiddenSize
        let attentionWidth = c.numAttentionHeads * c.headDim
        let kvWidth = c.numKeyValueHeads * c.headDim
        var layers: [TransformerLayer] = []
        for i in 0 ..< c.numHiddenLayers {
            let p = "pre_transformer.layers.\(i)"
            let qkv = try [
                r.matrix("\(p).self_attn.q_proj.weight", out: attentionWidth, in: hidden),
                r.matrix("\(p).self_attn.k_proj.weight", out: kvWidth, in: hidden),
                r.matrix("\(p).self_attn.v_proj.weight", out: kvWidth, in: hidden),
            ]
            let gateUp = try [
                r.matrix("\(p).mlp.gate_proj.weight", out: c.intermediateSize, in: hidden),
                r.matrix("\(p).mlp.up_proj.weight", out: c.intermediateSize, in: hidden),
            ]
            layers.append(
                TransformerLayer(
                    inputNorm: try r.vector("\(p).input_layernorm.weight", hidden),
                    qkv: Linear(
                        weight: contiguous(concatenated(qkv, axis: 0)), bias: nil),
                    output: Linear(
                        weight: try r.matrix(
                            "\(p).self_attn.o_proj.weight", out: hidden, in: attentionWidth),
                        bias: nil),
                    attentionScale: try r.vector("\(p).self_attn_layer_scale.scale", hidden),
                    postNorm: try r.vector("\(p).post_attention_layernorm.weight", hidden),
                    gateUp: Linear(
                        weight: contiguous(concatenated(gateUp, axis: 0)), bias: nil),
                    down: Linear(
                        weight: try r.matrix(
                            "\(p).mlp.down_proj.weight", out: hidden, in: c.intermediateSize),
                        bias: nil),
                    mlpScale: try r.vector("\(p).mlp_layer_scale.scale", hidden)))
        }

        return FrontEnd(
            codebooks: codebooks,
            preConv: preConv,
            inputProjection: try r.linear(
                "pre_transformer.input_proj", out: hidden, in: c.latentDim),
            layers: layers,
            finalNorm: try r.vector("pre_transformer.norm.weight", hidden),
            outputProjection: try r.linear(
                "pre_transformer.output_proj", out: c.latentDim, in: hidden))
    }

    static func readConvStack(
        config c: Qwen3TTSTokenizerDecoderConfig, reader r: WeightReader
    ) throws -> ConvStack {
        var upsamplers: [Upsampler] = []
        for (i, factor) in c.upsamplingRatios.enumerated() {
            let p = "upsample.\(i)"
            let dim = c.latentDim
            upsamplers.append(
                Upsampler(
                    upsample: try r.upsample(
                        "\(p).0.conv", in: dim, out: dim, kernel: factor, stride: factor),
                    block: ConvNeXt(
                        depthwise: try r.conv(
                            "\(p).1.dwconv.conv", in: dim, out: dim, kernel: 7, groups: dim),
                        normWeight: try r.vector("\(p).1.norm.weight", dim),
                        normBias: try r.vector("\(p).1.norm.bias", dim),
                        pointwise1: try r.linear("\(p).1.pwconv1", out: 4 * dim, in: dim),
                        pointwise2: try r.linear("\(p).1.pwconv2", out: dim, in: 4 * dim),
                        gamma: try r.vector("\(p).1.gamma", dim))))
        }

        let initialConv = try r.conv(
            "decoder.0.conv", in: c.latentDim, out: c.decoderDim, kernel: 7)
        var blocks: [Block] = []
        for (i, rate) in c.upsampleRates.enumerated() {
            let p = "decoder.\(i + 1).block"
            let inDim = c.decoderDim >> i
            let outDim = c.decoderDim >> (i + 1)
            var units: [ResidualUnit] = []
            for (u, dilation) in [1, 3, 9].enumerated() {
                let q = "\(p).\(u + 2)"
                units.append(
                    ResidualUnit(
                        snake1: try r.snake("\(q).act1", outDim),
                        conv1: try r.conv(
                            "\(q).conv1.conv", in: outDim, out: outDim, kernel: 7,
                            dilation: dilation),
                        snake2: try r.snake("\(q).act2", outDim),
                        conv2: try r.conv("\(q).conv2.conv", in: outDim, out: outDim, kernel: 1)))
            }
            blocks.append(
                Block(
                    snake: try r.snake("\(p).0", inDim),
                    upsample: try r.upsample(
                        "\(p).1.conv", in: inDim, out: outDim, kernel: 2 * rate, stride: rate),
                    units: units))
        }
        let outputDim = c.decoderDim >> c.upsampleRates.count
        let finalIndex = c.upsampleRates.count + 1
        return ConvStack(
            upsamplers: upsamplers,
            initialConv: initialConv,
            blocks: blocks,
            finalSnake: try r.snake("decoder.\(finalIndex)", outputDim),
            finalConv: try r.conv(
                "decoder.\(finalIndex + 1).conv", in: outputDim, out: 1, kernel: 7))
    }
}

/// SnakeBeta as one fused kernel.
private let compiledSnake: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { x, alpha, inverseBeta in
        let s = sin(x * alpha)
        return x + inverseBeta * (s * s)
    }
}()

/// Reads the checkpoint's tensors into MLX's layout: checks every shape,
/// converts PyTorch conv layouts, casts to the decoder's dtype and makes each
/// weight contiguous (a strided weight would be copied on every conv call).
/// Each tensor is handed over once and its conversion evaluated at once, so
/// loading holds one fp32 original at a time, not all of them.
final class WeightReader {
    private var tensors: [String: MLXArray]
    let dtype: DType

    init(tensors: [String: MLXArray], dtype: DType) {
        self.tensors = tensors
        self.dtype = dtype
    }

    func raw(_ key: String) throws -> MLXArray {
        guard let t = tensors.removeValue(forKey: key) else {
            throw AudioGenerationError.modelNotInitialized(
                "The speech tokenizer checkpoint has no decoder weight \(key).")
        }
        return t
    }

    private func finish(_ t: MLXArray) -> MLXArray {
        let converted = contiguous(t.asType(dtype))
        eval(converted)
        return converted
    }

    private func mismatch(_ key: String, _ t: MLXArray, _ expected: String) -> Error {
        AudioGenerationError.modelNotInitialized(
            "Speech tokenizer weight \(key) has shape \(t.shape), expected \(expected).")
    }

    func vector(_ key: String, _ count: Int) throws -> MLXArray {
        let t = try raw(key)
        guard t.shape == [count] else { throw mismatch(key, t, "[\(count)]") }
        return finish(t)
    }

    func matrix(_ key: String, out: Int, in inputs: Int) throws -> MLXArray {
        let t = try raw(key)
        // A 1×1 conv stored as [out, in, 1] is a matrix too.
        let m = t.ndim == 3 && t.dim(2) == 1 ? t.reshaped(t.dim(0), t.dim(1)) : t
        guard m.shape == [out, inputs] else { throw mismatch(key, t, "[\(out), \(inputs)]") }
        return finish(m)
    }

    func linear(_ prefix: String, out: Int, in inputs: Int) throws -> Qwen3TTSCodecDecoder.Linear {
        .init(
            weight: try matrix("\(prefix).weight", out: out, in: inputs),
            bias: try vector("\(prefix).bias", out))
    }

    /// PyTorch stores a conv as `[out, in/groups, kernel]`; MLX wants
    /// `[out, kernel, in/groups]`. Either is accepted.
    func conv(
        _ prefix: String, in inputs: Int, out: Int, kernel: Int, dilation: Int = 1,
        groups: Int = 1
    ) throws -> Qwen3TTSCodecDecoder.Conv {
        let key = "\(prefix).weight"
        let t = try raw(key)
        let perGroup = inputs / groups
        let weight: MLXArray
        if t.shape == [out, perGroup, kernel] {
            weight = t.transposed(0, 2, 1)
        } else if t.shape == [out, kernel, perGroup] {
            weight = t
        } else {
            throw mismatch(key, t, "[\(out), \(perGroup), \(kernel)]")
        }
        return .init(
            weight: finish(weight), bias: try vector("\(prefix).bias", out), kernel: kernel,
            dilation: dilation, groups: groups)
    }

    /// A transposed conv (PyTorch `[in, out, kernel]`, or MLX's
    /// `[out, kernel, in]`) with kernel `stride` or `2·stride`, as its
    /// polyphase conv. Tap 1 applies kernel column `j` to x[t]; tap 0 applies
    /// column `j + stride` to x[t-1].
    func upsample(
        _ prefix: String, in inputs: Int, out: Int, kernel: Int, stride: Int
    ) throws -> Qwen3TTSCodecDecoder.Upsample {
        let key = "\(prefix).weight"
        let t = try raw(key)
        let w: MLXArray  // [out, kernel, in]
        if t.shape == [inputs, out, kernel] {
            w = t.transposed(1, 2, 0)
        } else if t.shape == [out, kernel, inputs] {
            w = t
        } else {
            throw mismatch(key, t, "[\(inputs), \(out), \(kernel)]")
        }
        guard kernel == stride || kernel == 2 * stride else {
            throw AudioGenerationError.modelNotInitialized(
                "Speech tokenizer transposed conv \(prefix) has kernel \(kernel) for stride \(stride).")
        }
        // Phase j of output channel c is output channel j·out + c.
        func phases(_ columns: MLXArray) -> MLXArray {  // [out, stride, in]
            columns.transposed(1, 0, 2).reshaped(stride * out, inputs)
        }
        let current = phases(w[0..., ..<stride, 0...])
        let weight =
            kernel == stride
            ? current.reshaped(stride * out, 1, inputs)
            : stacked([phases(w[0..., stride..., 0...]), current], axis: 1)
        let bias = tiled(try raw("\(prefix).bias").reshaped(1, out), repetitions: [stride, 1])
            .reshaped(stride * out)
        return .init(
            conv: .init(
                weight: finish(weight), bias: finish(bias), kernel: kernel / stride,
                dilation: 1, groups: 1),
            stride: stride)
    }

    func snake(_ prefix: String, _ channels: Int) throws -> Qwen3TTSCodecDecoder.Snake {
        let alpha = try raw("\(prefix).alpha").asType(.float32)
        let beta = try raw("\(prefix).beta").asType(.float32)
        guard alpha.shape == [channels], beta.shape == [channels] else {
            throw mismatch("\(prefix).alpha", alpha, "[\(channels)]")
        }
        return .init(alpha: finish(exp(alpha)), inverseBeta: finish(1 / (exp(beta) + 1e-9)))
    }
}
