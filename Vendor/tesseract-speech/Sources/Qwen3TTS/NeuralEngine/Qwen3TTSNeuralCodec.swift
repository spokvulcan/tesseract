import CoreML
import Foundation
@preconcurrency import MLX

/// The codec decoder's conv stack on the Apple Neural Engine.
///
/// The conv stack (the two ×2 upsamplers, the initial conv, the four
/// upsampling blocks and the output conv) is 98% of the decoder's work and
/// suits the Neural Engine: convolutions, fixed shapes, fp16 headroom (its
/// largest activation is a few hundred). There it runs beside the GPU, which
/// keeps the talker and the code predictor, instead of queueing behind them.
/// The front end (codebooks, pre-conv, the windowed transformer, about 2%)
/// stays in MLX and hands over a latent per chunk.
///
/// The model is built on this machine from the checkpoint's own weights, so
/// nothing extra is downloaded: an fp16 ML program in the Neural Engine's
/// `[1, C, 1, W]` layout, one fixed chunk of `frames` frames per call, every
/// causal conv's context passed in and out as state (20 small tensors). It
/// is compiled once and cached; Core ML specializes it for the Neural Engine
/// on first load and caches that too. Loading refuses a plan that would put
/// any op off the Neural Engine, and then the MLX conv stack does the work.
package final class Qwen3TTSNeuralCodec: @unchecked Sendable {
    /// Frames per call.
    package let frames: Int
    package let samplesPerFrame: Int
    private let latentDim: Int
    private let model: MLModel
    private let states: [StateSpec]

    /// Bump when the graph changes: part of the cache key.
    static let graphVersion = 1

    struct StateSpec {
        let input: String
        let output: String
        let shape: [Int]
    }

    private init(
        frames: Int, samplesPerFrame: Int, latentDim: Int, model: MLModel, states: [StateSpec]
    ) {
        self.frames = frames
        self.samplesPerFrame = samplesPerFrame
        self.latentDim = latentDim
        self.model = model
        self.states = states
    }

    // MARK: - Running

    /// One stream's conv contexts, carried from chunk to chunk. Its chunks
    /// decode one at a time, in order.
    package final class Stream: @unchecked Sendable {
        fileprivate var state: [String: MLMultiArray]

        fileprivate init(state: [String: MLMultiArray]) {
            self.state = state
        }
    }

    /// A new stream: every conv context zeros.
    package func makeStream() throws -> Stream {
        var state: [String: MLMultiArray] = [:]
        for spec in states {
            let array = try MLMultiArray(
                shape: spec.shape.map { NSNumber(value: $0) }, dataType: .float16)
            array.withUnsafeMutableBytes { raw, _ in
                _ = raw.initializeMemory(as: UInt8.self, repeating: 0)
            }
            state[spec.input] = array
        }
        return Stream(state: state)
    }

    /// Decodes `stream`'s next chunk: `latent` is up to `frames` frames of
    /// `latentDim` fp16 values, frame-major (the MLX latent's own layout).
    /// Returns their samples. A shorter chunk, a stream's last, is padded
    /// with zeros: every layer is causal, so padding changes only the state,
    /// which nothing reads after it.
    package func decode(latent: [Float16], stream: Stream) throws -> [Float] {
        let valid = latent.count / latentDim
        precondition(latent.count == valid * latentDim && valid > 0 && valid <= frames)
        let input = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim), 1, NSNumber(value: frames)], dataType: .float16)
        // [T, C] in, [1, C, 1, T] (strided) out.
        input.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            let channelStride = strides[1]
            let timeStride = strides[3]
            for t in 0 ..< frames {
                for c in 0 ..< latentDim {
                    buffer[c * channelStride + t * timeStride] =
                        t < valid ? latent[t * latentDim + c] : 0
                }
            }
        }
        var features: [String: MLFeatureValue] = ["latent": MLFeatureValue(multiArray: input)]
        for spec in states {
            features[spec.input] = MLFeatureValue(multiArray: stream.state[spec.input]!)
        }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        for spec in states {
            guard let next = output.featureValue(for: spec.output)?.multiArrayValue else {
                throw AudioGenerationError.modelNotInitialized("The Neural Engine codec lost its state.")
            }
            stream.state[spec.input] = next
        }
        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw AudioGenerationError.modelNotInitialized("The Neural Engine codec returned no audio.")
        }
        let count = valid * samplesPerFrame
        var samples = [Float](repeating: 0, count: count)
        audio.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
            let stride = audio.strides[3].intValue
            for i in 0 ..< count { samples[i] = Float(buffer[i * stride]) }
        }
        return samples
    }

    /// A whole latent (`frames × latentDim` values each chunk, frame-major),
    /// decoded on a new stream.
    package func decodeAll(latent: [Float16]) throws -> [Float] {
        let stream = try makeStream()
        let chunk = frames * latentDim
        var samples: [Float] = []
        for start in stride(from: 0, to: latent.count, by: chunk) {
            samples += try decode(
                latent: Array(latent[start ..< min(start + chunk, latent.count)]), stream: stream)
        }
        return samples
    }

    /// The signal-to-noise ratio of `actual` against `expected`, in dB.
    package static func snr(_ actual: [Float], _ expected: [Float]) -> Double {
        var signal = 0.0
        var noise = 0.0
        for (a, e) in zip(actual, expected) {
            signal += Double(e * e)
            noise += Double((a - e) * (a - e))
        }
        return 10 * log10(signal / max(noise, 1e-20))
    }

    // MARK: - Loading

    /// The codec for `decoder`'s weights: loaded from `cacheDirectory` when
    /// built before, else built, compiled and cached there. Throws when the
    /// compute plan would put any op off the Neural Engine.
    package static func load(
        decoder: Qwen3TTSCodecDecoder, frames: Int, cacheDirectory: URL, sourceKey: String,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine, requireNeuralEngine: Bool = true
    ) async throws -> (codec: Qwen3TTSNeuralCodec, placement: NeuralPlacement) {
        let config = decoder.config
        let states = stateSpecs(config: config)
        let compiled = try await NeuralModelStore.compiled(
            key: cacheKey(config: config, frames: frames, sourceKey: sourceKey), in: cacheDirectory,
            metadata: [
                "com.tesseract.qwen3tts.codec": "conv stack, \(frames) frames",
                "com.tesseract.qwen3tts.graphVersion": "\(graphVersion)",
            ]
        ) { try buildGraph($0, decoder: decoder, frames: frames, states: states) }
        let (model, placement) = try await NeuralModelStore.load(
            compiled, computeUnits: computeUnits, requireNeuralEngine: requireNeuralEngine,
            what: "The codec")
        let codec = Qwen3TTSNeuralCodec(
            frames: frames, samplesPerFrame: decoder.samplesPerFrame, latentDim: config.latentDim,
            model: model, states: states)
        return (codec, placement)
    }

    /// The graph version, the chunk size, the decoder's configuration and
    /// the checkpoint file's identity (`sourceKey`, e.g. size and date).
    static func cacheKey(
        config: Qwen3TTSTokenizerDecoderConfig, frames: Int, sourceKey: String
    ) -> String {
        NeuralModelStore.key(
            "qwen3tts-codec",
            "v\(graphVersion) f\(frames) \(config.latentDim) \(config.decoderDim) "
                + "\(config.upsampleRates) \(config.upsamplingRatios) \(sourceKey)")
    }

    // MARK: - The graph

    /// The state each causal conv carries, in graph order.
    static func stateSpecs(config c: Qwen3TTSTokenizerDecoderConfig) -> [StateSpec] {
        var specs: [StateSpec] = []
        func add(_ channels: Int, _ width: Int) {
            let i = specs.count
            specs.append(.init(input: "state_\(i)", output: "next_state_\(i)", shape: [1, channels, 1, width]))
        }
        for _ in c.upsamplingRatios { add(c.latentDim, 6) }  // ConvNeXt depthwise, k7
        add(c.latentDim, 6)  // initial conv, k7
        for i in 0 ..< c.upsampleRates.count {
            let inDim = c.decoderDim >> i
            let outDim = c.decoderDim >> (i + 1)
            add(inDim, 1)  // polyphase upsampler: the last input frame
            for dilation in [1, 3, 9] { add(outDim, 6 * dilation) }
        }
        add(c.decoderDim >> c.upsampleRates.count, 6)  // output conv, k7
        return specs
    }

    /// Emits the conv stack into `f`, carrying `specs` (`stateSpecs`) as its
    /// states in order.
    static func buildGraph(
        _ f: MILFunctionBuilder, decoder: Qwen3TTSCodecDecoder, frames: Int, states specs: [StateSpec]
    ) throws {
        let c = decoder.config
        let w = try decoder.convStack
        var nextState = 0

        // Shared small constants.
        let valid = f.string("valid")
        let unitStrides = f.ints([1, 1])
        let noPad = f.ints([0, 0, 0, 0])
        let axis3 = f.int(3)
        let noInterleave = f.bool(false)
        var dilationConsts: [Int: MILVar] = [:]
        var groupConsts: [Int: MILVar] = [:]

        // On the CPU stream: building runs beside generation and never
        // touches the GPU.
        func halves(_ a: MLXArray) -> MLXArray {
            contiguous(a.asType(.float16, stream: .cpu), stream: .cpu)
        }
        func channelVector(_ a: MLXArray) -> MILVar {
            f.weight(halves(a), shape: [1, a.dim(0), 1, 1])
        }

        /// A conv with MLX weight `[out, k, in/groups]` over `x` `[1, C, 1, W]`.
        func conv(_ x: MILVar, _ conv: Qwen3TTSCodecDecoder.Conv) -> MILVar {
            let out = conv.weight.dim(0)
            let k = conv.weight.dim(1)
            let perGroup = conv.weight.dim(2)
            let weight = f.weight(
                halves(conv.weight.transposed(0, 2, 1)), shape: [out, perGroup, 1, k])
            let dilation = dilationConsts[conv.dilation] ?? f.ints([1, conv.dilation])
            dilationConsts[conv.dilation] = dilation
            let groups = groupConsts[conv.groups] ?? f.int(conv.groups)
            groupConsts[conv.groups] = groups
            var inputs: [(String, [MILVar])] = [
                ("x", [x]), ("weight", [weight]), ("strides", [unitStrides]),
                ("pad_type", [valid]), ("pad", [noPad]), ("dilations", [dilation]),
                ("groups", [groups]),
            ]
            if let bias = conv.bias {
                inputs.append(("bias", [f.weight(halves(bias), shape: [out])]))
            }
            let width = x.type.shape[3] - (k - 1) * conv.dilation
            return f.op("conv", inputs, .fp16([1, out, 1, width]))
        }

        /// A causal conv: this stream's context in front, then a valid conv.
        func causalConv(_ x: MILVar, _ c: Qwen3TTSCodecDecoder.Conv) -> MILVar {
            let needed = (c.kernel - 1) * c.dilation
            guard needed > 0 else { return conv(x, c) }
            let spec = specs[nextState]
            nextState += 1
            let channels = x.type.shape[1]
            let width = x.type.shape[3]
            precondition(spec.shape == [1, channels, 1, needed], "state \(spec.input)")
            let context = f.input(spec.input, .fp16(spec.shape))
            let joined = f.op(
                "concat", [("values", [context, x]), ("axis", [axis3]), ("interleave", [noInterleave])],
                .fp16([1, channels, 1, needed + width]))
            let carried = f.op(
                "slice_by_index",
                [("x", [joined]), ("begin", [f.ints([0, 0, 0, width])]),
                 ("end", [f.ints([1, channels, 1, needed + width])])],
                .fp16(spec.shape), name: spec.output)
            f.output(carried)
            return conv(joined, c)
        }

        func snake(_ x: MILVar, _ s: Qwen3TTSCodecDecoder.Snake) -> MILVar {
            let t = x.type
            let scaled = f.op("mul", [("x", [x]), ("y", [channelVector(s.alpha)])], t)
            let sine = f.op("sin", [("x", [scaled])], t)
            let squared = f.op("mul", [("x", [sine]), ("y", [sine])], t)
            let term = f.op("mul", [("x", [squared]), ("y", [channelVector(s.inverseBeta)])], t)
            return f.op("add", [("x", [x]), ("y", [term])], t)
        }

        /// Polyphase upsampling: the two-tap conv's `[1, s·C, 1, T]`, channel
        /// `j·C + c` being phase j, reordered to `[1, C, 1, T·s]`.
        func upsample(_ x: MILVar, _ u: Qwen3TTSCodecDecoder.Upsample) -> MILVar {
            let y = causalConv(x, u.conv)
            let s = u.stride
            let width = y.type.shape[3]
            let channels = y.type.shape[1] / s
            let phases = f.op(
                "reshape", [("x", [y]), ("shape", [f.ints([1, s, channels, width])])],
                .fp16([1, s, channels, width]))
            let ordered = f.op(
                "transpose", [("x", [phases]), ("perm", [f.ints([0, 2, 3, 1])])],
                .fp16([1, channels, width, s]))
            return f.op(
                "reshape", [("x", [ordered]), ("shape", [f.ints([1, channels, 1, width * s])])],
                .fp16([1, channels, 1, width * s]))
        }

        func pointwise(_ x: MILVar, _ l: Qwen3TTSCodecDecoder.Linear) -> MILVar {
            conv(
                x,
                .init(
                    weight: l.weight.reshaped(l.weight.dim(0), 1, l.weight.dim(1)), bias: l.bias,
                    kernel: 1, dilation: 1, groups: 1))
        }

        var x = f.input("latent", .fp16([1, c.latentDim, 1, frames]))
        for u in w.upsamplers {
            x = upsample(x, u.upsample)
            var h = causalConv(x, u.block.depthwise)
            h = f.op(
                "layer_norm",
                [("x", [h]), ("axes", [f.ints([1])]), ("gamma", [f.weight(halves(u.block.normWeight), shape: [c.latentDim])]),
                 ("beta", [f.weight(halves(u.block.normBias), shape: [c.latentDim])]), ("epsilon", [f.half(1e-6)])],
                h.type)
            h = pointwise(h, u.block.pointwise1)
            h = f.op("gelu", [("x", [h]), ("mode", [f.string("EXACT")])], h.type)
            h = pointwise(h, u.block.pointwise2)
            h = f.op("mul", [("x", [h]), ("y", [channelVector(u.block.gamma)])], h.type)
            x = f.op("add", [("x", [x]), ("y", [h])], x.type)
        }
        x = causalConv(x, w.initialConv)
        for block in w.blocks {
            x = snake(x, block.snake)
            x = upsample(x, block.upsample)
            for unit in block.units {
                var h = snake(x, unit.snake1)
                h = causalConv(h, unit.conv1)
                h = snake(h, unit.snake2)
                h = conv(h, unit.conv2)
                x = f.op("add", [("x", [x]), ("y", [h])], x.type)
            }
        }
        x = snake(x, w.finalSnake)
        x = causalConv(x, w.finalConv)
        let audio = f.op(
            "clip", [("x", [x]), ("alpha", [f.half(-1)]), ("beta", [f.half(1)])], x.type,
            name: "audio")
        f.output(audio)
        precondition(nextState == specs.count, "every state is wired")
    }
}
