// spike(337) smoke harness — NOT part of the port.
// Validates the ported features at runtime against local VoiceDesign bf16 weights:
//   1. seed determinism (same seed => identical samples, different seed => different)
//   2. tokenizeForAlignment (non-empty, monotonic offsets)
//   3. voice anchor mechanical path (build from segment 1, generate segment 2 anchored)

import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

let modelDir = URL(fileURLWithPath:
    "/Users/owl/Library/Containers/app.tesseract.agent/Data/Library/Application Support/models/mlx-community_Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
)

func fail(_ msg: String) -> Never {
    fputs("SMOKE FAIL: \(msg)\n", stderr)
    exit(1)
}

do {
    let model = try await Qwen3TTSModel.fromModelDirectory(modelDir)
    print("model loaded")

    let voice = "A calm, warm female narrator with a clear, steady tone."
    let text = "Hello there, this is a short determinism test."
    let params = GenerateParameters(
        maxTokens: 200, temperature: 0.9, topP: 1.0, repetitionPenalty: 1.05
    )

    // 2. tokenizeForAlignment
    let offsets = model.tokenizeForAlignment(text: text)
    print("alignment offsets (\(offsets.count)):", offsets)
    if offsets.isEmpty || offsets != offsets.sorted() {
        fail("tokenizeForAlignment offsets empty or non-monotonic")
    }

    // 1. seed determinism (non-streaming path)
    func gen(seed: UInt64) async throws -> [Float] {
        model.seed = seed
        let audio = try await model.generate(
            text: text, voice: voice, refAudio: nil, refText: nil,
            language: nil, generationParameters: params
        )
        return audio.asArray(Float.self)
    }
    let a = try await gen(seed: 42)
    let b = try await gen(seed: 42)
    let c = try await gen(seed: 43)
    print("len(a)=\(a.count) len(b)=\(b.count) len(c)=\(c.count)")
    let sameSeedIdentical = (a == b)
    let diffSeedDifferent = !(a.count == c.count && a == c)
    print("same seed identical:", sameSeedIdentical)
    print("different seed different:", diffSeedDifferent)
    if !sameSeedIdentical { fail("same seed produced different audio") }
    if !diffSeedDifferent { fail("different seed produced identical audio") }

    // 3. voice anchor mechanical path (streaming)
    func genStream(_ text: String, useVoiceAnchor: Bool) async throws -> [Float] {
        model.seed = 42
        var samples = [Float]()
        let stream = model.generateStream(
            text: text, voice: voice, refAudio: nil, refText: nil,
            language: nil, generationParameters: params,
            streamingInterval: 2.0, useVoiceAnchor: useVoiceAnchor
        )
        for try await event in stream {
            if case .audio(let chunk) = event {
                samples.append(contentsOf: chunk.asArray(Float.self))
            }
        }
        return samples
    }

    let seg1 = try await genStream("First segment builds the anchor.", useVoiceAnchor: false)
    print("seg1 samples:", seg1.count)
    if seg1.count < 1000 { fail("segment 1 produced too little audio") }
    guard let codes = model.lastGeneratedCodes, !codes.isEmpty else {
        fail("lastGeneratedCodes not captured after streaming generation")
    }
    print("lastGeneratedCodes steps:", codes.count)

    model.buildVoiceAnchor(referenceCount: 48, instruct: voice, language: nil)
    let seg2 = try await genStream("Second segment continues in the same voice.", useVoiceAnchor: true)
    print("seg2 (anchored) samples:", seg2.count)
    if seg2.count < 1000 { fail("anchored segment produced too little audio") }

    model.clearVoiceAnchor()
    print("SMOKE OK")
} catch {
    fail("\(error)")
}
