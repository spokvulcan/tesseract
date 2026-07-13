// spike(337) smoke harness → v2 pinning suite — NOT part of the port.
// Validates the ported features at runtime against local VoiceDesign bf16 weights:
//   1. seed determinism (same seed => identical samples, different seed => different)
//   2. tokenizeForAlignment (non-empty, monotonic offsets)
//   3. voice anchor mechanical path (build from segment 1, generate segment 2 anchored)
//      — this is ALSO the warm-cache mask-fix pin (TESSERACT-PATCHES.md #6):
//      anchored generation restores KV then runs a multi-token forward; without
//      createCausalMask(n:offset:) it dies on a broadcast error. If this test
//      starts crashing after a re-vendor, the mask fix regressed.
//   4. cache-limit tolerance (ADR-0039): generation is bit-identical under the
//      LLM stack's process-global Memory.cacheLimit and across Memory.clearCache()
//   5. anchor rebuild from code frames (TESSERACT-PATCHES.md #10): an anchor
//      rebuilt from exported [[Int32]] frames conditions generation bit-identically
//      to the anchor built in place — the PinnedVoice (ADR-0038) guarantee.

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

    // 4. cache-limit tolerance (ADR-0039): the LLM stack owns the process-global
    // cache limit (2 GB in the app); TTS output must be bit-identical under it,
    // and across an explicit buffer-pool clear between bursts.
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 2 * 1024 * 1024 * 1024
    Memory.clearCache()
    let underLimit = try await gen(seed: 42)
    Memory.cacheLimit = previousLimit
    Memory.clearCache()
    let afterClear = try await gen(seed: 42)
    if underLimit != a { fail("generation differs under LLM cache limit") }
    if afterClear != a { fail("generation differs after Memory.clearCache()") }
    print("cache-limit + clearCache bit-identical: true")

    // 3. voice anchor mechanical path (streaming) — the mask-fix pin (#6).
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

    // 5. anchor rebuild equivalence (#10): export frames BEFORE building the
    // in-place anchor, then compare anchored generations from both paths.
    let exportedFrames = Array(model.lastGeneratedCodeFrames.prefix(48))
    if exportedFrames.isEmpty { fail("lastGeneratedCodeFrames empty") }

    model.buildVoiceAnchor(referenceCount: 48, instruct: voice, language: nil)
    let seg2 = try await genStream("Second segment continues in the same voice.", useVoiceAnchor: true)
    print("seg2 (anchored) samples:", seg2.count)
    if seg2.count < 1000 { fail("anchored segment produced too little audio") }

    model.clearVoiceAnchor()
    model.buildVoiceAnchor(
        fromCodeFrames: exportedFrames, referenceCount: 48, instruct: voice, language: nil
    )
    let seg2Rebuilt = try await genStream(
        "Second segment continues in the same voice.", useVoiceAnchor: true
    )
    print("seg2 (rebuilt anchor) samples:", seg2Rebuilt.count)
    if seg2Rebuilt != seg2 {
        fail("anchor rebuilt from code frames conditions generation differently")
    }
    print("anchor rebuild bit-identical: true")

    model.clearVoiceAnchor()
    print("SMOKE OK")
} catch {
    fail("\(error)")
}
