import Testing
import MLXLMCommon

@testable import Tesseract_Agent

struct AgentGenerateParametersTests {

    @MainActor
    @Test func defaultPrefillStepSizeIs1024() {
        #expect(AgentGenerateParameters.default.prefillStepSize == 1024)
        #expect(AgentGenerateParameters.qwen35.prefillStepSize == 1024)
        #expect(AgentGenerateParameters.default.triAttention.enabled == false)
        #expect(AgentGenerateParameters.default.triAttention.budgetTokens == 12_000)
        #expect(AgentGenerateParameters.default.triAttention.implementationVersion == .v1)
        #expect(AgentGenerateParameters.default.triAttention.calibrationArtifactIdentity == nil)
    }

    @MainActor
    @Test func qwen36IdMapsToQwen36ThinkingPreset() {
        // Qwen3.6 must not inherit qwen3.5's presencePenalty = 1.5 — inside
        // `<think>` the penalty drives paraphrased loops with changing
        // identifiers, which downstream signals don't catch as cleanly.
        let params = AgentGenerateParameters.forModel("qwen3.6-35b-a3b-ud")
        #expect(params.temperature == AgentGenerateParameters.qwen36Thinking.temperature)
        #expect(params.topP == AgentGenerateParameters.qwen36Thinking.topP)
        #expect(params.topK == AgentGenerateParameters.qwen36Thinking.topK)
        #expect(params.presencePenalty == nil)
        #expect(params.thinkingSafeguard.enabled == true)
    }

    @MainActor
    @Test func qwen36ThinkingPresetHasNoPresencePenalty() {
        #expect(AgentGenerateParameters.qwen36Thinking.temperature == 0.6)
        #expect(AgentGenerateParameters.qwen36Thinking.topP == 0.95)
        #expect(AgentGenerateParameters.qwen36Thinking.topK == 20)
        #expect(AgentGenerateParameters.qwen36Thinking.presencePenalty == nil)
        #expect(AgentGenerateParameters.qwen36Thinking.repetitionPenalty == nil)
    }

    // MARK: - SamplingPreset

    @MainActor
    @Test func automaticReturnsBaseUnchanged() {
        let base = AgentGenerateParameters.qwen35
        let out = SamplingPreset.automatic.apply(to: base)
        #expect(out.temperature == base.temperature)
        #expect(out.topP == base.topP)
        #expect(out.topK == base.topK)
        #expect(out.minP == base.minP)
        #expect(out.presencePenalty == base.presencePenalty)
        #expect(out.repetitionPenalty == base.repetitionPenalty)
    }

    @MainActor
    @Test func qwenThinkingGeneralMatchesSpec() {
        let out = SamplingPreset.qwenThinkingGeneral.apply(to: .default)
        #expect(out.temperature == 1.0)
        #expect(out.topP == 0.95)
        #expect(out.topK == 20)
        #expect(out.minP == 0.0)
        #expect(out.presencePenalty == 1.5)
        #expect(out.repetitionPenalty == nil)
    }

    @MainActor
    @Test func qwenThinkingCodingMatchesSpec() {
        let out = SamplingPreset.qwenThinkingCoding.apply(to: .default)
        #expect(out.temperature == 0.6)
        #expect(out.topP == 0.95)
        #expect(out.topK == 20)
        #expect(out.minP == 0.0)
        #expect(out.presencePenalty == 0.0)
        #expect(out.repetitionPenalty == nil)
    }

    @MainActor
    @Test func qwenInstructGeneralMatchesSpec() {
        let out = SamplingPreset.qwenInstructGeneral.apply(to: .default)
        #expect(out.temperature == 0.7)
        #expect(out.topP == 0.8)
        #expect(out.topK == 20)
        #expect(out.minP == 0.0)
        #expect(out.presencePenalty == 1.5)
        #expect(out.repetitionPenalty == nil)
    }

    @MainActor
    @Test func qwenInstructReasoningMatchesSpec() {
        let out = SamplingPreset.qwenInstructReasoning.apply(to: .default)
        #expect(out.temperature == 1.0)
        #expect(out.topP == 0.95)
        #expect(out.topK == 20)
        #expect(out.minP == 0.0)
        #expect(out.presencePenalty == 1.5)
        #expect(out.repetitionPenalty == nil)
    }

    @MainActor
    @Test func presetPreservesNonSamplingFields() {
        var base = AgentGenerateParameters.default
        base.maxTokens = 131_072
        base.kvBits = 4
        base.kvGroupSize = 128
        base.prefillStepSize = 2048
        base.thinkingSafeguard.enabled = false
        base.thinkingSafeguard.maxThinkingChars = 12_345
        base.frequencyPenalty = 0.7
        base.triAttention = .v1Disabled

        let out = SamplingPreset.qwenThinkingCoding.apply(to: base)

        #expect(out.maxTokens == 131_072)
        #expect(out.kvBits == 4)
        #expect(out.kvGroupSize == 128)
        #expect(out.prefillStepSize == 2048)
        #expect(out.thinkingSafeguard.enabled == false)
        #expect(out.thinkingSafeguard.maxThinkingChars == 12_345)
        #expect(out.frequencyPenalty == 0.7)
    }

    @MainActor
    @Test func allNonAutomaticPresetsDisableRepetitionPenalty() {
        let presets: [SamplingPreset] = [
            .qwenThinkingGeneral, .qwenThinkingCoding,
            .qwenInstructGeneral, .qwenInstructReasoning,
        ]
        for preset in presets {
            var base = AgentGenerateParameters.default
            base.repetitionPenalty = 1.1  // caller set something non-neutral
            let out = preset.apply(to: base)
            #expect(out.repetitionPenalty == nil, "preset \(preset) should clear repetitionPenalty")
        }
    }
}
