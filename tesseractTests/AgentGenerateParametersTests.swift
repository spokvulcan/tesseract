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
}
