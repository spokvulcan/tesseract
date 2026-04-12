import Testing

@testable import Tesseract_Agent

struct AgentGenerateParametersTests {

    @MainActor
    @Test func defaultPrefillStepSizeIs1024() {
        #expect(AgentGenerateParameters.default.prefillStepSize == 1024)
        #expect(AgentGenerateParameters.qwen35.prefillStepSize == 1024)
    }
}
