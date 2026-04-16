import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct TriAttentionConfigPlumbingTests {

    private func clearTriAttentionDefaults() {
        UserDefaults.standard.removeObject(forKey: "triattentionEnabled")
    }

    @Test
    func settingsDefaultToDisabledV1Configuration() {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        #expect(settings.triattentionEnabled == false)

        let config = settings.makeTriAttentionConfig()
        #expect(config.enabled == false)
        #expect(config.budgetTokens == TriAttentionConfiguration.v1BudgetTokens)
        #expect(config.implementationVersion == .v1)
        #expect(config.calibrationArtifactIdentity == nil)
    }

    @Test
    func settingsValueRoundTripsAcrossInstances() {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let first = SettingsManager()
        first.triattentionEnabled = true

        let second = SettingsManager()
        #expect(second.triattentionEnabled == true)
        #expect(second.makeTriAttentionConfig().enabled == true)
    }

    @Test
    func zeroArgEngineResolvesDisabledTriAttentionConfig() {
        let engine = AgentEngine()
        let config = engine.resolveTriAttentionConfig()

        #expect(config == .v1Disabled)
    }

    @Test
    func settingsBackedEngineResolvesHiddenTriAttentionSetting() {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        let engine = AgentEngine(settingsManager: settings)

        let disabled = engine.resolveTriAttentionConfig()
        #expect(disabled == .v1Disabled)

        settings.triattentionEnabled = true
        let enabled = engine.resolveTriAttentionConfig()
        #expect(enabled.enabled == true)
        #expect(enabled.budgetTokens == TriAttentionConfiguration.v1BudgetTokens)
        #expect(enabled.implementationVersion == .v1)
        #expect(enabled.calibrationArtifactIdentity == nil)

        var params = AgentGenerateParameters.forModel("qwen3.5-4b-paro")
        let originalTemperature = params.temperature
        let originalTopP = params.topP
        let originalPrefill = params.prefillStepSize
        params.triAttention = enabled

        #expect(params.temperature == originalTemperature)
        #expect(params.topP == originalTopP)
        #expect(params.prefillStepSize == originalPrefill)
        #expect(params.triAttention == enabled)
    }
}
