import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct TriAttentionConfigPlumbingTests {

    private func clearTriAttentionDefaults() {
        UserDefaults.standard.removeObject(forKey: "triattentionEnabled")
    }

    private func makeFakeModelDirectory(paro: Bool = false) throws -> URL {
        try TriAttentionTestFixtures.makeFakeModelDirectory(
            prefix: "triattention-plumbing-model",
            paro: paro
        )
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
        #expect(config.prefixProtectionMode == .protectStablePrefixOnly)
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
    func triAttentionConfigurationDecodesLegacyPayloadWithoutPrefixProtectionMode() throws {
        let legacyJSON = #"""
        {
          "enabled": true,
          "budgetTokens": 4096,
          "calibrationArtifactIdentity": { "rawValue": "artifact" },
          "implementationVersion": "v1"
        }
        """#.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(
            TriAttentionConfiguration.self,
            from: legacyJSON
        )

        #expect(decoded.enabled == true)
        #expect(decoded.budgetTokens == 4096)
        #expect(decoded.calibrationArtifactIdentity?.rawValue == "artifact")
        #expect(decoded.implementationVersion == .v1)
        #expect(decoded.prefixProtectionMode == .protectNone)
    }

    /// `resetToDefaults()` must bring `triattentionEnabled` back to `false` and
    /// persist that value — otherwise a fresh `SettingsManager()` would rehydrate
    /// the stale `true` from UserDefaults and re-enable the feature silently.
    @Test
    func resetToDefaultsRestoresDisabledAndPersists() {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        settings.triattentionEnabled = true
        #expect(settings.triattentionEnabled == true)

        settings.resetToDefaults()
        #expect(settings.triattentionEnabled == false)

        let rehydrated = SettingsManager()
        #expect(rehydrated.triattentionEnabled == false)
        #expect(rehydrated.makeTriAttentionConfig().enabled == false)
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
        #expect(enabled.prefixProtectionMode == .protectStablePrefixOnly)

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

    @Test
    func agentEngineLoadForwardsExplicitTriAttentionOverrideToActor() async throws {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        settings.triattentionEnabled = false
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory(paro: true)
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        let override = TriAttentionConfiguration(enabled: true)
        do {
            try await engine.loadModel(
                from: fakeDir,
                visionMode: false,
                triAttention: override
            )
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected: container load fails after install runs.
        }

        let selection = await engine.llmActor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.requestedConfiguration == override)
    }

    @Test
    func agentEngineLoadFallsBackToSettingsWhenNoExplicitTriAttention() async throws {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        settings.triattentionEnabled = true
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory(paro: true)
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected.
        }

        let selection = await engine.llmActor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.requestedConfiguration.enabled == true)
    }

    @Test
    func agentEngineLoadRecordsDenseFallbackReasonWhenUnsupported() async throws {
        clearTriAttentionDefaults()
        defer { clearTriAttentionDefaults() }

        let settings = SettingsManager()
        settings.triattentionEnabled = true
        let engine = AgentEngine(settingsManager: settings)
        let fakeDir = try makeFakeModelDirectory(paro: true)
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(
                from: fakeDir,
                visionMode: true,
                triAttention: TriAttentionConfiguration(enabled: true)
            )
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected.
        }

        let selection = await engine.llmActor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.fallbackReason == .visionMode)
    }
}
