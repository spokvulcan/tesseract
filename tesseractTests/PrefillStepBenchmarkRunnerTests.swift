import Testing

@testable import Tesseract_Agent

struct PrefillStepBenchmarkRunnerTests {

    @Test func benchmarkMatrixIncludesFiveColdAndFiveWarmCases() {
        let matrix = PrefillStepBenchmarkSupport.benchmarkMatrix()

        #expect(matrix.count == 10)
        #expect(matrix.filter { $0.mode == .cold }.count == 5)
        #expect(matrix.filter { $0.mode == .warm }.count == 5)
        #expect(matrix.map(\.prefillStepSize) == [256, 256, 512, 512, 1024, 1024, 2048, 2048, 4096, 4096])
    }

    @Test func adaptiveSuggestionUsesLowestPeakColdAndFastestWarm() {
        let measurements = [
            PrefillStepBenchmarkMeasurement(
                caseID: "cold-256",
                mode: .cold,
                prefillStepSize: 256,
                passed: true,
                error: nil,
                promptTimeSeconds: 2.0,
                externalTTFTSeconds: 2.2,
                promptTokenCount: 16_000,
                cachedTokenCount: 0,
                prefilledTokenCount: 16_000,
                peakMemoryMB: 3_000,
                activeMemoryBeforeMB: 1_000,
                activeMemoryAfterMB: 1_200
            ),
            PrefillStepBenchmarkMeasurement(
                caseID: "cold-1024",
                mode: .cold,
                prefillStepSize: 1024,
                passed: true,
                error: nil,
                promptTimeSeconds: 1.5,
                externalTTFTSeconds: 1.6,
                promptTokenCount: 16_000,
                cachedTokenCount: 0,
                prefilledTokenCount: 16_000,
                peakMemoryMB: 3_400,
                activeMemoryBeforeMB: 1_000,
                activeMemoryAfterMB: 1_100
            ),
            PrefillStepBenchmarkMeasurement(
                caseID: "warm-512",
                mode: .warm,
                prefillStepSize: 512,
                passed: true,
                error: nil,
                promptTimeSeconds: 0.9,
                externalTTFTSeconds: 1.0,
                promptTokenCount: 16_000,
                cachedTokenCount: 12_000,
                prefilledTokenCount: 4_000,
                peakMemoryMB: 2_100,
                activeMemoryBeforeMB: 1_000,
                activeMemoryAfterMB: 1_050
            ),
            PrefillStepBenchmarkMeasurement(
                caseID: "warm-2048",
                mode: .warm,
                prefillStepSize: 2048,
                passed: true,
                error: nil,
                promptTimeSeconds: 0.6,
                externalTTFTSeconds: 0.7,
                promptTokenCount: 16_000,
                cachedTokenCount: 12_000,
                prefilledTokenCount: 4_000,
                peakMemoryMB: 2_500,
                activeMemoryBeforeMB: 1_000,
                activeMemoryAfterMB: 1_050
            ),
        ]

        let summary = PrefillStepBenchmarkSupport.summarize(measurements: measurements)

        #expect(summary.lowestPeakColdStepSize == 256)
        #expect(summary.fastestWarmStepSize == 2048)
        #expect(summary.suggestedAdaptivePair == .init(coldStepSize: 256, warmStepSize: 2048))
    }

    @Test func warmCaseValidationRejectsMissAndExactRepeat() {
        let missFailure = PrefillStepBenchmarkSupport.validationFailure(
            mode: .warm,
            promptTokenCount: 16_000,
            cachedTokenCount: 0
        )
        let exactRepeatFailure = PrefillStepBenchmarkSupport.validationFailure(
            mode: .warm,
            promptTokenCount: 16_000,
            cachedTokenCount: 16_000
        )

        #expect(missFailure?.contains("cachedTokenCount > 0") == true)
        #expect(exactRepeatFailure?.contains("non-zero suffix prefill") == true)
    }

    @Test func fixtureBuilderKeepsWarmPromptSameLengthAsCold() {
        let measureTextTokens: (String) -> Int = { $0.count }
        let fixture = PrefillStepBenchmarkSupport.buildFixture(
            markerPair: ("A", "B"),
            targetStablePrefixTokens: 120,
            targetUserTokens: 40,
            measureTextTokens: measureTextTokens
        )

        let counts = fixture.promptTokenCounts { systemPrompt, userMessage, toolSpecs in
            systemPrompt.count + userMessage.count + toolSpecs.count * 10
        }

        #expect(counts.cold == counts.warm)
        #expect(counts.cold > 0)
    }
}
