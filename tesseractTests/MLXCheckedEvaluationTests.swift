import MLX
import Testing

@testable import Tesseract_Agent

struct MLXCheckedEvaluationTests {
    @Test func scopedBoundaryConvertsMLXErrorsToThrows() {
        do {
            try MLXCheckedEvaluation.withErrors { error in
                let left = MLXArray(0 ..< 10, [2, 5])
                let right = MLXArray(0 ..< 15, [3, 5])
                _ = left + right
                try error.check()
            }
            Issue.record("Expected MLX error")
        } catch let error as MLXError {
            if case .caught = error {
                #expect(true)
            } else {
                Issue.record("Unexpected MLX error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
