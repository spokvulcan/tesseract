import Foundation
import Testing
@testable import Tesseract_Agent

@MainActor
struct AppTerminationCoordinatorTests {

    @Test
    func prepareForTerminationExecutesShutdownStepsInOrderOnce() async {
        var order: [String] = []

        let coordinator = AppTerminationCoordinator(
            steps: .init(
                stopHotkeys: { order.append("stopHotkeys") },
                cancelForegroundGenerationAndWait: {
                    order.append("cancelForegroundGenerationAndWait")
                },
                stopHTTPServerAndDrain: { order.append("stopHTTPServerAndDrain") },
                cancelLLMGenerationAndWait: { order.append("cancelLLMGenerationAndWait") },
                stopSpeech: { order.append("stopSpeech") },
                unloadLLM: { order.append("unloadLLM") },
                awaitLLMUnload: { order.append("awaitLLMUnload") },
                unloadSpeech: { order.append("unloadSpeech") },
                synchronizeGPU: { order.append("synchronizeGPU") }
            ))

        await coordinator.prepareForTermination()
        await coordinator.prepareForTermination()

        #expect(
            order == [
                "stopHotkeys",
                "cancelForegroundGenerationAndWait",
                "stopHTTPServerAndDrain",
                "cancelLLMGenerationAndWait",
                "stopSpeech",
                "unloadLLM",
                "awaitLLMUnload",
                "unloadSpeech",
                "synchronizeGPU",
            ])
    }
}
