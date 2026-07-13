import Foundation

@MainActor
final class AppTerminationCoordinator {
    struct Steps {
        let stopHotkeys: @MainActor () -> Void
        let cancelForegroundGenerationAndWait: @MainActor () async -> Void
        let stopHTTPServerAndDrain: @MainActor () async -> Void
        let cancelLLMGenerationAndWait: @MainActor () async -> Void
        let stopSpeech: @MainActor () -> Void
        let unloadLLM: @MainActor () -> Void
        let awaitLLMUnload: @MainActor () async -> Void
        /// The v2 engine's deterministic teardown (ADR-0039): cancels any
        /// active utterance, frees weights/KV/caches, syncs the GPU stream.
        let unloadSpeech: @MainActor () async -> Void
        let synchronizeGPU: @MainActor () -> Void

        init(
            stopHotkeys: @escaping @MainActor () -> Void,
            cancelForegroundGenerationAndWait: @escaping @MainActor () async -> Void,
            stopHTTPServerAndDrain: @escaping @MainActor () async -> Void,
            cancelLLMGenerationAndWait: @escaping @MainActor () async -> Void,
            stopSpeech: @escaping @MainActor () -> Void,
            unloadLLM: @escaping @MainActor () -> Void,
            awaitLLMUnload: @escaping @MainActor () async -> Void,
            unloadSpeech: @escaping @MainActor () async -> Void,
            synchronizeGPU: @escaping @MainActor () -> Void
        ) {
            self.stopHotkeys = stopHotkeys
            self.cancelForegroundGenerationAndWait = cancelForegroundGenerationAndWait
            self.stopHTTPServerAndDrain = stopHTTPServerAndDrain
            self.cancelLLMGenerationAndWait = cancelLLMGenerationAndWait
            self.stopSpeech = stopSpeech
            self.unloadLLM = unloadLLM
            self.awaitLLMUnload = awaitLLMUnload
            self.unloadSpeech = unloadSpeech
            self.synchronizeGPU = synchronizeGPU
        }
    }

    private let steps: Steps
    private var hasPreparedForTermination = false

    init(steps: Steps) {
        self.steps = steps
    }

    func prepareForTermination() async {
        guard !hasPreparedForTermination else { return }
        hasPreparedForTermination = true

        steps.stopHotkeys()

        await steps.cancelForegroundGenerationAndWait()
        await steps.stopHTTPServerAndDrain()
        await steps.cancelLLMGenerationAndWait()

        steps.stopSpeech()

        steps.unloadLLM()
        await steps.awaitLLMUnload()
        await steps.unloadSpeech()
        steps.synchronizeGPU()
    }
}
