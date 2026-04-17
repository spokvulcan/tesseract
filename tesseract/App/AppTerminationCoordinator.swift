import Foundation

@MainActor
final class AppTerminationCoordinator {
    struct Steps {
        let cancelSchedulingSubscriptions: @MainActor () -> Void
        let stopHotkeys: @MainActor () -> Void
        let stopSchedulingPolling: @MainActor () async -> Void
        let stopSchedulingHeartbeat: @MainActor () async -> Void
        let persistInterruptedTask: @MainActor () async -> Void
        let abortInFlightBackgroundAgent: @MainActor () async -> Void
        let cancelForegroundGenerationAndWait: @MainActor () async -> Void
        let stopHTTPServerAndDrain: @MainActor () async -> Void
        let cancelLLMGenerationAndWait: @MainActor () async -> Void
        let stopSpeech: @MainActor () -> Void
        let cancelSpeechGeneration: @MainActor () async -> Void
        let clearSpeechVoiceAnchor: @MainActor () async -> Void
        let unloadLLM: @MainActor () -> Void
        let awaitLLMUnload: @MainActor () async -> Void
        let unloadSpeech: @MainActor () -> Void
        let synchronizeGPU: @MainActor () -> Void

        init(
            cancelSchedulingSubscriptions: @escaping @MainActor () -> Void,
            stopHotkeys: @escaping @MainActor () -> Void,
            stopSchedulingPolling: @escaping @MainActor () async -> Void,
            stopSchedulingHeartbeat: @escaping @MainActor () async -> Void,
            persistInterruptedTask: @escaping @MainActor () async -> Void,
            abortInFlightBackgroundAgent: @escaping @MainActor () async -> Void,
            cancelForegroundGenerationAndWait: @escaping @MainActor () async -> Void,
            stopHTTPServerAndDrain: @escaping @MainActor () async -> Void,
            cancelLLMGenerationAndWait: @escaping @MainActor () async -> Void,
            stopSpeech: @escaping @MainActor () -> Void,
            cancelSpeechGeneration: @escaping @MainActor () async -> Void,
            clearSpeechVoiceAnchor: @escaping @MainActor () async -> Void,
            unloadLLM: @escaping @MainActor () -> Void,
            awaitLLMUnload: @escaping @MainActor () async -> Void,
            unloadSpeech: @escaping @MainActor () -> Void,
            synchronizeGPU: @escaping @MainActor () -> Void
        ) {
            self.cancelSchedulingSubscriptions = cancelSchedulingSubscriptions
            self.stopHotkeys = stopHotkeys
            self.stopSchedulingPolling = stopSchedulingPolling
            self.stopSchedulingHeartbeat = stopSchedulingHeartbeat
            self.persistInterruptedTask = persistInterruptedTask
            self.abortInFlightBackgroundAgent = abortInFlightBackgroundAgent
            self.cancelForegroundGenerationAndWait = cancelForegroundGenerationAndWait
            self.stopHTTPServerAndDrain = stopHTTPServerAndDrain
            self.cancelLLMGenerationAndWait = cancelLLMGenerationAndWait
            self.stopSpeech = stopSpeech
            self.cancelSpeechGeneration = cancelSpeechGeneration
            self.clearSpeechVoiceAnchor = clearSpeechVoiceAnchor
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

        steps.cancelSchedulingSubscriptions()
        steps.stopHotkeys()

        await steps.stopSchedulingPolling()
        await steps.stopSchedulingHeartbeat()
        await steps.persistInterruptedTask()

        await steps.abortInFlightBackgroundAgent()
        await steps.cancelForegroundGenerationAndWait()
        await steps.stopHTTPServerAndDrain()
        await steps.cancelLLMGenerationAndWait()

        steps.stopSpeech()
        await steps.cancelSpeechGeneration()
        await steps.clearSpeechVoiceAnchor()

        steps.unloadLLM()
        await steps.awaitLLMUnload()
        steps.unloadSpeech()
        steps.synchronizeGPU()
    }
}
