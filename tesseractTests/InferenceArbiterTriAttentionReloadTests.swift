import Foundation
import Observation
import Testing

@testable import Tesseract_Agent

/// Epic 4 Tasks 3 + 4:
/// - `reloadLLMIfNeeded()` must propagate a settings change (model ID, vision
///   mode, TriAttention toggle) into an eager model reload via the arbiter.
/// - That reload must work regardless of `isServerEnabled` — the public HTTP
///   listener is a transport concern, not a dependency of the canonical
///   internal server-core inference path.
@MainActor
struct InferenceArbiterTriAttentionReloadTests {

    private func clearDefaults() {
        UserDefaults.standard.removeObject(forKey: "triattentionEnabled")
        UserDefaults.standard.removeObject(forKey: "selectedAgentModelID")
        UserDefaults.standard.removeObject(forKey: "isServerEnabled")
    }

    private func makeArbiter(_ settings: SettingsManager) -> InferenceArbiter {
        InferenceArbiter(
            agentEngine: AgentEngine(),
            speechEngine: SpeechEngine(),
            imageGenEngine: ImageGenEngine(),
            zimageGenEngine: ZImageGenEngine(),
            settingsManager: settings,
            modelDownloadManager: ModelDownloadManager()
        )
    }

    /// The reload primitive must actually run `ensureLoaded(.llm)` → `loadSlot`,
    /// not be a local no-op. Pointing the setting at a never-downloaded model
    /// ID forces `loadSlot` to throw `modelNotDownloaded` — observable proof
    /// the lease body reached the download check before returning.
    @Test func reloadLLMIfNeededSurfacesModelNotDownloadedWhenModelMissing() async throws {
        clearDefaults()
        defer { clearDefaults() }

        let settings = SettingsManager()
        let missingModelID = "tesseract-reload-test-missing-model"
        settings.selectedAgentModelID = missingModelID
        let arbiter = makeArbiter(settings)

        do {
            try await arbiter.reloadLLMIfNeeded()
            Issue.record("reloadLLMIfNeeded should have thrown modelNotDownloaded")
        } catch let AgentEngineError.modelNotDownloaded(modelID) {
            #expect(modelID == missingModelID)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    /// Task 4 invariant. `isServerEnabled = false` must not short-circuit the
    /// arbiter path — internal inference has no dependency on the public
    /// listener. Combine with `triattentionEnabled = true` so the test also
    /// exercises the TriAttention seam under "server off" conditions.
    @Test func reloadLLMIfNeededRunsWhenPublicListenerDisabled() async throws {
        clearDefaults()
        defer { clearDefaults() }

        let settings = SettingsManager()
        settings.isServerEnabled = false
        settings.triattentionEnabled = true
        let missingModelID = "tesseract-reload-test-missing-model"
        settings.selectedAgentModelID = missingModelID
        let arbiter = makeArbiter(settings)

        do {
            try await arbiter.reloadLLMIfNeeded()
            Issue.record("reloadLLMIfNeeded should have thrown modelNotDownloaded")
        } catch let AgentEngineError.modelNotDownloaded(modelID) {
            // Reaching the download check under `isServerEnabled = false` is
            // the signal: nothing in the arbiter path inspects that flag.
            #expect(modelID == missingModelID)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    /// Mirror of the `DependencyContainer.setup()` observation wiring.
    /// Verifies the two behaviors we rely on:
    ///   1. The initial `Observations` emit at subscription time does NOT
    ///      trigger a reload when the LLM is not loaded — preserves lazy load
    ///      at app launch.
    ///   2. A subsequent toggle flip DOES trigger a reload once the LLM slot
    ///      reports loaded.
    /// Uses a recorder stand-in for the arbiter; the primitive itself is
    /// covered by the two tests above.
    @Test func toggleObservationSkipsInitialEmitAndFiresOnChangeWhenLLMLoaded() async throws {
        clearDefaults()
        defer { clearDefaults() }

        let settings = SettingsManager()

        // `iterations` counts every observation emit; `records` counts only
        // emits that pass the "loaded" guard. Waiting on `iterations` instead
        // of fixed sleeps makes each stage deterministic under CI load.
        actor Counter {
            private(set) var iterations = 0
            private(set) var records = 0
            func bumpIteration() { iterations += 1 }
            func record() { records += 1 }
        }
        let counter = Counter()

        @MainActor final class LoadedFlag { var value = false }
        let loaded = LoadedFlag()

        let observer = Task { @MainActor in
            for await _ in Observations({ settings.triattentionEnabled }) {
                await counter.bumpIteration()
                guard loaded.value else { continue }
                await counter.record()
            }
        }
        defer { observer.cancel() }

        // Stage 1 — initial subscribe emit. Guard blocks because loaded=false.
        try await waitUntil({ await counter.iterations >= 1 })
        #expect(await counter.records == 0)

        // Stage 2 — flip while still "not loaded". Guard continues to block.
        settings.triattentionEnabled = true
        try await waitUntil({ await counter.iterations >= 2 })
        #expect(await counter.records == 0)

        // Stage 3 — mark loaded, flip again. Guard now passes.
        loaded.value = true
        settings.triattentionEnabled = false
        try await waitUntil({ await counter.iterations >= 3 })
        #expect(await counter.records == 1)
    }

    /// Poll a condition until it returns true or a deadline elapses. Avoids
    /// fixed `Task.sleep` delays that are flaky under CI load.
    private func waitUntil(
        timeout: Duration = .seconds(3),
        _ condition: @Sendable () async -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while !(await condition()) {
            try await Task.sleep(for: .milliseconds(10))
            if ContinuousClock.now >= deadline {
                Issue.record("waitUntil timed out after \(timeout)")
                return
            }
        }
    }
}
