import Foundation
import Testing
@testable import Tesseract_Agent

@MainActor
struct AgentEngineLoadStateTests {

    /// A failed load must reset `isLoading`, or the reentrancy guard at the
    /// top of `loadModel` silently swallows every later attempt (and the
    /// arbiter keeps a loaded slot with no model behind it).
    @Test
    func loadFailureResetsIsLoadingSoALaterLoadCanRun() async throws {
        let engine = AgentEngine()
        let fakeDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "load-state-fake-model-\(UUID().uuidString)", isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: fakeDir, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: fakeDir) }

        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected loadModel to throw for a non-model directory")
        } catch {
            // Expected — the container load fails on a directory without weights.
        }

        #expect(engine.isLoading == false)

        // The guard no longer short-circuits: a second attempt reaches the
        // real load path again (observable as the same throw, not a silent
        // early return).
        do {
            try await engine.loadModel(from: fakeDir, visionMode: false)
            Issue.record("expected the second loadModel to throw too")
        } catch {
            // Expected — proves the call got past the `isLoading` guard.
        }
    }
}
