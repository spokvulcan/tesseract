//
//  ProofreadPassTests.swift
//  tesseractTests
//
//  The **Proofread Pass** policy (map #283, ADR-0034), driven hermetically
//  through its injected closures — no MLX, no model files. Every skip path
//  must return `nil` (fail-open: the caller commits the raw text), the pass
//  must never touch the model while the GPU lease is held, and a budget
//  overrun must cut the pass off rather than stall the commit.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ProofreadPassTests {

    /// Records the model-side closure traffic so tests can assert not just
    /// the verdict but *whether the model was touched at all*.
    @MainActor
    final class ModelRecorder {
        private(set) var loadCount = 0
        private(set) var runCount = 0
        private(set) var unloadCount = 0
        var loadError: (any Error)?
        var reply: String = ""
        var runError: (any Error)?

        func load() throws {
            loadCount += 1
            if let loadError { throw loadError }
        }

        func run() throws -> String {
            runCount += 1
            if let runError { throw runError }
            return reply
        }

        func unload() { unloadCount += 1 }
    }

    struct Boom: Error {}

    private func makePass(
        recorder: ModelRecorder,
        enabled: Bool = true,
        gpuBusy: Bool = false,
        directory: URL? = URL(fileURLWithPath: "/tmp/proofread-model"),
        budget: Duration = .seconds(4)
    ) -> ProofreadPass {
        ProofreadPass(
            isEnabled: { enabled },
            isGPUBusy: { gpuBusy },
            modelDirectory: { directory },
            loadModel: { _ in try await recorder.load() },
            runModel: { _, _ in try await recorder.run() },
            unloadModel: { await recorder.unload() },
            budget: budget
        )
    }

    // MARK: - Skip paths (every one returns nil and fails open)

    @Test func disabledSkipsWithoutTouchingTheModel() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder, enabled: false)

        #expect(await pass.proofread("hello world") == nil)
        #expect(recorder.loadCount == 0)
        #expect(recorder.runCount == 0)
    }

    @Test func missingModelDirectorySkipsWithoutTouchingTheModel() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder, directory: nil)

        #expect(await pass.proofread("hello world") == nil)
        #expect(recorder.loadCount == 0)
    }

    /// Skip-when-busy is the owner's locked call: while the agent or server
    /// holds the GPU lease, the pass must not load, not run, not queue.
    @Test func gpuLeaseHeldSkipsWithoutTouchingTheModel() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder, gpuBusy: true)

        #expect(await pass.proofread("hello world") == nil)
        #expect(recorder.loadCount == 0)
        #expect(recorder.runCount == 0)
    }

    @Test func loadFailureSkipsAndLeavesResidencyFalse() async {
        let recorder = ModelRecorder()
        recorder.loadError = Boom()
        let pass = makePass(recorder: recorder)

        #expect(await pass.proofread("hello world") == nil)
        #expect(recorder.runCount == 0)
        #expect(!pass.isModelLoaded)
    }

    @Test func modelErrorSkips() async {
        let recorder = ModelRecorder()
        recorder.runError = Boom()
        let pass = makePass(recorder: recorder)

        #expect(await pass.proofread("hello world") == nil)
    }

    /// A model call that overruns the budget is cut off — the commit must
    /// never wait on a wedged generation.
    @Test func budgetOverrunSkips() async {
        let recorder = ModelRecorder()
        let pass = ProofreadPass(
            isEnabled: { true },
            isGPUBusy: { false },
            modelDirectory: { URL(fileURLWithPath: "/tmp/proofread-model") },
            loadModel: { _ in try await recorder.load() },
            runModel: { _, _ in
                try await Task.sleep(for: .seconds(60))
                return "never"
            },
            unloadModel: { await recorder.unload() },
            budget: .milliseconds(50)
        )

        #expect(await pass.proofread("hello world") == nil)
    }

    // MARK: - Verdicts

    @Test func replyIsParsedIntoTheVerdict() async {
        let recorder = ModelRecorder()
        recorder.reply = "piece of cake"
        let pass = makePass(recorder: recorder)

        let verdict = await pass.proofread("peace of cake")

        #expect(
            verdict
                == .corrected(
                    text: "piece of cake",
                    edits: [WordEdit(original: "peace", replacement: "piece")]))
        #expect(recorder.loadCount == 1)
        #expect(recorder.runCount == 1)
        #expect(pass.isModelLoaded)
    }

    @Test func rejectReplyParsesToRejected() async {
        let recorder = ModelRecorder()
        recorder.reply = "REJECT: unintelligible mumbling"
        let pass = makePass(recorder: recorder)

        #expect(
            await pass.proofread("asdf ghjk")
                == .rejected(reason: "unintelligible mumbling"))
    }

    @Test func echoedReplyParsesToUnchanged() async {
        let recorder = ModelRecorder()
        recorder.reply = "hello world"
        let pass = makePass(recorder: recorder)

        #expect(await pass.proofread("hello world") == .unchanged)
    }

    // MARK: - Residency lifecycle

    @Test func prewarmLoadsWhenEnabledAndDownloaded() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder)

        await pass.prewarm()

        #expect(recorder.loadCount == 1)
        #expect(pass.isModelLoaded)
    }

    @Test func prewarmSkipsWhenDisabled() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder, enabled: false)

        await pass.prewarm()

        #expect(recorder.loadCount == 0)
        #expect(!pass.isModelLoaded)
    }

    @Test func unloadClearsResidency() async {
        let recorder = ModelRecorder()
        let pass = makePass(recorder: recorder)
        await pass.prewarm()
        #expect(pass.isModelLoaded)

        await pass.unload()

        #expect(recorder.unloadCount == 1)
        #expect(!pass.isModelLoaded)
    }
}
