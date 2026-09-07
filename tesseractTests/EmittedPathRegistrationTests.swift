import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Emitted Path registration (ADR-0063 decisions 2/3/6/9) end to end on
/// the fake tokenizer: the path built from the fed ids and the stop id,
/// the fidelity gate, the key hashed from the stored render through its
/// last end-of-turn marker, and the skip-reason vocabulary the Leaf Store
/// logs — including the mapping from a refused live capture.
struct EmittedPathRegistrationTests {

    private static let fingerprint = "fp-reg"
    private let tokenizer = GreedyTokenizer(pieces: chatMLGreedyPieces + ["hello", "hel", "lo"])

    private var imEnd: Int { tokenizer.convertTokenToId("<|im_end|>")! }

    private func marker() throws -> EndOfTurnMarker {
        let probe = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "probe"],
                ["role": "assistant", "content": EndOfTurnMarker.probeContent],
            ],
            tools: nil, additionalContext: ["add_generation_prompt": false])
        return try #require(EndOfTurnMarker.derive(probeRender: probe, tokenizer: tokenizer))
    }

    /// The request-edge prompt (with generation prompt) and the stored
    /// render (without) for a one-turn conversation.
    private func renders(assistant: String) throws -> (prompt: [Int], storedBytes: [UInt8]) {
        let prompt = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "hi"]], tools: nil, additionalContext: nil)
        let stored = try tokenizer.renderChatTemplate(
            messages: [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": assistant],
            ],
            tools: nil, additionalContext: ["add_generation_prompt": false])
        return (prompt, Array(stored.utf8))
    }

    private func inputs(
        index: EmittedPathIndex,
        generated: [Int],
        stoppedOn: Int?,
        storedContent: String = "hello"
    ) throws -> EmittedPathRegistration.Inputs {
        let (prompt, storedBytes) = try renders(assistant: storedContent)
        return EmittedPathRegistration.Inputs(
            index: index,
            fingerprint: Self.fingerprint,
            marker: try marker(),
            tokenizer: tokenizer,
            storedRenderBytes: storedBytes,
            storedMessage: .assistant(content: storedContent),
            promptKeyPath: prompt,
            generatedTokens: generated,
            stoppedOn: stoppedOn,
            toolCallFormat: .xmlFunction,
            tools: nil,
            startsInsideThinkBlock: false)
    }

    @Test func aTurnStoppedOnTheMarkerRegistersThePathAsFed() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hello = try #require(tokenizer.convertTokenToId("hello"))
        let inputs = try inputs(index: index, generated: [hello, imEnd], stoppedOn: imEnd)
        guard case .registered(let registered) = EmittedPathRegistration.register(inputs) else {
            Issue.record("expected a registration")
            return
        }
        #expect(registered.pathLength == inputs.promptKeyPath.count + 2)
        #expect(registered.promptTokens == inputs.promptKeyPath.count)
        #expect(registered.generatedTokens == 2)
        #expect(registered.appendedEndOfTurn == false)
        #expect(registered.previousPathLength == nil)
        // The key covers the render through the marker (the trailing
        // newline after it is not part of the key).
        #expect(registered.prefixBytes == inputs.storedRenderBytes.count - 1)
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: inputs.storedRenderBytes, marker: Array("<|im_end|>".utf8))
        #expect(
            index.lookup(fingerprint: Self.fingerprint, hash: hashes.last!.hash)
                == inputs.promptKeyPath + [hello, imEnd])
    }

    @Test func aForeignStopIdIsKeptAndTheMarkerAppended() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hello = try #require(tokenizer.convertTokenToId("hello"))
        let foreign = 424_242
        let inputs = try inputs(index: index, generated: [hello, foreign], stoppedOn: foreign)
        guard case .registered(let registered) = EmittedPathRegistration.register(inputs) else {
            Issue.record("expected a registration")
            return
        }
        #expect(registered.appendedEndOfTurn == true)
        #expect(registered.pathLength == inputs.promptKeyPath.count + 3)
        let hashes = EmittedPathIndex.prefixHashes(
            renderedBytes: inputs.storedRenderBytes, marker: Array("<|im_end|>".utf8))
        #expect(
            index.lookup(fingerprint: Self.fingerprint, hash: hashes.last!.hash)
                == inputs.promptKeyPath + [hello, foreign, imEnd])
    }

    @Test func aTokenLimitCutAppendsTheMarker() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hello = try #require(tokenizer.convertTokenToId("hello"))
        let inputs = try inputs(index: index, generated: [hello], stoppedOn: nil)
        guard case .registered(let registered) = EmittedPathRegistration.register(inputs) else {
            Issue.record("expected a registration")
            return
        }
        #expect(registered.appendedEndOfTurn == true)
        #expect(registered.pathLength == inputs.promptKeyPath.count + 2)
    }

    @Test func aFidelityMismatchRegistersNothingAndCounts() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let goodbye = tokenizer.encode(text: "goodbye", addSpecialTokens: false)
        let inputs = try inputs(index: index, generated: goodbye + [imEnd], stoppedOn: imEnd)
        guard case .skipped(let skip) = EmittedPathRegistration.register(inputs) else {
            Issue.record("expected a skip")
            return
        }
        #expect(skip.reason == .fidelityRejected)
        #expect(skip.fidelity?.field == .content)
        let stats = index.statsSnapshot()
        #expect(stats.fidelityRejections == 1)
        #expect(stats.entryCount == 0)
    }

    @Test func theSameKeyIsOverwrittenByTheLaterSplit() throws {
        // `hello` as one piece, then as `hel`+`lo`: identical text, a
        // different emitted split — last writer wins, both lengths reported.
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hello = try #require(tokenizer.convertTokenToId("hello"))
        let hel = try #require(tokenizer.convertTokenToId("hel"))
        let lo = try #require(tokenizer.convertTokenToId("lo"))
        _ = EmittedPathRegistration.register(
            try inputs(index: index, generated: [hello, imEnd], stoppedOn: imEnd))
        let second = try inputs(index: index, generated: [hel, lo, imEnd], stoppedOn: imEnd)
        guard case .registered(let registered) = EmittedPathRegistration.register(second) else {
            Issue.record("expected a registration")
            return
        }
        #expect(registered.previousPathLength == second.promptKeyPath.count + 2)
        #expect(registered.pathLength == second.promptKeyPath.count + 3)
        #expect(index.statsSnapshot().overwrites == 1)
    }

    @Test func aStoredRenderWithoutTheMarkerRegistersNothing() throws {
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let hello = try #require(tokenizer.convertTokenToId("hello"))
        var inputs = try inputs(index: index, generated: [hello, imEnd], stoppedOn: imEnd)
        inputs = EmittedPathRegistration.Inputs(
            index: inputs.index, fingerprint: inputs.fingerprint, marker: inputs.marker,
            tokenizer: inputs.tokenizer, storedRenderBytes: Array("no markers here".utf8),
            storedMessage: inputs.storedMessage, promptKeyPath: inputs.promptKeyPath,
            generatedTokens: inputs.generatedTokens, stoppedOn: inputs.stoppedOn,
            toolCallFormat: inputs.toolCallFormat, tools: inputs.tools,
            startsInsideThinkBlock: inputs.startsInsideThinkBlock)
        guard case .skipped(let skip) = EmittedPathRegistration.register(inputs) else {
            Issue.record("expected a skip")
            return
        }
        #expect(skip.reason == .noEndOfTurnMarker)
    }

    // MARK: - Skip vocabulary

    @Test func refusedLiveCapturesMapToTheTicketsGuards() {
        typealias Reason = EmittedPathRegistration.SkipReason
        #expect(EmittedPathRegistration.skipReason(for: .intervened).reason == .intervened)
        #expect(
            EmittedPathRegistration.skipReason(for: .nonIdentityKeySpace).reason
                == .nonIdentityKeySpace)
        #expect(
            EmittedPathRegistration.skipReason(for: .noGeneratedTokens).reason == .noGeneratedTokens
        )
        #expect(
            EmittedPathRegistration.skipReason(
                for: .cacheOffsetOutsideLivePath(cacheOffset: 9, promptCount: 3, liveCount: 4)
            ).reason == .cacheOffsetOutsideLivePath)
        let longer = EmittedPathRegistration.skipReason(
            for: .liveLongerThanStored(cacheOffset: 9, storedLen: 8))
        #expect(longer.reason == .notProvenLive)
        #expect(longer.detail == "liveLongerThanStored")
        let diverged = EmittedPathRegistration.skipReason(
            for: .divergence(
                offset: 3, liveToken: 1, storedToken: 2, liveContext: [1], storedContext: [2]))
        #expect(diverged.reason == .notProvenLive)
        #expect(diverged.detail == "divergence")
    }

    @Test func skipReasonsAreTheTicketsCamelCaseWireStrings() {
        typealias Reason = EmittedPathRegistration.SkipReason
        #expect(Reason.nonIdentityKeySpace.rawValue == "nonIdentityKeySpace")
        #expect(Reason.noGeneratedTokens.rawValue == "noGeneratedTokens")
        #expect(Reason.cacheOffsetOutsideLivePath.rawValue == "cacheOffsetOutsideLivePath")
        #expect(Reason.intervened.rawValue == "intervened")
        #expect(Reason.fidelityRejected.rawValue == "fidelityRejected")
        #expect(EmittedPathRegistration.stage == "emittedPathRegister")
    }

    // MARK: - Event wire format

    @Test func registerEventCarriesPrefixAndPathLengths() {
        let event = EmittedPathRegistration.RegisterEvent(
            registered: EmittedPathRegistration.Registered(
                prefixBytes: 120, pathLength: 40, promptTokens: 30, generatedTokens: 10,
                appendedEndOfTurn: false, previousPathLength: 38, evicted: 0),
            registerSeconds: 0.0025)
        #expect(event.eventName == "emittedPathRegister")
        #expect(
            event.fields.map { [$0.0, $0.1] } == [
                ["prefixBytes", "120"], ["pathLength", "40"], ["promptTokens", "30"],
                ["generatedTokens", "10"], ["appendedEndOfTurn", "false"], ["evicted", "0"],
                ["registerMs", "2.500"], ["overwrote", "true"], ["previousPathLength", "38"],
            ])
    }

    @Test func fidelityEventNamesTheFieldAndTheDifference() {
        let event = EmittedPathRegistration.FidelityEvent(
            mismatch: EmittedPathFidelity.Mismatch(
                field: .content, emittedLength: 11, storedLength: 12, firstDifference: 6))
        #expect(event.eventName == "emittedPathFidelity")
        #expect(
            event.fields.map { [$0.0, $0.1] } == [
                ["result", "mismatch"], ["field", "content"], ["emittedLength", "11"],
                ["storedLength", "12"], ["firstDifference", "6"],
            ])
    }

    @Test func overwriteEventCarriesBothLengths() {
        let event = EmittedPathRegistration.OverwriteEvent(pathLength: 40, previousPathLength: 38)
        #expect(event.eventName == "emittedPathOverwrite")
        #expect(
            event.fields.map { [$0.0, $0.1] } == [
                ["pathLength", "40"], ["previousPathLength", "38"],
            ])
    }
}
