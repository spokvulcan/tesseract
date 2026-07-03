import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The #136 harness (PRD #137, user story 11): a hermetic bench for the
/// intermittent GPU kill on request-path restore of SSD-resident leaves.
/// A **Segment Chain** is written through to a temp-dir SSD store, a fresh
/// module warm-starts over it — RAM tier empty, descriptors only — and the
/// next request must hydrate the chain from disk, restore it onto the toy
/// model *inside the session*, prefill the suffix, and resume generation:
/// exactly the restore → prefill → decode window the #136 kills fire in,
/// minus the multi-GB model. The kill itself is out of scope; this pins the
/// sequencing and gives the parked MLX-UAF discriminate plan a fast seat.
@Suite struct ServerCompletionSSDRestoreTests {

    @MainActor
    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        // The toy's head dimension is far below any quantization group size;
        // KV quantization stays a recorded no-op verb in these suites.
        parameters.kvBits = nil
        return parameters
    }

    @Test func warmStartedSegmentChainHydratesRestoresAndResumesGeneration() async throws {
        let tokenizer = ToySequencingTokenizer()
        let opener = HTTPPrefixCacheMessage(
            role: .user, content: String(repeating: "a", count: 600))
        let round1 = HTTPPrefixCacheConversation(systemPrompt: nil, messages: [opener])
        let round2 = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                opener,
                .assistant(content: "Hello!"),
                HTTPPrefixCacheMessage(role: .user, content: "More?"),
            ]
        )
        let render2 = try tokenizer.applyChatTemplate(
            messages: round2.promptMessages, tools: nil, additionalContext: nil
        )
        let script = render2 + Array("Sure.".utf8).map(Int.init)

        let ssdRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-restore-harness-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: ssdRoot) }
        let ssdConfig = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: ssdRoot,
            budgetBytes: 1 << 30,
            maxPendingBytes: 1 << 30
        )
        let fingerprint = "toy-fingerprint"
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: script),
            tokenizer: tokenizer
        )

        // -- Session A: the cold generation whose leaf writes through to the
        // SSD tier as a committed Segment Chain; flush persists the manifest.
        let sessionA = ServerCompletionFixture(
            provider: provider, fingerprint: fingerprint, ssdConfig: ssdConfig
        )
        let handle1 = try await sessionA.start(
            conversation: round1, parameters: Self.parameters()
        )
        let (text1, _) = try await collectServerText(handle1)
        #expect(text1 == "Hello!")
        await sessionA.drain()
        await sessionA.flush()

        // The chain is durably on disk before the "restart".
        let onDisk = try #require(
            FileManager.default.enumerator(at: ssdRoot, includingPropertiesForKeys: nil)?
                .compactMap { $0 as? URL }
                .filter { !$0.hasDirectoryPath }
        )
        #expect(onDisk.isEmpty == false)

        // -- Session B: a fresh module over the same store, as after an app
        // relaunch. Warm start restores descriptors only — the RAM tier is
        // empty, so the hit below can only be served by SSD hydration.
        let sessionB = ServerCompletionFixture(
            provider: provider, fingerprint: fingerprint, ssdConfig: ssdConfig
        )
        let verbsBefore = provider.recorder.verbs.count

        let storedTokens1 = try tokenizer.applyChatTemplate(
            messages: round1.appendingAssistant(.assistant(content: "Hello!")).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        let handle2 = try await sessionB.start(
            conversation: round2, parameters: Self.parameters()
        )
        #expect(handle2.cachedTokenCount == storedTokens1.count)
        let (text2, _) = try await collectServerText(handle2)
        #expect(text2 == "Sure.")

        // The hydrated snapshot went through the session's restore verb onto
        // the model — the #136 crash site — before the suffix prefill.
        let verbs = Array(provider.recorder.verbs.dropFirst(verbsBefore))
        #expect(
            verbs == [
                .prepare, .restore, .prefill, .quantizeKVCache, .makeDecodeIterator,
                .captureSnapshot,
            ]
        )
        await sessionB.drain()
    }
}
