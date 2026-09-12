import CoreGraphics
import Foundation
import ImageIO
import MLXLMCommon
import Testing
import UniformTypeIdentifiers

@testable import Tesseract_Agent

/// The Stage 1 replay gate's synthesized cases (ADR-0063, ticket #477):
/// every history shape the recordings cannot show, run hermetically through
/// the real **Server Completion** module — real prefix cache, Leaf Store
/// fast path, Emitted Path Index, SSD tier — over the content-relative toy
/// Model Session and the Qwen3.8-shaped toy template. Each case reads the
/// request's telemetry (the `leafStore`, `emittedPathResolve`,
/// `emittedPathRegister` and skip events), the handle's restored offset,
/// and the toy's tape of what the model was actually fed: the served
/// composition on a hit, the canonical encode on a miss, never a wrong
/// prompt.
@MainActor
struct EmittedPathSynthesizedReplayTests {

    /// The preserve-thinking render every case runs under unless it is
    /// about the think-stripping boundary: stop turns stay live.
    nonisolated static let preserving = TemplateRenderContext(
        kwargs: [.preserveThinking: true], preservesThinking: true)

    @MainActor
    private static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        return parameters
    }

    // MARK: - Cases

    @Test func quantizedPartitionKeepsCaptureCopyAndReportsWhy() async throws {
        let session = Session()
        var parameters = Self.parameters()
        parameters.kvBits = 8
        let turn = try await session.turn(
            Self.conversation([Self.user("hi")]),
            generated: session.completion(thinking: "plan", text: "hello world"),
            parameters: parameters)
        #expect(turn.leafStore["source"] == "live")
        #expect(turn.leafStore["copyReason"] == "quantized")
        #expect(turn.leafStore["emittedPath"] == "registered")
    }

    @Test func imageBearingRequestKeepsCopyEvenWhenTheTextProcessorDropsImages() async throws {
        let session = Session()
        let image = HTTPPrefixCacheImage(data: try Self.tinyPNG())
        let turn = try await session.turn(
            Self.conversation([
                HTTPPrefixCacheMessage(role: .user, content: "hi", images: [image])
            ]))
        #expect(turn.leafStore["source"] == "live")
        #expect(turn.leafStore["copyReason"] == "imageKeySpace")
    }

    @Test(arguments: [false, true])
    func liveTurnRegistersAndTheNextRequestServesTheWholePath(mtpLoaded: Bool) async throws {
        // A loaded but ineligible drafter must not turn an ordinary cold
        // or warm generation's leaf handoff into a full capture copy.
        let session = Session(hasMTPDrafter: mtpLoaded)
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        #expect(turn1.text == "hello world")
        #expect(turn1.thinking == "plan")
        #expect(turn1.leafStore["path"] == "live")
        #expect(turn1.leafStore["source"] == "handoff")
        #expect(turn1.leafStore["emittedPath"] == "registered")
        let pathLength = try #require(turn1.registeredPathLength)
        // The path: the prompt and the fed ids, the marker the model
        // stopped on last.
        #expect(pathLength == turn1.render.count + turn1.fedGenerated.count, turn1.account)
        #expect(turn1.fedPrompt == turn1.render, turn1.account)

        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("hello world"), Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.leafStore["source"] == "handoff")
        #expect(turn2.requestResolve["result"] == "hit")
        #expect(turn2.requestResolve["indexedPrefix"] == "\(pathLength)")
        // The leaf was captured at the cache's offset, the whole path: the
        // next request restores it and prefills the newline after the
        // marker, its new messages and the generation prompt.
        #expect(turn2.cached == pathLength, turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[pathLength...]), turn2.account)
        let newMessages =
            turn2.storedRender.count - (try turn1.storedRender(appending: "hello world").count)
        let glue = turn2.fedPrompt.count - newMessages
        #expect(glue == 1 + session.tokenizer.generationPrompts[0].count, "glue=\(glue)")
        #expect(glue == EmittedPathReplayGate.glueTokenAllowance)
        let stats = session.index.statsSnapshot()
        #expect(stats.registrations == 2)
        #expect(stats.overwrites == 0)
        #expect(stats.fidelityRejections == 0)
        #expect(stats.hits >= 1)
    }

    @Test func editedEarlierUserMessageFallsBackToCanonicalPastTheEdit() async throws {
        let session = Session()
        let turn1 = try await session.turn(Self.conversation([Self.user("hi there")]))
        #expect(turn1.registeredPathLength != nil)

        let request2 = Self.conversation([
            Self.user("hi again"), Self.assistant("hello world"), Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        // The restore lands at the deepest checkpoint below the edit
        // (**Chain-Prefix Restore**, ADR-0012); the canonical encode is
        // prefilled from there.
        let shared = Self.commonPrefix(turn1.livePath, turn2.render)
        #expect(shared > 0)
        #expect(turn2.cached <= shared, "restored \(turn2.cached) past the edit at \(shared)")
        #expect(turn2.cached > 0, "nothing before the edit reused")
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
    }

    @Test func editedAssistantMessageMissesOnThatMessageOnly() async throws {
        let session = Session()
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        #expect(turn1.registeredPathLength != nil)

        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("hello there"), Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        // The renders share the prompt and the unedited head of the turn;
        // the restore lands at the deepest checkpoint below the edit
        // (**Chain-Prefix Restore**, ADR-0012). The leaf's checkpoints are
        // its stable prefix — the system block and the user header, the
        // `system` checkpoint — and the leaf itself; the edit lies between
        // them, so the stable prefix is served and the user message and
        // the edited turn are prefilled canonically.
        let shared = Self.commonPrefix(turn1.livePath, turn2.render)
        #expect(shared > turn1.render.count)
        #expect(turn2.cached <= shared)
        let stablePrefix = try session.stablePrefix(of: turn1)
        #expect(turn2.cached == stablePrefix, turn2.account)
        #expect(turn2.diagnostics.cacheReason.hasPrefix("hit(system"), turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
        #expect(turn2.leafStore["emittedPath"] == "registered")
    }

    @Test func compactedHistoryIsAnOrdinaryMiss() async throws {
        let session = Session()
        _ = try await session.turn(Self.conversation([Self.user("hi")]))
        let turn2 = try await session.turn(
            Self.conversation([Self.user("hi"), Self.assistant("hello world"), Self.user("more")]),
            text: "again")
        #expect(turn2.requestResolve["result"] == "hit")

        let compacted = Self.conversation([Self.user("summary"), Self.user("more")])
        let turn3 = try await session.turn(compacted, text: "again")
        #expect(turn3.text == "again")
        #expect(turn3.requestResolve["result"] == "miss")
        #expect(turn3.requestResolve["reason"] == "noEntry")
        #expect(turn3.fedPrompt == Array(turn3.render[turn3.cached...]), turn3.account)
        #expect(turn3.leafStore["emittedPath"] == "registered")
    }

    @Test func reasoningEffortChangeRePrefillsFromTokenZero() async throws {
        let session = Session()
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        #expect(turn1.registeredPathLength != nil)

        let lowEffort = TemplateRenderContext(
            kwargs: [.preserveThinking: true], preservesThinking: true, reasoningEffort: .low)
        let request2 = Self.conversation(
            [Self.user("hi"), Self.assistant("hello world"), Self.user("more")], context: lowEffort)
        let turn2 = try await session.turn(request2, text: "again", context: lowEffort)
        #expect(turn2.text == "again")
        // The effort sits in the system block: the renders differ before
        // the first message (ADR-0060), so the index has no entry and the
        // previous effort's prefix is never served.
        #expect(Self.commonPrefix(turn1.render, turn2.render) < turn1.render.count)
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        #expect(turn2.cached == 0)
        #expect(turn2.fedPrompt == turn2.render, turn2.account)
        #expect(turn2.leafStore["emittedPath"] == "registered")
    }

    @Test func enableThinkingFlipMissesThePartitionAndNeverFeedsAWrongPrompt() async throws {
        let session = Session()
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        let path1 = try #require(turn1.registeredPathLength)

        let noThinking = TemplateRenderContext(
            kwargs: [.preserveThinking: true, .enableThinking: false], preservesThinking: true)
        let request2 = Self.conversation(
            [Self.user("hi"), Self.assistant("hello world"), Self.user("more")], context: noThinking
        )
        let turn2 = try await session.turn(
            request2, thinking: nil, text: "again", context: noThinking)
        #expect(turn2.text == "again")
        #expect(turn2.thinking.isEmpty)
        // The flip changes only the generation prompt: the history bytes are
        // identical, so the partition-agnostic index serves the stored turn
        // — and the flipped kwarg selects its own, empty cache partition
        // (`templateContextDigest`), today's miss of the whole history (spec
        // #471 decision 23): the composition is prefilled from token 0.
        #expect(Self.commonPrefix(turn1.render, turn2.render) >= turn1.storedRender.count)
        #expect(turn2.requestResolve["result"] == "hit")
        #expect(turn2.requestResolve["indexedPrefix"] == "\(path1)")
        #expect(turn2.requestResolve["markerDepth"] == "1")
        #expect(turn2.cached == 0, turn2.account)
        // Never a wrong prompt: the served turn, then the canonical encode
        // under the flipped context — the closed think block last.
        #expect(turn2.fedPrompt == turn1.livePath + Array(turn2.render[path1...]), turn2.account)
        let closedPrompt = session.tokenizer.generationPrompts[1]
        #expect(turn2.fedPrompt.suffix(closedPrompt.count).elementsEqual(closedPrompt))
    }

    @Test func sameParentDifferentSplitsLastWriterWins() async throws {
        let session = Session()
        let parent = Self.conversation([Self.user("hi")])
        let thinking = session.completion(thinking: "plan", text: "")
        let joined = session.tokenizer.encode(text: "KNI", addSpecialTokens: false)
        let split = ["K", "NI"].flatMap {
            session.tokenizer.encode(text: $0, addSpecialTokens: false)
        }
        #expect(joined.count == 1 && split.count == 2)

        let turn1 = try await session.turn(parent, generated: thinking + joined)
        #expect(turn1.text == "KNI")
        let path1 = try #require(turn1.registeredPathLength)
        let turn1b = try await session.turn(parent, generated: thinking + split)
        #expect(turn1b.text == "KNI")
        let path2 = try #require(turn1b.registeredPathLength)
        #expect(path2 == path1 + 1)
        #expect(turn1b.register["overwrote"] == "true")
        #expect(turn1b.register["previousPathLength"] == "\(path1)")
        #expect(turn1b.events.contains { $0.eventName == "emittedPathOverwrite" })
        #expect(session.index.statsSnapshot().overwrites == 1)

        // The earlier branch's next request resolves to the later path and
        // hits the later leaf while it is resident.
        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("KNI"), Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "hit")
        #expect(turn2.requestResolve["indexedPrefix"] == "\(path2)")
        #expect(turn2.cached == path2, turn2.account)
        #expect(!turn2.fedPrompt.contains(joined[0]))

        // Otherwise one miss: a module with an empty cache over the same
        // index feeds the served composition — the later split, not the
        // canonical encode — from token 0.
        session.restart(index: session.index)
        let turn3 = try await session.turn(request2, text: "again")
        #expect(turn3.text == "again")
        #expect(turn3.requestResolve["result"] == "hit")
        #expect(turn3.cached == 0)
        #expect(Array(turn3.fedPrompt.prefix(path2)) == turn1b.livePath, turn3.account)
        #expect(!turn3.fedPrompt.contains(joined[0]))
        #expect(turn3.fedPrompt.count == path2 + turn2.fedPrompt.count)
    }

    @Test func responseConversionFaultIsRejectedAndRePrefilledFromTheFirstDifference() async throws
    {
        let tokenizer = EmittedPathToyTokenizer()
        let worldID = try #require(tokenizer.encode(text: "world", addSpecialTokens: false).first)
        let session = Session(fault: { inner in
            FaultyStreamTokenizer(inner: inner, targetID: worldID) {
                $0.replacingOccurrences(of: "world", with: "wor1d")
            }
        })
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        #expect(turn1.text.contains("wor1d"), "stream text: \(turn1.text)")
        #expect(turn1.leafStore["path"] == "live")
        #expect(turn1.leafStore["source"] == "handoff")
        #expect(turn1.leafStore["emittedPath"] == "skipped")
        #expect(turn1.leafStore["emittedPathSkip"] == "fidelityRejected")
        let fidelity = try #require(turn1.event("emittedPathFidelity"))
        #expect(Self.fields(fidelity)["result"] == "mismatch")
        #expect(turn1.level(of: fidelity) == .warning, "the mismatch is a warning event")
        let stats = session.index.statsSnapshot()
        #expect(stats.fidelityRejections == 1)
        #expect(stats.registrations == 0)

        // The client echoes what it streamed: the next request misses the
        // index and re-prefills canonically from the first differing token,
        // the prefix before it reused.
        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant(turn1.text, reasoning: turn1.thinking),
            Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        let divergence = Self.commonPrefix(turn1.livePath, turn2.render)
        #expect(divergence > turn1.render.count)
        #expect(turn2.cached <= divergence, turn2.account)
        // The deepest checkpoint below the fault: the stable prefix, as in
        // the edited-assistant case.
        let stablePrefix = try session.stablePrefix(of: turn1)
        #expect(turn2.cached == stablePrefix, turn2.account)
        #expect(turn2.diagnostics.cacheReason.hasPrefix("hit(system"), turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
    }

    @Test func imageBearingRequestNeitherRegistersNorResolves() async throws {
        // The vision-container instance: images key into a non-identity
        // key space (a flat text instance drops them and keys text-only,
        // issue #439). The template renders each image's placeholder run
        // in place; the stub supplies the grid.
        let imagePadID = EmittedPathToyTokenizer().imagePadID
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5", "image_token_id": imagePadID,
                "vision_config": ["num_heads": 16, "spatial_merge_size": 2],
            ],
            chatTemplate: nil)
        let session = Session(
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: imagePadID, padRunLength: EmittedPathToyTokenizer.imagePadRunLength,
                frame: THW(1, 8, 8), inlineRuns: true),
            identity: identity)
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        #expect(turn1.text == "hello world")
        #expect(turn1.leafStore["emittedPath"] == "skipped")
        #expect(turn1.event("emittedPathRegister") == nil)

        let image = HTTPPrefixCacheImage(data: try Self.tinyPNG())
        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("hello world"),
            HTTPPrefixCacheMessage(role: .user, content: "look", images: [image]),
        ])
        let turn2 = try await session.turn(request2, text: "nice")
        #expect(turn2.text == "nice")
        #expect(turn2.requestResolve.isEmpty)
        #expect(turn2.resolveSkips.contains("media"), "resolve skips: \(turn2.resolveSkips)")
        #expect(turn2.leafStore["path"] != "live")
        #expect(turn2.leafStore["emittedPath"] == "skipped")
        #expect(turn2.event("emittedPathRegister") == nil)
        // Today's path: the processor's placeholder run is what the model
        // was fed — never a key-space pseudo-token.
        #expect(turn2.fedPrompt.contains(imagePadID), turn2.account)
        #expect(turn2.feeds.allSatisfy { $0.id >= 0 && $0.id < ToyVocabulary.size })
        #expect(turn2.cached > 0, "today's path did not reuse the text prefix")
        let stats = session.index.statsSnapshot()
        #expect(stats.registrations == 0)
        #expect(stats.hits == 0)
    }

    @Test func evictionPastTheByteBoundMissesCanonicallyThenReRegisters() async throws {
        // A budget that holds the longest path of the scenario — the second
        // request's: its prompt, its generated ids and the stop token — and
        // not two of the first turn's.
        let probe = Session()
        let request1 = Self.conversation([Self.user("hi")])
        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("hello world"), Self.user("more"),
        ])
        let pathLength =
            try probe.render(request1).count
            + probe.completion(thinking: "plan", text: "hello world").count + 1
        let path2Length =
            try probe.render(request2).count
            + probe.completion(thinking: "plan", text: "again").count + 1
        #expect(path2Length < 2 * pathLength - 2)
        let index = EmittedPathIndex(byteBudget: path2Length * MemoryLayout<Int>.size)
        let session = Session(index: index)

        let turn1 = try await session.turn(request1)
        #expect(turn1.registeredPathLength == pathLength, turn1.account)
        let other = try await session.turn(Self.conversation([Self.user("there")]))
        #expect(other.register["evicted"] == "1")
        #expect(index.statsSnapshot().evictions == 1)

        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        // The leaf itself is still resident: the canonical encode finds it.
        #expect(turn2.cached == pathLength, turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
        #expect(turn2.leafStore["emittedPath"] == "registered")
        #expect(turn2.registeredPathLength == path2Length)
        let stats = index.statsSnapshot()
        #expect(stats.registrations == 3)
        #expect(stats.evictions == 2)
        #expect(stats.entryCount == 1)
    }

    @Test func restartWithASurvivingSSDLeafMissesOnceWithNoWrongState() async throws {
        let ssdRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("emitted-path-restart-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: ssdRoot) }
        let ssdConfig = SSDPrefixCacheConfig(
            enabled: true, rootURL: ssdRoot, budgetBytes: 1 << 30, maxPendingBytes: 1 << 30)
        let session = Session(ssdConfig: ssdConfig)
        let turn1 = try await session.turn(Self.conversation([Self.user("hi")]))
        let pathLength = try #require(turn1.registeredPathLength)
        await session.fixture.drain()
        await session.fixture.flush()

        // The restart: a fresh module and a fresh (empty) index over the
        // same SSD root; the leaf survives on disk, the path does not.
        session.restart(index: EmittedPathIndex(), ssdConfig: ssdConfig)
        let request2 = Self.conversation([
            Self.user("hi"), Self.assistant("hello world"), Self.user("more"),
        ])
        let turn2 = try await session.turn(request2, text: "again")
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        #expect(turn2.cached == pathLength, turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
        #expect(turn2.leafStore["emittedPath"] == "registered")
        let path2 = try #require(turn2.registeredPathLength)

        let request3 = Self.conversation([
            Self.user("hi"), Self.assistant("hello world"), Self.user("more"),
            Self.assistant("again"), Self.user("again"),
        ])
        let turn3 = try await session.turn(request3, text: "again")
        #expect(turn3.requestResolve["result"] == "hit")
        #expect(turn3.requestResolve["indexedPrefix"] == "\(path2)")
        #expect(turn3.cached == path2, turn3.account)
        #expect(session.index.statsSnapshot().registrations == 2)
    }

    @Test func thinkStrippingTemplateKeepsTheBoundaryPathAtAUserBoundary() async throws {
        let session = Session()
        let request1 = Self.conversation([Self.user("hi")], context: .canonical)
        let turn1 = try await session.turn(request1, context: .canonical)
        #expect(turn1.text == "hello world")
        #expect(turn1.leafStore["path"] == "boundary")
        #expect(turn1.leafStore["source"] == "boundary")
        #expect(turn1.leafStore["emittedPath"] == "skipped")
        #expect(turn1.leafStore["emittedPathSkip"] == "thinkStrippingUserBoundary")
        #expect(turn1.event("emittedPathRegister") == nil)

        // The next user message re-renders the turn stripped; the boundary
        // leaf serves it as before, the index has nothing to say.
        let request2 = Self.conversation(
            [Self.user("hi"), Self.assistant("hello world"), Self.user("more")], context: .canonical
        )
        let turn2 = try await session.turn(request2, text: "again", context: .canonical)
        #expect(turn2.text == "again")
        #expect(turn2.requestResolve["result"] == "miss")
        #expect(turn2.requestResolve["reason"] == "noEntry")
        #expect(turn2.cached > 0, turn2.account)
        let stripped = try turn1.storedRender(appending: "hello world", context: .canonical)
        #expect(turn2.cached <= stripped.count)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
        #expect(session.index.statsSnapshot().registrations == 0)
    }

    // MARK: - Harness

    /// One toy session: the module, its private index, the completion
    /// queue and the telemetry capture, plus the turn helper every case
    /// drives.
    @MainActor
    final class Session {
        let tokenizer: EmittedPathToyTokenizer
        let queue: ToyCompletionQueue
        let provider: ToyModelSessionProvider
        let fingerprint: String
        let modelID: String
        let identity: ModelIdentity?
        let capture: TelemetryCapture
        private(set) var index: EmittedPathIndex
        private(set) var fixture: ServerCompletionFixture

        init(
            index: EmittedPathIndex = EmittedPathIndex(),
            fault: ((EmittedPathToyTokenizer) -> FaultyStreamTokenizer)? = nil,
            vision: ToyUserInputProcessor.VisionStub? = nil,
            identity: ModelIdentity? = nil,
            ssdConfig: SSDPrefixCacheConfig? = nil,
            hasMTPDrafter: Bool = false
        ) {
            let uuid = UUID().uuidString
            let fingerprint = "toy-emitted-path-\(uuid)"
            let modelID = "toy/emitted-path/\(uuid)"
            let tokenizer = EmittedPathToyTokenizer()
            let queue = ToyCompletionQueue(
                generationPrompts: tokenizer.generationPrompts, eosTokenId: tokenizer.endOfTurnID)
            var configuration = ModelConfiguration(id: modelID)
            configuration.eosTokenIds = [tokenizer.endOfTurnID]
            let streamTokenizer: any Tokenizer = fault.map { $0(tokenizer) } ?? tokenizer
            let provider = ToyModelSessionProvider(
                model: ToyLanguageModel(completions: queue),
                tokenizer: streamTokenizer,
                configuration: configuration,
                vision: vision,
                reportsFlatTextTokens: vision == nil,
                anchorsVision: vision != nil,
                hasMTPDrafter: hasMTPDrafter)
            self.tokenizer = tokenizer
            self.queue = queue
            self.provider = provider
            self.fingerprint = fingerprint
            self.modelID = modelID
            self.identity = identity
            self.index = index
            self.capture = TelemetryCapture(modelID: modelID)
            self.fixture = Self.makeFixture(
                provider: provider, fingerprint: fingerprint, identity: identity, index: index,
                ssdConfig: ssdConfig, modelID: modelID)
        }

        deinit { capture.stop() }

        /// A fresh module — empty RAM tier — over the same toy and the given index.
        func restart(index: EmittedPathIndex, ssdConfig: SSDPrefixCacheConfig? = nil) {
            self.index = index
            fixture = Self.makeFixture(
                provider: provider, fingerprint: fingerprint, identity: identity, index: index,
                ssdConfig: ssdConfig, modelID: modelID)
        }

        private static func makeFixture(
            provider: ToyModelSessionProvider, fingerprint: String, identity: ModelIdentity?,
            index: EmittedPathIndex, ssdConfig: SSDPrefixCacheConfig?, modelID: String
        ) -> ServerCompletionFixture {
            ServerCompletionFixture(
                provider: provider, fingerprint: fingerprint, ssdConfig: ssdConfig,
                identity: identity, promptStartsThinking: true, emittedPathIndex: index,
                modelID: modelID)
        }

        /// The turn's stable prefix — the render up to the last user
        /// message's content, where the Leaf Store plants its `system`
        /// checkpoint: what the render shares with any other last message.
        func stablePrefix(of turn: Turn) throws -> Int {
            let other = EmittedPathSynthesizedReplayTests.conversation(
                [EmittedPathSynthesizedReplayTests.user("probe")], context: turn.context)
            return EmittedPathSynthesizedReplayTests.commonPrefix(
                turn.render, try render(other, context: turn.context))
        }

        /// The ids the toy emits for a turn: the reasoning that closes the
        /// generation prompt's open think block, then the text.
        func completion(thinking: String?, text: String) -> [Int] {
            let emitted = thinking.map { "\($0)\n</think>\n\n" + text } ?? text
            return tokenizer.encode(text: emitted, addSpecialTokens: false)
        }

        func render(
            _ conversation: HTTPPrefixCacheConversation,
            context: TemplateRenderContext = EmittedPathSynthesizedReplayTests.preserving,
            generationPrompt: Bool = true
        ) throws -> [Int] {
            let extra: [String: any Sendable]? =
                generationPrompt ? nil : ["add_generation_prompt": false]
            return try tokenizer.applyChatTemplate(
                messages: conversation.promptMessages, tools: nil,
                additionalContext: context.additionalContext(merging: extra))
        }

        func turn(
            _ conversation: HTTPPrefixCacheConversation,
            thinking: String? = "plan",
            text: String = "hello world",
            context: TemplateRenderContext = EmittedPathSynthesizedReplayTests.preserving
        ) async throws -> Turn {
            try await turn(
                conversation, generated: completion(thinking: thinking, text: text),
                context: context)
        }

        func turn(
            _ conversation: HTTPPrefixCacheConversation,
            generated: [Int],
            context: TemplateRenderContext = EmittedPathSynthesizedReplayTests.preserving,
            parameters: AgentGenerateParameters? = nil
        ) async throws -> Turn {
            queue.enqueue(generated)
            _ = queue.drainFeeds()
            _ = capture.drain()
            _ = capture.drainLines()
            let handle = try await fixture.start(
                conversation: conversation,
                parameters: parameters ?? EmittedPathSynthesizedReplayTests.parameters(),
                renderContext: context)
            var text = ""
            var thinking = ""
            for try await event in handle.stream {
                switch event {
                case .text(let chunk): text += chunk
                case .thinking(let chunk): thinking += chunk
                default: break
                }
            }
            // The stream has ended: the prompt and every decode step — the
            // stop token included, which the iterator forwards before the
            // stop check — are on the tape; the Leaf Store's own forwards
            // (a boundary residual, a speculative seed) come after.
            let feeds = queue.drainFeeds()
            await handle.waitForCompletion()
            let (fedPrompt, fedGenerated) = Self.split(
                feeds, generated: generated, eos: queue.eosTokenId)
            return Turn(
                conversation: conversation, context: context,
                text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                thinking: thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                cached: handle.cachedTokenCount, diagnostics: handle.diagnostics,
                render: try render(conversation, context: context),
                storedRender: try render(conversation, context: context, generationPrompt: false),
                feeds: feeds, fedPrompt: fedPrompt, fedGenerated: fedGenerated,
                tailFeeds: queue.drainFeeds(), events: capture.drain(),
                lines: capture.drainLines(), session: self)
        }

        /// The request's own run on the tape: the generated ids (plus the
        /// fed stop token) locate the decode steps; the prompt is the
        /// contiguous run of positions feeding straight into them.
        private static func split(
            _ feeds: [(position: Int, id: Int)], generated: [Int], eos: Int
        ) -> (prompt: [Int], generated: [Int]) {
            let decode = generated + [eos]
            let ids = feeds.map(\.id)
            guard decode.count <= ids.count,
                let start = (0...(ids.count - decode.count)).last(where: { index in
                    ids[index..<(index + decode.count)].elementsEqual(decode)
                })
            else { return ([], []) }
            var promptStart = start
            while promptStart > 0,
                feeds[promptStart - 1].position == feeds[promptStart].position - 1
            {
                promptStart -= 1
            }
            return (Array(ids[promptStart..<start]), decode)
        }
    }

    /// One request's account.
    nonisolated struct Turn {
        let conversation: HTTPPrefixCacheConversation
        let context: TemplateRenderContext
        let text: String
        let thinking: String
        /// The handle's restored offset.
        let cached: Int
        let diagnostics: HTTPServerGenerationStart.Diagnostics
        /// The canonical request render (generation prompt included).
        let render: [Int]
        /// The canonical render without the generation prompt.
        let storedRender: [Int]
        /// Everything fed while the stream ran, in feed order.
        let feeds: [(position: Int, id: Int)]
        /// What the request prefilled, in order.
        let fedPrompt: [Int]
        /// The model's own tokens fed back during decode, the stop token last.
        let fedGenerated: [Int]
        /// What the Leaf Store fed after the stream ended.
        let tailFeeds: [(position: Int, id: Int)]
        let events: [PromptCacheTelemetryEvent]
        /// The diagnostics lines as logged, each with its level.
        let lines: [(line: String, level: PrefixCacheDiagnostics.Level)]
        let session: Session

        /// The live path the turn's leaf was captured under: the prompt and
        /// every fed id, the stop token included.
        var livePath: [Int] { render + fedGenerated }

        /// The tape and the lookup, for a failure to read.
        var account: Comment {
            Comment(
                rawValue: "cached=\(cached) reason=\(diagnostics.cacheReason) "
                    + "shared=\(diagnostics.sharedPrefixLength) "
                    + "feeds=\(feeds.map { "\($0.position):\($0.id)" }.joined(separator: " ")) "
                    + "tail=\(tailFeeds.map { "\($0.position):\($0.id)" }.joined(separator: " ")) "
                    + "leafStore=\(leafStore)")
        }

        func event(_ name: String) -> PromptCacheTelemetryEvent? {
            events.first { $0.eventName == name }
        }
        /// The level the event's line was logged at.
        func level(of event: PromptCacheTelemetryEvent) -> PrefixCacheDiagnostics.Level? {
            guard let requestID = event.requestID?.uuidString else { return nil }
            return lines.first {
                $0.line.contains("event=\(event.eventName) ")
                    && $0.line.contains("requestID=\(requestID) ")
            }?.level
        }
        var leafStore: [String: String] { event("leafStore").map(fields) ?? [:] }
        var register: [String: String] { event("emittedPathRegister").map(fields) ?? [:] }
        var registeredPathLength: Int? { leafStore["emittedPathLength"].flatMap(Int.init) }
        /// The request-edge resolve, empty when the edge never consulted the index.
        var requestResolve: [String: String] {
            events.first {
                $0.eventName == "emittedPathResolve" && fields($0)["spelling"] == "request"
            }.map(fields) ?? [:]
        }
        var resolveSkips: [String] {
            events.filter { $0.eventName == "skip" && fields($0)["stage"] == "emittedPathResolve" }
                .compactMap { fields($0)["reason"] }
        }

        /// The stored render of this conversation plus the assistant turn.
        @MainActor
        func storedRender(
            appending text: String, reasoning: String = "plan",
            context: TemplateRenderContext = EmittedPathSynthesizedReplayTests.preserving
        ) throws -> [Int] {
            try session.render(
                conversation.appendingAssistant(.assistant(content: text, reasoning: reasoning)),
                context: context, generationPrompt: false)
        }
    }

    nonisolated static func fields(_ event: PromptCacheTelemetryEvent) -> [String: String] {
        Dictionary(event.fields.map { ($0.key, $0.value) }, uniquingKeysWith: { first, _ in first })
    }

    nonisolated static func commonPrefix(_ lhs: [Int], _ rhs: [Int]) -> Int {
        zip(lhs, rhs).prefix { $0 == $1 }.count
    }

    nonisolated static func conversation(
        _ messages: [HTTPPrefixCacheMessage], context: TemplateRenderContext = preserving
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(
            systemPrompt: "sys", messages: messages, templateContextDigest: context.digest)
    }

    nonisolated static func user(_ content: String) -> HTTPPrefixCacheMessage {
        HTTPPrefixCacheMessage(role: .user, content: content)
    }

    nonisolated static func assistant(
        _ content: String, reasoning: String = "plan"
    ) -> HTTPPrefixCacheMessage {
        .assistant(content: content, reasoning: reasoning)
    }

    /// A real 1×1 PNG: the keyed path proves attachments `CIImage`-decodable.
    private static func tinyPNG() throws -> Data {
        let context = try #require(
            CGContext(
                data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
        return data as Data
    }
}

/// Collects the structured diagnostics events one model's requests emit
/// (`PrefixCacheDiagnostics.addTelemetrySink`), across isolations.
nonisolated final class TelemetryCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [PromptCacheTelemetryEvent] = []
    private var lines: [(line: String, level: PrefixCacheDiagnostics.Level)] = []
    private var handle: PrefixCacheDiagnostics.TelemetrySinkHandle?
    private var lineHandle: PrefixCacheDiagnostics.TestSinkHandle?

    init(modelID: String) {
        handle = PrefixCacheDiagnostics.addTelemetrySink { [weak self] event in
            guard event.modelID == modelID, let self else { return }
            self.lock.withLock { self.events.append(event) }
        }
        // The rendered lines carry what the events do not: the level.
        lineHandle = PrefixCacheDiagnostics.addTestSink(withLevel: { [weak self] line, level in
            guard line.contains(" modelID=\(modelID) "), let self else { return }
            self.lock.withLock { self.lines.append((line, level)) }
        })
    }

    /// The events since the last drain.
    func drain() -> [PromptCacheTelemetryEvent] {
        lock.withLock {
            let drained = events
            events.removeAll()
            return drained
        }
    }

    /// The logged lines since the last drain.
    func drainLines() -> [(line: String, level: PrefixCacheDiagnostics.Level)] {
        lock.withLock {
            let drained = lines
            lines.removeAll()
            return drained
        }
    }

    func stop() {
        if let handle { PrefixCacheDiagnostics.removeTelemetrySink(handle) }
        if let lineHandle { PrefixCacheDiagnostics.removeTestSink(lineHandle) }
        handle = nil
        lineHandle = nil
    }
}
