import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// A minimal chat-template tokenizer whose renders are *row-consistent* with
/// what generation physically leaves in the KV cache: every message renders
/// as `[roleMark] + content-bytes + [EOT]`, the generation prompt is the
/// assistant's own role mark, and EOT doubles as the EOS the model emits.
/// A stored render (prompt + assistant turn, no generation prompt) is then
/// token-identical to the rows the drive produced — prompt, completion
/// bytes, forwarded EOS — so leaf offsets line up exactly, the property the
/// real ChatML template + real tokenizer pair provides in production.
nonisolated struct ToySequencingTokenizer: Tokenizer {
    static let eotTokenId = 300
    /// Assistant role mark == generation prompt: opening the assistant turn
    /// in history renders the same token generation started from.
    static let assistantMarkTokenId = 301
    static let systemMarkTokenId = 310
    static let userMarkTokenId = 311
    static let toolMarkTokenId = 313

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        // Lossy byte decode; marker ids (≥ 256) are dropped like specials.
        // swiftlint:disable:next optional_data_string_conversion
        String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
    }
    func tokenize(text: String) -> [String] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eot>" }
    var eosTokenId: Int? { Self.eotTokenId }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    private func roleMark(_ role: String) -> Int {
        switch role {
        case "assistant": Self.assistantMarkTokenId
        case "system": Self.systemMarkTokenId
        case "tool": Self.toolMarkTokenId
        default: Self.userMarkTokenId
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            tokens.append(roleMark(message["role"] as? String ?? "user"))
            // Image-bearing messages arrive in the content-array form; the
            // toy renders their text parts (images ride only through the
            // processor's prepared tokens, as in production).
            if let text = message["content"] as? String {
                tokens += encode(text: text, addSpecialTokens: false)
            } else if let parts = message["content"] as? [[String: any Sendable]] {
                for part in parts where part["type"] as? String == "text" {
                    tokens += encode(text: part["text"] as? String ?? "", addSpecialTokens: false)
                }
            }
            tokens.append(Self.eotTokenId)
        }
        let addGenerationPrompt = (additionalContext?["add_generation_prompt"] as? Bool) ?? true
        if addGenerationPrompt {
            tokens.append(Self.assistantMarkTokenId)
        }
        return tokens
    }
}

/// PR B gating coverage (PRD #137, ADR-0016): the keyed spine — Snapshot
/// Resolution → restore → suffix prefill → drive → leaf capture → Snapshot
/// Admission — and the cancellation orderings, all through the module's
/// public entry with the toy-model-backed **Model Session**.
@Suite struct ServerCompletionKeyedSequencingTests {

    @MainActor
    @Test(arguments: [0, 3])
    func thinkStrippingTurnRetainsOnlyWholeStateBoundaryBytes(recurrentElements: Int) async throws {
        var tokenizer = FakeChatMLTokenizer()
        tokenizer.stripsThinkBeforeLastUser = true
        let conversation = Self.conversation([.init(role: .user, content: "question")])
        let prompt = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let modelID = "boundary-memory-\(UUID())"
        let telemetry = TelemetryCapture(modelID: modelID)
        defer { telemetry.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: prompt + Array("thought</think>ok".utf8).map(Int.init),
                    recurrentElements: recurrentElements),
                tokenizer: tokenizer),
            promptStartsThinking: true, modelID: modelID)
        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters())
        #expect(try await collectServerText(handle).text == "ok")
        let events = telemetry.drain()
        let prefilled = try #require(
            events.first {
                $0.eventName == "requestMemory" && $0.field("phase") == "prefilled"
                    && $0.field("boundaryCheckpointArrayBytes") != nil
            })
        #expect(prefilled.field("boundaryCheckpointCount") == "1")
        // The hybrid toy owns three float32 recurrent values (12 bytes);
        // the attention-only toy owns none. Neither boundary owns KV rows.
        #expect(
            prefilled.field("boundaryCheckpointArrayBytes") == (recurrentElements == 0 ? "0" : "12")
        )
        // No system/older checkpoint exists: this turn's checked-in leaf
        // must back the transient view to build the canonical leaf.
        let canonical = try #require(
            events.last {
                $0.eventName == "capture" && $0.field("source") == "canonicalLeaf"
            })
        #expect(canonical.intField("offset") ?? 0 > 0)
        #expect(events.last { $0.eventName == "leafStore" }?.field("path") == "boundary")
        // The live leaf checked in to back the transient views is released
        // once the canonical leaf can back them: one resident leaf per turn.
        let backer = try #require(
            events.first {
                $0.eventName == "capture" && $0.field("source") == "boundaryBackingLeaf"
            })
        let released = try #require(
            events.first { $0.eventName == "leafSupersession" && $0.field("mode") == "released" })
        #expect(released.intField("offset") == backer.intField("offset"))
        let stats = try #require(fixture.cacheAdmin.stats)
        #expect(stats.snapshotsByType[.leaf] == 1)
        #expect(stats.snapshotsByType[.branchPoint, default: 0] == 0)
        await fixture.drain()
    }

    @MainActor
    @Test(arguments: [false, true])
    func plannedBranchViewReportsCaptureAndLookupThroughTheModelSession(ssdEnabled: Bool)
        async throws
    {
        let tokenizer = ToySequencingTokenizer()
        let completions = ToyCompletionQueue(
            generationPrompts: [[ToySequencingTokenizer.assistantMarkTokenId]],
            eosTokenId: ToySequencingTokenizer.eotTokenId)
        let modelID = "view-sequencing-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "view-ssd-turn-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(completions: completions), tokenizer: tokenizer),
            fingerprint: ssdEnabled ? String(repeating: "d", count: 64) : nil,
            ssdConfig: ssdEnabled
                ? .init(
                    enabled: true, rootURL: root, budgetBytes: 1_000_000, maxPendingBytes: 1_000_000
                ) : nil,
            modelID: modelID)
        for text in ["abcd-one", "abcd-two"] {
            completions.enqueue(Array("ok".utf8).map(Int.init))
            let handle = try await fixture.start(
                conversation: Self.conversation([.init(role: .user, content: text)]),
                parameters: Self.parameters())
            #expect(try await collectServerText(handle).text == "ok")
        }
        let events = capture.drain()
        let branch = try #require(
            events.last {
                $0.eventName == "capture" && $0.field("checkpointType") == "branchPoint"
            })
        #expect(branch.field("checkpointKind") == "prefixView")
        #expect(branch.field("bytes") == "0")
        completions.enqueue(Array("ok".utf8).map(Int.init))
        let fork = try await fixture.start(
            conversation: Self.conversation([.init(role: .user, content: "abcd-three")]),
            parameters: Self.parameters())
        #expect(try await collectServerText(fork).text == "ok")
        let lookup = try #require(capture.drain().first { $0.eventName == "lookup" })
        #expect(lookup.field("source") == "view")
        #expect(lookup.field("backingLeafForm") == "ownedBody")
        #expect(lookup.field("restoreMode") == "copy")
        #expect(lookup.field("copyReason") == "checkpoint")
        let backingOffset = try #require(lookup.field("backingLeafOffset").flatMap(Int.init))
        #expect(backingOffset > fork.cachedTokenCount)
        if ssdEnabled {
            // The second fork proves reuse. Its successful-turn tail must
            // fulfil the deferred view intent after checking in the leaf.
            completions.enqueue(Array("ok".utf8).map(Int.init))
            let next = try await fixture.start(
                conversation: Self.conversation([.init(role: .user, content: "abcd-four")]),
                parameters: Self.parameters())
            #expect(try await collectServerText(next).text == "ok")
            await fixture.flush()
            let manifest = try JSONDecoder().decode(
                SnapshotManifest.self,
                from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
            #expect(manifest.snapshots.values.contains { $0.checkpointType == "branchPoint" })
        }
        await fixture.drain()
    }

    @MainActor
    @Test func canonicalFallbackRestoresAPlannedBranchView() async throws {
        let tokenizer = ToySequencingTokenizer()
        let completions = ToyCompletionQueue(
            generationPrompts: [[ToySequencingTokenizer.assistantMarkTokenId]],
            eosTokenId: ToySequencingTokenizer.eotTokenId)
        let modelID = "canonical-view-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(completions: completions), tokenizer: tokenizer),
            promptStartsThinking: true, modelID: modelID)
        let preserving = TemplateRenderContext(
            kwargs: [.preserveThinking: true], preservesThinking: true)
        // Seed a full leaf on the preserving fast path. Assistant-only
        // histories have no transient last-user boundary, so canonical
        // reconstruction must resolve the planned branch checkpoint.
        for (index, text) in ["abcd-one", "abcd-two", "abcd-three"].enumerated() {
            completions.enqueue(Array("</think>ok".utf8).map(Int.init))
            _ = capture.drain()
            let handle = try await fixture.start(
                conversation: Self.conversation([.assistant(content: text)]),
                parameters: Self.parameters(),
                renderContext: index == 0 ? preserving : .canonical)
            #expect(try await collectServerText(handle).text == "ok")
            let events = capture.drain()
            if index == 2 {
                #expect(events.first { $0.eventName == "lookup" }?.field("source") == "view")
                #expect(events.last { $0.eventName == "leafStore" }?.field("path") == "boundary")
                // The re-prefilled cache is request-private, so the leaf
                // moves it in rather than deep-copying it.
                #expect(events.last { $0.eventName == "leafStore" }?.field("source") == "handoff")
                #expect(!events.contains { $0.field("reason") == "prefill-threw" })
            }
        }
        await fixture.drain()
    }

    @MainActor
    @Test(arguments: [nil, 4] as [Int?])
    func creationAndRestoreReservePromptRowsWithoutReservingOutput(kvBits: Int?) async throws {
        let tokenizer = ToySequencingTokenizer()
        let first = Self.conversation([
            .init(role: .user, content: String(repeating: "a", count: 998))
        ])
        let next = Self.conversation([
            .init(role: .user, content: String(repeating: "a", count: 998)),
            .assistant(content: "b"),
            .init(role: .user, content: String(repeating: "c", count: 994)),
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: next.promptMessages, tools: nil, additionalContext: nil)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: render + [100], headDim: 32), tokenizer: tokenizer))
        var parameters = await Self.parameters(kvBits: kvBits)
        parameters.prefillStepSize = 64
        parameters.maxTokens = 8192
        let cold = try await fixture.start(conversation: first, parameters: parameters)
        #expect(try await collectServerText(cold).text == "b")
        let coldCapacity = try #require(fixture.provider.recorder.prefillCapacities.last)
        #expect(coldCapacity == 1024)
        let warm = try await fixture.start(conversation: next, parameters: parameters)
        #expect(warm.cachedTokenCount > 0)
        #expect(try await collectServerText(warm).text == "d")
        let restoredCapacity = try #require(fixture.provider.recorder.prefillCapacities.last)
        #expect(restoredCapacity >= render.count)
        #expect(restoredCapacity < render.count + 256)
        if kvBits != nil { #expect(fixture.provider.recorder.verbs.contains(.restore)) }
        await fixture.drain()
    }

    @MainActor
    @Test func canonicalLeafRestoreReservesTheStoredPath() async throws {
        let tokenizer = FakeParoThinkingTokenizer()
        let conversation = Self.conversation([
            .init(role: .user, content: String(repeating: "a", count: 998))
        ])
        let prompt = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let reply = "reasoning\n</think>\n\n" + String(repeating: "b", count: 1000)
        let model = ToyLanguageModel(script: prompt + Array(reply.utf8).map(Int.init))
        let records = model.capacityRecords
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(model: model, tokenizer: tokenizer),
            promptStartsThinking: true, modelID: "capacity-canonical-\(UUID())")
        var parameters = Self.parameters()
        parameters.prefillStepSize = 64
        let handle = try await fixture.start(conversation: conversation, parameters: parameters)
        _ = try await collectServerText(handle)
        #expect(fixture.provider.recorder.verbs.contains(.restore))
        #expect(fixture.provider.recorder.prefillCapacities.count >= 2)
        // A restore starts a second monotonic run of forward offsets.
        let values = records.values
        let rewind = try #require(
            values.indices.dropFirst().first { values[$0].offset < values[$0 - 1].offset })
        let canonical = Array(values[rewind...])
        let finalOffset = try #require(canonical.last?.offset)
        #expect(canonical.allSatisfy { $0.capacity >= finalOffset })
        #expect(canonical.allSatisfy { $0.capacity < finalOffset + 256 })
        #expect(Set(canonical.map(\.capacity)).count == 1)
        await fixture.drain()
    }

    @Test func toyDecodeUsesGeometricCapacityGrowth() async throws {
        let prompt = [65, 66, 67]
        let completion = Array(repeating: 68, count: 2048)
        let provider = ToyModelSessionProvider(
            model: ToyLanguageModel(script: prompt + completion, layers: 1))
        let capacities = try await provider.withSession { session in
            let parameters = GenerateParameters(maxTokens: 2048, temperature: 0)
            let cache = try session.newCache(parameters: parameters)
            let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
            var iterator = try session.makePreparingDecodeIterator(
                input, cache: cache, parameters: parameters, prepare: nil)
            var capacities = [cache[0].innerState()[0].dim(2)]
            var generated = 0
            while let token = iterator.next(), generated < 2048 {
                #expect(token == 68)
                generated += 1
                let capacity = cache[0].innerState()[0].dim(2)
                if capacity != capacities.last { capacities.append(capacity) }
            }
            #expect(generated == 2048)
            return capacities
        }
        #expect(capacities == [256, 768, 1792, 3840])
    }

    // MARK: - Compaction against the next turn's growth (#554, item 6)

    /// A leaf keeps the capacity its turn grew into when compaction leaves it
    /// alone: a next turn whose suffix fits that capacity prefills and decodes
    /// without reallocating the attention body.
    @MainActor
    @Test func aNextTurnThatFitsTheKeptCapacityDoesNotReallocate() async throws {
        let tokenizer = ToySequencingTokenizer()
        let first = Self.conversation([.init(role: .user, content: "Hi")])
        let next = Self.conversation([
            .init(role: .user, content: "Hi"), .assistant(content: "Hello!"),
            .init(role: .user, content: "More?"),
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: next.promptMessages, tools: nil, additionalContext: nil)
        let model = ToyLanguageModel(script: render + Array("Sure.".utf8).map(Int.init))
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(model: model, tokenizer: tokenizer),
            modelID: "compaction-fits-\(UUID())")
        #expect(
            try await collectServerText(
                try await fixture.start(conversation: first, parameters: Self.parameters())
            ).text == "Hello!")
        let kept = try #require(model.capacityRecords.values.last).capacity
        let before = model.capacityRecords.values.count

        let handle = try await fixture.start(conversation: next, parameters: Self.parameters())
        #expect(handle.cachedTokenCount > 0)
        #expect(try await collectServerText(handle).text == "Sure.")
        let turn = model.capacityRecords.values.dropFirst(before)
        #expect(!turn.isEmpty)
        #expect(
            turn.allSatisfy { $0.capacity == kept },
            "the next turn fits the kept capacity: \(turn.map(\.capacity)) vs \(kept)")
        await fixture.drain()
    }

    /// A cancelled long generation leaves the rewound leaf far more capacity
    /// than its body, and compaction rebuilds it at the offset plus one step.
    /// The resend adds more than that step, so it grows the body once, and
    /// only once: compaction never makes the next turn pay two whole-body
    /// copies.
    @MainActor
    @Test func theNextTurnAfterACompactionGrowsTheBodyAtMostOnce() async throws {
        let tokenizer = ToySequencingTokenizer()
        let first = Self.conversation([.init(role: .user, content: "Hi")])
        let long = Self.conversation([
            .init(role: .user, content: "Hi"), .assistant(content: "Hello!"),
            .init(role: .user, content: String(repeating: "m", count: 300)),
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: long.promptMessages, tools: nil, additionalContext: nil)
        let gate = ForwardGate(threshold: render.count + 300, armed: false)
        let model = ToyLanguageModel(
            script: render + Array(String(repeating: "x", count: 1_000).utf8).map(Int.init),
            onForward: gate.onForward)
        let modelID = "compaction-grows-once-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(model: model, tokenizer: tokenizer),
            modelID: modelID)
        _ = try await collectServerText(
            try await fixture.start(conversation: first, parameters: Self.parameters()))

        // The long generation, cancelled well past the prompt: the rewind
        // returns the first leaf with the rows the decode grew into.
        gate.arm()
        let cancelled = try await fixture.start(conversation: long, parameters: Self.parameters())
        await gate.reached()
        cancelled.cancel()
        gate.open()
        for try await _ in cancelled.stream {}
        await cancelled.waitForCompletion()
        let rewind = try #require(capture.drain().last { $0.eventName == "leafRewind" })
        let compacted = try #require(rewind.intField("compactedBytes"))
        #expect(compacted > 0, "the rewound leaf is compacted")
        // Two toy layers of one 4-wide float32 head, keys and values: 64 B a row.
        let keptRows = try #require(rewind.intField("fullAttentionArrayBytes")) / 64
        let leafOffset = try #require(rewind.intField("offset"))
        #expect(keptRows == leafOffset + 256)

        var short = Self.parameters()
        short.maxTokens = 4
        let before = model.capacityRecords.values.count
        let resend = try await fixture.start(conversation: long, parameters: short)
        #expect(resend.cachedTokenCount == leafOffset)
        _ = try await collectServerText(resend)
        let capacities = model.capacityRecords.values.dropFirst(before).map(\.capacity)
        #expect(!capacities.isEmpty)
        #expect(Set(capacities).count == 1, "one growth, not two: \(capacities)")
        #expect(capacities.allSatisfy { $0 > keptRows && $0 >= render.count })
        await fixture.drain()
    }

    @MainActor
    @Test func emptyDirectTurnReturnsItsLeafWithoutTryingToCaptureTheRewoundCache() async throws {
        let tokenizer = ToySequencingTokenizer()
        let first = Self.conversation([.init(role: .user, content: "Hi")])
        let next = Self.conversation([
            .init(role: .user, content: "Hi"), .assistant(content: "Hello!"),
            .init(role: .user, content: "More?"),
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: next.promptMessages, tools: nil, additionalContext: nil)
        let modelID = "direct-rewind-\(UUID())"
        let capture = TelemetryCapture(modelID: modelID)
        defer { capture.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: render + Array("Sure.".utf8).map(Int.init)),
                tokenizer: tokenizer), modelID: modelID)
        let firstHandle = try await fixture.start(
            conversation: first, parameters: Self.parameters())
        #expect(try await collectServerText(firstHandle).text == "Hello!")
        _ = capture.drain()
        var empty = Self.parameters()
        empty.maxTokens = 0
        let handle = try await fixture.start(conversation: next, parameters: empty)
        #expect(handle.cachedTokenCount > 0)
        #expect(try await collectServerText(handle).text.isEmpty)
        let events = capture.drain()
        let leaf = try #require(events.last { $0.eventName == "leafStore" })
        #expect(leaf.field("source") == "rewind")
        #expect(leaf.field("boundary") == "no-generated-tokens")
        #expect(!events.contains { $0.field("reason") == "no-reusable-cache-state" })
        let terminal = try #require(events.last { $0.field("sampleKind") == "terminal" })
        #expect(terminal.field("treeLeaseCount") == "0")
        let resend = try await fixture.start(conversation: next, parameters: Self.parameters())
        #expect(resend.cachedTokenCount == handle.cachedTokenCount)
        #expect(try await collectServerText(resend).text == "Sure.")
        await fixture.drain()
    }

    private static func conversation(
        _ messages: [HTTPPrefixCacheMessage]
    ) -> HTTPPrefixCacheConversation {
        HTTPPrefixCacheConversation(systemPrompt: nil, messages: messages)
    }

    @MainActor
    private static func parameters(kvBits: Int? = nil) -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        // Quantized cases use headDim 64 so the real quantizer can replace
        // the attention cache objects before decode.
        parameters.kvBits = kvBits
        return parameters
    }

    /// The keyed spine, cold then warm. Round 1 (cold miss) must run
    /// prepare → newCache → chunked prefill → (no-op) quantize → decode
    /// iterator, then hand off the post-generation leaf without a copy.
    /// Round 2 extends the
    /// conversation, must resolve the admitted leaf, restore it *before*
    /// prefilling only the suffix, and decode the scripted continuation —
    /// proving the restored rows landed where the script expects them.
    @Test(arguments: [nil, 4] as [Int?])
    func keyedSpineRestoresAdmittedLeafAndPrefillsOnlyTheSuffix(kvBits: Int?) async throws {
        let tokenizer = ToySequencingTokenizer()
        let round1 = Self.conversation([HTTPPrefixCacheMessage(role: .user, content: "Hi")])
        let round2 = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: "Hi"),
            .assistant(content: "Hello!"),
            HTTPPrefixCacheMessage(role: .user, content: "More?"),
        ])

        // The toy believes in the full two-round transcript: round 2's
        // render (whose prefix is round 1's render, then round 1's
        // completion + EOS) followed by round 2's completion.
        let render1 = try tokenizer.applyChatTemplate(
            messages: round1.promptMessages, tools: nil, additionalContext: nil
        )
        let render2 = try tokenizer.applyChatTemplate(
            messages: round2.promptMessages, tools: nil, additionalContext: nil
        )
        #expect(Array(render2.prefix(render1.count)) == render1)
        let script = render2 + Array("Sure.".utf8).map(Int.init)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script, headDim: 64),
                tokenizer: tokenizer
            )
        )
        let parameters = await Self.parameters(kvBits: kvBits)

        // -- Round 1: cold.
        let handle1 = try await fixture.start(
            conversation: round1, parameters: parameters
        )
        #expect(handle1.cachedTokenCount == 0)
        let (text1, info1) = try await collectServerText(handle1)
        #expect(text1 == "Hello!")
        #expect(try #require(info1).promptTokenCount == render1.count)
        let round1Verbs = fixture.provider.recorder.verbs
        #expect(
            round1Verbs == [
                .prepare, .newCache, .prefill, .quantizeKVCache, .makeDecodeIterator,
            ] + (kvBits == nil ? [] : [.captureSnapshot])
        )

        // Row-consistency: the drive left prompt + completion + forwarded
        // EOS in the cache — exactly the stored render's length.
        let storedTokens1 = try tokenizer.applyChatTemplate(
            messages: round1.appendingAssistant(.assistant(content: "Hello!")).promptMessages,
            tools: nil,
            additionalContext: ["add_generation_prompt": false]
        )
        #expect(storedTokens1.count == render1.count + "Hello!".utf8.count + 1)

        // -- Round 2: warm. Unquantized leaves move; quantized leaves copy.
        // Both must retain the post-decode cache. Quantization replaces cache
        // objects, so retaining the array before that step would lose the
        // generated rows and this exact cached-token count would fail.
        let handle2 = try await fixture.start(
            conversation: round2, parameters: parameters
        )
        #expect(handle2.cachedTokenCount == storedTokens1.count)
        let (text2, info2) = try await collectServerText(handle2)
        #expect(text2 == "Sure.")
        #expect(try #require(info2).promptTokenCount == render2.count)
        let round2Verbs = Array(fixture.provider.recorder.verbs.dropFirst(round1Verbs.count))
        #expect(
            round2Verbs
                == (kvBits == nil
                    ? [
                        .prepare, .prefill, .quantizeKVCache, .makeDecodeIterator,
                    ]
                    : [
                        .prepare, .restore, .prefill, .quantizeKVCache, .makeDecodeIterator,
                        .captureSnapshot,
                    ])
        )

        await fixture.drain()
    }

    /// **Salvage-on-cancel** (issue #97): a cancel landing between prefill
    /// chunks must admit the completed progress as a RAM-only leaf and
    /// release the in-flight start, so a re-sent request resumes from the
    /// salvaged offset instead of the restore floor.
    @Test func cancelMidPrefillSalvagesProgressAndReleasesTheStart() async throws {
        let tokenizer = ToySequencingTokenizer()
        let longContent = String(repeating: "a", count: 4000)
        let conversation = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: longContent)
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let script = render + Array("OK".utf8).map(Int.init)

        // Pause the second prefill chunk (offset 1024, prefillStepSize 1024)
        // so the cancel deterministically lands at the following chunk
        // boundary — past the salvage progress threshold of 2,048 tokens.
        let gate = ForwardGate(threshold: 1024)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script, onForward: gate.onForward),
                tokenizer: tokenizer
            )
        )

        let startTask = Task {
            try await fixture.start(conversation: conversation, parameters: Self.parameters())
        }
        await gate.reached()
        startTask.cancel()
        gate.open()

        guard case .failure(let error) = await startTask.result else {
            Issue.record("cancelled start must throw, not return a handle")
            return
        }
        #expect(error is CancellationError)

        // The registry released the in-flight start: the unload drain has
        // nothing to park on.
        await fixture.drain()

        // The re-sent request resumes from the salvaged chunk boundary and
        // completes the scripted response.
        let retry = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        #expect(retry.cachedTokenCount >= 2048)
        #expect(retry.cachedTokenCount < render.count)
        let (text, _) = try await collectServerText(retry)
        #expect(text == "OK")
        await fixture.drain()
    }

    /// Abort during the drive (decode underway): the stream must end, the
    /// registry slot must be released to the drain, and nothing may be
    /// admitted for the aborted turn — a re-sent request starts cold.
    @Test func abortDuringDriveReleasesSlotAndAdmitsNothing() async throws {
        let tokenizer = ToySequencingTokenizer()
        let conversation = Self.conversation([
            HTTPPrefixCacheMessage(role: .user, content: "Hi")
        ])
        let render = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil
        )
        let completion = String(repeating: "x", count: 64)
        let script = render + Array(completion.utf8).map(Int.init)

        // Pause decode a few tokens in, cancel while generation is live.
        let gate = ForwardGate(threshold: render.count + 8)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(script: script, onForward: gate.onForward),
                tokenizer: tokenizer
            )
        )

        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        await gate.reached()
        handle.cancel()
        gate.open()
        // Consume whatever was emitted; the stream must terminate.
        for try await _ in handle.stream {}
        await handle.waitForCompletion()

        // Slot released — the drain returns instead of parking.
        await fixture.drain()

        // Nothing was admitted for the aborted turn: the re-send is cold and
        // completes the full scripted response.
        let retry = try await fixture.start(
            conversation: conversation, parameters: Self.parameters()
        )
        #expect(retry.cachedTokenCount == 0)
        let (text, _) = try await collectServerText(retry)
        #expect(text == completion)
        await fixture.drain()
    }
}
