import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Every consumer of a request's **Generation Prompt** through the real
/// Server Completion module on the toy Model Session (ADR-0070): the stream
/// parser's start, the leaf-store mode, MTP engagement and the last-message
/// boundary. Each prompt is measured from the toy's own template, so a test
/// fails when a call site feeds the wrong value, not only when a pure
/// function is wrong.
@MainActor
struct ServerCompletionGenerationPromptTests {

    typealias Replay = EmittedPathSynthesizedReplayTests

    /// A `qwen3_5` identity with a full-attention scratch profile, so a
    /// loaded MTP drafter is eligible wherever the predicted mode allows it.
    /// The toy drafter traps if the arm engages: a turn that completes under
    /// it proves MTP stayed off.
    nonisolated static let mtpEligibleIdentity = ModelIdentity(
        configJSON: [
            "model_type": "qwen3_5",
            "text_config": [
                "num_attention_heads": 4, "full_attention_interval": 4, "dtype": "float16",
            ] as [String: Any],
        ],
        chatTemplate: nil)

    static func parameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters()
        parameters.temperature = 0
        parameters.kvBits = nil
        return parameters
    }

    static func context(enableThinking: Bool, preserveThinking: Bool) -> TemplateRenderContext {
        var kwargs: [TemplateRenderFlag: Bool] = [:]
        if !enableThinking { kwargs[.enableThinking] = false }
        if preserveThinking { kwargs[.preserveThinking] = true }
        return TemplateRenderContext(kwargs: kwargs, preservesThinking: preserveThinking)
    }

    // MARK: - The thinking-off stop turn (#563's open finding)

    /// A thinking-off stop turn under the think-stripping render closes an
    /// empty think block the template drops from history once the next user
    /// message arrives. It takes the canonical user leaf, as a thinking turn
    /// does, and the next user turn restores it. Keyed on thinking start
    /// alone, the turn took the direct leaf under the fed path, closed block
    /// included, which the next render never reproduces: with recurrent
    /// state that leaf cannot be cut short, so the next turn re-prefilled
    /// from the system checkpoint.
    @Test func thinkingOffStopTurnStoresTheLeafTheNextUserTurnHits() async throws {
        let off = Self.context(enableThinking: false, preserveThinking: false)
        let session = Replay.Session(recurrentElements: 3)
        let turn1 = try await session.turn(
            Replay.conversation([Replay.user("hi")], context: off), thinking: nil, context: off)
        #expect(turn1.text == "hello world")
        #expect(turn1.thinking.isEmpty)
        #expect(turn1.leafStore["generationPrompt"] == "closed", turn1.account)
        #expect(turn1.leafStore["mode"] == HTTPLeafStoreMode.canonicalUserLeaf.rawValue)
        #expect(turn1.leafStore["path"] == "boundary", turn1.account)

        let request2 = Replay.conversation(
            [
                Replay.user("hi"), Replay.assistant("hello world", reasoning: ""),
                Replay.user("more"),
            ],
            context: off)
        let turn2 = try await session.turn(request2, thinking: nil, text: "again", context: off)
        #expect(turn2.text == "again")
        // The canonical leaf ends where the stored render and the next user
        // turn's render part: right after the assistant header.
        let leafOffset = try #require(turn1.leafStore["leafOffset"].flatMap(Int.init))
        #expect(leafOffset == Replay.commonPrefix(turn1.render, turn2.render), turn1.account)
        #expect(turn2.cached >= leafOffset, turn2.account)
        #expect(turn2.fedPrompt == Array(turn2.render[turn2.cached...]), turn2.account)
    }

    // MARK: - The stop-turn table

    struct StopTurnCase: Sendable, CustomTestStringConvertible {
        let enableThinking: Bool
        let preserveThinking: Bool
        let generationPrompt: String
        let mode: HTTPLeafStoreMode
        var testDescription: String {
            "thinking \(enableThinking ? "on" : "off"), "
                + (preserveThinking ? "preserve-thinking" : "think-stripping")
        }
    }

    /// A stop turn after a tool result, cold, on the Qwen3.8-shaped toy: its
    /// prompt's think block, the mode the rule gives it, and the
    /// last-message boundary at the key path minus the measured prompt,
    /// past the last-user boundary. Wherever the predicted mode is not the
    /// direct leaf an MTP drafter is loaded and must stay off, which the
    /// thinking-off think-stripping row used to break.
    @Test(arguments: [
        StopTurnCase(
            enableThinking: true, preserveThinking: false, generationPrompt: "opens",
            mode: .canonicalUserLeaf),
        StopTurnCase(
            enableThinking: false, preserveThinking: false, generationPrompt: "closed",
            mode: .canonicalUserLeaf),
        StopTurnCase(
            enableThinking: true, preserveThinking: true, generationPrompt: "opens",
            mode: .canonicalUserLeaf),
        StopTurnCase(
            enableThinking: false, preserveThinking: true, generationPrompt: "closed",
            mode: .directLeaf),
    ])
    func stopTurnAfterAToolResult(_ testCase: StopTurnCase) async throws {
        let context = Self.context(
            enableThinking: testCase.enableThinking, preserveThinking: testCase.preserveThinking)
        let mtp = testCase.mode != .directLeaf
        let session = Replay.Session(
            identity: mtp ? Self.mtpEligibleIdentity : nil, hasMTPDrafter: mtp)
        let conversation = Replay.conversation(
            [
                Replay.user("hi"), Replay.assistant("calling", reasoning: ""),
                HTTPPrefixCacheMessage(role: .tool, content: "result"),
            ], context: context)
        let turn = try await session.turn(
            conversation, thinking: testCase.enableThinking ? "plan" : nil, context: context)
        #expect(turn.text == "hello world")
        #expect(turn.thinking == (testCase.enableThinking ? "plan" : ""))
        #expect(turn.leafStore["generationPrompt"] == testCase.generationPrompt, turn.account)
        #expect(turn.leafStore["mode"] == testCase.mode.rawValue, turn.account)

        let prefilled = try #require(
            turn.events.first {
                $0.eventName == "requestMemory" && $0.field("phase") == "prefilled"
                    && $0.field("boundaryCheckpointOffsets") != nil
            })
        let offsets = (prefilled.field("boundaryCheckpointOffsets") ?? "")
            .split(separator: ",").compactMap { Int($0) }
        if testCase.preserveThinking {
            // A text-only preserve-thinking turn keeps no boundary helpers.
            #expect(offsets.isEmpty)
        } else {
            let prompt = session.tokenizer.generationPrompts[testCase.enableThinking ? 0 : 1]
            #expect(offsets.count == 2, turn.account)
            #expect(offsets.max() == turn.render.count - prompt.count, turn.account)
        }
    }

    // MARK: - The unknown state

    /// A template whose prompt names the message count measures on the
    /// one-message probe, but a longer request never feeds that prompt, so
    /// its Generation Prompt is `unknown(notFed)` and every consumer takes
    /// its own answer: the parser starts outside a think block (reasoning
    /// streams as content, the stray `</think>` reclassifies nothing
    /// already emitted), the stop turn takes the canonical user leaf, MTP
    /// stays off, no last-message boundary is placed, and the request logs
    /// why.
    @Test func unknownGenerationPromptTakesEveryConsumersUnknownAnswer() async throws {
        let tokenizer = TemplateShapeTokenizer(.conversationDependent)
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: "sys",
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: "question"),
                .assistant(content: "earlier"),
                HTTPPrefixCacheMessage(role: .user, content: "again"),
            ])
        let prompt = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let lastUser = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil,
            additionalContext: ["add_generation_prompt": false])
        let modelID = "unknown-generation-prompt-\(UUID())"
        let telemetry = TelemetryCapture(modelID: modelID)
        defer { telemetry.stop() }
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: prompt + Array("thought</think>ok".utf8).map(Int.init)),
                tokenizer: tokenizer, hasMTPDrafter: true),
            identity: Self.mtpEligibleIdentity, modelID: modelID)

        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters())
        var text = ""
        var thinking = ""
        for try await event in handle.stream {
            switch event {
            case .text(let chunk): text += chunk
            case .thinking(let chunk): thinking += chunk
            default: break
            }
        }
        await handle.waitForCompletion()
        #expect(text == "thoughtok")
        #expect(thinking.isEmpty)

        let events = telemetry.drain()
        func skip(_ stage: String) -> PromptCacheTelemetryEvent? {
            events.first { $0.eventName == "skip" && $0.field("stage") == stage }
        }
        #expect(skip("generationPrompt")?.field("reason") == "notFed")
        #expect(skip("lastMessageBoundary")?.field("generationPrompt") == "notFed")
        let leafStore = try #require(events.first { $0.eventName == "leafStore" })
        #expect(leafStore.field("generationPrompt") == "unknown(notFed)")
        #expect(leafStore.field("mode") == HTTPLeafStoreMode.canonicalUserLeaf.rawValue)
        let prefilled = try #require(
            events.first {
                $0.eventName == "requestMemory" && $0.field("phase") == "prefilled"
                    && $0.field("boundaryCheckpointOffsets") != nil
            })
        // Only the last-user boundary: the prompt's tokens are unknown.
        #expect(prefilled.field("boundaryCheckpointOffsets") == "\(lastUser.count)")
        await fixture.drain()
    }

    /// The parser under an unknown prompt, as the spec documents it: it
    /// starts outside a think block, so a first chunk of reasoning streams
    /// as content, and a `</think>` in the next chunk reclassifies only what
    /// is still buffered, which here is nothing.
    @Test func unknownPromptStreamsReasoningAsContentUntilTheCloseTag() {
        let unknown = measuredGenerationPrompt(TemplateShapeTokenizer(.mergesAcrossAppend))
        #expect(unknown.unknownReason == .unstableSplit)
        let parser = ToolCallParser(generationPrompt: unknown)
        var accumulator = GenerationAccumulator()
        for chunk in ["reasoning", "</think>answer"] {
            for event in parser.processChunk(chunk) {
                accumulator.ingest(AgentGeneration(parserEvent: event))
            }
        }
        for event in parser.finalize() { accumulator.ingest(AgentGeneration(parserEvent: event)) }
        #expect(accumulator.text == "reasoninganswer")
        #expect(accumulator.thinking == nil || accumulator.thinking?.isEmpty == true)
    }

    // MARK: - The Unkeyed Completion

    /// An Unkeyed Completion carries its Generation Prompt too, checked
    /// against its prompt tokens, so on a thinking template its reasoning
    /// streams as thinking. Here a vision container returns no grids for
    /// the request's image, so the key space cannot be built.
    @Test func unkeyedCompletionOnAThinkingTemplateStreamsReasoningAsThinking() async throws {
        let tokenizer = ToySequencingTokenizer(thinking: true)
        let conversation = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: "Hi",
                    images: [HTTPPrefixCacheImage(data: ImageTestFixtures.tinyPNGData)])
            ])
        let prompt = try tokenizer.applyChatTemplate(
            messages: conversation.promptMessages, tools: nil, additionalContext: nil)
        let fixture = ServerCompletionFixture(
            provider: ToyModelSessionProvider(
                model: ToyLanguageModel(
                    script: prompt + Array("thought</think>ok".utf8).map(Int.init)),
                tokenizer: tokenizer))
        let handle = try await fixture.start(
            conversation: conversation, parameters: Self.parameters())
        #expect(handle.diagnostics.cacheReason.hasPrefix("unkeyed"))
        var text = ""
        var thinking = ""
        for try await event in handle.stream {
            switch event {
            case .text(let chunk): text += chunk
            case .thinking(let chunk): thinking += chunk
            default: break
            }
        }
        await handle.waitForCompletion()
        #expect(thinking == "thought")
        #expect(text == "ok")
        await fixture.drain()
    }
}
