import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

//
//  ConversationRenderProbeParityTests.swift
//  tesseractTests
//
//  Ticket #473 of issue #471: the Leaf Admission Builder's future-shared-
//  prefix probe pair and the stable-prefix detector's two-probe used to apply
//  the chat template directly; both now go through the **Conversation
//  Render** module's cache-free probe verbs. Behaviour must not change, so
//  each suite spells the PRE-MODULE computation out by hand (raw
//  `applyChatTemplate`, the same guards) and asserts the module path yields
//  the byte-identical token path — on the fake fixtures here and on the real
//  PARO tokenizer when it is present on disk.
//

// MARK: - Pre-module spellings (the "before")

/// The future-shared-prefix probe pair exactly as `LeafAdmissionBuilder
/// .futureSharedPrefix` computed it before #473: two raw template
/// applications under the merged no-generation-prompt context, LCP, the
/// convergence guards, key-space translation.
private func preModuleFutureSharedPrefix(
    tokenizer: any MLXLMCommon.Tokenizer,
    storedConversation: HTTPPrefixCacheConversation,
    keySpace: CacheKeySpace,
    toolSpecs: [ToolSpec]?,
    renderContext: TemplateRenderContext
) throws -> Result<[Int], CacheKeySpace.TranslationFailure>? {
    let baseMessages = storedConversation.promptMessages
    let probeContext = renderContext.additionalContext(
        merging: ["add_generation_prompt": false]
    )
    let first = try tokenizer.applyChatTemplate(
        messages: baseMessages + [LeafAdmissionBuilder.Continuation.userTurn.probeMessage],
        tools: toolSpecs,
        additionalContext: probeContext
    )
    let second = try tokenizer.applyChatTemplate(
        messages: baseMessages + [LeafAdmissionBuilder.divergentUserProbeMessage],
        tools: toolSpecs,
        additionalContext: probeContext
    )
    let common = zip(first, second).prefix { $0 == $1 }.count
    guard common > 0, common < first.count, common < second.count else {
        return nil
    }
    return keySpace.translate(renderTokens: Array(first[0..<common]))
}

/// The stable-prefix two-probe exactly as `StablePrefixDetector.detect`
/// computed it before #473 (memo aside): two raw system+user renders under
/// the base context, LCP, the full-tokens verification, the ratio guard.
private func preModuleStablePrefix(
    tokenizer: any MLXLMCommon.Tokenizer,
    systemPrompt: String,
    toolSpecs: [ToolSpec]?,
    additionalContext: [String: any Sendable]?,
    fullTokens: [Int]
) throws -> Int? {
    func probe(_ userContent: String) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": userContent],
            ],
            tools: toolSpecs,
            additionalContext: additionalContext
        )
    }
    let probeA = try probe("A_prefix_probe")
    let probeB = try probe("Z_prefix_probe")
    let common = zip(probeA, probeB).prefix(while: ==).count
    guard common > 0,
        fullTokens.count >= common,
        fullTokens[0..<common].elementsEqual(probeA[0..<common]),
        fullTokens.count <= 1000 || common >= fullTokens.count / 3
    else { return nil }
    return common
}

// MARK: - Shared fixtures

private enum ProbeParityFixtures {
    static let systemPrompt = "You are a probe-parity assistant (#473)."

    static let tools: [ToolSpec] = [
        [
            "type": "function",
            "function": [
                "name": "read_file",
                "description": "Read a file.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    /// A stop answer with reasoning behind it — the shape the speculative
    /// pass targets (a user turn follows).
    static let stopTurn = HTTPPrefixCacheConversation(
        systemPrompt: systemPrompt,
        messages: [
            HTTPPrefixCacheMessage(role: .user, content: "What is in a.txt?"),
            HTTPPrefixCacheMessage(
                role: .assistant, content: "I should read it first.",
                reasoning: "The user wants the file contents."),
            HTTPPrefixCacheMessage(role: .user, content: "Go ahead."),
            HTTPPrefixCacheMessage(
                role: .assistant, content: "<think>reading now</think>\nHere it is: hello."),
        ]
    )

    /// A tool stretch: a tool call, its result, and the closing answer.
    static let toolStretch = HTTPPrefixCacheConversation(
        systemPrompt: systemPrompt,
        messages: [
            HTTPPrefixCacheMessage(role: .user, content: "Read a.txt"),
            HTTPPrefixCacheMessage(
                role: .assistant, content: "",
                reasoning: "Need the file.",
                toolCalls: [
                    HTTPPrefixCacheToolCall(
                        name: "read_file", argumentsJSON: "{\"path\": \"a.txt\"}")
                ]),
            HTTPPrefixCacheMessage(role: .tool, content: "hello"),
            HTTPPrefixCacheMessage(role: .assistant, content: "The file says hello."),
        ]
    )

    static let conversations = [stopTurn, toolStretch]

    static let preserveThinking = TemplateRenderContext(flags: [.preserveThinking])
}

// MARK: - Fake-fixture parity

/// `@MainActor` like the detector suites: the detector's memo is
/// process-global, so every suite that runs `detect` serializes on the main
/// actor and never races another suite's `resetMemo()`.
@MainActor
@Suite struct ConversationRenderProbeParityTests {

    /// Every fake template the server suites drive: byte-level ChatML with
    /// and without the think-strip, the PARO-shaped thinking template under
    /// the canonical and preserve-thinking contexts, and the greedy-merge
    /// tokenizer whose pieces span message seams.
    private struct Fixture {
        let name: String
        let tokenizer: any MLXLMCommon.Tokenizer
        let renderContext: TemplateRenderContext
    }

    private static let fixtures: [Fixture] = [
        Fixture(name: "chatml", tokenizer: FakeChatMLTokenizer(), renderContext: .canonical),
        Fixture(
            name: "chatml-strip",
            tokenizer: FakeChatMLTokenizer(stripsThinkBeforeLastUser: true),
            renderContext: .canonical),
        Fixture(name: "paro", tokenizer: FakeParoThinkingTokenizer(), renderContext: .canonical),
        Fixture(
            name: "paro-preserve",
            tokenizer: FakeParoThinkingTokenizer(),
            renderContext: ProbeParityFixtures.preserveThinking),
        Fixture(
            name: "greedy",
            tokenizer: GreedyTokenizer(pieces: [
                "<|im_start|>", "<|im_end|>", "\n", "user", "assistant", "system", "tool",
                "\nThe", "The", " file", "hello", "<think>", "</think>", "Aqkz", "Zqxv",
                "_strip_probe", "read_file", "a.txt", "\n<|im_start|>",
            ]),
            renderContext: .canonical),
    ]

    init() {
        StablePrefixDetector.resetMemo()
    }

    @Test func probeRenderMatchesTheRawTemplateApplication() throws {
        for fixture in Self.fixtures {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let render = ConversationRender.uncached(
                    tokenizer: fixture.tokenizer,
                    toolSpecs: toolSpecs,
                    renderContext: fixture.renderContext
                )
                for conversation in ProbeParityFixtures.conversations {
                    let messages =
                        conversation.promptMessages
                        + [LeafAdmissionBuilder.Continuation.userTurn.probeMessage]
                    let raw = try fixture.tokenizer.applyChatTemplate(
                        messages: messages,
                        tools: toolSpecs,
                        additionalContext: fixture.renderContext.additionalContext(
                            merging: ["add_generation_prompt": false])
                    )
                    #expect(
                        try render.uncachedContinuationRender(messages: messages) == raw,
                        "\(fixture.name) tools=\(toolSpecs != nil)")
                }
            }
        }
    }

    @Test func futureSharedPrefixMatchesThePreModuleProbePair() throws {
        for fixture in Self.fixtures {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let render = ConversationRender.uncached(
                    tokenizer: fixture.tokenizer,
                    toolSpecs: toolSpecs,
                    renderContext: fixture.renderContext
                )
                for conversation in ProbeParityFixtures.conversations {
                    let before = try preModuleFutureSharedPrefix(
                        tokenizer: fixture.tokenizer,
                        storedConversation: conversation,
                        keySpace: .identity(),
                        toolSpecs: toolSpecs,
                        renderContext: fixture.renderContext
                    )
                    let after = try LeafAdmissionBuilder.futureSharedPrefix(
                        storedConversation: conversation,
                        keySpace: .identity(),
                        render: render
                    )
                    // A fixture on which the probe pair diverges would pass
                    // vacuously — every fixture here converges.
                    #expect(try before?.get() != nil, Comment(rawValue: "\(fixture.name) diverged"))
                    #expect(
                        after == before,
                        Comment(rawValue: "\(fixture.name) tools=\(toolSpecs != nil)"))
                }
            }
        }
    }

    /// The image-bearing shape: the probe path comes back translated through
    /// the key space, before and after alike.
    @Test func futureSharedPrefixMatchesThePreModuleProbePairInImageKeySpace() throws {
        let tokenizer = FakeChatMLTokenizer()
        let stored = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [
                HTTPPrefixCacheMessage(
                    role: .user, content: "describe",
                    images: [HTTPPrefixCacheImage(data: Data([0x0A]))]),
                HTTPPrefixCacheMessage(role: .assistant, content: "a diagram"),
            ]
        )
        let keySpace = try FakeChatMLTokenizer.keySpace(for: stored, runLengths: [4])
        let before = try preModuleFutureSharedPrefix(
            tokenizer: tokenizer,
            storedConversation: stored,
            keySpace: keySpace,
            toolSpecs: nil,
            renderContext: .canonical
        )
        let after = try LeafAdmissionBuilder.futureSharedPrefix(
            storedConversation: stored,
            keySpace: keySpace,
            render: makeRender(tokenizer)
        )
        #expect(try before?.get().contains { $0 < 0 } == true)
        #expect(after == before)
    }

    /// The probe verb is cache-free by construction. The control proves the
    /// setup can tell: under the same eligible fingerprint and a rendering
    /// tokenizer, the CACHED continuation verb moves the private cache's
    /// telemetry (a cold tail-replacement fallback). The probe verb, twice
    /// over, applies the template on every call and leaves it untouched.
    @Test func uncachedContinuationRenderNeverTouchesTheRenderTokenCache() throws {
        let messages =
            ProbeParityFixtures.stopTurn.promptMessages
            + [LeafAdmissionBuilder.Continuation.userTurn.probeMessage]

        let controlCache = RenderTokenCache()
        let control = TemplateCallObservingTokenizer(GreedyTokenizer(pieces: chatMLGreedyPieces))
        _ = try makeRender(control, fingerprint: "probe-parity", cache: controlCache)
            .continuationRender(messages: messages)
        #expect(controlCache.statsSnapshot().replacedFallbacks == 1)

        let cache = RenderTokenCache()
        let tokenizer = TemplateCallObservingTokenizer(GreedyTokenizer(pieces: chatMLGreedyPieces))
        let render = makeRender(tokenizer, fingerprint: "probe-parity", cache: cache)
        let first = try render.uncachedContinuationRender(messages: messages)
        let second = try render.uncachedContinuationRender(messages: messages)

        #expect(first == second)
        #expect(tokenizer.templateCalls == 2)
        #expect(cache.statsSnapshot() == RenderTokenCache.Stats())
    }

    @Test func detectorProbeMatchesTheRawTemplateApplication() throws {
        for fixture in Self.fixtures {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let messages: [[String: any Sendable]] = [
                    ["role": "system", "content": ProbeParityFixtures.systemPrompt],
                    ["role": "user", "content": "A_prefix_probe"],
                ]
                let context = fixture.renderContext.additionalContext()
                let raw = try fixture.tokenizer.applyChatTemplate(
                    messages: messages, tools: toolSpecs, additionalContext: context)
                let probed = try ConversationRender.stablePrefixProbeRender(
                    tokenizer: fixture.tokenizer,
                    messages: messages,
                    tools: toolSpecs,
                    additionalContext: context
                )
                #expect(
                    probed == raw, Comment(rawValue: "\(fixture.name) tools=\(toolSpecs != nil)"))
            }
        }
    }

    @Test func stablePrefixDetectionMatchesThePreModuleTwoProbe() throws {
        for fixture in Self.fixtures {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let context = fixture.renderContext.additionalContext()
                let fullTokens = try fixture.tokenizer.applyChatTemplate(
                    messages: ProbeParityFixtures.stopTurn.promptMessages,
                    tools: toolSpecs,
                    additionalContext: context
                )
                let before = try preModuleStablePrefix(
                    tokenizer: fixture.tokenizer,
                    systemPrompt: ProbeParityFixtures.systemPrompt,
                    toolSpecs: toolSpecs,
                    additionalContext: context,
                    fullTokens: fullTokens
                )
                StablePrefixDetector.resetMemo()
                let after = try StablePrefixDetector.detect(
                    systemPrompt: ProbeParityFixtures.systemPrompt,
                    toolSpecs: toolSpecs,
                    additionalContext: context,
                    fullTokens: fullTokens,
                    tokenizer: fixture.tokenizer
                )
                #expect(before != nil, Comment(rawValue: "\(fixture.name) found no stable prefix"))
                #expect(
                    after == before, Comment(rawValue: "\(fixture.name) tools=\(toolSpecs != nil)"))
            }
        }
    }
}

// MARK: - Real-tokenizer parity

/// The same parity on the real PARO tokenizer and its Jinja template, when
/// the model directory is present (`TESSERACT_TOKENIZE_CACHE_MODEL`, or the
/// `RenderTokenCacheRealTests` default) — skipped otherwise, so it is safe
/// anywhere.
@MainActor
@Suite struct ConversationRenderProbeParityRealTests {

    private nonisolated static var modelDirectory: URL {
        let path =
            ProcessInfo.processInfo.environment["TESSERACT_TOKENIZE_CACHE_MODEL"]
            ?? "~/Library/Application Support/models/z-lab_Qwen3.5-4B-PARO"
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private nonisolated static var modelAvailable: Bool {
        FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("tokenizer_config.json").path)
    }

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static let renderContexts: [TemplateRenderContext] = [
        .canonical, ProbeParityFixtures.preserveThinking,
    ]

    @Test(.enabled(if: modelAvailable))
    func futureSharedPrefixMatchesThePreModuleProbePair() async throws {
        let tokenizer = try await Self.loadTokenizer()
        for renderContext in Self.renderContexts {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let render = ConversationRender.uncached(
                    tokenizer: tokenizer, toolSpecs: toolSpecs, renderContext: renderContext)
                for conversation in ProbeParityFixtures.conversations {
                    let before = try preModuleFutureSharedPrefix(
                        tokenizer: tokenizer,
                        storedConversation: conversation,
                        keySpace: .identity(),
                        toolSpecs: toolSpecs,
                        renderContext: renderContext
                    )
                    let after = try LeafAdmissionBuilder.futureSharedPrefix(
                        storedConversation: conversation,
                        keySpace: .identity(),
                        render: render
                    )
                    #expect(try before?.get() != nil, "probe pair diverged")
                    #expect(after == before)
                }
            }
        }
    }

    @Test(.enabled(if: modelAvailable))
    func stablePrefixDetectionMatchesThePreModuleTwoProbe() async throws {
        let tokenizer = try await Self.loadTokenizer()
        for renderContext in Self.renderContexts {
            for toolSpecs in [nil, ProbeParityFixtures.tools] {
                let context = renderContext.additionalContext()
                let fullTokens = try tokenizer.applyChatTemplate(
                    messages: ProbeParityFixtures.stopTurn.promptMessages,
                    tools: toolSpecs,
                    additionalContext: context
                )
                let before = try preModuleStablePrefix(
                    tokenizer: tokenizer,
                    systemPrompt: ProbeParityFixtures.systemPrompt,
                    toolSpecs: toolSpecs,
                    additionalContext: context,
                    fullTokens: fullTokens
                )
                StablePrefixDetector.resetMemo()
                let after = try StablePrefixDetector.detect(
                    systemPrompt: ProbeParityFixtures.systemPrompt,
                    toolSpecs: toolSpecs,
                    additionalContext: context,
                    fullTokens: fullTokens,
                    tokenizer: tokenizer
                )
                #expect(before != nil, "found no stable prefix")
                #expect(after == before)
            }
        }
    }
}
