import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import Tesseract_Agent

/// The Emitted Path Index on the real Qwen3.5 tokenizer and chat template
/// (ADR-0063 decision 1's premise and decision 4's composition): the
/// end-of-turn marker derives to the `<|im_end|>` special token, a special
/// token is a hard pretoken boundary — so encoding the render up to a
/// marker and the rest separately equals encoding the whole — and a
/// registered path is served back, exactly, at the next request's edge.
///
/// Skipped unless the PARO model directory is on disk (override with
/// `TESSERACT_TOKENIZE_CACHE_MODEL`, as `RenderTokenCacheRealTests` does).
struct EmittedPathResolveRealTests {

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

    private static let fingerprint = "real-emitted-path"

    /// A template that honours `preserve_thinking` (Qwen3.8 declares it;
    /// the PARO checkpoint's template strips prior thinking regardless),
    /// when its directory is on disk.
    private nonisolated static var preserveThinkingModelDirectory: URL {
        URL(
            fileURLWithPath: NSString(
                string: "~/Library/Application Support/models/mlx-community_Qwen3.8-27B-4bit"
            ).expandingTildeInPath)
    }

    private nonisolated static var preserveThinkingModelAvailable: Bool {
        FileManager.default.fileExists(
            atPath: preserveThinkingModelDirectory.appendingPathComponent("tokenizer_config.json")
                .path)
    }

    private static func loadTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        try await #huggingFaceTokenizerLoader().load(from: modelDirectory)
    }

    private static let readTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "read_file",
            "description": "Read a file.",
            "parameters": [
                "type": "object",
                "properties": [
                    "path": ["type": "string"] as [String: any Sendable],
                    "limit": ["type": "integer"] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    private static func encode(_ tokenizer: any MLXLMCommon.Tokenizer, _ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    private static func encode(
        _ tokenizer: any MLXLMCommon.Tokenizer, bytes: ArraySlice<UInt8>
    ) -> [Int] {
        // swiftlint:disable:next optional_data_string_conversion
        encode(tokenizer, String(decoding: bytes, as: UTF8.self))
    }

    // MARK: - Marker

    @Test(.enabled(if: modelAvailable))
    func theMarkerDerivesToTheImEndSpecialToken() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = try #require(
            ConversationRender.endOfTurnMarker(
                index: index, fingerprint: Self.fingerprint, tokenizer: tokenizer))
        #expect(marker.text == "<|im_end|>")
        #expect(marker.tokenID == tokenizer.convertTokenToId("<|im_end|>"))
        #expect(Self.encode(tokenizer, marker.text) == [marker.tokenID])
    }

    // MARK: - Suffix-encode equality at every end-of-turn boundary

    /// Assistant endings and user beginnings that stress the boundary: a
    /// pretoken that would merge across it if the marker were plain text.
    private static let assistantEndings = [
        "Hello.", "Hello", "Hello\n", "Hello  ", "Done ✅", "Résumé — ok", "42", ";", ")",
        "```swift\nlet x = 1\n```", "line one\n\nline two\n\n", "trailing space ", "…",
        "a\tb", "ends with newline\n", "日本語です",
    ]
    private static let userBeginnings = [
        "Next", " leading space", "\nnewline first", "1. list", "«quote»", "```code",
        "?", "and then", "日本語", "\t\ttabbed", "", "  ", "🙂 emoji first",
    ]

    /// Every marker of every render in the battery: encoding the bytes up
    /// to the marker and the bytes after it separately equals encoding the
    /// whole — the composition's premise, on the real pretokenizer.
    @Test(.enabled(if: modelAvailable))
    func encodingSplitsExactlyAtEveryMarker() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = try #require(
            ConversationRender.endOfTurnMarker(
                index: index, fingerprint: Self.fingerprint, tokenizer: tokenizer))
        let context = TemplateRenderContext.canonical.additionalContext()

        var conversations: [(messages: [[String: any Sendable]], tools: [ToolSpec]?)] = []
        for (offset, ending) in Self.assistantEndings.enumerated() {
            let beginning = Self.userBeginnings[offset % Self.userBeginnings.count]
            let reasoning = offset.isMultiple(of: 2) ? "Thinking about \(offset)." : ""
            conversations.append(
                (
                    [
                        ["role": "system", "content": "You are terse."],
                        ["role": "user", "content": "First \(offset)"],
                        [
                            "role": "assistant", "content": ending,
                            "reasoning_content": reasoning,
                        ],
                        ["role": "user", "content": beginning],
                        ["role": "assistant", "content": "Second answer \(ending)"],
                        ["role": "user", "content": "Third \(beginning)"],
                    ], nil
                ))
        }
        // A tool-call turn followed by its result, with tools declared.
        let toolCallConversation = HTTPPrefixCacheConversation(
            systemPrompt: "You can read files.",
            messages: [
                HTTPPrefixCacheMessage(role: .user, content: "Read /tmp/a"),
                .assistant(
                    content: "Reading.", reasoning: "I need the file.",
                    toolCalls: [
                        HTTPPrefixCacheToolCall(
                            name: "read_file",
                            argumentsJSON: "{\"path\": \"/tmp/a\", \"limit\": 10}")
                    ]),
                HTTPPrefixCacheMessage(role: .tool, content: "line 1\nline 2\n"),
                .assistant(content: "Two lines.", reasoning: "Short."),
                HTTPPrefixCacheMessage(role: .user, content: "Thanks"),
            ]
        ).promptMessages
        conversations.append((toolCallConversation, [Self.readTool]))

        var boundaries = 0
        for conversation in conversations {
            for generationPrompt in [true, false] {
                var additional = context ?? [:]
                additional["add_generation_prompt"] = generationPrompt
                let rendered = try rendering.renderChatTemplate(
                    messages: conversation.messages, tools: conversation.tools,
                    additionalContext: additional)
                let bytes = Array(rendered.utf8)
                let full = Self.encode(tokenizer, rendered)
                let hashes = EmittedPathIndex.prefixHashes(
                    renderedBytes: bytes, marker: marker.bytes)
                #expect(!hashes.isEmpty)
                for prefix in hashes {
                    let head = Self.encode(tokenizer, bytes: bytes[..<prefix.end])
                    let tail = Self.encode(tokenizer, bytes: bytes[prefix.end...])
                    #expect(head + tail == full, "split at byte \(prefix.end) of \(bytes.count)")
                    #expect(head.last == marker.tokenID)
                    boundaries += 1
                }
            }
        }
        #expect(boundaries >= 2 * conversations.count * 3)
    }

    /// The composition against a populated index at every depth: with every
    /// prefix registered under its canonical encode, the deepest hit plus
    /// the canonical suffix equals the canonical whole.
    @Test(.enabled(if: modelAvailable))
    func theCompositionEqualsTheCanonicalEncodeAtEveryDepth() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = try #require(
            ConversationRender.endOfTurnMarker(
                index: index, fingerprint: Self.fingerprint, tokenizer: tokenizer))
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "One"],
            ["role": "assistant", "content": "First answer.", "reasoning_content": "r1"],
            ["role": "user", "content": "Two"],
            ["role": "assistant", "content": "Second answer.\n", "reasoning_content": "r2"],
            ["role": "user", "content": "Three"],
        ]
        let rendered = try rendering.renderChatTemplate(
            messages: messages, tools: nil,
            additionalContext: TemplateRenderContext.canonical.additionalContext())
        let bytes = Array(rendered.utf8)
        let canonical = Self.encode(tokenizer, rendered)
        let hashes = EmittedPathIndex.prefixHashes(renderedBytes: bytes, marker: marker.bytes)
        // Four turns, plus the system turn the template injects on its own.
        #expect(hashes.count >= 4)

        for (depth, prefix) in hashes.enumerated() {
            _ = index.register(
                fingerprint: Self.fingerprint, hash: prefix.hash,
                ids: Self.encode(tokenizer, bytes: bytes[..<prefix.end]))
            let composition = EmittedPathResolve.compose(
                index: index, fingerprint: Self.fingerprint, marker: marker,
                renderedBytes: bytes, tokenizer: tokenizer)
            guard
                case .indexed(let tokens, let indexedPrefix, _, let markerDepth, let markerCount) =
                    composition
            else {
                Issue.record("expected a hit at depth \(depth)")
                continue
            }
            #expect(tokens == canonical)
            // The hit is the deepest registered marker; its depth counts
            // the markers past it (0 = the last marker of the render).
            #expect(markerDepth == hashes.count - 1 - depth)
            #expect(markerCount == hashes.count)
            #expect(canonical[indexedPrefix - 1] == marker.tokenID)
        }
    }

    // MARK: - Request edge on a tool-call boundary

    /// The stored turn's registration and the next request's resolve on the
    /// real template, across a tool-call boundary: the Leaf Store registers
    /// the path (prompt + the ids the model would have emitted — here the
    /// canonical encode past the prompt, which the fidelity gate accepts),
    /// and the next request's `fullRender` hits it with the whole path as
    /// indexed prefix, the composition equal to the canonical tokens.
    @Test(.enabled(if: modelAvailable))
    func theRequestEdgeServesTheRegisteredPathAcrossAToolCallBoundary() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let identity = ModelIdentity(directory: Self.modelDirectory)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let tools = [Self.readTool]
        let context = TemplateRenderContext.canonical

        let previous = HTTPPrefixCacheConversation(
            systemPrompt: "You can read files.",
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Read /tmp/a")])
        let echo = HTTPPrefixCacheMessage.assistant(
            content: "Reading.", reasoning: "I need the file.",
            toolCalls: [
                HTTPPrefixCacheToolCall(
                    name: "read_file", argumentsJSON: "{\"path\": \"/tmp/a\", \"limit\": 10}")
            ])
        let stored = previous.appendingAssistant(echo)
        let next = HTTPPrefixCacheConversation(
            systemPrompt: stored.systemPrompt,
            messages: stored.messages + [HTTPPrefixCacheMessage(role: .tool, content: "line 1\n")])

        func makeRender() -> ConversationRender {
            ConversationRender.forTextOnlyRequest(
                tokenizer: tokenizer, toolSpecs: tools, renderContext: context, hasMedia: false,
                producesFlatTextTokens: true, modelFingerprint: Self.fingerprint,
                cache: RenderTokenCache(), emittedPathIndex: index, diagnostics: nil)
        }

        // Request N: its prompt is the generation-prompt render.
        let requestN = makeRender()
        let prompt = try #require(requestN.fullRender(messages: previous.promptMessages))
        guard case .eligible(_, _, let marker) = requestN.emittedPathEligibility() else {
            Issue.record("the real template must be eligible")
            return
        }

        // The Leaf Store's stored render, and the path a live turn fed.
        let storedRender = try requestN.storedRender(messages: stored.promptMessages)
        let storedBytes = try #require(storedRender.bytes)
        let markerIndex = try #require(storedRender.tokens.lastIndex(of: marker.tokenID))
        let path = Array(storedRender.tokens[...markerIndex])
        try #require(path.starts(with: prompt), "the prompt must be a token prefix of the path")
        let generated = Array(path[prompt.count...])
        #expect(generated.last == marker.tokenID)

        let outcome = EmittedPathRegistration.register(
            EmittedPathRegistration.Inputs(
                index: index, fingerprint: Self.fingerprint, marker: marker, tokenizer: tokenizer,
                storedRenderBytes: storedBytes, storedMessage: echo, promptKeyPath: prompt,
                generatedTokens: generated, stoppedOn: marker.tokenID, toolCallFormat: .qwen35,
                tools: tools,
                startsInsideThinkBlock: context.startsInsideThinkBlock(
                    promptStartsThinking: identity.promptStartsThinking)))
        guard case .registered(let registered) = outcome else {
            Issue.record("expected a registration, got \(outcome)")
            return
        }
        #expect(registered.pathLength == path.count)
        #expect(registered.appendedEndOfTurn == false)
        #expect(registered.promptTokens == prompt.count)

        // Request N+1 at its edge.
        let requestNext = makeRender()
        let tokens = try #require(requestNext.fullRender(messages: next.promptMessages))
        let canonical = try tokenizer.applyChatTemplate(
            messages: next.promptMessages, tools: tools,
            additionalContext: context.additionalContext())
        #expect(tokens == canonical)
        let summary = try #require(requestNext.emittedPathTelemetry?.summary)
        #expect(summary.requestEdgeIndexedPrefix == path.count)
        #expect(summary.requestEdgeSuffixTokens == canonical.count - path.count)
        #expect(index.statsSnapshot().fidelityRejections == 0)
    }

    /// The served composition on the real template: a registered path that
    /// splits the echoed text differently from the canonical encode is what
    /// the next request feeds — the index path, then the canonical encode of
    /// the bytes after its marker — never the canonical re-encode of the
    /// same bytes. A tool-stretch turn: the one boundary this template
    /// renders verbatim for the next request (it strips prior thinking at
    /// a user boundary, which takes the boundary path instead).
    @Test(.enabled(if: modelAvailable))
    func theRequestEdgeServesANonCanonicalSplit() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = try #require(
            ConversationRender.endOfTurnMarker(
                index: index, fingerprint: Self.fingerprint, tokenizer: tokenizer))
        let context = TemplateRenderContext.canonical
        let tools = [Self.readTool]
        let previous = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Read /tmp/a")])
        let stored = previous.appendingAssistant(
            .assistant(
                content: "Reading it now.", reasoning: "The user wants the file.",
                toolCalls: [
                    HTTPPrefixCacheToolCall(
                        name: "read_file", argumentsJSON: "{\"path\": \"/tmp/a\", \"limit\": 10}")
                ]))
        let next = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: stored.messages + [HTTPPrefixCacheMessage(role: .tool, content: "line 1\n")]
        ).promptMessages
        var storedContext = context.additionalContext() ?? [:]
        storedContext["add_generation_prompt"] = false
        let storedBytes = Array(
            try rendering.renderChatTemplate(
                messages: stored.promptMessages, tools: tools, additionalContext: storedContext
            ).utf8)
        let last = try #require(
            EmittedPathIndex.prefixHashes(renderedBytes: storedBytes, marker: marker.bytes).last)
        let canonicalPath = Self.encode(tokenizer, bytes: storedBytes[..<last.end])
        let respelled = try #require(Self.reSplit(canonicalPath, tokenizer: tokenizer))
        #expect(respelled != canonicalPath)
        #expect(
            tokenizer.decode(tokenIds: respelled, skipSpecialTokens: false)
                == tokenizer.decode(tokenIds: canonicalPath, skipSpecialTokens: false))
        _ = index.register(fingerprint: Self.fingerprint, hash: last.hash, ids: respelled)

        let render = ConversationRender.forTextOnlyRequest(
            tokenizer: tokenizer, toolSpecs: tools, renderContext: context, hasMedia: false,
            producesFlatTextTokens: true, modelFingerprint: Self.fingerprint,
            cache: RenderTokenCache(), emittedPathIndex: index, diagnostics: nil)
        let tokens = try #require(render.fullRender(messages: next))
        let canonical = try tokenizer.applyChatTemplate(
            messages: next, tools: tools, additionalContext: context.additionalContext())
        #expect(Array(tokens.prefix(respelled.count)) == respelled)
        #expect(
            Array(tokens.dropFirst(respelled.count))
                == Array(canonical.dropFirst(canonicalPath.count)))
        #expect(tokens != canonical)
        let summary = try #require(render.emittedPathTelemetry?.summary)
        #expect(summary.requestEdgeIndexedPrefix == respelled.count)
        #expect(summary.requestEdgeSuffixTokens == canonical.count - canonicalPath.count)
    }

    /// `ids` with its first token spelling three or more ASCII letters
    /// replaced by the encodes of its two halves — the same text, a split
    /// the canonical encode never produces.
    private static func reSplit(_ ids: [Int], tokenizer: any MLXLMCommon.Tokenizer) -> [Int]? {
        for (offset, id) in ids.enumerated() {
            let piece = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            let letters = piece.drop(while: { $0 == " " })
            guard letters.count >= 3, letters.allSatisfy({ $0.isLetter && $0.isASCII })
            else { continue }
            let split =
                encode(tokenizer, String(piece.dropLast()))
                + encode(tokenizer, String(piece.suffix(1)))
            guard split != [id],
                tokenizer.decode(tokenIds: split, skipSpecialTokens: false) == piece
            else { continue }
            return Array(ids[..<offset]) + split + Array(ids[(offset + 1)...])
        }
        return nil
    }

    /// A stop-finish answer at a user boundary under the Preserve-Thinking
    /// Render: the next request's render keeps the previous turn's think
    /// block, so the whole previous turn — reasoning included — is the hit,
    /// and the request encodes only its new message and the glue.
    @Test(.enabled(if: preserveThinkingModelAvailable))
    func theWholePreviousTurnHitsUnderThePreserveThinkingRender() async throws {
        let directory = Self.preserveThinkingModelDirectory
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        let identity = ModelIdentity(directory: directory)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let context = TemplateRenderContext(flags: [.preserveThinking])
        let fingerprint = "real-preserve-thinking"
        func makeRender() -> ConversationRender {
            ConversationRender.forTextOnlyRequest(
                tokenizer: tokenizer, toolSpecs: nil, renderContext: context, hasMedia: false,
                producesFlatTextTokens: true, modelFingerprint: fingerprint,
                cache: RenderTokenCache(), emittedPathIndex: index, diagnostics: nil)
        }

        let previous = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Say something.")])
        let echo = HTTPPrefixCacheMessage.assistant(
            content: "Something memorable, then.",
            reasoning: "A short thought about what to say.")
        let stored = previous.appendingAssistant(echo)
        let next = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: stored.messages + [HTTPPrefixCacheMessage(role: .user, content: "Again.")])

        // Request N, its stored turn, and the path a live turn fed.
        let requestN = makeRender()
        let prompt = try #require(requestN.fullRender(messages: previous.promptMessages))
        guard case .eligible(_, _, let marker) = requestN.emittedPathEligibility() else {
            Issue.record("the real template must be eligible")
            return
        }
        let storedRender = try requestN.storedRender(messages: stored.promptMessages)
        let storedBytes = try #require(storedRender.bytes)
        let markerIndex = try #require(storedRender.tokens.lastIndex(of: marker.tokenID))
        let path = Array(storedRender.tokens[...markerIndex])
        try #require(path.starts(with: prompt), "the prompt must be a token prefix of the path")
        let generated = Array(path[prompt.count...])
        #expect(
            tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)
                .contains("A short thought"))
        let outcome = EmittedPathRegistration.register(
            EmittedPathRegistration.Inputs(
                index: index, fingerprint: fingerprint, marker: marker, tokenizer: tokenizer,
                storedRenderBytes: storedBytes, storedMessage: echo, promptKeyPath: prompt,
                generatedTokens: generated, stoppedOn: marker.tokenID, toolCallFormat: .qwen35,
                tools: nil,
                startsInsideThinkBlock: context.startsInsideThinkBlock(
                    promptStartsThinking: identity.promptStartsThinking)))
        guard case .registered = outcome else {
            Issue.record("expected a registration, got \(outcome)")
            return
        }

        // Request N+1 at its edge: the whole previous turn is the hit.
        let requestNext = makeRender()
        let tokens = try #require(requestNext.fullRender(messages: next.promptMessages))
        #expect(Array(tokens.prefix(path.count)) == path)
        let summary = try #require(requestNext.emittedPathTelemetry?.summary)
        #expect(summary.requestEdgeIndexedPrefix == path.count)
        #expect(summary.requestEdgeSuffixTokens == tokens.count - path.count)
    }

    /// The fidelity gate on the real template's tool-call render: the
    /// replayed ids reconstruct the stored message; a different stored
    /// argument is rejected on the tool-call field.
    @Test(.enabled(if: modelAvailable))
    func theFidelityGateReadsTheRealToolCallRender() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let rendering = try #require(tokenizer as? any ChatTemplateRendering)
        let identity = ModelIdentity(directory: Self.modelDirectory)
        let index = EmittedPathIndex(byteBudget: 1 << 20)
        let marker = try #require(
            ConversationRender.endOfTurnMarker(
                index: index, fingerprint: Self.fingerprint, tokenizer: tokenizer))
        let tools = [Self.readTool]
        let context = TemplateRenderContext.canonical
        let startsInsideThinkBlock = context.startsInsideThinkBlock(
            promptStartsThinking: identity.promptStartsThinking)

        let previous = HTTPPrefixCacheConversation(
            systemPrompt: nil,
            messages: [HTTPPrefixCacheMessage(role: .user, content: "Read /tmp/a")])
        let echo = HTTPPrefixCacheMessage.assistant(
            content: "Reading it now.", reasoning: "The user wants the file.",
            toolCalls: [
                HTTPPrefixCacheToolCall(
                    name: "read_file", argumentsJSON: "{\"path\": \"/tmp/a\", \"limit\": 10}")
            ])
        let prompt = Self.encode(
            tokenizer,
            try rendering.renderChatTemplate(
                messages: previous.promptMessages, tools: tools,
                additionalContext: context.additionalContext()))
        var storedContext = context.additionalContext() ?? [:]
        storedContext["add_generation_prompt"] = false
        let storedTokens = Self.encode(
            tokenizer,
            try rendering.renderChatTemplate(
                messages: previous.appendingAssistant(echo).promptMessages, tools: tools,
                additionalContext: storedContext))
        let markerIndex = try #require(storedTokens.lastIndex(of: marker.tokenID))
        try #require(storedTokens.starts(with: prompt))
        let contentIDs = Array(storedTokens[prompt.count..<markerIndex])

        let replayed = EmittedPathFidelity.replay(
            contentIDs: contentIDs, tokenizer: tokenizer, toolCallFormat: .qwen35, tools: tools,
            startsInsideThinkBlock: startsInsideThinkBlock)
        #expect(replayed == echo)
        #expect(
            EmittedPathFidelity.check(
                contentIDs: contentIDs, tokenizer: tokenizer, toolCallFormat: .qwen35,
                tools: tools, startsInsideThinkBlock: startsInsideThinkBlock, stored: echo)
                == .match)

        let other = HTTPPrefixCacheMessage.assistant(
            content: "Reading it now.", reasoning: "The user wants the file.",
            toolCalls: [
                HTTPPrefixCacheToolCall(
                    name: "read_file", argumentsJSON: "{\"path\": \"/tmp/b\", \"limit\": 10}")
            ])
        let rejected = EmittedPathFidelity.check(
            contentIDs: contentIDs, tokenizer: tokenizer, toolCallFormat: .qwen35,
            tools: tools, startsInsideThinkBlock: startsInsideThinkBlock, stored: other)
        guard case .mismatch(let mismatch) = rejected else {
            Issue.record("a different argument must be rejected")
            return
        }
        #expect(mismatch.field == EmittedPathFidelity.Field.toolCalls)
    }
}
