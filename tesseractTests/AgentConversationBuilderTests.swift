import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The agent-side adapter into the prefix-cache conversation shape (PRD #72):
/// agent message history → conversation value, or `nil` when the history
/// carries content the shape cannot. The headline property is parity — the
/// same history pushed through the HTTP edge's `MessageConverter` and through
/// `AgentConversationBuilder` yields the *same* conversation value, so both
/// edges share one radix key for shared prefixes. (Tool-definition digests are
/// per-adapter by design; parity is asserted on the empty-tools case.)
@MainActor
struct AgentConversationBuilderTests {

    // MARK: - Role mapping

    @Test func textHistoryMapsRolesAndSystemPrompt() throws {
        let conversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "You are helpful.",
                messages: [
                    .user(content: "Hello"),
                    .assistant(content: "Hi there", toolCalls: nil),
                    .user(content: "How are you?"),
                ],
                toolSpecs: nil
            ))

        #expect(conversation.systemPrompt == "You are helpful.")
        #expect(conversation.messages.map(\.role) == [.user, .assistant, .user])
        #expect(conversation.messages.map(\.content) == ["Hello", "Hi there", "How are you?"])
        #expect(
            conversation.toolDefinitionsDigest
                == HTTPPrefixCacheConversation.emptyToolDefinitionsDigest)
    }

    @Test func midConversationSystemMessageKeepsItsPlace() throws {
        let conversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "System",
                messages: [
                    .user(content: "Hello"),
                    .system(content: "Mode switched"),
                    .user(content: "Continue"),
                ],
                toolSpecs: nil
            ))

        #expect(conversation.messages.map(\.role) == [.user, .system, .user])
        #expect(conversation.messages[1].content == "Mode switched")
    }

    @Test func reasoningIsCarriedOnAssistantTurns() throws {
        let conversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "System",
                messages: [
                    .user(content: "Question"),
                    .assistant(content: "Answer", reasoning: "thought about it", toolCalls: nil),
                    .user(content: "Follow-up"),
                ],
                toolSpecs: nil
            ))

        #expect(conversation.messages[1].reasoning == "thought about it")
    }

    // MARK: - Image attachments

    /// Identity is exact-byte SHA-256 of the attachment data — the same digest
    /// the HTTP edge computes, so a re-sent image keys the same prefix.
    @Test func imageAttachmentCarriesItsByteDigest() throws {
        let conversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "System",
                messages: [
                    .user(
                        content: "What is in this image?",
                        images: [
                            ImageAttachment(
                                data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
                        ])
                ],
                toolSpecs: nil
            ))

        let image = try #require(conversation.messages.first?.images.first)
        #expect(image.digest == ImageDigest(imageBytes: ImageTestFixtures.tinyPNGData))
        #expect(!conversation.images.isEmpty)
    }

    /// Resending identical bytes yields an equal conversation — exact-byte
    /// identity drives prefix matching, not per-request attachment ids.
    @Test func identicalBytesUnderFreshAttachmentIdsProduceEqualConversations() {
        func build() -> HTTPPrefixCacheConversation? {
            AgentConversationBuilder.conversation(
                systemPrompt: "System",
                messages: [
                    // `ImageAttachment.id` is a fresh UUID per value — it must
                    // not leak into the conversation identity.
                    .user(
                        content: "Look",
                        images: [
                            ImageAttachment(
                                data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
                        ])
                ],
                toolSpecs: nil
            )
        }

        let first = build()
        let second = build()
        #expect(first != nil)
        #expect(first == second)
    }

    /// An attachment that no longer decodes cannot be keyed — the builder
    /// bails and the request rides the standard route, uncached but correct.
    @Test func undecodableAttachmentReturnsNil() {
        let conversation = AgentConversationBuilder.conversation(
            systemPrompt: "System",
            messages: [
                .user(
                    content: "Look",
                    images: [
                        ImageAttachment(
                            data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png")
                    ])
            ],
            toolSpecs: nil
        )

        #expect(conversation == nil)
    }

    // MARK: - Tool turns

    @Test func toolCallTurnsMapWithCanonicalizedArgumentsAndDroppedIds() throws {
        let conversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "System",
                messages: [
                    .user(content: "Read the file"),
                    .assistant(
                        content: "Let me read it",
                        toolCalls: [
                            ToolCallInfo(
                                id: "call_1",
                                name: "read",
                                argumentsJSON: #"{"path":   "/tmp/a.txt"}"#
                            )
                        ]),
                    .toolResult(toolCallId: "call_1", content: "file contents"),
                    .user(content: "Thanks"),
                ],
                toolSpecs: nil
            ))

        let assistant = conversation.messages[1]
        #expect(
            assistant.toolCalls == [
                HTTPPrefixCacheToolCall(name: "read", argumentsJSON: #"{"path": "/tmp/a.txt"}"#)
            ])
        let toolResult = conversation.messages[2]
        #expect(toolResult.role == .tool)
        #expect(toolResult.content == "file contents")
    }

    // MARK: - Parity with the HTTP edge

    /// The same history through both adapters yields the same conversation
    /// value — agent chat and the OpenAI edge share radix keys for shared
    /// prefixes. Covers the full shape: system prompt, an image turn, a
    /// tool-call turn with its result, and a trailing user turn.
    @Test func sameHistoryThroughBothAdaptersYieldsTheSameConversation() throws {
        let argumentsJSON = #"{"path": "/tmp/a.txt"}"#
        let httpMessages: [OpenAI.ChatMessage] = [
            .init(role: .system, content: .text("You are helpful.")),
            .init(
                role: .user,
                content: .parts([
                    .init(type: .text, text: "What is in this image?"),
                    .init(
                        type: .image_url,
                        image_url: .init(
                            url: "data:image/png;base64,\(ImageTestFixtures.tinyPNGBase64)")
                    ),
                ])),
            .init(
                role: .assistant,
                content: .text("Let me read the file"),
                tool_calls: [
                    .init(
                        id: "call_1",
                        type: "function",
                        function: .init(name: "read", arguments: argumentsJSON)
                    )
                ]
            ),
            .init(role: .tool, content: .text("file contents"), tool_call_id: "call_1"),
            .init(role: .user, content: .text("Thanks — summarize it.")),
        ]
        let httpConversation = try #require(MessageConverter.normalizeConversation(httpMessages))

        let agentConversation = try #require(
            AgentConversationBuilder.conversation(
                systemPrompt: "You are helpful.",
                messages: [
                    .user(
                        content: "What is in this image?",
                        images: [
                            ImageAttachment(
                                data: ImageTestFixtures.tinyPNGData, mimeType: "image/png")
                        ]),
                    .assistant(
                        content: "Let me read the file",
                        toolCalls: [
                            ToolCallInfo(id: "call_1", name: "read", argumentsJSON: argumentsJSON)
                        ]),
                    .toolResult(toolCallId: "call_1", content: "file contents"),
                    .user(content: "Thanks — summarize it."),
                ],
                toolSpecs: nil
            ))

        #expect(agentConversation == httpConversation)
    }

    // MARK: - Tool-definitions digest

    /// Per-adapter stability is the requirement: the same specs digest the
    /// same on every call (session replay stays continuous turn over turn),
    /// and present-vs-absent tools are distinguished.
    @Test func toolSpecsDigestIsStablePerAdapterAndDistinguishesEmpty() throws {
        let specs: [ToolSpec] = [
            [
                "type": "function",
                "function": ["name": "read", "description": "Read a file"]
                    as [String: any Sendable],
            ]
        ]

        func digest(_ toolSpecs: [ToolSpec]?) throws -> String {
            try #require(
                AgentConversationBuilder.conversation(
                    systemPrompt: "System",
                    messages: [.user(content: "Hello")],
                    toolSpecs: toolSpecs
                )
            ).toolDefinitionsDigest
        }

        #expect(try digest(specs) == digest(specs))
        #expect(try digest(specs) != HTTPPrefixCacheConversation.emptyToolDefinitionsDigest)
        #expect(try digest(nil) == HTTPPrefixCacheConversation.emptyToolDefinitionsDigest)
        #expect(try digest([]) == HTTPPrefixCacheConversation.emptyToolDefinitionsDigest)
    }
}
