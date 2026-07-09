import AppKit
import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - ToolResultImageFlowTests

/// Tool-result images (browser screenshots) must reach the model's context —
/// the 2026-07-09 incident: `toLLMMessage()` dropped image blocks, the model
/// received only "Screenshot of <url>", and confabulated an entire Hacker News
/// front page. These tests pin every hop of the repaired path, plus the
/// text-only degradation and the prefix-cache eligibility bail.
@MainActor
struct ToolResultImageFlowTests {

    private func pngFixture(width: Int = 8, height: Int = 8) -> Data {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        return rep.representation(using: .png, properties: [:])!
    }

    private func screenshotResultMessage() -> ToolResultMessage {
        ToolResultMessage.create(
            toolCallId: "call-1", toolName: "browser.screenshot",
            result: AgentToolResult(content: [
                .image(data: pngFixture(), mimeType: "image/png"),
                .text("Screenshot of https://example.com"),
            ]),
            isError: false
        )
    }

    // MARK: - Hop 1: transcript message → LLMMessage

    @Test
    func toolResultImagesSurviveTheLLMMessageProjection() {
        let message = screenshotResultMessage()
        guard case .toolResult(_, let content, let images)? = message.toLLMMessage() else {
            Issue.record("expected a toolResult LLMMessage")
            return
        }
        #expect(content == "Screenshot of https://example.com")
        #expect(images.count == 1)
        #expect(images[0].mimeType == "image/png")
        #expect(!images[0].data.isEmpty)
    }

    // MARK: - Hop 2: LLMMessage → Chat.Message

    @Test
    func visionActiveAttachesImagesToTheToolMessage() {
        let llm = LLMMessage.toolResult(
            toolCallId: "call-1", content: "Screenshot of https://example.com",
            images: [ImageAttachment(data: pngFixture(), mimeType: "image/png")])

        let chat = toLLMCommonMessages([llm], visionActive: true)
        #expect(chat.count == 1)
        #expect(chat[0].role == .tool)
        #expect(chat[0].content == "Screenshot of https://example.com")
        #expect(chat[0].images.count == 1)
    }

    @Test
    func textOnlySessionDegradesImagesToAnExplicitNote() {
        let llm = LLMMessage.toolResult(
            toolCallId: "call-1", content: "Screenshot of https://example.com",
            images: [ImageAttachment(data: pngFixture(), mimeType: "image/png")])

        let chat = toLLMCommonMessages([llm], visionActive: false)
        #expect(chat[0].images.isEmpty)
        // The model is told the pixels are absent instead of being invited
        // to hallucinate them.
        #expect(chat[0].content.contains("text-only"))
        #expect(chat[0].content.contains("1 image(s)"))
    }

    @Test
    func imagelessToolResultsAreUntouchedEitherWay() {
        let llm = LLMMessage.toolResult(toolCallId: "call-1", content: "plain result")
        for visionActive in [true, false] {
            let chat = toLLMCommonMessages([llm], visionActive: visionActive)
            #expect(chat[0].content == "plain result")
            #expect(chat[0].images.isEmpty)
        }
    }

    // MARK: - Hop 3: UserInput collects tool-message images

    @Test
    func buildUserInputCollectsToolResultImages() {
        let llm = LLMMessage.toolResult(
            toolCallId: "call-1", content: "Screenshot of https://example.com",
            images: [ImageAttachment(data: pngFixture(), mimeType: "image/png")])

        let input = AgentEngine.buildUserInput(
            systemPrompt: "system", messages: [llm], toolSpecs: nil,
            visionActive: true)
        #expect(input.images.count == 1)

        let textOnly = AgentEngine.buildUserInput(
            systemPrompt: "system", messages: [llm], toolSpecs: nil,
            visionActive: false)
        #expect(textOnly.images.isEmpty)
    }

    // MARK: - Prefix-cache eligibility

    /// The prefix-cache conversation shape carries images on user messages
    /// only; an image-bearing tool result must make the whole request
    /// ineligible (nil) so it rides the standard route with pixels intact —
    /// mirroring the HTTP edge's `.nonTextToolMessage` bail.
    @Test
    func imageBearingToolResultsBailOutOfTheCacheShape() {
        let withImages: [LLMMessage] = [
            .user(content: "take a screenshot"),
            .assistant(content: "", reasoning: nil, toolCalls: nil),
            .toolResult(
                toolCallId: "call-1", content: "Screenshot of https://example.com",
                images: [ImageAttachment(data: pngFixture(), mimeType: "image/png")]),
        ]
        #expect(
            AgentConversationBuilder.conversation(
                systemPrompt: "system", messages: withImages, toolSpecs: nil)
                == nil)

        let textOnly: [LLMMessage] = [
            .user(content: "read the page"),
            .assistant(content: "", reasoning: nil, toolCalls: nil),
            .toolResult(toolCallId: "call-1", content: "Title: Example"),
        ]
        #expect(
            AgentConversationBuilder.conversation(
                systemPrompt: "system", messages: textOnly, toolSpecs: nil)
                != nil)
    }
}
