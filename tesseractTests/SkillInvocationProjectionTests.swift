//
//  SkillInvocationProjectionTests.swift
//  tesseractTests
//
//  The **Skill Invocation Row** (PRD #174): pure tests for the
//  `SkillInvocationBlock` parser and for the Chat Transcript projection of a
//  user message that carries an injected skill block — collapsed and expanded
//  shapes, attachment carry-through, and the expansion-pruning contract. Same
//  style as ChatTranscriptTests: input → output row assertions, no SwiftUI.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SkillInvocationProjectionTests {

    // MARK: - Fixtures

    private func skillContent(
        name: String = "proofread",
        body: String = "Fix objective errors only.",
        arguments: String = ""
    ) -> String {
        var content = """
            <skill name="\(name)" location="/skills/\(name)/SKILL.md">
            References are relative to /skills/\(name).

            \(body)
            </skill>
            """
        if !arguments.isEmpty {
            content += "\n\n\(arguments)"
        }
        return content
    }

    private func ctx(expandedDetails: Set<String> = []) -> ChatTranscript.Context {
        ChatTranscript.Context(
            expandedDetails: expandedDetails,
            formatTimestamp: { _ in "TS" }
        )
    }

    // MARK: - SkillInvocationBlock parsing

    @Test func parsesNameBlockAndArguments() throws {
        let content = skillContent(name: "proofread-tweet", arguments: "my draft tweet")
        let block = try #require(SkillInvocationBlock.parse(content))
        #expect(block.skillName == "proofread-tweet")
        #expect(block.argumentText == "my draft tweet")
        #expect(block.injectedBlock.hasPrefix("<skill name=\"proofread-tweet\""))
        #expect(block.injectedBlock.hasSuffix("</skill>"))
        #expect(block.displayLabel == "Proofread Tweet")
    }

    @Test func bareInvocationParsesWithEmptyArguments() throws {
        let block = try #require(SkillInvocationBlock.parse(skillContent()))
        #expect(block.argumentText.isEmpty)
    }

    @Test func multilineArgumentsSurviveIntact() throws {
        let arguments = "line one\nline two\n\nDefault target language: Ukrainian"
        let block = try #require(SkillInvocationBlock.parse(skillContent(arguments: arguments)))
        #expect(block.argumentText == arguments)
    }

    @Test func plainTextIsNotASkillInvocation() {
        #expect(SkillInvocationBlock.parse("hello there") == nil)
        #expect(SkillInvocationBlock.parse("<skillet name=\"x\">no</skillet>") == nil)
        #expect(SkillInvocationBlock.parse("") == nil)
    }

    @Test func handTypedSkillTagWithoutLocationStaysAUserMessage() {
        // Only the injection path writes both `name` and `location`; a user
        // literally typing a bare `<skill name=…>` tag keeps their bubble.
        let handTyped = "<skill name=\"proofread\">how does this tag work?</skill>"
        #expect(SkillInvocationBlock.parse(handTyped) == nil)
    }

    @Test func unterminatedBlockIsNotASkillInvocation() {
        #expect(SkillInvocationBlock.parse("<skill name=\"proofread\">no closing tag") == nil)
    }

    // MARK: - Chat Transcript projection

    @Test func skillMessageProjectsToSkillInvocationRow() throws {
        let user = UserMessage(content: skillContent(arguments: "fix this"))
        let projection = ChatTranscript.rows(from: [user], ctx())

        let row = try #require(projection.rows.first)
        guard case .skillInvocation(let data) = row.kind else {
            Issue.record("Expected .skillInvocation, got \(row.kind)")
            return
        }
        #expect(data.skillName == "proofread")
        #expect(data.displayLabel == "Proofread")
        #expect(data.argumentText == "fix this")
        #expect(data.isExpanded == false)
        #expect(data.messageID == user.id)
        #expect(data.timestamp == "TS")
    }

    @Test func expandedRowCarriesTheFullInjectedBlock() throws {
        let user = UserMessage(content: skillContent())
        let collapsed = ChatTranscript.rows(from: [user], ctx())
        let rowID = try #require(collapsed.rows.first?.id)

        let expanded = ChatTranscript.rows(
            from: [user], ctx(expandedDetails: [rowID]))
        guard case .skillInvocation(let data) = expanded.rows.first?.kind else {
            Issue.record("Expected .skillInvocation")
            return
        }
        #expect(data.isExpanded == true)
        #expect(data.injectedBlock.hasPrefix("<skill name=\"proofread\""))
        #expect(data.injectedBlock.contains("Fix objective errors only."))
    }

    @Test func skillRowIDRegistersForExpansionPruning() throws {
        // The projection reports the skill row id alongside tool-call row ids so
        // the controller's stale-expansion pruning keeps it alive across
        // rebuilds — collapsing then re-expanding must survive.
        let user = UserMessage(content: skillContent())
        let projection = ChatTranscript.rows(from: [user], ctx())
        let rowID = try #require(projection.rows.first?.id)
        #expect(projection.detailRowIDs.contains(rowID))
    }

    @Test func attachedImagesRideOnTheRow() throws {
        let image = ImageAttachment(
            data: Data([0x89, 0x50]), mimeType: "image/png", filename: "shot.png")
        let user = UserMessage(content: skillContent(), images: [image])
        let projection = ChatTranscript.rows(from: [user], ctx())

        guard case .skillInvocation(let data) = projection.rows.first?.kind else {
            Issue.record("Expected .skillInvocation")
            return
        }
        #expect(data.images.map(\.id) == [image.id])
    }

    @Test func plainUserMessageStillProjectsToUserRow() {
        let user = UserMessage(content: "just a normal message")
        let projection = ChatTranscript.rows(from: [user], ctx())
        guard case .user(let data) = projection.rows.first?.kind else {
            Issue.record("Expected .user")
            return
        }
        #expect(data.content == "just a normal message")
    }

    @Test func assistantAnswerFollowsTheSkillRow() throws {
        let user = UserMessage(content: skillContent(arguments: "text"))
        let assistant = AssistantMessage(content: "corrected text")
        let projection = ChatTranscript.rows(from: [user, assistant], ctx())

        #expect(projection.rows.count == 2)
        guard case .skillInvocation = projection.rows[0].kind,
            case .assistantText(let answer) = projection.rows[1].kind
        else {
            Issue.record("Expected skill row then answer row")
            return
        }
        #expect(answer.content == "corrected text")
    }
}
