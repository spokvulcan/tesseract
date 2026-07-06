//
//  AgentConversationStoreV3Tests.swift
//  tesseractTests
//
//  Seam 2 of the chat rewrite (ADR-0024): the conversation-store protocol.
//  Round-trips the parts-based canonical model through the real store against
//  a temp directory, and proves the storage-version bump wipes pre-v3 data on
//  first launch (the explicit no-migrator decision).
//
//  Serialized: the real store touches the filesystem, and the test scheme runs
//  suites twice in parallel processes — each test uses its own unique temp dir
//  so parallel processes can't collide.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentConversationStoreV3Tests {

    private func makeTempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-store-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Parts round-trip

    @Test func partsBasedConversationRoundTripsThroughDisk() throws {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let user = UserMessage(content: "run the tool")
        let assistant = AssistantMessage(
            content: [
                .thinking(ThinkingPart(thinking: "let me check")),
                .text(TextPart(text: "Running it now.")),
                .toolCall(
                    ToolCallPart(
                        id: "call-9", name: "read_file",
                        argumentsJSON: #"{"path":"a.txt"}"#)),
            ],
            model: "test-model",
            stopReason: .toolUse
        )
        let result = ToolResultMessage(
            toolCallId: "call-9", toolName: "read_file", content: [.text("contents")])

        let store = AgentConversationStore(directory: dir)
        store.createNew()
        store.updateCurrentMessages([CoreMessage.user(user), assistant, result])
        store.saveCurrent()
        let savedID = store.currentConversation?.id

        // Fresh store instance over the same directory — full disk round-trip.
        let reloaded = AgentConversationStore(directory: dir)
        reloaded.loadMostRecent()
        let conversation = reloaded.currentConversation
        #expect(conversation?.id == savedID)
        #expect(conversation?.messages.count == 3)

        let roundTripped = conversation?.messages[1].asAssistant
        #expect(roundTripped == assistant)
        #expect(roundTripped?.content.count == 3)
        #expect(roundTripped?.stopReason == .toolUse)
        #expect(roundTripped?.toolCalls.first?.id == "call-9")
    }

    // MARK: - Wipe on version mismatch

    @Test func preV3DataIsWipedOnFirstLaunch() throws {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Simulate a v2 install: version marker + a stray conversation file.
        try "2".write(
            to: dir.appendingPathComponent(".storage_version"), atomically: true, encoding: .utf8)
        let strayFile = dir.appendingPathComponent("\(UUID().uuidString).json")
        try #"{"legacy":"v2"}"#.write(to: strayFile, atomically: true, encoding: .utf8)

        let store = AgentConversationStore(directory: dir)

        #expect(store.conversations.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: strayFile.path))
        let marker = try String(
            contentsOf: dir.appendingPathComponent(".storage_version"), encoding: .utf8)
        #expect(marker == "3")
    }

    @Test func matchingVersionKeepsExistingConversations() throws {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let first = AgentConversationStore(directory: dir)
        first.createNew()
        first.updateCurrentMessages([CoreMessage.user(UserMessage(content: "keep me"))])
        first.saveCurrent()

        let second = AgentConversationStore(directory: dir)
        #expect(second.conversations.count == 1)
        second.loadMostRecent()
        #expect(second.currentConversation?.messages.first?.asUser?.content == "keep me")
    }
}
