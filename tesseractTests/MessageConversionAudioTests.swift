//
//  MessageConversionAudioTests.swift
//  tesseractTests
//
//  The Native Audio Turn through the agent conversion layer: spoken takes
//  reach the vendor chat shape as sample arrays when the loaded container
//  can hear, degrade to an explicit text note when it can't, and survive
//  the persistence round trip like image attachments.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct MessageConversionAudioTests {

    private static func takeAttachment(seconds: Double = 0.1) -> AudioAttachment {
        let samples = [Float](repeating: 0.25, count: Int(16_000 * seconds))
        return AudioAttachment(
            data: VoiceTakeWAV.encode(samples: samples, sampleRate: 16_000),
            duration: seconds)
    }

    @Test func audioActiveAttachesTakesToTheUserTurn() {
        let messages = toLLMCommonMessages(
            [.user(content: "", audios: [Self.takeAttachment()])],
            audioActive: true)

        #expect(messages.count == 1)
        #expect(messages[0].audios.count == 1)
        #expect(messages[0].content.isEmpty)
    }

    /// A deaf session must be told the audio is absent — an empty user turn
    /// invites hallucination exactly like a dropped screenshot did.
    @Test func audioInactiveDegradesToATextNote() {
        let messages = toLLMCommonMessages(
            [.user(content: "hello", audios: [Self.takeAttachment()])],
            audioActive: false)

        #expect(messages[0].audios.isEmpty)
        #expect(messages[0].content.contains("hello"))
        #expect(messages[0].content.contains("cannot hear"))
    }

    @Test func textOnlyTurnsAreUntouchedEitherWay() {
        for audioActive in [true, false] {
            let messages = toLLMCommonMessages(
                [.user(content: "plain")], audioActive: audioActive)
            #expect(messages[0].content == "plain")
            #expect(messages[0].audios.isEmpty)
        }
    }

    /// An undecodable persisted blob drops from the model input leniently —
    /// the turn still renders, mirroring undecodable images.
    @Test func undecodableTakeDropsFromModelInput() {
        let corrupt = AudioAttachment(data: Data("junk".utf8), duration: 1)
        let messages = toLLMCommonMessages(
            [.user(content: "say", audios: [corrupt])], audioActive: true)
        #expect(messages[0].audios.isEmpty)
        #expect(messages[0].content == "say")
    }

    // MARK: - Persistence

    @Test func userMessageAudiosRoundTripThroughCodable() throws {
        let original = UserMessage(
            content: "", audios: [Self.takeAttachment()], injectedContext: nil)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(UserMessage.self, from: data)
        #expect(decoded.audios.count == 1)
        #expect(decoded.audios[0].data == original.audios[0].data)
        #expect(decoded == original)
    }

    /// Conversations persisted before the audio field must keep decoding.
    @Test func legacyUserMessagesDecodeWithoutTheAudiosKey() throws {
        let legacy = """
            {"id":"\(UUID().uuidString)","content":"hi",
             "timestamp":774663000.0}
            """
        let decoded = try JSONDecoder().decode(UserMessage.self, from: Data(legacy.utf8))
        #expect(decoded.audios.isEmpty)
        #expect(decoded.content == "hi")
    }

    /// The memory wrapper preserves attachments — the enrichment path
    /// rebuilds the message and must not drop the take.
    @Test func injectedContextKeepsAudios() throws {
        let message = UserMessage(
            content: "what did I say", audios: [Self.takeAttachment()],
            injectedContext: "<memory>owner likes tea</memory>")
        let llm = try #require(message.toLLMMessage())
        guard case .user(let content, _, let audios) = llm else {
            Issue.record("Expected .user")
            return
        }
        #expect(content.contains("<memory>"))
        #expect(audios.count == 1)
    }
}
