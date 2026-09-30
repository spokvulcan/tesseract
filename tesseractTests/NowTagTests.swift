//
//  NowTagTests.swift
//  tesseractTests
//
//  The Now Tag at the agent-message rendering seam: every user message the
//  model sees starts with the local date, weekday, time and time zone, and a
//  message renders the same bytes every time it is replayed.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct NowTagTests {

    /// 2026-09-30 12:05:00 UTC.
    private let noonUTC = Date(timeIntervalSince1970: 1_790_769_900)

    @Test(arguments: [
        (
            "Europe/Berlin",
            "<now>Wednesday, 30 September 2026, 14:05, Europe/Berlin (UTC+02:00)</now>"
        ),
        (
            "America/New_York",
            "<now>Wednesday, 30 September 2026, 08:05, America/New_York (UTC-04:00)</now>"
        ),
        (
            "Asia/Kolkata",
            "<now>Wednesday, 30 September 2026, 17:35, Asia/Kolkata (UTC+05:30)</now>"
        ),
        // Foundation names the zero-offset zone "GMT".
        ("GMT", "<now>Wednesday, 30 September 2026, 12:05, GMT (UTC)</now>"),
    ])
    func rendersLocalDateWeekdayTimeAndZone(zone: String, expected: String) throws {
        let timeZone = try #require(TimeZone(identifier: zone))
        #expect(NowTag.render(noonUTC, timeZone: timeZone) == expected)
    }

    /// Every user message carries the tag on its own first line, followed by
    /// the owner's words unchanged.
    @Test func everyUserMessageCarriesTheTag() throws {
        let message = UserMessage(content: "What's next today?", timestamp: noonUTC)
        guard case .user(let content, _) = try #require(message.toLLMMessage()) else {
            Issue.record("a user message must render as a user turn")
            return
        }
        #expect(content == "\(NowTag.render(noonUTC))\nWhat's next today?")
        #expect(content.hasPrefix("<now>"))
    }

    /// The tag is stored with the message: it survives a round trip byte for
    /// byte, even when the Mac's time zone has changed since, so a replayed
    /// conversation keeps its cached prefix.
    @Test func storedTagSurvivesRoundTrip() throws {
        let berlin = try #require(TimeZone(identifier: "Europe/Berlin"))
        let message = UserMessage(
            content: "hi", timestamp: noonUTC, nowTag: NowTag.render(noonUTC, timeZone: berlin))
        let data = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(UserMessage.self, from: data)
        #expect(decoded.nowTag == message.nowTag)
        #expect(decoded.toLLMMessage() == message.toLLMMessage())
    }

    /// A message saved before the tag existed gets one from its own timestamp,
    /// not from the time it was reopened.
    @Test func legacyMessageGetsATagFromItsOwnTime() throws {
        let legacy = """
            {"id":"\(UUID().uuidString)","content":"old words","images":[],"timestamp":\(noonUTC.timeIntervalSinceReferenceDate)}
            """
        let decoded = try JSONDecoder().decode(UserMessage.self, from: Data(legacy.utf8))
        #expect(decoded.nowTag == NowTag.render(noonUTC))
    }
}
