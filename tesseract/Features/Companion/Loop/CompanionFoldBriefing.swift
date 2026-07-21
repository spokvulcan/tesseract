//
//  CompanionFoldBriefing.swift
//  tesseract
//
//  The Fold Briefing (ADR-0052): every owner conversation opens as the one
//  mind. A code-rendered account of the fold's recent life — today's
//  contract, due and recently-fired wakes, recent deliveries, the entity's
//  last fold-turn conclusions verbatim — rides the conversation's outgoing
//  message on `injectedContext`, beside `<jarvis-identity>` and `<memory>`
//  (the ADR-0045 rule: the transcript records exactly what the turn saw).
//
//  Gathering is mechanical; interpreting is the entity's job — the
//  Situation Briefing's discipline, pointed at the chat side. Unlike
//  identity's once-per-conversation block, the briefing re-injects when the
//  fold advanced past the last briefing's stamp: a resumed chat is never
//  behind its own mind.
//

import Foundation

@MainActor
final class CompanionFoldBriefing {

    /// The transcript-scan marker — also the re-brief stamp's carrier.
    nonisolated static let blockMarker = "<fold-briefing"
    /// Deliveries older than this stop riding the briefing; conclusions are
    /// count-capped instead (the last thing on his mind stays relevant no
    /// matter how quiet the night was).
    nonisolated static let deliveryWindow: TimeInterval = 24 * 3600
    nonisolated static let maxConclusions = 3
    nonisolated static let maxDeliveries = 5

    private let store: MemoryStore
    /// The fold's standing conversation — the store's warm cache, so a
    /// per-send stamp check costs no disk after the first read.
    private let missionControl: () -> AgentConversation?
    private let isEnabled: () -> Bool
    private let now: () -> Date
    /// The fold stamp this conversation was last briefed at. The transcript
    /// scan is the durable half (a reopened chat carries its own stamp);
    /// this is the cheap in-session half.
    private var lastBriefedStamp: Date?

    init(
        store: MemoryStore,
        missionControl: @escaping () -> AgentConversation?,
        isEnabled: @escaping () -> Bool,
        now: @escaping () -> Date = { Date() }
    ) {
        self.store = store
        self.missionControl = missionControl
        self.isEnabled = isEnabled
        self.now = now
    }

    /// Decorate the outgoing message with the fold briefing — on the first
    /// message, and again whenever the fold advanced since this
    /// conversation's last briefing.
    func decorate(
        _ user: UserMessage, transcript: [any AgentMessageProtocol & Sendable]
    ) async -> UserMessage {
        guard isEnabled(), let fold = missionControl(), !fold.messages.isEmpty
        else { return user }
        let stamp = fold.updatedAt
        let known = [lastBriefedStamp, Self.stampInTranscript(transcript)]
            .compactMap { $0 }.max()
        // Compare at the encoded (millisecond) granularity: the stamp a
        // transcript carries went through `%.3f`, and full-precision
        // `updatedAt` must not read as "newer" than its own round-trip.
        if let known, Self.quantized(stamp) <= Self.quantized(known) { return user }
        let block = await render(fold: fold, stamp: stamp)
        lastBriefedStamp = stamp
        let combined = [block, user.injectedContext].compactMap { $0 }
            .joined(separator: "\n\n")
        return user.with(injectedContext: combined)
    }

    /// The pipeline door — `decoratingUser` is the shared wrapper-restore
    /// contract.
    func decorate(
        _ outgoing: any AgentMessageProtocol & Sendable,
        transcript: [any AgentMessageProtocol & Sendable]
    ) async -> any AgentMessageProtocol & Sendable {
        await outgoing.decoratingUser { await decorate($0, transcript: transcript) }
    }

    /// A conversation switch — the next conversation re-derives its stamp
    /// from its own transcript.
    func reset() {
        lastBriefedStamp = nil
    }

    // MARK: - Rendering

    private func render(fold: AgentConversation, stamp: Date) async -> String {
        let now = now()
        let todayKey = TrackingDay.key(for: now)
        let today = (try? await store.day(todayKey)) ?? nil
        let due = (try? await store.dueWakes(asOf: now)) ?? []
        let fired =
            (try? await store.recentWakeActivity(
                since: now.addingTimeInterval(-Self.deliveryWindow))) ?? []
        let upcoming = (try? await store.upcomingWakes(after: now)) ?? []
        let activity = Self.extractActivity(
            from: fold.messages,
            deliveriesSince: now.addingTimeInterval(-Self.deliveryWindow))

        var lines: [String] = []
        lines.append(
            "Your own fold's recent life (Mission Control), as of "
                + "\(now.formatted(date: .abbreviated, time: .shortened)):")
        lines.append(CompanionFoldRender.contractLine(today: today))
        if !due.isEmpty {
            lines.append(CompanionFoldRender.dueHeading(triggeredThisTurn: false))
            for wake in due {
                lines.append(CompanionFoldRender.dueWakeLine(wake))
            }
        }
        if !fired.isEmpty {
            lines.append("Recently fired:")
            for wake in fired {
                lines.append(CompanionFoldRender.firedWakeLine(wake))
            }
        }
        if !upcoming.isEmpty {
            lines.append(CompanionFoldRender.upcomingHeading)
            for wake in upcoming {
                lines.append(CompanionFoldRender.upcomingWakeLine(wake))
            }
        }
        if !activity.deliveries.isEmpty {
            lines.append("Recent deliveries to the owner:")
            lines.append(contentsOf: activity.deliveries)
        }
        if !activity.conclusions.isEmpty {
            lines.append("Your last fold conclusions:")
            lines.append(contentsOf: activity.conclusions)
        }
        lines.append(
            "This conversation is one of your own (ADR-0052): if it concludes "
                + "anything Mission Control should know — a decision, a promise, "
                + "anything owed — deposit it with report_back before it ends.")
        let body = lines.joined(separator: "\n")
        return "\(Self.blockMarker) at=\"\(Self.encodeStamp(stamp))\">\n\(body)\n</fold-briefing>"
    }

    // MARK: - Fold transcript extraction (mechanical: the entity's own
    // words and delivery tool calls, verbatim — no summarizing)

    nonisolated static func extractActivity(
        from messages: [any AgentMessageProtocol & Sendable], deliveriesSince: Date
    ) -> (deliveries: [String], conclusions: [String]) {
        var deliveries: [String] = []
        var conclusions: [String] = []
        for message in messages {
            guard let assistant = message.asAssistant else { continue }
            let text = assistant.text.trimmingCharacters(in: .whitespacesAndNewlines)
            // "Silence." is a recorded decision, not a conclusion worth
            // re-reading — count-capped conclusions would otherwise be
            // nothing else on a quiet day.
            if !text.isEmpty, text != "Silence." {
                let when = assistant.timestamp.formatted(date: .omitted, time: .shortened)
                conclusions.append("- \(when): \(text)")
            }
            guard assistant.timestamp >= deliveriesSince else { continue }
            for call in assistant.toolCalls {
                guard let line = call.deliveryLine else { continue }
                let when = assistant.timestamp.formatted(date: .omitted, time: .shortened)
                deliveries.append("- \(when) \(line)")
            }
        }
        return (
            deliveries.suffix(maxDeliveries).map { $0 },
            conclusions.suffix(maxConclusions).map { $0 }
        )
    }

    // MARK: - Stamp round-trip

    /// Epoch seconds at millisecond precision — survives the transcript
    /// round-trip without ISO8601's fractional-second truncation re-briefing
    /// spuriously.
    private nonisolated static func encodeStamp(_ date: Date) -> String {
        String(format: "%.3f", date.timeIntervalSince1970)
    }

    nonisolated static func quantized(_ date: Date) -> Double {
        (date.timeIntervalSince1970 * 1000).rounded() / 1000
    }

    nonisolated static func stampInTranscript(
        _ transcript: [any AgentMessageProtocol & Sendable]
    ) -> Date? {
        for message in transcript.reversed() {
            guard
                let injected = message.asUser?.injectedContext,
                let range = injected.range(of: "\(blockMarker) at=\"", options: .backwards)
            else { continue }
            let rest = injected[range.upperBound...]
            guard let end = rest.firstIndex(of: "\""), let value = Double(rest[..<end])
            else { continue }
            return Date(timeIntervalSince1970: value)
        }
        return nil
    }
}

extension ToolCallInfo {
    /// The owner-facing line inside a delivery-ladder call — `nil` for any
    /// call that isn't a delivery.
    fileprivate nonisolated var deliveryLine: String? {
        let key: String
        switch name {
        case "notify": key = "body"
        case "speak": key = "text"
        case "summon_overlay": key = "line"
        default: return nil
        }
        guard
            let data = argumentsJSON.data(using: .utf8),
            let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let line = args[key] as? String
        else { return nil }
        return "\(name): \"\(line)\""
    }
}
