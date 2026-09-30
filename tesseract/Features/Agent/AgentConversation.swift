import CryptoKit
import Foundation

/// The conversation-kind and turn-class tag. Raw values are the persisted
/// tag; the store and the index keep plain strings, so files written before a
/// tag existed (or with a retired tag) load as ordinary chats.
nonisolated enum TurnOrigin: String, Codable, Sendable {
    /// The owner's own typed (or spoken) chat, never badged.
    case interactive
    /// A Day Thread: the one append-only conversation per day that the Today
    /// page's chat shows. A conversation kind, never a turn class.
    case dayThread = "day-thread"
    /// A Companion moment's request inside a Day Thread (Morning Plan,
    /// Breakpoint, Triage, Evening Wrap-up, Night Reflection). Tags the
    /// message, never a conversation.
    case moment

    /// The lenient read of a persisted tag: nil, or an unknown or retired tag
    /// (the old Companion's `wake`, `beat`, `mission-control`, `dialogue`…),
    /// reads as nil instead of failing the whole file's decode.
    init?(persisted raw: String?) {
        guard let raw, let origin = TurnOrigin(rawValue: raw) else { return nil }
        self = origin
    }

    /// Whether launch recency may land on this conversation kind on the Agent
    /// page. Day Threads belong to Today; the Agent page opens on the owner's
    /// own last chat.
    var opensAtLaunch: Bool { self == .interactive }
}

struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol & Sendable]
    let createdAt: Date
    var updatedAt: Date
    /// Which kind of conversation this is.
    var origin: TurnOrigin

    /// The retired Mission Control conversation's id. The one-time migration
    /// deletes it, and the store never lists it.
    nonisolated static let retiredMissionControlID = UUID(
        uuidString: "AD460046-0367-4366-B301-000000000001")!

    /// A Day Thread's id is a pure function of its day, so the Companion finds
    /// the day's thread across relaunches without scanning the index.
    nonisolated static func dayThreadID(for day: DayKey) -> UUID {
        var bytes = Array(
            Insecure.SHA1.hash(data: Data("tesseract.day-thread.\(day.rawValue)".utf8)))
        bytes[6] = (bytes[6] & 0x0F) | 0x50  // version 5
        bytes[8] = (bytes[8] & 0x3F) | 0x80  // RFC 4122 variant
        return UUID(
            uuid: (
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
            ))
    }

    var isDayThread: Bool { origin == .dayThread }

    /// Derive title from first user message content. A Day Thread's name is
    /// its date: its first message is the Day Opening, not the owner's words.
    var title: String {
        if isDayThread {
            return "Today · " + createdAt.formatted(.dateTime.weekday(.wide).day().month(.wide))
        }
        for msg in messages {
            if let user = msg.asUser {
                let text = user.content.prefix(80)
                return text.isEmpty ? "New Conversation" : String(text)
            }
        }
        return "New Conversation"
    }

    var messageCount: Int { messages.count }

    init(
        id: UUID = UUID(),
        messages: [any AgentMessageProtocol & Sendable] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        origin: TurnOrigin = .interactive
    ) {
        self.id = id
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.origin = origin
    }
}

/// Lightweight summary for the conversation index (avoids loading full message history).
struct AgentConversationSummary: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int
    /// Raw string, optional, so a pre-tag index decodes unchanged; nil (or an
    /// unknown tag) reads as interactive.
    var origin: String?

    /// The typed view of the raw tag.
    var turnOrigin: TurnOrigin { TurnOrigin(persisted: origin) ?? .interactive }

    init(from conversation: AgentConversation) {
        self.id = conversation.id
        self.title = conversation.title
        self.createdAt = conversation.createdAt
        self.updatedAt = conversation.updatedAt
        self.messageCount = conversation.messageCount
        self.origin = conversation.origin.rawValue
    }

    init(
        id: UUID, title: String, createdAt: Date, updatedAt: Date, messageCount: Int,
        origin: String? = nil
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messageCount = messageCount
        self.origin = origin
    }
}
