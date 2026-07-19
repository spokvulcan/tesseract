import Foundation

/// The turn-class vocabulary (#327 §2, ADR-0040 §8) — and, since ADR-0046, the
/// conversation-kind tag. `interactive` is the owner typing; `missionControl`
/// is the one standing conversation the loop folds into (a conversation kind,
/// never a turn class — no turn is "opened by" it); the rest are the
/// Companion's turn classes, which since ADR-0046 tag the opening message of
/// each turn inside Mission Control rather than a conversation of their own.
/// Raw values are the persisted tag — the store and the index keep plain
/// strings so pre-tag files load unchanged.
nonisolated enum TurnOrigin: String, Codable, Sendable {
    /// The owner's own typed (or spoken) chat — never badged.
    case interactive
    /// A rhythm beat's turn — the fired batch carried a rhythm-class wake.
    case beat
    /// A booked wake fired on time (promise, follow-up, re-summons).
    case wake
    /// A fold turn granted by pending Events alone — no wake due (#371).
    case event
    /// RETIRED (ADR-0046 #371): the unoccasioned cadence-granted turn died
    /// with the Event Fold. The case survives only so historical transcripts
    /// keep their tag; nothing emits it.
    case ambient
    /// Overdue wakes triaged late (past the catch-up grace).
    case catchup
    /// Reserved (#327 §2): sleep passes ride `internalCompletion`, not the
    /// turn machinery, so nothing emits this yet — the tag waits for them.
    case sleep
    /// Mission Control (ADR-0046): the fold's standing conversation. Tags the
    /// conversation, never a turn — turns inside it carry their own class on
    /// their opening message (`UserMessage.turnOrigin`).
    case missionControl = "mission-control"
    /// A summoned dialogue (ADR-0046 #372): the separate chat a summons
    /// engagement opens, voice or typed — the loop's sub-agent conversation
    /// that owes Mission Control a Report-Back. A conversation kind like
    /// `missionControl`, never a turn class.
    case dialogue

    /// The lenient read of a persisted tag: nil, or an unknown future tag,
    /// reads as nil instead of failing the whole file's decode — pre-tag
    /// histories load unchanged. The one home of the rule; message decode and
    /// both conversation-level readers go through it.
    init?(persisted raw: String?) {
        guard let raw, let origin = TurnOrigin(rawValue: raw) else { return nil }
        self = origin
    }

    /// Whether launch recency may land on this conversation kind. The fold is
    /// excluded: the loop appends to it around the clock, so it would win
    /// recency nearly always, and launch must open on the owner's own last
    /// chat (ADR-0046). The one home of the rule — the real store and the
    /// in-memory test fixture both filter on it.
    var opensAtLaunch: Bool { self != .missionControl }

    /// Whether launch-time index validation fully parses this kind's backing
    /// file. The fold is exempt: validating means parsing a file that grows
    /// all day, on every launch — and a corrupt fold already degrades
    /// gracefully (`missionControl()` re-seeds it empty).
    var validatesAtLaunch: Bool { self != .missionControl }
}

struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol & Sendable]
    let createdAt: Date
    var updatedAt: Date
    /// Which turn class opened this conversation (#327's one-interface tag).
    var origin: TurnOrigin

    /// Mission Control's identity is a constant of the domain (ADR-0046):
    /// there is exactly one standing conversation, so the loop finds it by id
    /// across relaunches, and deleting it just means the next turn re-seeds it
    /// empty under the same id — no index scan, no second source of truth.
    nonisolated static let missionControlID = UUID(
        uuidString: "AD460046-0367-4366-B301-000000000001")!

    /// Whether this is the fold's standing conversation (ADR-0046).
    var isMissionControl: Bool { origin == .missionControl }

    /// Derive title from first user message content. Mission Control's name is
    /// fixed — its first message is a turn opening (instructions + briefing),
    /// which would make a meaningless title.
    var title: String {
        if isMissionControl { return "Mission Control" }
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
