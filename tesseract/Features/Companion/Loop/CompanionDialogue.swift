//
//  CompanionDialogue.swift
//  tesseract
//
//  The conversation ledger (ADR-0046 #372, widened by ADR-0052): dialogue
//  out, Report-Back in. Every owner conversation — a summons-minted
//  dialogue or the owner's own chat, voice or typed — owes Mission Control
//  a deposit through `report_back`, and this type tracks the active one's
//  debt. A conversation that ends or goes quiet without depositing gets
//  exactly one harness nudge, on the record; what the deposit says, and
//  whether one is warranted at all, remains the entity's judgment — the
//  harness only asks.
//

import Foundation

@MainActor
final class CompanionDialogue {

    /// A dialogue with no exchange for this long has gone quiet — the one
    /// nudge fires so its conclusions don't silently vanish from the fold.
    static let quietAfter: TimeInterval = 10 * 60
    /// How often the quiet watch looks. Coarse on purpose: the nudge is a
    /// courtesy deadline, not an endpointer.
    static let quietCheckInterval: Duration = .seconds(60)

    /// The harness's one nudge — a plain ask, rendered in the dialogue's own
    /// transcript (that visibility IS the record's other half).
    static let nudgeMessage = """
        <harness-nudge>
        This conversation is winding down. If it concluded anything Mission \
        Control should know — a decision, a promise, anything owed — deposit it \
        now with report_back. If nothing concluded, say so briefly; never invent \
        a deposit.
        </harness-nudge>
        """

    private let recorder: CompanionFlightRecorder
    /// Mints the dialogue conversation (seeded with the summons line) and
    /// presents it — `ChatSession.beginDialogue` behind the container's door.
    private let openDialogue: (String?) -> UUID?
    /// A foreground run is queued or active — the nudge waits it out
    /// (bounded) before sending.
    private let isAgentBusy: () -> Bool
    /// The chat the owner is looking at — the nudge only lands while the
    /// dialogue is still current; the harness never hijacks his screen.
    private let currentConversationID: () -> UUID?
    /// Sends the nudge text into the current chat.
    private let sendNudge: (String) -> Void

    private(set) var activeConversationID: UUID?
    /// The owner actually said something — an engaged-then-abandoned summons
    /// with zero exchanges owes nothing and is never nudged.
    private var hadExchange = false
    private var deposited = false
    private var nudged = false
    private var lastActivityAt: Date?
    private var quietWatch: Task<Void, Never>?

    init(
        recorder: CompanionFlightRecorder,
        openDialogue: @escaping (String?) -> UUID?,
        isAgentBusy: @escaping () -> Bool,
        currentConversationID: @escaping () -> UUID?,
        sendNudge: @escaping (String) -> Void
    ) {
        self.recorder = recorder
        self.openDialogue = openDialogue
        self.isAgentBusy = isAgentBusy
        self.currentConversationID = currentConversationID
        self.sendNudge = sendNudge
    }

    // MARK: - Lifecycle

    /// A summons engagement opens the dialogue: mint the conversation (the
    /// summons line seeds it as the entity's own first words) and arm the
    /// ledger. One dialogue at a time — a new summons supersedes the last.
    func begin(line: String?, via: String) {
        guard let id = openDialogue(line) else { return }
        if let previous = activeConversationID, previous != id, hadExchange, !deposited {
            recorder.record(.dialogueSuperseded, conversationID: previous)
        }
        arm(on: id)
        recorder.record(.dialogueBegan, conversationID: id, snapshot: ["via": via])
    }

    /// Every send into a dialogue-origin conversation lands here (the chat
    /// session's door). A dialogue reopened from history arms the ledger the
    /// moment he talks in it — same debts, same one nudge.
    func activity(in conversationID: UUID) {
        if activeConversationID != conversationID { arm(on: conversationID) }
        hadExchange = true
        lastActivityAt = Date()
    }

    /// The `report_back` tool's flag: the debt is settled — this dialogue is
    /// never nudged. Further milestone deposits are welcome and change nothing
    /// here.
    func depositLanded(in conversationID: UUID?) {
        guard
            conversationID == nil || conversationID == activeConversationID
                || activeConversationID == nil
        else { return }
        deposited = true
        quietWatch?.cancel()
        quietWatch = nil
    }

    /// The voice session ended — the dialogue's "end" trigger. (A typed
    /// dialogue has no end button; the quiet watch is its trigger.) Returns
    /// the nudge task so tests can await the outcome.
    @discardableResult
    func voiceSessionEnded() -> Task<Void, Never> {
        Task { await attemptNudge(reason: "voice-session-ended") }
    }

    // MARK: - The one nudge

    private func arm(on conversationID: UUID) {
        activeConversationID = conversationID
        hadExchange = false
        deposited = false
        nudged = false
        lastActivityAt = Date()
        startQuietWatch()
    }

    private func startQuietWatch() {
        quietWatch?.cancel()
        quietWatch = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: CompanionDialogue.quietCheckInterval)
                guard let self, !Task.isCancelled else { return }
                if let last = self.lastActivityAt,
                    Date().timeIntervalSince(last) >= CompanionDialogue.quietAfter
                {
                    await self.attemptNudge(reason: "quiet")
                    return
                }
            }
        }
    }

    /// Exactly one nudge per dialogue, delivered or not: a missed delivery
    /// (he already switched away) is recorded and the debt is let go — the
    /// harness asks once, it never nags.
    private func attemptNudge(reason: String) async {
        guard let conversationID = activeConversationID, hadExchange, !deposited, !nudged
        else { return }
        nudged = true
        quietWatch?.cancel()
        quietWatch = nil

        // An exiting session's last reply may still be settling — give it a
        // bounded moment before deciding the nudge can't land.
        var waited = 0
        while isAgentBusy(), waited < 60 {
            try? await Task.sleep(for: .milliseconds(500))
            waited += 1
        }
        let delivered =
            !isAgentBusy() && currentConversationID() == conversationID && !deposited
        if delivered { sendNudge(Self.nudgeMessage) }
        recorder.record(
            delivered ? .dialogueNudged : .dialogueNudgeMissed,
            conversationID: conversationID,
            snapshot: ["reason": reason])
    }
}
