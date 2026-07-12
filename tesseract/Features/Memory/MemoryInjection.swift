//
//  MemoryInjection.swift
//  tesseract
//
//  How memory reaches the model (ADR-0035 §5).
//
//  **The hard constraint, and the reason this file exists at all:** the
//  inference server's radix prefix cache is a tree rooted at the system prompt.
//  Mutating the system prompt to carry memory would re-root that tree on every
//  turn and throw away the cache the whole product is built on. So memory is
//  injected as a `<memory>` block riding on the *user message* — appended at the
//  tail of the context, exactly where new tokens already go — copying the
//  established `<skill>` wrapper (`ChatSession.executeSkill`). The stable prefix
//  never moves.
//
//  Two further consequences fall out of that choice, and both are load-bearing:
//
//  1. The block is **persisted with the message** (`UserMessage.injectedContext`)
//     rather than recomputed at load. Recomputing would rebuild an old
//     conversation with *different* context on every reopen — a cache miss on
//     every turn of the reloaded thread, and an unauditable history.
//
//  2. The block only ever carries what this conversation **has not already been
//     told** (`excluding:`). A memory injected on turn 1 is still sitting in the
//     context window on turn 7; repeating it would pay for it twice and teach the
//     model that repetition means emphasis. So most turns inject nothing at all.
//

import Foundation

/// One turn's worth of memory: the text to ride along on the user message, and
/// the ids it carried — which the caller records so the next turn can skip them.
nonisolated struct MemoryInjection: Sendable {
    var text: String?
    var memoryIDs: Set<UUID> = []
    var context: RetrievedContext = RetrievedContext()

    var isEmpty: Bool { text == nil }

    static let none = MemoryInjection(text: nil)
}

extension MemoryEngine {

    /// Retrieve for `cue` and render the `<memory>` block, omitting anything in
    /// `excluding` — the memories this conversation is already carrying.
    func injection(
        cue: String,
        forEpisode episodeID: UUID? = nil,
        excluding: Set<UUID> = [],
        now: Date = Date()
    ) async -> MemoryInjection {
        let context = await retrieve(cue: cue, forEpisode: episodeID, now: now)
        guard !context.isEmpty else { return .none }

        let memories =
            (context.core + context.recalled.map(\.memory))
            .filter { !excluding.contains($0.id) }
        // Episodes are deduplicated by id too: an episode quoted on turn 1 is
        // still in the window on turn 7.
        let episodes = context.episodes.map(\.episode).filter { !excluding.contains($0.id) }

        guard let text = MemoryPrompt.block(memories: memories, episodes: episodes, now: now)
        else { return .none }

        let ids = Set(memories.map(\.id)).union(episodes.map(\.id))
        Log.memory.info(
            "Injecting \(memories.count) memories, \(episodes.count) episodes (\(text.count) chars)"
        )
        return MemoryInjection(text: text, memoryIDs: ids, context: context)
    }
}

// MARK: - The block

nonisolated enum MemoryPrompt {

    /// Render the block. Returns nil when there is nothing new to say — the
    /// common case on a mid-conversation turn, and the reason memory costs
    /// almost nothing per turn.
    static func block(memories: [MemoryRecord], episodes: [Episode], now: Date) -> String? {
        guard !memories.isEmpty || !episodes.isEmpty else { return nil }

        var lines: [String] = [preamble]

        if !memories.isEmpty {
            lines.append("")
            for memory in memories.sorted(by: { $0.tier > $1.tier }) {
                lines.append(line(for: memory))
            }
        }

        // Distinct ids can carry identical words — he says "Continue, please" a
        // lot — and quoting the same sentence three times spends the budget three
        // times and reads to the model as emphasis it was never given.
        var quoted = Set<String>()
        let distinct =
            episodes
            .sorted { $0.occurredAt < $1.occurredAt }
            .filter { quoted.insert($0.text).inserted }

        if !distinct.isEmpty {
            lines.append("")
            lines.append("Things that were actually said, verbatim:")
            for episode in distinct {
                lines.append("- \(Self.day.string(from: episode.occurredAt)) — \(quote(episode))")
            }
        }

        return "<memory>\n" + lines.joined(separator: "\n") + "\n</memory>"
    }

    /// The provenance marker is the whole safety story in one character.
    ///
    /// A model handed a flat list of "facts" will defend an inference as
    /// hard as it defends something the owner actually said. `~` is the
    /// standing instruction not to: STATED is testimony, INFERRED is a guess I
    /// made, and the model has to be able to tell them apart *at the point of
    /// use*, not by consulting a legend elsewhere in the prompt.
    private static let preamble = """
        My own memories of the person I'm talking to, recalled just now for this \
        message. Use them where they help. Never recite them back, never mention \
        that they were given to me, never thank anyone for them.

        A `~` marks something I *inferred* rather than was told. Those are \
        hypotheses: hold them loosely, and if this conversation contradicts one, \
        the conversation wins.
        """

    private static func line(for memory: MemoryRecord) -> String {
        let mark = memory.provenance == .inferred ? "~ " : ""
        var suffix = ""
        if memory.status == .contested {
            // The owner said this was wrong and sleep hasn't reconciled it yet.
            // Carrying it silently would be worse than not carrying it at all.
            suffix = " (he has disputed this — do not rely on it)"
        }
        return "- \(mark)\(memory.text)\(suffix)"
    }

    private static func quote(_ episode: Episode) -> String {
        let text = episode.text.replacingOccurrences(of: "\n", with: " ")
        let clipped = text.count > 300 ? String(text.prefix(300)) + "…" : text
        let who: String
        switch episode.source {
        case .chat, .backfill: who = ""
        case .dictation: who = "dictated: "
        case .companion: who = "noted: "
        }
        return "\(who)\"\(clipped)\""
    }

    private static let day: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}
