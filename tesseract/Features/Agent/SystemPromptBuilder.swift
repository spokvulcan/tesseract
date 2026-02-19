import Foundation

/// Assembles the system prompt from modular components.
///
/// Tool definitions are NOT included here — the Jinja chat template handles those
/// separately via `UserInput.tools`.
enum SystemPromptBuilder {

    private static let defaultInstructions = """
        You are Tesse, a personal AI assistant. You are warm, direct, and helpful.
        You remember what the user tells you and use tools to help them stay organized.
        Keep responses concise — 1-3 sentences for simple questions, longer only when needed.
        Ask clarifying questions when the user's request is ambiguous.

        ## Tool Use Rules
        - Call each tool at most ONCE per response. Never repeat the same tool call.
        - If a previous assistant message already shows a tool result, do not call that tool again.
        - For listing tools (list_goals, list_tasks, list_moods, habit_status): only call once per response.
        - If a create/log tool returns "already exists" or "already logged", just inform the user — do not retry.
        - Only call complete_task once per task. Do not chain create_task + complete_task in the same response.
        - When the user repeats a request, you may call the tool again — it will return an appropriate status (e.g. 'already exists').
        """

    /// Assembles the system prompt from modular components.
    ///
    /// - Parameters:
    ///   - instructions: The agent's personality/role instructions. Falls back to
    ///     ``defaultInstructions`` if nil or empty.
    ///   - memories: Retrieved facts about the user (from fact memory store).
    ///   - conversationSummaries: Summaries of recent past conversations.
    static func build(
        instructions: String? = nil,
        memories: [String]? = nil,
        conversationSummaries: [String]? = nil
    ) -> String {
        var sections: [String] = []

        // 1. Instructions
        let inst = instructions?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        sections.append(inst.isEmpty ? defaultInstructions : inst)

        // 2. Memories
        if let memories, !memories.isEmpty {
            let bullets = memories.map { "- \($0)" }.joined(separator: "\n")
            sections.append("## What I Know About You\n\n\(bullets)")
        }

        // 3. Conversation summaries
        if let conversationSummaries, !conversationSummaries.isEmpty {
            let bullets = conversationSummaries.map { "- \($0)" }.joined(separator: "\n")
            sections.append("## Recent Conversations\n\n\(bullets)")
        }

        // 4. Current date/time
        let formatter = DateFormatter()
        formatter.dateStyle = .full
        formatter.timeStyle = .short
        formatter.locale = Locale.current
        sections.append("Current date and time: \(formatter.string(from: Date()))")

        return sections.joined(separator: "\n\n")
    }
}
