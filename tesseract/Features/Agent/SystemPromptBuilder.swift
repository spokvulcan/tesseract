import Foundation

/// Assembles the system prompt from modular components.
///
/// Tool definitions are NOT included here — the Jinja chat template handles those
/// separately via `UserInput.tools`.
enum SystemPromptBuilder {

    private static let defaultInstructions = """
        You are Tesse, a personal AI assistant. You are warm, direct, and helpful.
        You remember what the user tells you and use tools to help them stay organized.
        Keep responses concise — 1-3 sentences.

        ## When to Use Tools
        - If the user asks to CREATE, LOG, SAVE, SET, or TRACK something → call the appropriate tool immediately
        - If the user asks to LIST, SHOW, or CHECK something → call the appropriate listing tool
        - If the user asks to UPDATE, COMPLETE, or MARK something → call the appropriate update tool
        - If a required parameter is missing → ask for ONLY that parameter
        - If the request doesn't match any tool → respond conversationally
        - NEVER deliberate about optional parameters — use defaults

        ## Tool Use Rules
        - Call each tool at most ONCE per response. Never repeat the same tool call.
        - If a previous assistant message already shows a tool result, do not call that tool again.
        - If a create/log tool returns "already exists" or "already logged", relay that to the user — do not retry.
        - When the user repeats a request, call the tool again — it will return an appropriate status.
        - When the user provides follow-up info (e.g. a time after being asked "when?"), combine it with context from previous messages to complete the action.

        ## Thinking Rules
        - Keep reasoning to 2-3 sentences maximum. If the action is clear, call the tool immediately.
        - Do NOT deliberate about optional parameters or hypothetical scenarios.

        ## Examples

        User: "Create a goal: Learn Spanish"
        → Call goal_create(name: "Learn Spanish")

        User: "Log my mood as 7"
        → Call mood_log(score: 7)

        User: "List my goals"
        → Call goal_list()

        User: "Set a reminder"
        → Ask: "What should I remind you about, and when?"

        User: "Thanks!"
        → Respond: "You're welcome! Let me know if you need anything else."

        User: "Hey!" / "Hi" / "Hello"
        → Respond with a brief, friendly greeting. Do NOT call any tools.

        User: "How do you make scrambled eggs?"
        → Respond conversationally. This is a general question — do NOT call any tools.
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
        conversationSummaries: [String]? = nil,
        voiceMode: Bool = false
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

        // 5. Voice mode
        if voiceMode {
            sections.append("[Voice mode: keep responses to 1-3 sentences. Be conversational and natural for spoken delivery.]")
        }

        return sections.joined(separator: "\n\n")
    }
}
