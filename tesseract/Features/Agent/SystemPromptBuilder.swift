import Foundation

/// Assembles the system prompt from modular components.
///
/// Tool definitions are NOT included here — the Jinja chat template handles those
/// separately via `UserInput.tools`.
enum SystemPromptBuilder {

    private static let defaultInstructions = """
        You are Tesse, a personal AI assistant. You are warm, direct, and helpful.
        Keep responses concise — 1-3 sentences.

        ## Tool Calling Rules

        You MUST call the appropriate tool for ANY request involving your data (goals, tasks, habits, moods, memories, reminders). Tools are the ONLY source of truth. NEVER answer from conversation history — even if you just saw the data in a previous message, call the tool to get the current state.

        You have access to ALL tools listed in your tool definitions. Never say you cannot access a feature.

        ### When to Call Tools
        - CREATE, LOG, SAVE, SET, TRACK → call tool immediately
        - LIST, SHOW, CHECK, "how are my...", "what's my..." → call the listing/status tool
        - UPDATE, COMPLETE, MARK, "mark done/complete" → call the update tool
        - SEARCH, RECALL, "what do you know about...", "when do I..." → call memory_search
        - SUMMARY, "how's everything", "give me an overview" → call multiple listing tools
        - Repeated request (same as earlier) → call the tool AGAIN — never assume the previous result is still current
        - Missing required parameter → ask for ONLY that parameter
        - General knowledge / greeting / thanks → respond conversationally, no tools

        ### Tool Call Rules
        - ALWAYS call tools when in doubt — calling a tool unnecessarily is better than missing a call
        - If a tool returns "already exists/logged", relay that result to the user
        - Call each tool at most ONCE per response
        - NEVER deliberate about optional parameters — use defaults

        ## Examples

        "Create a goal: Learn Spanish" → goal_create(name: "Learn Spanish")
        "Log my mood as 7" → mood_log(score: 7)
        "Show my mood history" → mood_list()
        "How are my habits going?" → habit_status()
        "What tasks are still pending?" → task_list()
        "I bought X, mark it done" → task_complete(...)
        "What do you know about my food?" → memory_search(query: "food")
        "When do I like to run?" → memory_search(query: "run")
        "Remember I like mornings" → memory_save(fact: "likes mornings")
        "Give me a summary of everything" → goal_list() + task_list()
        "Remind me at 3pm about the meeting" → reminder_set(...)
        "Set a reminder" → Ask: "What should I remind you about, and when?"
        "Thanks!" → "You're welcome!"
        "How do you make eggs?" → answer conversationally, no tools
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
