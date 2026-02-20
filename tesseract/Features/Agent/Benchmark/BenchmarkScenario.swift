import Foundation

// MARK: - Turn Expectation

/// What we expect the model to do for a given user message.
struct TurnExpectation {
    let userMessage: String
    /// Tool names expected to be called (empty = no tools expected).
    let expectedTools: [String]
    /// Tool names that must NOT be called.
    let forbiddenTools: [String]
    /// Substrings expected in the assistant's text response (case-insensitive).
    let expectedSubstrings: [String]
    /// Required argument key-value pairs per tool (tool name → dict of key → substring match).
    let expectedArguments: [String: [String: String]]
    /// If true, the model should ask for clarification rather than call tools.
    let expectsClarification: Bool
    /// If true, skip tool correctness check entirely (tools are optional for this turn).
    let toolsOptional: Bool

    init(
        _ userMessage: String,
        tools: [String] = [],
        forbidden: [String] = [],
        substrings: [String] = [],
        arguments: [String: [String: String]] = [:],
        expectsClarification: Bool = false,
        toolsOptional: Bool = false
    ) {
        self.userMessage = userMessage
        self.expectedTools = tools
        self.forbiddenTools = forbidden
        self.expectedSubstrings = substrings
        self.expectedArguments = arguments
        self.expectsClarification = expectsClarification
        self.toolsOptional = toolsOptional
    }
}

// MARK: - Scenario Protocol

protocol BenchmarkScenario {
    var id: String { get }
    var description: String { get }
    var turns: [TurnExpectation] { get }
}

// MARK: - S1: Simple Q&A

struct SimpleQAScenario: BenchmarkScenario {
    let id = "S1"
    let description = "Simple Q&A — no tools expected"
    let turns: [TurnExpectation] = [
        TurnExpectation(
            "What's the capital of France?",
            substrings: ["Paris"]
        ),
        TurnExpectation(
            "How do you make scrambled eggs?"
        ),
        TurnExpectation(
            "Thanks!"
        ),
    ]
}

// MARK: - S2: Single Tool Calls

struct SingleToolScenario: BenchmarkScenario {
    let id = "S2"
    let description = "Single tool calls — one tool per turn"
    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Remember that I'm allergic to peanuts",
            tools: ["memory_save"],
            arguments: ["memory_save": ["fact": "peanut"]]
        ),
        TurnExpectation(
            "Log my mood as 7",
            tools: ["mood_log"],
            arguments: ["mood_log": ["score": "7"]]
        ),
        TurnExpectation(
            "Create a habit called meditation, daily",
            tools: ["habit_create"],
            arguments: ["habit_create": ["name": "meditation", "frequency": "daily"]]
        ),
        TurnExpectation(
            "Show my mood history",
            tools: ["mood_list"]
        ),
    ]
}

// MARK: - S3: Multi-Tool Sequence

struct MultiToolScenario: BenchmarkScenario {
    let id = "S3"
    let description = "Multi-tool sequence — chained operations"
    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Create a goal: Learn Spanish",
            tools: ["goal_create"],
            arguments: ["goal_create": ["name": "Spanish"]]
        ),
        TurnExpectation(
            "Add tasks: buy textbook and download a language app",
            tools: ["task_create", "task_create"]
        ),
        TurnExpectation(
            "List my goals",
            tools: ["goal_list"]
        ),
        TurnExpectation(
            "List my tasks",
            tools: ["task_list"]
        ),
        TurnExpectation(
            "I bought the textbook, mark it complete",
            tools: ["task_complete"]
        ),
        TurnExpectation(
            "Update my Spanish goal with a note that I started studying",
            tools: ["goal_update"]
        ),
        TurnExpectation(
            "How are my goals looking?",
            tools: ["goal_list"]
        ),
        TurnExpectation(
            "What tasks are still pending?",
            tools: ["task_list"]
        ),
    ]
}

// MARK: - S4: Long Conversation (50 turns)

struct LongConversationScenario: BenchmarkScenario {
    let id = "S4"
    let description = "Long conversation — 49 turns, stability test"
    let turns: [TurnExpectation] = Self.buildTurns()

    private static func buildTurns() -> [TurnExpectation] {
        var turns: [TurnExpectation] = []

        // Phase 1: Setup (turns 1-10)
        turns.append(TurnExpectation(
            "Create a goal: Run a marathon",
            tools: ["goal_create"],
            arguments: ["goal_create": ["name": "marathon"]]
        ))
        turns.append(TurnExpectation(
            "Create a goal: Read 20 books this year",
            tools: ["goal_create"]
        ))
        turns.append(TurnExpectation(
            "Create a goal: Learn to cook Italian food",
            tools: ["goal_create"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Sign up for a running club",
            tools: ["task_create"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Buy running shoes",
            tools: ["task_create"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Pick up a library card",
            tools: ["task_create"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Order an Italian cookbook",
            tools: ["task_create"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Research marathon training plans",
            tools: ["task_create"]
        ))
        turns.append(TurnExpectation(
            "Create a habit called running, daily",
            tools: ["habit_create"],
            arguments: ["habit_create": ["name": "running", "frequency": "daily"]]
        ))
        turns.append(TurnExpectation(
            "Create a habit called reading, daily",
            tools: ["habit_create"],
            arguments: ["habit_create": ["name": "reading", "frequency": "daily"]]
        ))

        // Phase 2: Daily routine (turns 11-30)
        turns.append(TurnExpectation(
            "Remember that my favorite Italian dish is carbonara",
            tools: ["memory_save"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 8, feeling motivated",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "I signed up for a running club, mark that task complete",
            tools: ["task_complete"],
            toolsOptional: true  // Task created at turn 4, outside 20-msg context window
        ))
        turns.append(TurnExpectation(
            "How are my habits going?",
            tools: ["habit_status"]
        ))
        turns.append(TurnExpectation(
            "What tasks do I still need to do?",
            tools: ["task_list"]
        ))
        turns.append(TurnExpectation(
            "Update my marathon goal — signed up for the club and started training",
            tools: ["goal_update"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 7",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Show my mood history",
            tools: ["mood_list"]
        ))
        turns.append(TurnExpectation(
            "I bought the running shoes, mark complete",
            tools: ["task_complete"]
        ))
        turns.append(TurnExpectation(
            "I got my library card too",
            tools: ["task_complete"]
        ))
        turns.append(TurnExpectation(
            "Remember that I run best in the morning before 8am",
            tools: ["memory_save"]
        ))
        turns.append(TurnExpectation(
            "What are my goals?",
            tools: ["goal_list"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 6, feeling a bit tired",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "How's my running streak?",
            tools: ["habit_status"]
        ))
        turns.append(TurnExpectation(
            "Update my reading goal — picked up the library card and started my first book",
            tools: ["goal_update"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["habit_log"],
            toolsOptional: true  // Habit created at turn 10, outside 20-msg context window
        ))
        // Phase 3: Past context window boundary (turns 31-40)
        turns.append(TurnExpectation(
            "I ordered the Italian cookbook, mark it done",
            tools: ["task_complete"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 9, had a great run today",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Remember I prefer to read fiction before bed",
            tools: ["memory_save"]
        ))
        turns.append(TurnExpectation(
            "Update my Italian cooking goal — the cookbook arrived, trying first recipe this weekend",
            tools: ["goal_update"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "What tasks are still pending?",
            tools: ["task_list"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 8",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "How are all my habits looking?",
            tools: ["habit_status"]
        ))
        turns.append(TurnExpectation(
            "Show my mood trends",
            tools: ["mood_list"]
        ))

        // Phase 4: Reference early items, test graceful context loss (turns 41-50)
        turns.append(TurnExpectation(
            "What do you know about my food preferences?",
            tools: ["memory_search"],
            substrings: ["carbonara"],
            toolsOptional: true  // Memory saved at turn 11, outside 20-msg context window
        ))
        turns.append(TurnExpectation(
            "When do I like to run?",
            tools: ["memory_search"],
            substrings: ["morning"],
            toolsOptional: true  // Memory saved at turn 23, outside 20-msg context window
        ))
        turns.append(TurnExpectation(
            "Update my marathon goal with a progress note — completed first 10K run",
            tools: ["goal_update"]
        ))
        turns.append(TurnExpectation(
            "How are my goals looking overall?",
            tools: ["goal_list"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 9, feeling accomplished",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "What's my average mood been?",
            tools: ["mood_list"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit",
            tools: ["habit_log"]
        ))
        turns.append(TurnExpectation(
            "Give me a summary of everything I'm working on",
            tools: ["goal_list", "task_list"]
        ))
        turns.append(TurnExpectation(
            "Thanks for keeping track of everything!",
            substrings: []
        ))

        return turns
    }
}

// MARK: - S5: Duplicate Detection

struct DuplicateDetectionScenario: BenchmarkScenario {
    let id = "S5"
    let description = "Duplicate detection — tools handle repeated requests gracefully"
    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Create a habit called running, daily",
            tools: ["habit_create"],
            arguments: ["habit_create": ["name": "running", "frequency": "daily"]]
        ),
        // Repeated request — tool returns "already exists", model should relay that
        TurnExpectation(
            "Create a habit called running, daily",
            tools: ["habit_create"],
            substrings: ["already"]
        ),
        TurnExpectation(
            "Log my running habit for today",
            tools: ["habit_log"]
        ),
        // Repeated log — tool returns "already logged", model should relay that
        TurnExpectation(
            "Log my running habit for today",
            tools: ["habit_log"],
            substrings: ["already"],
            toolsOptional: true  // Model may skip tool when prior success visible in context
        ),
        TurnExpectation(
            "Remember I like coffee",
            tools: ["memory_save"]
        ),
        // Near-duplicate memory — tool returns "already saved"
        TurnExpectation(
            "Remember that I like coffee",
            tools: ["memory_save"],
            substrings: ["already"],
            toolsOptional: true  // Model may skip tool when prior save visible in context
        ),
    ]
}

// MARK: - S6: Context Window Stress

struct ContextStressScenario: BenchmarkScenario {
    let id = "S6"
    let description = "Context window stress — boundary behavior"
    let turns: [TurnExpectation] = [
        // Create many items to fill context
        TurnExpectation(
            "Remember my favorite color is blue",
            tools: ["memory_save"]
        ),
        TurnExpectation(
            "Create a goal: Learn piano",
            tools: ["goal_create"]
        ),
        TurnExpectation(
            "Create a habit called piano practice, daily",
            tools: ["habit_create"]
        ),
        TurnExpectation(
            "Add a task: Buy a keyboard",
            tools: ["task_create"]
        ),
        // Ask about turn 1 items — answer may come from context or recall tool
        TurnExpectation(
            "What's my favorite color?",
            substrings: ["blue"],
            toolsOptional: true
        ),
        TurnExpectation(
            "List all my goals and tasks",
            tools: ["goal_list", "task_list"]
        ),
    ]
}

// MARK: - S7: Error Recovery

struct ErrorRecoveryScenario: BenchmarkScenario {
    let id = "S7"
    let description = "Error recovery — ambiguous/missing info, clarification"
    let turns: [TurnExpectation] = [
        // Ambiguous — should ask what/when, not hallucinate
        TurnExpectation(
            "Set a reminder",
            forbidden: ["reminder_set"],
            expectsClarification: true
        ),
        // Still missing time
        TurnExpectation(
            "Remind me about the meeting",
            forbidden: ["reminder_set"],
            expectsClarification: true
        ),
        // Now complete — self-contained so 3B model doesn't need to infer from prior turns
        TurnExpectation(
            "At 3pm tomorrow, remind me about the meeting",
            tools: ["reminder_set"]
        ),
        // Nonsensical request
        TurnExpectation(
            "Do something weird with the thingy",
            expectsClarification: true
        ),
    ]
}

// MARK: - All Scenarios

enum BenchmarkScenarios {
    static let all: [any BenchmarkScenario] = [
        SimpleQAScenario(),
        SingleToolScenario(),
        MultiToolScenario(),
        LongConversationScenario(),
        DuplicateDetectionScenario(),
        ContextStressScenario(),
        ErrorRecoveryScenario(),
    ]

    static func filtered(by ids: [String]?) -> [any BenchmarkScenario] {
        guard let ids else { return all }
        let idSet = Set(ids)
        return all.filter { idSet.contains($0.id) }
    }
}
