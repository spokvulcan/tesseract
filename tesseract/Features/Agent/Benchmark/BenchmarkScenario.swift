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
            "What time is it?",
            tools: ["get_current_time"]
        ),
        TurnExpectation(
            "Remember that I'm allergic to peanuts",
            tools: ["remember"],
            arguments: ["remember": ["fact": "peanut"]]
        ),
        TurnExpectation(
            "Log my mood as 7",
            tools: ["mood_log"],
            arguments: ["mood_log": ["score": "7"]]
        ),
        TurnExpectation(
            "Create a habit called meditation, daily",
            tools: ["create_habit"],
            arguments: ["create_habit": ["name": "meditation", "frequency": "daily"]]
        ),
        TurnExpectation(
            "Show my mood history",
            tools: ["list_moods"]
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
            tools: ["create_goal"],
            arguments: ["create_goal": ["name": "Spanish"]]
        ),
        TurnExpectation(
            "Add tasks: buy textbook and download a language app",
            tools: ["create_task", "create_task"]
        ),
        TurnExpectation(
            "List my goals",
            tools: ["list_goals"]
        ),
        TurnExpectation(
            "List my tasks",
            tools: ["list_tasks"]
        ),
        TurnExpectation(
            "I bought the textbook, mark it complete",
            tools: ["complete_task"]
        ),
        TurnExpectation(
            "Update my Spanish goal with a note that I started studying",
            tools: ["update_goal"]
        ),
        TurnExpectation(
            "How are my goals looking?",
            tools: ["list_goals"]
        ),
        TurnExpectation(
            "What tasks are still pending?",
            tools: ["list_tasks"]
        ),
    ]
}

// MARK: - S4: Long Conversation (50 turns)

struct LongConversationScenario: BenchmarkScenario {
    let id = "S4"
    let description = "Long conversation — 50 turns, stability test"
    let turns: [TurnExpectation] = Self.buildTurns()

    private static func buildTurns() -> [TurnExpectation] {
        var turns: [TurnExpectation] = []

        // Phase 1: Setup (turns 1-10)
        turns.append(TurnExpectation(
            "Create a goal: Run a marathon",
            tools: ["create_goal"],
            arguments: ["create_goal": ["name": "marathon"]]
        ))
        turns.append(TurnExpectation(
            "Create a goal: Read 20 books this year",
            tools: ["create_goal"]
        ))
        turns.append(TurnExpectation(
            "Create a goal: Learn to cook Italian food",
            tools: ["create_goal"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Sign up for a running club",
            tools: ["create_task"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Buy running shoes",
            tools: ["create_task"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Pick up a library card",
            tools: ["create_task"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Order an Italian cookbook",
            tools: ["create_task"]
        ))
        turns.append(TurnExpectation(
            "Add a task: Research marathon training plans",
            tools: ["create_task"]
        ))
        turns.append(TurnExpectation(
            "Create a habit called running, daily",
            tools: ["create_habit"],
            arguments: ["create_habit": ["name": "running", "frequency": "daily"]]
        ))
        turns.append(TurnExpectation(
            "Create a habit called reading, daily",
            tools: ["create_habit"],
            arguments: ["create_habit": ["name": "reading", "frequency": "daily"]]
        ))

        // Phase 2: Daily routine (turns 11-30)
        turns.append(TurnExpectation(
            "Remember that my favorite Italian dish is carbonara",
            tools: ["remember"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 8, feeling motivated",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "I signed up for a running club, mark that task complete",
            tools: ["complete_task"]
        ))
        turns.append(TurnExpectation(
            "How are my habits going?",
            tools: ["habit_status"]
        ))
        turns.append(TurnExpectation(
            "What tasks do I still need to do?",
            tools: ["list_tasks"]
        ))
        turns.append(TurnExpectation(
            "Update my marathon goal — signed up for the club and started training",
            tools: ["update_goal"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 7",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Show my mood history",
            tools: ["list_moods"]
        ))
        turns.append(TurnExpectation(
            "I bought the running shoes, mark complete",
            tools: ["complete_task"]
        ))
        turns.append(TurnExpectation(
            "I got my library card too",
            tools: ["complete_task"]
        ))
        turns.append(TurnExpectation(
            "Remember that I run best in the morning before 8am",
            tools: ["remember"]
        ))
        turns.append(TurnExpectation(
            "What are my goals?",
            tools: ["list_goals"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 6, feeling a bit tired",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "How's my running streak?",
            tools: ["habit_status"]
        ))
        turns.append(TurnExpectation(
            "Update my reading goal — picked up the library card and started my first book",
            tools: ["update_goal"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "What's the current time?",
            tools: ["get_current_time"]
        ))

        // Phase 3: Past context window boundary (turns 31-40)
        turns.append(TurnExpectation(
            "I ordered the Italian cookbook, mark it done",
            tools: ["complete_task"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 9, had a great run today",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "Remember I prefer to read fiction before bed",
            tools: ["remember"]
        ))
        turns.append(TurnExpectation(
            "Update my Italian cooking goal — the cookbook arrived, trying first recipe this weekend",
            tools: ["update_goal"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit for today",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "What tasks are still pending?",
            tools: ["list_tasks"]
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
            tools: ["list_moods"]
        ))

        // Phase 4: Reference early items, test graceful context loss (turns 41-50)
        turns.append(TurnExpectation(
            "What do you know about my food preferences?",
            tools: ["recall"],
            substrings: ["carbonara"]
        ))
        turns.append(TurnExpectation(
            "When do I like to run?",
            tools: ["recall"],
            substrings: ["morning"]
        ))
        turns.append(TurnExpectation(
            "Update my marathon goal with a progress note — completed first 10K run",
            tools: ["update_goal"]
        ))
        turns.append(TurnExpectation(
            "How are my goals looking overall?",
            tools: ["list_goals"]
        ))
        turns.append(TurnExpectation(
            "Log my mood as 9, feeling accomplished",
            tools: ["mood_log"]
        ))
        turns.append(TurnExpectation(
            "What's my average mood been?",
            tools: ["list_moods"]
        ))
        turns.append(TurnExpectation(
            "Log my running habit",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "Log my reading habit",
            tools: ["log_habit"]
        ))
        turns.append(TurnExpectation(
            "Give me a summary of everything I'm working on",
            tools: ["list_goals", "list_tasks"]
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
            tools: ["create_habit"],
            arguments: ["create_habit": ["name": "running", "frequency": "daily"]]
        ),
        // Repeated request — tool returns "already exists", model should relay that
        TurnExpectation(
            "Create a habit called running, daily",
            tools: ["create_habit"],
            substrings: ["already"]
        ),
        TurnExpectation(
            "Log my running habit for today",
            tools: ["log_habit"]
        ),
        // Repeated log — tool returns "already logged", model should relay that
        TurnExpectation(
            "Log my running habit for today",
            tools: ["log_habit"],
            substrings: ["already"]
        ),
        TurnExpectation(
            "Remember I like coffee",
            tools: ["remember"]
        ),
        // Near-duplicate memory — tool returns "already remembered"
        TurnExpectation(
            "Remember that I like coffee",
            tools: ["remember"],
            substrings: ["already"]
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
            tools: ["remember"]
        ),
        TurnExpectation(
            "Create a goal: Learn piano",
            tools: ["create_goal"]
        ),
        TurnExpectation(
            "Create a habit called piano practice, daily",
            tools: ["create_habit"]
        ),
        TurnExpectation(
            "Add a task: Buy a keyboard",
            tools: ["create_task"]
        ),
        // Ask about turn 1 items — answer may come from context or recall tool
        TurnExpectation(
            "What's my favorite color?",
            substrings: ["blue"],
            toolsOptional: true
        ),
        TurnExpectation(
            "List all my goals and tasks",
            tools: ["list_goals", "list_tasks"]
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
            forbidden: ["set_reminder"],
            expectsClarification: true
        ),
        // Still missing time
        TurnExpectation(
            "Remind me about the meeting",
            forbidden: ["set_reminder"],
            expectsClarification: true
        ),
        // Now complete — self-contained so 3B model doesn't need to infer from prior turns
        TurnExpectation(
            "At 3pm tomorrow, remind me about the meeting",
            tools: ["set_reminder"]
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
