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
    /// Required argument key-value pairs per tool (tool name -> dict of key -> substring match).
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

// MARK: - S1: Memory Save

/// Tests file-based memory workflow: read memories.md, edit/write to add entries.
struct MemorySaveScenario: BenchmarkScenario {
    let id = "S1"
    let description = "Memory save — read + write/edit on memories.md"
    let turns: [TurnExpectation] = [
        // Save a new memory — model should read memories.md then edit/write to append
        TurnExpectation(
            "Remember that I'm allergic to peanuts",
            toolsOptional: true  // Flexible: may read+write, read+edit, or just write
        ),
        // Query pre-seeded memory — model should read memories.md
        TurnExpectation(
            "What do you know about my birthday?",
            tools: ["read"],
            substrings: ["March 15"],
            arguments: ["read": ["path": "memories"]]
        ),
        // Query just-written memory — model should read memories.md
        TurnExpectation(
            "What are my allergies?",
            tools: ["read"],
            substrings: ["peanut"],
            toolsOptional: true  // May recall from conversation context
        ),
    ]
}

// MARK: - S2: Task Creation

/// Tests file-based task creation: read tasks.md, edit/write to add checkbox items.
struct TaskCreationScenario: BenchmarkScenario {
    let id = "S2"
    let description = "Task creation — read + write/edit on tasks.md"
    let turns: [TurnExpectation] = [
        // Create a new task — model should read tasks.md then edit to append
        TurnExpectation(
            "Add a task: Water the plants",
            toolsOptional: true  // Flexible tool sequence
        ),
        // Query tasks — model should read tasks.md and confirm
        TurnExpectation(
            "What are my tasks?",
            tools: ["read"],
            substrings: ["plants"],
            arguments: ["read": ["path": "tasks"]]
        ),
        // Create another task
        TurnExpectation(
            "Add a task: Pick up dry cleaning",
            toolsOptional: true
        ),
        // List all tasks — should include both new and seeded tasks
        TurnExpectation(
            "List all my pending tasks",
            tools: ["read"],
            substrings: ["cleaning"]
        ),
    ]
}

// MARK: - S3: Task Listing

/// Tests reading and understanding task files from pre-seeded data.
struct TaskListingScenario: BenchmarkScenario {
    let id = "S3"
    let description = "Task listing — read tasks.md and summarize"
    let turns: [TurnExpectation] = [
        // List pre-seeded tasks
        TurnExpectation(
            "Show me my tasks",
            tools: ["read"],
            substrings: ["groceries"]
        ),
        // Ask about task count — model should read and count
        TurnExpectation(
            "How many pending tasks do I have?",
            tools: ["read"],
            toolsOptional: true  // May answer from context
        ),
        // Thanks — no tools
        TurnExpectation(
            "Thanks for the summary!"
        ),
    ]
}

// MARK: - S4: Task Completion

/// Tests editing task checkboxes: read tasks.md, edit [ ] to [x].
struct TaskCompletionScenario: BenchmarkScenario {
    let id = "S4"
    let description = "Task completion — read + edit checkbox in tasks.md"
    let turns: [TurnExpectation] = [
        // List current tasks (from seed data)
        TurnExpectation(
            "What tasks do I need to do?",
            tools: ["read"],
            substrings: ["groceries"]
        ),
        // Complete a task — model should read tasks.md then edit checkbox
        TurnExpectation(
            "I bought the groceries, mark that task as done",
            tools: ["read", "edit"],
            arguments: ["edit": ["path": "tasks"]]
        ),
        // Verify remaining tasks
        TurnExpectation(
            "What tasks are still pending?",
            tools: ["read"]
        ),
        // Query completed tasks
        TurnExpectation(
            "What have I already completed?",
            tools: ["read"],
            substrings: ["groceries"],
            toolsOptional: true  // May answer from context
        ),
    ]
}

// MARK: - S6: Multi-step Workflow

/// Tests combining memory and task operations in a multi-step sequence.
struct MultiStepScenario: BenchmarkScenario {
    let id = "S6"
    let description = "Multi-step workflow — memory + task file operations"
    let turns: [TurnExpectation] = [
        // Save a memory
        TurnExpectation(
            "Remember that I love Italian food",
            toolsOptional: true  // read + write/edit on memories.md
        ),
        // Create a task
        TurnExpectation(
            "Add a task: Order an Italian cookbook",
            toolsOptional: true  // read + write/edit on tasks.md
        ),
        // List tasks — should show new + seeded tasks
        TurnExpectation(
            "What tasks do I have?",
            tools: ["read"],
            substrings: ["cookbook"]
        ),
        // Complete the new task
        TurnExpectation(
            "I ordered the cookbook, mark it done",
            tools: ["read", "edit"],
            arguments: ["edit": ["path": "tasks"]]
        ),
        // Query memory
        TurnExpectation(
            "What's my favorite food?",
            tools: ["read"],
            substrings: ["Italian"],
            toolsOptional: true  // May recall from context
        ),
        // List remaining tasks
        TurnExpectation(
            "What tasks are still pending?",
            tools: ["read"]
        ),
    ]
}

// MARK: - S7: Simple Conversation

/// Tests pure conversation without tools — model should respond naturally.
struct SimpleConversationScenario: BenchmarkScenario {
    let id = "S7"
    let description = "Simple conversation — no tools expected"
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

// MARK: - All Scenarios

enum BenchmarkScenarios {
    static let all: [any BenchmarkScenario] = [
        MemorySaveScenario(),
        TaskCreationScenario(),
        TaskListingScenario(),
        TaskCompletionScenario(),
        // S5 removed — goal tracking intentionally dropped (see PLAN.md scope decision)
        MultiStepScenario(),
        SimpleConversationScenario(),
    ]

    static func filtered(by ids: [String]?) -> [any BenchmarkScenario] {
        guard let ids else { return all }
        let idSet = Set(ids)
        return all.filter { idSet.contains($0.id) }
    }
}
