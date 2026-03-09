import Foundation

// MARK: - Tool Expectations

enum ToolMatchMode {
    case noTools
    case containsSequence
    case exactSequence
}

struct ExpectedToolCall {
    let name: String
    let arguments: [String: String]

    init(_ name: String, arguments: [String: String] = [:]) {
        self.name = name
        self.arguments = arguments
    }
}

// MARK: - File Expectations

struct FileAssertion {
    let path: String
    let mustContain: [String]
    let mustNotContain: [String]

    init(path: String, mustContain: [String] = [], mustNotContain: [String] = []) {
        self.path = path
        self.mustContain = mustContain
        self.mustNotContain = mustNotContain
    }
}

struct BenchmarkSeedFile {
    let relativePath: String
    let content: String
}

// MARK: - Turn Expectation

/// What we expect the model to do for a given user message.
struct TurnExpectation {
    let userMessage: String
    let toolMatchMode: ToolMatchMode
    let expectedToolCalls: [ExpectedToolCall]
    let forbiddenTools: [String]
    let expectedSubstrings: [String]
    let expectsClarification: Bool
    let fileAssertions: [FileAssertion]
    let preTurnFiles: [BenchmarkSeedFile]

    init(
        _ userMessage: String,
        tools: [ExpectedToolCall] = [],
        toolMatchMode: ToolMatchMode? = nil,
        forbidden: [String] = [],
        substrings: [String] = [],
        expectsClarification: Bool = false,
        fileAssertions: [FileAssertion] = [],
        preTurnFiles: [BenchmarkSeedFile] = []
    ) {
        self.userMessage = userMessage
        self.expectedToolCalls = tools
        self.toolMatchMode = toolMatchMode ?? (tools.isEmpty ? .noTools : .containsSequence)
        self.forbiddenTools = forbidden
        self.expectedSubstrings = substrings
        self.expectsClarification = expectsClarification
        self.fileAssertions = fileAssertions
        self.preTurnFiles = preTurnFiles
    }
}

// MARK: - Benchmark Skill

/// Describes a skill file to seed in the benchmark sandbox.
struct BenchmarkSkill {
    /// Relative path within the sandbox (e.g. "skills/tasks/SKILL.md").
    let relativePath: String
    /// File content to write.
    let content: String
    /// Skill name for prompt XML (e.g. "task-management").
    let name: String
    /// Skill description for prompt XML.
    let description: String
}

// MARK: - Scenario Protocol

protocol BenchmarkScenario {
    var id: String { get }
    var description: String { get }
    var turns: [TurnExpectation] { get }
    var benchmarkSkills: [BenchmarkSkill] { get }
    var benchmarkFiles: [BenchmarkSeedFile] { get }
}

extension BenchmarkScenario {
    var benchmarkSkills: [BenchmarkSkill] { [] }
    var benchmarkFiles: [BenchmarkSeedFile] { [] }
}

// MARK: - Helpers

private enum BenchmarkTool {
    static func read(_ pathSubstring: String) -> ExpectedToolCall {
        ExpectedToolCall("read", arguments: ["path": pathSubstring])
    }

    static func edit(path: String) -> ExpectedToolCall {
        ExpectedToolCall("edit", arguments: ["path": path])
    }
}

private enum BenchmarkSkillFixtures {
    static let taskManagement = BenchmarkSkill(
        relativePath: "skills/tasks/SKILL.md",
        content: """
            # Task Management Skill

            ## When to Use
            Use this skill when the user wants to create, list, complete, or manage tasks.

            ## Workflow

            ### Creating a Task
            1. Read `tasks.md` to see existing tasks
            2. Use `edit` to append a new checkbox item: `- [ ] <task description>`
            3. Do NOT overwrite the file — always append to existing content

            ### Completing a Task
            1. Read `tasks.md` to find the task
            2. Use `edit` to change `- [ ]` to `- [x]` for that task

            ### Listing Tasks
            1. Read `tasks.md` and summarize for the user

            ## Format
            Tasks use markdown checkboxes:
            - `- [ ]` for pending tasks
            - `- [x]` for completed tasks
            """,
        name: "task-management",
        description: "Use this skill when the user wants to create, list, complete, or manage tasks and to-do items."
    )
}

// MARK: - S1: Memory Save

struct MemorySaveScenario: BenchmarkScenario {
    let id = "S1"
    let description = "Memory save — strict read + edit workflow on memories.md"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Remember that I'm allergic to peanuts",
            tools: [BenchmarkTool.read("memories"), BenchmarkTool.edit(path: "memories")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "memories.md",
                    mustContain: [
                        "Alex prefers dark mode for all apps",
                        "Alex's birthday is March 15th",
                        "Alex is allergic to peanuts",
                    ]
                )
            ]
        ),
        TurnExpectation(
            "What do you know about my birthday?",
            tools: [BenchmarkTool.read("memories")],
            substrings: ["March 15"]
        ),
        TurnExpectation(
            "What are my allergies?",
            tools: [BenchmarkTool.read("memories")],
            substrings: ["peanut"]
        ),
    ]
}

// MARK: - S2: Task Creation

struct TaskCreationScenario: BenchmarkScenario {
    let id = "S2"
    let description = "Task creation — strict read + edit workflow on tasks.md"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Add a task: Water the plants",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Buy groceries", "Schedule dentist appointment", "Call mom", "Water the plants"]
                )
            ]
        ),
        TurnExpectation(
            "What are my tasks?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["plants", "groceries"]
        ),
        TurnExpectation(
            "Add a task: Pick up dry cleaning",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Water the plants", "Pick up dry cleaning", "Buy groceries"]
                )
            ]
        ),
        TurnExpectation(
            "List all my pending tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["cleaning", "plants"]
        ),
    ]
}

// MARK: - S3: Task Listing

struct TaskListingScenario: BenchmarkScenario {
    let id = "S3"
    let description = "Task listing — authoritative reads for summaries"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["groceries"]
        ),
        TurnExpectation(
            "How many pending tasks do I have?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["3"]
        ),
        TurnExpectation(
            "Thanks for the summary!"
        ),
    ]
}

// MARK: - S4: Task Completion

struct TaskCompletionScenario: BenchmarkScenario {
    let id = "S4"
    let description = "Task completion — read, exact edit, then reread"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "What tasks do I need to do?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["groceries"]
        ),
        TurnExpectation(
            "I bought the groceries, mark that task as done",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["- [x] Buy groceries", "- [ ] Schedule dentist appointment", "- [ ] Call mom"]
                )
            ]
        ),
        TurnExpectation(
            "What tasks are still pending?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["Schedule dentist appointment", "Call mom"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustNotContain: ["- [ ] Buy groceries"]
                )
            ]
        ),
        TurnExpectation(
            "What have I already completed?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["groceries"]
        ),
    ]
}

// MARK: - S6: Multi-step Workflow

struct MultiStepScenario: BenchmarkScenario {
    let id = "S6"
    let description = "Multi-step workflow — authoritative rereads across memory and tasks"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Remember that I love Italian food",
            tools: [BenchmarkTool.read("memories"), BenchmarkTool.edit(path: "memories")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(path: "memories.md", mustContain: ["Alex loves Italian food"])
            ]
        ),
        TurnExpectation(
            "Add a task: Order an Italian cookbook",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(path: "tasks.md", mustContain: ["Order an Italian cookbook", "Buy groceries"])
            ]
        ),
        TurnExpectation(
            "What tasks do I have?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["cookbook", "groceries"]
        ),
        TurnExpectation(
            "I ordered the cookbook, mark it done",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["- [x] Order an Italian cookbook", "- [ ] Buy groceries"]
                )
            ]
        ),
        TurnExpectation(
            "What's my favorite food?",
            tools: [BenchmarkTool.read("memories")],
            substrings: ["Italian"]
        ),
        TurnExpectation(
            "What tasks are still pending?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["Buy groceries", "Call mom"],
            fileAssertions: [
                FileAssertion(path: "tasks.md", mustNotContain: ["- [ ] Order an Italian cookbook"])
            ]
        ),
    ]
}

// MARK: - S7: Simple Conversation

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

// MARK: - S8: Skill-Guided Task Creation

struct SkillGuidedTaskScenario: BenchmarkScenario {
    let id = "S8"
    let description = "Skill-guided task creation — must read skill before tasks"

    let benchmarkSkills: [BenchmarkSkill] = [
        BenchmarkSkillFixtures.taskManagement
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Create a task: Watch data streams more regularly",
            tools: [
                BenchmarkTool.read("SKILL"),
                BenchmarkTool.read("tasks"),
                BenchmarkTool.edit(path: "tasks"),
            ],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Watch data streams more regularly", "Buy groceries", "Call mom"]
                )
            ]
        ),
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["watch data streams", "groceries"]
        ),
    ]
}

// MARK: - S9: Read-Before-Write Safety

struct ReadBeforeWriteScenario: BenchmarkScenario {
    let id = "S9"
    let description = "Read-before-write safety — edit existing files without overwriting"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Remember that I love hiking on weekends",
            tools: [BenchmarkTool.read("memories"), BenchmarkTool.edit(path: "memories")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "memories.md",
                    mustContain: ["dark mode", "March 15th", "love hiking on weekends"]
                )
            ]
        ),
        TurnExpectation(
            "What do you know about me?",
            tools: [BenchmarkTool.read("memories")],
            substrings: ["dark mode", "March 15", "hiking"]
        ),
        TurnExpectation(
            "Add a task: Go hiking this Saturday",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Buy groceries", "Call mom", "Go hiking this Saturday"]
                )
            ]
        ),
        TurnExpectation(
            "What are all my tasks?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["groceries", "hiking"]
        ),
    ]
}

// MARK: - S10: Skill vs Tool Shadowing

struct SkillShadowScenario: BenchmarkScenario {
    let id = "S10"
    let description = "Skill shadowing — named skill must be read, not called as a tool"

    let benchmarkSkills: [BenchmarkSkill] = [
        BenchmarkSkillFixtures.taskManagement
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Use the task-management skill to add a task: Review benchmark regressions",
            tools: [
                BenchmarkTool.read("SKILL"),
                BenchmarkTool.read("tasks"),
                BenchmarkTool.edit(path: "tasks"),
            ],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Review benchmark regressions", "Buy groceries"]
                )
            ]
        ),
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["benchmark regressions"]
        ),
    ]
}

// MARK: - S11: External Mutation

struct ExternalMutationScenario: BenchmarkScenario {
    let id = "S11"
    let description = "External mutation — reread authoritative state after files change"

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["groceries", "Call mom"]
        ),
        TurnExpectation(
            "What tasks are pending now?",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["Pay rent", "Buy groceries"],
            preTurnFiles: [
                BenchmarkSeedFile(
                    relativePath: "tasks.md",
                    content: """
                        # Tasks

                        - [ ] Buy groceries
                        - [ ] Schedule dentist appointment
                        - [ ] Call mom
                        - [ ] Pay rent
                        """
                )
            ]
        ),
    ]
}

// MARK: - S12: Relative Skill Resource

struct RelativeSkillResourceScenario: BenchmarkScenario {
    let id = "S12"
    let description = "Relative skill resource — resolve skill-relative paths before acting"

    let benchmarkSkills: [BenchmarkSkill] = [
        BenchmarkSkill(
            relativePath: "skills/checklists/SKILL.md",
            content: """
                # Checklist Skill

                ## When to Use
                Use this skill when the user asks for the default checklist task.

                ## Workflow
                1. Read `templates/default-task.md`
                2. Read `tasks.md`
                3. Use `edit` to append the template task to `tasks.md`
                """,
            name: "checklist-skill",
            description: "Use this skill to create checklist tasks from a skill-local template."
        )
    ]

    let benchmarkFiles: [BenchmarkSeedFile] = [
        BenchmarkSeedFile(
            relativePath: "skills/checklists/templates/default-task.md",
            content: "Review agent benchmark logs"
        )
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Use the checklist skill to add the default task",
            tools: [
                BenchmarkTool.read("SKILL"),
                BenchmarkTool.read("default-task"),
                BenchmarkTool.read("tasks"),
                BenchmarkTool.edit(path: "tasks"),
            ],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["Review agent benchmark logs", "Buy groceries"]
                )
            ]
        ),
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["Review agent benchmark logs"]
        ),
    ]
}

// MARK: - S13: Ambiguous Edit Requires Clarification

struct AmbiguousTaskScenario: BenchmarkScenario {
    let id = "S13"
    let description = "Ambiguous task completion — ask clarification before editing"

    let benchmarkFiles: [BenchmarkSeedFile] = [
        BenchmarkSeedFile(
            relativePath: "tasks.md",
            content: """
                # Tasks

                - [ ] Call mom
                - [ ] Call mom about taxes
                - [ ] Buy groceries
                """
        )
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Mark call mom done",
            substrings: ["which", "call mom"],
            expectsClarification: true
        ),
        TurnExpectation(
            "The one about taxes",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["- [ ] Call mom", "- [x] Call mom about taxes", "- [ ] Buy groceries"]
                )
            ]
        ),
    ]
}

// MARK: - S14: Negative Skill Control

struct NegativeSkillControlScenario: BenchmarkScenario {
    let id = "S14"
    let description = "Negative skill control — do not read skills or files for casual requests"

    let benchmarkSkills: [BenchmarkSkill] = [
        BenchmarkSkillFixtures.taskManagement
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "What's the capital of France?",
            forbidden: ["read", "edit", "write", "ls"],
            substrings: ["Paris"]
        ),
        TurnExpectation(
            "Thanks!",
            forbidden: ["read", "edit", "write", "ls"]
        ),
    ]
}

// MARK: - S15: Exact Edit Safety

struct ExactEditSafetyScenario: BenchmarkScenario {
    let id = "S15"
    let description = "Exact edit safety — update only the intended matching task"

    let benchmarkFiles: [BenchmarkSeedFile] = [
        BenchmarkSeedFile(
            relativePath: "tasks.md",
            content: """
                # Tasks

                - [ ] Call mom
                - [ ] Call mom about taxes
                - [ ] Schedule dentist appointment
                """
        )
    ]

    let turns: [TurnExpectation] = [
        TurnExpectation(
            "Mark 'Call mom about taxes' as done",
            tools: [BenchmarkTool.read("tasks"), BenchmarkTool.edit(path: "tasks")],
            forbidden: ["write"],
            fileAssertions: [
                FileAssertion(
                    path: "tasks.md",
                    mustContain: ["- [ ] Call mom", "- [x] Call mom about taxes", "- [ ] Schedule dentist appointment"],
                    mustNotContain: ["- [x] Call mom\n"]
                )
            ]
        ),
        TurnExpectation(
            "Show me my tasks",
            tools: [BenchmarkTool.read("tasks")],
            substrings: ["Call mom about taxes", "Call mom"]
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
        MultiStepScenario(),
        SimpleConversationScenario(),
        SkillGuidedTaskScenario(),
        ReadBeforeWriteScenario(),
        SkillShadowScenario(),
        ExternalMutationScenario(),
        RelativeSkillResourceScenario(),
        AmbiguousTaskScenario(),
        NegativeSkillControlScenario(),
        ExactEditSafetyScenario(),
    ]

    static func filtered(by ids: [String]?) -> [any BenchmarkScenario] {
        guard let ids else { return all }
        let idSet = Set(ids)
        return all.filter { idSet.contains($0.id) }
    }
}
