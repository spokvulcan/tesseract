# Epic 3: Context and Prompt

> Implement context file loading, skill discovery, Pi-style prompt assembly, and the compaction system. All tasks are additive.

## Prerequisites

- Epic 0 (Foundation Types) — `AgentMessageProtocol`, `AgentToolDefinition`, `ContentBlock`
- Epic 2 (Core Loop) — `CompactionSummaryMessage` (Task 2.6), `convertToLlm` (Task 2.1)

**Note**: Epic 3 cannot run in parallel with Epic 2. The compaction system (Task 3.5) creates `CompactionSummaryMessage` instances and the context manager's `transformContext` hook uses `convertToLlm`. Both are defined in Epic 2.

## Directory

All new files go in `Features/Agent/Context/`.

---

## Task 3.1: Implement ContextLoader

**File**: `Features/Agent/Context/ContextLoader.swift` (~120 lines)

**What to build**:

Discovers and loads context files (`AGENTS.md`, `CLAUDE.md`) and prompt override files (`SYSTEM.md`, `APPEND_SYSTEM.md`) from the fixed set of sandbox-accessible locations.

```swift
struct ContextLoader {
    let agentRoot: URL  // ~/Library/Application Support/Tesseract Agent/agent/

    struct LoadedContext {
        let contextFiles: [(path: String, content: String)]  // In load order
        let systemOverride: String?       // SYSTEM.md content (replaces default prompt)
        let systemAppend: String?         // APPEND_SYSTEM.md content (appended to prompt)
    }

    /// Load all context files from known locations
    func load(
        packagePaths: [URL],    // From enabled packages
        extensionPaths: [URL]   // From resources_discover events
    ) -> LoadedContext
}
```

**Context file discovery** (load order = precedence):
1. **Global agent directory**: `{agentRoot}/AGENTS.md` (or `CLAUDE.md`)
2. **Package-provided context**: Files at each `packagePaths` URL
3. **Extension-provided context**: Files at each `extensionPaths` URL
4. Deduplicate by absolute path (each file loaded at most once)

**Prompt override discovery** (first found wins):
1. `{agentRoot}/SYSTEM.md` → if exists, replaces entire default prompt
2. Package-provided `SYSTEM.md` files
3. Same for `APPEND_SYSTEM.md` → appended to prompt

**Rules**:
- Check for `AGENTS.md` first; if not found, check `CLAUDE.md` (matches Pi)
- No ancestor directory walking (sandbox constraint)
- Files that don't exist are silently skipped
- Empty files are skipped

**Acceptance criteria**:
- Loads `AGENTS.md` from agent root when it exists
- Falls back to `CLAUDE.md` when `AGENTS.md` not found
- `SYSTEM.md` replaces default prompt
- `APPEND_SYSTEM.md` is appended
- Package paths are checked in order
- Deduplication by path works
- Build succeeds

**Spec reference**: Sections 10.3, J.2, J.4

---

## Task 3.2: Implement SkillRegistry

**File**: `Features/Agent/Context/SkillRegistry.swift` (~180 lines)

**What to build**:

Discovers skill files, parses YAML frontmatter, and generates the XML listing for the system prompt.

```swift
struct SkillMetadata: Sendable {
    let name: String            // lowercase, hyphens
    let description: String     // max 1024 chars
    let filePath: String        // absolute path for `read` tool
    let disableModelInvocation: Bool  // if true, not listed in prompt
}

struct SkillRegistry {
    /// Discover all skills from known locations
    static func discover(
        locations: [URL],       // Skill directories in load order
        packageSkillPaths: [URL]  // From packages
    ) -> [SkillMetadata]

    /// Format discovered skills as XML for system prompt
    static func formatForPrompt(_ skills: [SkillMetadata]) -> String
}
```

**Skill discovery rules** (from spec F.2):
1. **Root `.md` files**: Direct `.md` children of a skills directory (any `.md` with valid frontmatter, not just `SKILL.md`)
2. **`SKILL.md` in subdirectories**: Recursive scan for `SKILL.md` under subdirectories
3. **Deduplication**: By name (first loaded wins)
4. **Skip**: Files without `description` in frontmatter

**Skill locations** (load order):
```
1. ~/Library/Application Support/Tesseract Agent/agent/skills/    (user-global)
2. Package-provided skill paths                               (from packages)
3. Extension-provided skill paths                             (from resources_discover)
```

**YAML frontmatter parsing**:
- Parse between `---` delimiters at top of file
- Extract: `name` (optional, fallback to directory name), `description` (required), `disable-model-invocation` (optional)
- Use simple regex/string parsing (avoid heavy YAML dependency)
- Frontmatter format is simple key-value, not nested YAML

**XML prompt format** (from spec F.4):
```xml
The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory.

<available_skills>
  <skill>
    <name>memory-management</name>
    <description>Use this skill when the user wants to remember, update, or forget personal facts.</description>
    <location>/path/to/skill/SKILL.md</location>
  </skill>
</available_skills>
```

Only include skills where `disableModelInvocation == false`.

**Acceptance criteria**:
- Discovers `.md` files at skill directory root
- Discovers `SKILL.md` in subdirectories
- Parses frontmatter correctly (name, description, disable-model-invocation)
- Skips files without description
- Deduplicates by name (first wins)
- Generates correct XML format
- Build succeeds

**Spec reference**: Sections F.1-F.4

---

## Task 3.3: Implement SystemPromptAssembler

**File**: `Features/Agent/Context/SystemPromptAssembler.swift` (~120 lines)

**What to build**:

Replaces `SystemPromptBuilder` with Pi-style prompt assembly. The old builder remains functional until Epic 6 integration.

```swift
struct SystemPromptAssembler {
    /// Assemble the complete system prompt
    static func assemble(
        defaultPrompt: String,
        loadedContext: ContextLoader.LoadedContext,
        skills: [SkillMetadata],
        tools: [AgentToolDefinition],
        dateTime: Date,
        agentRoot: String
    ) -> String
}
```

**Assembly order** (from spec J.3):

1. **Base prompt**:
   - If `loadedContext.systemOverride` exists → use it (replaces default)
   - Else → use `defaultPrompt`

2. **Append system**:
   - If `loadedContext.systemAppend` exists → append it

3. **Context files**:
   - For each context file in `loadedContext.contextFiles`:
     ```
     # Project Context: {filename}

     {content}
     ```

4. **Skills listing**:
   - Only if tools include a `read` tool (the model needs `read` to load skills)
   - `SkillRegistry.formatForPrompt(skills)`

5. **Date/time**: `"Current date and time: {formatted}"`

6. **Working directory**: `"Current working directory: {agentRoot}"`

**Default core prompt** (from spec section 10.2):
```
You are an expert local assistant operating inside Tesseract, a tool-calling agent harness.
You help users by reading files, editing files, writing files, and using other tools provided by the current package or project.

Available tools:
- read: Read file contents
- write: Create or overwrite files
- edit: Make surgical edits to files (find exact text and replace)
- list: List files and directories

In addition to the tools above, you may have access to other custom tools depending on the current package or project.

Guidelines:
- Use list to discover files and directories
- Use read to examine files before editing
- Use edit for precise changes (old text must match exactly)
- Use write only for new files or complete rewrites
- Be concise in your responses
- Show file paths clearly when working with files

Tesseract resources:
- Project context files are loaded from AGENTS.md or CLAUDE.md
- The system prompt may be replaced by SYSTEM.md or extended by APPEND_SYSTEM.md
- Skills may be available and should be used when relevant
```

Store this as a static string constant in the assembler.

**Acceptance criteria**:
- `SYSTEM.md` replaces default prompt entirely
- `APPEND_SYSTEM.md` is appended
- Context files appear as "# Project Context" sections
- Skills listed in XML only when `read` tool is available
- Date/time and working directory appended at end
- Build succeeds

**Spec reference**: Sections 10.1, 10.2, J.3

---

## Task 3.4: Implement Token Estimation

**File**: `Features/Agent/Context/ContextManager.swift` (Part 1: ~80 lines for estimation)

**What to build**:

Token estimation utilities for compaction decisions.

```swift
struct TokenEstimator {
    /// Estimate token count for a single message
    static func estimate(_ message: any AgentMessageProtocol) -> Int

    /// Estimate total tokens for a message array
    static func estimateTotal(_ messages: [any AgentMessageProtocol]) -> Int

    /// Estimate tokens for a string
    static func estimate(_ text: String) -> Int
}
```

**Estimation algorithm** (from spec G.2):
- Sum all text content in the message:
  - For `UserMessage`: content length
  - For `AssistantMessage`: content + thinking + tool call names + JSON args
  - For `ToolResultMessage`: content blocks text lengths
  - For image content blocks: estimate 4,800 chars (≈1,200 tokens)
- Apply heuristic: `(totalChars + 3) / 4` (ceil division, ~4 chars per token)

This is a coarse estimate. Pi also uses actual `usage.totalTokens` from the last assistant response when available. We'll add that refinement during integration.

**Acceptance criteria**:
- Empty message → 0 tokens
- 400-char message → ~100 tokens
- Image block → ~1,200 tokens
- Build succeeds

**Spec reference**: Section G.2

---

## Task 3.5: Implement Compaction Algorithm

**File**: `Features/Agent/Context/ContextManager.swift` (Part 2: ~200 lines for compaction)

**What to build**:

The full compaction system that runs inside `transformContext`.

```swift
struct CompactionSettings: Sendable {
    var enabled: Bool = true
    var reserveTokens: Int = 16_384
    var keepRecentTokens: Int = 20_000
}

actor ContextManager {
    let settings: CompactionSettings
    private var lastSummary: String?

    /// Should compaction run? (check before each LLM call)
    func shouldCompact(contextTokens: Int, contextWindow: Int) -> Bool

    /// Run compaction on the message array
    /// Returns the compacted message array with summary prepended
    func compact(
        messages: [any AgentMessageProtocol],
        contextWindow: Int,
        summarize: (String) async throws -> String  // LLM call for summarization
    ) async throws -> [any AgentMessageProtocol]
}
```

**`shouldCompact`** (from spec G.1):
```swift
guard settings.enabled else { return false }
return contextTokens > contextWindow - settings.reserveTokens
```

**Cut point detection** (from spec G.3):
1. Walk backward from newest message, accumulating token estimates
2. Stop when accumulated ≥ `keepRecentTokens`
3. Find closest valid cut point at or after that position
4. Valid cut points: user, assistant, custom, compaction_summary messages
5. Never cut at tool results (they must follow their tool call)

**Summarization** (from spec G.4):

Build a summarization prompt from the messages being compacted:

```
Summarize the following conversation history into a structured checkpoint.

Format:
## Goal
## Constraints & Preferences
## Progress
### Done / In Progress / Blocked
## Key Decisions
## Next Steps
## Critical Context

Focus on preserving actionable information. Be concise.

---

{messages to summarize, formatted as text}
```

If a previous compaction summary exists, use the update prompt instead:
```
Update the following summary with new information from the conversation.
Keep the same structure. Merge, don't duplicate.

Previous summary:
{lastSummary}

New conversation:
{new messages}
```

After summarization:
1. Create `CompactionSummaryMessage` with the summary text
2. Replace compacted messages with the summary message
3. Keep recent messages (after cut point) intact
4. Store summary for future update-style compaction

**Result**: `[CompactionSummaryMessage] + [recent messages]`

**`transformContext` integration**:
```swift
func makeTransformContext(contextManager: ContextManager, contextWindow: Int) -> @Sendable ([any AgentMessageProtocol], CancellationToken?) async -> ContextTransformResult {
    return { messages, signal in
        let tokens = TokenEstimator.estimateTotal(messages)
        guard await contextManager.shouldCompact(contextTokens: tokens, contextWindow: contextWindow) else {
            return ContextTransformResult(messages: messages, didMutate: false, reason: .compaction)
        }
        do {
            let compacted = try await contextManager.compact(
                messages: messages,
                contextWindow: contextWindow,
                summarize: { prompt in
                    // Call LLM for summarization (wired during integration)
                    // For now, return a placeholder
                    fatalError("Summarization not yet wired")
                }
            )
            return ContextTransformResult(messages: compacted, didMutate: true, reason: .compaction)
        } catch {
            Log.agent.error("Compaction failed: \(error)")
            return ContextTransformResult(messages: messages, didMutate: false, reason: .compaction)
        }
    }
}
```

**Acceptance criteria**:
- `shouldCompact` returns true when tokens exceed threshold
- Cut point detection never splits mid-tool-result
- First compaction uses initial summary prompt
- Subsequent compactions use update prompt
- Compacted array is shorter than original
- Build succeeds

**Spec reference**: Sections G.1-G.5

---

## Task 3.6: Default Compaction Settings for Tesseract

**Add to**: `ContextManager.swift` (small addition)

**What to build**:

Default settings tuned for Tesseract's 120K context window:

```swift
extension CompactionSettings {
    /// Default for 120K context window models
    static let standard = CompactionSettings(
        enabled: true,
        reserveTokens: 16_384,
        keepRecentTokens: 20_000
    )

    /// Conservative for smaller models (4-8K context)
    static let small = CompactionSettings(
        enabled: true,
        reserveTokens: 2_048,
        keepRecentTokens: 2_048
    )
}
```

Document: Compaction triggers at ~104K tokens for standard settings. At that point, older history is summarized and the 20K most recent tokens are preserved verbatim.

**Acceptance criteria**:
- `CompactionSettings.standard` triggers compaction at 120K - 16K = 104K tokens
- `CompactionSettings.small` triggers at proportionally lower thresholds
- Build succeeds

---

## Summary

After this epic, the project has the complete context loading, skill discovery, prompt assembly, and compaction system. None of it is wired to the existing agent yet.

**New files created**: 4 (ContextLoader.swift, SkillRegistry.swift, SystemPromptAssembler.swift, ContextManager.swift)
**Lines added**: ~700
**Existing files modified**: 0
