# Epic 5: Package System

> Implement the package manifest model, registry, and the first-party personal assistant package. This epic creates the infrastructure for bundling skills, prompts, extensions, and seed data.

## Prerequisites

- Epic 0 (Foundation Types)
- Epic 3 (Context and Prompt) — `SkillRegistry`, `ContextLoader`
- Epic 4 (Extension System) — `AgentExtension`, `ExtensionHost`

## Directories

- Swift code: `Features/Agent/Packages/`
- Package resources: `Resources/AgentPackages/personal-assistant/`

---

## Task 5.1: Define AgentPackage Manifest Model

**File**: `Features/Agent/Packages/AgentPackage.swift` (~60 lines)

**What to build**:

```swift
struct AgentPackageManifest: Codable, Sendable {
    let name: String
    let enabled: Bool
    let skills: [String]?            // Relative paths to skill files
    let promptAppendFiles: [String]? // Relative paths to APPEND_SYSTEM.md files
    let extensions: [String]?        // Extension module identifiers
    let seedFiles: [String]?         // Relative paths to seed data files
    let contextFiles: [String]?      // Relative paths to context files
}

struct ResolvedPackage: Sendable {
    let manifest: AgentPackageManifest
    let baseURL: URL                  // Directory containing package.json

    /// Resolved absolute paths
    var skillPaths: [URL]
    var promptAppendPaths: [URL]
    var extensionIdentifiers: [String]
    var seedFilePaths: [URL]
    var contextFilePaths: [URL]
}
```

**Resolution**: When loading a manifest, resolve all relative paths against `baseURL`:
```swift
func resolve() -> ResolvedPackage {
    let skills = (manifest.skills ?? []).map { baseURL.appendingPathComponent($0) }
    // ... etc
}
```

**Acceptance criteria**:
- Manifest decodes from JSON correctly
- Path resolution produces absolute URLs
- Build succeeds

**Spec reference**: Section 8.4

---

## Task 5.2: Implement PackageRegistry

**File**: `Features/Agent/Packages/PackageRegistry.swift` (~120 lines)

**What to build**:

```swift
@MainActor
final class PackageRegistry {
    private var packages: [String: ResolvedPackage] = [:]

    /// Load packages from known locations
    func loadPackages(
        bundledPackagesDir: URL,     // App bundle resources
        userPackagesDir: URL         // ~/Library/.../agent/packages/
    )

    /// Get all enabled packages (in load order)
    var enabledPackages: [ResolvedPackage] { get }

    /// Get all skill paths from enabled packages
    var allSkillPaths: [URL] { get }

    /// Get all prompt append paths from enabled packages
    var allPromptAppendPaths: [URL] { get }

    /// Get all context file paths from enabled packages
    var allContextFilePaths: [URL] { get }

    /// Get all seed file paths from enabled packages
    var allSeedFilePaths: [URL] { get }

    /// Get all extension identifiers from enabled packages
    var allExtensionIdentifiers: [String] { get }

    /// Enable/disable a package
    func setEnabled(_ name: String, enabled: Bool)
}
```

**Package discovery**:
1. Scan `bundledPackagesDir` for directories containing `package.json`
2. Scan `userPackagesDir` for directories containing `package.json`
3. Parse each `package.json` as `AgentPackageManifest`
4. Resolve paths
5. Store enabled packages

**Load order**:
- Bundled packages first, then user packages
- Within each directory: alphabetical by name

**Seed file handling**:
- When a package is first enabled, copy seed files to the agent data directory if they don't already exist
- Never overwrite existing user data
- Seed files are templates (e.g., empty `memories.md`, `tasks.md`)

```swift
func seedDataIfNeeded(package: ResolvedPackage, agentRoot: URL) {
    for seedPath in package.seedFilePaths {
        let targetName = seedPath.lastPathComponent
        let targetURL = agentRoot.appendingPathComponent(targetName)
        if !FileManager.default.fileExists(atPath: targetURL.path) {
            try? FileManager.default.copyItem(at: seedPath, to: targetURL)
        }
    }
}
```

**Acceptance criteria**:
- Discovers packages from bundled and user directories
- Parses manifest JSON correctly
- Resolves paths to absolute URLs
- Seeds data files on first enable (doesn't overwrite existing)
- Enable/disable persists (via UserDefaults or file)
- Build succeeds

**Spec reference**: Sections 8.3, 8.4

---

## Task 5.3: Create Personal Assistant Package Manifest and Skills

**Files**: Multiple resource files

### package.json
**File**: `Resources/AgentPackages/personal-assistant/package.json`

```json
{
    "name": "personal-assistant",
    "enabled": true,
    "skills": [
        "skills/memory/SKILL.md",
        "skills/tasks/SKILL.md",
        "skills/notes/SKILL.md"
    ],
    "promptAppendFiles": [
        "prompts/APPEND_SYSTEM.md"
    ],
    "extensions": [
        "PersonalAssistantExtension"
    ],
    "seedFiles": [
        "data/memories.md",
        "data/tasks.md"
    ]
}
```

### Memory Skill
**File**: `Resources/AgentPackages/personal-assistant/skills/memory/SKILL.md`

```markdown
---
name: memory-management
description: Use this skill when the user wants to remember, update, or forget personal facts, preferences, or important information.
---

# Memory Management

## File
`memories.md` in the agent working directory.

## Format
One fact per line. Plain text. No bullet points or numbering needed — the line position is the identifier.

## Workflow

### Saving a new memory
1. Read `memories.md` using the read tool
2. Check for duplicates or near-duplicates
3. If a similar memory exists, use edit to update it instead of adding a new line
4. If truly new, use edit to append a new line at the end
5. If the file doesn't exist, use write to create it

### Updating a memory
1. Read `memories.md`
2. Find the exact line to update
3. Use edit with the exact old text and the new text

### Deleting a memory
1. Read `memories.md`
2. Find the exact line to remove
3. Use edit to replace the line (and its trailing newline) with empty string

### Consolidation
If memories.md has grown large or has duplicates:
1. Read the full file
2. Write a consolidated version that merges duplicates and removes outdated facts
```

### Tasks Skill
**File**: `Resources/AgentPackages/personal-assistant/skills/tasks/SKILL.md`

```markdown
---
name: task-management
description: Use this skill when the user wants to create, list, complete, or manage tasks and to-do items.
---

# Task Management

## File
`tasks.md` in the agent working directory.

## Format
```
- [ ] Task description
- [x] Completed task description
```

Markdown checkbox format. One task per line.

## Workflow

### Creating a task
1. Read `tasks.md`
2. Use edit to append a new line: `- [ ] {task description}`
3. If file doesn't exist, use write to create it

### Listing tasks
1. Read `tasks.md`
2. Summarize pending tasks (lines with `- [ ]`)
3. Optionally mention recently completed tasks

### Completing a task
1. Read `tasks.md`
2. Find the exact line matching the task
3. Use edit to replace `- [ ]` with `- [x]`

### Deleting a task
1. Read `tasks.md`
2. Find and remove the line using edit
```

### Notes Skill
**File**: `Resources/AgentPackages/personal-assistant/skills/notes/SKILL.md`

```markdown
---
name: note-capture
description: Use this skill when the user wants to save a note, write down thoughts, or capture information for later reference.
---

# Note Capture

## Directory
`notes/` in the agent working directory.

## Format
Each note is a separate markdown file named with a slug derived from the content or title.

Example: `notes/meeting-with-sarah.md`, `notes/recipe-ideas.md`

## Workflow

### Saving a note
1. Use list to check what notes already exist in `notes/`
2. Choose a descriptive filename (lowercase, hyphens, .md extension)
3. Use write to create the note file
4. Include a `# Title` header and the content

### Finding a note
1. Use list on `notes/` to see all notes
2. Read the relevant note file

### Updating a note
1. Read the existing note
2. Use edit to make precise changes
3. Or use write for a full rewrite if the structure has changed significantly
```

### Assistant Prompt Append
**File**: `Resources/AgentPackages/personal-assistant/prompts/APPEND_SYSTEM.md`

```markdown
## Personal Assistant

You are a helpful personal assistant. You help the user manage their life by tracking memories, tasks, goals, and notes.

### Personality
- Be warm, concise, and proactive
- Acknowledge what you've done clearly ("Saved to memory", "Task completed")
- When the user shares facts about themselves, save them to memory proactively
- When the user mentions something they need to do, offer to create a task

### Memory Awareness
- At the start of a conversation, read `memories.md` to know what you already know about the user
- Use this knowledge naturally in conversation — don't repeat facts back unnecessarily
- Update memories when the user corrects or adds to existing information

### File Paths
- Memories: `memories.md`
- Tasks: `tasks.md`
- Notes: `notes/` directory
```

### Seed Data Files
**File**: `Resources/AgentPackages/personal-assistant/data/memories.md`
```markdown
```
(Empty file — the user starts with a clean slate)

**File**: `Resources/AgentPackages/personal-assistant/data/tasks.md`
```markdown
```
(Empty file)

**Acceptance criteria**:
- All files created in correct locations
- `package.json` is valid JSON
- Skill files have valid YAML frontmatter
- `APPEND_SYSTEM.md` provides assistant personality
- Seed data files are empty (fresh start)
- Build succeeds (resources are in the bundle)

**Spec reference**: Sections 9.1-9.4

---

## Task 5.4: Create PersonalAssistantPackage Swift Registration

**File**: `Features/Agent/Packages/BuiltIn/PersonalAssistantPackage.swift` (~80 lines)

**What to build**:

A Swift module that registers the personal assistant as a first-party package with the extension host.

```swift
final class PersonalAssistantExtension: AgentExtension, @unchecked Sendable {
    let path = "personal-assistant"
    let commands: [String: RegisteredCommand] = [:]

    // Extension tools (for features that can't be file-only)
    var tools: [String: AgentToolDefinition] {
        // For MVP: no custom tools. The package relies entirely on
        // file tools (read/write/edit) guided by skills.
        // Future: create_reminder, schedule_notification, etc.
        [:]
    }

    var handlers: [ExtensionEventType: [ExtensionEventHandler]] {
        [
            .sessionStart: [
                ExtensionEventHandler { _, context in
                    Log.agent.info("[PersonalAssistant] Session started, cwd: \(context.cwd)")
                    return nil
                }
            ],
            .beforeAgentStart: [
                ExtensionEventHandler { payload, context in
                    // Could inject memories into context here in the future
                    return nil
                }
            ]
        ]
    }
}
```

For MVP, the personal assistant package is purely skill-driven. It doesn't need custom tools because the file tools (read, write, edit, list) handle everything via the skill instructions.

Future tasks (post-MVP) can add:
- `create_reminder` tool using `UNUserNotificationCenter`
- `capture_current_app_context` tool using Accessibility APIs
- Custom memory indexing tools

**Acceptance criteria**:
- Conforms to `AgentExtension` protocol
- Registers with `ExtensionHost` without errors
- Build succeeds

**Spec reference**: Section 9.4

---

## Task 5.5: Wire Package Loading into Resource Pipeline

**File**: `Features/Agent/Packages/PackageBootstrap.swift` (~60 lines)

**What to build**:

A bootstrap function that loads packages, seeds data, and connects to the context/skill/extension systems:

```swift
struct PackageBootstrap {
    /// Called once at app startup (or when packages change)
    @MainActor
    static func bootstrap(
        packageRegistry: PackageRegistry,
        extensionHost: ExtensionHost,
        agentRoot: URL
    ) {
        // 1. Load packages
        let bundledDir = Bundle.main.url(forResource: "AgentPackages", withExtension: nil)
        let userDir = agentRoot.appendingPathComponent("packages")
        packageRegistry.loadPackages(
            bundledPackagesDir: bundledDir ?? agentRoot,
            userPackagesDir: userDir
        )

        // 2. Seed data for enabled packages
        for package in packageRegistry.enabledPackages {
            packageRegistry.seedDataIfNeeded(package: package, agentRoot: agentRoot)
        }

        // 3. Register built-in extensions referenced by packages
        for extId in packageRegistry.allExtensionIdentifiers {
            switch extId {
            case "PersonalAssistantExtension":
                extensionHost.register(PersonalAssistantExtension())
            default:
                Log.agent.warning("Unknown extension identifier: \(extId)")
            }
        }
    }

    /// Get skill discovery paths from packages
    static func skillPaths(from registry: PackageRegistry) -> [URL] {
        registry.allSkillPaths
    }

    /// Get prompt append content from packages
    static func promptAppends(from registry: PackageRegistry) -> [String] {
        registry.allPromptAppendPaths.compactMap {
            try? String(contentsOf: $0, encoding: .utf8)
        }
    }
}
```

**Note**: This is a coordination point that will be called from the rewritten `DependencyContainer` or a new startup function during Epic 6.

**Acceptance criteria**:
- Loads bundled personal-assistant package
- Seeds memories.md and tasks.md if not present
- Registers PersonalAssistantExtension
- Collects skill paths and prompt appends from all packages
- Build succeeds

---

## Summary

After this epic, the project has a complete package system with the personal assistant as the first bundled package. Skills define memory/task/note workflows. The package's prompt append provides the assistant personality.

**New files created**: ~10 (3 Swift files + 7 resource files)
**Lines added**: ~320 Swift + ~200 Markdown
**Existing files modified**: 0
