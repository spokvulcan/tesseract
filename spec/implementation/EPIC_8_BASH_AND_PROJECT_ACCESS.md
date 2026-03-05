# Epic 8: Bash Tool & Project Directory Access

> Add a bash tool to the agent, enable user-granted project directory access via security-scoped bookmarks, and update the system prompt and sandbox to support coding-agent workflows. Inspired by Pi's `read`, `write`, `edit`, `bash` tool surface — adapted for macOS App Sandbox constraints.

## Motivation

The on-device LLM is trained to use bash commands for exploration and automation. Currently the agent can only read/write/edit files inside its own `~/Library/Application Support/tesse-ract/agent/` container. By adding:

1. A **bash tool** that executes shell commands via `Process`
2. **Security-scoped bookmark** support so the user can grant access to a project directory
3. A **multi-root PathSandbox** that can serve both the agent data dir and user-granted directories

…the agent becomes a capable coding assistant that can `ls`, `grep`, `cat`, run scripts, and modify code — all within the macOS App Sandbox.

## Prerequisites

- Epics 0-1 (Foundation Types + Built-In Tools) — specifically `AgentToolDefinition`, `PathSandbox`, `BuiltInToolFactory`
- Epic 2 (Core Loop) — `AgentLoop`, `Agent`, `AgentContext`
- Epic 3 (Context and Prompt) — `SystemPromptAssembler`

## Design Decisions

### Sandbox Strategy: Option A (Sandbox-Contained with User-Granted Access)

The app stays sandboxed (App Store compatible). Shell commands run inside the sandbox — the child `Process` inherits the app's sandbox profile. The user grants access to project directories via `NSOpenPanel`, stored as security-scoped bookmarks.

### What Works Inside the Sandbox

| Command | Works? | Notes |
|---------|--------|-------|
| `/bin/ls`, `/bin/cat`, `/bin/mkdir`, `/bin/rm`, `/bin/cp`, `/bin/mv` | Yes | Within sandbox + user-granted paths |
| `/usr/bin/grep`, `/usr/bin/sed`, `/usr/bin/awk`, `/usr/bin/sort`, `/usr/bin/wc` | Yes | On accessible files |
| `/usr/bin/find` | Yes | Within accessible paths |
| `/usr/bin/diff`, `/usr/bin/head`, `/usr/bin/tail` | Yes | On accessible files |
| `/bin/echo`, `/usr/bin/printf`, `/usr/bin/tr` | Yes | Always (stdin/stdout only) |
| `/usr/bin/env`, `/usr/bin/which` | Limited | Restricted PATH |
| `/bin/ps`, `/usr/sbin/system_profiler` | No | System info blocked |
| `/usr/bin/open` | No | Launch Services restricted |
| Homebrew tools (`/opt/homebrew/bin/*`) | No | Outside sandbox |
| `curl`, `python3`, `node`, `git` | Depends | If in `/usr/bin/` and entitlements allow |

### Entitlement Changes

Add to **both** `tesseract.entitlements` (Debug) and `tesseractRelease.entitlements` (Release):

```xml
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

This entitlement is required for `NSOpenPanel` to grant read-write access and for security-scoped bookmarks to work. It is a standard App Store entitlement.

### No Additional Entitlements Needed

- `com.apple.security.network.client` — already present (for model downloads; also enables `curl`)
- No temporary exceptions needed — all file access is user-granted

## Directory

```
Features/Agent/Tools/BuiltIn/
├── BashTool.swift              ← NEW: Bash tool implementation
├── PathSandbox.swift           ← MODIFIED: Multi-root support
├── BuiltInToolFactory.swift    ← MODIFIED: Add bash tool
├── ReadTool.swift              ← unchanged
├── WriteTool.swift             ← unchanged
├── EditTool.swift              ← unchanged
└── ListTool.swift              ← unchanged

Features/Agent/ProjectAccess/
├── ProjectAccessManager.swift  ← NEW: Security-scoped bookmarks + NSOpenPanel
└── ProjectAccessView.swift     ← NEW: UI for managing granted directories

Features/Agent/Context/
└── SystemPromptAssembler.swift ← MODIFIED: Add bash tool description + working dir

App/
└── DependencyContainer.swift   ← MODIFIED: Wire project access + multi-root sandbox

tesseract/
├── tesseract.entitlements      ← MODIFIED: Add user-selected.read-write
└── tesseractRelease.entitlements ← MODIFIED: Add user-selected.read-write
```

---

## Task 8.1: Add `user-selected.read-write` Entitlement

**Files**:
- `tesseract/tesseract.entitlements`
- `tesseract/tesseractRelease.entitlements`

**What to change**: Add the `com.apple.security.files.user-selected.read-write` entitlement to both files.

```xml
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

**Acceptance criteria**:
- Both entitlement files contain the new key
- Build succeeds
- Existing entitlements unchanged

---

## Task 8.2: Implement ProjectAccessManager

**File**: `Features/Agent/ProjectAccess/ProjectAccessManager.swift` (~150 lines)

**What to build**:

```swift
@MainActor
@Observable
final class ProjectAccessManager {
    /// Currently granted project directories (resolved bookmark URLs).
    private(set) var grantedDirectories: [URL] = []

    /// The currently active working directory for the agent.
    /// Defaults to the agent data directory if no project is selected.
    private(set) var activeProjectDirectory: URL?

    /// Present NSOpenPanel to let the user pick a project directory.
    func requestProjectAccess() async -> URL?

    /// Start accessing a bookmarked URL.
    func activateProject(_ url: URL)

    /// Stop accessing a bookmarked URL.
    func deactivateProject(_ url: URL)

    /// Load persisted bookmarks on launch.
    func loadBookmarks()

    /// Remove a bookmark.
    func removeBookmark(for url: URL)
}
```

**Implementation details**:

1. **Bookmark storage**: Persist security-scoped bookmark `Data` blobs in UserDefaults under key `"grantedProjectBookmarks"` as `[String: Data]` (path string → bookmark data).

2. **Requesting access**:
   ```
   1. Create NSOpenPanel with:
      - canChooseDirectories = true
      - canChooseFiles = false
      - allowsMultipleSelection = false
      - message = "Select a project directory for the agent to access"
      - prompt = "Grant Access"
   2. Run panel, get selected URL
   3. Create security-scoped bookmark:
      url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
   4. Store bookmark data in UserDefaults
   5. Start accessing: url.startAccessingSecurityScopedResource()
   6. Add to grantedDirectories
   7. Return URL
   ```

3. **Loading bookmarks on launch**:
   ```
   1. Read [String: Data] from UserDefaults
   2. For each entry:
      a. Resolve bookmark: URL(resolvingBookmarkData:options:.withSecurityScope, bookmarkDataIsStale:)
      b. If stale, re-create bookmark and update storage
      c. Call url.startAccessingSecurityScopedResource()
      d. Add to grantedDirectories
   3. If any resolve fails, remove from storage
   ```

4. **Deactivation**:
   ```
   1. Call url.stopAccessingSecurityScopedResource()
   2. Remove from grantedDirectories
   ```

5. **Active project**: The `activeProjectDirectory` is the directory the agent treats as its `cwd` for bash commands and file operations. Set via `activateProject()`.

**Acceptance criteria**:
- `NSOpenPanel` presents correctly
- Bookmark is persisted and survives app restart
- `startAccessingSecurityScopedResource()` called on load
- Stale bookmarks re-created
- Build succeeds

**Spec reference**: macOS App Sandbox documentation, security-scoped bookmarks

---

## Task 8.3: Extend PathSandbox for Multi-Root Support

**File**: `Features/Agent/Tools/BuiltIn/PathSandbox.swift` (modify existing)

**What to change**:

Currently `PathSandbox` has a single `root: URL`. Extend it to support multiple allowed roots so the agent can access both its data directory and user-granted project directories.

```swift
nonisolated struct PathSandbox: Sendable {
    let roots: [URL]          // ← was: let root: URL

    /// Primary root (first in list). Used for relative path resolution.
    var primaryRoot: URL { roots[0] }

    /// Legacy accessor for backward compatibility.
    var root: URL { primaryRoot }

    /// Single-root initializer (backward compatible)
    init(root: URL) { self.roots = [root] }

    /// Multi-root initializer
    init(roots: [URL]) {
        precondition(!roots.isEmpty, "PathSandbox requires at least one root")
        self.roots = roots
    }
}
```

**Changes to `resolve()`**:
```
1. If path is relative, resolve against primaryRoot (first root)
2. If path is absolute, check if it falls within ANY root
3. Symlink checks: verify resolved path falls within ANY root
```

**Changes to `displayPath()`**:
```
1. Check each root in order
2. If absolute path is within a root, strip that root prefix
3. If no root matches, return absolute path
```

**New method — `isWithinSandbox(_ url: URL) -> Bool`**:
```
Check if the URL falls within any of the allowed roots.
Used by BashTool to validate Process cwd.
```

**Acceptance criteria**:
- Single-root behavior unchanged (backward compatible)
- Multi-root: `resolve("/Users/me/project/src/main.swift")` succeeds when `/Users/me/project/` is a root
- Multi-root: `resolve("/etc/passwd")` still throws `.outsideSandbox`
- `displayPath` shows relative path from the matching root
- All existing tool tests still pass
- Build succeeds

---

## Task 8.4: Implement BashTool

**File**: `Features/Agent/Tools/BuiltIn/BashTool.swift` (~250 lines)

**What to build**:

```swift
func createBashTool(sandbox: PathSandbox) -> AgentToolDefinition
```

**Schema**: `bash(command: String, timeout?: Int)`

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | The bash command to execute |
| `timeout` | integer | no | Timeout in seconds (default: 120, max: 300) |

**Tool description** (shown to LLM):
```
Execute a bash command. The command runs in the current working directory with the same sandbox restrictions as the app. Output (stdout + stderr combined) is truncated to the last 2000 lines or 50KB, whichever is hit first. Use timeout for long-running commands.
```

**Implementation**:

```swift
// Core execution flow
func executeBash(command: String, cwd: URL, timeout: Int, signal: CancellationToken?,
                 onUpdate: ToolProgressCallback?) async throws -> AgentToolResult
{
    // 1. Create Process
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", command]
    process.currentDirectoryURL = cwd

    // 2. Set up pipes
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    process.standardInput = FileHandle.nullDevice

    // 3. Merge stdout + stderr into single output stream
    // Use DispatchIO or readabilityHandler to stream data

    // 4. Launch
    try process.run()

    // 5. Set up timeout
    let timeoutTask = Task {
        try await Task.sleep(for: .seconds(timeout))
        process.terminate()  // SIGTERM first
        try? await Task.sleep(for: .seconds(2))
        if process.isRunning { process.interrupt() } // SIGINT fallback
    }

    // 6. Set up cancellation
    // Poll signal.isCancelled periodically or use Task cancellation

    // 7. Collect output with rolling buffer
    // Keep last 50KB or 2000 lines (tail truncation, matching Pi)

    // 8. Wait for process exit
    process.waitUntilExit()
    timeoutTask.cancel()

    // 9. Format result
    let exitCode = process.terminationStatus
    let output = collectedOutput  // truncated

    if exitCode == 0 {
        return AgentToolResult(
            content: [.text(output.isEmpty ? "(no output)" : output)],
            details: BashToolDetails(exitCode: Int(exitCode), wasTruncated: wasTruncated)
        )
    } else {
        return .error("Exit code \(exitCode)\n\(output)")
    }
}
```

**Output handling** (following Pi's approach):
- Combine stdout and stderr into a single stream (interleaved, as user would see)
- Rolling buffer: keep accumulating, but only retain the last `maxBytes` (50KB) or `maxLines` (2000)
- If truncated, prepend notice: `[Output truncated. Showing last {N} lines of {total}.]`
- Strip ANSI escape codes from output before returning to LLM
- Handle binary output gracefully (replace non-UTF-8 bytes)

**Timeout handling**:
- Default: 120 seconds
- Max: 300 seconds (5 minutes)
- On timeout: SIGTERM → wait 2s → SIGINT
- Return error: `"Command timed out after {N} seconds"`

**Cancellation**:
- Check `signal.isCancelled` before launch
- On cancel: `process.terminate()`
- Return error: `"Command cancelled"`

**Streaming progress** (via `onUpdate` callback):
- Emit partial output every ~500ms while command runs
- Allows UI to show real-time bash output

**Working directory**:
- Use `sandbox.primaryRoot` as default cwd
- If `ProjectAccessManager.activeProjectDirectory` is set, use that instead
- The cwd is injected at tool creation time (not per-call)

**Details for UI**:
```swift
nonisolated struct BashToolDetails: Sendable, Hashable {
    let command: String
    let exitCode: Int
    let wasTruncated: Bool
    let timedOut: Bool
    let durationMs: Int
}
```

**Acceptance criteria**:
- `bash(command: "echo hello")` → `"hello\n"`
- `bash(command: "ls")` in project dir → lists project files
- `bash(command: "cat nonexistent")` → error with exit code 1
- `bash(command: "sleep 999", timeout: 2)` → timeout error
- Output truncation works (test with `seq 10000`)
- ANSI codes stripped
- Cancellation terminates process
- Streaming progress emitted
- Build succeeds

**Spec reference**: Pi `bash.ts`

---

## Task 8.5: Add ANSI Stripping Utility

**File**: `Features/Agent/Tools/BuiltIn/ANSIStripper.swift` (~40 lines)

**What to build**:

```swift
nonisolated enum ANSIStripper: Sendable {
    /// Remove ANSI escape sequences from a string.
    /// Handles: SGR (colors), cursor movement, erase, OSC (title).
    static func strip(_ input: String) -> String

    /// Remove non-printable control characters (except \t \n \r).
    static func sanitize(_ input: String) -> String
}
```

**Implementation**:
- Regex: `\x1B\[[0-9;]*[A-Za-z]` for CSI sequences
- Regex: `\x1B\][^\x07]*\x07` for OSC sequences
- Regex: `\x1B[()][AB012]` for charset sequences
- Strip control chars: `[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]`
- Use `String.replacing(_:with:)` (Swift Regex, available on macOS 13+)

**Acceptance criteria**:
- `strip("\u{1B}[31mred\u{1B}[0m")` → `"red"`
- `strip("\u{1B}]0;title\u{07}")` → `""`
- `sanitize("hello\u{00}world")` → `"helloworld"`
- Preserves tabs, newlines, carriage returns
- Build succeeds

---

## Task 8.6: Update BuiltInToolFactory

**File**: `Features/Agent/Tools/BuiltIn/BuiltInToolFactory.swift` (modify existing)

**What to change**:

Add the bash tool to the factory. Make bash optional (only included when a working directory is available).

```swift
nonisolated enum BuiltInToolFactory: Sendable {

    /// Create all file tools (read, write, edit, list).
    static func createFileTools(sandbox: PathSandbox) -> [AgentToolDefinition] {
        [
            createReadTool(sandbox: sandbox),
            createWriteTool(sandbox: sandbox),
            createEditTool(sandbox: sandbox),
            createListTool(sandbox: sandbox),
        ]
    }

    /// Create all tools including bash.
    static func createAll(sandbox: PathSandbox) -> [AgentToolDefinition] {
        createFileTools(sandbox: sandbox) + [
            createBashTool(sandbox: sandbox),
        ]
    }
}
```

**Acceptance criteria**:
- `createAll()` returns 5 tools (read, write, edit, list, bash)
- `createFileTools()` returns 4 tools (backward compatible)
- Build succeeds

---

## Task 8.7: Update SystemPromptAssembler

**File**: `Features/Agent/Context/SystemPromptAssembler.swift` (modify existing)

**What to change**:

Update the default core prompt to include bash tool description and working directory context.

```swift
static let defaultCorePrompt = """
    You are an expert local assistant operating inside Tesseract, a tool-calling agent harness.
    You help users by reading files, editing files, writing files, running commands, and using other tools provided by the current package or project.

    Built-in tools:
    - read: Read file contents (text or images)
    - write: Create or overwrite files
    - edit: Make surgical edits to files (find exact text and replace)
    - list: List files and directories
    - bash: Execute bash commands (ls, grep, find, git, etc.)

    Guidelines:
    - Use bash for file discovery (ls, find, grep), running tests, and automation
    - Use read to examine file contents before editing
    - Use edit for precise, surgical changes (old_text must match exactly once)
    - Use write only for new files or complete rewrites
    - Use list for quick directory overviews
    - Be concise in your responses
    - Show file paths clearly when working with files
    - Prefer edit over write for existing files
    - Prefer bash grep/find for searching across many files

    Environment:
    - Commands run inside the macOS App Sandbox
    - File access is limited to the agent data directory and any user-granted project directories
    - Standard Unix tools (/bin/*, /usr/bin/*) are available
    - Homebrew tools and user-installed tools are NOT available
    - git is available if installed at /usr/bin/git

    Tesseract resources:
    - Project context files are loaded from AGENTS.md or CLAUDE.md
    - The system prompt may be replaced by SYSTEM.md or extended by APPEND_SYSTEM.md
    - Skills may be available and should be used when relevant
    """
```

Also add the active project directory to section 6 of the assembly:

```swift
// 6. Working directory
if let projectDir = projectDirectory {
    sections.append("Working directory: \(projectDir.path)")
    sections.append("Agent data directory: \(agentRoot.path)")
} else {
    sections.append("Working directory: \(agentRoot.path)")
}
```

**Changes to `assemble()` signature**:

```swift
static func assemble(
    defaultPrompt: String = defaultCorePrompt,
    loadedContext: ContextLoader.LoadedContext,
    skills: [SkillMetadata],
    tools: [AgentToolDefinition],
    dateTime: Date = Date(),
    agentRoot: String,
    projectDirectory: URL? = nil    // ← NEW
) -> String
```

**Acceptance criteria**:
- System prompt includes bash tool description
- Working directory shows project dir when set
- Backward compatible (projectDirectory defaults to nil)
- Build succeeds

---

## Task 8.8: Implement ProjectAccessView

**File**: `Features/Agent/ProjectAccess/ProjectAccessView.swift` (~120 lines)

**What to build**:

A SwiftUI view for managing project directory access. Can be embedded in the agent chat header or settings.

```swift
struct ProjectAccessView: View {
    @Environment(ProjectAccessManager.self) var projectAccess

    var body: some View {
        // 1. Current active project (if any)
        // 2. List of granted directories with remove buttons
        // 3. "Open Project" button → NSOpenPanel
    }
}
```

**UI layout**:

```
┌─────────────────────────────────────┐
│ Project Access                      │
│                                     │
│ Active: ~/projects/my-app      [×]  │
│                                     │
│ Granted Directories:                │
│ • ~/projects/my-app            [×]  │
│ • ~/projects/other-project     [×]  │
│                                     │
│ [+ Open Project Directory]          │
└─────────────────────────────────────┘
```

**Features**:
- Show shortened paths (`~` prefix for home directory)
- Click a granted directory to make it active
- Active directory highlighted
- Remove button revokes access
- "Open Project Directory" triggers `NSOpenPanel` via `ProjectAccessManager`

**Acceptance criteria**:
- View renders with correct layout
- Open button triggers NSOpenPanel
- Selected directory appears in list
- Remove button works
- Active directory is visually indicated
- Build succeeds

---

## Task 8.9: Wire Everything in DependencyContainer

**File**: `App/DependencyContainer.swift` (modify existing)

**What to change**:

1. Add `ProjectAccessManager` as a dependency
2. Load bookmarks on app launch
3. Create multi-root `PathSandbox` that includes both agent root and active project directory
4. Pass project directory to `SystemPromptAssembler`
5. Rebuild tools when active project changes

```swift
// In DependencyContainer:

lazy var projectAccessManager = ProjectAccessManager()

lazy var agentSandbox: PathSandbox = {
    var roots = [PathSandbox.defaultRoot]
    if let projectDir = projectAccessManager.activeProjectDirectory {
        roots.insert(projectDir, at: 0)  // Project dir is primary root
    }
    return PathSandbox(roots: roots)
}()
```

**On project change**:
When `ProjectAccessManager.activeProjectDirectory` changes, the sandbox and tools need to be recreated. This can be done via:

1. `Agent` exposes a `reconfigure(tools:systemPrompt:)` method
2. `AgentCoordinator` (or whoever wires things) observes `projectAccessManager.activeProjectDirectory`
3. On change: rebuild sandbox → rebuild tools → rebuild system prompt → call `agent.reconfigure()`

**This task defines the wiring strategy. The actual `reconfigure()` method on `Agent` may already exist or can be added as a simple setter.**

**Acceptance criteria**:
- `ProjectAccessManager` created and initialized on app launch
- Bookmarks loaded on launch
- Sandbox includes project directory when set
- Tools use the multi-root sandbox
- System prompt reflects active project
- Build succeeds

---

## Task 8.10: Inject ProjectAccessManager as EnvironmentObject

**File**: `App/TesseractApp.swift` (modify existing)

**What to change**:

Add `ProjectAccessManager` to the environment so views can access it:

```swift
.environment(container.projectAccessManager)
```

**Also**: Add a UI trigger point for project access. Options:
- Add to the agent chat toolbar/header
- Add to the sidebar as a section
- Add to settings

Recommended: Add a toolbar button in the agent chat view that opens a popover with `ProjectAccessView`.

**Acceptance criteria**:
- `ProjectAccessManager` available via `@Environment`
- UI trigger point exists in the agent chat view
- Build succeeds

---

## Summary

After this epic, the agent has:

1. **Bash tool**: Execute shell commands with timeout, cancellation, output truncation, and streaming progress
2. **Project directory access**: User grants access to project directories via NSOpenPanel, persisted as security-scoped bookmarks
3. **Multi-root sandbox**: File tools work on both agent data and project files
4. **Updated system prompt**: Describes bash capabilities and environment constraints to the LLM

**New files created**: 4
- `BashTool.swift` (~250 lines)
- `ANSIStripper.swift` (~40 lines)
- `ProjectAccessManager.swift` (~150 lines)
- `ProjectAccessView.swift` (~120 lines)

**Modified files**: 5
- `PathSandbox.swift` (multi-root support)
- `BuiltInToolFactory.swift` (add bash)
- `SystemPromptAssembler.swift` (bash description + project dir)
- `DependencyContainer.swift` (wire project access)
- `TesseractApp.swift` (environment injection)

**Entitlement changes**: 2
- `tesseract.entitlements` (add user-selected.read-write)
- `tesseractRelease.entitlements` (add user-selected.read-write)

**Lines added**: ~560 new, ~100 modified
**App Store compatible**: Yes — uses standard entitlements only

## Risk Areas

1. **Process in sandbox**: Child processes inherit the sandbox. Some commands may fail unexpectedly. The system prompt warns the LLM about limitations. Testing with real projects will reveal edge cases.

2. **Security-scoped bookmark staleness**: Bookmarks can become stale after macOS updates or disk changes. The `loadBookmarks()` implementation handles this by re-creating stale bookmarks when possible.

3. **Working directory changes during conversation**: If the user switches projects mid-conversation, the agent's context may reference files from the old project. Consider clearing conversation or warning the user.

4. **Output encoding**: Process output may contain non-UTF-8 bytes (e.g., binary file reads). The `sanitize()` function handles this, but edge cases may exist.

5. **Process lifecycle**: If the app crashes or is killed during bash execution, orphaned processes may remain. Using `Process.terminate()` in the app's `applicationWillTerminate` handler mitigates this.
