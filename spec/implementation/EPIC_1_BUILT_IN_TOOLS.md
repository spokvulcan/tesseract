# Epic 1: Built-In Tools

> Implement the four Pi-aligned built-in tools (read, write, edit, list) and the path sandboxing layer. All tasks are additive — old tools remain untouched. New tools use the `AgentToolDefinition` struct from Epic 0.

## Prerequisites

- Epic 0 (Foundation Types) — specifically `AgentToolDefinition`, `AgentToolResult`, `ContentBlock`, `CancellationToken`

## Directory

All new files go in `Features/Agent/Tools/BuiltIn/`.

---

## Task 1.1: Implement PathSandbox

**File**: `Features/Agent/Tools/BuiltIn/PathSandbox.swift` (~80 lines)

**What to build**:

```swift
struct PathSandbox: Sendable {
    let root: URL  // The sandbox root directory

    /// Resolve a user-provided path to an absolute path within the sandbox.
    /// - Relative paths resolve against root
    /// - Absolute paths must be within root
    /// - Path traversal (../) that escapes sandbox is rejected
    /// - Symlinks pointing outside sandbox are rejected
    func resolve(_ path: String) throws -> URL

    /// Resolve and verify the file exists
    func resolveExisting(_ path: String) throws -> URL

    /// Resolve for write (parent directory must exist or be creatable within sandbox)
    func resolveForWrite(_ path: String) throws -> URL

    /// Get a display path (relative to sandbox root)
    func displayPath(_ absolutePath: URL) -> String
}

enum PathSandboxError: LocalizedError {
    case outsideSandbox(String)
    case pathTraversal(String)
    case symlinkEscape(String, target: String)
    case fileNotFound(String)
    case isDirectory(String)
    case notDirectory(String)
}
```

**Default sandbox root**: `~/Library/Application Support/tesse-ract/agent/`

**Implementation details**:
- `resolve()`:
  1. If path is relative, join with root
  2. Standardize path (resolve `.`, `..`)
  3. Check `standardizedPath.hasPrefix(root.standardizedPath)` — reject if escapes
  4. Resolve symlinks via `URL.resolvingSymlinksInPath()` — check again after resolution
- `resolveExisting()`: Call `resolve()` + check `FileManager.fileExists()`
- `resolveForWrite()`: Call `resolve()` + verify parent directory is within sandbox
- `displayPath()`: Strip sandbox root prefix to show clean relative path

**Acceptance criteria**:
- `resolve("memories.md")` → `{root}/memories.md`
- `resolve("../etc/passwd")` → throws `.pathTraversal`
- `resolve("/etc/passwd")` → throws `.outsideSandbox`
- `resolve("sub/../other.md")` → `{root}/other.md` (valid, stays in sandbox)
- Build succeeds

**Spec reference**: Section I.5

---

## Task 1.2: Implement ReadTool

**File**: `Features/Agent/Tools/BuiltIn/ReadTool.swift` (~120 lines)

**What to build**:

A factory function that creates an `AgentToolDefinition`:

```swift
func createReadTool(sandbox: PathSandbox) -> AgentToolDefinition
```

**Schema**: `read(path: String, offset?: Int, limit?: Int)`

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | File path relative to working directory |
| `offset` | integer | no | Line number to start from (1-indexed) |
| `limit` | integer | no | Max lines to return |

**Implementation**:
1. Resolve path via `sandbox.resolveExisting()`
2. Detect if image by MIME type (check UTType: .png, .jpeg, .gif, .webp, .tiff)
   - If image: read raw data, return `ContentBlock.image(data:mimeType:)`
3. Read file as UTF-8 string
4. Split into lines
5. Apply offset (convert 1-indexed to 0-indexed)
6. Apply limits:
   - Max 2,000 lines
   - Max 30KB total text
   - Whichever limit hits first
7. Format as `cat -n` style output: `   1\tline content`
8. If truncated, append notice:
   ```
   [Showing lines {start}-{end} of {total}. Use offset={end+1} to continue.]
   ```
9. If a single line exceeds 30KB, truncate with notice:
   ```
   [Line {N} is {size}KB, truncated to 30KB. Content may be incomplete.]
   ```
10. Return `AgentToolResult.text(formattedOutput)`

**Details for UI**: Include `ReadToolDetails` struct with `path`, `lineCount`, `wasTruncated`, `totalLines`.

**Acceptance criteria**:
- Reads a text file and returns numbered lines
- Truncates at 2,000 lines with continuation notice
- Handles offset/limit parameters correctly (1-indexed)
- Returns image content block for image files
- Rejects paths outside sandbox
- Build succeeds

**Spec reference**: Section I.1

---

## Task 1.3: Implement WriteTool

**File**: `Features/Agent/Tools/BuiltIn/WriteTool.swift` (~70 lines)

**What to build**:

```swift
func createWriteTool(sandbox: PathSandbox) -> AgentToolDefinition
```

**Schema**: `write(path: String, content: String)`

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | File path to write |
| `content` | string | yes | Content to write |

**Implementation**:
1. Resolve path via `sandbox.resolveForWrite()`
2. Check cancellation token before write
3. Create parent directories if needed (`FileManager.createDirectory(withIntermediateDirectories: true)`)
4. Write content as UTF-8 data
5. Check cancellation token after write
6. Return success message: `"Wrote {byteCount} bytes to {displayPath}"`

**Details for UI**: `WriteToolDetails` with `path`, `byteCount`, `created` (bool, true if file was new).

**Acceptance criteria**:
- Creates file with correct content
- Creates parent directories automatically
- Returns byte count in result
- Rejects paths outside sandbox
- Build succeeds

**Spec reference**: Section I.2

---

## Task 1.4: Implement EditTool

**File**: `Features/Agent/Tools/BuiltIn/EditTool.swift` (~160 lines)

**What to build**:

```swift
func createEditTool(sandbox: PathSandbox) -> AgentToolDefinition
```

**Schema**: `edit(path: String, old_text: String, new_text: String)`

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | yes | File path to edit |
| `old_text` | string | yes | Exact text to find and replace |
| `new_text` | string | yes | Replacement text |

**Implementation**:
1. Resolve path via `sandbox.resolveExisting()`
2. Read file content
3. Detect line ending style (LF vs CRLF)
4. Strip BOM if present
5. Normalize to LF for matching
6. Count occurrences of `old_text` in content

**Case: Zero matches** →
1. Attempt fuzzy match: normalize whitespace (collapse runs to single space, trim lines)
2. If fuzzy match found, return error with suggestion:
   ```
   No exact match found. Did you mean:

   {fuzzy matched text}

   (Copy the exact text above into old_text)
   ```
3. If no fuzzy match: return error `"No match found for the specified old_text"`

**Case: Multiple matches** →
- Return error: `"Found {count} matches. old_text must match exactly once. Add surrounding context to make it unique."`

**Case: Exactly one match** →
1. Replace `old_text` with `new_text`
2. Restore original line endings
3. Write file
4. Generate unified diff (for UI)
5. Report first changed line number
6. Return success: `"Edited {displayPath}: replaced {old_text_preview} with {new_text_preview}"`

**Diff generation**: Simple unified diff format:
```
--- a/{path}
+++ b/{path}
@@ -{old_start},{old_count} +{new_start},{new_count} @@
-old line
+new line
```

**Details for UI**: `EditToolDetails` with `path`, `diff`, `firstChangedLine`.

**Acceptance criteria**:
- Single exact match: replaces and writes file
- Zero matches: attempts fuzzy, returns helpful error
- Multiple matches: returns count error
- Preserves line endings (CRLF → CRLF, LF → LF)
- Handles BOM
- Returns unified diff in details
- Rejects paths outside sandbox
- Build succeeds

**Spec reference**: Section I.3

---

## Task 1.5: Implement ListTool

**File**: `Features/Agent/Tools/BuiltIn/ListTool.swift` (~80 lines)

**What to build**:

```swift
func createListTool(sandbox: PathSandbox) -> AgentToolDefinition
```

**Schema**: `list(path?: String, recursive?: Bool, limit?: Int)`

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | no | Directory path (defaults to root) |
| `recursive` | boolean | no | List recursively (default: false) |
| `limit` | integer | no | Max entries (default: 200, hard cap: 500) |

**Implementation**:
1. Resolve path via `sandbox.resolve()` (defaults to sandbox root if not provided)
2. Verify it's a directory
3. If recursive:
   - Use `FileManager.enumerator(at:includingPropertiesForKeys:options:)`
   - Collect entries up to limit
   - Skip hidden files (`.` prefix) by default
4. If non-recursive:
   - Use `FileManager.contentsOfDirectory(at:includingPropertiesForKeys:)`
   - Sort alphabetically
5. Format output:
   ```
   {path}/
   ├── file1.md (2.3 KB)
   ├── file2.txt (145 B)
   ├── subdir/
   │   ├── nested.md (1.1 KB)
   │   └── other.txt (890 B)
   └── last.md (500 B)
   ```
6. If truncated: append `[Showing {shown} of {total} entries. Use limit={higher} to see more.]`

**Details for UI**: `ListToolDetails` with `path`, `entryCount`, `wasTruncated`.

**Acceptance criteria**:
- Non-recursive: lists immediate children with sizes
- Recursive: lists full tree up to limit
- Truncates at hard cap (500)
- Tree-style formatting with `├──`, `└──`, `│` characters
- Rejects paths outside sandbox
- Build succeeds

**Spec reference**: Section I.4

---

## Task 1.6: Create BuiltInToolFactory

**File**: `Features/Agent/Tools/BuiltIn/BuiltInToolFactory.swift` (~30 lines)

**What to build**:

A convenience factory that creates all four built-in tools:

```swift
struct BuiltInToolFactory {
    static func createAll(sandbox: PathSandbox) -> [AgentToolDefinition] {
        [
            createReadTool(sandbox: sandbox),
            createWriteTool(sandbox: sandbox),
            createEditTool(sandbox: sandbox),
            createListTool(sandbox: sandbox),
        ]
    }
}
```

This is used later by the `ToolRegistry` rewrite (Epic 6) and the `ExtensionHost` (Epic 4) to get the default tool set.

**Acceptance criteria**:
- Returns array of 4 `AgentToolDefinition` instances
- Each tool has correct name, description, and schema
- Build succeeds

---

## Summary

After this epic, the project has 6 new files in `Features/Agent/Tools/BuiltIn/` implementing the complete Pi-aligned built-in tool surface. The old domain tools remain untouched and functional.

**New files created**: 6
**Lines added**: ~540
**Existing files modified**: 0
