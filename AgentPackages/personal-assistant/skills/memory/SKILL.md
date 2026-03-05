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
4. If truly new and the file is empty or has no content, use write to create the file with the new fact
5. If truly new and the file already has content, use edit to append a new line at the end
6. If the file doesn't exist, use write to create it

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
