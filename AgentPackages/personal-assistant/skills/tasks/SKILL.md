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
2. If the file is empty or doesn't exist, use write to create it with `- [ ] {task description}`
3. If the file already has content, use edit to append a new line: `- [ ] {task description}`

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
