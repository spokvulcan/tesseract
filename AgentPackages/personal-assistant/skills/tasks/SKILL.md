---
name: task-management
description: Create, list, complete, or delete tasks in tasks.md. Load when the user mentions to-dos, reminders, or things they need to do.
---

# Task Management

File: `tasks.md` in the working directory. One task per line, markdown checkboxes:

```
- [ ] Pending task
- [x] Completed task
```

## Creating

Read `tasks.md`, then append `- [ ] {description}` with edit — or write the file if it doesn't exist yet.

## Listing

Read `tasks.md`. Report the pending tasks; mention recent completions only briefly.

## Completing

Read the file, find the task's exact line, edit its `- [ ]` to `- [x]`.

## Deleting

Read the file, remove the task's line with edit.
