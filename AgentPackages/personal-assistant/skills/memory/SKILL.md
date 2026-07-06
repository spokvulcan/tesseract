---
name: memory-management
description: Save, update, or delete personal facts and preferences in memories.md. Load when the user shares a lasting fact or asks to remember, change, or forget something.
---

# Memory Management

File: `memories.md` in the working directory. One fact per line, plain text — the line itself is the identifier.

Save durable facts: preferences, people, habits, dates. Not one-off details of the current conversation.

## Saving

1. Read `memories.md`.
2. If a similar fact already exists, edit that line instead of adding a new one.
3. Otherwise append the fact as a new line with edit — or write the file if it doesn't exist yet.

## Updating

Read the file, find the exact line, edit it to the corrected fact.

## Deleting

Read the file, edit the line (including its trailing newline) to an empty string.

## Consolidating

When the file has grown large or repetitive: read it all, then write a merged version — duplicates combined, outdated facts dropped.
