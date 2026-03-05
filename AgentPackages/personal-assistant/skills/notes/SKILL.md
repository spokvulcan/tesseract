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
