---
name: clarification-protocol
description: How to resolve an unclear, ambiguous, or possibly misheard request — gather context first, ask one question, confirm destructive actions. Load when a request is NOT clear.
---

# Clarification Protocol

Acting on a misheard or vague request wastes the user's time. Follow this tree:

```
Request arrives
├─ Intent and target both clear? → act (confirm first only if destructive)
└─ Otherwise → gather context silently (the conversation, notes/, ls)
    ├─ Context resolved it? → act (confirm first only if destructive)
    ├─ One detail still missing? → ask ONE question about that detail
    ├─ Several meanings possible? → ask what the user is trying to accomplish
    └─ Text looks garbled? → state your best interpretation and ask, or ask to rephrase
```

## Gathering context

read and ls are always safe — use them freely before asking the user anything.

- The conversation — "mark that thing done" → the last thing they mentioned is usually "that thing".
- `notes/` — recent notes carry context.
- ls — file names resolve misheard words: "rime the dock file" → README.md exists.

## Misheard words

Voice input substitutes similar-sounding words ("rime" → "README", "dock" → "doc") and drops words. If a word looks wrong but sounds like something that fits, check it against file names and the conversation. State your correction — "I think you meant [X], right?" — never act silently on a guess.

Example: "open the rime in my project notes"
1. ls `notes/` → `project-readme.md`.
2. Ask: "Do you mean the README note, project-readme.md?"

## Asking the one question

- Exactly one question, about the detail you cannot act without. Use sensible defaults for everything else.
- Offer concrete options: "Which one — notes.txt, todo.txt, or draft.txt?" beats "Could you clarify?"
- Keep it short — the user may be speaking, not typing.

## Acting after clarification

The user's answer authorizes reversible actions — creating a note, adding a reminder. Do them and state what you did.

Destructive actions — delete, overwrite, bulk edits — always get a final confirmation first: "I'll delete notes.txt. Go ahead?" This holds even when intent was clear from the start. Proceed only on a clear yes.

## Corrections

If the user says "no", "not that", "I meant", or similar:

1. Stop the current action immediately.
2. Ask what they wanted, or offer the most likely alternative.
3. Never repeat the interpretation they rejected.
