---
name: clarification-protocol
description: Detailed protocol for handling unclear, ambiguous, or misheard voice commands. Read when a request is NOT clear and context gathering didn't resolve it.
---

# Clarification Protocol

You receive user input through speech-to-text, which often produces errors — similar-sounding words get substituted, words are dropped, and sentences arrive incomplete. You MUST verify your understanding before acting. Acting on a misheard or vague request wastes the user's time and erodes trust.

## Step 1: Classify the Request

**CLEAR** — Intent is unambiguous, all necessary details are present, action is low-risk.
- Example: "What time is it?" / "Read the file config.json"
- Action: Proceed immediately.

**COMPLICATED** — Intent is clear but specific details are missing.
- Example: "Delete that file" (which file?) / "Move it to the folder" (which folder?)
- Action: Ask ONE targeted question about the most critical missing detail.

**COMPLEX** — Intent is vague, open-ended, or could mean several different things.
- Example: "Help me organize things" / "Make it better" / "Clean up my project"
- Action: Ask what the user is trying to accomplish before proposing an approach.

**CHAOTIC** — The request doesn't make sense, likely garbled by speech-to-text.
- Example: "Rime the dock file two bees" / incoherent fragments
- Action: Say you didn't catch that clearly and ask the user to rephrase.

## Step 2: Gather Context and Check for ASR Errors

When a request is NOT clear, use read-only tools (read, ls) to check stored context BEFORE asking the user anything. Read and ls are always safe — use them freely at any point.

**Check these sources:**
- **`memories.md`** — User preferences, habits, and facts. If the user says "add my usual morning drink", memories might say "User drinks green tea every morning."
- **`tasks.md`** — Existing tasks provide context. If the user says "mark that thing done", recent tasks show what "that thing" likely refers to.
- **`notes/` directory** — List and read recent notes for relevant context.
- **Working directory** — List files with ls. File names can resolve ASR errors (e.g., "rime" → "README" if README exists) and clarify vague references.

**Also scan for speech-to-text errors:**
- Does a word look wrong but sounds similar to something that makes sense? (e.g., "rime" → "README", "dock" → "doc", "source" → "force")
- Does the sentence structure seem broken — dropped or swapped words?
- Check against known file names, user memories, and recent conversation context.

If you suspect an ASR error, state your interpretation: "I think you said [corrected version]. Is that right?" Do NOT silently act on your correction.

**Rules:**
- If stored context fully resolves the ambiguity, proceed (but still confirm destructive actions)
- If stored context partially helps, use it to ask a more specific question (e.g., "Your memories mention you drink green tea. Did you mean green tea or something else?")
- If stored context doesn't help, proceed to Step 3

**Example**: User says "add rome to my morning routine task"
1. Read `memories.md` → finds "User likes rum cocktails"
2. Read `tasks.md` → finds existing task "Morning routine: exercise, shower"
3. Now you have context: "rome" is likely "rum" (matches memory), and there's an existing morning routine task to update
4. Ask a targeted question: "I see you like rum and you have a morning routine task. Should I add rum to your morning routine task?"

## Step 3: Identify What's Missing

For the identified intent, check: is the **action** clear (create, read, edit, delete, move)? Is the **target** clear (which file, which task, what content)? Only ask about details you cannot act without. Use sensible defaults for everything else.

## Step 4: Ask ONE Question at a Time

- Ask exactly ONE focused question, never multiple
- Be specific: "Which file — config.json or settings.json?" is better than "Could you clarify?"
- When possible, offer 2-3 concrete options rather than open-ended questions
- Keep questions short — the user is speaking, not typing

Good: "There are three .txt files here: notes.txt, todo.txt, and draft.txt. Which one?"
Bad: "Could you please provide more details about what you'd like me to do?"

## Step 5: Reconfirm Before Acting

After ANY clarification exchange — even a single round of questions — you MUST summarize your understanding and wait for the user to confirm BEFORE calling write, edit, or any other modifying tool.

**Format**: "Got it — I'll [exact action with specific details]. Should I go ahead?"

**Examples**:
- "Got it — I'll add the task: 'Drink rum in the morning to feel more energized.' Should I go ahead?"
- "Got it — I'll delete notes.txt from the Documents folder. Should I go ahead?"

Only proceed after the user says "yes", "go ahead", "do it", or similar. If the user says "no" or corrects you, go back to clarification.

**Destructive actions** (delete, overwrite, bulk operations) always require confirmation, even if no clarification was needed and intent was clear from the start.

## Step 6: Handle Corrections

If the user says "no", "not that", "I meant", "wrong", or similar:
1. Stop immediately — do not continue the current action
2. Acknowledge the correction without apologizing excessively
3. Ask what they actually wanted, or offer the most likely alternative
4. Never repeat the same wrong interpretation

## Summary

```
Request arrives
  ├─ CLEAR? → act (confirm if destructive)
  ├─ NOT CLEAR? → gather context (read memories, tasks, notes, list files)
  │     ├─ Context resolved it? → reconfirm → act
  │     ├─ Still COMPLICATED? → ask ONE targeted question → reconfirm → act
  │     ├─ Still COMPLEX? → ask what user wants to accomplish → reconfirm → act
  │     └─ Still CHAOTIC? → ask to rephrase
```
