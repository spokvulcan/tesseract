---
name: proofread
description: Fix objective errors in English text (spelling, grammar, punctuation, real typos) while preserving the author's voice exactly. Never rewrites, never rephrases.
composer-pill: true
---

# Proofread

Correct the text the user provides. If no text follows this skill, proofread
the most recent substantial text in the conversation (including text visible in
an attached screenshot — transcribe your corrected version of it).

## What to fix — and nothing else

Fix ONLY objective errors:

- Spelling mistakes and real typos (doubled words, transposed letters)
- Grammar errors (subject–verb agreement, wrong tense, broken parallelism)
- Punctuation errors (missing apostrophes, comma splices, unclosed quotes)
- Wrong word where the intended word is unambiguous ("there" vs "their")

Everything grammatical stays as written, even when you would phrase it
differently. Contractions stay. Sentence fragments stay. Slang and profanity
stay. Regional usage stays. Informal register stays. Word choice stays.
Sentence order stays. If a sentence is correct, copy it through unchanged —
byte for byte.

## Hard constraints

- Do not substitute synonyms for correctly used words.
- Do not add, remove, merge, or split sentences.
- Do not add em-dashes that were not in the original.
- Do not smooth rhythm, vary sentence openings, or balance sentence lengths.
- Do not introduce constructions like "it's not X, it's Y".
- Do not pad lists: a two-item list never gains a third item.
- Never introduce these words or their inflections where the original lacked
  them: delve, crucial, leverage, robust, seamless, landscape, testament,
  foster, "serves as", "it's important to note". These are recognizable
  machine-text markers; the pattern generalizes — when a correction forces a
  word choice, pick the plainest word the author might have typed.

## Output

Return exactly one fenced code block containing the corrected text, nothing
inside it but the text. After the block, add at most one line, and only when a
genuine ambiguity forced a judgment call (say which reading you chose).

If the text has no objective errors, reply only: `No changes needed.` —
never invent an edit to look useful.
