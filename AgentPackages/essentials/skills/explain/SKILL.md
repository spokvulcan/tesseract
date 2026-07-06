---
name: explain
description: Explain the thing on screen in plain language — code, an error, a config, a contract clause — what it is, then what it means for the user.
composer-pill: true
disable-model-invocation: true
---

# Explain

Explain the content the user provides — code, an error message, a config file,
a legal clause, a chart, anything. If nothing follows this skill, explain the
most recent substantial content in the conversation (including an attached
screenshot).

## How to explain

- **Lead with what it is.** One sentence: "This is a …". Then what it does or
  says, then what it means for the user right now — in that order.
- **Plain language.** No jargon unless the input itself is jargon, and then
  define each term the first time you use it. Prefer a concrete example over
  an abstract definition.
- **Match depth to the input.** A one-line error gets a short paragraph; a
  whole file gets a walkthrough of the parts that matter, not line-by-line
  narration.
- **Call out the sharp edges.** If the thing has a consequence the user likely
  cares about (this error is fatal vs ignorable, this clause waives a right,
  this code has a side effect), say so plainly.
- **Say what you can't tell.** If the meaning depends on context you don't
  have, name what's missing instead of guessing.
- If the user adds a question after this skill, answer that question first.
