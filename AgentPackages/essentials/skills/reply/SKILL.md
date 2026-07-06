---
name: reply
description: Draft a reply to the message on screen — email, Slack, X, or chat — in the user's voice, matched to the medium's register and length.
composer-pill: true
disable-model-invocation: true
---

# Reply

Draft a reply to the content the user provides — usually a screenshot of an
email, a Slack or chat message, or an X thread. If nothing follows this skill,
reply to the most recent message content in the conversation.

## How to draft

- **Identify the medium** from the content and match its register: emails get
  greetings and sign-offs, Slack gets neither, X replies are tight and public.
- **Match the length to the medium and the message.** A two-line Slack ping
  gets a two-line answer. Do not pad.
- **Write in the user's voice.** If the user's own words appear in the
  conversation, mirror their tone and formality. Plain words, no corporate
  filler ("I hope this finds you well", "per my last message").
- **Answer the actual asks.** If the message contains questions or requests,
  address each one. If information you need is missing, draft around it with a
  clearly marked placeholder like `[date]` rather than inventing facts.
- If the user adds instructions after this skill (tone, decision, key points),
  those win.

## Output

Return the draft in a single fenced code block, ready to paste. If a decision
in the draft was a coin flip (accept vs decline, formal vs casual), add one
line after the block naming the choice you made.
