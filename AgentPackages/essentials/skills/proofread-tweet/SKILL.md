---
name: proofread-tweet
description: Polish a tweet or tweet thread for reach — hook-first, formatted for skimming, engagement-bait stripped. Returns 2–3 copyable variants for a single tweet, one polished pass for a thread.
composer-pill: true
disable-model-invocation: true
---

# Proofread Tweet

Polish the tweet or thread the user provides. If nothing follows this skill,
use the most recent draft in the conversation (including text visible in an
attached screenshot).

**On a screenshot:** the draft is the text inside the compose box if one is
visible — not the surrounding timeline, replies, or UI text. Only when no
compose box exists, take the most draft-like text on screen.

## Craft rules

- **Hook first.** The first line decides everything. Make it clear, not
  clever: concrete specifics, numbers, or a curiosity gap the tweet actually
  pays off. No throat-clearing ("So I've been thinking…").
- **Format for skimming.** Short sentences. Line breaks between ideas.
  Whitespace is free. One idea per tweet.
- **No hashtags.** They read as spam and add nothing to distribution.
- **Links leave the body.** A link belongs in the first reply, never in the
  post itself — link posts measurably underperform. If the draft has a link,
  move it and say so.
- **Optimize for replies and dwell time.** Replies the author engages with are
  weighted far above likes in ranking (directionally, not gospel). Prefer
  phrasing that invites a response over phrasing that closes the topic.
- **Strip engagement bait and machine-text tells.** No "Let that sink in", no
  "Thread 🧵👇", no "Who's with me?", no emoji garlands, no "delve"/"game-
  changer" vocabulary. Keep the author's voice — edit toward reach, not toward
  a different person.

## Single tweet → variants

Return 2–3 variants, strongest first, each in its own fenced code block with
nothing else inside it. After the blocks, one line on why the first variant
wins. Respect the 280-character limit; show a character count only if a
variant is close to it.

## Thread → one polished pass

Return the whole thread once, each tweet in its own fenced code block, numbered
outside the blocks. Structure: hook tweet → one point per tweet → TL;DR →
a single call to action at the end. Re-hook the reader every 1–2 tweets (a
forward pull: "here's where it gets interesting"). 6–8 tweets is the sweet
spot; suggest cuts if the draft runs long.
