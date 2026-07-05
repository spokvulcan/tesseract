---
name: translate
description: Translate anything into the user's configured target language; text already in the target language goes to English instead. Faithful register, copyable output.
composer-pill: true
---

# Translate

Translate the content the user provides — pasted text or text visible in an
attached screenshot. If nothing follows this skill, translate the most recent
substantial text in the conversation.

## Direction rule

The invocation ends with a line like `Default target language: Ukrainian` —
that is the user's configured target.

1. If the user names a language anywhere in their message ("to German",
   "in French"), that language wins. Ignore the default.
2. Otherwise translate into the default target language.
3. If the text is *already* in the default target language, translate it to
   English instead (one pill covers both directions of the user's workflow).
   When the default target is English, this flip disappears — everything goes
   to English.

## How to translate

- **Faithful register.** Casual stays casual, formal stays formal, profanity
  stays profanity. Translate what was said, not a politer version of it.
- **Meaning over word order.** Produce natural text in the target language;
  do not calque source syntax.
- **Keep untranslatables.** Proper nouns, code, @handles, and URLs pass
  through unchanged.
- **One translation.** Offer alternatives only when the source is genuinely
  ambiguous — then give each variant its own fenced block with a one-line
  label outside the block.

## Output

Return the translation in a single fenced code block, ready to paste. No
commentary, no transliteration, no back-translation — unless the user asked
for it.
