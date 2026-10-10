# ADR-0090: Jarvis sees a picture in the turn it's shown

- Status: Accepted (built; the loaded-model gate is under As built)
- Date: 2026-10-10
- Amends: ADR-0080 (the Day Thread takes pictures)
- Relates to: ADR-0007 (Cache Key Space, Minimum Warm Offset), ADR-0012
  (chain-prefix restore points), ADR-0013 (the vision container loads
  eagerly), ADR-0014 (the vision tower's patch guard), ADR-0089 (images
  speculate)

## Context

The Companion took words only. Much of the owner's day arrives as pictures —
a letter from school, an invite in a chat, a flyer, a booking confirmation —
and getting one into the day meant typing it out, or asking the agent chat,
which can see it but doesn't keep the day.

The Day Thread is not the agent chat. Moments append to it all day, and each
one sends the whole thread; the agent chat grows only when the owner writes.
Kept the way the agent chat keeps them, a picture would ride every later
request of the day:

- About 1,300 tokens each at the default Vision Token Budget, in every
  moment's prompt.
- A cold re-prefill of the thread — a relaunch whose SSD tier missed, the
  web-access switch (it changes the tool list), a model switch, compaction
  rewriting the head — feeds every picture still in it to the vision tower in
  one forward. The tower's global attention allocates one `[16, ΣP, ΣP]`
  score matrix: about 0.84 GB × N² for N screenshots of 5,120 patches each.
  The ADR-0014 guard rejects such a forward past the Metal buffer limit,
  after a handful of screenshots on a 48 GB Mac and fewer on smaller ones;
  below the limit it still allocates gigabytes, for a moment that never asked
  about a picture.
- An image a tool returns (a browser screenshot) already took every later
  request of its conversation off the prefix cache: the cache-aware request
  shape carries images on user messages only (`AgentConversationBuilder`), so
  in the Day Thread one screenshot sent every later moment down the uncached
  route.

## Decision

1. **Where pictures come in.** The Today composer takes them by paste, by a
   drop anywhere on Today and by a picture button; the Jarvis Panel by paste
   into its field (a key equivalent the panel sees before the field,
   `GlassPanel.interceptKeyEquivalent`) and by a drop anywhere on the panel
   (`GlassPanel.Drop`). The agent chat's Image Gesture readers, Image Ingest
   (its types, 10 MB, eight a message) and Image Input Availability verdict
   and remedy apply unchanged. Capture stays words only: Add task and the
   panel's + are not offered with a picture in the field.
2. **What it asks.** A picture goes into the Day Thread with the owner's
   message. One sent without words asks "What in this picture matters for my
   day?": the field's placeholder shows the question before it is sent, and
   the transcript after.
3. **A picture is seen in its turn.** The model sees a picture — the owner's
   or a tool's — from the thread's latest user message on: the turn it
   arrives in, with every tool round that turn runs. Before that message it
   reads a line in the picture's place: "[The owner showed you a picture
   here: school-letter.png. You no longer see it; ask to see it again if you
   need it.]" One pure function, `DayThreadPictures.llmMessages`, renders the
   thread for the day agent's own turns (its `convertToLlm`) and for every
   moment.
4. **The thread keeps the pictures.** They are stored with their message, as
   the agent chat stores them, so Today's Chat shows them after a relaunch.
   Today has its own Composer Draft, so Quick Look pages through the Day
   Thread's pictures and Edit & Resend lands in Today's composer; a turn
   stopped before it reached the thread gives its words and pictures back to
   it.
5. **Compaction** marks a summarized message's images ("[1 image
   attached]"), so the summary keeps that one was shown.
6. **The trace** records `thread.picture`: how many, from where (Today or the
   panel), and whether words came with them.

## Considered options

- **Keep pictures in sight, as the agent chat does.** Every later moment pays
  for them, and a cold re-prefill feeds them all to the tower at once.
- **Keep only the latest picture, or the latest few.** When a newer one
  arrives, an older one's render changes and everything after it
  re-prefills: for a picture shown in the morning, most of the day's thread.
- **Cap the pictures a day takes.** An arbitrary refusal, and a cap high
  enough to be useful still crosses the tower's cliff.
- **Show the picture in a side request outside the thread.** The thread
  would hold the owner's words and Jarvis's reply, and the next request
  diverges at the same message: the same cost, with a turn whose tool calls
  the thread doesn't hold.

## Consequences

- After the turn, Jarvis knows a picture only through what he said about it
  and did with it. A later question about something he didn't read out
  ("what else does it say?") needs the picture shown again; the line tells
  him to ask.
- The request after a picture's turn re-reads that turn once. Its render
  changed, so the cache serves the thread up to the picture's message — the
  boundary the picture turn's leaf extended, kept on SSD as a Chain-Prefix
  Restore point, since the picture turn took the leaf by Leaf Handoff — and
  prefills the rest. Every later request restores past it, as text. With the
  SSD tier off, that restore falls back to the last checkpoint in RAM before
  the picture.
- A browser screenshot in the Day Thread no longer takes the rest of the day
  off the prefix cache: from the next request it is a line, and the request
  is cache-aware again.
- The day's thread file holds the pictures' bytes (base64 in its JSON, up to
  10 MB each).

## As built (2026-10-10)

- `Features/Companion/Thread/DayThreadPictures.swift`: the rule and the line;
  `DayThread.send(_:images:from:)` and `DayThread.question`.
- `Features/Companion/Today/TodayComposer.swift`: the agent composer's
  image-aware text view, the picture strip and button, the image notices;
  `TodayView` hosts Quick Look and the page's drop. `Features/Agent/Views/
  ComposerImageSupport.swift` holds what both composers share.
- `Features/Companion/Delivery/JarvisPanel.swift`: pictures waiting above the
  field, the exchange's thumbnails, `JarvisPanelController.take`;
  `Platform/GlassPanel.swift`: `interceptKeyEquivalent` and `Drop`.
- `DayThreadPicturesTests`: the rule's decision table; a Day Thread over the
  in-memory arbiter, whose owner turn hands the model the picture and whose
  next turn and moment read the line; the panel's rules. The Today and panel
  galleries render pictures waiting and asked.
- The loaded-model gate is Step P of the prefix-cache e2e
  (`TESSERACT_E2E_ONLY=day-thread-pictures scripts/dev.sh prefix-cache-e2e
  --bench-model-id <model>`). On an SSD-backed vision load it sends a Day
  Thread rendered through `DayThreadPictures`: a text turn, the picture's
  turn, the request after it and the one after that, greedy, 32 tokens a
  reply. Debug builds on an Apple M3 Max with 48 GB:

  | Tokens: prompt / restored | Qwen3.5-4B PARO | Qwen3.8-27B 4-bit, DFlash2 |
  |---|---|---|
  | The thread (text) | 206 / 0 | 244 / 0 |
  | The picture's turn | 355 / 204 | 398 / 280 |
  | The request after it | 392 / 204, 1 hydration | 439 / 280, 1 hydration |
  | The request after that | 473 / 390 | 524 / 475 |

  The request after the picture's turn restores exactly as far as the
  picture's turn did — the thread before the picture, from SSD — and
  re-reads only the picture's turn, as text (188 and 159 tokens). The one
  after restores the whole request before it, less the generation prompt a
  re-rendered leaf leaves out (2 tokens on the 4B; none on the 27B, whose
  leaf was captured live with its reply). On the 27B every turn decoded with
  the draft, the picture's included.
