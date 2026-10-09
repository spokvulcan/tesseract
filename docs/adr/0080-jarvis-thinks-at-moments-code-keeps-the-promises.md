# ADR-0080: Jarvis thinks at moments; code keeps the promises

- Status: Accepted (built on `feat/companion-v2`, spec #600)
- Date: 2026-09-30
- Supersedes: ADR-0035 (living memory), ADR-0040 (the entity loop), ADR-0043
  (the Wake Evaluator), ADR-0045 (Conversation Memory), ADR-0046 (the Event
  Fold and Mission Control), ADR-0051 (the fold reducer), ADR-0052 (every chat
  joins the fold)
- Relates to: ADR-0060 (reasoning effort as a template kwarg), ADR-0073 (test
  runs never reach the owner's data), ADR-0047 (the non-sandboxed agent),
  ADR-0031 (the trace vocabulary pattern)

## Context

The first Companion was an entity in a harness: a standing Mission Control
conversation that every notification, app switch and wake folded into, with
self-authored Standing Instructions, a nightly Digest, and a living memory
consolidated whenever the Mac went idle. Two days of July dogfooding measured
what it did:

- About 185 full-model turns in two days, most of them triggered by other
  apps' notifications and app switches; roughly 70% produced nothing for the
  owner, and most of the rest repeated banners they had already seen.
- Its prompts averaged about 40k tokens because the instructions were re-sent
  every turn, and the standing conversation was rewritten four times a day.
- Memory consolidation started after three minutes without input (reading,
  video, calls) and ran generations back to back with no cooldown and no
  thermal or power awareness. The Mac ran hot.
- Memory injection placed a block in front of almost every chat message. An
  A/B on the same model: a simple reasoning question answered correctly 4 of
  5 times without the block, 0 of 4 times with it.
- Promises broke silently: "remind me tomorrow" booked a wake that never fired
  with the Companion off; "now" was stamped into the system prompt at launch;
  a generation that failed mid-turn was recorded as a completed turn.
- No screen showed the day, the calendar was read-only, and tasks lived in a
  markdown file and tables nothing read.

The owner switched it off after three days.

## Decision

Jarvis thinks at moments, when there is something to judge. Every decision
becomes an artifact — a reminder, a calendar event, a scheduled nudge, a rule,
a Profile fact or a card — and code and the OS keep it without the model.

1. **Apple Reminders and Calendar are the single source of truth.** One Agenda
   port over EventKit (with an in-memory store for the test host and every
   test), five agenda tools in every conversation, and a capture hotkey with a
   deterministic parser. "Remind me" always becomes a real reminder; a timed
   one carries its own alarm, so Reminders delivers it on every device.
   `tasks.md` and the task skill are gone.
2. **The Day Engine is the one decider.** A pure function in the gather →
   decide → perform shape: signals (clock, presence, meetings ending,
   notifications, apps coming forward, coding agents, the agenda, moment
   results, card actions, power) and a snapshot in; the day's state and an
   ordered list of effects out. The Companion runtime performs them.
3. **Moments are single model calls with a JSON card.** Morning Plan,
   Breakpoint, Triage, Evening Wrap-up, Night Reflection. Each reply is
   validated against the facts it was shown; an invalid or failed reply gets
   one retry, then a deterministic card built from the snapshot. No moment
   runs without new input, and none waits on the model to deliver something
   urgent: a Breakpoint's card goes up at once, built by code, and the model
   only refines it when there are notifications to judge.
4. **The Day Thread replaces Mission Control.** One append-only conversation
   per day, on its own agent with the same system prompt and tools as every
   chat. It opens with the Day Opening (the Profile, the Areas, today's agenda,
   last night's carry-over note); moments append a request and a card; the
   owner's Today chat appends to it too. A new day starts a new thread; within
   a day compaction runs only past a ceiling (80k tokens by default).
5. **The prefix-cache contract.** The system prompt is static — no time, no
   per-conversation content — and the tool list is identical everywhere (tool
   audiences are gone), so every chat and every moment shares one cached
   system-and-tools root, and the Day Thread prefills only its delta. The time
   rides every user message as a stored Now Tag.
6. **Memory is small and the owner's.** A Profile of approved facts (`remember`
   is an explicit approval; Night Reflection proposes, the owner decides), a
   `recall` tool over the Profile and past conversations, and nothing injected
   into ordinary chats. The old store is deleted by a one-time clean start.
7. **Delivery is code's decision.** The Delivery Ladder picks glyph, the Jarvis
   panel, a banner or voice from importance, presence, the app in front and
   quiet hours. The Jarvis panel is a Siri-style Liquid Glass panel over any
   app that never steals typing.
8. **Notifications are triaged in batches, never one by one.** A seen ledger
   keeps what the owner already saw away from the model; owner rules apply
   first; Triage runs at most every ten minutes while the owner works.
9. **Coding agents report in** through Claude Code's hooks to a
   localhost-only route; the "speak once after two minutes out of the terminal"
   rule is code, not a model call.
10. **The governor defers, it does not budget.** No daily GPU budget;
    Triage and the Night Reflection wait while the Mac is hot or low on
    battery, and the Night Reflection runs only on power with a nominal
    thermal state. The selected model is never switched.
11. **The Companion Trace** records every decision, card, reaction and agenda
    change in a closed vocabulary, one JSONL file per day, each record stamped
    with its Day Thread; model calls carry tokens (with the cache's share),
    latency, the model, and the thermal and power state.

## Departures from the spec

- **One reasoning effort for every moment.** The spec asked for low effort
  at Breakpoints and Triage and medium for the plan and wrap-up. On the current
  checkpoints effort is written into the first system block (ADR-0060), so a
  per-moment effort would give each moment its own prefix and re-read the
  whole Day Thread every time — the opposite of decision 5. Every moment runs
  at the owner's configured effort, and each is bounded by its own output cap
  instead; a reply that hits the cap falls back.
- **Focus modes are not an input to the ladder.** macOS exposes no public,
  unentitled way for an app to read the current Focus. Quiet hours and the app
  in front (calls, presentations) stand in for it.
- **The day's plan is Tesseract's, not Reminders'.** A task's slot in the day
  (start and length) lives in the Day Engine's state, because Reminders has no
  duration. The reminder keeps its own date; checking a task off in Today
  completes it in Reminders.

## Consequences

- Almost all of the old Companion is deleted: Mission Control, the Event Fold,
  the Digest and both briefings, Report-Back, wakes and the Wake Evaluator,
  the fold reducer, Standing Instructions, Companion Sleep, Memory Sleep and
  idle consolidation, memory injection and Conversation Memory, the memory
  lifecycle and backfill, the Memory window, tracking and `log_feedback`.
- Kept as building blocks: idle and presence detection, the Notification
  Center watcher, the banner notifier, the voice session and overlay, the
  menu-bar glyph, the rotating JSONL writer, the embedder, the Turn Replay
  Breaker.
- The Companion's logic is testable without a model and without EventKit:
  decision tables over the Day Engine, the in-memory Agenda behind the tools,
  and card parsing on canned replies.
- Loaded-model quality (does the model read a 60k-token Day Thread well?) is
  measured through the trace, not assumed.

## As built

- `Features/Companion/Agenda/` — the port, EventKit and in-memory stores, the
  Agenda facade (snapshot, confirmations, undo), Areas, the tools.
- `Features/Companion/Capture/` — the parser, the capture door, the hotkey panel.
- `Features/Companion/Engine/` — `DayEngine` (+Moments, +Breakpoints),
  `DayTypes`, `Delivery` (ladder and governor), `Nudges`.
- `Features/Companion/Moments/` — moment kinds, cards, prompts, parsing,
  fallbacks, the Breakpoint and Triage moments.
- `Features/Companion/Thread/DayThread.swift` — the Day Thread and its store.
- `Features/Companion/Loop/CompanionRuntime.swift` — the loop and the day's
  saved state.
- `Features/Companion/Today/` — the Today page (the Timeline layout).
- `Features/Companion/Delivery/` — the Jarvis panel, banners and nudges, the
  glyph state.
- `Features/Companion/Perception/` — the Notification Center watcher, the seen
  ledger, owner rules and their tool.
- `Features/Companion/CodingAgents/` — agent signals and the Claude Code hooks.
- `Features/Companion/Profile/`, `Recall/` — the Profile, proposals, the
  memory tools, the recall index and embedder.
- `Features/Companion/Trace/` — the Companion Trace.
- `Features/Companion/Migration/` — the one-time clean start.
- `Platform/GlassPanel.swift` — the floating glass panel both panels use.

## Amendments (2026-10-01, after the first full day)

The first day's Companion Trace showed one Morning Plan ready 2 min 42 s after
the sit-down, three cards (all dismissed, none acted on), nine Triages that
raised nothing — two of them about Game Mode and a game — and a Night Reflection
skipped because the Mac was on battery (at 85%). What changed:

- **Only people reach Triage.** Code sorts other apps' banners on arrival
  (`NotificationSources`): a person (messaging apps, or a messaging site in a
  browser), an app's own news, or noise — the system's banners such as Game Mode,
  and games, known by category, Game Mode support, store folder or publisher.
  Noise is never shown; an app's news waits for the next Breakpoint; owner rules
  still come first. Triage never runs while a game is in front, a game in front
  gets no panel and no voice (like a call), and Triage's output cap is 1,024
  tokens.
- **The Morning Plan never makes the owner wait.** A card built by code goes up
  at once and Jarvis's version replaces it in place; if he fails, the code card
  stands. When the Mac is awake and on power in the morning window before the
  first sit-down, the plan is made ahead and comes forward at the sit-down.
- **The Night Reflection runs on a battery at least half full** with a nominal
  thermal state, as well as on power (fair thermal state allowed).
- **"Nothing needs you" stays in Today**: a Breakpoint with nothing for the owner
  never takes the panel. The lock screen is never "where you were".
- **The capture hotkey is one key**: Right ⌥ alone — tap to type, hold to speak;
  another key pressed with it cancels (`ModifierKeyDetector`).
- **The Day Thread compacts past 64k tokens** (was 80k): about 4 GB of KV at the
  ceiling on the 27B checkpoints instead of 5.
- **`delete_event`**: Jarvis deletes a block or slot the owner keeps for
  themselves when asked (never a meeting with other people), with undo. The
  agenda tools are six.
- **The trace measures what happened.** A moment's `prefillSeconds` is the
  server's whole prompt time (lookup, restore and prefill; it used to be only the
  last residual chunk), `waitSeconds` is the wait for the model, and every nudge
  macOS delivered is recorded once, read back from Notification Center on the
  tick. A reply's model label is the model selected at that turn.
- Moments take their turn at the LLM Gate (ADR-0081) and compact inside it.

## Amendments (2026-10-04, Today as steps)

The owner found Today hard to read: one column of sections at the same weight
(the day, Anytime, Capture, Waiting on You, the Inbox), a long paragraph of
Jarvis's in secondary text at the top, empty sections taking room, and
nothing that said what to do now. What changed:

- **The Now Card tops Today**: the step the day is on, with one-click offers
  that move it on (Done, Start now, "Do it at 16:30", Tomorrow, Plan my day,
  Wrap up the day). Code builds it from the Timeline (`NowCardBuilder`, a
  decision table pinned by `NowCardTests`), so it is there the moment Today
  opens. Jarvis's latest card line rides on it while it is fresh, and Waiting
  on You and the Evening Wrap-up's leftovers moved onto it, so each thing is
  said once.
- **The day is steps on one Day Line**: a table (time, task, Area, length) on
  a wide page and a list shaped for a phone below 640 pt. The phone list is a
  starting point for the Companion's phone UX (ADR-0084, release 4); the
  Companion itself stays Mac-only.
- **Offers, not searches**: an Inbox item offers a concrete slot ("At 16:30",
  clear of the Now Card's offer and of the items above it) or Tomorrow, in
  place of "Find a time", which did nothing once the day was full.
- **One field, the Today composer**: the agent composer simplified (its glass,
  notice slot and action row, design-language §1) to Add task (⌘↩, Capture
  with no model), a hold-to-talk mic and send. Capture's own box is gone. The
  latest agenda change confirms itself in the notice slot, with an undo.
- `TodayGalleryTests` renders Today over fixture days at three widths;
  `TEST_RUNNER_TODAY_GALLERY_DIR` writes the renders as PNGs, for judging the
  page by eye.

## Amendments (2026-10-05, tomorrow on the Day Line)

The owner found the days disconnected: at night the steps stopped at the Now
line and tomorrow was one sentence on the Now Card. What changed:

- **The Day Line runs on into tomorrow.** After today's steps and its Anytime
  tasks, a day break ("Tomorrow · Monday, 5 October") and tomorrow's
  all-day events, timed events, tasks due at a time and Anytime tasks, on the
  same line (`TomorrowTimeline`). Tomorrow is not planned yet, so it shows no
  free time. A task moved to tomorrow lands there in sight; a task due
  tomorrow and done early stays there, out of today's count.
- **A done day says how tomorrow starts** ("Next: Work, tomorrow at
  09:00.") instead of listing its first three events.
- **Today is the owner's day.** `DayFacts` and the Agenda's snapshot follow
  the Day Key: until 04:00 the page and the moments keep the day that is
  ending (its steps, its done count, the snapshot's events from that day's
  start), and tomorrow is the next date. Before, the page jumped to the new
  date at midnight while the Day Thread and the day's cards stayed on the
  old one, and a wrap-up after midnight read the new date's tasks as its
  leftovers.

## Amendments (2026-10-09, the plan keeps time)

Nine days of the Companion Trace showed the Morning Plan placing two to four
tasks a day, and the owner using "Start now" and "Do it at 16:30", but a slot
lives only in the Day Engine's state (Reminders has no duration), so nothing
marked its start: a plan was kept only if the owner happened to open Today at
the right minute. That broke this ADR's own rule that every decision becomes
an artifact code keeps. Help that lands at the moment of doing works; a list
the owner must remember to read does not. What changed:

- **The Step Cue.** When a planned step's slot starts and the owner is at the
  Mac, code puts it on the Jarvis Panel (a shorter panel than a card's): the
  task, until when and what follows, with Start (the slot starts this minute),
  In 15 min (it moves on and is cued again then), Tomorrow and Done. No model.
  Each slot is cued once, within ten minutes of its start; never while away,
  in quiet hours, a call, a game or a meeting with other people, nor over a
  panel the owner hasn't closed (it waits a tick, and a card that takes the
  panel on the same tick goes first). A slot whose reminder rings at the same
  minute is left to Reminders. Closing it changes nothing.
- **The panel's buttons are drawn by hand** (capsules, the main one in the
  accent): the panel is never key, and a system prominent button turns gray
  in a window that isn't.
- **The trace measures it**: `cue.presented` (how late, the slot's length,
  the must-do) and `cue.reaction` (the choice, the time to react).
- `JarvisPanelGalleryTests` renders the panel's content over fixture cards;
  `TEST_RUNNER_PANEL_GALLERY_DIR` writes the renders as PNGs.
