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
  Each slot is cued once, as soon as the owner can see it; never while away,
  in quiet hours (but the morning's end of them is over once the owner sat
  down to start the day), a call, a game or a meeting with other people, nor
  over a panel the owner hasn't closed (it waits a tick, and a card that
  takes the panel on the same tick goes first). What came due meanwhile is
  cued late, once they can see it — a start while its slot still runs, an
  end that day — and says so ("Still time for", "How did it go?"): a cue
  first lived only ten minutes past its moment, so a slot that started while
  the owner was away, in a meeting or behind another panel slid unseen. A
  cue on the panel when the app quits is cued again after the relaunch. A
  slot whose reminder rings at the same minute is left to Reminders. Closing
  it changes nothing.
- **A started step checks in when its time is up.** A step the owner started
  (Start on a cue, or Start now on Today, which needs no cue of its own) is
  put back on the panel at its end while its task is open: Done, 15 more min
  (fifteen minutes from now, or from its end if that is still ahead; it checks
  in again), Tomorrow. While it runs, no other cue interrupts it: what comes
  due meanwhile waits (the check-in names a step already under way) and is
  cued once the owner is free; of two check-ins, the step that ended last
  comes first.
- **The panel's buttons are drawn by hand** (capsules, the main one in the
  accent): the panel is never key, and a system prominent button turns gray
  in a window that isn't.
- **A card on the panel says it all.** Every Morning Plan and Evening Wrap-up
  that reached the panel was dismissed: the panel said "Open Today to see it
  all". The Morning Plan now lists the steps ahead (its tasks among the day's
  events, the must-do starred) and its tips, with Looks Good; the Evening
  Wrap-up lists what got done and each leftover with Tomorrow, Later and Let
  go (Jarvis's pick in the accent) or Do What Jarvis Suggests. Looks Good and
  Good Night take the card in (`card.reaction` "kept"): it leaves the panel
  and stays in Today, unlike the close button; a kept card never takes the
  panel again (Jarvis's version of a kept plan updates in Today) and its
  items no longer light the glyph. A cue a card takes the panel from is held
  by the engine, which knows the cue on the panel, and comes back by the
  cue's own rules once the panel is free, while its slot still runs. The panel
  shows the card as the day has it now, so an item handled there leaves it
  (it used to stay), and it is as tall as what it says (it was always 560 pt).
- **The plan meets the first sit-down on the panel, quiet hours or not.**
  Since plans were made ahead (5 October on), the owner sat down at 06:12,
  07:55, 07:58 and 07:59 — every time inside the default quiet hours, which
  end at 08:00 — so the plan never reached the panel and waited unseen in
  Today. The day's first sit-down is the owner starting their day, not
  Jarvis reaching out at night: the Morning Plan, made ahead or made then,
  takes the panel at it (a game or a call still keeps it in Today).
- **"Nothing needs you" is no news.** 39 of the 43 Breakpoint cards had
  nothing for the owner, yet each became Jarvis's word on the Now Card for an
  hour, pushing the Morning Plan's line and tips off it after the first
  break. Jarvis's word is now the latest card with something to say; a
  Breakpoint speaks only when something needs the owner (or while Jarvis is
  judging what came in). The Breakpoint is titled "While you were away", so
  the greeting is said once, by Jarvis's line.
- **Time left can be seen.** During a meeting or a task's slot the Now Card
  said "Until 15:00": a clock reading, which a time-blind owner has to turn
  into "how long" each time. It now leads with the time left ("40 min left,
  until 15:00 · then Design review at 15:00") over a short accent bar that
  drains as the minutes go (`NowCard.span`).
- **The evening closes the day.** The owner's most frequent manual fix was
  taking planned tasks off the plan by hand — five of six times between 23:52
  and 00:24 — because at night the Now Card still led with a slid task
  ("Slid past 11:10", Tomorrow, Done). In the evening window the card now
  shows what is still ahead tonight, or else counts what got done and names
  what is still open ("1 of 3 done today." · "Still open: …"), with the
  Evening Wrap-up as the way to settle it; once wrapped up, it looks at
  tomorrow.
- **A plan cut short by a quit runs again, once.** On 8 October the plan
  was made ahead at 05:25 and the app was killed three minutes later; a
  relaunch cleared the moment in flight, and since the plan's time was set
  when its code card went up, nothing ever finished it: the day had no
  plan, and so nothing to cue. A relaunch now records the moment it cut
  short (`DayState.relaunched()`), and when the Companion comes on the
  engine runs a Morning Plan again with the trigger `resumed` — once, not
  after the owner closed the card, not in the evening or the small hours. Every other moment
  already runs again on its own trigger.
- **The wind-down.** The owner was at the Mac past midnight most nights,
  with mornings that start at 07:15 or 07:30, and nothing in the Companion
  spoke to sleep (Health's own bedtime banners are held as an app's news).
  As quiet hours begin with the owner still at the Mac, one banner a night
  says when tomorrow starts and how far off that is ("Tomorrow starts with
  All Hands at 07:30 — 8 h 30 min from now."). Code, no model; for quiet
  hours that start at night, within their first hour and while they hold,
  once a night (across the 04:00 rollover too); never in a game or a call; a
  setting beside quiet hours turns it off (`night.wind-down` in the trace).
- **A bot in a chat app is an app's news.** Slack counts as people, so
  every Jira comment relayed through it went to Triage: 7 of the 18 Slack
  banners in the ledger were the Jira bot, and on 7 October three Triage
  runs judged nothing else. A messaging app's banner whose sender line, or
  the speaker before a channel message's colon, is a known integration
  (Jira, GitHub, CI, calendars, Notion, …) is now an app's news: it waits
  for the next Breakpoint and costs no model call. Paging tools stay people,
  and the title is never read (in Slack it is the workspace).
- **The night's draft of tomorrow reaches the morning.** The Night
  Reflection writes a first draft of tomorrow (on 7 October: send the request
  a colleague asked for early, documents ready for both ID checks, the free
  mid-morning for the Companion work), but only its carry-over note opened
  the next day; the draft lived on a card no one saw by morning. The draft
  now rides into the next Day Opening as "Last night's first draft of
  today", where the Morning Plan reads it.
- **A time to leave.** The owner's Profile says an in-person class means
  leaving 30 minutes early, and Jarvis wrote "Leave by 12:30" as a plan tip,
  but the only alarm was the event nudge ten minutes before the start. The
  Morning Plan now sees the events still ahead with short ids and places,
  and may answer `leave: [{event, at}]` for one in person; code keeps a
  departure only for a listed event, before it starts, at most three hours
  ahead and still to come. Each becomes a Nudge the OS keeps ("Time to leave
  for Class", `nudge.leave.*`), the Now Card says "Leave at 12:30" as the
  event comes up and "Time to leave" when it is time, over any task still
  running; the way there is busy time, so free time and "Do it at" offers end
  at the departure; and the Day Line shows it under the event. Jarvis's own
  re-plan sets the day's departures, none included.
- **The time left of a started step is in the menu bar.** The Now Card's
  draining bar helps only while Today is open, and during a step the owner
  is in another app. While a step the owner started runs (its task still
  open), the status item shows its time left beside the glyph ("25m",
  "1h 5m"), refreshed on its own clock, with the step in the tooltip; it
  goes back to the glyph alone when the step ends or is done
  (`DayEngine.focus`, pushed through `CompanionPresence`).
- **A promise in a message becomes a task with one click.** On 7 October a
  Breakpoint noted a colleague "still waiting on that request from you" and
  the Night Reflection's draft said to send it early, but nothing made it a
  reminder. The Night Reflection may now answer `tasks` (at most three, most
  nights none): code drops any already in Reminders, whatever its date,
  fixes the due day (the reflection's tomorrow, or the Inbox), and keeps them
  under "Jarvis noticed" that night and the next day
  (`DayState.taskProposals`). Add makes the reminder through the Agenda (with
  its undo), unless the owner wrote it down meanwhile — then the question
  leaves Today and Add makes no twin; No lets it go (`task.proposed`,
  `task.decided`).
- **A week's focus.** Goals span weeks; the Companion saw one day at a
  time. On the week's last day the Evening Wrap-up also looks back on the
  week (the Agenda's snapshot now holds the week's completed reminders, read
  in the same EventKit query as today's) and may answer `week` and `focus`:
  next week's one thing, in a few words. Code keeps the focus a week
  (`DayState.weekFocus`); it opens each day's thread, the Morning Plan is
  asked to let the must-do serve it, and Today shows it beside the date.
  The look-back also counts the week's must-dos ("The must-do got done on 4
  of the 6 days it was set"): the engine notes when the day's must-do is seen
  done and keeps each day's outcome a week (`DayState.mustDoDays`).
- **An online meeting reads as its service.** A Zoom event's location is
  its link with the password; it showed under the event on the Day Line, in
  its nudge, in the agenda tool and in the plan's list of events to leave
  for. `AgendaEvent.place` reads it as "Zoom" (text around the link stays:
  "Room 4 · Microsoft Teams"; every link goes, and a passcode written beside
  one), which also tells the plan there is nothing to travel to; the agent's
  listing keeps the link without its query.
- **The plan sees the likeliest tasks first.** The Day Opening and the
  Morning Plan listed up to 25 undated reminders in the store's order, most
  of them from collection lists (films, books, places) no day plans. They now
  list 15: the Inbox first, then lists that hold dated work, then the rest,
  with what is left summed by list ("…and 12 more, in Movies, Books").
- **The capture hotkey is taught where it pays.** In ten days it was never
  used; tasks came through Today's composer and the agent's tools. An empty
  Inbox now says "Tap Right ⌥ in any app to write a thought down, or hold it
  to say one" (the configured key, by name).
- **Once the must-do is done, the rest is a bonus.** Finishing the day's one
  thing that mattered most changed nothing on Today: the next free-time card
  offered the next task at the same weight. Now it says "The must-do is done;
  this one's a bonus", and the evening counts it ("2 of 4 done today, the
  must-do among them") — the pressure comes off once the main thing is in.
- **The trace measures it**: `cue.presented` (start or end, how late, the
  slot's length, the must-do) and `cue.reaction` (the choice, the time to
  react).
- `JarvisPanelGalleryTests` renders the panel's content over fixture cards,
  each at the height the panel fits to it; `TEST_RUNNER_PANEL_GALLERY_DIR`
  writes the renders as PNGs.
