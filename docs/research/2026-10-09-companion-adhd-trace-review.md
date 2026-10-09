# The Companion as a day manager for an ADHD brain — what ten days of its trace showed

**Date:** 2026-10-09 · **Scope:** the Companion as built by ADR-0080 and its
amendments, measured against its own record of ten days of use.
**Method:** the Companion Trace (`~/Library/Application Support/CompanionTrace/trace-*.jsonl`,
30 September – 9 October 2026), the saved day state (`companion/day-state.json`,
its seen ledger) and the Day Threads, read locally. Only aggregate counts and
times appear here; no names, messages or places. The principles section names
the ideas the changes lean on; it makes no claims beyond them.

---

## What the trace showed

1. **Cards were mostly noise.** 43 Breakpoint cards: 39 went to Today only,
   with nothing for the owner ("Nothing needs you right now."); 2 were acted
   on (one notification opened, one reminder done). Each quiet card became
   Jarvis's word on the Now Card for an hour, and pushed the Morning Plan's
   line and tips off it after the first break of the day.
2. **The panel said "Open Today to see it all", and was dismissed.** Every
   Morning Plan (4) and Evening Wrap-up (5) that reached the panel was
   dismissed, in 12 s to 8 min. Only 2 leftover decisions were ever made, both
   from Today.
3. **The plan was not kept.** A slot (`Placement`) lives only in the Day
   Engine's state, so nothing marked its start: no cue, no alarm. The owner's
   only frequent manual actions were placing a task (3) and taking one off the
   plan (6 — five of them between 23:52 and 00:24, tidying slid tasks off the
   Now Card before bed).
4. **The prepared plan never reached the panel.** From 5 October the plan was
   made ahead; the owner sat down at 06:12, 07:55, 07:58 and 07:59, every time
   inside the default quiet hours (until 08:00), so it waited unseen in Today.
5. **A plan cut short was never finished.** On 8 October the plan started at
   05:25 and the app was killed at 05:28; the relaunch cleared the moment in
   flight and nothing ran it again. The day had no plan.
6. **Triage judged bots.** 24 Triage runs; most were offered 1–3 banners and
   raised none. 7 of the 18 Slack banners in the ledger were a Jira bot, and on
   7 October three runs judged nothing else.
7. **Cold prefills cost minutes.** The Evening Wrap-up ran with a cold prefix
   cache 6 times out of 8, re-reading 8–27k tokens (up to 195 s); each followed
   a ~7 s model load, that is, an app relaunch.
8. **The night's draft of tomorrow was lost.** The Night Reflection writes up
   to five concrete lines for tomorrow; only its carry-over note opened the
   next day.
9. **Late nights, early mornings, no word about sleep.** The trace is active
   past midnight most nights, with commitments at 07:15–08:00; Health's own
   bedtime banners are held as an app's news.
10. **Travel was known but not kept.** The Profile says an in-person class
    means leaving 30 minutes early, and Jarvis wrote "leave by 12:30" as a tip;
    the only alarm was the event nudge ten minutes before the start.

## Principles the changes lean on

- **Help at the point of performance** (Barkley's phrase for ADHD support):
  a cue at the moment of doing beats a list the owner has to remember to read.
- **If-then plans** (Gollwitzer's implementation intentions): a plan that says
  *when* a step starts works when something marks that moment.
- **Externalised time:** time left shown, not computed from a clock reading —
  a draining bar, a countdown in the menu bar.
- **One thing at a time, less noise:** say each thing once; "nothing needs you"
  is no news; never stack interruptions.
- **No shame, credit the win:** nothing is "missed"; finishing the must-do is
  said aloud, and the rest becomes a bonus.
- **Transitions and the day's edges:** when to leave, when to wind down, when
  the day closes.

## What changed (branch `feat/companion-adhd-loop`)

| Finding | Change |
|---|---|
| 3 | **Step Cue**: a planned slot comes to the owner on the Jarvis Panel at its start (Start, In 15 min, Tomorrow, Done); a started one checks in at its end; no start interrupts it, a held start is cued once the owner is free |
| 3 | A cue that came due while the owner was away, in a meeting or behind another panel is shown late when they can see it ("Still time for", "How did it go?"), instead of lapsing after ten minutes |
| 3 | A step put off twice is offered as five minutes ("Just five minutes?"), and five minutes in the check-in asks to keep going |
| 2 | The Morning Plan and the Evening Wrap-up are whole on the panel (steps, tips, leftovers with their choices); the panel shows the live card and fits its content; Looks Good keeps the card instead of dismissing it |
| 4 | The Morning Plan meets the first sit-down on the panel, quiet hours or not; Step Cues follow after an early sit-down |
| 4 | A day that starts late (4 October: first at the Mac at 14:43) gets its plan at the first sit-down, made then, instead of only on asking |
| 1 | A quiet Breakpoint no longer takes Jarvis's word; it is titled "While you were away", so the greeting is said once |
| — | The Now Card leads with the time left of the step under way, over a draining bar; the menu bar shows it from any app, and counts down the last half hour to the next event or time to leave |
| 3 | In the evening the Now Card closes the day (what is ahead tonight, or what got done and what is still open) instead of heading it with a slid task |
| 3 | Several tasks that slid are fitted back into the day's free slots in one click ("Fit all 3 in") |
| 5 | A Morning Plan cut short by a quit runs again, once |
| 9 | The wind-down: one banner as quiet hours begin, saying when tomorrow starts |
| 9 | After a night at the Mac past midnight, the Morning Plan is asked to keep the day light |
| 9 | A done day's Now Card says how far off tomorrow's start is ("— 6 h 50 min from now"), within half a day |
| 2 | The Evening Wrap-up waits for a step the owner started, as Step Cues do |
| 6 | A bot posting through a chat app is an app's news, not a person |
| 8 | The night's draft of tomorrow opens the next day's thread |
| 10 | The plan sets a time to leave for an event in person; the OS nudges then, the Now Card and the Day Line say it |
| — | Once the must-do is done, the Now Card says the rest is a bonus; Done on a Step Cue is answered for a moment — the win, and what comes next |
| — | A week's focus: the week's last evening looks back (done by Area, must-dos kept) and names next week's one thing, shown beside Today's date |
| — | The week ends on a clean slate: that evening also asks about up to five overdue tasks, oldest first, suggesting Later for what has waited |
| — | An online meeting reads as its service ("Zoom"), never its link and password |
| — | The plan's candidates come the likeliest first; collection lists are summed |
| — | An empty Inbox teaches the capture hotkey; a Step Cue chimes softly |
| — | The menu bar menu opens on the day: what is on now or next, Open Today, Write a Thought Down… |
| — | Two code reviews of these changes, with every finding fixed and tested |

The trace measures the new behaviour: `cue.presented` / `cue.reaction`,
`card.reaction` with `kept` apart from `dismissed`, `night.wind-down`,
`nudge.scheduled` for `nudge.leave.*`.

## Recommendations not built

1. **Run Triage and the Evening Wrap-up on a lean context.** Both judge what
   their request already holds, yet read the whole Day Thread; after a relaunch
   that is a full prefill of 8–27k tokens. A short conversation (system prompt,
   Profile, the request) would share the cached root and cost seconds. This
   revisits ADR-0080 decision 4 and needs the owner's call. Not the cause of
   the cold prefills: the Day Opening is committed to the thread once, so it
   does not change across a relaunch, and the SSD prefix cache is on by
   default. Its manifest (metadata only) shows where it stops: of the 18
   snapshots on disk (4–8 October), the 4.4k-token system root and two
   branch points were read back after a relaunch, but none of the nine leaf
   snapshots written at the end of a moment's reply (3–20k tokens) ever was
   — each one's last access is its creation. So a relaunched thread restores
   the root and re-reads everything after it. Why the next request's path
   misses the leaf (a reply rendered differently from how it was generated,
   for one) is for the prefix cache's own investigation, not the Companion's.
2. **Keep someday lists out of the plan's candidates.** Partly done: the
   requests now list 15 undated reminders, the Inbox and lists with dated work
   first, the rest summed by list. Mapping Areas (or marking lists Jarvis
   never plans from) would go further.
3. **A weekly look-back.** Built as part of the Evening Wrap-up on the week's
   last day: the week's done reminders by Area, the must-dos kept, next
   week's one focus (kept for the week, steering the plan's must-do), and up
   to five overdue tasks to decide, so the week ends on a clean slate.
4. **A leaner agenda listing.** Measured and set aside: event and reminder
   ids are about 2% of the listings' text, and the threads show no tool call
   that failed on a mistyped id. Meeting links already lose their password.
5. **Make capture discoverable.** Built: an empty Inbox names the capture
   hotkey.
6. **Re-read the trace after a week** with the new events, and keep what moves
   the owner to act: `scripts/companion-trace-report.py [days]` prints, per
   day, moments and cold prefills, cards by rung, reactions (kept apart from
   dismissed), Step Cues (late ones apart) and the owner's choices — "later"
   and "startSmall" among them, the measure of whether five minutes beats a
   snooze — the wind-down, event and leave nudges, notifications by source,
   Triage, and task proposals with what became of them.

## Appendix: the queries

Tallies were made with short scripts over the JSONL: events per day; for
`card.presented` the pair (moment, rungs); for `card.reaction` the pair
(moment, action); for `moment.finished` the prompt and cached tokens, prefill
and generate seconds; for `notification.arrived` the triple (source, app,
rule); and the seen ledger's Slack entries by sender.
