#!/usr/bin/env python3
"""Summarise the Companion Trace: what Jarvis did each day and what the owner
did with it.

Usage: scripts/companion-trace-report.py [days]   (default: the last 7)

Reads ~/Library/Application Support/CompanionTrace/trace-*.jsonl (one file per
day, ADR-0080) and prints, per day: moments (and how many ran with a cold
prefix cache), cards by delivery rung, card reactions by action ("kept" apart
from "dismissed"), Step Cues by phase and the owner's choices, the wind-down,
nudges (event and leave), notifications by source, Triage, and the tasks the
Night Reflection proposed and what the owner decided. A Step Cue shown ten
minutes or more after its moment counts as late. The night before is how long
the Mac was left before the day's first sit-down. Counts only — no titles,
messages or names.
"""

import collections
import glob
import json
import os
import sys

TRACE = os.path.expanduser("~/Library/Application Support/CompanionTrace")


def records(path):
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "event" in record:
                yield record


def summarise(path):
    moments = collections.Counter()
    cold = collections.Counter()
    prefill = 0.0
    cards = collections.Counter()
    reactions = collections.Counter()
    cues = collections.Counter()
    choices = collections.Counter()
    nudges = collections.Counter()
    sources = collections.Counter()
    triage = [0, 0]
    wind_down = 0
    tasks = collections.Counter()
    night = None
    for record in records(path):
        event = record["event"]
        fields = record.get("fields", {})
        if event == "moment.finished":
            kind = fields.get("moment", "?")
            moments[kind] += 1
            prefill += fields.get("prefillSeconds", 0)
            if fields.get("cachedTokens", 0) < 0.5 * fields.get("promptTokens", 1):
                cold[kind] += 1
        elif event == "moment.failed":
            moments[fields.get("moment", "?") + " (failed)"] += 1
        elif event == "card.presented":
            cards["%s→%s" % (fields.get("moment", "?"), fields.get("rungs", "?"))] += 1
        elif event == "card.reaction":
            reactions[fields.get("action", "?")] += 1
        elif event == "cue.presented":
            phase = fields.get("phase", "start")
            cues[phase] += 1
            if fields.get("late", 0) >= 600:
                cues[phase + " late"] += 1
        elif event == "cue.reaction":
            choices[fields.get("action", "?")] += 1
        elif event == "nudge.scheduled":
            kind = "leave" if str(fields.get("id", "")).startswith("nudge.leave.") else "event"
            nudges[kind + " scheduled"] += 1
        elif event == "nudge.fired":
            kind = "leave" if str(fields.get("id", "")).startswith("nudge.leave.") else "event"
            nudges[kind + " fired"] += 1
        elif event == "notification.arrived":
            sources[fields.get("source") or "unsorted"] += 1
        elif event == "notification.triaged":
            triage[0] += 1
            triage[1] += fields.get("raised", 0)
        elif event == "night.wind-down":
            wind_down += 1
        elif event == "night.ended":
            minutes = fields.get("minutesAway", 0)
            night = "%d h %02d min away%s" % (
                minutes // 60, minutes % 60, ", up past midnight" if fields.get("upLate") else "")
        elif event == "task.proposed":
            tasks["proposed"] += fields.get("count", 0)
        elif event == "task.decided":
            if fields.get("added"):
                tasks["added"] += 1
            elif fields.get("existed"):
                tasks["already there"] += 1
            else:
                tasks["let go"] += 1
    return {
        "moments": dict(moments),
        "cold prefills": dict(cold),
        "prefill seconds": round(prefill),
        "cards": dict(cards),
        "reactions": dict(reactions),
        "step cues": dict(cues),
        "cue choices": dict(choices),
        "nudges": dict(nudges),
        "notifications": dict(sources),
        "triage runs / raised": "%d / %d" % tuple(triage),
        "night before": night,
        "wind-down": wind_down,
        "task proposals": dict(tasks),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    paths = sorted(glob.glob(os.path.join(TRACE, "trace-*.jsonl")))[-days:]
    if not paths:
        print("No Companion Trace in %s" % TRACE)
        return
    for path in paths:
        print("== " + os.path.basename(path)[len("trace-") : -len(".jsonl")])
        for name, value in summarise(path).items():
            if value:
                print("   %-22s %s" % (name, value))


if __name__ == "__main__":
    main()
