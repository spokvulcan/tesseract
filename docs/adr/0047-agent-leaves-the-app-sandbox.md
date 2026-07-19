# ADR-0047: The agent leaves the App Sandbox

- Status: Accepted
- Date: 2026-07-19
- Relates to: #381 (the cutover), PRD #376 (the Notification Hub this unblocks),
  #378 (the capture mechanism, recorded below), #379/#380 (the seed and the
  evidence), #357 (the ambient-intelligence vision), #331 (the AX-grant prior
  art), map #301

## Context

The Companion's next step is a Notification Hub — Jarvis reading every
notification on the Mac and interrupting only when it is worth the owner's
time (#357). The only offline, no-private-entitlement way to read other apps'
notifications is Accessibility observation of the `NotificationCenter`
process. Verified live (#377): under the App Sandbox, cross-process
`AXUIElementCopyAttributeValue` returns `-25204 (kAXErrorAPIDisabled)` for
*any* target — Finder as much as NotificationCenter — even with
`AXIsProcessTrusted() == true`; non-sandboxed, the identical call returns
`AXError=0` with the full window tree. The sandbox forbids a process from
being an AX assistive *client* of other processes; #331's tap-class grant
never generalized to reads.

The sandbox is not free either way. An ambient agent whose whole job is to
observe the machine — notifications now, the screen next (#357's Eyes) — is
the exact workload the sandbox is built to prevent. The containment was
fighting the product.

## Decision

**The agent ships non-sandboxed.** `ENABLE_APP_SANDBOX = NO` in both build
configs; `com.apple.security.app-sandbox` dropped from both entitlements
files. Hardened Runtime stays on (notarization requires it and it is
independent of the sandbox), so the Developer-ID pipeline is unchanged.

This is legitimate because distribution is **Developer ID from the website,
not the Mac App Store** (docs/releasing.md). The App Sandbox is mandatory only
for the App Store; Developer ID permits its absence. The product's future
shape reinforces the split: the **server** — a clean, contained product — can
go to the App Store (sandboxed); the **agent**, which needs deep system
access, is the website download that never will.

The privacy promise is untouched. "Nothing leaves the Mac" is about network
egress; leaving the sandbox enables zero new egress. What changes is local
containment — precisely the reach the agent's job requires.

**Migration is models-only.** Off the sandbox, `applicationSupportDirectory`
resolves to `~/Library/Application Support` instead of the per-app container,
so everything the sandboxed build wrote is otherwise orphaned. The owner's
call: carry only the downloaded models across (a same-volume rename — instant,
no copy) and let memory, conversations, and settings start fresh, so the
un-sandboxed build boots as a clean install with models already present. The
shim (`SandboxMigration`) runs at `applicationDidFinishLaunching`, ahead of the
first model-path read; it merges per model (never clobbering a model the new
path already holds, so a pre-existing non-sandboxed `models` dir survives),
is idempotent, and never deletes from the old container.

## Considered options

- **Stay sandboxed, find another capture path** — rejected: the `usernoted`
  notification store is TCC-locked (blocked even for an unsandboxed CLI), and
  `getDeliveredNotifications` is own-app only. No sandboxed path to other apps'
  notifications exists.
- **A non-sandboxed XPC helper beside a sandboxed app** — rejected for v1: it
  reintroduces the sandbox's constraints on the main app for no gain once the
  product is Developer-ID anyway, and complicates signing for a split the
  future server/agent architecture makes cleanly instead.
- **Migrate everything, or nothing** — rejected in favor of models-only: a
  full migration carries stale state the owner wants to shed; a nothing
  migration re-downloads ~160 GB. Models-only is the one expensive-to-refetch,
  cheap-to-move slice.

## Consequences

- Cross-process AX reads work, unblocking the Notification Hub (#376) and, on
  the same posture, later screen observation (#357's Eyes).
- The Mac App Store is foreclosed **for the agent** — it would require the
  sandbox back. The MAS path (`archive.sh SIGNING_METHOD=app-store`) now
  belongs to the future server, not the agent.
- TCC grants (Accessibility, microphone, calendar) re-consent once at the
  cutover, as expected for any signing-identity/entitlement change.
- The old sandbox container is left intact (356 MB: memory, conversations,
  settings) — recoverable if the fresh-start decision is ever reversed, and
  the owner's to delete when ready. The three overlapping TTS models the merge
  kept at the destination remain as harmless container dupes until then.
- The now-inert sandbox-scoped entitlements (calendars, audio-input, network)
  are left in place; TCC still governs calendar and microphone by prompt, and
  they cause no signing or launch issue non-sandboxed.

## The capture mechanism (#378)

The reach this posture buys, made concrete. The Notification Hub reads other
apps' notifications by **Accessibility observation of the `NotificationCenter`
process** (`com.apple.notificationcenterui`) — the only offline,
no-private-entitlement path that returns another app's source/title/body, all
verified live on macOS 26.5 (research spike, 2026-07-18). An `AXObserver` on the
process fires on window/element creation; the watcher walks the new subtree for
the banner group (subrole `AXNotificationCenterBanner`), which carries an
`AXIdentifier` UUID and an `AXDescription` of "app, title, subtitle, body" plus
`AXStaticText` children. The Event id derives from that UUID, so a re-observed
banner collapses at admission; Tesseract's own banners are excluded on display
name, because the tree exposes **no bundle ids** — only localized display names.
The AX tree is unofficial UI, not API, so the reader is defensive (structured
children first, the flattened description as fallback) and every OS bump owes a
re-probe.

**The runner-up is the `usernoted` store** (`~/Library/Group
Containers/group.com.apple.usernoted/db2/db`). It records every delivery
regardless of how it was presented — the best coverage on paper — but it is
TCC-locked: unreadable even from an unsandboxed process without Full Disk
Access, behind a protected container with a private, historically-changing
schema. It is the thing to re-test each macOS release, not a v1 bet.

**The structural blind spots are the load-bearing consequence.** The live-banner
path is blind to silent deliveries, Focus/DND-suppressed notifications, and
anything that arrives while the screen is locked — none of them banner, so none
are seen. This makes the full-hub end vision (#357: the owner turns on Do Not
Disturb and Jarvis becomes the sole interface) **mechanism-gated, not merely
evidence-gated**: with DND on there are no banners for AX to observe — exactly
the column where this mechanism goes dark. Reaching the full hub needs a
presentation-independent source (the store, a supported successor API), not a
policy toggle on top of v1. So the capture source sits behind an interface that
can be swapped, and every escalation and reaction lands on the flight recorder
(#380), so the eventual flip is earned on a real record with a real successor
mechanism to stand on — not AX with the banners turned off.
