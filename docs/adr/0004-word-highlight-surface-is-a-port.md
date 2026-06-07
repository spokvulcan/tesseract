---
status: accepted
---

# The spoken-word highlight surface is a port, so the segment-boundary switch is testable

This records a design decision for the **Word Timeline** / **Segment Playback**
deepening (see `CONTEXT.md` → **Speech word timeline**). The modules named below are
agreed but **not yet implemented** in the tree — this ADR exists so the seam is not
re-litigated once it lands.

The TTS notch overlay — until now a concrete `TTSNotchPanelController?` that
`SpeechCoordinator` held and called directly — becomes a **port**,
`WordHighlightSurface`. It is the fourth speech seam in the ADR-0003 family and the same
**`@MainActor` sibling** shape as `AudioPlayback`: class-bound, main-actor-isolated, and
called **synchronously**. The **Segment Playback** loop drives the per-segment calls
(`switchText`, `updateTotalDuration`, `markSegmentComplete`); `SpeechCoordinator` makes
the session-level ones (`show`, `markGenerationComplete`, `dismiss`). It is deliberately
not an actor, for the same reason `AudioPlayback` is not — the calls are already
main-actor-bound, so an actor would add cross-actor hops on a hot path for no isolation
gain.

Two adapters, exactly as ADR-0003 requires to make a seam real rather than hypothetical:

- **`TTSNotchPanelController`** (production) — the only code touching `NSPanel` /
  `NSScreen` and the hosted **TTS Word Tracker**.
- **`RecordingHighlightSurface`** (test peer, in `tesseractTests`) — records the call
  sequence.

The recording peer is the point. **Segment Playback** is a deep module whose reason to
exist is to own the branchiest, least-tested logic in the speech path — the
deferred-duration gate, the post-stream ~50 ms boundary-drain poll, and the overlay text
switch at the **Segment Window**. The interface is the test surface; with the overlay
concrete, every `SpeechCoordinator` test passed `notchOverlay: nil`, so the switch and its
ordering against the (ADR-0003) virtual playback clock were **untested**. The peer makes
assertable, for the first time, that segment 0 shows and segment 1 does not switch in while
the playback head sits below the boundary, that the switch fires exactly when the virtual
clock crosses it, and that the switch precedes this segment's first duration update (the
"don't corrupt the previous segment's pacing" rule, today guarded only by a local bool).

Promoting the surface also collapses the overlay's
`updateText(segmentTimeBase:segmentDurationBase:)` — always passed the same value twice —
into one `switchText(…, segmentBase:)` over the single **Segment Window**.

## Considered / rejected

An architecture review that re-suggests any of these should treat them as already-decided:

- **"The notch is the only highlight surface — keep it concrete, don't add a port."** No.
  The second adapter is the recording test peer, a real peer implementation (ADR-0003), not
  a mock; it unlocks the boundary-switch test surface that does not otherwise exist. The
  seam is justified by testability of **Segment Playback**, not by a second *production*
  surface.
- **"Make `WordHighlightSurface` an actor, or pass closures instead of a protocol."** No.
  It mirrors `AudioPlayback`: synchronous `@MainActor` calls on the hot path. A bag of
  correlated closures with a contract ordering is loose flags by another name; the named
  port plus documented ordering is the established idiom (the speech model ports,
  `AudioPlayback`).
- **"Fold the notch into the Overlay Panel."** No — unchanged from ADR-0003. The notch's
  spoken-word highlight is a separate surface from the dictation HUD; `WordHighlightSurface`
  is the notch's seam only.
- **"Port the pacing math too — a `WordTimeline` port, or a clock port."** No. **Word
  Timeline** is a pure `nonisolated` value; its locality needs no seam, and a port around a
  pure function is ceremony. The clock is already faked *through* `AudioPlayback`'s virtual
  clock, so a separate clock port would be a one-adapter duplicate.
