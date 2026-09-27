# ADR-0076: The Speech page is a Reader on one Read-Along clock

- Status: Accepted
- Date: 2026-09-27
- Replaces: the notch adapter of the Word Highlight Surface (ADR-0004) and the
  TTS Word Tracker that drove it
- Relates to: ADR-0038 (engine v2: Segment Scripts and `startFrame`), ADR-0072
  (Reference Take), ADR-0073 (tests never reach the owner's data)

## Context

The Speech page was a composer: a SwiftUI text field, a transport bar and an
inspector of sampling knobs. Speech showed in a notch overlay with no settings
on the page. Three things were wrong with it.

- Pasting a book made the page crawl. SwiftUI text lays out the whole string,
  and the page redrew it as speech progressed.
- The notch ran ahead of the voice. It switched to a segment's text when the
  segment's script arrived, and under lookahead pacing a script arrives up to
  8 s before its audio plays.
- Reading couldn't be followed in the text itself, and the notch couldn't be
  set up from the page.

Five prototypes were built on a throwaway branch (`prototype/speech-page`,
9a953d1e), after a survey of TTS studios
([research](../research/2026-09-27-tts-studio-ui-patterns.md)). The owner
chose the Reader (B) with the Voice Lab's designer (C), and asked for a clean
page that stays fast with a whole book pasted in.

## Decision

1. **The page is a Reader.** An AppKit text view on TextKit 2, which lays out
   only what is on screen, holds the text. It is an editor at rest and
   read-only while reading, when a click jumps the reading and Space pauses.
   One floating glass bar holds sentence back, play and pause, sentence
   forward, where you are (drag to go anywhere) with the time left, stop,
   speed and the voice. One Display popover holds text and overlay settings.
   The sampler's knobs move to Settings > Speech.
2. **One clock.** The **Read-Along** (`SpeechReadAlong`) is the production
   Word Highlight Surface, and the Reader and the Speech Overlay both follow
   it; nothing else runs a speech timer. A segment starts when the playback
   head reaches its `startFrame`, never when its script arrives. Inside a
   segment, words are placed by characters over the segment's audio: at the
   learned pace until the segment's end is known, then on to that end from
   wherever the pace had got to, so the word never moves back. It samples the
   head 30 times a second and publishes only when the heard word changes.
3. **No whole-document index.** The Reader maps each segment by counting
   words forward from where the previous one ended (the engine, the Word
   Timeline and the Reader split words the same way), and finds sentences
   only inside the segment at hand. Skips and seeks look at a window of 4,000
   characters around the offset. Highlights are TextKit 2 rendering
   attributes, so a word costs no layout. TextKit 2 doesn't redraw when those
   change, so the Reader marks the lines it changed.
4. **The Bookmark** is where reading resumes. The text and the bookmark are
   saved in the app-support `Speech` folder. The bookmark follows the sentence
   being heard, moves with edits before it, and goes back to the start once the
   text has been read to the end.
5. **The Speech Overlay replaces the notch.** It is an island at the top of
   the screen or captions at the bottom, in a chosen size and word color, with
   pause and stop on hover. In its automatic scope it hides while the Speech
   page is in front. Changing a setting previews it on screen.
6. **Voices by Voice Source.** A Voices sheet lists built-in designs, saved
   designs and designed voices never named. The designer shows only when the
   checkpoint designs voices (VoiceDesign). Its trait chips write Qwen's compact
   description shape
   ([research](../research/2026-09-27-qwen3-tts-voice-design.md)), each
   language has its own reference line, and each finished take joins a strip
   to compare and keep (ADR-0072). A CustomVoice checkpoint lists its own
   speakers and nothing else.
7. **Speed stretches time.** A time-pitch unit sits between the player node
   and the mixer. The player's clock stays in the audio's own time, so the
   Read-Along needs no rate arithmetic.
8. **A cancelled request leaves its successor alone.** When `stop()` cancels a
   request, its late cleanup no longer resets the state or drops the
   completion callback: by then both belong to the request that replaced it.
   That cleanup had ended a reading at every jump.

## Consequences

- An 80,000-word book opens at once and starts reading in about a second.
  Each word costs a redraw of a line or two.
- Deleted: the notch panel and view, the TTS Word Tracker and its pacing fold,
  the composer, the transport bar and the parameters inspector. The Word
  Highlight Surface no longer carries token offsets; the engine still
  announces them.
- Placing words by characters drifts at pauses, as the notch did. There is no
  forced alignment.
- One document, plain text: no file import, no chapters, and formatting from
  a paste is dropped.
- Saving audio and editing it were prototyped (D, Takes) and not chosen. The
  Reader leaves room for an export later.

## Alternatives considered

- **SwiftUI text with an attributed highlight.** It lays out the whole string
  again on every change, which is unusable for a book.
- **Indexing words and sentences up front.** That costs O(n) on every paste
  and edit, and the engine speaks in order anyway.
- **Fixing the notch's clock and keeping the notch.** A notch-wide strip shows
  too little of the text, and displays without a notch had no good place
  for it.
