# ADR-0097: The reply speaks as it is written, and a short turn ends sooner

- Status: Accepted
- Date: 2026-10-27
- Amends: ADR-0082 (an interrupt while the reply is still being written also
  cancels the generation; Barge-In stays a key or a click), ADR-0042 (the Voice
  Session Machine takes the reply's text as it streams)
- Relates to: ADR-0081 (speech never waits for the LLM), ADR-0074 and ADR-0075
  (the voice's first audio), #310 (the voice session),
  [the LFM3 note](../research/2026-10-27-lfm3-live-voice-companion.md) §2 and
  §7.1

## Context

A voice turn kept the owner waiting for the whole of Jarvis's work before he
said a word:

- **The session spoke only after `agentEnd`.** ChatSession called the reply
  hook with the last assistant message's text once the run was over, so
  thinking, the full answer and every tool round came before the first word.
  The voice itself starts in about 54 ms (ADR-0074, ADR-0075); the wait was
  the generation. Qwen3.8-27B decodes at 21 to 50 tokens a second on this Mac
  (ADR-0058, ADR-0089), so a 60-token answer alone is one to three seconds
  more, and text Jarvis wrote before a tool call was never spoken at all.
- **Every turn closed on 1.8 s of silence** (the End-of-Speech Silence
  Setting), the same for "what's next?" as for a long thought.

The obvious fix for the second, a shorter fixed silence, cuts the owner off.
Replaying the endpointer's rule over the last 99 Capture Dump takes (median
10.2 s of speech) gave the share of takes that a given silence would have
closed before the owner finished:

| Close after | 1.0 s | 1.2 s | 1.4 s | 1.8 s |
| --- | --- | --- | --- | --- |
| Takes cut short | 31% | 21% | 10% | 4% |

The long pauses sit in long takes. Within the first 3 s of speech a 1.0 s
silence would have cut 1% of takes, and none of the 25 takes with 5 s of
speech or less had a pause that long. A cut is costly in a half-duplex loop:
the mic closes, and Jarvis answers half a thought.

## Decision

1. **The reply speaks as it is written.** ChatSession forwards the reply's
   visible text as the model writes it: each text part's opening text and
   deltas, then the part's end. Thinking and tool calls never reach it. The
   Voice Session Machine keeps a **Reply Speech Buffer** that cuts the text
   into whole sentences: a line break, or a word that ends a sentence as a
   take's words do (`TakeText`: `.` `!` `?`, never an abbreviation, an
   initialism, a decimal point or a pause), once the whitespace after it has
   been written. The
   first sentence starts the voice and later ones queue behind it. A text
   part's end makes what it held complete, so "Let me check your calendar"
   before a tool call is spoken too. `agentEnd` still closes the reply: the
   rest of the text, then the end of the speech. A reply that never reached a
   sentence end, or never streamed, is spoken whole then, as before.
2. **One playback stream per reply.** The Speech Coordinator gains
   `startStreamedSpeech`, `appendStreamedSpeech` and `finishStreamedSpeech`.
   Each round hands the engine everything queued so far as one utterance, so
   pieces after the first run longer, and every round's audio goes into one
   playback stream. While it waits for text the state stays live, so the
   session's watchdog doesn't read a reply paused on a tool call as over.
   The success callback fires once the finish came and the audio drained.
3. **The mic stays closed for the whole reply.** Half-duplex (ADR-0082) is
   unchanged: the machine closes the mic before the first sentence speaks
   and ignores `speechDone` until the turn has ended.
4. **A reply cut short while it is still being written is dropped.** The rest
   of its text is not spoken, and its `agentEnd` is not taken as the next
   turn's reply: the agent runs one run at a time, and every run, a cancelled
   one too, ends with exactly one `agentEnd`, so the next reply end after the
   cut is always the old run's. An interrupt (a key or a click) cancels the
   generation at once: the owner wants the floor. If the watchdog finds the
   voice dead, the chat may finish writing the reply, but the owner's next
   turn cancels a run that is still writing before it goes out, so it never
   queues behind it at the LLM Gate.
5. **A short turn ends sooner.** The silence that closes a turn grows with
   the turn: 1.0 s while the owner has spoken up to 1.5 s, rising to the
   End-of-Speech Silence Setting (1.8 s by default) at 3.5 s of speech. A
   Setting under 1.0 s closes every turn at the Setting. On the same 99 takes
   this rule cuts 5 (fixed 1.8 s cut 4), none of the 25 short ones.

## Considered and rejected

- **A shorter fixed silence.** 1.4 s cuts 10% of takes, 1.0 s 31%.
- **Speculative transcription at a short silence, committed at the long
  one.** It would also take Whisper's 0.65 s off every turn without any cut,
  but it needs Voice Capture to transcribe a snapshot of a take that is still
  recording. Left for later, with its own decision.
- **Speak the reply per sentence through `speakText`.** Each call stops the
  speech before it, so sentences would cut each other off.
- **Let an interrupted run finish in the background.** The owner's next turn
  would wait behind it, and a tool call the owner just objected to would
  still run.

## Consequences

- Jarvis starts speaking when the first sentence of his answer is written,
  after his thinking, instead of after the whole run. Text written before a
  tool call is now spoken.
- A question said in a breath closes 0.8 s sooner; a turn past 3.5 s of
  speech waits as before.
- Interrupting a reply that is still being written leaves a cut-short reply
  in the chat, as the chat's own stop button does.
- `voice.reply-spoken` marks when speech starts. A streamed reply's record
  carries `streamed: true`, and its `chars` count the first sentence.
- The overlay's spoken line grows sentence by sentence, and the companion
  line joins the transcript once the reply is complete or cut short.
- Each round of a streamed reply is its own utterance to the speech engine,
  so its frames and segment indices restart every round, and the default
  `MemoryPolicy` clears MLX's buffer pool (`trimCaches()`) after every round
  while the LLM is still decoding: a speed cost ADR-0081 already accepts,
  paid a few times per reply instead of once. The deeper fix is an utterance
  the engine keeps open for appended text, with `speakText` as its one-piece
  case; that is a change to `Vendor/tesseract-speech` and ADR-0038's contract,
  left for its own decision.
- Not changed: thinking still runs before the first word. Moving the voice
  turn's reasoning effort breaks Qwen3.8's cached prefix (it is written into
  the first system block), so that needs its own decision.

## As built

- `Features/Companion/Voice/ReplySpeechBuffer.swift`: the sentence cutter.
- `Features/Companion/Voice/VoiceSessionMachine.swift`: `.replyText`,
  `.replyPartEnded`, `.speakMore`, `.finishSpeaking`, `.cancelReply`,
  `.settleCompanionLine`, `wantsReply`, and `ReplyStream` (closed, open,
  dropping) for a reply still being written.
- `Features/Companion/Voice/VoiceEndpointer.swift`: `shortTurnSilence`,
  `patienceRamp`, `closingSilence(afterSpeaking:)`.
- `Features/Companion/Voice/CompanionVoiceSessionController.swift`:
  `replyText(_:)`, and `replyCompleted(_:)` asks the machine.
- `Features/Agent/ChatSession.swift`: `voiceReplyTextHandler`,
  `ChatSession.ReplyText`.
- `Features/Speech/SpeechCoordinator.swift`: the streamed-reply API.
  `speakText` loses `showsOverlay`: the voice session was its only
  audio-only caller.
- `App/DependencyContainer.swift`: the wiring; `cancelReply` is
  `ChatSession.cancelGeneration()`.
- Tests: `ReplySpeechBufferTests`, `VoiceSessionMachineTests` (the reply as
  it is written), `VoiceEndpointerTests` (the growing close),
  `SpeechCoordinatorTests` (one stream, nothing taken after a stop),
  `ChatSessionTests` (the hook hears visible text only).
