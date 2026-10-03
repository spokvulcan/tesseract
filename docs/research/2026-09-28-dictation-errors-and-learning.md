# Dictation: what goes wrong, and how the app should learn from a fix

**Date:** 2026-09-28
**Question:** Which errors reach the text Tesseract Agent inserts, what does each stage of the dictation pipeline add or remove, and which signals and mechanisms would let one correction stop the same mistake from coming back?

**Method.** Read-only snapshot of the owner's dictation data on 2026-09-28: 1,000 takes in the Correction Pair store (2026-07-27 to 2026-09-27, 222 minutes, 21,933 words) and the 99 takes whose audio was still in the Capture Dump (2026-09-25 to 2026-09-27, 34 minutes, 3,459 words). The audio was replayed offline through the production transcription path: the same WhisperKit revision, the same decoding options as the app's recognizer, the same resampler and the app's regex post-processor, compiled as a command-line tool. The replay reproduces the stored raw text exactly on 82 of 99 takes; the rest differ by punctuation or one word. The Proofread Pass was replayed on all 1,000 stored texts with the same checkpoint, prompt and greedy decoding. Reference transcripts for the 99 takes were written by hand from the production text, a second recognizer from another model family (Parakeet TDT 0.6B v2), Whisper's own alternative tokens, the audio level and the context. Only confident calls went into the references, so the error counts below are a floor. Nothing in the owner's data was written to.

Per-take analysis quotes the owner's dictation and stays out of the repository. The numbers here are aggregates, and the examples are technical terms.

---

## 1. Findings

### 1.1 Generic accuracy is high. Nearly every error is one of the owner's own words.

On the 99 replayed takes, Whisper large-v3-turbo gets 99.4% of words right before any cleanup (word error rate 0.61%, ignoring case and punctuation). Of the 16 word errors left, 14 are the owner's names and technical terms:

| Type | Errors | Examples (heard → meant) |
|---|---|---|
| Product and proper names | 11 | "whisper flow" → Wispr Flow, "Eleven Labs" → ElevenLabs, "QWEN 3.TTS, Void Design" → Qwen3-TTS VoiceDesign, "test flight" → TestFlight, and one of the owner's project names |
| Technical terms | 3 | "vendor at" → vendored, "work trees" → worktrees |
| Common words | 1 | "Depagination" → "The pagination" |
| Text with no speech behind it | 1 | "Thank you." from a silent capture (1.5) |

The same holds across all 1,000 takes. Reading them turned up 83 misrecognitions of about 30 terms in 68 takes (6.8% of takes), and that list is certainly incomplete. The app's own name came out as "SRACT", "SRAX", "TSRAC" and "struct"; "Claude" was never transcribed correctly and came out as "cloud" (so `CLAUDE.md` became "cloud.md"); DFlash2 was spelled six ways; "the PR" and "the MR" merged into "APR", "DPR", "DMR". A generic proofreader cannot know any of these.

Numbers were fine except inside versioned model names ("Qwen 3.8 27B" became "3.827B"). The language setting is fixed to English and all 99 replayed takes were English, so the data says nothing about mixed-language speech.

### 1.2 The Proofread Pass makes the text worse more often than it fixes it

Over the 1,000 takes the pass left 874 unchanged, corrected 123, skipped 3 and rejected none. Reading all 123 corrections:

| What the correction did | Count |
|---|---|
| Changed the meaning: deleted a clause, flipped a pronoun, answered a question, finished a cut-off sentence | 13 |
| Smaller damage: broke a technical term or the grammar | 8 |
| Fixed a misheard word | 4 |
| Fixed punctuation, or a break the regex had introduced | 23 |
| Rewrote the speaker's own phrasing: grammar, fillers, "Okay, got it." | 69 |
| No net effect | 6 |

The worst cases are the ones a reader would never notice: a question came out as its own answer, a whole sentence was dropped from an instruction, and the sentence introducing a pasted code review was deleted from three messages that depended on it.

On the 99 takes with audio, the text the pass committed differs from what was said in 1.39% of words, against 0.69% before the pass, and deletions went from 3 to 15. The acceptance guard caught 13 outputs in the replay (1.3%), all of them the model answering or summarizing instead of proofreading ("No", "I am Qwen3.5…"). Anything subtler passes it.

Giving the model the owner's vocabulary does not help: with a glossary of the 40 terms in its system prompt it fixed none of them and its error rate went to 1.42%. That matches a report from VoiceInk's issue tracker that a 1B local model cannot reliably apply even a single-word replacement (see the correction UX note).

Cost is not the problem. In the replay a pass takes a median 69 ms (p90 192 ms, slowest 1.1 s), nowhere near its 4 s budget. The persistent KV cache that ADR-0034 describes never engages, because Qwen3.5's linear-attention layers cannot be trimmed, so every pass prefills the whole system prompt again; at this size that costs little.

### 1.3 The regex post-processor introduces errors of its own

- It splits file names and domains: `CLAUDE.md` and `versions.txt` became "cloud. Md" and "versions. Txt" (14 takes).
- It turns Whisper's pause marker "..." into a sentence break with a capital ("I... don't" became "I. Don't"): 61 false breaks in 47 takes.
- It deletes deliberate repetition: "very very" and "much much" lose a word (10 takes). It also removes real stutters ("current current"), which is right.

### 1.4 A silent capture gets text, and the no-speech check never fires

One capture in the dump peaks at −78 dBFS, which is silence, and it was transcribed as "Thank you." and inserted. Seven of the 1,000 takes are exactly "Thank you.", all between 1.3 and 3.1 s long. The app passes WhisperKit a no-speech threshold, but WhisperKit never computes the probability it compares against: at this revision it is a [hard-coded zero with a TODO](https://github.com/argmaxinc/argmax-oss-swift/blob/25c62997041c134b03ca82731ce2f6fd2cae1eb9/Sources/WhisperKit/Core/TextDecoder.swift#L802), and all 320 replayed segments report 0.

### 1.5 Whisper's confidence does not point at these errors

The name errors are confident. Whisper wrote "whisper flow", "vendor at" and "Eleven Labs" with every token at a probability of about 0.5 or higher. Flagging words whose weakest token has a log-probability below −1.0 marks 1.5% of words, and only about one in twelve of those is an error. Underlining uncertain words would mostly highlight stutters and punctuation, which is also what the published studies found (correction UX note, section 6).

### 1.6 Whisper's prompt cannot carry a vocabulary here

Setting the decoder prompt to the owner's terms made WhisperKit 1.0.0, the version the app pins, return empty text for 74 of 99 takes. It feeds the prompt one token at a time, and when the model predicts end-of-text at a forced prompt position, [the loop ends the segment](https://github.com/argmaxinc/argmax-oss-swift/blob/25c62997041c134b03ca82731ce2f6fd2cae1eb9/Sources/WhisperKit/Core/TextDecoder.swift#L668) before transcribing anything. This is [WhisperKit issue 501](https://github.com/argmaxinc/argmax-oss-swift/issues/501), fixed in 1.1.0.

On 1.1.0 the prompt still doesn't pay. Against 0.92% and 35 of 48 terms on 1.0.0 without a prompt:

| WhisperKit 1.1.0 | Word error rate | Terms right | Median decode |
|---|---|---|---|
| No prompt | 2.05% | 35 | 1.3 s |
| The terms as a list | 2.75% | 35 | 2.1 s |
| A sentence using the terms | 6.50% | 37 | 1.6 s |

At best two more terms, bought with more errors and slower takes. The published work explains why: Whisper reads a prompt as the transcript that came before, so a static term list is mostly distractors for any one take (personalization note, §1). VoiceInk put dictionary words in the Whisper prompt in version 1.40 and has since taken them out. The 1.1.0 upgrade on its own also lost the first 38 words of one 34-second take, so it needs its own check before the app moves to it.

### 1.7 Nudging the decoder toward known terms works, gently

WhisperKit accepts custom logits filters, so a vocabulary can bias decoding without a prompt: when the tokens just emitted start one of the owner's terms, the term's next token gets a bonus, and optionally each term's first token gets a smaller one. Measured on the 99 takes, against 48 occurrences of the terms that matter:

| Configuration | Word error rate | Terms right |
|---|---|---|
| No biasing | 0.92% | 35 / 48 |
| Vocabulary learned from the older takes, continuation bonus only | 0.90% | 37 / 48 |
| Same, plus canonical spelling of known terms | 0.78% | 39 / 48 |
| Vocabulary after each term was fixed once, continuation bonus only | 0.78% | 39 / 48 |
| Same, with a start bonus too | 0.87% | 42 / 48 |
| Same, plus canonical spelling | 1.18% | 46 / 48 |
| Too strong a bonus | 3.15% | 40 / 48 |

The continuation bonus alone made five fixes and no regressions. The start bonus finds more terms but also nudges ordinary words when a term begins with one ("And" from a person's name that starts with it, "App" from App Store), so it belongs only on rare first tokens. Canonical spelling helps only when it comes from a fix: applied blindly from a vocabulary, it turned every "voice design" the owner said as plain words into VoiceDesign. Biasing costs about 2% more decode time.

### 1.8 One fix per term would catch half of the later repeats

Replaying the 83 misrecognitions from 1.3 in time order, as if the owner fixed each term the first time it went wrong:

- 32 are first sightings, which the owner fixes once.
- 39 of the rest (47%) repeat an exact misheard form already fixed, so a learned "heard → meant" replacement fixes them with no extra action.
- 12 more repeat a known term in a new spelling ("SRAX" after "SRACT"). Phonetic matching against known terms catches 10 of them but also raises 31 false alarms on correct words ("backend" matched "Pi agent"), so it may suggest a fix but must not apply one.

### 1.9 Latency by stage

From release of the key, on the replayed takes: resampling under 1 ms, transcription a median 1.31 s (p90 2.7 s, 4.7 s for a 165 s take), regex under 1 ms, proofreading a median 69 ms, and the paste about 100 ms. The app logs these spans at the info level, which macOS does not keep, so there is no history of them on the owner's machine.

### 1.10 The loop has no input

All 1,000 stored takes are candidates. None is gold: the owner never flagged or corrected a take, because correcting means opening the history window, finding the take and editing it there. The living memory stores dictated text as it was inserted (283 of its 418 episodes are dictation), so memory inherits every misrecognition too.

---

## 2. What the best products do, and what the research adds

Three notes carry the sources: `2026-09-28-dictation-correction-ux.md`, `2026-09-28-dictation-personalization.md` and `2026-09-28-dictation-accessibility-learning.md`. In short:

- **Two actions, at the text.** Dragon (keypad-minus or "Correct that", then choose), macOS Dictation (click the underlined word, pick), Gboard (tap, pick) and Windows Voice Access ("Correct X", "Click 1"). A history window is the slow outlier.
- **The edit is the correction, where the app can see it.** Wispr Flow and VoiceInk watch the field they pasted into and learn from edits. It is blind where this owner dictates most: terminals expose their whole scrollback as one read-only string, and Electron apps only expose text after being switched into screen-reader mode, which changes how they behave (VS Code turns on its screen-reader mode). A Claude Code `UserPromptSubmit` hook sees the prompt as it was actually sent, with no Accessibility at all.
- **Learn two artifacts, gate them, show them.** A deterministic "heard → meant" replacement plus a vocabulary entry; a check that the edit sounds like the original and is not a rewrite; a "Learned X → Y" toast with Undo; a ledger of what was learned, with deleted entries remembered so they are not learned again.
- **Bias the decoder, not the prompt.** Shallow fusion over a prefix trie is the published version of 1.7, and its risk is biasing toward terms that were not said; the fix is to gate on matched prefixes and test on takes without the terms.
- **Fine-tune only with thousands of pairs.** Small fine-tuned correctors lost at about 2,000 pairs and won at about 4,000, and relabelling unfixable pairs to "no change" cut over-editing from 43% to 14%.

## 3. Recommendation

**Signals to collect.**
1. A fix made where the text is, in one or two actions, from the overlay or the target app. Each fix makes the take gold and records what changed, how it was fixed and which app received the text.
2. For Claude Code, an opt-in `UserPromptSubmit` hook: the prompt as sent, diffed against what was dictated into it, filtered like any edit. For this owner it should yield more pairs than anything that reads the screen.
3. Edits made in place, read through Accessibility, only in apps the owner turns on that expose their text well (Safari and WebKit apps, native fields). Never in terminals, never in secure fields, never by switching an app's accessibility on. These are candidates until the owner confirms them.
4. Vocabulary added or removed by hand, and suggestions accepted from the living memory.
5. Undo of anything learned or applied automatically, recorded as a lesson against it.
6. Per take: target app, stage timings and what the learning layer changed, so the next measurement has data. Keep the audio of gold takes, as the Capture Dump already does.

**Apply at runtime.**
1. Fix the regex bugs and skip transcription of a silent capture. Zero cost.
2. Learned replacements from fixes, applied straight after recognition, shown in the overlay, one-step undo. Automatic when the rule was confirmed twice, the heard form is not a dictionary word, or Whisper was unsure of it; otherwise offered.
3. Decoder biasing toward learned terms through a logits filter: continuation bonus only, two matched tokens when a term starts with an ordinary word, a capped list ranked by use and recency, checked on takes that contain none of the terms.
4. Keep Whisper's per-token confidence with the take, to gate uncertain rules. Don't show it.
5. Turn the Proofread Pass off by default and stop treating it as the corrector. Formatting, which the owner has asked for, can come back as a separate opt-in step once it passes an evaluation on gold takes.
6. Offer names and projects from the living memory as vocabulary suggestions, never apply them silently.
7. Leave `promptTokens` unset, and upgrade WhisperKit only after a replay shows no loss.

**Fine-tune later.** Fine-tuning the small model (#294) starts once there are a few thousand gold and "no change" pairs, and ships only if it beats "no pass" on a replay of the gold takes; synthetic pairs made with the app's own voice and the owner's terms can top up the rare words. Adapting Whisper has the highest ceiling (minutes of corrected speech moved name recall from 2% to 74% on another recognizer) and needs its own project and the gold audio, which is kept. The Neural Engine port (#293) matters only if a pass comes back.

**For the design.** Every fix is at most two actions from where the owner already is, and one global shortcut works in any app, terminals included. Alternatives are shown only in the fix surface, drawn from the learned vocabulary. Uncertainty is not underlined. What was learned is shown at the moment of learning, with Undo, and kept in a ledger on the Dictation page.
