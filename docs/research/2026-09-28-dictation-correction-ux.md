# Dictation correction: how the best products let you fix a word, and what they learn from it

**Date:** 2026-09-28
**Question:** How do the best dictation products let a user fix a misheard word, and how do they learn from the fix so it does not come back? Which of their patterns fit Tesseract, where a fix must take one or two actions from where the owner already is, every fix must teach the app, and the page and overlay must stay clean?

**Method.** Primary sources first: help centers, changelogs, developer docs, open-source code read at pinned commits, patents and papers. Forums and reviews only for complaints, marked as such. The web-search budget ran out partway through, so later lookups were direct fetches; Reddit, the KnowBrainer Dragon forum and the ACM Digital Library were blocked. Product behavior is as documented on the date above. The full working notes, with about 360 sources, stay outside the repository.

---

## 1. Two actions, at the text, is the norm

| Product | How a word is fixed | Actions |
|---|---|---|
| Dragon | Keypad-minus or "Correct that", then pick from up to 9 choices, arrow keys, or type ([Correction options](https://www.nuance.com/products/help/dragon16/dragon-for-pc/enx/dpg-cp/Content/DialogBoxes/options/options_dialog_correct_tab.htm), [NaturallySpeaking 13 guide](https://speechrecsolutions.com/assets/user_guides/dns13_userguide.pdf)) | 2 utterances, or 3 to 4 keys |
| macOS Dictation | Click a word with a blue underline, pick an alternative ([Mac User Guide](https://support.apple.com/guide/mac-help/mh40584/27/mac/27)) | 2 |
| Windows Voice Access | "Correct X", then "Click 1" ([Microsoft](https://support.microsoft.com/en-us/accessibility/windows/voice-access/correct-text-with-voice)) | 2 utterances |
| Gboard | Tap the word, pick an alternative ([Gboard help](https://support.google.com/gboard/answer/11197787?hl=en)) | 2 |
| Aqua Voice Edit Mode | Select the text in any app, hold the dictation key, say or spell the corrected text; no command words ([Edit Mode](https://aquavoice.com/guide/edit-mode)) | about 2 |
| Wispr Flow | Retype the word in the target app; the app watches the field ([Data Controls](https://wisprflow.ai/data-controls)) | the edit itself |
| Wispr Flow dictionary, Otter editor | A separate window ([Wispr dictionary](https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary), [Otter](https://help.otter.ai/hc/en-us/articles/360047731754-Edit-a-conversation)) | 5 to 7 |

Correcting in a history window is the slow outlier everywhere it exists, which is also why Tesseract has zero corrections in 1,000 takes.

## 2. The edit is the correction, where the app can see it

- Wispr Flow watches the text box it pasted into and adds a changed spelling to the dictionary automatically, marking learned words with a badge ([Data Controls](https://wisprflow.ai/data-controls), [bulk import and badges](https://docs.wisprflow.ai/articles/8955301725-how-do-i-bulk-import-for-dictionary-and-snippets)).
- VoiceInk 2.20 is the most complete open-source version ([release](https://github.com/Beingpax/VoiceInk/releases/tag/v2.20)). After a real paste it waits 120 ms, reads the focused field through Accessibility ([reader](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Features/Dictionary/AutoLearn/AutoLearnAXTextReader.swift)), and finishes when focus leaves, at the next dictation or paste, or after 60 s ([limits](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Features/Dictionary/AutoLearn/AutoLearnTypes.swift#L174-L188)). It finds the pasted region again by anchors and keeps only small edits ([diff](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Features/Dictionary/AutoLearn/FinalSnapshotDiffEngine.swift)).
- Dragon learns from keyboard fixes only in fields it fully controls, and offers a separate Dictation Box everywhere else ([Dictation Box](https://www.nuance.com/products/help/dragon161/dragon-for-pc/enx/dlg-cp/Content/Dictation/using_the_dictation_box.htm)).
- Watching is blind in many apps. Wispr lists Slack, Teams, Gmail, Notion, Claude and ChatGPT among apps that return no surrounding text ([Wispr help](https://docs.wisprflow.ai/articles/4518230910-why-flow-capitalizes-words-mid-sentence-and-how-to-control-it)), and VoiceInk's auto-learn misses Electron apps such as Claude Desktop ([issue 977](https://github.com/Beingpax/VoiceInk/issues/977)). Every product that watches also has a fallback fix surface.

## 3. Tell a correction from a rewrite before learning

- Dragon separates Correct (learn) from Select (don't), and warns that "correcting" words it got right degrades accuracy ([Dragon for Mac](https://www.nuance.com/products/help/dragon/dragon-for-mac/enx/Content/Correction/AboutCorrection.htm)). Its patents gate adaptation on a recognition-score difference ([US7315818](https://patents.google.com/patent/US7315818B2/en)) and train only on confirmed misrecognitions ([US5794189](https://patents.google.com/patent/US5794189A/en)).
- Wispr's Canto model separates recognition errors from rewrites by how well the edited word matches the audio plus the edit's place and shape, keeping "cloud" → "Claude" and discarding a rewritten sentence ([Canto](https://wisprflow.ai/canto)).
- VoiceInk asks an LLM to accept only edits that sound like the original and are not semantic rewrites, and rejects grammar, style, number and case-only edits ([reviewer](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Features/Dictionary/AutoLearn/AutoLearnAIReviewer.swift)).
- Without a gate the dictionary fills with junk: Dragon users describe nonsense words and deleted words coming back ([forum](https://forums.knowbrainer.com/forum/dragon-speech-recognition/3138-can-a-word-be-permanently-deleted-from-the-built-in-vocabulary), unverified snippet), and Wispr had to start filtering duplicate entries ([changelog](https://wisprflow.ai/whats-new)).

## 4. Learn two things per fix, and know where each one acts

- An exact heard → meant replacement: deterministic, safe for a mishearing that repeats. Wispr's "Correct a misspelling", Superwhisper replacements, Deepgram [find and replace](https://developers.deepgram.com/docs/find-and-replace), AssemblyAI [custom spelling](https://www.assemblyai.com/docs/pre-recorded-audio/correct-spelling-of-terms). VoiceInk runs replacements before its LLM pass.
- A small, ranked bias list for the recognizer, which covers spellings the rule has not seen. It degrades as it grows: Deepgram describes force-fitting ([large vocabularies](https://deepgram.com/learn/large-vocabulary-speech-recognition)) and recommends a few dozen [key terms](https://developers.deepgram.com/docs/keyterm).
- Whisper's own bias channel is small and leaky. Only the last 223 prompt tokens are used ([decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py)), the model copies the prompt's style ([OpenAI prompting guide](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Whisper_prompting_guide.ipynb)), and prompt text can leak into the transcript ([discussion 1150](https://github.com/openai/whisper/discussions/1150)). OpenAI suggests a text post-pass for longer glossaries ([misspelling cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Whisper_correct_misspelling.ipynb)). VoiceInk put dictionary words in the Whisper prompt in version 1.40 ([v1.40](https://github.com/Beingpax/VoiceInk/blob/v1.40/VoiceInk/Whisper/WhisperPrompt.swift)) and no longer does ([current](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Infrastructure/Providers/Transcription/Whisper/WhisperPrompt.swift)). Our own measurement agrees: see the errors note, section 1.6.
- A small local model does not apply a glossary reliably: a 1B model could not follow even a single-word replacement instruction ([VoiceInk issue 976](https://github.com/Beingpax/VoiceInk/issues/976)).
- Handy, fully offline, uses custom words as Whisper's prompt and fuzzy spelling-and-sound matching (Levenshtein plus Soundex) for other models ([code](https://github.com/cjpais/Handy/blob/8f9cf53cd1410cda26beea39ff802ac306e39585/src-tauri/src/audio_toolkit/text.rs)).

## 5. Show the learning when it happens

- VoiceInk shows "Learned X → Y" with an Undo button for 4 seconds, or queues proposals for review if the user prefers ([code](https://github.com/Beingpax/VoiceInk/blob/593fadb/VoiceInk/Features/Dictionary/AutoLearn/AutoLearnService.swift#L649-L699)).
- Descript can notify when a word is added to its glossary, which it does after three corrections ([help](https://help.descript.com/script-editing/transcription-glossary)). Talon confirms additions in a notification ([vocabulary.py](https://github.com/talonhub/community/blob/dafe1dc/core/vocabulary/vocabulary.py)).
- Wispr marks learned words with a badge and never learns a deleted word again ([dictionary](https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary)).
- Apple, Gboard and Otter learn silently, and users ask whether the app will ever learn.

## 6. Alternatives and confidence marks help less than they seem

- The right word was in a list of six alternatives only 24% of the time, and speaking it again fixed 35% of errors ([Suhm et al. 2001](https://www.cs.cmu.edu/~cpof/papers/suhm_tochi.pdf)). Spoken corrections alone were right 21% of the time ([Vertanen and Kristensson 2010](https://www.keithv.com/pub/second/spoken_corrections.pdf)).
- Underlining low-confidence words did not help people find more errors overall, and they missed the unmarked ones ([Vertanen and Kristensson 2008](http://pokristensson.com/pubs/VertanenKristenssonCHI2008.pdf)). With Whisper, confidence-based flagging had precision 0.48 and recall 0.54, and users fixed about half the errors either way ([Kuhn et al. 2025](https://arxiv.org/html/2503.15124v1)). On the owner's takes the name errors were confident (errors note, section 1.5).

## 7. Every automatic change needs a one-step revert, and a revert is a lesson

Wispr's "Undo AI edit" ([History](https://docs.wisprflow.ai/articles/5096240724-navigating-the-wispr-flow-app-desktop-ios-and-android)), Aqua's History undo ([History](https://aquavoice.com/guide/history)) and Windows fluid dictation's revert ([Microsoft](https://support.microsoft.com/en-us/accessibility/windows/voice-access/fluid-dictation)) all restore the raw transcript. AppKit's spell checker reports reverted corrections back to the system precisely so it can learn from them ([NSSpellChecker.CorrectionResponse](https://developer.apple.com/documentation/appkit/nsspellchecker/correctionresponse)).

## 8. Anti-patterns

- Correction only in a history window.
- Speaking it again as the only fix.
- Learning silently, with nothing to review or undo.
- Adding every edit to the dictionary without a gate.
- A bias list that only grows.
- Confidence underlines as the main correction surface.
- Commands the user has to memorize.
- A learned vocabulary that never reaches the recognizer.
- An LLM cleanup with no one-step way back to the raw text.
