# Dictation personalization: what teaches a local recognizer the owner's words, cheapest first

**Date:** 2026-09-28
**Question:** How can one owner's corrections teach Tesseract's dictation (WhisperKit large-v3-turbo, regex cleanup, the Qwen3.5-0.8B Proofread Pass) so a fixed word stays fixed? For each technique: the published effect, the cost, the risk, and what the owner's own replayed audio said.

**Method.** Papers read from their text (arXiv, ACL Anthology, ISCA), source code of WhisperKit (argmax-oss-swift 1.0.0, which the app pins, and 1.1.0), mlx-lm and mlx-swift-lm, and vendor docs, compared against measurements on the 99 replayed takes described in `2026-09-28-dictation-errors-and-learning.md`. The web-search budget ran out partway through, so later papers came from the arXiv API. Full working notes stay outside the repository.

---

## 1. Whisper's prompt: drop it

- Whisper conditions on the prompt as the transcript that came before the audio, not as instructions ([Radford et al. §2.3](https://arxiv.org/abs/2212.04356); an OpenAI maintainer says the same for turbo in [discussion 2363](https://github.com/openai/whisper/discussions/2363)). OpenAI's guide says spelling prompts "are not especially reliable" ([cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb)).
- A static 70-word list lowered rare-word error (23.7 to 18.0) and raised everything else (9.1 to 11.6), with overall WER worse on 6 of 11 test sets ([Jogi et al. 2025](https://arxiv.org/abs/2502.11572)). Prompts helped only when they held just the words spotted in that utterance ([CB-Whisper](https://aclanthology.org/2024.lrec-main.262/)); a personal vocabulary is mostly distractors for any one take.
- WhisperKit 1.0.0 returns empty text when the model predicts end-of-text on a forced prompt token ([issue 501](https://github.com/argmaxinc/argmax-oss-swift/issues/501)); [PR 514](https://github.com/argmaxinc/argmax-oss-swift/pull/514), released in 1.1.0, fixes it. WhisperKit fits the prompt, the task tokens and the output into one 224-token window and feeds the prompt through the decoder one token at a time.
- **On the owner's audio** (99 takes, 48 term occurrences):

| WhisperKit | Prompt | Word error rate | Terms right | Median decode |
|---|---|---|---|---|
| 1.0.0 (app) | none | 0.92% | 35 | 1.3 s |
| 1.0.0 | term list | empty on 74 of 99 takes | | |
| 1.1.0 | none | 2.05% | 35 | 1.3 s |
| 1.1.0 | term list | 2.75% | 35 | 2.1 s |
| 1.1.0 | sentence using the terms | 6.50% | 37 | 1.6 s |

  WhisperKit 1.1.0 also lost the first 38 words of one 34-second take without any prompt, so an upgrade needs its own check.

## 2. Bias the decoder toward known terms: do first

- WhisperKit applies custom `logitsFilters` at every decoding step, in 1.0.0 and 1.1.0. A filter that adds a bonus to the next token of a term whose first tokens were just emitted is shallow fusion over a prefix trie: no prompt, no model, no extra decoder steps.
- Shallow-fusion biasing halved contact-name errors (15.8 to 7.5), and the same paper warns that an always-on list with no prefix gating took general voice search from 6.9 to 12.5 ([Zhao et al. 2019](https://www.isca-archive.org/interspeech_2019/zhao19d_interspeech.pdf)).
- **On the owner's audio:** a continuation-only bonus fixed 5 missed terms with no regressions; adding a first-token bonus fixed 7 but also nudged ordinary words; decode time rose about 2%.
- Risk specific to this vocabulary: terms that begin with an ordinary token (a person's name that starts with "And", "Whis"+"per"+"Kit", "Swift"+"UI") can turn "And I" into that name. Require two matched tokens when the first is an ordinary word, and test on takes that contain none of the terms.

## 3. Learned replacements: do first

- An exact "heard → meant" rule from a fix is deterministic and cheap, and every product with a dictionary ships one (Wispr Flow, VoiceInk, Superwhisper, Deepgram find-and-replace, AssemblyAI custom spelling).
- Apply a rule automatically when it has been confirmed twice, or when the heard form is not a dictionary word, or when Whisper was unsure of it; otherwise offer it.
- Phonetic codes only suggest: Double Metaphone gives "can" and "Qwen" the same code. On the owner's 1,000 takes, sound-alike matching caught 10 more repeats but raised 31 false alarms.

## 4. Word confidence: keep it as a gate, not a display

- WhisperKit computes a log-probability per token; the app discards it. Whisper confidence separates errors at AUC 0.69 to 0.86 ([C-Whisper](https://arxiv.org/abs/2502.13446)), and confidence-based flagging in a correction interface had precision 0.48 and recall 0.54 ([Kuhn et al. 2025](https://arxiv.org/abs/2503.15124)).
- On the owner's takes the name errors were confident, so confidence cannot find them. It can decide when an uncertain learned rule may fire.
- WhisperKit's beam search is not implemented, so there are no n-best lists.

## 5. A small LLM corrector: turn it off, fine-tune later

- Zero-shot correction of a strong Whisper is neutral to harmful even with large models: GPT-3.5 made large-v3 output 7.4% worse ([Naderi et al. 2024](https://arxiv.org/abs/2407.21414)). No study shows a model of 1.5B parameters or less doing it. On whisper-large-v3-turbo, zero-shot GPT-4o-mini barely moved WER, while a corrector fine-tuned on synthetic rare-word data (LLM sentences, TTS, Whisper) raised rare-word recall from 80.5 to 91.7 ([arXiv 2505.17410](https://arxiv.org/abs/2505.17410)).
- What helps is gating: correct only uncertain spans and keep everything else ([Pu et al.](https://arxiv.org/abs/2310.11532)).
- Fine-tuned correctors of 250M to 3B lost at about 2,000 pairs and won at about 4,000 ([FlanEC](https://arxiv.org/abs/2501.12979)); gains grew from 1,700 to 4,000 pairs in [HyPoradise](https://arxiv.org/abs/2309.15701). Relabelling pairs whose fix can't be inferred to "no change" cut over-editing from 43% to 14% ([IBM](https://arxiv.org/abs/2407.13300)).
- Tooling: mlx-lm trains LoRA, but Qwen3.5's Gated DeltaNet layers have no Metal backward kernel yet and LoRA runs out of memory on the 4B and 9B ([mlx-lm](https://github.com/ml-explore/mlx-lm)); mlx-swift-lm's gated-delta recurrence became trainable in [PR 616](https://github.com/ml-explore/mlx-swift-lm/pull/616), which the vendored fork has.
- **On the owner's audio:** the zero-shot pass doubled the words that differ from what was said; with a 40-term glossary in its prompt it fixed no terms.

## 6. Adapting Whisper itself: later, separately

- It has the highest ceiling: 50 corrected utterances (4.6 minutes) took person-name recall from 2.4% to 73.5% on an on-device recognizer ([Sim et al. 2019](https://arxiv.org/abs/1912.09251)), and 14 minutes of accented speech gave 70% of a 35% relative gain ([Shor et al. 2019](https://arxiv.org/abs/1907.13511)).
- It needs a PyTorch training run and a Core ML conversion ([whisperkittools](https://github.com/argmaxinc/whisperkittools)); MLX can't train Whisper today. The gold takes' audio, which the Capture Dump already keeps, is the input.

## Order

Decoder bias fed by learned terms, learned replacements, confidence gating, the Proofread Pass off by default, then fine-tuning once there are a few thousand gold and "no change" pairs, and Whisper adaptation as its own project.
