# Model parameters reference

The numbers we keep looking up: context windows, recommended output
lengths, sampling presets, thinking defaults, and what each checkpoint
ships. One row per catalog entry (`tesseract/Features/Models/ModelDefinition.swift`),
sourced from the official model cards and the checkpoint `config.json` /
`generation_config.json` on disk. Last verified **2026-09-17**.

Two columns matter when they disagree:

- **Card** — what the model authors recommend.
- **App** — what Tesseract applies (`AgentGenerateParameters` presets in
  `tesseract/Features/Agent/AgentGeneration.swift`, chosen by id prefix).

## At a glance (agent models)

| Catalog id | Base card | Params | Arch (`model_type`) | Layers | Quant | Thinking default | Card output length | App preset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.8-27b-paro` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | `qwen3_5` | 64 | PARO 4-bit, group 128 | on | 131,072 final / 262,144 reasoning | `qwen38Thinking` |
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | `qwen3_5` | 64 | affine 4-bit, group 64 | on | 131,072 final / 262,144 reasoning | `qwen38Thinking` |
| `bonsai-2-27b` | [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) (Qwen3.8-27B) | 27B dense | `prism_hadamard_qwen35` (base `qwen3_5`) | 64 | ternary in affine 2-bit, group 128, Hadamard-rotated | on | 131,072 final / 262,144 reasoning | `qwen38Thinking` |
| `qwen3.6-27b-paro` | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 27B dense | `qwen3_5` | 64 | PARO 4-bit, group 128 | on | 32,768 (81,920 hard problems) | `qwen36Thinking` |
| `qwen3.6-27b` | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 27B dense | `qwen3_5` | 64 | affine 4-bit, group 64 | on | 32,768 (81,920) | `qwen36Thinking` |
| `qwen3.6-35b-a3b-paro` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B total / 3B active, 256 experts (8 routed + 1 shared) | `qwen3_5_moe` | 40 | PARO 4-bit, group 128 | on | 32,768 (81,920) | `qwen36Thinking` |
| `qwen3.6-35b-a3b-ud` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B / 3B active | `qwen3_5_moe` | 40 | Unsloth UD 4-bit, group 64 | on | 32,768 (81,920) | `qwen36Thinking` |
| `qwen3.5-9b-paro` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B dense | `qwen3_5` | 32 | PARO 4-bit, group 128 | on | 32,768 (81,920) | `qwen35` |
| `qwen3.5-4b-paro` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B dense | `qwen3_5` | 32 | PARO 4-bit, group 128 | on | 32,768 (81,920) | `qwen35` |
| `qwen3.5-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B dense | `qwen3_5` | 24 | bf16 | **off** | 32,768 (81,920) | `qwen35` |
| `nanbeige4.2-3b-8bit` | [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) | 4B total / 3B non-embedding, looped (22 layers × 2) | `nanbeige` | 22 | affine 8-bit, group 64 | on | 65,536 agentic / 131,072 reasoning+chat | `nanbeige42` |
| `ornith-9b` | [deepreinforce-ai/Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) | 9B dense (Qwen3.5 post-train) | `qwen3_5` | 32 | affine 6-bit, group 64 | on | card gives none (examples use 512) | `ornith9b` |
| `ornith-35b` | [deepreinforce-ai/Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) | 35B MoE (Qwen3.5-A3B post-train) | `qwen3_5_moe` | 40 | affine 4-bit, group 64 | on | card gives none (examples use 512 / 2,048 tool calls) | `ornith35b` |

**Context window: 262,144 tokens natively for every model above** (checkpoint
`max_position_embeddings`, no `rope_scaling`). The Qwen cards describe YaRN
extension to about 1,000,000 tokens; the app does not enable it and
`ChatSession` / `AgentFactory` pin `contextWindow` at 262,144.

## Sampling, per family

Values are verbatim from the cards. `presence_penalty` is the one that moves
between families and modes, so it gets its own column.

### Qwen3.8 (27B)

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| Thinking (card) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct / non-thinking (card) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |
| **App `qwen38Thinking`** | 1.0 | 0.95 | 20 | 0.0 | none |

- Output length (card, Best Practices): "Reasoning Content: 262,144 tokens.
  Final Response: 131,072 tokens." Clients with one limit use the final
  figure (Pi is set to 131,072).
- Reasoning effort levels `low` / `medium` / `xhigh`, default `xhigh`; the
  server maps OpenAI `reasoning_effort` onto them (ADR-0060). Thinking off
  per request via `chat_template_kwargs: {"enable_thinking": false}`.
- Vision-language checkpoint (`vision_config` present). Both 27B entries
  carry the **Text-Only Override** in the catalog until map #457 lands.
- `generation_config.json`: temperature 1.0, top_p 0.95, top_k 20.

### Bonsai 2 27B (Rotated Ternary Checkpoint of Qwen3.8-27B)

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| Thinking (card) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| **App `qwen38Thinking`** | 1.0 | 0.95 | 20 | 0.0 | none |

- Qwen3.8-27B with ternary language-model weights carried in MLX affine
  2-bit form in a Hadamard-rotated input basis (ADR-0067): `model_type`
  `prism_hadamard_qwen35`, **Base Architecture** `qwen3_5`, and the Model
  Identity family facts key on the base, so every Qwen3.8 rule above applies.
  The vision tower is FP16 and unrotated; the MTP head is dropped, so no
  MTP drafting.
- Chat template byte-identical to Qwen3.8-27B: same tool-call format, thinking
  on by default, `reasoning_effort` `low` / `medium` / `xhigh`. The card
  notes **`low` behaves like `xhigh`** on this checkpoint; the server passes
  it through unchanged.
- `generation_config.json`: temperature 1.0, top_p 0.95, top_k 20 (the
  Qwen3.8 thinking preset).
- Weights ~8.6 GB. Recommended 24 GB+; runs on 16 GB at short context.
- Decode uses the stock 2-bit `quantizedMM` kernels plus a float32
  `hadamardTransform` per shared input: siblings that share a sign vector
  (q|k|v, gate|up, the GDN qkv|z — every group on this pack) are stacked
  into one rotated layer at load, 144 fewer rotations per token than one per
  packed matmul; the fork's tuned matmul kernels are 4-bit only. Measured
  against `qwen3.8-27b` on this machine (48 GB, greedy, 5,963-token prompt,
  192 new tokens, 2026-09-18): plain decode **30.8 vs 22.1 tok/s**, prefill
  27.9 vs 27.9 s (the rotation costs nothing measurable at prefill), peak
  memory 9.8 vs 32.5 GB (the 4-bit run carries the DFlash2 draft, which
  reaches 45.1 tok/s there and is refused here — see Speculative decoding).
  The stacking itself moves nothing measurable (same day, paired against
  the unstacked build, two passes each: 29.0 / 31.0 vs 31.4 / 29.1 tok/s
  median, inside the ±4% drift between passes): decode sits at the
  memory-bandwidth ceiling, so launch count is not the limiter
  (ADR-0067, consequences).
- Agent quick bench at this preset, same day, two passes: 7/14 and 5/14
  scenarios (`qwen3.8-27b`: 5/14), tool accuracy 77% and 74% (86%),
  duplicate tool calls 9.5% and 3.8% (3.4%), 33.3 and 29.6 tok/s on the
  short agent turns. Same band; the runs sample at temperature 1.0, so a
  single pass is noisy. Its misses are behavioral (answering from context
  instead of re-reading, a `write` where an `edit` was required, acting on
  an ambiguous request), not malformed tool calls.

### Qwen3.6 (27B dense, 35B-A3B MoE)

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| Thinking, general (27B card) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking, general (35B-A3B card) | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| Thinking, precise coding (both) | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct / non-thinking (both) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |
| **App `qwen36Thinking`** | 0.6 | 0.95 | 20 | 0.0 | none |

- The app runs the coding profile. No presence penalty on purpose: inside
  `<think>` it drives paraphrase loops instead of preventing repetition.
- Output length: 32,768 for most queries, 81,920 for math/programming
  competition problems.
- Both are vision-language checkpoints. `generation_config.json`: 1.0 /
  0.95 / 20.

### Qwen3.5 (9B, 4B, 2B, 0.8B)

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| Thinking, general text | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| Thinking, vision or coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking, general text (4B/9B "general") | 0.7 | 0.80 | 20 | 0.0 | 1.5 |
| Non-thinking, text (2B/0.8B "text tasks") | 1.0 | 1.00 | 20 | 0.0 | 2.0 |
| Non-thinking, vision-language (2B/0.8B) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |
| Non-thinking, reasoning (4B/9B) | 1.0 | 1.00 | 40 | 0.0 | 2.0 |
| **App `qwen35`** | 1.0 | 0.95 | 20 | 0.0 | 1.5 |

- Thinking on by default for 4B and 9B; **off by default for 2B and 0.8B**
  (enable with `enable_thinking: true`).
- Output length: 32,768 for most queries, 81,920 for hard problems.
- All are vision-language checkpoints. Qwen3.5-0.8B is the proofread model
  (`qwen3.5-0.8b-proofread`), not an agent entry.

### Nanbeige4.2-3B

| Scenario (card) | temperature | top_p | top_k | max_new_tokens |
| --- | --- | --- | --- | --- |
| Agentic / tool use | 1.0 | 0.95 | 20 | 65,536 |
| Reasoning and chat | 0.6 | 0.95 | 20 | 131,072 |
| **App `nanbeige42`** | 1.0 | 0.95 | 20 | — |

- Thinking on by default; `preserve_thinking=true` recommended for multi-turn
  tool use. The chat/reasoning profile is offered in the app as the
  `nanbeigeChatReasoning` sampling override.
- `generation_config.json`: 0.6 / 0.95 / 20. Text only.

### Ornith 1.0 (9B dense, 35B MoE)

| Setting | temperature | top_p | top_k | min_p | repetition_penalty |
| --- | --- | --- | --- | --- | --- |
| Card (both sizes) | 0.6 | 0.95 | 20 | — | — |
| Card, reproduce benchmarks | 1.0 | — | — | — | — |
| **App `ornith9b`** | 0.6 | 0.95 | 20 | 0.0 | none |
| **App `ornith35b`** (vendor Terminal-Bench recipe) | 1.0 | 1.0 | 40 | 0.01 | 1.05 |

- Both open a `<think>` block by default. Agentic-coding post-trains of
  Qwen3.5; the 9B card also names Gemma 4 as a base for other family members.
- The 35B recipe's repetition penalty is the one the Qwen3 notes warn can end
  think blocks early; kept by explicit decision.
- The cards give no output-length recommendation; their examples use 512
  (basic) and 2,048 (tool calls). The 35B checkpoint has `vision_config`;
  the 9B does not.

## Speculative decoding

| Catalog id | MTP head (`mtp.*` in checkpoint) | DFlash2 draft |
| --- | --- | --- |
| `qwen3.8-27b` | yes | `qwen3.8-27b-dflash2-draft` |
| `qwen3.8-27b-paro` | **grafted** by `scripts/graft_mtp_head.py` (not in the upstream file) | `qwen3.8-27b-dflash2-draft` |
| `bonsai-2-27b` | dropped from the pack | **refused by identity**: pairs by shape, but measured 0.64× (19.6 vs 30.8 tok/s, 22% acceptance, 2026-09-18) — see below |
| `qwen3.5-2b` | yes | — |
| every other entry | no | — |

- **DFlash2 draft** ([incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)):
  2B params, 5 layers, block size 8 (7 draft tokens per verify),
  target layers 5/19/33/47/61 of a 64-layer target, lossless (greedy output
  matches the target). Card recommends the target's own sampling (1.0 / 0.95 /
  20). The app quantizes it to 4-bit, group 64. Loads only when the target is
  the MLXLLM text class with 64 layers and not a Rotated Ternary Checkpoint
  (`DFlash2Support`).
- **MTP** (ADR-0056): greedy only, block size 4; the drafter borrows the
  target's embedding and head.
- Measured on this machine (quick bench, greedy, 5,976-token prompt,
  192 new tokens, 2026-09-03): PARO AR 21.1 tok/s, DFlash2 bs8 30.8 (53%
  acceptance), bs5 32.7 (62%). Uniform 27B record bs8f 47.9 (ADR-0058).
  Bonsai 2 27B (same bench, 2026-09-18): AR 30.8 tok/s, DFlash2 bs8 19.6
  (22% acceptance, 0.64×) — the draft was distilled against the
  full-precision target, so the pairing is refused at load.

## KV cache compression (KV Scheme, ADR-0083)

Settings → Agent → KV Cache Compression picks a TurboQuant **KV Scheme** for
`qwen3.8-27b` and `qwen3.8-27b-paro` (`KVScheme.supports`); every other model
keeps full precision. Off by default. The prompt prefills at full precision,
the 16 attention layers convert once prefill ends, and the prefix cache
stores them compressed.

| Setting | Scheme | Keys | Values | KV bytes per token (27B) | vs bf16 |
| --- | --- | --- | --- | --- | --- |
| Off | — | bf16 | bf16 | 65,536 | 1× |
| Smallest | `turbo8v4` | 8-bit affine | 4-bit TurboQuant | 25,856 | 2.53× smaller |
| Balanced | `turbo0v4` | bf16 | 4-bit TurboQuant | 41,216 | 1.59× smaller |

Plain decode runs within about 2% of bf16 at 8K to 64K under either scheme
on the 4-bit checkpoint (#603). On the PARO pack, whose decode is bound by
the CPU's per-step dispatch work, `turbo0v4` decodes 2% faster than bf16
at 8K and `turbo8v4` 1% slower with occasional stalls, so Balanced is the
better pick there (`benchmarks/turboquant/2026-10-03/README.md`, loop 2).
DFlash2 speculates over both; MTP does not. Per DFlash2 round at a
29K prompt (greedy, 2026-10-03): bf16 83.9 ms, `turbo8v4` 89.0 ms,
`turbo0v4` 75.7 ms, at the same acceptance.

## Client settings that follow from this

- **Pi** (`~/.pi/agent/models.json`, provider `tesseract`): `contextWindow`
  262144, `maxTokens` 131072 for all three 27B entries (`qwen3.8-27b`,
  `qwen3.8-27b-paro`, `bonsai-2-27b`), `reasoning: true`.
- **Server** (`/v1/chat/completions`): `max_tokens` /
  `max_completion_tokens` pass straight through to generation; there is no
  clamp against the remaining context. `chat_template_kwargs.enable_thinking`
  toggles thinking per request; `reasoning_effort` maps to Qwen3.8 levels.
- **App default** (`AgentGenerateParameters.default`): temperature 0.6,
  top_p 0.95, no penalties, max_tokens 262,144, prefill step 1,024.

## Companion moments (ADR-0080)

Every Companion moment is one generation over the Day Thread on the selected
agent model, at the owner's reasoning effort and sampling preset: on Qwen3.8
the effort is written into the first system block, so one effort for every
moment keeps one cached prefix. Each moment has its own output cap (thinking
included); a reply that hits it falls back to the deterministic card.

| Moment | Output cap (tokens) |
|---|---|
| Morning Plan | 6,000 |
| Breakpoint | 3,000 |
| Triage | 1,024 |
| Evening Wrap-up | 4,000 |
| Night Reflection | 8,000 |

The Day Thread compacts past its ceiling (Settings → Companion, 64k tokens by
default). On the 27B hybrid checkpoints only 16 of 64 layers hold a KV cache,
64 KB per token at full precision, so a 64k-token thread is about 4 GB during
each moment (a full day measured 22.7k tokens, about 1.4 GB).

The MLX buffer pool (`LLMActor.Defaults.cacheLimitMB`, process-wide) keeps at
most 512 MB of freed buffers for reuse: decode's small buffers fit, and a long
prefill's large ones go back to the system. It was 2 GB, which a long prefill
filled on every turn.

## Non-LLM entries

| Catalog id | Repo | Facts from the checkpoint |
| --- | --- | --- |
| `qwen3-embedding-0.6b` | mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ | `qwen3`, 28 layers, context 32,768, 4-bit group 64 |
| `qwen3-tts-voicedesign` | mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit (`ModelDefinition.textToSpeechModelSpec`, the spec `SpeechEngine` loads) | `qwen3_tts`, 6-bit group 64; `generation_config`: temperature 0.9, top_p 1.0, top_k 50, repetition_penalty 1.05 |
| `whisper-large-v3-turbo` / `-compact` | argmaxinc/whisperkit-coreml | CoreML; no generation_config |
| `qwen3.5-0.8b-proofread` | mlx-community/Qwen3.5-0.8B-4bit | see Qwen3.5 above; non-thinking by default, 24 layers |

### Qwen3-TTS VoiceDesign sampling

Every Qwen3-TTS checkpoint ships the same `generation_config.json`, and
Qwen gives no tuning advice beyond it
([research](research/2026-09-27-qwen3-tts-voice-design.md)). App values are
`TTSParameters` in `Vendor/tesseract-speech`, set in Settings > Speech
("Reset to Recommended" restores them). Verified **2026-09-27**.

| Parameter | Card (`generation_config.json`) | App | Settings > Speech |
| --- | --- | --- | --- |
| Talker temperature | 0.9 | 0.9 | Expressiveness |
| Talker top_k | 50 | 50, fixed | — |
| Talker top_p | 1.0 | 1.0 | Top-p |
| Repetition penalty | 1.05, whole request | 1.05, last 64 frames (ADR-0072) | Repetition penalty |
| Code predictor ("sub-talker") temperature | 0.9 | **0.5** (ADR-0072: keeps a voice the same person across passages) | Voice steadiness |
| Code predictor top_k / top_p | 50 / 1.0 | 50 / 1.0, fixed | — |
| `max_new_tokens` | 8192 (Qwen's evals and Space use 2048) | 4096, then at most 6 frames per text token | Longest passage |
| Language | Auto when omitted; set it when known | always named (English by default) | Voices sheet |

Codec frames are 80 ms: 4096 frames is about 5½ minutes of audio in one passage.
Variety between designed voices comes from the description and from
re-rolling takes, not from sampling: Qwen never suggests raising the
temperature for it.

### Qwen3-TTS alignment heads

The talker head whose attention follows the text, which times each spoken
word (ADR-0077). A property of the network, found by scoring every head
against Whisper's word timestamps
([research](research/2026-09-27-word-timing-from-attention.md)). Verified
**2026-09-27**.

| Checkpoint family | Talker | Head (layer, head) | Where it is set |
| --- | --- | --- | --- |
| 1.7B (VoiceDesign, 6- and 8-bit) | 28 layers × 16 heads | 3, 0 | `AlignmentHead.qwen3TTS17B`, set by `TTSModelSpec.voiceDesign17B` |
| 0.6B (CustomVoice, 8-bit) | 28 layers × 16 heads | 6, 5 | `AlignmentHead.qwen3TTS06B` (no spec ships it) |

Voice, speaker, language and precision don't move the head; a different
network does. A checkpoint family added later needs its head found before its
words are timed; until then its segments fall back to spreading characters.

## Keeping this current

Re-verify when a catalog entry is added or a base card changes. The cheap
checks: `config.json` (`max_position_embeddings`, `num_hidden_layers`,
`quantization_config`, `vision_config`), `generation_config.json`, and a
header scan of the shards for `mtp.*` (the rule in
`MTPDrafterSupport.checkpointShipsMTPHead`). The card is the source for
sampling and output length; quote it with the mode it applies to.
