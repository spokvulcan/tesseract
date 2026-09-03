# Model parameters reference

The numbers we keep looking up: context windows, recommended output
lengths, sampling presets, thinking defaults, and what each checkpoint
ships. One row per catalog entry (`tesseract/Features/Models/ModelDefinition.swift`),
sourced from the official model cards and the checkpoint `config.json` /
`generation_config.json` on disk. Last verified **2026-09-03**.

Two columns matter when they disagree:

- **Card** — what the model authors recommend.
- **App** — what Tesseract applies (`AgentGenerateParameters` presets in
  `tesseract/Features/Agent/AgentGeneration.swift`, chosen by id prefix).

## At a glance (agent models)

| Catalog id | Base card | Params | Arch (`model_type`) | Layers | Quant | Thinking default | Card output length | App preset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.8-27b-paro` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | `qwen3_5` | 64 | PARO 4-bit, group 128 | on | 131,072 final / 262,144 reasoning | `qwen38Thinking` |
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense | `qwen3_5` | 64 | affine 4-bit, group 64 | on | 131,072 final / 262,144 reasoning | `qwen38Thinking` |
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
  think blocks early; kept by explicit decision, thinking-loop safeguard armed.
- The cards give no output-length recommendation; their examples use 512
  (basic) and 2,048 (tool calls). The 35B checkpoint has `vision_config`;
  the 9B does not.

## Speculative decoding

| Catalog id | MTP head (`mtp.*` in checkpoint) | DFlash2 draft |
| --- | --- | --- |
| `qwen3.8-27b` | yes | `qwen3.8-27b-dflash2-draft` |
| `qwen3.8-27b-paro` | **grafted** by `scripts/graft_mtp_head.py` (not in the upstream file) | `qwen3.8-27b-dflash2-draft` |
| `qwen3.5-2b` | yes | — |
| every other entry | no | — |

- **DFlash2 draft** ([incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)):
  2B params, 5 layers, block size 8 (7 draft tokens per verify),
  target layers 5/19/33/47/61 of a 64-layer target, lossless (greedy output
  matches the target). Card recommends the target's own sampling (1.0 / 0.95 /
  20). The app quantizes it to 4-bit, group 64. Loads only when the target is
  the MLXLLM text class with 64 layers (`DFlash2Support`).
- **MTP** (ADR-0056): greedy only, block size 4; the drafter borrows the
  target's embedding and head.
- Measured on this machine (quick bench, greedy, 5,976-token prompt,
  192 new tokens, 2026-09-03): PARO AR 21.1 tok/s, DFlash2 bs8 30.8 (53%
  acceptance), bs5 32.7 (62%). Uniform 27B record bs8f 47.9 (ADR-0058).

## Client settings that follow from this

- **Pi** (`~/.pi/agent/models.json`, provider `tesseract`): `contextWindow`
  262144, `maxTokens` 131072 for both 27B entries, `reasoning: true`.
- **Server** (`/v1/chat/completions`): `max_tokens` /
  `max_completion_tokens` pass straight through to generation; there is no
  clamp against the remaining context. `chat_template_kwargs.enable_thinking`
  toggles thinking per request; `reasoning_effort` maps to Qwen3.8 levels.
- **App default** (`AgentGenerateParameters.default`): temperature 0.6,
  top_p 0.95, no penalties, max_tokens 262,144, prefill step 1,024.

## Non-LLM entries

| Catalog id | Repo | Facts from the checkpoint |
| --- | --- | --- |
| `qwen3-embedding-0.6b` | mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ | `qwen3`, 28 layers, context 32,768, 4-bit group 64 |
| `qwen3-tts-voicedesign` | mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 | `qwen3_tts`; `generation_config`: temperature 0.9, top_p 1.0, top_k 50, repetition_penalty 1.05 |
| `whisper-large-v3-turbo` / `-compact` | argmaxinc/whisperkit-coreml | CoreML; no generation_config |
| `qwen3.5-0.8b-proofread` | mlx-community/Qwen3.5-0.8B-4bit | see Qwen3.5 above; non-thinking by default, 24 layers |

## Keeping this current

Re-verify when a catalog entry is added or a base card changes. The cheap
checks: `config.json` (`max_position_embeddings`, `num_hidden_layers`,
`quantization_config`, `vision_config`), `generation_config.json`, and a
header scan of the shards for `mtp.*` (the rule in
`MTPDrafterSupport.checkpointShipsMTPHead`). The card is the source for
sampling and output length; quote it with the mode it applies to.
