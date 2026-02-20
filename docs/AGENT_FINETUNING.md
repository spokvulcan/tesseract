# Agent LLM Benchmark & Tuning

## Model

- **Nanbeige4.1-3B-8bit** (8-bit quantized, ~3.8GB)
- ChatML format with `<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>` special tokens
- 15 tools: time_get, memory_save, memory_search, goal_create, goal_list, goal_update, task_create, task_list, task_complete, habit_create, habit_log, habit_status, mood_log, mood_list, reminder_set

## Benchmark System

Run benchmarks via:

```bash
scripts/bench.sh quick              # Default params, all 7 scenarios
scripts/bench.sh quick S1,S2        # Specific scenarios
scripts/bench.sh full               # Parameter sweep (36 configs)
```

Outputs:
- **Results**: `benchmarks/results/bench_{date}_{sweep}_{hash}.json` (git-tracked)
- **Transcripts**: `/private/tmp/tesseract-debug/benchmark/transcripts/{scenario}.transcript.txt`
- **Log**: `/private/tmp/tesseract-debug/benchmark/latest.log`

Transcripts show the **exact raw ChatML prompt** the model receives (including all `<|im_start|>` tags, tool definitions in `<tools>` block, and generation prompt), plus per-round raw output with `<think>` and `<tool_call>` tags.

### Scenarios

| ID | Name | Turns | Tests |
|----|------|-------|-------|
| S1 | Simple Q&A | 3 | No false tool calls on plain questions |
| S2 | Single tool calls | 5 | Correct tool + args for one tool per turn |
| S3 | Multi-tool sequence | 8 | Chained operations, cross-turn ID references |
| S4 | Long conversation | 50 | Stability, coherence at context boundary |
| S5 | Duplicate detection | 6 | Avoid redundant ops on repeated requests |
| S6 | Context window stress | 6 | Graceful degradation when context drops |
| S7 | Error recovery | 4 | Clarification on ambiguous requests |

### Scoring

Per-turn evaluation:
- **toolsCorrect**: Expected tools called (no more, no fewer)
- **duplicateToolCalls**: Same tool+args called 2+ times in one turn
- **argumentsCorrect**: Required key-value pairs present
- **responseRelevant**: Expected substrings in response
- **noForbiddenTools**: Tools that should NOT have been called

---

## Baseline Results — 2026-02-19

**Hardware**: Mac15,9 (M3 Max), 48GB
**Parameters**: temp=0.6, topP=0.95, no repetition penalty
**Context window**: 20 messages, maxToolRounds=5

| Scenario | Pass | Tool Acc | Dup Rate | Avg tok/s |
|----------|------|----------|----------|-----------|
| S1: Simple Q&A | FAIL | 67% | 0% | 53.7 |
| S2: Single tool calls | FAIL | 100% | 38% | 46.3 |
| S3: Multi-tool sequence | FAIL | 62% | 44% | 44.6 |
| S4: Long conversation (50t) | FAIL | 76% | 37% | 67.6 |
| S5: Duplicate detection | FAIL | 67% | 17% | 64.6 |
| S6: Context window stress | FAIL | 100% | 18% | 77.2 |
| S7: Error recovery | FAIL | 75% | 0% | 76.4 |
| **Aggregate** | **0/7** | **74.7%** | **34.0%** | **64.5** |

**Latency**: p50=37s, p95=121s
**Peak memory**: 16.7GB

### Performance Notes

- tok/s is lower in early turns (~35-45) and recovers to 73-79 in later turns (KV cache warm-up)
- S4 sustained 73-78 tok/s across turns 10-50 with no degradation
- Long think blocks dominate latency — a 420s turn at 46 tok/s means ~19K tokens generated (mostly thinking)

---

## Failure Analysis

### 1. Tool Call Looping (Critical — 34% duplicate rate)

**Affected**: S2-T2 (memory_save 3x), S2-T5 (mood_list 2x), S3-T8 (task_list 2x), S4-T38 (mood_log 4x), S5-T1 (habit_create 2x)

**Pattern**: After a successful tool call, the model re-enters the thinking phase and generates the exact same tool call again instead of proceeding to the response. The cycle repeats 2-4 times.

**Transcript evidence (S2-T2 — `memory_save` called 3x)**:
```
Round 1: <think>The user wants to remember an allergy...</think>
         <tool_call>{"name":"memory_save","arguments":{"fact":"Allergic to peanuts","category":"health"}}</tool_call>
         → Tool returns: "Remembered: Allergic to peanuts (category: health)"

Round 2: <think>Okay, let me see. The user said, 'Remember that I'm allergic to peanuts'.
         I need to figure out which tool to use here.</think>
         <tool_call>{"name":"memory_save","arguments":{"fact":"Allergic to peanuts","category":"health"}}</tool_call>
         → Tool returns: "Remembered: Allergic to peanuts (category: health)"

Round 3: <think>[identical thinking repeated]</think>
         <tool_call>[identical call]</tool_call>
         → Same result

Round 4: Finally generates response text.
```

**Transcript evidence (S4-T38 — `mood_log` called 4x)**:
```
Round 1: Correct thinking, correct tool call → success
Round 2: (empty output), then identical tool_call
Round 3: (empty output), then identical tool_call
Round 4: (empty output), then identical tool_call
Round 5: "Okay, let me try to figure out what's going on here..."
```

**Root cause**: The model doesn't properly advance its internal state after receiving a tool result. Instead of generating a text response, it re-enters tool-calling mode and regenerates identical logic. The empty raw output in S4-T38 suggests the model's text generation stalls, triggering a retry loop.

**Possible mitigations**:
- Add "Do not call the same tool twice with the same arguments" to system prompt
- Add post-tool-result injection: "Tool completed successfully. Now respond to the user."
- Reduce `maxToolRounds` from 5 to 3 (limits damage from loops)
- In `AgentRunner`, detect same tool+args in consecutive rounds and break the loop
- Try `repetitionPenalty=1.05-1.1` to discourage repeating tool call tokens

### 2. Over-Eager Tool Use (S1 — false positive tool calls)

**Affected**: S1-T2 (called `habit_create` on "How do you make scrambled eggs?")

**Transcript evidence**:
```
<think>
The user is asking how to make scrambled eggs, which is a cooking question.
None of the provided tools are designed to provide cooking instructions...
I should answer directly and clearly.
...
Would you like me to help you track this as a habit or set a reminder for cooking?
</think>
<tool_call>{"name":"habit_create",...}</tool_call>
```

**Root cause**: The model's thinking correctly identifies no tools are needed, but then proactively offers to create a habit and actually calls it. The `<tools>` block in the prompt may be too prominent, biasing the model toward tool use even when its reasoning says otherwise.

**Possible mitigations**:
- Strengthen system prompt: "Only use tools when the user explicitly requests an action. Do not proactively create goals, tasks, or habits unless asked."
- Move tool definitions to the end of the system prompt (less visual weight)

### 3. Over-Clarification (S3 — paralysis on non-ambiguous requests)

**Affected**: S3-T1 ("Create a goal: Learn Spanish" → no tool call), S3-T2 ("Add tasks: buy textbook..." → no tool call)

**Transcript evidence (S3-T1 — 392 lines of thinking!)**:
```
<think>
The goal_create tool requires 'name' (provided), others are optional...
But a goal without target date is not useful...
Instructions say: 'Ask clarifying questions when ambiguous'...
[debates for 392 lines whether to call tool or ask for clarification]
...
The assistant should respond with a question to the user.
</think>
[Asks for target date, category, and focus instead of calling goal_create]
```

**Root cause**: The system prompt instruction "Ask clarifying questions when the user's request is ambiguous" is interpreted too broadly. Requests like "Create a goal: Learn Spanish" have a clear name — the model should create the goal with required fields and let optional fields default, not ask about every optional parameter.

**Possible mitigations**:
- Reword clarification instruction: "Ask clarifying questions only when required parameters are missing. If a request provides the required fields, proceed with the action."
- Add examples of when NOT to clarify: "If the user says 'Create a goal: Learn Spanish', create it immediately — don't ask about optional fields."

### 4. Parameter Confusion (S5 — wrong parameter name)

**Affected**: S5-T1 (used `habit_name` instead of `name` for habit_create)

**Transcript evidence**:
```
Round 1: <tool_call>{"name":"habit_create","arguments":{"habit_name":"running","frequency":"daily"}}</tool_call>
         → Error: Missing required argument: name
Round 2: "Wait, the parameters should be 'name', not 'habit_name'."
         <tool_call>{"name":"habit_create","arguments":{"name":"running","frequency":"daily"}}</tool_call>
         → Success
```

**Root cause**: The model confuses parameter names across tools. `habit_log` uses `habit_name` while `habit_create` uses `name`. The model mixes them up, especially when both habit tools appear in the same conversation.

**Possible mitigations**:
- Normalize parameter names across tools (use `name` everywhere, or `habit_name` everywhere)
- Add parameter name reminders in tool descriptions

### 5. Context Loss (S7 — doesn't reuse info from prior turns)

**Affected**: S7-T3 (user says "At 3pm tomorrow" — should combine with T2's "about the meeting")

**Transcript evidence**:
```
Turn 2: User: "Remind me about the meeting" → Model asks for when
Turn 3: User: "At 3pm tomorrow" → Model asks for message AGAIN instead of using
         "about the meeting" from Turn 2
```

**Root cause**: The model treats each turn somewhat independently, not strongly connecting information across turns. When the user provides the time in T3, the model should infer the message from T2's "about the meeting" context.

**Possible mitigations**:
- Add system prompt guidance: "When the user provides additional information in a follow-up message, combine it with context from previous messages to complete the action."

---

## Thinking Token Budget

The model generates extremely long thinking blocks, especially on ambiguous or multi-step requests:

| Scenario | Turn | Think tokens (est) | Latency | Issue |
|----------|------|--------------------|---------|-------|
| S3-T1 | 1 | ~8000 | 147s | 392 lines debating whether to clarify |
| S4-T7 | 7 | ~10000 | 488s | Circular reasoning about task creation |
| S1-T3 | 3 | ~4500 | 99s | Excessive deliberation on "Thanks!" |

Long thinking is the primary latency driver. The model generates at 35-78 tok/s, but when it thinks for 10K tokens, that's 130-285 seconds of wall time.

**Possible mitigations**:
- Cap thinking tokens (if model supports `max_thinking_tokens` or similar)
- System prompt instruction: "Keep your reasoning brief. If the action is clear, proceed without extensive deliberation."

---

## Recommended Next Steps

### Priority 1: System Prompt Tuning
1. Add anti-looping instruction: "After receiving a tool result, respond to the user. Never call the same tool with the same arguments twice."
2. Narrow clarification scope: "Only ask for clarification when required parameters are missing or the request is genuinely unclear. If the user provides enough information to fill required fields, proceed."
3. Discourage proactive tool use: "Only use tools when the user explicitly requests an action."
4. Add brevity instruction for thinking: "Reason concisely. If the action is clear, call the tool immediately."

### Priority 2: Parameter Sweep
Run `scripts/bench.sh full` to test:
- `repetitionPenalty`: 1.05, 1.1 (may reduce tool call looping)
- Lower `temperature`: 0.3 (may reduce false tool calls and over-deliberation)
- Lower `topP`: 0.8 (tighter sampling)

### Priority 3: Agent Loop Improvements
1. **Dedup detection in AgentRunner**: If the model calls the same tool+args as the previous round, inject "You already called this tool. Please respond to the user." instead of executing it again.
2. **Reduce maxToolRounds** from 5 to 3.
3. **Normalize tool parameter names** across habit tools.

### Priority 4: Benchmark Expansion
- Add scenarios for multi-tool calls in a single round (model should batch)
- Add scenarios testing tool result comprehension (use tool output in response)
- Run full parameter sweep and analyze with `scripts/bench_viz.py`

---

## Optimizations Applied — 2026-02-20

### Changes (targeting 7/7 pass rate)

**1. Strip `<think>` blocks from conversation history** (`AgentRunner.swift`)
- `reconstructAssistantMessage` no longer includes `<think>...</think>` in `workingMessages`
- Thinking is still stored in `newMessages` for transcript/UI display
- Impact: Reduces context consumption by 50-80%, removes the "example" of long deliberation that the model copies, fixes S4 degradation (turns 25-50 all fail when context is full of thinking)

**2. Rewrite system prompt with decision framework** (`SystemPromptBuilder.swift`)
- Action-word triggers: "CREATE → call tool", "LIST → call listing tool"
- 7 few-shot examples covering common failure patterns (tool calls, greetings, general questions)
- Explicit thinking budget: "2-3 sentences maximum"
- Follow-up context rule: "combine info from previous messages"
- Replaces vague "ask clarifying questions when ambiguous" that caused over-clarification

**3. Reduce maxToolRounds from 5 to 3** (`AgentRunner.swift`, `BenchmarkRunner.swift`)
- Transcript evidence shows rounds 3-5 are always duplicate attempts
- Limits damage from any remaining looping behavior

**4. Add repetitionPenalty=1.05** (`AgentGeneration.swift`)
- Discourages repeating exact `<tool_call>` token sequences
- Qwen team recommends 1.05 as starting point for tool-calling models

### Expected Impact
- S3 (multi-tool): 62% → ~85%+ (few-shot examples fix paralysis on "Create a goal")
- S4 (long conversation): 66% → ~80%+ (stripped thinking prevents context rot)
- S5 (duplicate detection): 83% → ~90%+ (decision framework + rep penalty)
- S7 (error recovery): 75% → ~90%+ (follow-up context rule + few-shot for "Set a reminder")

---

## File Reference

| File | Purpose |
|------|---------|
| `tesseract/Features/Agent/Benchmark/BenchmarkRunner.swift` | Orchestrator: model loading, scenario execution, report writing |
| `tesseract/Features/Agent/Benchmark/BenchmarkScenario.swift` | 7 scenario definitions with turn expectations |
| `tesseract/Features/Agent/Benchmark/BenchmarkEvaluator.swift` | Per-turn scoring and aggregate computation |
| `tesseract/Features/Agent/Benchmark/BenchmarkReport.swift` | Codable JSON report types |
| `tesseract/Features/Agent/Benchmark/BenchmarkConfig.swift` | CLI args, parameter sweep presets |
| `tesseract/Features/Agent/Benchmark/BenchmarkTranscript.swift` | Per-scenario plain text transcript writer |
| `tesseract/Features/Agent/SystemPromptBuilder.swift` | System prompt assembly |
| `tesseract/Features/Agent/AgentRunner.swift` | Agent loop (generate → tools → re-generate) |
| `scripts/bench.sh` | Build + run + tail convenience script |
| `scripts/bench_viz.py` | Matplotlib charts from JSON results |
| `benchmarks/results/` | Git-tracked JSON reports |
| `benchmarks/transcripts/` | Gitignored, regenerated per run |
