# Agent Clarification System: Research & Proposed Solutions

## Problem Statement

Tesseract Agent receives user input through WhisperKit speech-to-text, which introduces an inherent error layer: phonetically similar words get substituted, words are dropped, and sentences arrive incomplete or garbled. The agent's current architecture is a pure tool-calling loop — the LLM decides what to do based solely on the system prompt and conversation context. There is **no mechanism** to:

1. Detect that a request is ambiguous or underspecified
2. Ask the user for clarification before acting
3. Confirm high-risk actions before execution
4. Correct likely ASR transcription errors using context

The result: the agent acts on vague or misinterpreted commands, producing incorrect outcomes that erode user trust. A user who says "move the report file" gets a file moved to an arbitrary location instead of being asked "which report file, and where should I move it?"

### The STT Error Amplification Problem

```
User speaks → WhisperKit transcribes (with errors) → LLM interprets (compounds errors) → Agent acts (wrong action)
```

Each stage amplifies uncertainty. A small ASR error ("rime" instead of "README") becomes a confident wrong action because the LLM fills in gaps rather than flagging them. Research confirms this: LLMs are systematically overconfident and default to non-interactive behavior — they act rather than ask (AMBIG-SWE, ICLR 2026).

### Current Architecture Gaps

The agent loop (`AgentLoop.swift`) runs until no more tool calls are generated. The extension system provides hooks at `input`, `toolCall`, `toolResult`, and `beforeAgentStart` — but none are currently used for clarification. The steering queue infrastructure exists (`Agent.pushSteering()`) but is never populated. All the plumbing is there; the logic is missing.

---

## Research Findings

### 1. Human Decision-Making Frameworks

#### 1.1 OODA Loop (Observe → Orient → Decide → Act)

**Origin**: Military strategist Colonel John Boyd. Designed for rapid adaptive decision-making in dynamic, adversarial environments.

**The Four Phases**:
- **Observe**: Gather raw data (ASR output, confidence scores, N-best hypotheses)
- **Orient**: Analyze and contextualize — parse intent, check against known patterns, assess ambiguity using conversation history and app state. Boyd considered this the most critical phase.
- **Decide**: Select a course of action: execute, clarify, confirm, or reject
- **Act**: Execute the decision, then loop back to Observe the outcome

**Key Insight** (Schneier, 2025): AI agents' OODA loops operate on untrusted inputs at every stage. ASR output is noisy, the orient phase relies on potentially biased model weights, and the decide phase lacks ground truth. The security trilemma applies: "Fast, smart, secure; pick any two."

**Applicability**: Maps directly to a voice command pipeline. The critical gap in Tesseract is the Orient phase — the agent skips straight from Observe to Act.

Sources:
- [OODA Loop: AI-Driven Decision Framework](https://www.lowtouch.ai/ooda-loop-ai-decision-framework/)
- [Schneier: Agentic AI's OODA Loop Problem](https://www.schneier.com/blog/archives/2025/10/agentic-ais-ooda-loop-problem.html)
- [Harnessing the OODA Loop for Agentic AI](https://labs.sogeti.com/harnessing-the-ooda-loop-for-agentic-ai-from-generative-foundations-to-proactive-intelligence/)

#### 1.2 Recognition-Primed Decision (RPD) Model

**Origin**: Gary Klein (late 1980s). Core of Naturalistic Decision Making (NDM) — how experts actually decide under time pressure.

**How It Works**: Experts don't compare options analytically. Instead:
1. **Recognize** the situation as typical (pattern matching against experience)
2. **Generate** a plausible course of action (first one that fits the pattern)
3. **Mentally simulate** the action to check for problems
4. If simulation reveals issues, **modify** the plan or consider the next option
5. If simulation succeeds, **execute**

**Three Variations**:
- **Variation 1** (Simple match): Situation is typical, response is obvious → act immediately. Example: "Open Safari" → execute.
- **Variation 2** (Diagnose): Need to assess which pattern applies → ask diagnostic question. Example: "Move that file" → which file?
- **Variation 3** (Evaluate): Mentally simulate consequences → confirm if high-risk. Example: "Delete everything in this folder" → confirm scope.

**Key Limitation**: RPD fails on novel or misidentified situations. ASR errors frequently create these — the agent "recognizes" a wrong pattern because the transcription is wrong.

**Applicability**: Directly implementable as a tiered decision system. High confidence → act. Ambiguous → diagnose. High risk → simulate and confirm. Low confidence → full clarification.

Sources:
- [RPD Model - Mindtools](https://www.mindtools.com/a5wclfo/the-recognition-primed-decision-rpd-process/)
- [RPD for Artificial Agents - SpringerOpen](https://hcis-journal.springeropen.com/articles/10.1186/s13673-019-0197-2)

#### 1.3 Cynefin Framework

**Origin**: Dave Snowden (1999), IBM Global Services. A sense-making framework that classifies situations into domains, each requiring a different decision strategy.

**Five Domains**:

| Domain | Characteristics | Response Strategy |
|--------|----------------|-------------------|
| **Clear** | Known knowns, cause-effect obvious | Sense → Categorize → Respond (best practice) |
| **Complicated** | Known unknowns, expert analysis needed | Sense → Analyze → Respond (good practice) |
| **Complex** | Unknown unknowns, emergent patterns | Probe → Sense → Respond (emergent practice) |
| **Chaotic** | No discernible cause-effect | Act → Sense → Respond (novel practice) |
| **Disorder** | Can't tell which domain applies | Break down into parts |

**Mapping to Voice Commands**:
- **Clear**: "Set a timer for 5 minutes" → all parameters present, well-formed → execute
- **Complicated**: "Schedule a meeting with the team" → intent clear, details missing → analyze what's needed, ask specific questions
- **Complex**: "Help me organize my project" → open-ended, no single answer → probe to understand scope
- **Chaotic**: User frustrated, repeating commands, system misunderstanding → acknowledge the problem, reset, ask what they need
- **Disorder**: Can't classify the utterance → break it down

**Applicability**: The most powerful meta-framework. Classify each command, then apply the corresponding strategy. Prevents both over-asking (on clear commands) and under-asking (on complex ones).

Sources:
- [Cynefin Framework - Wikipedia](https://en.wikipedia.org/wiki/Cynefin_framework)
- [Cynefin Framework - Untools](https://untools.co/cynefin-framework/)

#### 1.4 DECIDE Model (Medicine)

**Six Steps**: Define the problem → Establish criteria → Consider alternatives → Identify best alternative → Develop plan → Evaluate outcome.

**Applicability**: The "Establish criteria" step maps to defining what information is needed before acting. The "Consider alternatives" step maps to generating multiple interpretations of an ambiguous command. Useful as the agent's internal reasoning template.

Sources:
- [DECIDE Model - PubMed](https://pubmed.ncbi.nlm.nih.gov/32701610/)

#### 1.5 PDCA (Plan-Do-Check-Act)

**Applicability**: The emphasis on "test before full implementation" maps to a confirmation pattern — propose interpretation before executing. Also useful at the system improvement level: track which commands are frequently ambiguous and build specialized handling.

Sources:
- [PDCA Cycle - ASQ](https://asq.org/quality-resources/pdca-cycle)

---

### 2. AI Agent Clarification Research

#### 2.1 AMBIG-SWE: LLMs Default to Non-Interactive Behavior (ICLR 2026)

**Finding**: Even with explicit prompting ("proactively seek clarification"), LLMs struggle to distinguish ambiguous from well-specified inputs. Only Claude Sonnet 3.5 achieved 84% accuracy at this task; other models ranged 48-69%.

**Three prompting levels tested**: neutral ("may ask"), moderate ("verifies completeness"), strong ("proactively seeks clarification"). Strong prompting helped but improvement was model-dependent.

**Critical result**: Interactivity (actually asking questions) boosted performance on underspecified inputs by up to **74%**. But the agent must be architecturally forced to be interactive — it won't do it spontaneously.

**Implication**: System prompt instructions alone are insufficient. The clarification behavior must be enforced through architecture (extensions, tools, pre-processing), not just prompt engineering.

Source: [AMBIG-SWE - arXiv](https://arxiv.org/html/2502.13069v1)

#### 2.2 Clarify-or-Answer (CoA): RL-Optimized Clarification

**Architecture**: Three modular components:
1. **Controller**: Binary classifier — "Answer" or "Clarify"
2. **Clarifier**: Generates a single focused clarification question
3. **Answerer**: Produces final answer with or without clarification context

**GRPO-CR** (RL optimization) uses five reward signals for clarification quality:
1. Format: Well-formed (question mark, under 50 words)
2. Focused relevance: Targets missing contextual factors
3. Ambiguity resolution: Answer would change the outcome
4. Novelty: Not a trivial rephrasing of the input
5. Ground-truth alignment

**Result**: +15.3 percentage points in end-to-end accuracy over baselines.

**Implication**: The controller/clarifier separation is a clean architecture. A dedicated "should I clarify?" gate upstream of the main agent intercepting ambiguous commands.

Source: [CoA - arXiv](https://arxiv.org/html/2601.16400)

#### 2.3 MAC: Multi-Agent Clarification Framework

**Ambiguity taxonomy**: Parameter underspecification (missing info) vs. value ambiguity/vagueness (subjective terms).

**Architecture**: Supervisor agent (high-level ambiguity) + domain expert agents (domain-specific underspecification).

**Key result**: Clarification at both levels increased task success by 7.8% AND **reduced** average dialogue turns (6.53 → 4.86). Asking upfront saves time by preventing failed attempts and backtracking.

**Implication**: Asking clarifying questions early actually reduces total interaction time.

Source: [MAC Framework - arXiv](https://arxiv.org/abs/2512.13154)

#### 2.4 A2H: Agent-to-Human Interaction Protocol

**Four interaction primitives**:
1. **PERMISSION**: Authorization for high-risk actions (hard blocking, Boolean)
2. **CLARIFICATION**: Multiple valid paths exist (soft blocking, user selects)
3. **SOLICITATION**: Missing information needed (soft blocking, structured data)
4. **NOTIFICATION**: Informational updates (non-blocking)

**Three trigger conditions**:
1. **Ambiguity**: Confidence below threshold → structured solicitation
2. **Criticality**: Irreversible action → require permission
3. **Resource exhaustion**: Agent loops or exceeds max steps

**Design principle**: "Agent-to-human communication must be minimized to avoid cognitive overload."

**Implication**: Concrete taxonomy of when and how to interrupt. The three triggers are directly implementable.

Source: [A2H Protocol - arXiv](https://arxiv.org/html/2602.15831v1)

#### 2.5 Bayesian Information-Seeking ("Shoot First, Ask Questions Later?")

**Framework**: Rank questions by Expected Information Gain (EIG) — how much uncertainty each question reduces. The optimal question is equally likely to get "yes" or "no" (maximum entropy).

**Human behavior finding**: Skilled decision-makers don't ask ALL questions first. They ask the single most informative question, then reassess. More questions correlates with success (rho=0.684), but targeted questions beat blanket interrogation.

**Result**: Bayesian enhancements enabled even weak LMs to reach superhuman performance (81-82% win rate).

**Implication**: Don't blanket-ask. Calculate which single question would most reduce uncertainty. Ask that one, reassess, repeat if needed.

Source: [Shoot First, Ask Questions Later - arXiv](https://arxiv.org/html/2510.20886)

#### 2.6 LLM Uncertainty Estimation

**Key finding**: LLMs are systematically overconfident when verbalizing uncertainty. They imitate human confidence patterns rather than accurately reflecting their actual uncertainty.

**Practical detection**: Use structural signals (ASR confidence, slot completeness, parameter count) rather than LLM self-reported confidence. Semantic entropy (uncertainty over meanings, not tokens) is more reliable than sampling-based approaches.

Sources:
- [Survey of Uncertainty Estimation in LLMs - arXiv](https://arxiv.org/abs/2410.15326)
- [Can LLMs Faithfully Express Uncertainty? - arXiv](https://arxiv.org/abs/2405.16908)

---

### 3. Conversational AI & Disambiguation

#### 3.1 Dialogue State Tracking (Slot-Filling)

**Core concept**: Task-oriented dialogue systems maintain a "belief state" — a structured assignment of values to predefined slots. Each turn extracts user goals and fills slots. Missing slots trigger targeted questions.

**Example**: User says "open that document" → slots: `{action: "open", target_type: "document", target_name: ???, target_location: ???}`. Missing slots trigger: "Which document?"

**Applicability**: Define slot schemas for common command types. The belief state carries forward across turns, so earlier conversation can fill slots automatically.

Sources:
- [Dialogue State Tracking - Emergent Mind](https://www.emergentmind.com/topics/dialogue-state-tracking)

#### 3.2 Grounding in Communication (Clark & Brennan, 1991)

**Core principle**: Communication requires collaborative establishment of "common ground." Both parties must coordinate until mutual understanding is achieved. If the grounding criterion is not met, repair is needed.

**Voice-only is harder**: The cost of grounding varies by medium. Voice-only (no screen) requires MORE proactive grounding than voice+visual. The agent should echo its understanding before acting on ambiguous commands.

Source: [Grounding in Communication - Clark & Brennan](https://web.stanford.edu/~clark/1990s/Clark,%20H.H.%20_%20Brennan,%20S.E.%20_Grounding%20in%20communication_%201991.pdf)

#### 3.3 Conversational Repair

**Four repair types**:
1. Self-initiated self-repair (SISR): Speaker corrects own talk
2. Self-initiated other-repair (SIOR): Speaker signals problem, listener fixes
3. **Other-initiated self-repair (OISR)**: Listener flags problem ("huh?"), speaker corrects — *most relevant for agent*
4. Other-initiated other-repair (OIOR): Listener explicitly corrects

**Third-position repair**: "The last defense of intersubjectivity" — when the speaker notices from the listener's response that they were misunderstood, and corrects. The agent should implement this: if the user's next utterance suggests the action was wrong, proactively offer correction.

**User preference**: Users most prefer clarification requests and explicit acknowledgments of misunderstanding. They least prefer being redirected to web searches.

Sources:
- [Repair: Interaction and Cognition - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6849777/)
- [Dialogue Repair in Virtual Assistants - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11586770/)

---

### 4. ASR Error Handling

#### 4.1 Confidence-Based Error Detection

Transformer-based confidence modules generate word-level predictions (0-1). F1 > 0.98 for error detection, 21% error rate reduction. But confidence alone produces many false positives — contextual understanding is also needed.

Source: [Confidence-Guided ASR Error Correction - arXiv](https://arxiv.org/html/2407.12817v1)

#### 4.2 N-Best List Correction

Using 5-10 ASR hypotheses instead of just the top one yields up to 38.4% WER reduction. LLM-based selection/correction on N-best lists achieves 25-32% improvement. Constrained decoding prevents hallucinated corrections.

**Applicability**: If WhisperKit can provide multiple hypotheses, pass all to a pre-processing layer. When hypotheses diverge significantly, flag the utterance as uncertain.

Source: [ASR Error Correction using LLMs - arXiv](https://arxiv.org/html/2409.09554v2)

#### 4.3 Contextual Correction

The agent has context that WhisperKit lacks: available tools, existing file names, recent conversation, user history. This context can bias ASR correction. Example: "open the rime file" → context knows "README" exists → correct before processing.

Source: [Contextual ASR Error Handling - ACL](https://aclanthology.org/2025.coling-industry.32.pdf)

---

### 5. Cognitive Psychology of Questioning

#### 5.1 Socratic Questioning (Six Types)

1. **Clarification**: "What do you mean by...?" — resolve ambiguous terms
2. **Probing assumptions**: "What are you assuming?" — surface hidden assumptions
3. **Probing evidence**: "Which file are you referring to? I see three .txt files." — ground with evidence
4. **Exploring viewpoints**: "I can either move or copy. Which?" — present alternatives
5. **Probing consequences**: "This will permanently delete 47 files. Sure?" — flag consequences
6. **Meta-questions**: "Is 'which file?' the right question, or should I ask 'what are you trying to accomplish?'" — agent self-reflection

Sources:
- [6 Types of Socratic Questions - UMich](https://websites.umich.edu/~elements/probsolv/strategy/cthinking.htm)

#### 5.2 Information Gap Theory (Loewenstein, 1994)

The agent should model its own "information gaps" explicitly: parse the command into known and unknown components, rank unknowns by importance, and ask about the most critical gap. Frame questions to invoke the user's awareness: "I need to know which folder you mean before I can proceed."

Source: [Information Gap Theory - Psychology Fanatic](https://psychologyfanatic.com/information-gap-theory/)

#### 5.3 5W1H Framework + MoSCoW Prioritization

**5W1H**: Who, What, When, Where, Why, How — six dimensions to check for completeness.

**MoSCoW for missing parameters**:
- **Must-have**: Without this, execution is impossible or dangerous → ask
- **Should-have**: Improves accuracy, has reasonable default → use default
- **Could-have**: Nice to know, can be assumed → assume
- **Won't-have**: Don't ask → ignore

Only ask clarifying questions for Must-have missing parameters. Use sensible defaults for the rest.

Sources:
- [MoSCoW Method - Wikipedia](https://en.wikipedia.org/wiki/MoSCoW_method)

---

### 6. Ambiguity Taxonomy for Voice Commands

| Type | Example | Detection Signal | Response |
|------|---------|-----------------|----------|
| **ASR Error** | "Open the rime file" (README) | Low word confidence, phonetic similarity to known entity | Contextual correction or confirm |
| **Referential** | "Delete that file" | Missing referent, deictic pronoun ("that", "this", "it") | Ask which, or infer from recency |
| **Lexical** | "Run the program" (execute vs. test) | Multiple valid intents, close confidence scores | Present alternatives |
| **Underspecification** | "Set up a meeting" | Missing required slots (when, with whom) | Ask for must-have slot |
| **Vagueness** | "Make it better" | Subjective predicate, no measurable target | Probe for specific criteria |
| **Scope** | "Delete all the old files" | Quantifier + vague qualifier ("old") | Confirm scope and threshold |
| **Speech Act** | "Can you open Safari?" | Indirect speech act (question vs. request) | Default to request interpretation |

---

## Proposed Solutions

Six concrete implementations, ordered from simplest to most sophisticated. Each maps to existing Tesseract extension hooks.

### Solution 1: Clarification Skill (Prompt-Only)

**Complexity**: Low
**Latency Impact**: None (prompt only)
**Extension Hook**: `resourcesDiscover` (provides skill file path)
**Mechanism**: A skill markdown file that teaches the agent to follow a clarification protocol before acting.

**How it works**: A skill file (e.g., `skills/clarification/SKILL.md`) is discovered by `SkillRegistry` and listed in the system prompt. When the agent reads it (via the `read` tool), it gets detailed instructions on:
- Parsing commands into known/unknown components (Information Gap Theory)
- Classifying requests using the Cynefin mapping (Clear/Complicated/Complex)
- Asking the single most informative question before acting (Bayesian EIG)
- Using 5W1H to identify missing dimensions
- Only asking about Must-have parameters (MoSCoW)

**Skill content outline**:
```markdown
---
name: clarification-protocol
description: Use this skill before acting on any user request to ensure you have all needed information
---

# Clarification Protocol

Before taking action on a user request, follow this protocol:

## Step 1: Classify the Request
- CLEAR: All parameters present, intent unambiguous, low-risk → act immediately
- COMPLICATED: Intent clear but details missing → ask ONE targeted question about the most critical gap
- COMPLEX: Intent unclear or open-ended → ask what the user is trying to accomplish
- CHAOTIC: Request unintelligible → ask the user to rephrase

## Step 2: Check Required Information (5W1H)
For the identified intent, check which dimensions are specified:
- WHAT: What action to perform?
- WHERE: What target (file, folder, app)?
- WHEN: Immediate or scheduled?
- HOW: What method or approach?

Only ask about dimensions that are MUST-HAVE (without which you cannot act correctly).

## Step 3: Confirm High-Risk Actions
Before destructive actions (delete, overwrite, bulk operations), state what you will do and ask for confirmation.
Include the scope and consequences.

## Step 4: Act and Verify
After acting, briefly state what you did. If the user corrects you, acknowledge and fix immediately.
```

**Pros**: Zero code changes, easy to iterate, can be tested immediately.
**Cons**: Relies on LLM compliance (AMBIG-SWE shows this is unreliable), no enforcement, can be ignored by the model.

---

### Solution 2: Input Gate Extension (Pre-Processing)

**Complexity**: Medium
**Latency Impact**: Low (one LLM call for classification)
**Extension Hook**: `input` event handler
**Mechanism**: An extension that intercepts every user message via the `input` event. It runs a lightweight classification pass to detect ambiguity, then either passes the message through (clear) or injects a clarification prompt.

**How it works**:
1. Extension registers an `input` event handler
2. When a user message arrives, the handler analyzes it:
   - Check for deictic pronouns ("that", "this", "it") without clear referents
   - Check for vague quantifiers ("all", "some", "the old ones")
   - Check for missing action targets (verb without object)
   - Check for subjective predicates ("better", "cleaner", "nice")
3. If ambiguity detected, return `.modifyInput()` that appends a clarification instruction:

```
[SYSTEM NOTE: The user's request may be ambiguous. Before acting, identify what
information is missing and ask ONE targeted clarification question. Do not proceed
with tool calls until you have the needed information.]
```

4. If clear, return `.none` (pass through)

**Detection heuristics** (no LLM needed):
- Regex for deictic pronouns without antecedent in recent context
- Missing noun phrases after action verbs
- Presence of vague qualifiers ("some", "a few", "the old")
- Very short messages (< 5 words) for non-trivial intents
- Presence of hedging language ("maybe", "I think", "kind of")

**Pros**: Architecturally enforced (not just prompt), low overhead (regex-based), the LLM gets an explicit instruction to clarify.
**Cons**: Rule-based detection has false positives/negatives, doesn't understand semantic ambiguity.

---

### Solution 3: Tool Gate Extension (Action Confirmation)

**Complexity**: Medium
**Latency Impact**: Medium (blocks until user responds)
**Extension Hook**: `toolCall` event handler
**Mechanism**: An extension that intercepts tool calls via the `toolCall` event. For destructive or high-risk tools, it blocks execution and asks for confirmation. For ambiguous tool arguments, it flags the issue.

**How it works**:
1. Extension registers a `toolCall` event handler
2. When a tool call arrives, classify its risk level:
   - **High risk**: `write` (overwriting existing file), `edit` (modifying code) → block with confirmation
   - **Medium risk**: `write` (new file), `edit` (adding content) → allow with notification
   - **Low risk**: `read`, `ls` → pass through (read operations are safe)
3. For high-risk calls, return `.block(reason:)` with a message:

```
"Before writing to [filename], please confirm: Is this what you intended?
The file currently contains [X lines]. This action will [overwrite/modify] it."
```

4. The block reason becomes a tool result message in context, prompting the agent to ask the user

**Risk classification criteria**:
- File exists + write tool = HIGH (destructive overwrite)
- Edit to file not recently read = HIGH (blind edit)
- Multiple file operations in sequence = MEDIUM (bulk operation)
- Operations on system/config files = HIGH
- All read operations = LOW

**Pros**: Catches dangerous actions regardless of prompt, works as a safety net, can be tuned per-tool.
**Cons**: Can be annoying for routine operations, doesn't help with ambiguous intent (only catches at execution time).

---

### Solution 4: Structured Clarification Tool

**Complexity**: Medium-High
**Latency Impact**: Medium (adds clarification turns)
**Extension Hook**: Extension `tools` dictionary (custom tool)
**Mechanism**: A custom tool called `clarify` that the agent can call to ask the user a structured question. The tool presents options and collects the response.

**How it works**:
1. Extension provides a `clarify` tool via its `tools` dictionary
2. The tool accepts parameters:
   - `question`: The clarification question (string)
   - `options`: Optional list of choices (array of strings)
   - `context`: Why the agent is asking (string)
3. Tool execution presents the question to the user (via UI or TTS)
4. Tool result contains the user's response
5. The system prompt instructs the agent to use this tool when uncertain

**Tool definition**:
```
name: "clarify"
description: "Ask the user a clarification question when the request is ambiguous or missing required information. Use this before taking action when you're unsure about the user's intent."
parameters:
  question: string (required) — the specific question to ask
  options: [string] (optional) — possible answers to choose from
  context: string (optional) — brief explanation of why you're asking
```

**System prompt addition** (via `beforeAgentStart` or APPEND_SYSTEM.md):
```
You have a `clarify` tool. Use it BEFORE taking action when:
- The user's request could be interpreted multiple ways
- Required information is missing (which file, what format, where to save)
- The action is destructive or irreversible
- You're less than 80% confident in your interpretation

Ask ONE focused question at a time. Prefer providing options when possible.
```

**Pros**: Gives the agent an explicit mechanism to ask questions (not just text), structured format, options reduce user effort.
**Cons**: Requires UI changes to handle the tool output, agent may not call it without strong prompting, adds a round-trip per clarification.

---

### Solution 5: OODA Pre-Processor Extension (Full Pipeline)

**Complexity**: High
**Latency Impact**: Medium (one pre-processing LLM call)
**Extension Hooks**: `input` + `beforeAgentStart`
**Mechanism**: A full OODA-inspired pre-processing pipeline that analyzes every input, classifies it, and either passes it through with enriched context or triggers a clarification flow.

**How it works**:

**Phase 1 — Observe**: Capture the raw ASR output. If WhisperKit provides confidence scores or N-best alternatives, capture those too.

**Phase 2 — Orient**: Run a lightweight LLM analysis (or the main model with a focused prompt) to produce a structured assessment:
```json
{
  "transcription": "move the rime file to the desktop",
  "likely_corrections": [{"rime": "README", "confidence": 0.85}],
  "intent": "file_move",
  "intent_confidence": 0.78,
  "identified_parameters": {"action": "move", "destination": "desktop"},
  "missing_parameters": ["source_file"],
  "ambiguity_type": "asr_error + referential",
  "risk_level": "medium",
  "cynefin_domain": "complicated"
}
```

**Phase 3 — Decide**: Based on the assessment:
- **Clear** (confidence > 0.9, all params present, low risk): Pass through with corrected transcription
- **Complicated** (confidence > 0.7, missing params): Inject a targeted clarification instruction
- **Complex** (confidence < 0.7, open-ended): Inject a probing instruction
- **Chaotic** (confidence < 0.4): Ask to rephrase

**Phase 4 — Act**: The processed input enters the main agent loop, either as-is or with injected guidance.

**Implementation via extensions**:
- `beforeAgentStart`: Inject the OODA assessment instructions into the system prompt
- `input`: Intercept each message, run the Orient phase, modify the message with enriched context

**Context enrichment** appended to user message:
```
[ORIENT ANALYSIS]
Likely intent: file_move (confidence: 0.78)
Possible ASR correction: "rime" → "README" (phonetically similar, file exists)
Missing: which file to move (source_file)
Risk: MEDIUM (file will be relocated)
Recommendation: Ask which file before proceeding
```

**Pros**: Most thorough approach, catches ASR errors, classifies ambiguity systematically, provides rich context to the agent.
**Cons**: Highest complexity, adds latency (extra LLM call), may over-process simple commands.

---

### Solution 6: Adaptive Clarification Skill with Slot Templates

**Complexity**: Medium-High
**Latency Impact**: None (prompt-based, but with structured templates)
**Extension Hook**: `resourcesDiscover` (skills) + APPEND_SYSTEM.md
**Mechanism**: A skill that provides the agent with structured slot templates for common command categories. Each template defines required/optional parameters and default values, enabling systematic gap detection.

**How it works**: The skill file contains a library of command templates:

```markdown
---
name: adaptive-clarification
description: Provides structured templates for common commands to detect missing information
---

# Command Templates

When a user request matches one of these categories, check which slots are filled.
Ask about MUST-HAVE slots that are empty. Use defaults for SHOULD-HAVE slots.

## File Operations
- action: MUST (create/read/edit/delete/move/copy/rename)
- target_file: MUST (which file or pattern)
- destination: MUST for move/copy (where to put it)
- content: MUST for create/edit (what to write)
- confirmation: MUST for delete (explicit yes)
- format: SHOULD (defaults to current format)
- backup: SHOULD (defaults to no)

## Search Operations
- query: MUST (what to search for)
- scope: SHOULD (defaults to current directory)
- file_types: COULD (defaults to all)
- case_sensitive: WON'T (defaults to insensitive)

## System Commands
- action: MUST (open/close/restart/install)
- target: MUST (which app, service, or package)
- version: COULD (defaults to latest)

## Communication
- action: MUST (send/read/reply/draft)
- recipient: MUST for send/reply
- content: MUST for send/draft
- channel: SHOULD (defaults to last used)
- urgency: COULD (defaults to normal)

# Clarification Rules
1. Count the MUST-HAVE slots that are empty
2. If 0 empty: proceed
3. If 1 empty: ask about it directly
4. If 2+ empty: ask about the most critical one first
5. Never ask about more than one thing at a time
6. For destructive actions, ALWAYS confirm even if all slots are filled
```

**Pros**: No code changes, provides structured reasoning, templates can be customized per-user, easy to add new command categories.
**Cons**: Templates are static (can't adapt to novel commands), relies on LLM compliance, requires maintenance as capabilities grow.

---

## Comparison Matrix

| Solution | Complexity | Latency | Enforcement | ASR Handling | Scope |
|----------|-----------|---------|-------------|--------------|-------|
| 1. Clarification Skill | Low | None | Weak (prompt) | None | All commands |
| 2. Input Gate Extension | Medium | Low | Medium (code) | Basic (regex) | All commands |
| 3. Tool Gate Extension | Medium | Medium | Strong (blocks) | None | Tool calls only |
| 4. Clarify Tool | Medium-High | Medium | Medium (tool) | None | Agent-initiated |
| 5. OODA Pre-Processor | High | Medium | Strong (pipeline) | Strong (LLM) | All commands |
| 6. Slot Templates Skill | Medium-High | None | Weak (prompt) | None | Templated commands |

### Recommended Testing Order

1. **Start with Solution 1** (Clarification Skill) — zero code, immediate test of whether prompt-based guidance helps at all
2. **Add Solution 3** (Tool Gate) — catches dangerous actions as a safety net, independent of Solution 1
3. **Try Solution 4** (Clarify Tool) — gives the agent an explicit mechanism to ask questions
4. **If prompt-based insufficient**, build Solution 2 (Input Gate) — architectural enforcement
5. **If ASR errors are the dominant problem**, build Solution 5 (OODA Pre-Processor)
6. **Refine with Solution 6** (Slot Templates) — once you know which command categories are most problematic

### Combinations

The solutions are composable:
- **1 + 3**: Skill teaches the agent to clarify; Tool Gate catches anything it misses (good baseline)
- **2 + 3 + 4**: Input Gate pre-screens; Tool Gate blocks dangerous actions; Clarify Tool gives agent a structured way to ask (comprehensive medium-complexity setup)
- **5 + 3 + 4**: OODA pre-processes everything; Tool Gate is the safety net; Clarify Tool for structured interaction (maximum coverage, highest complexity)

---

## Key Design Principles

1. **Ask early, ask once**: Clarifying upfront reduces total interaction time (MAC: fewer turns overall)
2. **Ask the minimum**: Only ask about must-have information; use defaults for everything else
3. **Be specific, not generic**: "Which folder?" beats "Could you clarify?"
4. **Present alternatives when possible**: "Did you mean A or B?" is faster than open-ended questions
5. **Support repair**: Always allow correction after action ("No, I meant...")
6. **Don't trust LLM self-confidence**: Use structural signals (slot completeness, ASR confidence) over self-reported certainty
7. **Context is king**: Use conversation history, app state, and available entities to resolve ambiguity before asking
8. **Minimize interruption**: Only interrupt when the cost of getting it wrong exceeds the cost of asking (A2H principle)

---

## All Sources

### Decision-Making Frameworks
- [OODA Loop AI Decision Framework](https://www.lowtouch.ai/ooda-loop-ai-decision-framework/)
- [Schneier: Agentic AI's OODA Loop Problem](https://www.schneier.com/blog/archives/2025/10/agentic-ais-ooda-loop-problem.html)
- [Harnessing the OODA Loop for Agentic AI](https://labs.sogeti.com/harnessing-the-ooda-loop-for-agentic-ai-from-generative-foundations-to-proactive-intelligence/)
- [PDCA Cycle - ASQ](https://asq.org/quality-resources/pdca-cycle)
- [DECIDE Model - PubMed](https://pubmed.ncbi.nlm.nih.gov/32701610/)
- [RPD Model - Mindtools](https://www.mindtools.com/a5wclfo/the-recognition-primed-decision-rpd-process/)
- [RPD for Artificial Agents - SpringerOpen](https://hcis-journal.springeropen.com/articles/10.1186/s13673-019-0197-2)
- [Cynefin Framework - Wikipedia](https://en.wikipedia.org/wiki/Cynefin_framework)
- [Cynefin Framework - Untools](https://untools.co/cynefin-framework/)

### AI Agent Clarification Research
- [AMBIG-SWE: Interactive Agents (ICLR 2026)](https://arxiv.org/html/2502.13069v1)
- [Clarify-or-Answer (CoA)](https://arxiv.org/html/2601.16400)
- [Agent-Based Detection of Incompleteness/Ambiguity](https://arxiv.org/html/2507.03726)
- [MAC: Multi-Agent Clarification](https://arxiv.org/abs/2512.13154)
- [A2H: Agent-to-Human Protocol](https://arxiv.org/html/2602.15831v1)
- [Shoot First, Ask Questions Later](https://arxiv.org/html/2510.20886)
- [Deciding Whether to Ask Clarifying Questions](https://arxiv.org/abs/2109.12451)

### Uncertainty Estimation
- [Survey of Uncertainty Estimation in LLMs](https://arxiv.org/abs/2410.15326)
- [Can LLMs Faithfully Express Uncertainty?](https://arxiv.org/abs/2405.16908)

### Conversational AI
- [Dialogue State Tracking - Emergent Mind](https://www.emergentmind.com/topics/dialogue-state-tracking)
- [Grounding in Communication - Clark & Brennan](https://web.stanford.edu/~clark/1990s/Clark,%20H.H.%20_%20Brennan,%20S.E.%20_Grounding%20in%20communication_%201991.pdf)
- [Repair: Interaction and Cognition - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6849777/)
- [Dialogue Repair in Virtual Assistants - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11586770/)
- [Amazon Lex: Intent Confidence Scores](https://docs.aws.amazon.com/lexv2/latest/dg/using-intent-confidence-scores.html)

### ASR Error Handling
- [Confidence-Guided ASR Error Correction](https://arxiv.org/html/2407.12817v1)
- [ASR Error Correction using LLMs](https://arxiv.org/html/2409.09554v2)
- [Contextual ASR Error Handling - ACL](https://aclanthology.org/2025.coling-industry.32.pdf)

### Cognitive Psychology
- [Socratic Questioning Types - UMich](https://websites.umich.edu/~elements/probsolv/strategy/cthinking.htm)
- [Information Gap Theory - Psychology Fanatic](https://psychologyfanatic.com/information-gap-theory/)
- [Clinical Problem Solving - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC1122649/)
- [MoSCoW Method - Wikipedia](https://en.wikipedia.org/wiki/MoSCoW_method)
