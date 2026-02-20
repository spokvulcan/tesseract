# Agent Dynamic Island — Design & Engineering Brief

**Role:** Act as a World-Class Senior Motion Designer and Lead SwiftUI Engineer specializing in premium micro-interactions. **Objective:** Refine and perfect a macOS Dynamic Island overlay for an AI voice assistant called Tesse. The island appears at the screen notch and visualizes a full voice-to-response lifecycle. **Aesthetic Identity:** Minimalistic, dark, fluid. The island is a solid black pill — no glow, no gradients behind it, no ambient lighting effects. It communicates entirely through shape, motion, and typography. Every frame must feel intentional. Zero generic AI patterns.

## CURRENT ARCHITECTURE (DO NOT RESTRUCTURE)

```
tesseract/Features/Agent/AgentNotch/
├── AgentNotchState.swift          — AgentNotchPhase enum + @Observable state
├── AgentNotchOverlayView.swift    — SwiftUI view (all visuals + animations)
└── AgentNotchPanelController.swift — NSPanel lifecycle (show/dismiss/forceClose)
```

Reused from existing codebase:
- `DynamicIslandShape` — animatable notch path (concave top, rounded bottom) at `tesseract/Features/Speech/NotchOverlay/DynamicIslandShape.swift`
- `NotchFrameTracker` — syncs SwiftUI frame → NSPanel frame at `tesseract/Features/Speech/NotchOverlay/TTSNotchOverlayView.swift`
- `AudioBarsView` — timeline-style waveform bars at `tesseract/Features/Dictation/Views/AudioBarsView.swift`
- `ProcessingDotsView` — pulsing white dots at `tesseract/Features/Dictation/Views/ProcessingDotsView.swift`

State is driven by `AgentCoordinator` (at `tesseract/Features/Agent/AgentCoordinator.swift`) which calls `notchController.show()`, `.updatePhase()`, and `.dismiss()` as voice input and generation progress.

## INTERACTION LIFECYCLE (7 PHASES)

The island progresses through these states sequentially. Each transition must feel like a single continuous gesture — never jarring, never abrupt.

### Phase 1 — LISTENING
**Trigger:** User presses and holds agent hotkey (Control+Space).
**Duration:** As long as hotkey is held.
**Data:** `audioLevel: Float` (0.0–1.0), updated ~20× per second from microphone.
**Current:** AudioBarsView + "Listening" text label.
**Intent:** The island should feel *alive* and *receptive* — breathing with the user's voice. This is the first impression. It must be magnetic.

### Phase 2 — TRANSCRIBING
**Trigger:** User releases hotkey.
**Duration:** 0.5–3 seconds (Whisper model inference).
**Data:** Initially empty string, then the full transcribed text appears at once.
**Current:** ProcessingDotsView + text preview or "Transcribing..." label.
**Intent:** A brief liminal state. Should feel like the system is processing with quiet confidence, not anxious loading.

### Phase 3 — THINKING
**Trigger:** Agent LLM starts generating, enters `<think>` reasoning block.
**Duration:** 1–10 seconds depending on complexity.
**Data:** None visible (thinking content is hidden from user).
**Current:** Pulsing brain icon + "Thinking..." label in purple.
**Intent:** The mind at work. Should feel contemplative, not stalled. The user should sense intelligence happening behind the surface.

### Phase 4 — TOOL CALL
**Trigger:** Agent invokes a tool (task_create, recall, reminder_set, etc.).
**Duration:** 0.2–1 second per tool call. May appear multiple times interleaved with Thinking.
**Data:** Tool name string (mapped to human-readable via `AgentNotchPhase.toolDisplayName()`).
**Current:** Wrench icon in circle + tool display name in mint green.
**Intent:** Brief, purposeful flash — the assistant is taking action. Should feel like a confident keystroke, not a loading spinner.

### Phase 5 — RESPONDING
**Trigger:** Agent streams response text tokens.
**Duration:** 2–15 seconds depending on response length.
**Data:** `text: String` — grows incrementally as tokens arrive (updated every ~10 characters).
**Current:** Sparkles icon + streaming text, 3-line max.
**Intent:** The assistant speaks. Text should materialize naturally — not pop in block by block. The island stays the same size; text flows within the fixed container. This is the payoff moment.

### Phase 6 — COMPLETE
**Trigger:** Generation finishes.
**Duration:** Holds for 2.5 seconds, then auto-dismisses.
**Data:** Final response text (full content, truncated in display).
**Current:** Green checkmark in circle + text excerpt.
**Intent:** Satisfaction. Done. A gentle signal of completion before the island retreats.

### Phase 7 — ERROR
**Trigger:** Mic in use, recording too short, transcription failed, generation failed.
**Duration:** Holds for 2.5 seconds, then auto-dismisses.
**Data:** Error message string.
**Current:** Red warning icon in circle + message text.
**Intent:** Acknowledge failure without drama. Informative, not alarming.

## GESTURES

- **Tap** anywhere on island → navigates to agent window (via `onTap` callback)
- **Swipe up** → immediate dismiss with spring physics + upward slide-out
- **Swipe down** → rubber-band resistance (15% of drag distance), snaps back

## CORE DESIGN SYSTEM (STRICT)

### Palette
- **Background:** Pure black (`Color.black`). The island is a solid dark pill — no glow, no halo, no ambient effects.
- **Primary Text:** White at 85–95% opacity. Never 100% — that's harsh.
- **Thinking Accent:** Soft purple `Color(red: 0.8, green: 0.6, blue: 1.0)`
- **Tool/Complete Accent:** Mint green `Color(red: 0.5, green: 0.9, blue: 0.7)`
- **Error Accent:** System red, but contained in a subtle circle badge — never overwhelming.

### Typography
- **All text:** `.system(design: .rounded)` — clean, modern feel.
- **Labels** (Listening, Thinking, tool names): 14pt semibold.
- **Body text** (responses, previews): 13–14pt regular.
- **Error text:** 13pt medium.

### Shape
- `DynamicIslandShape` with animatable `topInset` (8→14pt) and `bottomRadius` (8→22pt).
- Collapsed state: 160pt wide × 37pt tall (matches physical notch).
- Expanded: **One fixed size for all visible phases.** No per-phase width/height variation.

### Sizing (STRICT: uniform for all phases)

**All visible phases use the same expanded size.** The island expands once on appear and stays that size until dismiss. No resizing between phases — content transitions happen inside a fixed container. This prevents distracting shape jitter during phase changes.

| State | Width | Content Height |
|-------|-------|----------------|
| hidden (collapsed) | 160 | 0 |
| **all visible phases** | **320** | **64** |

The 320×64 container is large enough to hold 2–3 lines of response text but compact enough to feel unobtrusive during simple states like listening or thinking. Content within the fixed container should be vertically centered for single-line phases and top-aligned for multi-line text.

## ANIMATION REQUIREMENTS (STRICT)

### Global Rules
- **Spring physics everywhere.** No linear animations except text streaming. Use `.spring(response: 0.35–0.5, dampingFraction: 0.7–0.85)` — weighted and satisfying, never bouncy.
- **Choreographed sequences.** On appear: shape expands first, content follows 150–200ms later. On dismiss: content fades first, shape collapses after. Between phases: only content transitions — the shape stays fixed.
- **60fps minimum.** Use `TimelineView(.animation)` for continuous animations. Use `.drawingGroup()` for composited renders.
- **No jumpcuts.** Every state transition must crossfade, blur-replace, or morph. The user should never see a frame where content pops in from nothing.

### Appear Sequence
1. Island springs from notch size (160×37) to target size — `.spring(response: 0.45, dampingFraction: 0.72)`
2. Content fades in 200ms after shape settles — `.easeOut(duration: 0.3)`

### Phase Transitions
- Content swaps use `.blurReplace` for neighboring phases (listening→transcribing→thinking→responding)
- Tool calls use `.push(from: .bottom)` insertion — brief, confident
- Complete uses `.push(from: .bottom)` — a gentle arrival from below
- Error uses `.scale(scale: 0.9)` insertion — draws attention without shouting
- All phase content animated with `.spring(response: 0.35, dampingFraction: 0.8)`

### Dismiss Sequence
1. Content fades out — `.easeOut(duration: 0.2)`
2. 150ms later: shape collapses to notch + slides up 80pt — `.spring(response: 0.4, dampingFraction: 1.0)` (critically damped, no bounce on exit)

## TECH STACK

- **SwiftUI** (macOS 26+, Swift 6.2)
- **NSPanel** — borderless, non-activating, `.screenSaver` level, `canJoinAllSpaces`
- **@Observable** for state (`AgentNotchState`)
- **`NotchFrameTracker`** syncs SwiftUI layout → NSPanel frame via `didSet`
- Logging: `Log.agent` (never `print()`)

## QUALITY BAR

Do not build a notification badge; build a living interface element. Every expansion should feel like it's *growing*, not *resizing*. Every text appearance should feel like it's *materializing*, not *rendering*. The dismiss should feel like the island is retreating *into* the notch, not *disappearing* from the screen. The pill itself is the entire visual — no ambient effects, no halos, no decorations. Shape and motion do all the talking.

Reference: iPhone 15 Pro Dynamic Island interactions. The weight and timing of iOS spring animations. That is the bar.
