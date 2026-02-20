# Tesseract Animation — Design Brief

**Objective:** Design a single, continuous tesseract (4D hypercube) animation that serves as the unified visual indicator across all Dynamic Island phases. Instead of swapping between separate icons (brain, wrench, sparkles, checkmark), one geometric figure morphs fluidly between states. The tesseract is the app's identity — it should feel like a living sigil.

## THE SHAPE

A tesseract is an inner cube nested inside an outer cube, with 8 edges connecting their corresponding vertices. When projected into 2D, it produces an elegant wireframe geometry — two overlapping squares (or a square within a square) with diagonal connecting lines.

```
    ┌─────────┐
    │  ┌───┐  │
    │  │   │  │
    │  └───┘  │
    └─────────┘
     + 4 diagonal edges connecting corners
```

**Rendering:** Thin white strokes on transparent background. No fills. Line weight ~1.5–2pt. Rounded line caps. The geometry should be small (20–24pt square) and sit where the phase icon currently lives — left side of the island content.

## STATE BEHAVIORS

Each phase maps to a distinct motion behavior of the same tesseract wireframe. Transitions between states should be continuous — the shape never disappears and reappears; it morphs from one behavior to the next.

### IDLE / APPEAR
The tesseract fades in from zero opacity and begins a slow, gentle rotation around one axis. Calm, ambient, barely moving. This is the resting state — the shape is present but quiet.

### LISTENING
The tesseract breathes — the inner cube scales in and out rhythmically, synced to `audioLevel`. Louder voice → inner cube pushes closer to outer cube edges. Silence → inner cube shrinks toward center. The connecting edges stretch and compress like tendons. The shape feels alive, reactive to the user's voice.

### TRANSCRIBING
The tesseract enters a steady, slow rotation — a single smooth 360° loop around the Y axis on a ~2–3 second cycle. Constant, predictable motion. Communicates "processing" without urgency. Slightly dimmed opacity (70–80%) to feel recessive while the text preview takes focus.

### THINKING
The tesseract rotates faster and along two axes simultaneously — a complex, hypnotic tumble. The motion should feel like the shape is "working through something." The rotation speed is moderate (not frantic) — contemplative, not anxious. Full white opacity.

### TOOL CALL
The tesseract snaps to a fixed orientation and pulses once — a quick scale-up (1.0 → 1.15 → 1.0) over ~300ms. Like a heartbeat. Then holds still. Each new tool call triggers another single pulse. The stillness between pulses communicates precision and confidence.

### RESPONDING
The tesseract returns to a slow single-axis rotation (similar to transcribing) but at full opacity. Smooth, steady, unhurried. The text is the star of this phase — the animation is a calm companion, not competing for attention.

### COMPLETE
The tesseract stops rotating and settles into a fixed orientation. A brief, satisfying scale-down (1.0 → 0.95 → 1.0) signals "done" — like a gentle nod. Then holds perfectly still until dismiss begins.

### ERROR
The tesseract jitters — a small, rapid horizontal shake (±2pt, 3 cycles over 400ms), then holds still. Unmistakable "something went wrong" without being dramatic. Could also slightly tint the stroke red for the error duration.

## TRANSITION RULES

- **Between states:** The rotation speed, axis, and inner-cube scale interpolate smoothly over ~300ms. Never snap. Use ease-in-out timing.
- **Appear:** Fade in from 0 opacity over 200ms, simultaneously beginning the idle rotation.
- **Dismiss:** Rotation decelerates to zero over 150ms while opacity fades to 0.

## IMPLEMENTATION NOTES

This could be implemented as:

1. **SwiftUI Canvas / Shape** — Draw the tesseract wireframe procedurally using `Path`. Animate rotation angles and scale via `TimelineView(.animation)`. This gives full control over vertex positions and allows `audioLevel` reactivity.

2. **Lottie / SVG animation** — Pre-built animation file with markers for each state. Simpler to implement but harder to make reactive to live audio data.

**Recommendation:** SwiftUI `Canvas` or custom `Shape` is the better fit. The listening state requires real-time reactivity to audio levels, which pre-baked animations can't provide. A procedural approach also makes transitions between states trivial — just interpolate the parameters.

### Geometry Reference

A 2D-projected tesseract has 16 vertices (8 inner + 8 outer) and 32 edges. For a simplified, clean version suitable for 20pt rendering, use the front-face projection:

- **Outer square:** 4 vertices at the corners of the bounding box
- **Inner square:** 4 vertices at a smaller, centered square (scale ~0.4–0.5 of outer)
- **Connecting edges:** 4 diagonal lines from each outer corner to the corresponding inner corner
- **Total:** 8 vertices, 12 edges (4 outer + 4 inner + 4 connecting)

This simplified projection reads clearly at small sizes and animates smoothly.

## CONSTRAINTS

- **No color fills.** Strokes only. White on black.
- **No particle effects, blur, or glow.** Pure geometry.
- **Keep it small.** The animation sits in a ~24pt square alongside text. It must not dominate.
- **Performant.** Must run at 60fps alongside the rest of the island content. Avoid expensive path recalculations — precompute vertex positions where possible.
- **One element, many behaviors.** The tesseract is always the same shape. States are expressed purely through motion, scale, and speed. The user should never feel like icons are being swapped — it's one living object that shifts its energy.
