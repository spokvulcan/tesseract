# Tesseract design language

One page of ratified rules for every face-lifted surface. Cite this doc
instead of re-deciding; changing a rule means amending this doc in the same
PR. Ratified 2026-07-09 (map #211, ticket #214). Locked designs — agent chat
(PR #184), Server Drawer, Skill Cluster (ADR-0030), speech transport bar,
overlay HUD — are audited for consistency against this doc but never
redesigned under it.

## 1. Liquid Glass

Grounded in Apple's [Adopting Liquid Glass](https://developer.apple.com/documentation/technologyoverviews/adopting-liquid-glass),
the [Materials](https://developer.apple.com/design/human-interface-guidelines/materials)
and [Settings](https://developer.apple.com/design/human-interface-guidelines/settings) HIG pages.

1. **Two layers, always.** Liquid Glass is the functional/navigation layer
   (toolbars, sidebars, tab bars, floating controls). The content layer uses
   standard materials — **never `glassEffect`**. The only content-layer glass
   is the transient knob of an active slider/toggle, which the system draws.
2. **Adopt by subtraction.** Build against the macOS 26 SDK and *remove*
   custom backgrounds, scrims, and blurs on bars, sheets, sidebars, and split
   views — they fight the automatic glass and the scroll-edge effect. Window
   chrome, toolbars, and scroll edges take the automatic treatment; no
   per-surface chrome styling.
3. **Custom glass: one `GlassEffectContainer` per cluster.** Separate
   containers cannot sample each other's refraction — this is a correctness
   rule. Apply `.glassEffect` after layout/appearance modifiers.
4. **Tint is semantic only** — primary action, alert, status. System colors
   or light+dark+increased-contrast custom variants. No decorative tinting.
5. **`.regular` by default; `.clear` only floating over media**, with a
   35 %-opacity dim layer over bright media.
6. **Overlay windows: known ceiling.** `.clear` + forced-aqua is the public
   maximum; the Control Center variant is private; `appearsActive` /
   key-window-faking do nothing. Don't relitigate.
7. **Settings is a `Settings` scene with a `TabView` of panes** — a
   noncustomizable toolbar of panes, not a sidebar. Window sizes to the
   current pane; title reflects the pane; last pane restored; minimize/zoom
   dimmed; entry via ⌘, and the App menu only.
8. **Settings content is content-layer:** `formStyle(.grouped)`, standard
   materials (see §4).
9. **Custom glass ships with accessibility passes:** Reduce Transparency,
   Reduce Motion, Increase Contrast. Standard components adapt for free;
   custom ones must be tested.

**Sanctioned custom-glass inventory** — the complete list of surfaces that
may carry a custom `glassEffect`. Adding one means amending this list in the
same PR:

- Agent composer (`AgentComposerView`)
- Slash-command popup (`SlashCommandPopupView`)
- Skill Cluster (`SkillClusterView`, ADR-0030)
- Speech transport bar (`SpeechTransportBar`)
- Global overlay HUD (`GlobalOverlayHUD`)
- Dictation recording button (`RecordingButtonView`)
- Models action bar (`ModelsActionBar`)

The inventory governs custom `.glassEffect` surfaces. The system glass
button styles (`.glass` / `.glassProminent`) are standard components — they
don't need an entry, including when a `GlassEffectContainer` merely gives a
group of such buttons a shared sampling context (onboarding footer nav,
Prompt Cache canvas zoom controls).

## 2. Reference aesthetic — what generalizes from the agent chat

The chat is the reference surface. Its *principles* graduate app-wide; its
literal numbers (16 pt body, 720 pt column, 16 pt rhythm —
`ChatViewSupport.swift`) stay chat's own. Standard forms and controls keep
native macOS metrics untouched.

- **One type size per surface.** Hierarchy comes from weight and color,
  never from size (chat: `chatBodyFontSize`, one 16 pt size for body,
  thinking, tool rows, badges alike).
- **One spacing rhythm per surface.** A single vertical spacing constant per
  surface, no clustering exceptions (chat: `ChatLayout.rowSpacing`).
- **Readable column on prose-like content.** Cap the content column; never
  let prose stretch the window (chat: `ChatLayout.columnMaxWidth`).
- **Icon-light rows; actions in context menus.** Rows carry no visible
  button chrome; row actions live in the context menu (PR #184).
- **Quiet loading.** Small inline `ProgressView` plus plain status text
  (`AgentComposerView` model-loading states). No skeletons, no shimmers.

## 3. Accent

- **The app accent is the markdown warm orange**, set as the asset-catalog
  `AccentColor`: light `#D68C27`, dark `#F5A742`, each with an
  increased-contrast variant. Every standard control, selection highlight,
  focus ring, and `Color.accentColor` call-site picks it up automatically —
  no per-view code. (macOS users who pick a custom system accent override
  it; that's platform behavior, accept it.)
- **`.tint` stays semantic** (rule 4): destructive red, status colors. Don't
  re-tint what the accent already covers.
- **The Prose Accent Palette is chat-local and locked**
  (`ChatMarkdownStyle.swift`): headings purple in dark mode, warm
  strong/emphasis, peach links. It is the *origin* of the accent, not a
  consumer of it — don't retrofit `accentColor` into it. The lock is on the
  *colors* (OpenCode-exact, verified 2026-07-11); markdown chrome around them
  — the neutral inline-code chip, the warm blockquote bar — is tunable, and
  tuned in the Markdown Gallery (Window menu), never by re-deciding colors.

## 4. Settings idioms

- Grouped `Form` panes, **Title-Case section headers**.
- **Help text on every non-obvious knob** — one or two short sentences,
  secondary style, under the control (or a `Section` footer for group-level
  context). Written *effect-first*; never restate the label. Obvious knobs
  (Launch at Login) stay bare.
  - ✅ "Keeps the last 10 recordings on disk for replay and re-transcription."
  - ✅ "Slower first response after switching models, but image questions
    answer immediately."
  - ❌ "Enables web access." (under a toggle already labeled *Web Access*)
- **Controls:** toggles for booleans; segmented pickers for ≤ 3 always-visible
  options; menu pickers otherwise; no custom control where a standard one
  exists.
- **Never duplicate system settings** (appearance, accessibility, scrolling).

## 5. Charts

Ratified with the Cache cutover (map #269, ticket #277). Charts are Swift
Charts; the shared pieces live in `ChartSupport.swift`. The reference
surfaces are the Cache Overview's three charts and the Activity rail.

- **The categorical palette is `ChartPalette` — four fixed slots, assigned
  in fixed order, never cycled.** Light `#2A78D6 · #1BAF7A · #D68C27 ·
  #4A3AA7` on `#fcfcfb`; dark `#3987E5 · #199E70 · #C67F16 · #9085E9` on
  `#1a1a19`. Validated (CVD separation, lightness band, chroma floor)
  2026-07-10 — re-run the dataviz palette validator before changing any
  step. Slot 3 is the brand warm orange with its dark step lowered into the
  dark lightness band; **prefill wears slot 3 on every chart, app-wide** —
  color follows the entity, never its rank. A fifth series is never a new
  hue: fold into "Other," split the chart, or encode differently.
- **One axis per chart.** Never a dual-axis chart. Two measures of
  different scale get two charts or an indexed base.
- **A legend whenever a chart draws ≥ 2 series;** a single series is named
  by its title. Identity never rides on color alone — legends, axis labels,
  and tooltips carry it in text.
- **Text wears text tokens, never the series color.** Values, labels, and
  legends stay in primary/secondary/muted ink; a small color dot beside the
  text carries identity (`ChartTooltipRow`).
- **Hover is standard equipment**: a full-plot `ChartHoverOverlay`, a 1 px
  quaternary `RuleMark` cursor snapped to the nearest point, the hovered
  point emphasized, and a `ChartTooltipChrome` annotation — a thin-material
  chip with a hairline quaternary ring (content layer: standard materials,
  no glass), fitted inside the chart bounds.
- **Heavy-tailed measures cap the y-domain near p95** (when
  `max > p95 × 1.6`); off-scale marks clip at the top edge and the footnote
  counts them ("N slow requests run off-scale, hover for exact"). Truthful
  and readable beats fitted and unreadable. Swift Charts does not clip
  marks to the plot area on its own — a capped domain requires
  `.chartPlotStyle { $0.clipped() }`, or the off-scale marks paint over
  whatever sits above the chart. (Annotations survive the clip.)
- **Footnotes state the window and the identity** ("last 80 of 3 781
  requests in 30 d, oldest → newest") in caption-weight secondary text —
  no silent truncation of what a chart covers.
- **Swift Charts gotcha (learned the hard way):** `BarMark`
  `width: .ratio` is only defined for binned/categorical axes — on a
  continuous (Int) x-axis it renders zero-width, invisible bars while the
  legend and tooltips keep working. Use a `.fixed` width computed from the
  measured chart width ÷ point count (clamped), via `onGeometryChange`.
  Automatic width doesn't shrink with density (bars merge at ~400 pt), and
  `xStart:/xEnd:` ranges break y-stacking — both probed and rejected.

## Revision expectations

One revision loop is budgeted after the first surface (the native Settings
window, map #211) is built against this doc. Amendments are normal; silent
divergence is not.
