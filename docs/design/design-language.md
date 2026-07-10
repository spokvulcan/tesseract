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
  consumer of it — don't retrofit `accentColor` into it.

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

## Revision expectations

One revision loop is budgeted after the first surface (the native Settings
window, map #211) is built against this doc. Amendments are normal; silent
divergence is not.
