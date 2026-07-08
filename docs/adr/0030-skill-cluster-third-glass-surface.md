# The Skill Cluster is the third custom glass surface

Status: accepted (amends ADR-0024's two-surface clause)

ADR-0024 restricted Liquid Glass to *exactly two* custom surfaces — the composer
bar and the slash popup — sharing one `GlassEffectContainer`, with the content
layer never getting glass. The **Skill Pill** strip that shipped under that budget
lives *inline in the composer's action row* (PRD #183 US 18), which keeps the
composer's chrome permanently loaded: six-plus capsules crowd the action row by
default, scroll-clipped at ordinary window widths, whether or not the user is
reaching for a skill.

**Decision:**

- **The Skill Pills leave the composer and become the Skill Cluster** — a
  floating glass surface above the composer's trailing corner: a collapsed
  ~38pt circular bubble (`sparkles` icon, `.regular` glass, `.interactive()`)
  that morphs open into the fanned pills. Hover opens (~150ms delay, ~250ms
  exit grace), click pins until click-away/Esc, firing a pill collapses.
- **It is the third custom glass surface, in its own
  `GlassEffectContainer(spacing: 16)`** in `AgentContentView`'s bottom
  safe-area inset — the collapsed⇄expanded morph uses `glassEffectID` within
  that container (all verified available on macOS 26: `GlassEffectContainer`,
  `glassEffectID`, `glassEffectUnion`, `GlassEffectTransition` in the macOS
  26.5 SDK swiftinterface). *Amended 2026-07-08 after live testing:* the first
  build shared the composer's container so the two could sample each other's
  glass, and at close range the pills liquid-fused into the composer's top
  edge. Apple's Landmarks Liquid Glass sample (`BadgesView`) gives its
  morphing badge cluster its own container for exactly this reason; the
  one-container rule (ADR-0024) scopes to elements that *should* blend. The
  transcript content itself still never gets glass; the cluster floats in the
  inset layer, not the content layer.
- **Geometry:** pills fan leftward from the bubble at its height, most-used
  nearest (the **Skill Usage Ranking**'s new spatial mapping), wrapping upward
  right-aligned when out of width. No scrolling, no overflow menu.
- **Contracts carried over unchanged:** the composer-draft ride-along (fire
  drains text + images, restores on failure, bare fire works), disabled-not-
  hidden while generating (bubble dims, won't expand), ranking recomputed only
  at conversation start, and the "show skill pills" Setting (same stored key)
  now gates the cluster.

**Considered options:**

- *Collapse the strip into a composer action-row icon expanding to a glass
  popover* — no new glass surface, smallest ADR bend; rejected: keeps the skills
  affordance inside the composer's chrome and loses the free-floating morph that
  motivated the redesign.
- *Free-floating FAB pinned over the transcript* — rejected: glass directly on
  the content layer, the exact thing ADR-0024 and the HIG forbid, and it would
  need a second `GlassEffectContainer` whose refraction cannot match the
  composer's.
- *Mode toggle keeping the inline strip as an alternative rendering* — rejected:
  two permanent renderings of one controller.

**Consequences:**

- ADR-0024's "exactly two custom surfaces" clause is amended to three; the
  clause "the content layer never gets glass" stands.
- `SkillPillRowView` is replaced by the cluster rendering; the action row's
  `Spacer()` returns. `SkillPillController` (derivation, ranking, argument
  assembly) is unchanged.
- The slash popup and the cluster are mutually exclusive occupants of the
  above-composer space: while the popup is open the cluster is suppressed
  (hover and clicks inert) and hidden.
- Keyboard path is unchanged: slash commands remain the keyboard route to
  skills; the cluster is a pointer surface (its buttons join the normal
  Full-Keyboard-Access order but no default tab stops are added).
- Hover-open is pointer-dependent by design; if the hover grace tuning feels
  slippery in practice, the click-pin path is the fallback, not a redesign.
