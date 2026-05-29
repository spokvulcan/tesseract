---
status: accepted
---

# The Settings Store seam sits below the facade, not at the module interface

Settings persistence now lives behind a **Settings Store** seam (`SettingsStore`
protocol; `UserDefaultsSettingsStore` and `InMemorySettingsStore` adapters), with
every default declared once in a **Settings Catalogue**. The seam is injected
*below* the `@Observable @MainActor` **Settings Facade** (`SettingsManager`) — it
is deliberately **not** the module's public interface.

Making the store the interface was rejected in a design-it-twice session. The
facade keeps one bindable stored property per setting because the 23 SwiftUI
`$settings.foo` binding sites and per-property Observation depend on it; a
store-shaped interface (`store.bool(for:)`) has neither bindings nor fine-grained
invalidation. The store moves bytes and never learns what a setting *means* — the
two genuine side effects (launch-at-login via `SMAppService`, dock visibility via
`NSApp`) stay in the facade's `didSet`, above the store. Vocabulary for this area
is in `CONTEXT.md` → **Language → Settings persistence**.

## Deferred: the `@Setting` macro

Collapsing each setting's stored-property + `didSet` + `init`/`reset` assignment to
a one-line declaration is a future ergonomics pass behind this *same* seam. It is
deferred because it first requires spiking whether a macro can emit the
`access`/`withMutation` calls that compose with `@Observable`'s own synthesis. Not
in scope for the seam itself (issue #16).

## Consequences

An architecture review that re-suggests "expose the store as the settings
interface" or "fold the catalogue into a keypath-driven `init`" should treat both
as already-decided:

- **Store-as-interface** breaks bindings + Observation — keep it injected below the
  facade.
- **Catalogue fold in `init`** cannot compile: under `@Observable` a property's
  hydrating first assignment must be a *direct, property-named* `self.foo = …`
  (it routes through the synthesized storage-restrictions init accessor and so
  skips `didSet`); a keypath/closure assignment fails definite initialization.
  The per-property `init`/`resetToDefaults` assignments therefore stay explicit,
  each reading its one default from the catalogue. This is also why the stored
  properties are declared *without* a default value — only then is the `init`
  line the genuine first write that skips `didSet`, keeping construction
  write-free and side-effect-free (the one exception is stale-value migration,
  which runs after hydration and so persists through the store).

The `@Setting` macro remains the open ergonomics lever; revisit once the
Observation-composition spike is done.
