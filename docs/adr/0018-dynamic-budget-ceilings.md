---
status: accepted
---

# Dynamic budget ceilings with fast retreat — constants demoted to bootstrap

The prefix-cache RAM ceiling was `(physicalRAM − modelWeights − 20 GiB) / 2`, and
the SSD tier had a fixed 20 GiB budget. Both constants are wrong at both ends of
the hardware range: on a 48 GiB Mac running a 21 GiB 35B model the formula
yields a 4.1 GiB cache (~two 96k-token leaves; measured 2026-07-04, ornith-35b,
~21.5 KB/token), and under system memory pressure the band collapsed to ~1 GiB —
smaller than one turn's leaf. On a 192 GiB Studio the same 20 GiB headroom tax is
pointless stinginess. The 20 GiB constant itself was a reaction to a real
incident (a 4 GiB headroom once pushed peak MLX to 36 GiB and 14 GiB of swap on
a 48 GiB machine), so simply raising limits statically walks back into swap.
Decided in the 2026-07-04 grilling.

## Decision

- **Ceilings come from measurement, not constants.** The RAM ceiling tracks
  measured machine headroom (free + purgeable memory, other-process footprint),
  re-evaluated periodically — not computed once at model load from physical RAM
  and fixed taxes. The SSD budget defaults to a function of *free disk space*
  (fraction with an absolute cap, floored at the old 20 GiB default),
  re-evaluated periodically. The old constants survive only as bootstrap values
  before the first measurement.

- **The swap guardrail is fast retreat, not a static tax.** The
  pressure-reactive band (ADR-0011) is unchanged as a mechanism — fast-down on
  OS pressure events, slow hysteresis regrowth — and is what makes an
  aggressive ceiling safe. This ADR changes where the ceiling comes from, not
  how retreat works.

- **The `/2` divisor is replaced by an explicit active-inference reserve.** The
  halving was an unexplained safety factor; the actual thing it protected — the
  in-flight generation's KV working set — becomes a named reserve, sized per
  in-flight request so future batch (N lanes) subtracts N reserves instead of
  relying on slack.

- **User overrides are caps, never floors.** Both budgets are configurable in
  the app ("Automatic (recommended)" default), but a user value only lowers the
  effective ceiling; pressure retreat always wins. A user cannot configure the
  swap incident back into existence.

## Considered and rejected

- *Raise the constants* (e.g. 20 GiB → 8 GiB headroom): right on one machine
  size, wrong on the rest; re-litigated every hardware generation.
- *A user-visible "RAM-first vs SSD-first" mode switch*: thresholds misclassify
  (a 25B model should be RAM-first on a 128 GiB Studio, SSD-first on a 36 GiB
  Air); the continuous formula already decides by bytes left over, and the
  switch adds a settings surface with a behavior discontinuity. SSD-first
  behavior *emerges* when measured headroom is small.
