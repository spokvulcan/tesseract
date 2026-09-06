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

## Amendment 2026-09-06 — the Working-Set Bound

The measured headroom was widened in issue #236 to the kernel's inactive and
speculative buckets, because a free+purgeable sample read ~4.6 GB next to a
35B model and zeroed the ceiling. Those buckets are not safe on their own: the
cache's own cold snapshot pages age into "inactive", so a growing cache raised
its own ceiling. Replaying a 65-turn agent session as dead-end branches (each
turn admitting a ~3 GB leaf and a ~3 GB branch point that nothing superseded)
moved the ceiling from 8.7 GB to 38.7 GB in 17 minutes on a 48 GB machine with
a 16 GB model; the process reached ~70 GB, swap filled the disk, and the
machine rebooted. The fast retreat could not save it: the pressure events
arrive on the main actor, which was inside an admission drain that swapping
had stretched to 25–50 s.

- **Headroom is the smaller of the kernel buckets and the Working-Set Bound**:
  the per-process working set the GPU driver recommends
  (`recommendedMaxWorkingSetSize`, ~78% of physical memory on Apple Silicon)
  minus this process's physical footprint. Footprint counts compressed pages
  and Metal buffers, so growth cannot hide in the compressor. Each admitted
  byte is one byte less headroom, and the ceiling converges at
  working set − footprint − 1.25 × reserve instead of tracking the cache.
  The #236 case keeps a usable ceiling: ~40 GB working set − 18.6 GB weights
  − live KV leaves ~14 GB before the reserve.
- **The SSD budget never exceeds what the disk can hold**: what the tier
  already holds plus free space above a 10 GiB reserve. The 20 GiB floor
  stays the default where the volume can keep it; it no longer writes the
  last bytes of a full disk.
- The `budgetMeasure` trace carries the bound (`workingSetHeadroomBytes`), so
  a small ceiling is attributable to the process rather than the machine.
