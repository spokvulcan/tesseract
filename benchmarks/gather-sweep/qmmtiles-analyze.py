#!/usr/bin/env python3
"""Parse a qmmtiles driver log into per-shape ABBA tables.

Log lines of interest:
  QT <cfg> <shape> R=<n> <ms> ms <tflops> TFLOPs
  QTLAUNCH cfg=<cfg> rc=<rc> ts=...
  QTBLOCK cand=<cand> start/end=...
  QTGATE <cand> <shape> IDENT|DIFF|NOREF
"""
import sys, re
from collections import defaultdict

log = open(sys.argv[1]).read().splitlines()

# launches in order, each with the timing lines that preceded its QTLAUNCH
# marker (the driver appends QTLAUNCH after the binary exits)
launches = []  # (cfg, rc, {shape: (ms, tflops, R)})
pending = {}
qt_re = re.compile(r"^QT (\S+) (\S+) R=(\d+) ([\d.]+) ms ([\d.]+) TFLOPs")
for line in log:
    m = qt_re.match(line)
    if m:
        cfg, shape, R, ms, tf = m.groups()
        pending[shape] = (float(ms), float(tf), int(R))
        continue
    if line.startswith("QTLAUNCH"):
        kv = dict(p.split("=", 1) for p in line.split()[1:])
        launches.append({"cfg": kv["cfg"], "rc": int(kv["rc"]), "shapes": pending})
        pending = {}

gates = {}
for line in log:
    if line.startswith("QTGATE"):
        _, cand, shape, verdict = line.split()
        gates[(cand, shape)] = verdict

# group launches into candidate blocks of 8: s c c s s c c s
blocks = defaultdict(list)  # cand -> list of 8-launch groups
i = 0
while i + 7 < len(launches) + 1 and i + 8 <= len(launches):
    grp = launches[i:i + 8]
    cands = {g["cfg"] for g in grp if g["cfg"] != "stock"}
    if len(cands) == 1 and [g["cfg"] for g in grp] == [
            "stock", grp[1]["cfg"], grp[1]["cfg"], "stock",
            "stock", grp[1]["cfg"], grp[1]["cfg"], "stock"]:
        blocks[cands.pop()].append(grp)
        i += 8
    else:
        i += 1

shapes_order = []
for L in launches:
    for s in L["shapes"]:
        if s not in shapes_order:
            shapes_order.append(s)

print(f"launches parsed: {len(launches)} "
      f"(rc!=0: {sum(1 for L in launches if L['rc'] != 0)})")
for L in launches:
    if L["rc"] != 0:
        print(f"  CRASH cfg={L['cfg']} rc={L['rc']} shapes_completed={len(L['shapes'])}")

for cand, grps in blocks.items():
    print(f"\n=== candidate {cand} ({len(grps)} ABBA blocks) ===")
    print(f"{'shape':<26}{'stock TF':>9}{'cand TF':>9}{'ABBA ratio':>11}"
          f"{'pair ratios':>34}  gate")
    for shape in shapes_order:
        # collect per-pair (stock, cand) samples across blocks
        ratios, s_tfs, c_tfs = [], [], []
        for grp in grps:
            order = [(0, 1), (2, 3), (4, 5), (6, 7)]  # (stock idx, cand idx)
            for si, ci in order:
                s, c = grp[si], grp[ci]
                if shape in s["shapes"] and shape in c["shapes"]:
                    s_ms, s_tf, _ = s["shapes"][shape]
                    c_ms, c_tf, _ = c["shapes"][shape]
                    ratios.append(s_ms / c_ms)
                    s_tfs.append(s_tf)
                    c_tfs.append(c_tf)
        if not ratios:
            print(f"{shape:<26}{'-':>9}{'-':>9}{'-':>11}  no data")
            continue
        mr = sum(ratios) / len(ratios)
        gate = gates.get((cand, shape), "?")
        print(f"{shape:<26}{sum(s_tfs)/len(s_tfs):>9.3f}"
              f"{sum(c_tfs)/len(c_tfs):>9.3f}{mr:>11.3f}"
              f"  {' '.join(f'{r:.3f}' for r in ratios):>32}  {gate}")
