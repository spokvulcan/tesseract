"""Paired comparison of the probe's variants against the Python rendering."""
import json
import random
import sys

results = json.load(open(sys.argv[1]))
reference = results["python"]["rows"]
random.seed(0)


def bootstrap(diffs, n=10000):
    means = sorted(sum(random.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


print(f"{'variant':15s} {'prefix':>6s} {'fn':>6s} {'values':>6s} {'NLL':>7s} {'ΔNLL vs python [95% CI]':>28s} {'worse/better':>12s}")
for name, data in results.items():
    rows = data["rows"]
    nll = [-r["name_logp"] for r in rows]
    diffs = [-r["name_logp"] - -ref["name_logp"] for r, ref in zip(rows, reference)]
    mean = sum(diffs) / len(diffs)
    lo, hi = bootstrap(diffs) if name != "python" else (0.0, 0.0)
    worse = sum(d > 1e-6 for d in diffs)
    better = sum(d < -1e-6 for d in diffs)
    fn = sum(r["fn_ok"] for r in rows)
    values = sum(r["values_ok"] for r in rows)
    print(f"{name:15s} {data['prefix_tokens']:6d} {fn:3d}/{len(rows)} {values:3d}/{len(rows)} {sum(nll) / len(nll):7.3f} "
          f"{mean:+8.3f} [{lo:+.3f}, {hi:+.3f}] {worse:5d}/{better:<5d}")

print("\nTasks any variant got wrong:")
for i, ref in enumerate(reference):
    wrong = [n for n, d in results.items() if not d["rows"][i]["values_ok"]]
    if wrong:
        print(f"  {ref['user'][:60]:60s} wrong in: {', '.join(wrong)}")
        for n in wrong:
            print(f"      {n:15s} {results[n]['rows'][i]['output'][:160]!r}")

print("\nParameter order the model emitted (python vs swift_today), multi-parameter calls:")
for i, ref in enumerate(reference):
    a = results["python"]["rows"][i]["param_order"]
    b = results["swift_today"]["rows"][i]["param_order"]
    if len(a) > 1 or len(b) > 1:
        print(f"  {ref['function']:16s} python={a}  swift_today={b}")
