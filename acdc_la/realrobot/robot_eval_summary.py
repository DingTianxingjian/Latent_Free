"""Score the collaborator's robot evaluation (anhdao69/terra, 2026-11-13/14) -- our FIRST success rates.

Reads only the small `results/summary.json` of each model directory of that repo, downloaded into its own
directory.  Run it with `python -I` and pass the path, so nothing in the download directory is importable:

  envs/univla_vla/bin/python -I acdc_la/realrobot/robot_eval_summary.py --root <download dir>

Fisher's exact test is implemented here (exact hypergeometric tail sum, two-sided by the "equal or less probable
table" convention) rather than pulled from scipy, so the number does not depend on which env this is run in.
Wilson score interval at 95 %, because 0/3 and 5/5 make a normal-approximation CI meaningless.
"""
import argparse, json, math, os
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fisher2(a, b, c, d):
    """Two-sided p for [[a, b], [c, d]] (a = successes arm1, b = failures arm1, ...)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9)))


def load(root, d):
    with open(os.path.join(root, d, "results", "summary.json")) as f:
        return json.load(f)


def rate(eps):
    k = sum(1 for e in eps if e["outcome"] == "success")
    lo, hi = wilson(k, len(eps))
    return k, len(eps), f"{k}/{len(eps)} = {100*k/len(eps):5.1f}%  [95% CI {100*lo:4.1f}, {100*hi:5.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding the downloaded <model>_res/ folders")
    A = ap.parse_args()
    S = {d: load(A.root, d) for d in sorted(os.listdir(A.root)) if d.endswith("_res")}

    print("=" * 100)
    print("PRESS_BUTTON -- the matched pair (both single-task, per-step delta, same recipe, same 5 layouts)")
    print("=" * 100)
    t = S["press_button_terra_delta_res"]["episodes"]
    c = S["press_button_openvla_delta_res"]["episodes"]
    kt, nt, st = rate(t)
    kc, nc, sc = rate(c)
    print(f"  TERRA   press_button_terra_delta    {st}")
    print(f"  control press_button_openvla_delta  {sc}")
    print(f"  Fisher exact (two-sided): p = {fisher2(kt, nt-kt, kc, nc-kc):.2e}   odds ratio = "
          f"{(kt/(nt-kt))/max(kc/(nc-kc), 1e-9):.1f}x")
    for nm, eps in (("TERRA", t), ("control", c)):
        dur = [e["duration_s"] for e in eps]
        stp = [e["policy_steps"] for e in eps]
        print(f"  {nm:8s} episode length: {sum(dur)/len(dur):5.1f} s mean ({min(dur):.1f}-{max(dur):.1f}), "
              f"{sum(stp)/len(stp):5.1f} policy steps")
    for nm, eps in (("TERRA", t), ("control", c)):
        print(f"\n  {nm} by COLOUR:")
        by = defaultdict(list)
        for e in eps:
            by[e["prompt"].split()[2]].append(e["outcome"] == "success")
        for col in sorted(by):
            v = by[col]
            print(f"    {col:7s} {sum(v)}/{len(v)}")
        print(f"  {nm} by LAYOUT:")
        by = defaultdict(list)
        for e in eps:
            by[e.get("layout", "?")].append(e["outcome"] == "success")
        for lay in sorted(by):
            v = by[lay]
            print(f"    {lay:11s} {sum(v)}/{len(v)}")

    print("\n" + "=" * 100)
    print("piper_real3_anchored_fwd -- the 3-task mixture, by task (NO control was run for any of these)")
    print("=" * 100)
    by = defaultdict(list)
    for e in S["piper_real3_anchored_fwd_res"]["episodes"]:
        by[e["task"]].append(e)
    for task in sorted(by):
        print(f"  {task:14s} {rate(by[task])[2]}")
    bb = by.get("press_button", [])
    if bb:
        print(f"    press_button colours seen: {dict(Counter(e['prompt'].split()[2] for e in bb))}")

    print("\n" + "=" * 100)
    print("PICK_CAN -- every arm, pooled (the camera pose differs from the other three tasks: see LEDGER 20:30)")
    print("=" * 100)
    pooled = []
    for d, key in (("piper_pick_can_terra_delta_res", None), ("piper_pick_can_openvla_delta_res", None),
                   ("piper_real3_anchored_fwd_res", "pick_can")):
        eps = [e for e in S[d]["episodes"] if key is None or e.get("task") == key]
        pooled += eps
        print(f"  {d.replace('_res',''):34s}{'' if key is None else ' ['+key+']':12s} {rate(eps)[2]}")
    print(f"  {'ALL ARMS POOLED':46s} {rate(pooled)[2]}")

    print("\n" + "=" * 100)
    print("CHUNK actually executed on the robot (our cards say chunk = 1 for per-step delta)")
    print("=" * 100)
    for d in S:
        ch = Counter(e.get("chunk") for e in S[d]["episodes"])
        hz = Counter(e.get("hz") for e in S[d]["episodes"])
        print(f"  {d.replace('_res',''):34s} chunk {dict(ch)}   hz {dict(hz)}")


if __name__ == "__main__":
    main()
