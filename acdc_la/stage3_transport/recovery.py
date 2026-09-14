"""Few-label cross-context action recovery R2_recovery(p, c): fit a = W z on a fraction p of the labelled
context-A episodes (per source), freeze W, read R2 / cosine in context B.
    c = episode : held-out episodes (random third)          c = task : held-out INSTRUCTIONS (md5 bucket %3 == 0;
    sources with < 2 instructions fall back to the episode split and are counted in n_task_fallback)
p in {0.01, 0.05, 0.1, 0.25, 1.0}, 5 draws of the labelled subset.  Ledger (2026-09-08): swap did not improve
this over rec at any p -- the Stage-III gain is in the operator, not in z's linear recoverability.
Usage: python -m acdc_la.stage3_transport.recovery rec_src swap_fz ...  -> results/recovery_curve.json
"""
import sys, json, os, hashlib
import numpy as np

from ..paths import RESULTS
from ..common import readers as R
from ..data import oxe_mix

PS = (0.01, 0.05, 0.1, 0.25, 1.0)


def instruction_by_global_ep(names):
    m, off = {}, 0
    for name in names:
        d = oxe_mix.source_dir(name); z = np.load(os.path.join(d, "index.npz")); n = len(z["starts"])
        sm = os.path.join(d, "scene_meta.npz")
        ins = np.load(sm, allow_pickle=True)["instruction"] if os.path.exists(sm) else np.array([""] * n)
        for e in range(n): m[off + e] = str(ins[e]) if e < len(ins) else ""
        off += n
    return m


def per_dump(d, ps=PS, draws=5):
    Z, NET, GE, SI, names = d["Z"].astype(np.float64), d["NET"], d["GE"], d["SI"], d["names"]
    ins = instruction_by_global_ep(names); res = {}
    for ctx in ("episode", "task"):
        table = {}
        for s in np.unique(SI):
            m = SI == s; ug = np.unique(GE[m]); rs = np.random.RandomState(1); fallback = False
            te_eps = rs.choice(ug, max(1, int(len(ug) / 3)), replace=False)
            if ctx == "task":
                uins = sorted({ins.get(int(g), "") for g in ug})
                cand = np.array([g for g in ug if int(hashlib.md5(ins.get(int(g), "").encode()).hexdigest(), 16) % 3 == 0])
                if len(uins) >= 2 and 2 <= len(cand) <= len(ug) - 3: te_eps = cand
                else: fallback = True
            te = m & np.isin(GE, te_eps); tr_eps = np.setdiff1d(ug, te_eps)
            if te.sum() < 10 or len(tr_eps) < 4: continue
            row = dict(n_test=int(te.sum()), n_train_eps=int(len(tr_eps)), fallback=fallback)
            for p in ps:
                k = max(3, int(round(p * len(tr_eps)))); r2s, coss = [], []
                for dr in range(draws if p < 1 else 1):
                    lab = np.random.RandomState(100 + dr).choice(tr_eps, min(k, len(tr_eps)), replace=False); tr = m & np.isin(GE, lab)
                    if tr.sum() < 8: continue
                    G, _ = R.fit_reader(Z, NET, GE, tr); P = G(Z[te]); ym = NET[te].mean(0)
                    r2s.append(R.r2(NET[te], P)); coss.append(float(R.cosr(NET[te] - ym, P - ym).mean()))
                if r2s: row[f"r2_p{p}"], row[f"cos_p{p}"], row[f"neps_p{p}"] = float(np.mean(r2s)), float(np.mean(coss)), k
            table[names[s]] = row
        agg = R.aggregate(table); agg["n_task_fallback"] = int(sum(r.get("fallback", False) for r in table.values()))
        res[ctx] = agg
    return res


def main(specs):
    res_path = os.path.join(RESULTS, "recovery_curve.json")
    out = json.load(open(res_path)) if os.path.exists(res_path) else {}
    tags = []
    for spec in specs:
        d = R.load_arrays(spec); tag = os.path.basename(d["path"]).replace("_val.npz", "").replace("swaplam_", "").replace("_readout_arrays.npz", "")
        out[tag] = per_dump(d); tags.append(tag); json.dump(out, open(res_path, "w"), indent=1)
    for ctx in ("episode", "task"):
        print(f"\n[{ctx} context]  R2 (cos) at label fraction p")
        print("%-16s" % "tag" + "".join("%16s" % f"p={p}" for p in PS))
        for t in tags:
            a = out[t][ctx]; print("%-16s" % t[:15] + "".join("%16s" % f"{a.get(f'r2_p{p}', float('nan')):.3f} ({a.get(f'cos_p{p}', float('nan')):.2f})" for p in PS))


if __name__ == "__main__":
    main(sys.argv[1:] or ["rec_src", "swap_fz"])
