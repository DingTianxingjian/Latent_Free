"""Scene-leakage probes on the code z (2026-10-27, registered as the cheap side experiment of the counterfactual plan).

Linear probes on a validation dump (results/<tag>_val.npz), one episode-disjoint split shared by every tag:
  ACTION  (higher = better):  ridge z -> net displacement (3) and the 4-segment sequence (12), pooled R^2
  LEAKAGE (lower  = better):  logistic z -> source dataset id (29 classes), balanced accuracy
                              ridge z -> first-frame appearance, PCA-32 of F0Q (pooled DINO features of frame 0), R^2
Reference rows: the same probes from F0Q itself (how much scene information exists to leak) and chance.
Caveat: source id also carries embodiment / action-space statistics, so it is not pure scene; the appearance probe
is the cleaner scene measure.  Indirect evidence only -- the replay experiment is the direct test.

    python -m acdc_la.stage1_preserve.leakage_probes latp2_inv_ref latp2_anchor_ref tc_cont_ref
"""
import sys, json, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from ..paths import RESULTS


def split(ge, frac=0.3, seed=0):
    eps = np.unique(ge); rs = np.random.RandomState(seed)
    test_eps = set(rs.choice(eps, int(len(eps) * frac), replace=False))
    te = np.array([g in test_eps for g in ge]); return ~te, te


def ridge_r2(X, Y, tr, te):
    # alpha by efficient leave-one-out CV on the training clips: a fixed alpha over-fits the small per-source sets
    sx = StandardScaler().fit(X[tr]); sy = StandardScaler().fit(Y[tr])
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sx.transform(X[tr]), sy.transform(Y[tr]))
    return float(r2_score(sy.transform(Y[te]), m.predict(sx.transform(X[te])), multioutput="uniform_average"))


def source_acc(X, si, tr, te):
    sx = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=2000, C=1.0).fit(sx.transform(X[tr]), si[tr])
    return float(balanced_accuracy_score(si[te], m.predict(sx.transform(X[te]))))


def main(tags):
    ref = np.load(os.path.join(RESULTS, f"{tags[0]}_val.npz"), allow_pickle=True)
    ge, si = ref["GE"], ref["SI"]; tr, te = split(ge)
    f0 = ref["F0Q"].astype(np.float32)
    app = PCA(32, random_state=0).fit(f0[tr]).transform(f0)               # first-frame appearance target
    act = ref["ACT"].astype(np.float32); net, seq = act.sum(1), act.reshape(len(act), -1)
    out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_sources": int(len(np.unique(si))),
           "chance_balanced_acc": 1.0 / len(np.unique(si))}
    rows = {"F0Q (first-frame features, reference)": f0}
    for t in tags:
        d = np.load(os.path.join(RESULTS, f"{t}_val.npz"), allow_pickle=True)
        assert np.array_equal(d["GE"], ge) and np.array_equal(d["SI"], si), f"{t}: different validation clips"
        rows[t] = d["Z"].astype(np.float32)
    print(f"[leak] {out['n_train']} train / {out['n_test']} test clips, episode-disjoint; {out['n_sources']} sources, "
          f"chance balanced acc {out['chance_balanced_acc']:.3f}", flush=True)
    print(f"{'representation':42s} {'dims':>5s} | {'act net R2':>10s} {'act seq R2':>10s} | {'source acc':>10s} {'appear R2':>10s}")
    for name, X in rows.items():
        r = dict(dims=int(X.shape[1]), action_net_r2=ridge_r2(X, net, tr, te), action_seq_r2=ridge_r2(X, seq, tr, te),
                 source_bal_acc=source_acc(X, si, tr, te), appearance_r2=ridge_r2(X, app, tr, te))
        out[name] = r
        print(f"{name:42s} {r['dims']:5d} | {r['action_net_r2']:10.3f} {r['action_seq_r2']:10.3f} | "
              f"{r['source_bal_acc']:10.3f} {r['appearance_r2']:10.3f}", flush=True)
    # WITHIN-SOURCE version (2026-10-27): source id carries embodiment / action space, so the pooled probes above confound
    # "knows the robot" with "knows the scene".  Per source (>= 80 test clips): ridge z -> appearance PCA (fit per
    # source) and ridge z -> net / seq action; clip-weighted mean over sources.
    print(f"\n{'WITHIN-SOURCE':42s} {'dims':>5s} | {'act net R2':>10s} {'act seq R2':>10s} | {'':>10s} {'appear R2':>10s}  (sources used)")
    srcs = [k for k in np.unique(si) if (te & (si == k)).sum() >= 80 and (tr & (si == k)).sum() >= 200]
    for name, X in rows.items():
        acc = {"net": [], "seq": [], "app": [], "w": []}
        for k in srcs:
            m = si == k; trk, tek = tr & m, te & m
            appk = PCA(16, random_state=0).fit(f0[trk]).transform(f0)
            acc["net"].append(ridge_r2(X, net, trk, tek)); acc["seq"].append(ridge_r2(X, seq, trk, tek))
            acc["app"].append(ridge_r2(X, appk, trk, tek)); acc["w"].append(int(tek.sum()))
        w = np.array(acc["w"], float); r = {q: float(np.average(acc[q], weights=w)) for q in ("net", "seq", "app")}
        out[name].update(within_action_net_r2=r["net"], within_action_seq_r2=r["seq"], within_appearance_r2=r["app"], within_sources=len(srcs))
        print(f"{name:42s} {X.shape[1]:5d} | {r['net']:10.3f} {r['seq']:10.3f} | {'':>10s} {r['app']:10.3f}  ({len(srcs)})", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "leakage_probes.json"), "w"), indent=1)
    print("[leak] ->", os.path.join(RESULTS, "leakage_probes.json"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["latp2_inv_ref", "latp2_anchor_ref", "tc_cont_ref"])
