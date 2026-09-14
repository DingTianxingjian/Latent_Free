"""Stage III metrics -- does the transport operator REALISE the donor's action on a new state?  Offline, from a
validation dump.  The reader G is a per-source ridge fitted ONLY on the real frozen dF -> net xyz of the
training-split episodes; it never sees anything the tokenizer or the operator produced, so E(D(.)) ~ z
cannot satisfy it (the circularity of the old round-trip number).

    indep_self      G(D(s_j, z_j))   vs a_j      can the operator's own output be read at all
    indep_transfer  G(D(s_j, z_i))   vs a_i      the donor's action realised on j?           <- the number
    indep_replay    G(D(s_j, z_i))   vs a_j      or j's own motion replayed from the state?
    indep_matched   G(D(s_j, z_m))   vs a_j      matched (nearest-action) donor realises j's action?
    frozen_ref      G(dF_j)          vs a_j      the reader's own quality
Transfer is dissected as the reviews asked: NMSE = radial + angular, |p|/|a|, magnitude-weighted cosine,
scalar calibration alpha (fitted on half the test episodes, read on the other half), reader-input scale
sensitivity (gamma in {0.5, 2}), C_pair / C_real_pair / C_real_self; plus E_self / E_correct / E_wrong
(u-zeroed relative MSE on the real recipient future), donor discrimination and swap sensitivity.
Usage: python -m acdc_la.stage3_transport.probes rec_src swap_fz s3_latp ...  -> results/stage3_probes.json
"""
import sys, json, os
import numpy as np

from ..paths import RESULTS
from ..common import readers as R

COLS = ["frozen_ref_r2", "clean_r2_net", "indep_self_r2", "indep_self_cos", "indep_transfer_r2", "indep_transfer_cos", "indep_transfer_wcos",
        "indep_transfer_nmse_radial", "indep_transfer_nmse_angular", "indep_transfer_pred_over_true", "alpha", "indep_transfer_r2_calibrated",
        "indep_transfer_cos_gamma0.5", "indep_transfer_cos_gamma2.0", "indep_replay_r2", "indep_replay_cos", "indep_matched_r2",
        "E_self", "E_correct", "E_wrong", "donor_discrim_acc", "swap_sens", "C_pair_centered", "C_real_pair", "C_real_self"]


def dissect(Y, P):
    ym = Y.mean(0); Yc, Pc = Y - ym, P - ym
    r, p, c = np.linalg.norm(Yc, axis=1), np.linalg.norm(Pc, axis=1), R.cosr(Yc, Pc)
    den = (Yc ** 2).sum()
    return dict(r2=R.r2(Y, P), cos=float(c.mean()), nmse=float(((Pc - Yc) ** 2).sum() / den), nmse_radial=float(((p - r) ** 2).sum() / den),
                nmse_angular=float((2 * p * r * (1 - c)).sum() / den), wcos=float((Yc * Pc).sum() / (r * p).sum()), pred_over_true=float(p.mean() / r.mean()))


def per_dump(d):
    DF, NET, GE, SI = d["DF"].astype(np.float32), d["NET"], d["GE"], d["SI"]
    n = len(NET); part = np.arange(n) ^ 1; SAME = d["SAME"] > 0.5 if "SAME" in d else np.ones(n, bool)
    rows = {}
    for s, tr, te in R.source_splits(GE, SI):
        G, _ = R.fit_reader(DF, NET, GE, tr)
        row = dict(n_test=int(te.sum()), frozen_ref_r2=R.r2(NET[te], G(DF[te])))
        Gz, _ = R.fit_reader(d["Z"].astype(np.float64), NET, GE, tr); row["clean_r2_net"] = R.r2(NET[te], Gz(d["Z"][te]))
        if "DSELF" in d:
            for k, v in dissect(NET[te], G(d["DSELF"][te].astype(np.float32))).items():
                if k in ("r2", "cos"): row[f"indep_self_{k}"] = v
        if "DMATCH" in d: row["indep_matched_r2"] = R.r2(NET[te], G(d["DMATCH"][te].astype(np.float32)))
        sel = te & SAME
        if "DSW" in d and sel.sum() >= 10:
            Y, P = d["ASW_NET"][sel], G(d["DSW"][sel].astype(np.float32))
            for k, v in dissect(Y, P).items(): row[f"indep_transfer_{k}"] = v
            for k, v in dissect(NET[sel], P).items():
                if k in ("r2", "cos"): row[f"indep_replay_{k}"] = v
            ym = Y.mean(0); Yc, Pc = Y - ym, P - ym
            eps = GE[sel]; ue = np.unique(eps); rs = np.random.RandomState(7); rs.shuffle(ue); half = np.isin(eps, ue[: len(ue) // 2])
            r2c, al = [], []
            for f, g in ((half, ~half), (~half, half)):
                if f.sum() < 5 or g.sum() < 5: continue
                a_ = max(0.0, float((Pc[f] * Yc[f]).sum() / ((Pc[f] ** 2).sum() + 1e-9))); al.append(a_); r2c.append(R.r2(Y[g], ym + a_ * Pc[g]))
            if r2c: row["indep_transfer_r2_calibrated"], row["alpha"] = float(np.mean(r2c)), float(np.mean(al))
            for gm in (0.5, 2.0): row[f"indep_transfer_cos_gamma{gm}"] = float(R.cosr(Yc, G(gm * d["DSW"][sel].astype(np.float32)) - ym).mean())
        for k, key in (("E_self", "E_SELF"), ("E_correct", "E_CORRECT"), ("E_wrong", "E_WRONG"), ("swap_sens", "SENS")):
            if key in d: row[k] = float(d[key][te].mean())
        if "E_CORRECT" in d: row["donor_discrim_acc"] = float((d["E_CORRECT"][te] < d["E_WRONG"][te]).mean())
        i = np.where(te)[0]; j = part[i]; ok = (j < n) & (SI[j] == s) & te[np.minimum(j, n - 1)]; i, j = i[ok], j[ok]
        if len(i) >= 10:
            sm = NET[SI == s].mean(0); Gj = G(DF[j])
            row["C_pair_centered"] = float(R.cosr(NET[i] - sm, NET[j] - sm).mean())
            row["C_real_pair"] = float(R.cosr(Gj - sm, NET[i] - sm).mean()); row["C_real_self"] = float(R.cosr(Gj - sm, NET[j] - sm).mean())
        rows[d["names"][s]] = row
    return R.aggregate(rows)


def main(specs):
    res_path = os.path.join(RESULTS, "stage3_probes.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    tags = []
    for spec in specs:
        d = R.load_arrays(spec); tag = os.path.basename(d["path"]).replace("_val.npz", "").replace("swaplam_", "").replace("_readout_arrays.npz", "")
        res[tag] = per_dump(d); res[tag]["path"] = d["path"]; tags.append(tag)
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"[{tag}] transfer r2 {res[tag].get('indep_transfer_r2', np.nan):.3f} cos {res[tag].get('indep_transfer_cos', np.nan):.3f} "
              f"calibrated {res[tag].get('indep_transfer_r2_calibrated', np.nan):.3f} clean {res[tag].get('clean_r2_net', np.nan):.3f}", flush=True)
    R.table(res, COLS, tags)


if __name__ == "__main__":
    main(sys.argv[1:] or ["rec_src", "swap_fz"])
