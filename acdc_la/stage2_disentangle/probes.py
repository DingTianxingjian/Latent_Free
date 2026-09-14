"""Stage II metric -- the separation gap.  Offline, from a validation dump that has both channels
(z = task-centric / effect, u = task-irrelevant / nuisance) and both renderings (clean, moving patch):

    S_TC = R2(action | z)        S_TI = R2(action | u)             action = net xyz of the clip (per-source ridge)
    N_TC = R2(nuisance | z_p)    N_TI = R2(nuisance | u_p)         nuisance = the patch's motion: net displacement (2) and
                                                                    per-step velocity (2*(T-1)), read on the patch rendering's
                                                                    codes.  The patch law is the same in every source, so this
                                                                    reader is GLOBAL (one episode-disjoint split over all clips,
                                                                    ~8k train rows); a per-source reader (270 rows, ~130
                                                                    episodes, per-episode target) reads nothing -- 2026-09-09.
                                                                    Position targets are not linearly readable from any code
                                                                    and are not used.
    G_sep = (S_TC - S_TI) + (N_TI - N_TC)                          in [-2, 2]; 2 = perfect assignment of both signals

Also reported: the model-free references R2(action | frozen dq) and R2(nuisance | frozen dq_patch) (what the
features carry at all), the patch drop of the action probe on z, and the relative shift ||z_p - z|| / ||z||.
For UniVLA (baselines/univla.py dump) z = z_q (TC head), u = z_q_uncontrol (TI head): the same formula on
the same clips is the comparison the paper line needs.
Usage: python -m acdc_la.stage2_disentangle.probes s2_tcti univla_official ...  -> results/stage2_probes.json
"""
import sys, json, os
import numpy as np

from ..paths import RESULTS
from ..common import readers as R

COLS = ["S_TC", "S_TI", "N_TC", "N_TI", "G_sep", "N_TC_clean", "N_TI_clean", "ref_action_dq", "ref_nuis_dqp", "patch_drop_pct", "z_shift_rel", "u_shift_rel"]


def nuisance_target(xy):
    """PATCH_XY (n,T,2) -> (n, 2 + 2*(T-1)): net displacement and per-step velocity, in pixels."""
    return np.concatenate([xy[:, -1] - xy[:, 0], (xy[:, 1:] - xy[:, :-1]).reshape(len(xy), -1)], 1).astype(np.float64)


def global_split(GE, seed=1):
    ug = np.unique(GE); rs = np.random.RandomState(seed); te = np.isin(GE, rs.choice(ug, len(ug) // 3, replace=False)); return ~te, te


def global_r2(X, Y, GE):
    tr, te = global_split(GE); G, _ = R.fit_reader(X, Y, GE, tr); return R.r2(Y[te], G(X[te]))


def per_dump(d, skip_ref=False):
    need = ("Z", "U", "ZP", "UP", "PATCH_XY")
    assert all(k in d for k in need), f"dump lacks {[k for k in need if k not in d]} (Stage II needs both channels and the patch rendering)"
    Z, U, ZP, UP = (d[k].astype(np.float64) for k in ("Z", "U", "ZP", "UP"))
    NET, GE, SI = d["NET"], d["GE"], d["SI"]; NUI = nuisance_target(d["PATCH_XY"])
    DF, DFP = d["DF"].astype(np.float32), d["DFP"].astype(np.float32)
    rows = {}
    for s, tr, te in R.source_splits(GE, SI):                        # action half: per source (units differ)
        row = dict(n_test=int(te.sum()))
        for name, X, Y in (("S_TC", Z, NET), ("S_TI", U, NET), ("ref_action_dq", DF, NET)):
            G, _ = R.fit_reader(X, Y, GE, tr); row[name] = R.r2(Y[te], G(X[te]))
        Gz, _ = R.fit_reader(Z, NET, GE, tr); rp = R.r2(NET[te], Gz(ZP[te]))
        row["patch_drop_pct"] = float(100 * (1 - rp / row["S_TC"])) if row["S_TC"] > 0.02 else float("nan")
        row["z_shift_rel"] = float(np.linalg.norm(ZP[te] - Z[te], axis=1).mean() / (np.linalg.norm(Z[te] - Z[tr].mean(0), axis=1).mean() + 1e-9))
        row["u_shift_rel"] = float(np.linalg.norm(UP[te] - U[te], axis=1).mean() / (np.linalg.norm(U[te] - U[tr].mean(0), axis=1).mean() + 1e-9))
        rows[d["names"][s]] = row
    out = R.aggregate(rows)
    for name, X in (("N_TC", ZP), ("N_TI", UP), ("N_TC_clean", Z), ("N_TI_clean", U)):   # nuisance half: global; *_clean = must be ~0 (no patch there)
        out[name] = global_r2(X, NUI, GE)
    if not skip_ref: out["ref_nuis_dqp"] = global_r2(DFP, NUI, GE)
    out["G_sep"] = (out["S_TC"] - out["S_TI"]) + (out["N_TI"] - out["N_TC"])
    return out


def main(specs):
    skip_ref = "--no-ref" in specs; specs = [s for s in specs if not s.startswith("--")]
    res_path = os.path.join(RESULTS, "stage2_probes.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    tags = []
    for spec in specs:
        d = R.load_arrays(spec); tag = os.path.basename(d["path"]).replace("_val.npz", "")
        res[tag] = per_dump(d, skip_ref); res[tag]["path"] = d["path"]; tags.append(tag); json.dump(res, open(res_path, "w"), indent=1)
        g = lambda k: res[tag].get(k, float("nan"))
        print(f"[{tag}] G_sep {g('G_sep'):+.3f}  S_TC {g('S_TC'):.3f} S_TI {g('S_TI'):.3f} N_TC {g('N_TC'):.3f} N_TI {g('N_TI'):.3f}  (sources >= min clips: {res[tag]['n_sources']})", flush=True)
    R.table(res, COLS, tags)


if __name__ == "__main__":
    main(sys.argv[1:])
