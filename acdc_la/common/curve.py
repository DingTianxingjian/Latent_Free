"""Cheap proxy evaluation DURING pretraining (PROTOCOL.md Tier 0 + Tier 1), at every 10% of the sample budget.

Runs the standard validation dump on a small stratified subset (100 clips per source, ~2.9k clips), then:
  Tier 0 (health)  rec, kl_z, kl_u, swap sensitivity, effective rank of z (exp entropy of the normalised singular values),
                   fraction of z dims with posterior std < 0.9 (collapsed-to-prior dims)
  Tier 1 (representation)  per-source ridge z -> net / seg1 / prof / seq, patch drop, global nuisance leak N_TC (patch motion
                   readable from z on the patch rendering), N_TC_clean as the zero check
Appends one row to results/<tag>_curve.json and prints one line, so a run that is not moving on the proxies that carry the
claim (profile vs the pooled baseline, nuisance leak) can be stopped before the budget is spent.
"""
import os, json, time
import numpy as np
import torch

from ..paths import RESULTS
from .dump import dump_val
from ..common import readers as R
from ..common.probes import per_source_readouts


def health(d):
    Z = d["Z"].astype(np.float64); Zc = Z - Z.mean(0)
    sv = np.linalg.svd(Zc, compute_uv=False); p = sv ** 2 / (sv ** 2).sum()
    return dict(eff_rank=float(np.exp(-(p * np.log(p + 1e-12)).sum())), z_std_mean=float(Z.std(0).mean()),
                swap_sens=float(np.median(d["SENS"])), rec=float(d["agg_rec"]), kl_z=float(d["agg_kl_z"]))


def curve_eval(model, feat, dl_curve, distractor, tag, step, total_steps, train_stats=None):
    t0 = time.time(); os.environ.setdefault("ACDC_MIN_CLIPS", "20")
    from ..paths import CKPT
    torch.save(dict(state_dict=model.state_dict(), step=step, n_cells=model.n_cells, K=model.K, d=model.d, d_u=model.d_u,
                    dstats=float(feat.dstats), args=dict(tag=tag, T=model.K + 1)), os.path.join(CKPT, f"{tag}_step{step}.pt"))   # matched-budget snapshot
    path = os.path.join(RESULTS, f"{tag}_curve_step{step}_val.npz")
    dump_val(model, feat, dl_curve, distractor, path)
    d = R.load_arrays(path); row = dict(step=step, frac=step / total_steps, **(train_stats or {}), **health(d))
    P = per_source_readouts(d["Z"].astype(np.float64), dict(net=d["NET"], seg1=d["SEG1"], prof=d["PROF"], seq=d["SEQ"]), d["GE"], d["SI"], d["names"],
                            extra={"net_patch": (d["ZP"], "net", None)}, min_clips=20)
    for k in ("net", "seg1", "prof", "seq", "net_patch"): row[f"r2_{k}"] = P.get(f"mean_r2_{k}", float("nan"))
    row["patch_drop_pct"] = float(100 * (1 - row["r2_net_patch"] / row["r2_net"])) if row["r2_net"] > 0.02 else float("nan")
    from ..stage2_disentangle.probes import nuisance_target, global_r2
    NUI = nuisance_target(d["PATCH_XY"]); row["N_TC"] = global_r2(d["ZP"].astype(np.float64), NUI, d["GE"]); row["N_TC_clean"] = global_r2(d["Z"].astype(np.float64), NUI, d["GE"])
    row["seconds"] = int(time.time() - t0)
    cp = os.path.join(RESULTS, f"{tag}_curve.json"); curve = json.load(open(cp)) if os.path.exists(cp) else []
    curve = [r for r in curve if r["step"] != step] + [row]; json.dump(curve, open(cp, "w"), indent=1)
    print(f"[{tag}] CURVE step {step} ({row['frac']:.0%}) rec {row['rec']:.3f} kl_z {row['kl_z']:.1f} eff_rank {row['eff_rank']:.0f} sens {row['swap_sens']:.2f} | "
          f"net {row['r2_net']:.3f} seg1 {row['r2_seg1']:.3f} prof {row['r2_prof']:.3f} seq {row['r2_seq']:.3f} drop {row['patch_drop_pct']:.0f}% N_TC {row['N_TC']:.3f} ({row['seconds']}s)", flush=True)
    return row
