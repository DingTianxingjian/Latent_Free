"""UniVLA stage-2 baseline dump on the SAME validation clips, in this repo's dump format, so every stage's
probe compares against UniVLA on identical data:
    Z   = the deployed task-centric code z_q of the K consecutive stride pairs, concatenated (K*4*128)
    U   = the task-irrelevant code z_q_uncontrol, concatenated                          (Stage II's TI channel)
    ZP, UP  the same on the moving-patch rendering;  ZREV the same on the reversed clip
    DF, DFP, F0Q, ACT, PATCH_XY, GE, SI, names  from this repo's Featurizer (identical to the ACDC dumps)
No decodes are written (UniVLA's decoder path is the legacy eval_mix.py); Stage-III probes skip those keys.
Model loading reuses the audited port in claude_survey/eval_rala2.py (`official_stage2` = released lam-stage-2.ckpt;
any `s2_*` tag = a stage-2 model retrained on our data).  A UniVLA pair is 2 frames `stride` apart, so the K pairs of
a T-frame clip give UniVLA exactly the frames the ACDC tokenizer sees.
    python -m acdc_la.baselines.univla --arm official_stage2 --tag univla_official
"""
import os, sys, json, argparse, time
import numpy as np
import torch

from ..paths import RESULTS, CLAUDE_SURVEY, LF
from ..common import features as Fz
from ..common.pipeline import Featurizer
from ..common import loaders
from ..data.distractor import Distractor, patch_trajectory
from ..common.features import quadrants

for p in (CLAUDE_SURVEY, LF, os.path.join(LF, "diagnostics")):
    sys.path.insert(0, p)


@torch.no_grad()
def codes(model, frames, dev):
    """frames (B,T,3,224,224) in [0,1] -> z_q (B, K*4*128), z_q_unc (B, K*4*128)."""
    B, T = frames.shape[:2]; ZQ, ZU = [], []
    for k in range(T - 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            o = model.vq_encode(frames[:, k:k + 2])
        ZQ.append(o["z_q"].reshape(B, -1).float()); ZU.append(o["z_q_uncontrol"].reshape(B, -1).float())
    return torch.cat(ZQ, 1), torch.cat(ZU, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="official_stage2"); ap.add_argument("--tag", default="univla_official")
    ap.add_argument("--sources", nargs="*", default=None); ap.add_argument("--T", type=int, default=5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--val-per-source", type=int, default=400)
    ap.add_argument("--limit-eps", type=int, default=None); ap.add_argument("--val-batches", type=int, default=10 ** 9)
    ap.add_argument("--dstats", type=float, default=None, help="dF rms of the ACDC dumps (default: read from ckpt/s1_acdc.pt or legacy rec_src)")
    a = ap.parse_args(); a.steps, a.seed = 8, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import eval_rala2 as E
    model, T_model, _ = E.load_arm(a.arm, dev); assert T_model == 2
    srcs, ds, dv, dl, dl_val = loaders.build(a); loaders.describe(a, srcs, ds, dv, dev)
    dino = Fz.load_dino(dev)
    if a.dstats is None:
        from ..common.model import load_ckpt, legacy_path
        from ..paths import CKPT
        p = os.path.join(CKPT, "s1_acdc.pt"); p = p if os.path.exists(p) else legacy_path("rec_src")
        a.dstats = float(torch.load(p, map_location="cpu")["dstats"])
    feat = Featurizer(dino, a.T - 1, "acdc", 2, a.dstats, dev); dis = Distractor(dev)
    A = {k: [] for k in ("Z", "U", "ZP", "UP", "ZREV", "DF", "DFP", "F0Q", "ACT", "GE", "SI", "PATCH_XY")}
    t0 = time.time()
    for k, batch in enumerate(dl_val):
        if k >= a.val_batches: break
        b = feat(batch, dis)
        z, u = codes(model, b["frames"], dev); zp, up = codes(model, dis.render(b["frames"], b["ge"]), dev)
        zr, _ = codes(model, b["frames"].flip(1), dev)
        h16 = lambda t: quadrants(t).flatten(1).half().cpu()
        A["Z"].append(z.cpu()); A["U"].append(u.cpu()); A["ZP"].append(zp.cpu()); A["UP"].append(up.cpu()); A["ZREV"].append(zr.cpu())
        A["DF"].append(h16(b["dF"])); A["DFP"].append(h16(b["dF_patch"])); A["F0Q"].append(quadrants(b["f0"]).flatten(1).half().cpu())
        A["ACT"].append(b["act"]); A["GE"].append(b["ge"].cpu()); A["SI"].append(b["si"])
        A["PATCH_XY"].append(torch.from_numpy(patch_trajectory(b["ge"].cpu().numpy(), a.T)))
        if (k + 1) % 20 == 0: print(f"[{a.tag}] {k+1} batches {time.time()-t0:.0f}s", flush=True)
    out = {k: torch.cat(v).numpy() for k, v in A.items()}
    for k in ("Z", "U", "ZP", "UP", "ZREV", "ACT", "PATCH_XY"): out[k] = out[k].astype(np.float32)
    out["names"] = np.array(dv.src, dtype=object); out["K"] = a.T - 1; out["n_cells"] = a.T - 1; out["d"] = 4 * 128
    np.savez_compressed(os.path.join(RESULTS, f"{a.tag}_val.npz"), **out)
    json.dump(dict(tag=a.tag, arm=a.arm, n_val_clips=int(len(out["Z"])), baseline="univla_stage2", args=vars(a)),
              open(os.path.join(RESULTS, f"{a.tag}.json"), "w"), indent=1)
    print(f"[{a.tag}-done] {len(out['Z'])} clips", flush=True)


if __name__ == "__main__":
    main()
