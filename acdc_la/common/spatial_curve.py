"""How much task-relevant information does the SHARED 256 -> 16 patch pooling destroy?

Every tokenizer in this repo, ours and the pooled ablations, reads a cached 4x4 DINO grid produced by a fixed 4x4
mean pool of the 16x16 patch grid (`common/features.py::dino_g4`).  UniVLA's LAM works on the patch grid directly.
That pooling is therefore a limitation of OUR pipeline, not a controlled difference -- but it is also a fixed LINEAR
operator P, so it COMMUTES with the temporal decomposition:

    dF~_k = P(F_{k+1}) - P(F_k) = P(dF_k)   =>   N~ = P(N)   and   A1~ = P(A1)

so it cannot turn DC into AC and does not touch the "endpoints cannot express A1" argument.  What it can destroy is
spatial frequency: motion entirely inside one cell averages away.  It may also REMOVE NUISANCE, which is the same
capacity-robustness trade the paper makes about latent capacity, one level down.

This measures it directly and cheaply -- no tokenizer, no training.  For a subset of validation clips, extract the
patch grid once, pool it to g x g for g in {16, 8, 4, 2}, build the SAME evidence the tokenizer would see,

    N_p = sum_k dF_{k,p}        A_p = sum_k b_1(k) dF_{k,p}

reduce every resolution to a common PCA dimension so the ridge is not comparing dimensionalities, and read the action
off it with the repo's standard leave-one-source-out ridge.  Repeat with the moving-patch distractor for the nuisance
column.  The output is a spatial-resolution curve directly comparable to the latent-capacity results.

    python -m acdc_la.common.spatial_curve --n-clips 20000 --grids 16 8 4 2
"""
import argparse, json, os, sys, time
import numpy as np
import torch

from ..paths import RESULTS, DINO_DIM
from . import features as Fz
from . import readers as R
from . import loaders
from .pipeline import clip_to_frames
from ..data.distractor import Distractor

COLS = ["r2_net", "r2_seg1", "r2_prof", "r2_seq", "r2_net_patch", "patch_drop_pct", "cells", "dims"]


def pool_grid(patches, g):
    """(N,256,768) 16x16 patch tokens -> (N,g*g,768) by mean pooling blocks of (16/g)^2."""
    n = patches.shape[0]; b = 16 // g
    return patches.view(n, 16, 16, DINO_DIM).view(n, g, b, g, b, DINO_DIM).mean((2, 4)).reshape(n, g * g, DINO_DIM)


def evidence(dF, b1):
    """(B,K,C,768) segment deltas -> [N, A1] flattened: exactly what the tokenizer's encoder is given."""
    N = dF.sum(1)
    A = torch.einsum("k,bkcd->bcd", b1.to(dF.dtype), dF)
    return torch.cat([N.flatten(1), A.flatten(1)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-clips", type=int, default=20000); ap.add_argument("--grids", type=int, nargs="+", default=[16, 8, 4, 2])
    ap.add_argument("--pca", type=int, default=256, help="common dimension for every resolution before the ridge")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-per-source", type=int, default=400); ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--out", default="spatial_curve")
    ap.add_argument("--sources", nargs="*", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-eps", type=int, default=None); ap.add_argument("--pool", default="acdc")
    ap.add_argument("--modes", type=int, default=2); ap.add_argument("--tag", default="spatial_curve")
    a = ap.parse_args(); a.steps = 0; a.d = 32; a.d_u = 8; a.layers = 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    srcs, ds, dv, dl, dl_val = loaders.build(a, start_step=0)
    dino = Fz.load_dino(dev); K = a.T - 1
    b1 = Fz.dct_basis(K, 2, dev)[1]
    distractor = Distractor(dev)
    store = {g: [] for g in a.grids}; store_p = {g: [] for g in a.grids}
    ACT, GE, SI = [], [], []; t0 = time.time(); n = 0
    for batch in dl_val:
        clip, ge, lrow, act, lv, si = batch
        frames = clip_to_frames(clip, dev)
        B, T = frames.shape[:2]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for key, fr in (("clean", frames), ("patch", distractor.render(frames, ge))):
                p = dino.forward_features((fr.reshape(B * T, *fr.shape[2:]) - Fz.IMNET_MEAN.to(dev)) / Fz.IMNET_STD.to(dev))
                p = p["x_norm_patchtokens"].float()                                    # (B*T,256,768)
                for g in a.grids:
                    cells = pool_grid(p, g).view(B, T, g * g, DINO_DIM)
                    dF = cells[:, 1:] - cells[:, :-1]
                    (store if key == "clean" else store_p)[g].append(evidence(dF, b1).cpu().numpy().astype(np.float32))
        ACT.append(act[:, :, :3].numpy()); GE.append(ge.numpy()); SI.append(si.numpy()); n += B
        if n % (20 * a.batch) < a.batch: print(f"[spatial] {n}/{a.n_clips} clips ({time.time()-t0:.0f}s)", flush=True)
        if n >= a.n_clips: break

    A = np.concatenate(ACT).astype(np.float64); GE = np.concatenate(GE); SI = np.concatenate(SI)
    names = [str(s) for s in srcs]
    tgt = dict(net=A.sum(1), seg1=A[:, 0], seq=A.reshape(len(A), -1), prof=(A - A.mean(1, keepdims=True)).reshape(len(A), -1))
    print(f"[spatial] {len(A)} clips, {len(np.unique(SI))} sources, targets {[f'{k}{v.shape}' for k, v in tgt.items()]}", flush=True)

    from ..stage1_preserve.probes import pca_reduce
    res = {}
    for g in a.grids:
        X = np.concatenate(store[g]); Xp = np.concatenate(store_p[g])
        raw = X.shape[1]
        Xr = pca_reduce(X, a.pca) if raw > a.pca else X                     # same dimension for every resolution
        Xpr = pca_reduce(Xp, a.pca) if raw > a.pca else Xp
        rows = {}
        for s, tr, te in R.source_splits(GE, SI):
            row = dict(n_test=int(te.sum()), cells=g * g, dims=int(Xr.shape[1]))
            for k, y in tgt.items():
                G, _ = R.fit_reader(Xr, y, GE, tr); row[f"r2_{k}"] = R.r2(y[te], G(Xr[te]))
            Gp, _ = R.fit_reader(Xpr, tgt["net"], GE, tr); row["r2_net_patch"] = R.r2(tgt["net"][te], Gp(Xpr[te]))
            row["patch_drop_pct"] = 100 * (1 - row["r2_net_patch"] / row["r2_net"]) if row["r2_net"] > 0.02 else float("nan")
            rows[names[s]] = row
        res[f"g{g}"] = R.aggregate(rows); res[f"g{g}"]["raw_dims"] = int(raw)
        print(f"[spatial] {g}x{g} = {g*g} cells (raw {raw}d -> {Xr.shape[1]}d): "
              f"net {res[f'g{g}']['r2_net']:.3f} seg1 {res[f'g{g}']['r2_seg1']:.3f} prof {res[f'g{g}']['r2_prof']:.3f} "
              f"seq {res[f'g{g}']['r2_seq']:.3f} patch_drop {res[f'g{g}']['patch_drop_pct']:.1f}%", flush=True)
        json.dump(res, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    R.table(res, COLS, [f"g{g}" for g in a.grids])
    print("[spatial-done]", flush=True)


if __name__ == "__main__":
    main()
