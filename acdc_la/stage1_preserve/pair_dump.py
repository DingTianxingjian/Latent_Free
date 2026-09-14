"""Pair-wise use of a Stage-I tokenizer (UniVLA's usage, in-house).  A tokenizer trained on T=2 clips
(K=1: the DC cell of one stride segment, 4 cells x d) is applied to each of the K segments of a T-frame
validation clip and its K codes are CONCATENATED -- a per-segment code sequence, K*4*d dims -- and written
as a dump readable by stage1_preserve.probes.  With d = 32 that is 512 dims for K = 4 (vs 256 for the
DC+AC1 clip code); `--d 16` at training time gives 256, the bit-matched comparison.
    python -m acdc_la.stage1_preserve.pair_dump --tag s1_pair --eval-T 5
"""
import os, json, argparse
import numpy as np
import torch

from ..paths import RESULTS, CKPT
from ..common import features as Fz
from ..common.model import load_ckpt
from ..common.pipeline import Featurizer
from ..common import loaders
from ..data.distractor import Distractor, patch_trajectory
from ..common.features import quadrants


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="a K=1 tokenizer trained with --T 2"); ap.add_argument("--eval-T", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--sources", nargs="*", default=None); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--val-per-source", type=int, default=400); ap.add_argument("--limit-eps", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=10 ** 9)
    a = ap.parse_args(); a.T, a.steps, a.seed = a.eval_T, 8, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, dstats, ck = load_ckpt(os.path.join(CKPT, f"{a.tag}.pt"), dev); assert model.K == 1, "pair_dump needs a T=2 tokenizer"
    model.eval(); K = a.eval_T - 1
    a.tag = f"{a.tag}_x{K}"
    srcs, ds, dv, dl, dl_val = loaders.build(a); loaders.describe(a, srcs, ds, dv, dev)
    dino = Fz.load_dino(dev); feat = Featurizer(dino, K, "acdc", 2, dstats, dev); dis = Distractor(dev)
    one = Fz.dct_basis(1, 1, dev)                                          # the single DC cell of a segment
    @torch.no_grad()
    def codes(dF):                                                          # (B,K,16,768) -> (B, K*4*d): per-segment codes, concatenated
        return torch.cat([model.encode(Fz.effect_cells(quadrants(dF[:, k:k + 1]), one))[0].flatten(1) for k in range(dF.shape[1])], 1)
    A = {k: [] for k in ("Z", "ZP", "ZREV", "DF", "DFP", "F0Q", "ACT", "GE", "SI", "PATCH_XY")}
    with torch.no_grad():
        for kb, batch in enumerate(dl_val):
            if kb >= a.val_batches: break
            b = feat(batch, dis); h16 = lambda t: quadrants(t).flatten(1).half().cpu()
            A["Z"].append(codes(b["dF"]).cpu()); A["ZP"].append(codes(b["dF_patch"]).cpu()); A["ZREV"].append(codes(-b["dF"].flip(1)).cpu())
            A["DF"].append(h16(b["dF"])); A["DFP"].append(h16(b["dF_patch"])); A["F0Q"].append(quadrants(b["f0"]).flatten(1).half().cpu())
            A["ACT"].append(b["act"]); A["GE"].append(b["ge"].cpu()); A["SI"].append(b["si"])
            A["PATCH_XY"].append(torch.from_numpy(patch_trajectory(b["ge"].cpu().numpy(), a.eval_T)))
    out = {k: torch.cat(v).numpy() for k, v in A.items()}
    for k in ("Z", "ZP", "ZREV", "ACT", "PATCH_XY"): out[k] = out[k].astype(np.float32)
    out["names"] = np.array(dv.src, dtype=object); out["K"] = K; out["n_cells"] = 4 * K; out["d"] = model.d
    np.savez_compressed(os.path.join(RESULTS, f"{a.tag}_val.npz"), **out)
    json.dump(dict(tag=a.tag, pairwise_from=ck["args"].get("tag"), n_val_clips=int(len(out["Z"])), dims=int(out["Z"].shape[1])), open(os.path.join(RESULTS, f"{a.tag}.json"), "w"), indent=1)
    print(f"[{a.tag}-done] {len(out['Z'])} clips, {out['Z'].shape[1]} dims", flush=True)


if __name__ == "__main__":
    main()
