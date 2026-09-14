"""Does the decoder USE u?  For a Stage-II checkpoint, reconstruction of the patch rendering with its own u, with u
shuffled across the batch, and with u = 0.  If the three agree, u is decorative whatever its KL says.
    python -m acdc_la.stage2_disentangle.ushuffle --tags s2_tcti_bu s2_aug_bu s2_tcti
"""
import os, json, argparse
import numpy as np, torch, torch.nn.functional as F
from ..paths import RESULTS, CKPT
from ..common import features as Fz
from ..common.model import load_ckpt
from ..common.pipeline import Featurizer
from ..common import loaders
from ..data.distractor import Distractor


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+"); ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--T", type=int, default=5); ap.add_argument("--stride", type=int, default=4); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--val-per-source", type=int, default=100); ap.add_argument("--sources", nargs="*", default=None); ap.add_argument("--limit-eps", type=int, default=None)
    a = ap.parse_args(); a.steps, a.seed, a.tag = 8, 0, "ushuffle"; dev = "cuda"
    srcs, ds, dv, dl, dl_val = loaders.build(a); dino = Fz.load_dino(dev); dis = Distractor(dev); out = {}
    for tag in a.tags:
        model, dstats, _ = load_ckpt(os.path.join(CKPT, f"{tag}.pt"), dev); model.eval(); feat = Featurizer(dino, a.T - 1, "acdc", model.n_cells // 4, dstats, dev)
        acc = dict(own=[], shuffled=[], zero=[], kl_u=[])
        with torch.no_grad():
            for k, batch in enumerate(dl_val):
                if k >= a.batches: break
                b = feat(batch, dis); zmu, zlv, umu, ulv = model.encode(b["cells_patch"]); B = zmu.shape[0]
                acc["kl_u"].append(float(0.5 * (umu.pow(2) + ulv.exp() - 1 - ulv).sum(-1).mean()))
                for name, u in (("own", umu), ("shuffled", umu[torch.randperm(B, device=dev)]), ("zero", torch.zeros_like(umu))):
                    acc[name].append(float(F.mse_loss(model.decode(b["f0_patch"], b["c"], zmu, u), b["dF_patch"])))
        out[tag] = {k: float(np.mean(v)) for k, v in acc.items()}; out[tag]["d_u"] = model.d_u
        print(f"[ushuffle] {tag}: d_u {model.d_u} kl_u {out[tag]['kl_u']:.2f}  rec(patch) own {out[tag]['own']:.4f}  shuffled {out[tag]['shuffled']:.4f}  zero {out[tag]['zero']:.4f}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "ushuffle.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
