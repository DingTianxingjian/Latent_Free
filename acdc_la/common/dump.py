"""One validation pass -> one .npz.  Every metric in the repo is computed OFFLINE from this file
(`stage*/probes.py`, CPU), so the GPU pass is run once per checkpoint and every later probe reads
the same clips, the same codes and the same decodes.

Saved per val clip (val set = stratified nearest-action PAIRS per source, partner of i is i^1):
    Z, U                z_mu (n_cells*d) and u_mu (d_u) of the clean clip
    ZP, UP              same for the moving-patch rendering of the SAME clip
    ZREV                z_mu of the time-reversed clip
    DF, DFP             quadrant-pooled frozen dF, clean and patch (K*4*768, fp16) -- model-free references
    F0Q                 quadrant-pooled first-frame state (4*768, fp16)
    ACT                 (K,3) xyz displacement per segment;  PATCH_XY (T,2) patch centre per frame
    GE, SI, names       episode id, source index, source names
    DSELF               quadrants(decode(f0, c, z, u))                self-decode
    DSW, ASW, DONOR_GE, SAME   quadrants(decode(f0, c, z[pi], u)) with a same-source deranged donor pi, the donor's
                        action sequence (K,3), donor episode, and whether the donor is from the same source
    DMATCH, E_SELF, E_CORRECT, E_WRONG   u-zeroed decodes with the matched partner's z; relative MSE on the real future
    SENS                ||decode(z_i) - decode(z_k)|| / ||dF||  per clip (does z move the output at all)
"""
import os
import numpy as np
import torch

from .model import derangement
from .features import quadrants


@torch.no_grad()
def dump_val(model, feat, dl_val, distractor, path, max_batches=10 ** 9, extra_fn=None):
    model.eval()
    A = {k: [] for k in ("Z", "U", "ZP", "UP", "ZREV", "DF", "DFP", "F0Q", "ACT", "GE", "SI", "DSELF", "DSW", "ASW", "DONOR_GE",
                         "SAME", "DMATCH", "E_SELF", "E_CORRECT", "E_WRONG", "SENS", "PATCH_XY")}
    agg = {}
    dev = feat.dev
    for k, batch in enumerate(dl_val):
        if k >= max_batches: break
        b = feat(batch, distractor)
        B = b["dF"].shape[0]
        eb = getattr(model, "encode_batch", None)                 # models whose encoder needs the state, not just cells
        if eb is not None:
            zmu, zlv, umu, ulv = eb(b, "clean"); zp, _, up, _ = eb(b, "patch"); zr = eb(b, "rev")[0]
        else:
            zmu, zlv, umu, ulv = model.encode(b["cells"])
            zp, _, up, _ = model.encode(b["cells_patch"]); zr = model.encode(feat.cells_reversed(b["dF"]))[0]
        rec = model.decode(b["f0"], b["c"], zmu, umu)
        agg.setdefault("rec", []).append(float(torch.nn.functional.mse_loss(rec, b["dF"])))
        agg.setdefault("kl_z", []).append(float(0.5 * (zmu.pow(2) + zlv.exp() - 1 - zlv).sum(-1).mean()))
        ge = b["ge"]; pi = derangement(B, dev, ge); pk = derangement(B, dev, ge)
        usw = torch.zeros_like(umu) if getattr(model, "swap_u_zero", False) else umu     # PCRR: the cross path never sees the residual
        sw = model.decode(b["f0"], b["c"], zmu[pi], usw); sw2 = model.decode(b["f0"], b["c"], zmu[pk], usw)
        ia, ip = torch.arange(0, B, 2, device=dev), torch.arange(1, B, 2, device=dev)
        part = torch.empty(B, dtype=torch.long, device=dev); part[ia] = ip; part[ip] = ia
        u0 = torch.zeros_like(umu); nrm = b["dF"].flatten(1).pow(2).mean(1)
        rel = lambda d: ((d - b["dF"]).flatten(1).pow(2).mean(1) / nrm).cpu()
        dm = model.decode(b["f0"], b["c"], zmu[part], u0)
        h16 = lambda t: quadrants(t).flatten(1).half().cpu()
        A["Z"].append(zmu.flatten(1).cpu()); A["U"].append(umu.cpu()); A["ZP"].append(zp.flatten(1).cpu()); A["UP"].append(up.cpu())
        A["ZREV"].append(zr.flatten(1).cpu()); A["DF"].append(h16(b["dF"])); A["DFP"].append(h16(b["dF_patch"]))
        A["F0Q"].append(quadrants(b["f0"]).flatten(1).half().cpu()); A["ACT"].append(b["act"]); A["GE"].append(ge.cpu()); A["SI"].append(b["si"])
        A["DSELF"].append(h16(rec)); A["DSW"].append(h16(sw)); A["ASW"].append(b["act"][pi.cpu()]); A["DONOR_GE"].append(ge[pi].cpu())
        A["SAME"].append((b["si"][pi.cpu()] == b["si"]).float()); A["DMATCH"].append(h16(dm))
        A["E_SELF"].append(rel(model.decode(b["f0"], b["c"], zmu, u0))); A["E_CORRECT"].append(rel(dm))
        A["E_WRONG"].append(rel(model.decode(b["f0"], b["c"], zmu[derangement(B, dev, ge)], u0)))
        A["SENS"].append(((sw - sw2).flatten(1).norm(dim=1) / b["dF"].flatten(1).norm(dim=1).clamp_min(1e-3)).cpu())
        from ..data.distractor import patch_trajectory
        A["PATCH_XY"].append(torch.from_numpy(patch_trajectory(ge.cpu().numpy(), feat.K + 1)))
        if extra_fn is not None: extra_fn(A, b, model, zmu, umu)
    out = {k: torch.cat(v).numpy() for k, v in A.items() if v}
    for k in ("Z", "U", "ZP", "UP", "ZREV", "ACT", "ASW", "SAME", "SENS", "E_SELF", "E_CORRECT", "E_WRONG", "PATCH_XY"):
        out[k] = out[k].astype(np.float32)
    names = dl_val.dataset.dataset.src if isinstance(dl_val.dataset, torch.utils.data.Subset) else dl_val.dataset.src
    out["names"] = np.array(names, dtype=object)
    out["agg_rec"], out["agg_kl_z"] = np.float32(np.mean(agg["rec"])), np.float32(np.mean(agg["kl_z"]))
    out["n_cells"], out["d"], out["K"] = model.n_cells, model.d, feat.K
    np.savez_compressed(path, **out)
    model.train()
    return dict(rec=float(out["agg_rec"]), kl_z=float(out["agg_kl_z"]), n_val_clips=int(len(out["Z"])), n_val_eps=int(len(np.unique(out["GE"]))))


def load_dump(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
