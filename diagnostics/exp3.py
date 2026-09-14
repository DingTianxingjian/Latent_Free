"""Experiment 3 (OPTIONAL bonus) -- probe a real pretrained LAM's latent action.

Checkpoint: UniVLA `lam-stage-1.ckpt` (qwbu/univla-latent-action-model), the
*uncontrolled* stage-1 latent action model.  Stage 1 is the vanilla, purely
self-supervised LAM -- no language conditioning -- so it is the closest released
model to the LAPA/Genie family the theory is about.  (Stage 2 is UniVLA's
task-centric variant, which is explicitly designed to suppress task-irrelevant
motion; it needs a language embedding per sample and is not run here.)

The model is a VQ-VAE over DINOv2 patch tokens: it maps a frame pair to
num_codes=4 tokens drawn from a codebook of 16 entries, each 128-d.  We probe

  z_q  the quantized latent action (4x128 -> 512-d) -- what a downstream VLA
       actually consumes, and
  z    the pre-quantization continuous latent -- the same information before the
       VQ bottleneck, which bounds how much the bottleneck itself costs.

Protocol is byte-for-byte the Exp-1 protocol (same episode split, same pairs,
same targets, same ridge/logistic sweep), so the row drops straight into the
Exp-1 table.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import data
import probes

UNIVLA_ROOT = os.path.join(data.ROOT, "third_party", "UniVLA")
CKPT = os.path.join(data.ROOT, "cache", "univla", "lam-stage-1.ckpt")
TORCH_HOME = os.path.join(data.ROOT, "cache", "torch")
IMG_SIZE = 224

# from latent_action_model/config/lam-stage-1.yaml
LAM_KW = dict(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
              patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12,
              dropout=0.0)


def load_lam(device):
    """Instantiate the stage-1 LAM directly (bypassing the Lightning wrapper,
    which drags in wandb/piq) and load the released weights."""
    os.environ["TORCH_HOME"] = TORCH_HOME
    torch.hub.set_dir(os.path.join(TORCH_HOME, "hub"))
    cwd = os.getcwd()
    sys.path.insert(0, UNIVLA_ROOT)
    os.chdir(UNIVLA_ROOT)          # lam.py hard-codes the relative path './t5-base'
    try:
        from latent_action_model.genie.modules.lam import (
            UncontrolledDINOLatentActionModel)
        model = UncontrolledDINOLatentActionModel(**LAM_KW)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("lam."):]: v for k, v in sd.items() if k.startswith("lam.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
    finally:
        os.chdir(cwd)
    missing = [k for k in missing if "text_encoder" not in k]
    print(f"[exp3] loaded {CKPT}\n       missing={missing}\n"
          f"       unexpected={[u for u in unexpected][:5]}")
    assert not missing, f"unexpected missing weights: {missing}"
    return model.to(device).eval()


@torch.no_grad()
def encode_pairs(model, stride, device="cuda", batch=32, limit=None):
    """Encode every (o_t, o_{t+stride}) pair into the LAM latent action."""
    index = data.build_index()
    eps = data.list_episodes()
    if limit:
        eps = eps[:limit]
    is_val = index["is_val"]

    Zq, Zc, IDX, Yc, Yg, EPv = [], [], [], [], [], []
    for e, ep in enumerate(eps):
        frames = data.load_episode_frames(ep)
        x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255.)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False)
        t, _, _, yc, yg = data.episode_pairs(index, e, stride)
        if len(t) == 0:
            continue
        tt = torch.from_numpy(t).to(device)
        for k in range(0, len(t), batch):
            sl = tt[k:k + batch]
            # (B, 2, 3, H, W): the model applies its own ImageNet normalization
            vids = torch.stack([x[sl], x[sl + stride]], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                out = model.vq_encode(vids)
            B = vids.shape[0]
            # UniVLA's VectorQuantizer.forward returns (z_q, z, x, indices) and
            # vq_encode unpacks it as (z_q, z, emb, indices).  Both "z_q" and "z"
            # are the QUANTIZED code -- z_q = x + (z - x).detach() is just the
            # straight-through estimator, numerically equal to z.  The continuous
            # pre-quantization latent is "emb".  (Reading "z" here yields R^2
            # identical to z_q, which is how this was caught.)
            Zq.append(out["z_q"].reshape(B, -1).float().cpu().numpy())
            Zc.append(out["emb"].reshape(B, -1).float().cpu().numpy())
            IDX.append(out["indices"].reshape(B, -1).cpu().numpy())
        Yc.append(yc); Yg.append(yg)
        EPv.append(np.full(len(t), e, dtype=np.int64))
        if (e + 1) % 50 == 0 or e == len(eps) - 1:
            print(f"  [exp3] stride {stride}: {e + 1}/{len(eps)} episodes",
                  flush=True)

    Zq = np.concatenate(Zq); Zc = np.concatenate(Zc); IDX = np.concatenate(IDX)
    Yc = np.concatenate(Yc); Yg = np.concatenate(Yg); EPv = np.concatenate(EPv)
    assert len(Zq) == len(Yc) == len(EPv), (Zq.shape, Yc.shape, EPv.shape)
    # the pre-VQ latent must actually differ from the quantized one, otherwise we
    # are reading the wrong key out of vq_encode and the "cost of the bottleneck"
    # row is meaningless
    assert not np.allclose(Zq, Zc), \
        "pre-VQ latent is identical to the quantized one -- wrong vq_encode key"
    quant_err = np.abs(Zq - Zc).mean() / (np.abs(Zc).mean() + 1e-12)
    print(f"[exp3] mean relative quantization error |z_q - emb|/|emb| = {quant_err:.3f}")
    val = is_val[EPv].astype(bool)
    used = len(np.unique(IDX))
    print(f"[exp3] stride {stride}: z_q {Zq.shape}, z {Zc.shape}, "
          f"{int((~val).sum())} train / {int(val.sum())} val pairs, "
          f"codebook entries used: {used}/{LAM_KW['num_latents']}")
    return Zq, Zc, IDX, Yc, Yg, val, used


def cached_encode(stride, device, batch, limit, model_box):
    """Encode a stride's pairs, caching to disk so the GPU stage can be resumed
    across short debug-queue slices."""
    path = os.path.join(data.CACHE_DIR, f"exp3_z_s{stride}.npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"[exp3] loaded cached latents {path} {d['Zq'].shape}")
        return (d["Zq"], d["Zc"], d["IDX"], d["Yc"], d["Yg"],
                d["val"].astype(bool), int(d["used"]))
    if model_box[0] is None:
        model_box[0] = load_lam(device)
    out = encode_pairs(model_box[0], stride, device, batch, limit)
    Zq, Zc, IDX, Yc, Yg, val, used = out
    np.savez(path, Zq=Zq, Zc=Zc, IDX=IDX, Yc=Yc, Yg=Yg, val=val, used=used)
    print(f"[exp3] cached latents -> {path}")
    return out


def run(strides=data.STRIDES, device="cuda", batch=32, limit=None):
    model_box = [None]          # lazily loaded only if something needs encoding
    rows = []
    for stride in strides:
        Zq, Zc, IDX, Yc, Yg, val, used = cached_encode(stride, device, batch,
                                                       limit, model_box)
        for name, Z in (("univla_lam_z_q", Zq), ("univla_lam_z_prevq", Zc)):
            res, _ = probes.run_probe_set(Z[~val], Z[val], Yc[~val], Yc[val],
                                          Yg[~val], Yg[val])
            print(f"[exp3] {name:20s} s={stride} R2 xyz={res['r2_xyz']:.3f} "
                  f"rot={res['r2_rot']:.3f} cont6={res['r2_cont6']:.3f} "
                  f"grip={res['bal_acc_grip']:.3f}", flush=True)
            for g in ("xyz", "rot", "cont6"):
                rows.append(dict(condition="clean", stride=stride, feature=name,
                                 dim_group=g, metric="r2",
                                 value=res[f"r2_{g}"],
                                 hyperparam=f"alpha={res['alpha']:g}",
                                 n_train=res["n_train"], n_val=res["n_val"],
                                 codebook_used=used, latent_dim=Z.shape[1]))
            rows.append(dict(condition="clean", stride=stride, feature=name,
                             dim_group="gripper", metric="balanced_accuracy",
                             value=res["bal_acc_grip"],
                             hyperparam=f"C={res['C']:g}",
                             n_train=res["n_train"], n_val=res["n_val"],
                             codebook_used=used, latent_dim=Z.shape[1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--strides", type=int, nargs="+", default=list(data.STRIDES),
                    help="run one stride per job on a short debug queue; "
                         "latents are cached so reruns are cheap")
    a = ap.parse_args()

    os.makedirs(data.RESULTS_DIR, exist_ok=True)
    df = run(strides=tuple(a.strides), device=a.device, batch=a.batch,
             limit=a.limit_episodes)
    out = os.path.join(data.RESULTS_DIR, "exp3_lam.csv")
    df.to_csv(out, index=False)
    print(f"[exp3] wrote {out}")
    print(df[df.dim_group.isin(["xyz", "rot", "cont6", "gripper"])].to_string(index=False))
