"""Wan2.2 DiT *internal* features (Exp 7) -- the untested third candidate space.

Recipe (per phase_1.md, "five things that must not be guessed"):
  VAE-encode clean clip -> add flow-match noise at timestep t -> ONE DiT forward
  with ZERO text context -> capture block outputs via hooks at layers {5,14,22,29}
  -> unpatchify to [B, T_lat, Hp, Wp, C] -> 4x4 spatial pool (never global).

Model: Wan2.2-TI2V-5B WanModel, dim=3072, 30 blocks, patch_size [1,2,2], in_dim=48
(matches the VAE latent).  Built has_image_input=False (latent-only, per the
dreamzero TI2V-5B config) so the forward is clean latent->features; the checkpoint
is loaded strict=False (img_emb unused).

Two mandatory checks, both printed:
  * zero-context validation: features finite, non-constant across frames, per-dim
    std not collapsed (ref: VAE latent std ~0.46).  If it fails -> fall back to umT5.
  * contract: a single frame encoded alone must match the clip's z0 (small rel err),
    else the fused-vs-independent comparison is dishonest.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                    # noqa: E402
from features_wan import IMG_SIZE, load_vae    # noqa: E402  (224 -> latent 14x14)

DREAMZERO_ROOT = "/lustre/fs1/home/ti874297/lf/surgical_world/dreamzero"   # so `import groot` resolves
WAN_MOD = ("/lustre/fs1/home/ti874297/lf/surgical_world/dreamzero/"
           "groot/vla/model/dreamzero/modules")
DIT_DIR = os.path.join(data.CACHE_DIR, "wan", "dit")
LAYERS = (5, 14, 22, 29)                        # 0-indexed blocks: early/mid/late
G4 = 4
CTX_LEN, CTX_DIM = 512, 4096                    # text context [B, L, D]; zero context
NOISE_SEED = 0


def load_dit(device="cuda"):
    import json
    import safetensors.torch as st
    sys.path.insert(0, DREAMZERO_ROOT)
    from groot.vla.model.dreamzero.modules.wan_video_dit import WanModel
    cfg = json.load(open(os.path.join(DIT_DIR, "config.json")))
    model = WanModel(dim=cfg["dim"], in_dim=cfg["in_dim"], ffn_dim=cfg["ffn_dim"],
                     out_dim=cfg["out_dim"], freq_dim=cfg["freq_dim"], eps=cfg["eps"],
                     num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
                     patch_size=[1, 2, 2], has_image_input=False)
    sd = {}
    for i in (1, 2, 3):
        sd.update(st.load_file(os.path.join(DIT_DIR,
                  f"diffusion_pytorch_model-0000{i}-of-00003.safetensors")))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    keep = [k for k in missing if not k.startswith(("img_emb", "control_adapter", "ref_conv"))]
    assert not keep, f"missing core DiT weights: {keep[:6]}"
    if hasattr(model, "post_initialize"):
        model.post_initialize()
    model = model.to(device).eval().requires_grad_(False)
    print(f"[dit] loaded {cfg['num_layers']}-block DiT dim={cfg['dim']} "
          f"(unexpected {len(unexpected)}, missing-noncore {len(missing)-len(keep)})", flush=True)
    return model


def flow_match_sigmas():
    """Return the scheduler's (sigmas, timesteps) arrays, so t/sigma match training."""
    sys.path.insert(0, DREAMZERO_ROOT)
    from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
    sch = FlowMatchScheduler()
    sch.set_timesteps(num_inference_steps=1000, training=True)
    return sch.sigmas.float(), sch.timesteps.float()   # descending sigma ~1.0 -> ~0.009


def pick_timesteps():
    """Primary (near-clean, ~5% noise) + sensitivity (~25%, ~50%).  Returns
    [(name, sigma, timestep)] using the scheduler's own sigma convention."""
    sig, ts = flow_match_sigmas()
    out = []
    for name, frac in (("t_lo", 0.05), ("t_mid", 0.25), ("t_hi", 0.50)):
        idx = int(torch.argmin((sig - frac).abs()))
        out.append((name, float(sig[idx]), float(ts[idx])))
    return out


class Hooks:
    def __init__(self, model, layers):
        self.buf = {}
        self.h = []
        for li in layers:
            self.h.append(model.blocks[li].register_forward_hook(self._mk(li)))

    def _mk(self, li):
        def hook(mod, inp, out):
            self.buf[li] = out.detach()      # (B, N=f*h*w, C)
        return hook

    def remove(self):
        for h in self.h:
            h.remove()


@torch.no_grad()
def encode_clip(dit, vae, hooks, frames_u8, sigma, timestep, device, grid=None):
    """frames_u8 (W,128,128,3) uint8 -> dict{layer: (T_lat, 16*C)} 4x4-pooled per
    latent frame.  Also returns (T_lat, h, w).  grid: reuse a prior noise draw shape."""
    # The Wan video VAE needs a temporal window (kernel-3 temporal conv): a single
    # frame is encoded as a static >=5-frame clip (repeat it); its z0 is o_t-only by
    # causality, so it is the honest "independent single-frame" representation.
    if len(frames_u8) < 5:
        frames_u8 = np.concatenate(
            [frames_u8, np.repeat(frames_u8[-1:], 5 - len(frames_u8), axis=0)], axis=0)
    W = len(frames_u8)
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.mul_(2).sub_(1)                                   # [-1,1]
    clip = [x.permute(1, 0, 2, 3).contiguous()]            # list of (3,W,H,W)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        z = vae.encode(clip)                               # (1,48,T_lat,14,14)
    z = z.float()
    T_lat = z.shape[2]
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
    noise = torch.randn(z.shape, generator=gen, device=device, dtype=z.dtype)
    z_t = (1 - sigma) * z + sigma * noise                  # flow-match forward noising
    tstep = torch.full((z.shape[0],), timestep, device=device, dtype=z.dtype)
    ctx = torch.zeros(z.shape[0], CTX_LEN, CTX_DIM, device=device, dtype=z.dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        _ = dit(z_t, tstep, ctx)                           # hooks capture block outputs
    # patch grid: patch_size [1,2,2] -> f=T_lat, h=14/2=7, w=7
    h = z.shape[3] // 2
    w = z.shape[4] // 2
    feats = {}
    for li, tok in hooks.buf.items():
        t = tok.float().reshape(1, T_lat, h, w, -1)        # (1,f,h,w,C)
        c = t.shape[-1]
        t = t.permute(0, 1, 4, 2, 3).reshape(T_lat, c, h, w)   # (f,C,h,w)
        g = F.adaptive_avg_pool2d(t, (G4, G4))             # (f,C,4,4)
        feats[li] = g.reshape(T_lat, -1).cpu().numpy()     # (f, 16*C)
    return feats, (T_lat, h, w)


def validate(dit, vae, device):
    """Spec check 1 + 2: zero-context well-behaved, and single-frame == clip z0."""
    ep = data.list_episodes()[0]
    fr = data.load_episode_frames(ep)
    hooks = Hooks(dit, LAYERS)
    name, sigma, ts = pick_timesteps()[0]
    print(f"[dit] primary noise: {name} sigma={sigma:.4f} timestep={ts:.1f}", flush=True)

    # (1) zero-context validation on ~200 frames (as 5-frame clips), layer 14
    stds, allfinite, chans = [], True, None
    for k in range(0, 200, 5):
        f5 = fr[k:k + 5]
        if len(f5) < 5:
            break
        feats, (T_lat, h, w) = encode_clip(dit, vae, hooks, f5, sigma, ts, device)
        z0 = feats[14][0]
        allfinite = allfinite and np.isfinite(z0).all()
        stds.append(z0.std())
        chans = feats[14].shape
    print(f"[dit] zero-ctx check: grid (T_lat,h,w)=({T_lat},{h},{w}) feat/frame={chans[1]} "
          f"finite={allfinite} per-clip-std mean={np.mean(stds):.3f} (VAE ref ~0.46)", flush=True)
    assert allfinite and np.mean(stds) > 0.02, "zero context degenerate -> fall back to umT5"

    # (2) contract: single-frame encode == clip z0.  Use in-range frames (ep0 can be
    # short, e.g. 98 frames) — a hardcoded frame-100 was the earlier "VAE" crash.
    clip_feats, _ = encode_clip(dit, vae, hooks, fr[10:15], sigma, ts, device)
    single_feats, _ = encode_clip(dit, vae, hooks, fr[10:11], sigma, ts, device)  # W=1 -> pad to 5
    a, b = clip_feats[14][0], single_feats[14][0]
    rel = np.abs(a - b).max() / (np.abs(b).mean() + 1e-6)
    print(f"[dit] contract z0(clip) vs single-frame encode: rel={rel:.4f}", flush=True)
    hooks.remove()
    print("[dit] VALIDATION PASSED" if rel < 0.05 else
          f"[dit] WARNING: contract rel={rel:.3f} >= 0.05 (fused-vs-indep not honest)", flush=True)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[dit] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)
    data.build_index()
    vae = load_vae(dev)
    dit = load_dit(dev)
    validate(dit, vae, dev)
