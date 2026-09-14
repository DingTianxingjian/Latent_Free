"""Isolate the Wan-VAE encode failure: is it the VAE alone across many calls, or
the interleaved 5B-DiT forward corrupting state?  Prints the first failing iter."""
import os, sys, traceback
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
from features_wan import IMG_SIZE, load_vae
import features_dit as fd

dev = "cuda"
data.build_index()
fr = data.load_episode_frames(data.list_episodes()[0])
vae = load_vae(dev)

def prep(frames_u8):
    x = torch.from_numpy(frames_u8).to(dev).permute(0,3,1,2).float().div_(255.0)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.mul_(2).sub_(1)
    return [x.permute(1,0,2,3).contiguous()]

print("=== A) VAE-only, encode fr[100:105] 30x ===", flush=True)
for i in range(30):
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = vae.encode(prep(fr[100:105]))
    except Exception as e:
        print(f"  VAE-only FAILED at iter {i}: {type(e).__name__}: {str(e)[:80]}", flush=True); break
else:
    print("  VAE-only: 30/30 OK", flush=True)

print("=== B) VAE + DiT forward interleaved, 30x ===", flush=True)
dit = fd.load_dit(dev)
hooks = fd.Hooks(dit, fd.LAYERS)
name, sigma, ts = fd.pick_timesteps()[0]
for i in range(30):
    try:
        _ = fd.encode_clip(dit, vae, hooks, fr[100:105], sigma, ts, dev)
    except Exception as e:
        print(f"  VAE+DiT FAILED at iter {i}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        traceback.print_exc(); break
else:
    print("  VAE+DiT: 30/30 OK", flush=True)
print("DEBUG DONE", flush=True)
