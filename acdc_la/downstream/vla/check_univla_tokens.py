"""Is the univla arm of the controlled VLA comparison trained on UniVLA's OWN deployed tokens?

`finetune_libero_ctrl.py` loads `claude_survey/ckpt/s2mix_1ep.pt` (trained by rala2.py's subclass) into UniVLA's
own `ControllableDINOLatentActionModel` and labels windows with `vq_encode(...)["indices"]`.  This checks that the
two paths agree bit for bit on the same input, i.e. that the labels are the task-centric `vq_action` indices the
model was trained to emit, and reports exactly how much of the checkpoint came from the released stage 1."""
import sys, torch

sys.path.insert(0, "/anvil/scratch/x-tding2/latent_free/third_party/UniVLA")
sys.path.insert(0, "/anvil/scratch/x-tding2/latent_free/claude_survey")
torch.hub.set_dir("/anvil/scratch/x-tding2/latent_free/cache/torch/hub")
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
import eval_rala2 as E

dev = "cuda"
ck = torch.load("/anvil/scratch/x-tding2/latent_free/claude_survey/ckpt/s2mix_1ep.pt", map_location="cpu", weights_only=False)
sd = {k.replace("lam.", "", 1): v for k, v in ck["state_dict"].items()}
s1 = torch.load("/anvil/scratch/x-tding2/latent_free/cache/univla/lam-stage-1.ckpt", map_location="cpu", weights_only=False)["state_dict"]
s1 = {k.replace("lam.", "", 1): v for k, v in s1.items() if k.startswith("lam.")}
init = {k: v for k, v in s1.items() if ("vq" in k or "action_latent" in k)}      # genie/model.py:66-71, what stage 2 inherits
print(f"[check] stage-2 params {sum(v.numel() for v in sd.values())}; inherited at init {sum(v.numel() for v in init.items().__iter__().__next__()[1].reshape(-1).unsqueeze(0)) if False else sum(v.numel() for v in init.values())} "
      f"({', '.join(sorted(init))}); vq_action in stage 1: {any('vq_action' in k for k in s1)}", flush=True)
for k in sorted(init):
    if k in sd: print(f"[check] {k}: max|s2 - s1| = {float((sd[k].float() - init[k].float()).abs().max()):.4f}", flush=True)

m = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16, patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
res = m.load_state_dict(sd, strict=False); m = m.to(dev).eval()
print(f"[check] labeller class: {len(res.missing_keys)} missing (prefixes {sorted(set(k.split('.')[0] for k in res.missing_keys))}), unexpected {res.unexpected_keys}", flush=True)
m_rala, _, _ = E.load_arm("s2mix_1ep", dev); m_rala.eval()
g = torch.Generator().manual_seed(0); vid = torch.rand(4, 2, 3, 224, 224, generator=g).to(dev)
with torch.no_grad():
    a = m_rala.vq_encode(vid)["indices"].reshape(len(vid), -1).cpu()
    b = m.vq_encode(vid)["indices"].reshape(len(vid), -1).cpu()
print("[check] rala2 (training-time) :", a.tolist(), flush=True)
print("[check] labeller class        :", b.tolist(), flush=True)
print("[check] IDENTICAL:", bool(torch.equal(a, b)), flush=True)
