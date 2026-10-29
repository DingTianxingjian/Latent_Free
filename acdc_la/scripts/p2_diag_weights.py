"""How far did each phase-2 arm move OpenVLA, and where?  (2026-10-29, diagnosis of fft step 5000 vs the 1/10 LoRA)

Relative weight change ||W - W0||_F / ||W0||_F per component, against the openvla-7b base:
  fft   : the consolidated full-parameter checkpoint (vla_full.pt) minus the base
  lora  : the phase-2 adapter's own delta, scale * B @ A per targeted Linear (PEFT, lora_alpha / r), plus the
          embed_tokens copy it trains (modules_to_save)
Also checks that both arms' heads files hold the same keys and load strictly into the FlowHeads the trainer builds
(post-training loads heads with strict=False, so a silent mismatch would go unnoticed there).  CPU only, ~50 GB RAM.

    python acdc_la/scripts/p2_diag_weights.py --fft <run with vla_full.pt> --lora_adapter <adapters/<p2 run>> \
        --lora_run <p2 run> --out acdc_la/results/p2_diag_weights.json
"""
import argparse, json, os, re, sys
from collections import defaultdict
import torch
from safetensors.torch import load_file

LF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LF, os.path.join(LF, "third_party", "UniVLA")]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(LF, "cache", "vla", "openvla-7b"))
ap.add_argument("--fft", required=True); ap.add_argument("--lora_adapter", required=True); ap.add_argument("--lora_run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def group(k):
    if "vision_backbone.featurizer." in k: return "vision_dino"
    if "vision_backbone.fused_featurizer." in k: return "vision_siglip"
    if k.startswith("projector."): return "projector"
    if "embed_tokens" in k: return "embed_tokens"
    if "lm_head" in k: return "lm_head"
    if "self_attn" in k: return "llm_attn"
    if "mlp" in k: return "llm_mlp"
    if "layernorm" in k or k.endswith("norm.weight") or ".norm." in k: return "norms"
    return "other"


base = {}
for f in sorted(os.listdir(a.base)):
    if f.endswith(".safetensors"): base.update(load_file(os.path.join(a.base, f)))
print(f"[diag] base: {len(base)} tensors", flush=True)
res = {}

# ---- full-parameter arm
fft = torch.load(os.path.join(a.fft, "vla_full.pt"), map_location="cpu")
num, den, n_new = defaultdict(float), defaultdict(float), 0
for k, w in fft.items():
    if k not in base: n_new += 1; continue
    w0 = base[k].float(); w1 = w.float()
    if w0.shape != w1.shape:                      # resized embeddings / lm_head: compare the original rows
        n = min(w0.shape[0], w1.shape[0]); w0, w1 = w0[:n], w1[:n]
    g = group(k); num[g] += float((w1 - w0).pow(2).sum()); den[g] += float(w0.pow(2).sum())
res["fft_rel_change"] = {g: (num[g] / den[g]) ** 0.5 for g in sorted(num)}
res["fft_rel_change_all"] = (sum(num.values()) / sum(den.values())) ** 0.5
res["fft_keys_not_in_base"] = n_new
print("[diag] fft relative change:", json.dumps(res["fft_rel_change"], indent=1), flush=True)
del fft

# ---- LoRA arm
cfg = json.load(open(os.path.join(a.lora_adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
ad = load_file(os.path.join(a.lora_adapter, "adapter_model.safetensors"))
num, den = defaultdict(float), defaultdict(float)
for k in ad:
    if not k.endswith("lora_A.weight"): continue
    kb = k.replace("lora_A.weight", "lora_B.weight")
    tgt = re.sub(r"^base_model\.model\.", "", k.replace(".lora_A.weight", ".weight"))
    if tgt not in base: print("[diag] lora target not in base:", tgt); continue
    w0 = base[tgt].float()
    d = scale * (ad[kb].float().flatten(1) @ ad[k].float().flatten(1))          # Linear, Conv2d (patch embed) or Embedding
    if d.numel() != w0.numel():                   # embed_tokens: PEFT's embedding LoRA is (dim, vocab) and the vocab was resized
        d = d.T if d.shape[1] != w0.shape[1] else d
        n = min(d.shape[0], w0.shape[0]); d, w0 = d[:n], w0[:n]
    d = d.reshape(w0.shape)
    g = group(tgt); num[g] += float(d.pow(2).sum()); den[g] += float(w0.pow(2).sum())
for k in ad:                                      # modules_to_save copies (embed_tokens), however PEFT named them
    if "lora_" not in k:
        tgt = re.sub(r"^base_model\.model\.", "", k).replace("modules_to_save.default.", "").replace("modules_to_save.", "")
        if tgt not in base: print("[diag] adapter key not matched:", k); continue
        if True:
            w0 = base[tgt].float(); w1 = ad[k].float(); n = min(w0.shape[0], w1.shape[0])
            g = group(tgt); num[g] += float((w1[:n] - w0[:n]).pow(2).sum()); den[g] += float(w0[:n].pow(2).sum())
res["lora_rel_change"] = {g: (num[g] / den[g]) ** 0.5 for g in sorted(num)}
res["lora_scale"] = scale; res["lora_r"] = cfg["r"]; res["lora_targets"] = cfg.get("target_modules")
print("[diag] lora relative change:", json.dumps(res["lora_rel_change"], indent=1), flush=True)

# ---- heads: same keys, strict load into the trainer's FlowHeads
from acdc_la.downstream.vla.finetune_libero_ctrl import FlowHeads
heads = {}
for name, d in (("fft", a.fft), ("lora", a.lora_run)):
    sd = torch.load(os.path.join(d, "action_decoder.pt"), map_location="cpu"); heads[name] = sd
    meta = json.load(open(os.path.join(d, "ctrl_meta.json"))); fk = meta["flow"]
    h = FlowHeads(meta["dz"], meta["window_size"], width=fk["width"], layers=fk["layers"], z_tok=fk["z_tok"])
    r = h.load_state_dict(sd, strict=False)
    res[f"heads_{name}"] = dict(n=len(sd), missing=list(r.missing_keys), unexpected=list(r.unexpected_keys),
                                finite=all(bool(torch.isfinite(v.float()).all()) for v in sd.values()),
                                rms=float(torch.cat([v.float().flatten() for v in sd.values()]).pow(2).mean().sqrt()))
res["heads_same_keys"] = sorted(heads["fft"]) == sorted(heads["lora"])
print("[diag] heads:", {k: v for k, v in res.items() if k.startswith("heads")}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("[diag] wrote", a.out)
