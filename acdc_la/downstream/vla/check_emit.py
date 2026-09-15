"""Does the merged VLA actually emit latent-action tokens?  A silent LoRA/embedding merge failure would leave the model
generating ordinary text, and the evaluation would still run and report 0 %.  This prints the generated ids for a few
LIBERO prompts: they must be > 32000 (the `<ACT_i>` range)."""
import sys, json, argparse
import torch
from ...paths import UNIVLA_ROOT
sys.path.insert(0, UNIVLA_ROOT)
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from PIL import Image
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
meta = json.load(open(f"{a.ckpt}/ctrl_meta.json")); n = int(meta["n_tokens"]); Kc = int(meta["codebook_size"])
AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
proc = AutoProcessor.from_pretrained(a.ckpt, trust_remote_code=True)
m = AutoModelForVision2Seq.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
                                           attn_implementation="sdpa").to("cuda").eval()
print(f"[emit] {meta['latent']}: n_tokens {n}, codebook {Kc}, tokenizer len {len(proc.tokenizer)}, embed rows {m.get_input_embeddings().weight.shape[0]}", flush=True)
g = torch.Generator().manual_seed(0)
for k, task in enumerate(["pick up the black bowl and place it on the plate", "open the top drawer and put the bowl inside"]):
    pb = PurePromptBuilder("openvla"); pb.add_turn("human", f"What action should the robot take to {task}?")
    img = Image.fromarray((torch.rand(256, 256, 3, generator=g).numpy() * 255).astype(np.uint8))
    inp = proc(pb.get_prompt(), img)
    inp = {kk: (v.to("cuda", dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda")) for kk, v in inp.items()}
    with torch.no_grad():
        out = m.generate(**inp, min_new_tokens=n, max_new_tokens=n, do_sample=True, temperature=0.75, top_p=0.9)
    ids = out[0, -n:].tolist()
    print(f"[emit] task {k}: ids {ids} -> latent? {[i > 32000 for i in ids]} -> codes {[i - 32001 for i in ids]}", flush=True)
print("[emit] DONE", flush=True)
