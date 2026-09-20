"""Did LATP's own deployment prior learn the deployment mapping, and does it survive the move to LIBERO?

The tokenizer carries a prior head `p(z | F_0, language)` (`common/tcvq.py:prior_code`, review 2026-09-15) that sees
only the deployment information set.  The VLA arms never use it: their labels are the POSTERIOR code (it reads the
future frames) and the VLA re-learns the mapping from the prompt through its query tokens.  This script prices the
prior directly, on the HELD-OUT LIBERO split (`train[95%:]`, which no arm trained on), against the same target the
VLA is trained to hit:

    z*        posterior code  E_post(f0, dF)         (the label; standardised by z_stats, as the VLA sees it)
    z_prior   LATP's prior    P_LATP(f0, lang)       (current frame + T5 instruction, no future)
    z_VLA     the fine-tuned VLA's query-token prediction, flow-sampled exactly as at deployment (--vla <run dir>)

One pass, one set of rows, so R2(z_prior, z*) and R2(z_VLA, z*) are comparable.  R2 uses the project convention
(`common/probes._r2`: dims that do not vary on these rows are excluded).

    python -m acdc_la.downstream.vla.prior_gate --tag latp2_inv --suite libero_goal --n 2000 \
        [--vla acdc_la/ckpt/vla_runs/tc_cont+libero_goal_no_noops+...--latp2inv_fmq4_frac10]
"""
import argparse, json, os, sys

import numpy as np
import torch
import torchvision.transforms as transforms

from ...paths import LF, CKPT, IMG, RESULTS, UNIVLA_ROOT
from ...common import features as Fz
from ...common.probes import _r2
sys.path.insert(0, UNIVLA_ROOT)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder      # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO   # noqa: E402
from prismatic.vla.datasets import RLDSDataset                              # noqa: E402
from .tcvq_tokens import TCVQLatentTokenizer                                # noqa: E402
from . import datasets_ctrl as _dc                                          # noqa: E402
from .datasets_ctrl import RLDSBatchTransformCtrl                           # noqa: E402

# The training transform draws ro in {0, 1} per sample; the probe has to know which window the label came from, so
# every row here uses ro = 0 (and therefore no history tokens in the prompt) for the label, the prior and the VLA alike.
_dc.random.randint = lambda lo, hi: 0

T5_DIR = os.path.join(UNIVLA_ROOT, "t5-base")


class LangT5:
    """Masked-mean pooled frozen t5-base, the `c` the tokenizer was trained with (claude_survey/lang_cache.py +
    oxe_mix's `langvec`: mean of last_hidden_state over the attention mask)."""

    def __init__(self, dev):
        from transformers import T5EncoderModel, T5Tokenizer
        self.tok = T5Tokenizer.from_pretrained(T5_DIR)
        self.enc = T5EncoderModel.from_pretrained(T5_DIR).to(dev).eval().requires_grad_(False)
        self.dev, self.cache = dev, {}

    @torch.no_grad()
    def __call__(self, text):
        if text not in self.cache:
            e = self.tok([text], return_tensors="pt", truncation=False).to(self.dev)
            h = self.enc(**e).last_hidden_state[0]                       # (L, 768)
            m = e["attention_mask"][0].bool()
            self.cache[text] = (h[m].mean(0)).float()
        return self.cache[text]


class PriorProbeTransform:
    """The training transform (identical prompt / query tokens / label) plus the prior's own prediction for the row."""

    def __init__(self, labeller, base, lang, stride, T):
        self.labeller, self.base, self.lang, self.stride, self.T = labeller, base, lang, stride, T

    @torch.no_grad()
    def __call__(self, rlds_batch):
        out = self.base(rlds_batch)                                      # z* in out["latent_action_idx"], VLA inputs
        imgs = rlds_batch["observation"]["image_primary"]
        frames = [imgs[k * self.stride] for k in range(self.T)]          # ro = 0, the deterministic window
        x = torch.as_tensor(np.stack([np.asarray(f) for f in frames])).to(self.labeller.dev)
        x = x.permute(0, 3, 1, 2).float().div_(255.)
        x = torch.nn.functional.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.labeller.dev != "cpu")):
            g4 = Fz.dino_g4(self.labeller.dino, x).float()
        f0 = g4[:1]
        c = self.lang(rlds_batch["task"]["language_instruction"].decode().lower())[None]
        zp = self.labeller.tok.prior_code(f0, c).reshape(-1)
        out["z_prior"] = ((zp - self.labeller.zmean) / self.labeller.zstd).cpu()
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="latp2_inv"); ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vla", default=None, help="run dir of a trained continuous arm: adds z_VLA on the same rows")
    ap.add_argument("--flow_steps", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    labeller = TCVQLatentTokenizer(a.tag, dev=dev)
    lang = LangT5(dev)
    ds_name = f"{a.suite}_no_noops"

    vla = heads = None
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    src = a.vla or os.path.join(LF, "cache", "vla", "openvla-7b")
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    meta = {}
    if a.vla:
        from .finetune_libero_ctrl import FlowHeads, ContHeads
        meta = json.load(open(os.path.join(a.vla, "ctrl_meta.json")))
        assert meta.get("tc_tag") == a.tag, (meta.get("tc_tag"), a.tag)
        vla = AutoModelForVision2Seq.from_pretrained(a.vla, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                     trust_remote_code=True).to(dev).eval().requires_grad_(False)
        fk, dz = meta.get("flow", {}), int(meta["dz"])
        heads = (FlowHeads(dz, meta["window_size"], width=fk["width"], layers=fk["layers"], z_tok=fk["z_tok"])
                 if meta.get("head") == "flow" else ContHeads(dz, meta["window_size"], text_direct=meta.get("head") == "reg_text"))
        heads.load_state_dict(torch.load(os.path.join(a.vla, "action_decoder.pt"), map_location="cpu"))
        heads = heads.to(dev).eval().requires_grad_(False)
    else:                                                    # the labels need the <ACT_i> vocabulary either way
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(labeller.codebook_size or 16)]})

    nq = int(meta.get("query_tokens") or 4)
    base = RLDSBatchTransformCtrl(labeller, "tc_cont", processor.tokenizer, image_transform=processor.image_processor.apply_transform,
                                  image_transform_lam=transforms.ToTensor(), prompt_builder_fn=PurePromptBuilder,
                                  window_size=meta.get("window_size", 12), query_tokens=nq)
    tf = PriorProbeTransform(labeller, base, lang, meta.get("window_size", 12) // (labeller.T - 1), labeller.T)
    ds = RLDSDataset(os.path.join(LF, "data", "libero_rlds"), ds_name, tf, resize_resolution=(IMG, IMG),
                     shuffle_buffer_size=2048, image_aug=False, window_size=meta.get("window_size", 12) + 2,
                     training_phase="post-training", train=False)        # train=False -> train[95%:], the held-out side
    collator = PaddedCollatorForActionPrediction_LIBERO(processor.tokenizer.model_max_length,
                                                        processor.tokenizer.pad_token_id, padding_side="right")
    steps = int(a.flow_steps or meta.get("flow", {}).get("steps", 10))
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    Z, P, V, buf = [], [], [], []
    for s in ds:
        buf.append(s)
        if len(buf) < a.batch: continue
        Z += [b["latent_action_idx"].numpy() for b in buf]; P += [b["z_prior"].numpy() for b in buf]
        if vla is not None:
            batch = collator(buf)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                          pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev), labels=None, output_hidden_states=True)
            npatch = vla.vision_backbone.featurizer.patch_embed.num_patches
            vis = out.hidden_states[-1][:, :npatch].float(); txt = out.hidden_states[-1][:, npatch:]
            m = batch["input_ids"].to(dev) > 32000
            txt = torch.stack([txt[i][m[i]] for i in range(len(txt))]).float()
            zv = (heads.sample(vis, txt, steps, gen)[1] if meta.get("head") == "flow" else heads(vis, txt)[1])
            V += list(zv.float().cpu().numpy())
        buf = []
        if len(Z) >= a.n: break
    Z, P = np.stack(Z[:a.n]), np.stack(P[:a.n])
    res = dict(tag=a.tag, suite=a.suite, split="train[95%:]", n=len(Z), dz=int(Z.shape[1]), vla=a.vla,
               flow_steps=steps if a.vla else None, r2_prior=_r2(Z, P),
               r2_prior_per_token=[_r2(Z[:, i * (Z.shape[1] // labeller.tok.n_tokens):(i + 1) * (Z.shape[1] // labeller.tok.n_tokens)],
                                       P[:, i * (Z.shape[1] // labeller.tok.n_tokens):(i + 1) * (Z.shape[1] // labeller.tok.n_tokens)])
                                   for i in range(labeller.tok.n_tokens)],
               prior_rms=float(np.sqrt((P ** 2).mean())), target_rms=float(np.sqrt((Z ** 2).mean())))
    if V:
        V = np.stack(V[:a.n]); res["r2_vla"] = _r2(Z, V); res["vla_rms"] = float(np.sqrt((V ** 2).mean()))
        res["r2_prior_vs_vla"] = _r2(V, P)
    out = a.out or os.path.join(RESULTS, f"prior_gate_{a.tag}_{a.suite}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True); print(f"[prior-gate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
