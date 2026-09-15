"""How many BITS of its own latent label does each trained VLA actually recover at deployment?

The right instrument for the cross-method question (retraction of 2026-09-15).  `common/predictability.py` regresses
a continuous code from pooled DINO features with no language on held-out OXE sources: valid within one family, wrong
for UniVLA, whose deployed code is 4 indices into a 16-entry codebook read by a 7B VLM that is given the instruction.
Here the reader IS the trained policy and the units are the ones it optimises:

    residual = mean cross-entropy at the latent positions, in bits per sample
    H        = empirical entropy of the label on the same windows, in bits per sample
    recovered = H - residual                    "how much of its own label the policy can infer from what it sees"

and the same three restricted to AUGMENTATION-STABLE windows -- those whose label does not change under one further
draw of the training augmentation.  A label that flips under a nuisance re-draw is not a function of the scene, so
bits recovered on unstable windows are bits of our own augmentation noise, not of the action.

    python -m acdc_la.downstream.vla.token_bits --run univla+libero_spatial_no_noops+b16x4+lr0.00035+r32+s15000
    python -m acdc_la.downstream.vla.token_bits --run <acdc run> --dataset libero_object_no_noops      # held-out suite

NOTE on "held out": the LoRA fine-tune ran on the RLDS TRAIN split, so --split val is genuinely held out and is the
default here.  --split train gives the in-sample number that the 0.82 / 0.41 training token accuracies correspond to;
report the pair.
"""
import os, sys, json, time, math, random, argparse
import numpy as np
import torch
from PIL import Image

from ...paths import RESULTS, LF, UNIVLA_ROOT
sys.path.insert(0, UNIVLA_ROOT)
import torchvision.transforms as T
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets import RLDSDataset
from .datasets_ctrl import RLDSBatchTransformCtrl
from .finetune_libero_ctrl import build_labeller, Cfg, LATENT_TOKEN_BASE

JIT = T.Compose([T.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1.0, 1.0)), T.ColorJitter(0.2, (0.8, 1.2), (0.8, 1.2), 0.05)])
IGNORE = -100


class WithStability(RLDSBatchTransformCtrl):
    """The training example, plus the label the SAME window gets under one more draw of the augmentation."""
    def __call__(self, b):
        # the parent draws ro = randint(0, 1) as its first random call; re-seeding and re-drawing reproduces it, so
        # the stability pair is measured on the SAME window whose cross-entropy we record
        seed = random.randrange(1 << 30)
        random.seed(seed); out = super().__call__(b)
        random.seed(seed); ro = random.randint(0, 1)
        imgs = b["observation"]["image_primary"]
        with torch.no_grad():
            jit = np.stack([np.asarray(JIT(Image.fromarray(im))) for im in imgs])
            out["idx1"] = self.tokens(imgs, ro).reshape(-1).cpu()
            out["idx2"] = self.tokens(jit, ro).reshape(-1).cpu()
        return out


def collate(rows, pad_id):
    n = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.bool)
    for i, r in enumerate(rows):
        L = len(r["input_ids"]); ids[i, :L] = r["input_ids"]; lab[i, :L] = r["labels"]; att[i, :L] = True
    return ids, lab, att, torch.stack([r["pixel_values"] for r in rows])


def bits_table(nats_per_tok, idx, n_tokens, Kc):
    """nats_per_tok: (N, n_tokens) cross-entropy; idx: (N, n_tokens) labels.  Everything in bits per SAMPLE."""
    resid = float(nats_per_tok.sum(1).mean() / math.log(2))
    H = float(sum(-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(idx[:, j], minlength=Kc) / len(idx))
                  for j in range(n_tokens)))
    return dict(n=len(idx), residual_bits=resid, label_entropy_bits=H, recovered_bits=H - resid,
                recovered_frac=(H - resid) / H if H > 0 else float("nan"), max_bits=n_tokens * math.log2(Kc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory name under acdc_la/ckpt/vla_runs")
    ap.add_argument("--dataset", default="libero_spatial_no_noops")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="val = the RLDS validation split, which the LoRA fine-tune (train=True) never saw -- the held-out number")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="token_bits")
    a = ap.parse_args()
    dev = 0
    rd = os.path.join(LF, "acdc_la", "ckpt", "vla_runs", a.run)
    meta = json.load(open(os.path.join(rd, "ctrl_meta.json")))
    arm, n_tokens, Kc = meta["latent"], int(meta["n_tokens"]), int(meta["codebook_size"])
    assert Kc > 0, f"{arm} is a continuous arm: it has no token cross-entropy (use latent R2 instead)"
    print(f"[bits] {a.run} arm={arm} n_tokens={n_tokens} codebook={Kc} dataset={a.dataset}/{a.split}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(rd, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(rd, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa", trust_remote_code=True).to(dev).eval()
    n_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    cfg = Cfg(latent=arm, tc_tag=meta.get("tc_tag", "tc_cont"), acdc_tag=meta.get("acdc_tag", "s1_acdc_1ep"),
              acdc_codebook=meta.get("acdc_codebook", "vq_s1_acdc_1ep_pq2x64.npz"))
    labeller, nt, kc = build_labeller(cfg, dev)
    assert (nt, kc) == (n_tokens, Kc), (nt, kc, n_tokens, Kc)
    tr = WithStability(labeller, arm, processor.tokenizer, processor.image_processor.apply_transform,
                       T.ToTensor(), PurePromptBuilder, window_size=cfg.window_size)
    ds = RLDSDataset(cfg.data_root_dir, a.dataset, tr, resize_resolution=(224, 224), shuffle_buffer_size=4000,
                     image_aug=True, window_size=cfg.window_size + 2, training_phase="post-training", train=(a.split == "train"))

    CE, IDX, STA, rows, t0 = [], [], [], [], time.time()
    for k, ex in enumerate(ds):
        rows.append(ex)
        if len(rows) == a.batch or k + 1 >= a.n:
            ids, lab, att, pix = collate(rows, processor.tokenizer.pad_token_id)
            with torch.no_grad():
                out = vla(input_ids=ids.to(dev), attention_mask=att.to(dev), pixel_values=pix.to(torch.bfloat16).to(dev))
            lg = out.logits[:, n_patches:-1].float().cpu(); gt = lab[:, 1:]
            m = gt > LATENT_TOKEN_BASE
            for i in range(len(rows)):
                sel = m[i]
                assert int(sel.sum()) == n_tokens, (int(sel.sum()), n_tokens)
                CE.append(torch.nn.functional.cross_entropy(lg[i][sel], gt[i][sel], reduction="none").numpy())
                IDX.append(rows[i]["latent_action_idx"].numpy().astype(int).reshape(-1))
                STA.append(bool((rows[i]["idx1"] == rows[i]["idx2"]).all()))
            rows = []
            if (k + 1) % 100 < a.batch: print(f"[bits] {k+1}/{a.n} ({time.time()-t0:.0f}s)", flush=True)
        if k + 1 >= a.n: break
    CE, IDX, STA = np.stack(CE), np.stack(IDX), np.array(STA)

    R = dict(run=a.run, arm=arm, dataset=a.dataset, split=a.split, n_tokens=n_tokens, codebook=Kc,
             stable_frac=float(STA.mean()), all=bits_table(CE, IDX, n_tokens, Kc))
    if STA.sum() >= 50: R["stable"] = bits_table(CE[STA], IDX[STA], n_tokens, Kc)
    if (~STA).sum() >= 50: R["unstable"] = bits_table(CE[~STA], IDX[~STA], n_tokens, Kc)
    p = os.path.join(RESULTS, f"{a.out}.json"); allr = json.load(open(p)) if os.path.exists(p) else {}
    allr[f"{a.run}@{a.dataset}.{a.split}"] = R; json.dump(allr, open(p, "w"), indent=1)
    print(f"[bits] stable window fraction {R['stable_frac']:.3f}")
    print("%-10s %6s %14s %14s %14s %10s" % ("subset", "n", "residual bits", "label H bits", "recovered", "frac"))
    for k in ("all", "stable", "unstable"):
        if k in R:
            r = R[k]; print("%-10s %6d %14.2f %14.2f %14.2f %10.3f" % (k, r["n"], r["residual_bits"], r["label_entropy_bits"], r["recovered_bits"], r["recovered_frac"]))
    print("[bits-done]", flush=True)


if __name__ == "__main__":
    main()
