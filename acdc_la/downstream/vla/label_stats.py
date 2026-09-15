"""What does a training token accuracy of 0.82 (univla arm) vs 0.41 (acdc arm) actually mean?

Token accuracy is not comparable across the two arms: they predict a different number of tokens from a different
vocabulary, and above all their label distributions have very different floors.  For N sampled LIBERO-spatial windows
(the same RLDS pipeline and augmentation the VLA arms train on) this measures, per arm:

  majority          accuracy of a policy that always emits the most frequent code at each position (the real floor)
  entropy           empirical entropy of the label, bits per sample (upper bound: n_tokens * log2(codebook))
  usage             distinct codes actually used, per position and overall
  stability         agreement between the label of a window and the label of the SAME window under one more draw of
                    the training augmentation (crop 0.9, brightness 0.2, contrast/saturation 0.8-1.2, hue 0.05).
                    A label that flips under augmentation is not predictable from the observation: it caps accuracy.

  python -m acdc_la.downstream.vla.label_stats --arms univla acdc --n 1000
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
from PIL import Image

from ...paths import RESULTS, LF, UNIVLA_ROOT
sys.path.insert(0, UNIVLA_ROOT)
import torchvision.transforms as T
from prismatic.vla.datasets import RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from .datasets_ctrl import RLDSBatchTransformCtrl
from .finetune_libero_ctrl import build_labeller, Cfg

JIT = T.Compose([T.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1.0, 1.0)), T.ColorJitter(0.2, (0.8, 1.2), (0.8, 1.2), 0.05)])   # the pipeline delivers 224x224


class Collect(RLDSBatchTransformCtrl):
    """Labels only: the clean window as the pipeline delivers it, and the same window jittered once more."""
    def __call__(self, batch):
        imgs = batch["observation"]["image_primary"]; ro = 0
        idx = self.tokens(imgs, ro)
        g = torch.Generator().manual_seed(random.randrange(1 << 30)); torch.manual_seed(int(g.initial_seed()))
        jit = np.stack([np.asarray(JIT(Image.fromarray(im))) for im in imgs])          # one shared draw for the window
        idx2 = self.tokens(jit, ro)
        return dict(idx=idx.cpu().numpy().reshape(-1), idx2=idx2.cpu().numpy().reshape(-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["univla", "acdc"]); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--acdc-codebook", default="vq_s1_acdc_1ep_pq2x64.npz"); ap.add_argument("--out", default="vla_label_stats")
    a = ap.parse_args(); dev = 0 if torch.cuda.is_available() else "cpu"; R = {"n": a.n}
    for arm in a.arms:
        t0 = time.time(); cfg = Cfg(latent=arm, acdc_codebook=a.acdc_codebook)
        labeller, n_tokens, Kc = build_labeller(cfg, dev)
        tr = Collect(labeller, arm, None, None, T.ToTensor(), PurePromptBuilder, window_size=cfg.window_size)
        ds = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, tr, resize_resolution=(224, 224), shuffle_buffer_size=4000,
                         image_aug=True, window_size=cfg.window_size + 2, training_phase="post-training")
        I, I2 = [], []
        for k, ex in enumerate(ds):
            I.append(ex["idx"]); I2.append(ex["idx2"])
            if k + 1 >= a.n: break
            if (k + 1) % 200 == 0: print(f"[labelstats:{arm}] {k+1}/{a.n} ({time.time()-t0:.0f}s)", flush=True)
        I, I2 = np.stack(I), np.stack(I2)
        maj = np.array([np.bincount(I[:, j], minlength=Kc).max() / len(I) for j in range(I.shape[1])])
        ent = np.array([-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(I[:, j], minlength=Kc) / len(I)) for j in range(I.shape[1])])
        used = [int((np.bincount(I[:, j], minlength=Kc) > 0).sum()) for j in range(I.shape[1])]
        R[arm] = dict(n_tokens=int(n_tokens), codebook=int(Kc), chance=1.0 / Kc,
                      majority=float(maj.mean()), majority_per_pos=[round(float(x), 3) for x in maj],
                      entropy_bits_per_token=float(ent.mean()), entropy_bits_per_sample=float(ent.sum()),
                      max_bits_per_sample=float(I.shape[1] * np.log2(Kc)), codes_used=used,
                      stability=float((I == I2).mean()), stability_per_pos=[round(float(x), 3) for x in (I == I2).mean(0)])
        print(f"[labelstats:{arm}] {R[arm]}", flush=True)
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    print("%-8s %7s %7s %9s %10s %11s %10s" % ("arm", "tokens", "words", "chance", "majority", "bits/sample", "stability"))
    for arm in a.arms:
        r = R[arm]; print("%-8s %7d %7d %9.3f %10.3f %6.1f/%4.0f %10.3f" % (arm, r["n_tokens"], r["codebook"], r["chance"], r["majority"], r["entropy_bits_per_sample"], r["max_bits_per_sample"], r["stability"]))
    print("[labelstats-done]", flush=True)


if __name__ == "__main__":
    main()
