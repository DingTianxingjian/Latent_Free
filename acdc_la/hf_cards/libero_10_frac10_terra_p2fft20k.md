# `libero_10_frac10_terra_p2fft20k` — LIBERO-**Long** (libero_10), 43 demos (10 %), full TERRA stack

OpenVLA-7B with **TERRA phase 2** (`p2fft_20k`: full-parameter, 20,000 steps on OXE, `latp2_inv` labels) + LoRA r32
+ flow-matching head + 4 query tokens + our latent channel (`--latent tc_cont`). Merged weights. **Replicate 2.**

This is the low-data cell on the hardest LIBERO suite, where nothing is saturated — unlike Goal at 10 %, where every
arm including the no-latent control lands at 87-89 %.

## Result — **303 / 500 = 60.6 %** (standard LIBERO-Long eval, 50 trials x 10 tasks)

| success | task |
|---|---|
| 0.92 | put both the cream cheese box and the butter in the basket |
| 0.86 | turn on the stove and put the moka pot on it |
| 0.70 | put both the alphabet soup and the cream cheese box in the basket |
| 0.68 | put the black bowl in the bottom drawer of the cabinet and close it |
| 0.68 | pick up the book and place it in the back compartment of the caddy |
| 0.64 | put the yellow and white mug in the microwave and close it |
| 0.58 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 0.46 | put both moka pots on the stove |
| 0.36 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| **0.18** | put both the alphabet soup and the tomato sauce in the basket — the floor in **every** arm |

## The cell — and why this number does **not** settle the phase-2 question

All arms at 10 % Long, 500 episodes each, same interface and budget:

| arm | rep 1 | rep 2 | mean |
|---|---|---|---|
| p2ours — LoRA-scale phase 2 | 61.6 | 55.8 | 58.7 |
| **p2fft20k — 20k full-parameter phase 2 (this)** | *not relayed* | **60.6** | — |
| p2fft5000 — 5k full-parameter phase 2 | 57.6 | 51.2 | 52.9 |
| nop2 — no phase 2 | 56.4 | — | 56.4 |
| univla_noinh | 53.0 | — | — |
| ours_fm | 50.8 | — | — |

**The run is confirmed; the method question is not.** 60.6 sits *inside* p2ours's own spread (55.8-61.6) and above
nop2's single 56.4, but one 500-episode cell carries **±8 points at 95 %** (pooled sd ~4.3 points over identical
configs), so 60.6 vs 58.7 vs 56.4 is **not separable**. This checkpoint gives the 20k arm its first 10 % Long number;
replicate 1 is still missing. On the full split the same two phase-2 variants were 89.8 vs 89.0 — also "the split
cannot tell".

**Caveat that applies to every number in this suite:** Long episodes average ~203 steps against a 14-frame sample
window, so ~6 % of training samples had a padding-contaminated window. Identical across arms, so comparisons are
unaffected. (The real-robot pipeline has since moved to a forward sample window; LIBERO numbers here predate that.)

## Evaluate

```bash
git clone -b newton https://github.com/DingTianxingjian/Latent_Free.git latent_free && cd latent_free
huggingface-cli download Potestates/TERRA --include "libero_10_frac10_terra_p2fft20k/*" --local-dir .
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint ./libero_10_frac10_terra_p2fft20k
```

`action_decoder.pt` is **required** — the latent/action heads are not inside the merged VLA weights. The 500-episode
eval splits cleanly over two GPUs with `--task_ids 0,2,4,6,8` / `1,3,5,7,9`; check `complete: true` in the merged
result, and check that **both** shard files exist (a rank that dies at model load still exits 0).

## Training

`INIT=/data/ti874297/p2fft_20k`, 15,000 steps, global batch 64 (2 GPUs x micro-batch 8 x accum 4), lr 3.5e-4 x0.1 at
step 12,000, **10 h 41 m** (2.56 s/step) on 2x H100. `train[:10%]` of `libero_10_no_noops` = 43 demonstrations;
window 12, `dz` 256, flow head 512x4, 10 Euler steps. Final train batch: latent R² 0.821, `action_l1_1step` 0.0298
(action L1 0.168 -> 0.0298, latent R² 0.580 -> 0.821 over the run).

⚠ **`action_l1_1step` is oracle-conditioned** — track one run with it, **never rank two arms**; on the pick-can pair
in this repo it ranks them backwards. Rank with the suite eval.

Merged with `acdc_la/scripts/merge_ctrl_run.py` standalone (`EXTRA="--merge False"` at train time): peak RSS 30.9 GB
in 191 s, which is the recipe for merging on a 62 GB host.

## Provenance is in the files, not just in this card

`ctrl_meta.json` here records **`"init": "p2fft_20k"`** explicitly, because the metadata written by the training
script does *not* otherwise say which base a checkpoint was post-trained from — a reader could not tell this arm
from the `libero_goal_frac10_*` cell (stock `openvla-7b`, no phase 2) by looking at the files. The four directories
published on 2026-11-12 record it; the older ones state it only in their cards.
