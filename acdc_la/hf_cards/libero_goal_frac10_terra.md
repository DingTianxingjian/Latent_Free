# `libero_goal_frac10_terra` — LIBERO-Goal, 43 demos (10 %), **our latent** (TERRA `latp2_inv`)

OpenVLA-7B + LoRA r32 + flow-matching action head + **4 query tokens**, trained to predict our continuous latent
action (`--latent tc_cont`, labels from the `latp2_inv` labeller) next to the action chunk. Merged weights.

> **This arm has NO phase 2.** It starts from the official `openvla/openvla-7b`, not from `p2fft_20k`. It is the
> *latent-channel* arm of a three-way matched cell, not the full TERRA stack — the full stack is `libero_*_full`
> (LoRA phase 2) and the robot policies (20k full-parameter phase 2).

## Result — **434 / 500 = 86.8 %** (standard LIBERO-Goal eval, 50 trials x 10 tasks)

| success | task |
|---|---|
| 0.98 | put the bowl on the stove · put the wine bottle on top of the cabinet · turn on the stove |
| 0.90 | put the bowl on the plate |
| 0.88 | put the bowl on top of the cabinet |
| 0.86 | open the middle drawer of the cabinet |
| 0.84 | put the cream cheese in the bowl |
| 0.78 | put the wine bottle on the rack |
| 0.74 | open the top drawer and put the bowl inside · push the plate to the front of the stove |

## The cell — why this checkpoint is published at all

Three arms, **identical except the latent channel** (same base, same LoRA, same flow head, same 4 query tokens, same
43 demos, same 15k x 64 budget, same host, one seed each):

| dir | latent channel | success | train latent R² |
|---|---|---|---|
| `libero_goal_frac10_univla_cont` | UniVLA's deployed latent (`z_q` of `s2mix_1ep`) | **89.4 %** | 0.964 |
| `libero_goal_frac10_none` | **none** — no labeller, `dz=0`, no latent loss | **88.6 %** | — (identically 0) |
| **`libero_goal_frac10_terra`** (this) | ours, `latp2_inv` | **86.8 %** | 0.819 |

**Conclusion (negative, and it applies to both latents):** on Goal at 10 % demos through this interface the latent
channel is **decorative** — a policy with no latent input matches the best latent arm. 13 episodes in 500 is ~1.3σ
on a binomial, and one 500-episode cell carries **±8 points at 95 %** once training randomness is included (pooled
sd ~4.3 points, measured over identical configs), so none of these three is separated from the others.

**What the cell does settle:** an earlier version of this arm scored 9.2 % on Goal with the same latent and the same
flow head but **no query tokens** — that collapse was the **interface**, not the latent. Both latents reach 87-89 %
once the interface is repaired.

**What it does not say:** nothing about regimes where the visual features are insufficient (LIBERO-Long, or fewer
than 43 demos), and nothing about the latent's *content* — the probe advantage of `latp2_inv` is measured on the
oracle code, while the policy only ever sees its own prediction (ours carries more and is predicted worse: R² 0.819
vs their 0.964).

## Evaluate

```bash
git clone -b newton https://github.com/DingTianxingjian/Latent_Free.git latent_free && cd latent_free
huggingface-cli download Potestates/TERRA --include "libero_goal_frac10_terra/*" --local-dir .
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint ./libero_goal_frac10_terra
```

`action_decoder.pt` is **required** — the latent/action heads are not inside the merged VLA weights.

## Training

15,000 steps, global batch 64 (2 GPUs x micro-batch 8 x accum 4), lr 3.5e-4 decayed x0.1 near the end, 34,102 s
(2.27 s/step) on 2x H100 (UCF workstation). `train[:10%]` of `libero_goal_no_noops` = 43 demonstrations; window 12,
`dz` 256, flow head 512x4, 10 Euler steps at deployment. Final train batch: latent R² 0.819, `action_l1_1step` 0.0296.

⚠ **`action_l1_1step` is oracle-conditioned** (the flow loss scores a one-step correction from a point that already
contains the ground truth). Use it to track one run — **never to rank two arms**; on another pair in this repo it
ranks them backwards. Rank with the suite eval.

## Provenance is in the files, not just in this card

`ctrl_meta.json` here records **`"init": "none"`** explicitly, because the metadata written by the training script
does *not* otherwise say which base a checkpoint was post-trained from — a reader could not tell this arm (stock
`openvla-7b`) from `libero_10_frac10_terra_p2fft20k` (TERRA phase-2 base) by looking at the files. The four
directories published on 2026-11-12 record it; the older ones state it only in their cards.
