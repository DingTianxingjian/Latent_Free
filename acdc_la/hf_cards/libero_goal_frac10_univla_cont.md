# `libero_goal_frac10_univla_cont` — **BASELINE**: UniVLA's latent through our interface, LIBERO-Goal, 43 demos (10 %)

OpenVLA-7B + LoRA r32 + flow head + 4 query tokens, identical to `libero_goal_frac10_terra` in every respect
**except the latent labeller**: the targets here are **UniVLA's own deployed latent** — the `z_q` codebook vectors of
`vq_encode(...)` from the `s2mix_1ep` LAM checkpoint — used as continuous labels (`--latent univla_cont`, `dz=512`).
Base: official `openvla/openvla-7b`, **no phase 2**.

This is the baseline that answers "is *our* latent better than theirs, with everything else held fixed?".

## Result — **447 / 500 = 89.4 %**, i.e. **+2.6 points over our latent (86.8 %)**

| success | task |
|---|---|
| 1.00 | put the wine bottle on top of the cabinet · turn on the stove |
| 0.98 | put the bowl on the stove |
| 0.94 | open the top drawer and put the bowl inside · put the bowl on the plate |
| 0.90 | push the plate to the front of the stove |
| 0.84 | put the cream cheese in the bowl |
| 0.80 | put the bowl on top of the cabinet |
| 0.78 | open the middle drawer of the cabinet |
| 0.76 | put the wine bottle on the rack |

Per task vs ours (suite order, of 50): 39/43, 49/49, 50/49, **47/37**, 40/44, **45/37**, 42/42, 50/49, 47/45, 38/39.
The gap is carried by two tasks; ours is ahead on two others.

**Reading: not a significant difference.** 13 episodes in 500 is ~1.3σ, one seed each, and a single 500-episode cell
carries ±8 points at 95 % once training randomness is included. What it does settle is that **our latent is not
better than UniVLA's in this regime** — if anything it is behind — and that their latent is far easier for the VLA
to predict (train latent R² 0.964 vs ours 0.819).

Read this together with `libero_goal_frac10_none` (**88.6 %, no latent at all**): the matched no-latent control
matches both latent arms, so at 10 % Goal the latent channel is **decorative** and neither "ours beats UniVLA" nor
"latent actions help" can rest on this cell.

## Evaluate

```bash
huggingface-cli download Potestates/TERRA --include "libero_goal_frac10_univla_cont/*" --local-dir .
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint ./libero_goal_frac10_univla_cont
```

`action_decoder.pt` is required. Training: 15,000 steps x global batch 64, 45,058 s (3.0 s/step — their labeller is
heavier than ours), final latent R² 0.964, `action_l1_1step` 0.031 (oracle-conditioned — do not rank arms with it).

**Not** UniVLA's released policy: this is their latent *labeller* driving our post-training interface, so it is a
controlled ablation, not a reproduction of their published numbers. Their own token recipe at this cell scored
86.6 % (433/500).

## Provenance is in the files, not just in this card

`ctrl_meta.json` here records **`"init": "none"`** explicitly, because the metadata written by the training script
does *not* otherwise say which base a checkpoint was post-trained from — a reader could not tell this arm (stock
`openvla-7b`) from `libero_10_frac10_terra_p2fft20k` (TERRA phase-2 base) by looking at the files. The four
directories published on 2026-11-12 record it; the older ones state it only in their cards.
