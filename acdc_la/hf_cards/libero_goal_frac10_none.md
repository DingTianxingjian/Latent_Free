# `libero_goal_frac10_none` — **MATCHED CONTROL**: no latent at all, LIBERO-Goal, 43 demos (10 %)

OpenVLA-7B + LoRA r32 + flow head + 4 query tokens, identical to the two latent arms of this cell **except that
there is no latent channel**: `--latent none` means no labeller is built, `dz = 0`, there is no latent loss, and
`latent_r2` is **identically 0.000 on every line of `train_log.jsonl`** (checkable). Base: official
`openvla/openvla-7b`, no phase 2.

This is the arm that makes the other two interpretable. Publishing it is the point.

## Result — **443 / 500 = 88.6 %**

| success | task |
|---|---|
| 1.00 | put the wine bottle on top of the cabinet · turn on the stove |
| 0.96 | put the bowl on the stove · put the bowl on the plate |
| 0.90 | open the middle drawer of the cabinet |
| 0.88 | open the top drawer and put the bowl inside |
| 0.82 | put the wine bottle on the rack |
| 0.80 | put the cream cheese in the bowl |
| 0.78 | put the bowl on top of the cabinet |
| 0.76 | push the plate to the front of the stove |

| Goal @ 10 % demos, same interface | success | latent R² |
|---|---|---|
| UniVLA latent (`..._univla_cont`) | 89.4 % | 0.964 |
| **no latent (this)** | **88.6 %** | — |
| our latent (`..._terra`) | 86.8 % | 0.819 |

**Conclusion — negative, and it applies to both latents:** the no-latent control sits between them. At 10 % Goal
through the query-token flow interface the action head reads what it needs from the visual features alone, so the
latent channel is decorative here; 13 episodes in 500 is ~1.3σ and one cell carries ±8 points at 95 %. No claim of
the form "our latent action beats UniVLA's downstream", and no claim that "latent actions help", can rest on this
cell. (Same pattern as the 2026-09-13 toy closed-loop, now at full-VLA scale with a proper matched control.)

**What it does not cover:** suites/regimes where visual features are insufficient — LIBERO-Long at 10 % is the
harder cell (`libero_10_frac10_terra_p2fft20k`, 60.6 %, where nothing is saturated), and the real robot is where the
same kind of control *does* separate: on `press_button` the matched no-latent arm is 1.92° against 0.918° for ours
and **cannot read the instruction at all** (see `press_button_openvla_delta`).

## Evaluate

```bash
huggingface-cli download Potestates/TERRA --include "libero_goal_frac10_none/*" --local-dir .
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint ./libero_goal_frac10_none
```

`action_decoder.pt` is required (it still holds the flow head). Training: 15,000 steps x global batch 64, 31,727 s
(2.1 s/step — cheapest of the three, no labeller), final `action_l1_1step` 0.0281 (oracle-conditioned — do not rank
arms with it).

## Provenance is in the files, not just in this card

`ctrl_meta.json` here records **`"init": "none"`** explicitly, because the metadata written by the training script
does *not* otherwise say which base a checkpoint was post-trained from — a reader could not tell this arm (stock
`openvla-7b`) from `libero_10_frac10_terra_p2fft20k` (TERRA phase-2 base) by looking at the files. The four
directories published on 2026-11-12 record it; the older ones state it only in their cards.
