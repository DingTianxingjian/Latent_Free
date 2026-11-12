---
pipeline_tag: robotics
tags:
- openvla
- vision-language-action
- robotics
- libero
---

# TERRA — post-trained policies (real robot + LIBERO)

OpenVLA-7B with TERRA latent-action phase 2 (`latp2_inv` labels), post-trained with a flow-matching action head and
4 query tokens. **One directory per policy**, each self-contained: merged weights (`model-*.safetensors`), the
action/latent head (`action_decoder.pt` — **required**, it is not inside the merged VLA weights),
`dataset_statistics.json` (action un-normalisation), `ctrl_meta.json`, processor/tokenizer, and its own `README.md`
with the measured numbers and the exact deployment loop.

**Code:** https://github.com/DingTianxingjian/Latent_Free — branch `newton`, public, no token needed.

```bash
git clone -b newton https://github.com/DingTianxingjian/Latent_Free.git latent_free && cd latent_free
huggingface-cli download Potestates/TERRA --include "<dir>/*" --local-dir .
```

---

## Which one do I test?

| I want to… | use | why |
|---|---|---|
| **press a coloured button** | **`press_button_terra_delta`** | the only policy **confirmed working on hardware** — it presses the requested button |
| **pick the can, place on plate** | **`piper_pick_can_openvla_delta`** | **3x better open loop** than the arm currently on the robot (1.05 vs 3.24 deg); simplest of the equals |
| the same, with our full method | `piper_pick_can_terra_delta` | statistically **indistinguishable** from the row above (1.13 vs 1.05, CI contains 0) — either is fine |
| **one model for both tasks** | `piper_multi_anchored_step5500` | **experimental**, and a retrain will supersede it — see its README |
| post-train a new task | `p2fft_20k` | the **base**, not a policy |

> ### ⚠ Two things that will break a deployment if you skip them
> 1. **Action space differs per directory.** `piper_pick_can_10k` emits **absolute** joint targets; everything with
>    `delta` in the name emits **target minus measured joint angle** and must be sent as `q + a[:6]` with `q` re-read
>    **every tick**; the `anchored` ones use `q` read **once per chunk** and a different call. Every `ctrl_meta.json`
>    now records `"action_space"` explicitly and `PiperPolicy` prints it on load — **check the printed line.**
> 2. **Execute 1 chunk entry, then re-predict** (all per-step `delta` policies). The demos were teleoperated as a
>    continuous 30 Hz command, so later chunk entries are mis-anchored for an arm that settles: drift is
>    **0.855 deg/step on pick-can** and **~2 deg/step on press_button**, and it is opposite-signed on the two joints
>    that carry the motion, so it **bends the direction** instead of just scaling it. This was seen on the robot:
>    chunk 1 presses the button, chunk 2 "the direction is a bit off". Use an `anchored` checkpoint if you need
>    multi-step chunks.

---

## Real-robot policies — AgileX Piper

Open-loop error = mean |error| per joint in **degrees** on held-out episodes never trained on, chunk sampled from
noise through the deployment image path. Reproduce any row with
`envs/univla_vla/bin/python -m acdc_la.realrobot.piper_policy --checkpoint <dir> --replay <rlds root>` — it is
deterministic, and reproducing it is the check that your preprocessing matches ours.

### `press_button` — "press the {blue, orange, pink, red} button", 100 episodes, layout reshuffled per episode

| dir | base | action | `chunk[0]` | reads the instruction? | on the robot |
|---|---|---|---|---|---|
| **`press_button_terra_delta`** | our phase 2 | delta | **0.918** | **yes** — output moves 0.42 deg on a colour swap; true word wins 4/5 episodes | ✅ **presses the button** (chunk 1; slightly off centre) |
| `press_button_openvla_delta` | stock openvla-7b | delta | 1.920 | **no** — output moves **2.4e-7 deg**; true word wins 1/5 = chance | ❌ not for deployment |
| *null: "stay where you are"* | | | *1.691* | | |

The control is *worse than not moving*, and its invariance to language is **exact, not approximate**: its instruction
signal is 0.28% of the query-token state, which is below bfloat16 machine epsilon (7.8e-3), so it cannot survive its
own forward pass. With 4 colours and per-episode layouts a policy that cannot read the colour can only learn an
average press — which is what its j1/j2 error looks like. Caveats: **5 held-out episodes, 3 of 4 colours, one seed
per arm, offline**; and the pair changes *both* the base and the latent branch, so the effect cannot be attributed to
the latent branch alone.

### `piper_pick_can` — "pick up the can and place it on the plate", 150 episodes, **one** instruction throughout

| dir | base | action | `chunk[0]` | ensembled | note |
|---|---|---|---|---|---|
| **`piper_pick_can_openvla_delta`** | stock openvla-7b | delta | **1.054** | 0.914 | the matched control — and here a perfectly good policy |
| **`piper_pick_can_terra_delta`** | our phase 2 | delta | **1.126** | 0.950 | ours; **not separated** from the row above |
| `piper_pick_can_10k` | our phase 2 | **absolute** | 3.24 | 3.69 | the arm tested on the robot first; **superseded** by the two above |
| *null: "stay where you are"* | | | *1.20* | | |
| *null: repeat the last command* | | | *2.06* | | |
| *between-demonstration floor* | | | *0.61* | | **nothing can beat this** |

**Delta vs absolute is the real win here: 3.24 → ~1.05 deg, 3x**, and it moves the policy from 2.7x *worse* than
doing nothing to better than it. **TERRA vs control is not a win for either side**: paired cluster bootstrap over the
8 held-out episodes (the independent unit — 353 steps is not the sample size) gives a difference of +0.072 deg,
95% CI **[-0.038, +0.189]**, sign test p = 0.727.

A null here is what the mechanism's own story predicts: **all 150 episodes carry the identical instruction**, so
there is nothing for a language-grounding advantage to do. The same pair separates 2.1x on `press_button`.

### `piper_multi_anchored_step2500`, `piper_multi_anchored_step5500` — ⚠ experimental

One model for **both** tasks (press_button x2 + pick-can mixture), using **chunk-ANCHORED** deltas
(`target[s+k] - q[s]`, `q` read once per chunk) so that executing several entries open loop is valid. Better than the
per-step parameterisation from k>=2, slightly worse at k=0. Trained with the old backward sample window (magnitudes
under-predicted ~0.8x); **a forward-window retrain is in progress and will supersede these.** Different deployment
call (`chunk_targets`, no ensembling) — read their README.

---

## LIBERO policies

| dir | suite | demos | success, standard 500-episode eval |
|---|---|---|---|
| `libero_10_full` | `libero_10_no_noops` | full | **89.0** |
| `libero_goal_full` | `libero_goal_no_noops` | full | **94.4** |

```bash
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint <dir>
```

---

## `p2fft_20k` — the phase-2 base (not a deployable policy)

The **base** every TERRA policy above is post-trained from: OpenVLA-7B fully fine-tuned for 20,000 steps on OXE to
predict our continuous latent action (`latp2_inv`). `vla_full.pt` (15,081,838,026 B) is the VLA state dict, and
**`action_decoder.pt` (105,268,830 B) is required too** — `finetune_libero_ctrl.py:273` initialises the post-training
heads from it, so a run without it starts from a random head and will not reproduce these policies.

```bash
huggingface-cli download Potestates/TERRA --include "p2fft_20k/*" --local-dir .
INIT=./p2fft_20k  ...  bash acdc_la/scripts/run_vla_post2_portable.sh
```

The matched control is `INIT=none LATENT=none`, which post-trains the official `openvla/openvla-7b` with everything
else identical — that is how "what did phase 2 buy" is measured, and it is what the two `*_openvla_delta`
directories are.

---

## One metric trap, worth repeating

`action_l1_1step` in every `train_log.jsonl` is **oracle-conditioned**: `FlowHeads.loss` scores a one-step
correction from a point that already contains the ground truth. On the pick-can pair it ranks the two arms
**backwards** (TERRA 0.0563 vs control 0.0648, i.e. TERRA "13% better", while both free-running read-outs favour the
control). Use it to track a single run or pick a checkpoint within one run — **never to compare two arms.** Rank with
`--replay`.

`HANDOFF_press_button.md` at the root of this repo is the brief for training the next real-robot task on a fresh
host. The full experiment ledger, including everything retracted or superseded, is `LEDGER.md` in the code repo.
