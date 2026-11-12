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
Real-robot deployment wrapper: `acdc_la/realrobot/piper_policy.py`; LIBERO evaluator:
`acdc_la/downstream/vla/run_libero_eval_cont.py`.

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
| **compare the latent mechanism** in sim | the `libero_goal_frac10_*` **triplet** | ours / UniVLA's / **no latent at all**, identical otherwise |
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

## The three action spaces on the Piper — getting this wrong drives the arm somewhere else

`ctrl_meta.json -> action_space` says which one a directory uses, `PiperPolicy` reads it and prints it on load.
(It used to be *absent* on the pick-can directories and inferred from the joint means; since 2026-11-12 all three
record it explicitly, so **a missing key now means an old copy** — re-download.)

| `action_space` | what `chunk[k]` is | the call | how many steps per chunk |
|---|---|---|---|
| `absolute` | the joint target itself | send `chunk[k]` directly | — |
| `delta` | target − joints measured **at step k** | `to_joint_targets(a, q)`, `q` re-read **every tick** | **1** — the teleop lead adds up linearly and with opposite signs on j1/j2 (−0.74 / +0.75 deg per step on pick-can), so the direction bends |
| `delta_anchored` | `target[s+k]` − joints read **once** at the chunk's start | `chunk_targets(chunk, q)`, `q` read once per chunk; **no ensembling** (the policy disables it) | 1, then try 2 and 4: tracking drift does not exist here, what limits N is the model's own error growing with k |

---

## Inventory — which directories are **ours (TERRA)** and which are **controls / baselines**

Every policy here shares the same interface (LoRA r32 + flow-matching head + 4 query tokens, chunk 12). What differs
is the **base** and the **latent channel**, which is exactly what each comparison is about.

### TERRA — ours (latent channel `tc_cont`, `latp2_inv` labels)

| dir | base | task / suite | number |
|---|---|---|---|
| `p2fft_20k` | openvla-7b, full fine-tune | **phase-2 base, not a policy** | — |
| `press_button_terra_delta` | `p2fft_20k` | press_button, per-step delta | 0.918° ✅ on hardware |
| `press_button_anchored_fwd_step1500` | `p2fft_20k` | press_button, anchored, fixed window | 0.96° at k=0, flat in k |
| `piper_pick_can_terra_delta` | `p2fft_20k` | pick-can, per-step delta | 1.126° |
| `piper_pick_can_10k` | `p2fft_20k` | pick-can, **absolute** — superseded | 3.24° |
| `piper_multi_anchored_step2500` / `step5500` | `p2fft_20k` | both tasks, anchored — experimental | see card |
| `libero_goal_full` | LoRA-scale phase 2 | LIBERO-Goal, full split | 94.4 % |
| `libero_10_full` | LoRA-scale phase 2 | LIBERO-Long, full split | 89.0 % |
| `libero_10_frac10_terra_p2fft20k` | `p2fft_20k` | LIBERO-Long, 10 % (43 demos) | 60.6 % |
| `libero_goal_frac10_terra` | **openvla-7b, no phase 2** | LIBERO-Goal, 10 % | 86.8 % |

### Baselines / controls — no latent, or somebody else's latent

| dir | what it controls for | number |
|---|---|---|
| `press_button_openvla_delta` | stock openvla-7b, **`--latent none`** — the whole package vs nothing | 1.920° (worse than not moving) |
| `piper_pick_can_openvla_delta` | the same, on pick-can | **1.054°** — here it is a good policy |
| `libero_goal_frac10_none` | **no latent at all**, identical otherwise (`latent_r2` ≡ 0) | 88.6 % |
| `libero_goal_frac10_univla_cont` | **UniVLA's** deployed latent through our interface | 89.4 % |

> **One honest caveat about the two `*_openvla_delta` controls.** They change the base **and** the latent branch at
> once (`INIT=none LATENT=none`), so where they separate, the effect belongs to *the whole package*, not specifically
> to the latent branch. The missing arm that would split those two — phase-2 base with `--latent none` — has not been
> run. The `libero_goal_frac10_*` triplet *is* clean in that sense (one factor, no phase 2 anywhere), and there the
> answer is that the latent channel changes nothing.
>
> **Which base a policy came from is not in most of these files.** `ctrl_meta.json` records the latent, the head
> and the dataset, but the training script never wrote the phase-2 init. The four directories published on
> 2026-11-12 (`libero_goal_frac10_*`, `libero_10_frac10_terra_p2fft20k`) now carry an explicit `"init"` field;
> for the others the base is stated in the table above and in each card, and nowhere in the weights.

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
under-predicted ~0.8x). **`press_button_anchored_fwd_step1500` is the first checkpoint with the window fixed**
(press-button only, snapshot of a run still training): its error stops growing with chunk position — it levels
off near 3.5° instead of climbing to 13° — and it will supersede these for that task. Measured on the 5 held-out
press_button episodes, mean error over the steps actually executed: **exec 2 / exec 4 = 1.50 / 2.28°** for the
window-fixed snapshot, 1.44 / 2.65° for `piper_multi_anchored_step5500` (old window), 1.66 / 3.50° for the
per-step `press_button_terra_delta` under ideal tracking. Both anchored models are **snapshots of runs still
training** (1500 of 3000, and step 5500) and **neither has been on the robot**. Different deployment call
(`chunk_targets`, no ensembling) — read their READMEs.

---

## LIBERO policies

Standard eval, 50 trials x 10 tasks = **500 episodes**, continuous flow head:

```bash
python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint <dir>
```

| dir | suite | demos | base | latent | success |
|---|---|---|---|---|---|
| `libero_goal_full` | Goal | full | LoRA-scale phase 2 | ours | **94.4 %** |
| `libero_10_full` | Long | full | LoRA-scale phase 2 | ours | **89.0 %** |
| `libero_10_frac10_terra_p2fft20k` | Long | 10 % (43) | **`p2fft_20k`** | ours | **60.6 %** |
| `libero_goal_frac10_univla_cont` | Goal | 10 % (43) | openvla-7b | **UniVLA's** | 89.4 % |
| `libero_goal_frac10_none` | Goal | 10 % (43) | openvla-7b | **none** | 88.6 % |
| `libero_goal_frac10_terra` | Goal | 10 % (43) | openvla-7b | ours | 86.8 % |

### The one result in this repo that is a clean, single-factor comparison — and it is negative

The three `libero_goal_frac10_*` rows differ **only** in the latent channel. The **no-latent control lands between
the two latent arms** (88.6 % against 89.4 % and 86.8 %), so at 10 % Goal demos through this interface the latent
channel is **decorative**: the action head reads what it needs from the visual features alone. 13 episodes in 500 is
~1.3σ, one seed each, and a single 500-episode cell carries **±8 points at 95 %** once training randomness is
counted — so these three are one band, not a ranking.

What the triplet *does* settle: an earlier 9.2 % collapse on Goal with the same latent and no query tokens was an
**interface** failure, not the latent. What it does not cover: regimes where the visual features are insufficient.
LIBERO-Long at 10 % is that regime (nothing saturates there: 60.6 % for the best arm) but with one replicate per arm
it still cannot separate 60.6 / 58.7 / 56.4. The place where a matched control *does* separate is the real robot:
`press_button_openvla_delta` is 2.1x worse than ours and cannot read the instruction at all.

Not published here: the full-split 20k-phase-2 Long arm (89.8 %, our best Long number) and the remaining 10 % Long
replicates live on a cluster filesystem, not on this host.

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

---

*This card is version-controlled at `acdc_la/hf_cards/TERRA_root_README.md` in the code repo and is written
from both hosts. **Merge into it, do not overwrite it** — two hosts replaced each other's version of this file
on 2026-11-12 before anyone noticed.*
