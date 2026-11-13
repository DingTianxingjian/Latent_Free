# `stack_bowls_openvla_delta` — the matched CONTROL for stack_bowls (stock OpenVLA-7B, no latent)

**This is a control, not a policy to deploy.**  It exists to answer "what did TERRA's phase-2 base + latent branch
buy on this task", and it is published so that number can be checked.

**Read the floor first.**  On the held-out episodes this checkpoint is scored on, *repeating the previous command*
scores **0.667 deg** and *not moving at all* scores **0.863 deg**.  This model's deployment path scores
**0.695 deg** — i.e. **one step at a time it is not distinguishable from repeating the previous command**.  Any
number quoted for this task without those two nulls beside it is unreadable.

| | config |
|---|---|
| base | official `openvla/openvla-7b`, **no phase 2** (`INIT=none`) |
| latent | **none** (`--latent none`: `dz = 0`, no labeller, no latent loss; `latent_r2 == 0.000` on every logged step) |
| head | flow matching, 10 Euler steps, width 512, 4 layers, z_tok 32 |
| query tokens | 4 `<ACT_i>` placeholders (vocab 32005), embeddings trained |
| LoRA | r32, alpha 16, `all-linear`, merged into the weights here |
| data | `stack_bowls` **alone**, 100 episodes / 3,928 steps at 10 Hz, one instruction `stack the bowls` |
| action | per-step **delta** `target − q` (degrees), `"action_space": "delta"` in `ctrl_meta.json` |
| budget | 6,000 steps x global batch 64 = **97.8 epochs** (matched to pick-can's 99 and press_button's 104) |
| chunk | 12 at 10 Hz; front camera only, no proprio input |

## Measured (offline only — this has never run on the robot)

Held-out `train[95%:]` = the **last 5 episodes, 184 steps**, mean |error| over joints 0-5 in degrees:

| arm | error | vs repeat-last 0.667 | vs stay-still 0.863 |
|---|---|---|---|
| `chunk[0]` (the deployment path) | **0.695** | +0.031 [-0.118, +0.184] **NOT DISTINGUISHABLE** | -0.188 [-0.324, -0.054] better |
| temporally ensembled | **0.585** | -0.079 [-0.147, -0.015] better | -0.298 [-0.360, -0.246] better |

Intervals are a paired cluster bootstrap resampling the **5 episodes** — the episode is the independent unit, not
the 184 steps, and 5 clusters is a very small sample: the ensembled arm's interval would not survive one bad
episode, and ensembling is not uniformly good (on episode 0 it made the error *worse*, 0.605 -> 0.766).

Per joint, degrees: `chunk[0]` `[0.593, 1.254, 0.918, 0.125, 1.177, 0.102]` vs repeat-last
`[0.465, 1.343, 0.732, 0.128, 1.226, 0.110]`.  The error is concentrated in **j1 and j4**, exactly the two joints
where this model is level with repeat-last; its win over stay-still is almost entirely j1+j2, i.e. it learned the
teleop drift — which repeat-last also encodes.

```bash
python -m acdc_la.realrobot.piper_policy --checkpoint <this dir> \
    --replay data/real_rlds_delta --dataset stack_bowls --abs_root data/real_rlds \
    --dump perstep.npz --out replay.json
python -m acdc_la.realrobot.replay_vs_nulls --dump perstep.npz --dataset stack_bowls   # the nulls + the bootstrap
```

## What this does and does not pair with

**It does NOT pair with `piper_real3_anchored_fwd`.**  That checkpoint is a 3-task mixture
(press_button x2 + pick_can + stack_bowls, 350 episodes), **chunk-anchored** delta, with the **fixed forward**
sample window.  This one is stack_bowls alone, **per-step** delta, **backward** window.  Only the step count, LoRA,
head, query tokens, rate and chunk match.  A difference between the two cannot be attributed to anything.
Their `unnorm_key`s (`real4_*_anc`) are also built from `anhdao69/data_real_4tasks`' episode order, which differs
from ours, so their held-out 5 episodes are probably not these 5 — the split must be checked before pairing.

**What it would pair with:** a TERRA arm on `stack_bowls` alone, per-step delta, backward window, 6,000 steps at
global batch 64, scored on the same `train[95%:]` episodes.  The per-step errors of this arm are kept in the code
repo (`acdc_la/results/stack_bowls_openvla_delta_perstep.npz`, 24 KB) so that comparison can be done paired.

**Even then, two factors move.**  A TERRA arm initialised from `p2fft_20k` differs from this control in the **base**
*and* the **latent branch**.  That is the standard "post-train stock OpenVLA" baseline a reader asks for, but it
cannot attribute the effect to the latent.  Isolating the latent needs a third arm — **phase-2 base + `--latent
none`** — which has never been run on any real-robot task.

## Do not rank with the training metric

`train_log.jsonl`'s `action_l1_1step` is oracle-conditioned.  It ordered the pick-can pair the *opposite* way from
the free-running error (LEDGER.md 2026-11-12).  Use the replay above.

Method ledger: `LEDGER.md` in https://github.com/DingTianxingjian/Latent_Free (branch `newton`), entry
**2026-11-13 (07:00)**.  This card is version-controlled at `acdc_la/hf_cards/stack_bowls_openvla_delta.md` —
merge into it, do not overwrite it.
