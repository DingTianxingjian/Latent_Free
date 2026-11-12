# `piper_pick_can_openvla_delta` — the matched CONTROL for `piper_pick_can_terra_delta`, **DELTA actions**

Official `openvla/openvla-7b`, a **fresh** LoRA r32 post-training on the Piper pick-can delta data with the same
flow-matching head and the same 4 query tokens, and **no latent-action mechanism at all** (`--latent none`: no
labeller, `dz = 0`, no latent loss — `latent_r2` is identically 0.000 at all 400 logged points, verified). The LoRA
is already merged. It does **not** use our phase-2 base.

It is the standard "post-train stock OpenVLA" control: everything the TERRA arm has except the phase-2 base and the
latent branch. Unlike the `press_button` control (1.92 deg, and it ignores the instruction — not for deployment),
**this one is a perfectly good policy and is deployable**, because pick-can is a single-instruction task with
nothing to read.

## It is as good as the TERRA arm here, and simpler

| | deg |
|---|---|
| **this control, `chunk[0]`** | **1.054** |
| `piper_pick_can_terra_delta`, `chunk[0]` | 1.126 |
| `piper_pick_can_10k` (absolute targets) | 3.24 |
| null: output zero delta = "stay where you are" | 1.20 |
| null: repeat the last command | 2.06 |
| between-demonstration floor | 0.61 |

**The two arms are NOT statistically separated**: paired cluster bootstrap over the 8 held-out episodes,
difference +0.072 deg, 95% CI [-0.038, +0.189], sign test p = 0.727. The point estimate favours this control and it
trained 5.2 h faster (12.7 h vs 17.9 h), so if you want one pick-can policy, this is the simpler choice — but do not
read the 7% as a win for either side.

Every one of the 150 pick-can episodes carries the identical instruction, so a null result here is expected for a
language-grounding mechanism. On `press_button` (4 colours, layout reshuffled per episode) the same control is
**1.920 deg against TERRA's 0.918**, and its output moves **2.4e-7 deg** when the colour word is swapped — i.e. it is
exactly invariant to language, because its instruction signal (0.28% of the query-token state) is below bfloat16's
machine epsilon (7.8e-3) and so cannot survive its own forward pass.

## Held-out error, 8 episodes / 353 steps

| | j0 | j1 | j2 | j3 | j4 | j5 | **mean** | gripper (of 1.0) |
|---|---|---|---|---|---|---|---|---|
| `chunk[0]`, deg | 0.599 | 2.022 | 1.672 | 0.419 | 1.552 | 0.062 | **1.054** | 0.042 |
| temporally ensembled | 0.600 | 1.685 | 1.349 | 0.351 | 1.441 | 0.061 | **0.914** | 0.059 |

Normalised whole-chunk L1 **0.1142**. Per joint against the "stay still" null, this policy is better on j0/j1/j2
(ratio 0.84/0.72/0.86) and *worse* on j3/j4/j5 (1.46/1.10/1.30) — but measured against the per-joint demonstration
**floor** instead, **j2 and j4 carry 71.8% of all recoverable error**, while j1 (largest raw error) is nearly solved
at 1.18x its floor. `floor/null` is below 1 on every joint, so there is no joint where outputting zero is correct.

## Deployment

Identical to `piper_pick_can_terra_delta` — read that README's "⚠ The action is a DELTA" and "chunk length" sections;
they apply unchanged. In short: `ctrl_meta.json` says `"action_space": "delta"`, send `q + a[:6]` with `q` re-read
**every tick**, and **execute 1 entry per prediction** (the demos' 30 Hz teleop lead is 0.855 deg/step on this
dataset and is opposite-signed on j1/j2, so longer chunks bend the direction). Ensembling helped here offline
(0.914 vs 1.054) but is dataset-dependent; try both.

```python
from acdc_la.realrobot.piper_policy import PiperPolicy   # DingTianxingjian/Latent_Free, branch newton
p = PiperPolicy("<this dir>"); p.reset()                 # prints "action space DELTA"
while True:
    chunk = p.act_chunk(front_frame_rgb, "pick up the can and place it on the plate")
    q = read_joint_state_deg()
    send_joint_targets(p.to_joint_targets(chunk[0], q))
    send_gripper(PiperPolicy.gripper_to_raw(chunk[0][6]))
    wait_until_next_10hz_tick()
```

## Data and budget

`piper_pick_can` delta (`make_delta_rlds.py`, derived from the absolute RLDS so only `action` differs), 150 episodes
/ 6,462 steps at 10 Hz, front camera only, trained on `train[:95%]` = 142 episodes, `train[95%:]` = 8 held out.
10,000 steps at global batch 64 (1 GPU x BS 8 x accum 8), lr 3.5e-4, window 12, 12.7 h.

## Do not use the training metric to rank checkpoints

`action_l1_1step` is oracle-conditioned and ranks this pair **backwards** (it makes the TERRA arm look 13% better
while both free-running read-outs favour this one). Rank with `piper_policy.py --replay`.
