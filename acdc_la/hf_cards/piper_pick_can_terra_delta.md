# `piper_pick_can_terra_delta` — real-robot policy, AgileX Piper pick-can, **DELTA actions**

OpenVLA-7B with our latent-action phase 2 (full-parameter, 20k steps, `latp2_inv` labels; `p2fft_20k/` in this repo)
as the **base**, plus LoRA r32 post-training on the Piper pick-can data with a flow-matching head, 4 query tokens and
the latent branch (`--latent tc_cont`). The LoRA is **already merged** into the weights here.

This is the **delta-action replacement** for `piper_pick_can_10k` (which emits absolute joint targets and is 3x worse
open loop). Same task, same data, same held-out episodes — only the action parameterisation differs.

## ⚠ The action is a DELTA, not a joint target

Dims 0-5 are **target joint angle MINUS the arm's measured joint angle**, in degrees. Sending them as joint targets
drives every joint toward ~0 deg. `ctrl_meta.json` carries `"action_space": "delta"` and `to_joint_targets()` refuses
to run without a measured joint state.

```python
from acdc_la.realrobot.piper_policy import PiperPolicy   # DingTianxingjian/Latent_Free, branch newton
p = PiperPolicy("<this dir>")                            # prints "action space DELTA"
p.reset()
while True:                                              # 10 Hz
    chunk = p.act_chunk(front_frame_rgb, "pick up the can and place it on the plate")
    for a in chunk[:1]:                                  # EXECUTE 1 — see "chunk length" below
        q = read_joint_state_deg()                       # MEASURED joints, re-read EVERY tick
        send_joint_targets(p.to_joint_targets(a, q))     # = q[:6] + a[:6]
        send_gripper(PiperPolicy.gripper_to_raw(a[6]))
        wait_until_next_10hz_tick()
```

### chunk length: use 1 (measured, not guessed)

The demos were teleoperated as a **continuous 30 Hz** command, so the demo arm at step t+1 is already past the
command issued at t. Chunk entry `k` was trained as `a[t+k] - q[t+k]`, relative to the state at the step it is *for*,
so an arm that settles at each command falls behind by the **demo lead** `a[t-1] - q[t]` every step.
Measured on all 150 pick-can episodes (`acdc_la/realrobot/demo_lead.py`): **0.855 deg/step**, and it is
**opposite in sign on the two joints that carry the motion** (j1 -0.74, j2 +0.75) — so it *bends the direction*
rather than scaling it. Drift carried by entry k = k x 0.855 deg:

| entry k | 0 | 1 | 2 | 3 | 4 | 11 |
|---|---|---|---|---|---|---|
| drift, deg | 0 | 0.86 | 1.71 | 2.56 | 3.42 | 9.4 |

Entry 1's drift already equals this policy's whole `chunk[0]` error, and entry 4 reaches the old absolute arm's
3.24 deg. On `press_button` the same effect (there ~2 deg/step) was **seen on the robot**: chunk 1 presses the button,
chunk 2 "bends the direction". **So: execute 1 entry and re-predict.** If you need multi-step chunks, use a
chunk-ANCHORED checkpoint (`piper_multi_anchored_*`), which is parameterised to make that valid.

**Ensembling: try both.** On this dataset it *helped* offline (0.950 vs 1.126 deg); on `press_button` it was slightly
*worse*. It is coordinate-correct for this parameterisation — entry `k` from the prediction `k` steps ago is a delta
relative to *now* — but its benefit is dataset-dependent. `p.act(frame, instr)` gives the ensembled action; pass it
through `to_joint_targets` the same way.

**Smoothness:** `PiperPolicy(..., samples=8)` averages 8 head samples from one VLA forward and cut call-to-call
jitter 3.43 -> 1.28 deg (-63%) on the absolute arm. Caveat in the wrapper docstring: averaging a multimodal
distribution returns something that is not any mode. Fine for one can on a table.

## Data

`piper_pick_can`, AgileX Piper, 150 episodes of **one** task, 10 Hz, 6,462 steps. Front + wrist cameras in the
dataset but **this policy never sees the wrist** (`image_primary` only). Delta derived by
`acdc_la/realrobot/make_delta_rlds.py` from the absolute RLDS so that frames, subsampling, episode boundaries and
instruction are **bit-identical** and only `action` differs: `d[:,:6] = action[:,:6] - joint_state[:,:6]`, gripper
untouched. Trained on `train[:95%]` = 142 episodes; `train[95%:]` = **8 episodes** held out.

Every one of the 150 episodes carries the identical instruction `'pick up the can and place it on the plate'` —
this is a **single-instruction** dataset. Nothing here tests language grounding (see `press_button_terra_delta` for
a set that does).

## Training budget

10,000 steps at global batch 64 (1 GPU x BS 8 x accum 8), lr 3.5e-4, window 12, 17.9 h. Final train `latent_r2`
0.831, so the latent branch was active, not collapsed.

## Held-out error, measured the way the policy is actually run

Chunk sampled from noise (10 Euler steps), deployment image path, 8 held-out episodes / 353 steps.
`piper_policy.py --replay data/real_rlds_delta` reproduces this exactly (deterministic), and reproducing it is the
check that your preprocessing matches ours.

| | j0 | j1 | j2 | j3 | j4 | j5 | **mean** | gripper (of 1.0) |
|---|---|---|---|---|---|---|---|---|
| `chunk[0]`, deg | 0.652 | 2.201 | 1.730 | 0.437 | 1.672 | 0.064 | **1.126** | 0.048 |
| temporally ensembled | 0.638 | 1.801 | 1.364 | 0.387 | 1.447 | 0.065 | **0.950** | 0.064 |

Normalised whole-chunk L1 **0.1174**. Because the measured joint state is known at deployment, the delta error
**is** the absolute-target error; `--abs_root` confirms the `to_joint_targets` path end to end.

### What to compare it against

| | deg |
|---|---|
| this policy, `chunk[0]` | **1.126** |
| `piper_pick_can_openvla_delta` (the matched control) | 1.054 |
| `piper_pick_can_10k` (absolute targets, same arm) | 3.24 |
| null: output zero delta = "stay where you are" | 1.20 |
| null: repeat the last command | 2.06 |
| between-demonstration floor (nothing can beat this) | 0.61 |

**This policy and the control are NOT statistically separated.** Paired cluster bootstrap over the 8 held-out
episodes (the independent unit — 353 steps is not the sample size): A-B = +0.072 deg, 95% CI **[-0.038, +0.189]**,
worse in 5/8 episodes, sign test p = 0.727. So on this single-instruction task the latent mechanism shows **no
detected effect**; pick whichever checkpoint you prefer. On `press_button`, which has 4 instructions, the same pair
separates 2.1x — see that README.

Note also that the pair is **confounded for attribution**: the control changes both the base (stock `openvla-7b`
vs our phase 2) and the latent branch, so neither direction can be credited to the latent branch alone.

## Do not use the training metric to rank checkpoints

`action_l1_1step` in `train_log.jsonl` is **oracle-conditioned** (`FlowHeads.loss` scores a one-step correction from
a point that already contains the ground truth). It makes this policy look 13% *better* than the control
(0.05634 vs 0.06483) while both free-running read-outs make the control better. Use it to track one run; never to
rank two. Always rank with `--replay`.
