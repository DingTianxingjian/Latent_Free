# TERRA policy on the AgileX Piper (pick up the can, place it on the plate)

Everything needed to run the trained policy is in this repository plus the checkpoint on HuggingFace. You do **not**
need to clone UniVLA: the handful of UniVLA files the policy uses are vendored, byte-identical, in
`acdc_la/vendor/univla/` (Apache-2.0).

## 1. Install (Python 3.10, one CUDA GPU; inference measures 15.6 GB, so 24 GB is comfortable)

```bash
git clone https://github.com/DingTianxingjian/Latent_Free.git latent_free && cd latent_free
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r acdc_la/realrobot/requirements_deploy.txt
huggingface-cli download Potestates/TERRA --include "piper_pick_can_10k/*" --local-dir ckpt
```

## 2. Check the install (no robot)

```bash
PYTHONPATH=. python -m acdc_la.realrobot.piper_policy --checkpoint ckpt/piper_pick_can_10k
```

This should load the model and print `[piper-policy] ... window 12 | action dim 7 | ...`.

Optional: replay the held-out episodes and compare with our numbers. This needs the dataset converted to RLDS first:
`python -m acdc_la.realrobot.convert_piper_rlds --src <LeRobot download of jachansantiago/pick_can_place_plate_150_trimmed> --out data/real_rlds`.
That step also needs `pyarrow` and `av`.

```bash
PYTHONPATH=. python -m acdc_la.realrobot.piper_policy --checkpoint ckpt/piper_pick_can_10k --replay data/real_rlds
```

The reference result is about 3.2 degrees mean absolute error per joint for chunk[0].

For context on whether that is good: on the same episodes, predicting the training mean gives 14.70 deg, the average
demonstration time-warped to each episode gives 11.02 deg, and *repeating the last command* gives 2.06 deg.  So the
policy follows each episode's own trajectory rather than an average, but its per-step precision is the same order as
the per-step motion.  The step budget is **not** the explanation: held-out error is flat from step 6,000 to 10,000
(0.0972 / 0.0981 / 0.0947 / 0.0940 / 0.0930) while the training loss was still falling, so the policy is neither
under- nor over-trained.  Full reasoning in `LEDGER.md`, entry 2026-11-10.

## 3. Control loop

```python
from acdc_la.realrobot.piper_policy import PiperPolicy

policy = PiperPolicy("ckpt/piper_pick_can_10k", ensemble=False, samples=8)
policy.reset()                                    # at the start of every episode
while running:
    chunk = policy.act_chunk(front_rgb, "pick up the can and place it on the plate")
    for a in chunk[:2]:                           # execute 2 of the 12, then re-predict with a fresh image
        arm.send_joint_targets(a[:6])             # ABSOLUTE joint targets, degrees
        arm.send_gripper(PiperPolicy.gripper_to_raw(a[6]))
        wait_for_next_10hz_tick()
```

### The three settings above, and why (all measured on the 8 held-out episodes, 2026-11-10)

**`chunk[:2]` -- execute 2 steps per prediction, not 5.** The open-loop error grows across the chunk:

| steps ahead k | 0 | 1 | 2 | 3 | 4 | 5 | ... | 11 |
|---|---|---|---|---|---|---|---|---|
| error (deg/joint) | **3.24** | 3.94 | 4.58 | 4.98 | 5.19 | 5.32 | | 5.67 |

Executing the first *n* commands an average of: n=1 **3.24**, n=2 **3.59**, n=3 3.92, n=5 **4.38** deg.  `act_chunk`
costs 89.0 ms mean / 96.2 ms p90 on an RTX PRO 6000 against 100 ms per step at 10 Hz, so n=1 leaves 4 ms of headroom
-- too thin, and a control rate that slips is itself off-distribution because the policy only ever saw 10 Hz.  n=2
takes 69 % of the available gain with >2x headroom.  **Measure your own card first:
`python -m acdc_la.realrobot.latency`.**

**`ensemble=False`.** UniVLA's temporal ensembling is slightly *harmful* here, 3.69 vs 3.24 deg: it is tuned for
LIBERO, where the action is a delta and the lag it introduces is free.  These actions are absolute joint targets, so
the lag lands directly on the commanded position.

**`samples=8`.** One VLA forward, 8 flow-head samples, averaged.  The head is ~12M parameters against the VLA's 7B.
It improves accuracy by only 2 %, but it cuts the **call-to-call jitter** of the action actually sent by **63 %**
(3.43 -> 1.28 deg on a fixed image).  That matters because 3.43 deg of jitter is as large as the policy's entire
accuracy error -- on the robot it looks like hunting.  Caveat: averaging samples from a multimodal distribution
returns something that is not any of the modes.  For one can on a table that is a fine trade; for a task with
genuinely distinct options, drop it to 1.

The policy must see what training saw:

| | |
|---|---|
| Control rate | **10 Hz**. The 30 Hz data was subsampled by 3. |
| Camera | **Front camera only**, RGB, any resolution. It is squashed to 256x256 and centre-cropped inside the wrapper. The wrist camera is not used. |
| Instruction | `"pick up the can and place it on the plate"` |
| Output | `a[0:6]`: absolute joint targets in degrees, same convention as the dataset's `action`. `a[6]`: gripper in [0, 1]; `gripper_to_raw` maps it to the dataset's units (x8.2). |

The full reasoning behind each preprocessing step is in the docstring at the top of `piper_policy.py`.
