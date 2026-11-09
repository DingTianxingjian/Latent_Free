# TERRA policy on the AgileX Piper (pick up the can, place it on the plate)

Everything needed to run the trained policy is in this repository plus the checkpoint on HuggingFace. You do **not**
need to clone UniVLA: the handful of UniVLA files the policy uses are vendored, byte-identical, in
`acdc_la/vendor/univla/` (Apache-2.0).

## 1. Install (Python 3.10, one CUDA GPU, at least 24 GB)

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

## 3. Control loop

```python
from acdc_la.realrobot.piper_policy import PiperPolicy

policy = PiperPolicy("ckpt/piper_pick_can_10k")   # ensemble=False sends chunk[0] directly; try both
policy.reset()                                    # at the start of every episode
while running:                                    # 10 Hz
    a = policy.act(front_rgb, "pick up the can and place it on the plate")
    arm.send_joint_targets(a[:6])                 # ABSOLUTE joint targets, degrees
    arm.send_gripper(PiperPolicy.gripper_to_raw(a[6]))
```

The policy must see what training saw:

| | |
|---|---|
| Control rate | **10 Hz**. The 30 Hz data was subsampled by 3. |
| Camera | **Front camera only**, RGB, any resolution. It is squashed to 256x256 and centre-cropped inside the wrapper. The wrist camera is not used. |
| Instruction | `"pick up the can and place it on the plate"` |
| Output | `a[0:6]`: absolute joint targets in degrees, same convention as the dataset's `action`. `a[6]`: gripper in [0, 1]; `gripper_to_raw` maps it to the dataset's units (x8.2). |

The full reasoning behind each preprocessing step is in the docstring at the top of `piper_policy.py`.
