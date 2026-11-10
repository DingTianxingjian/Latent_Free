# Handoff: the `press_button` arm on a Jetstream2 g3.xl

Written 2026-11-10 by the workstation session for whoever picks this up on Jetstream2.  Everything below that is a
measurement was measured; everything that is a guess says so.  Read the "unverified" notes before trusting a number.

## 0. Read these first, in this order

1. `AGENTS.md` (repo root) -- **the experiment ledger is `LEDGER.md` at the repo root, and it must be updated before
   you report a result to the user.** Configuration, split, budget, the number, its control, conclusion status,
   artifact paths, next decision.  Do not leave a conclusion only in chat or in a subdirectory README.
2. `LEDGER.md`, the entry **2026-11-10 "FIRST PHYSICAL-ROBOT feedback on the Piper policy"**.  It has everything
   already measured about this robot and policy family.  Do not re-measure it.
3. `next_task.md` section 8: **never discretise our latent for the VLA; never change a shared setting for one arm
   only.**  The second one is the live constraint here -- see section 6.

## 1. What this run is for

`anhdao69/press_button_task` is the **first multi-task real-robot dataset in this project**.  The pick-can set has
exactly one task, which means the instruction carried **zero information**: the policy could ignore the language
entirely and nothing would change.  The press-button set contains at least `"press the red button"` and
`"press the blue button"` -- same scene, same motion class, the only disambiguator is the word.  **That is the first
clean test of whether this policy family reads its instruction at all**, and it is worth more than a second task
would be on its own.  Register that as the question before you train, not after.

## 2. The machine: what fits on a g3.xl

| | |
|---|---|
| g3.xl | 32 vCPU, **120 GB system RAM**, **40 GB GPU RAM** (a full A100), 64 SU/h |
| smaller flavors | g3.large 20 GB GPU / 60 GB RAM; g3.medium 10 GB / 30 GB.  **Neither is enough.** |
| multi-GPU | g3.2xl / g3.4xl exist but are **not available by default** -- a justified request to help@jetstream-cloud.org |

**Measured on the workstation (RTX PRO 6000, 2026-11-10): training takes 39.9-40.8 GB of VRAM at micro-batch 8.**
That does not fit in 40 GB with any headroom.  So:

```
BS=4 GA=16    estimated ~28 GB      <- use this
BS=8 GA=8     40 GB measured        <- will OOM on an A100
```

**The global batch must stay 64** (`BS x GA x GPUs = 64`); the launcher asserts it.  Every Piper and 10 % LIBERO cell
in this project is 64, and changing it makes the run incomparable with all of them.  The ~28 GB figure is an
extrapolation from one measured point (fixed ~16 GB + activations linear in batch), **not measured** -- check
`nvidia-smi` at step 50 and drop to BS=2 GA=32 if it is close to the limit.

Other facts worth having before you start:

* **Disk: plan 80-100 GB and attach a volume.**  `openvla-7b` is 15 GB, the env is ~11 GB, each merged checkpoint is
  16 GB, the phase-2 base (if you get it) is another 15 GB.
* **120 GB RAM is the one place Jetstream2 beats the workstation.**  The standalone merge peaks at **30.7 GB** and was
  SIGKILLed three times on the 62 GB workstation (LEDGER 2026-11-08).  On g3.xl it is a non-issue.
* **Measure the first 100 steps before quoting an ETA to anyone.**  On 2026-11-10 the workstation estimate of 13 h
  became 18 h because another user's job was on the same card.  `train_log.jsonl` has a `seconds` field; use it.

## 3. Environment

Follow **`REPRODUCE.md`, the `univla_vla` recipe** (torch 2.2.0 + cu121, `pip install -e third_party/UniVLA`, the
pinned `tensorflow_metadata` / `protobuf` / `wandb`).  That recipe targets sm80, which **is** the A100 -- do not copy
this workstation's env, which is torch 2.7.1+cu128 built for Blackwell.

Two things that save time:

* **Training never asks for flash-attention.**  Grep confirms `attn_implementation` appears only in the two inference
  scripts (`run_libero_eval_cont.py`, `realrobot/piper_policy.py`) and both use `"sdpa"`.  So you do not need to build
  flash-attn for sm80.
* **Conversion needs `pyarrow` and `av`** on top of the env (`cv2` too, which the env has).  The workstation does not
  have them, which is why this dataset was not converted there.

## 4. The dataset, and three things that will bite you

Verified from `meta/info.json` and `meta/stats.json` (read without downloading the videos):

```
robot_type piper | LeRobot codebase_version v3.0 | fps 30
total_episodes 100 | total_frames 5533 | total_tasks 4
action      float32 [7]   joint_1.pos .. joint_6.pos + gripper   (ABSOLUTE joint positions, degrees)
observation.state   float32 [14]
observation.images.front / .wrist   video 480x640x3 mp4
```

Same robot, same action space and same layout as `pick_can_place_plate_150_trimmed`, so
`acdc_la/realrobot/convert_piper_rlds.py` reads it **without a code change** to its parsing.  But:

### (a) `GRIP_SCALE = 8.2` is WRONG for this dataset

`convert_piper_rlds.py:20` divides the raw gripper by 8.2 because the pick-can set's raw gripper ranges 0 .. 8.15.
Here it does not:

```
action dim 6:   min 0.04   max 0.06   mean 0.05   std 0.01
```

Pressing a button never opens the gripper.  `clip(act[:,6] / 8.2, 0, 1)` maps all of it to ~0.005 -- a **constant**.
Numerically harmless, but that dimension carries no signal, and if the two datasets are ever trained jointly the
gripper calibrations are incompatible.  **Make a per-dataset decision and write it in the converter's docstring.**

### (b) The episodes are very short, and a 12-step chunk may not fit in them

```
5533 frames / 30 Hz, subsampled by 3  ->  ~1,844 steps at 10 Hz   (pick-can: 6,462, i.e. 3.5x more)
1,844 / 100 episodes                  ->  ~18 steps = 1.8 s per episode
window_size 12 = 1.2 s                ->  two thirds of an entire episode
RLDSDataset is built with window_size + 2 = 14 frames per sample  ->  ~5 valid windows per episode
```

**Verify what prismatic's chunking actually does at an episode boundary before you believe any loss curve.**  I did
not check this and it is the most likely silent failure: if it pads, you are training on padding for a large fraction
of every sample.  `prismatic/vla/datasets/rlds/traj_transforms.py::chunk_act_obs_libero` is the function.

### (c) Overfitting is a real risk here -- unlike on pick-can

```
1,844 steps of data / global batch 64  ->  29 steps per epoch
10,000 steps                           ->  347 epochs      (pick-can at 10k was 104)
95/5 split of 100 episodes             ->  95 train, only 5 held out
```

The collaborator's "is it under- or over-trained?" question was a **negative** for pick-can (held-out L1 flat from
step 6,000 to 10,000 while train L1 still fell).  **On this dataset it will genuinely bite.**  So:

* Pick a smaller budget.  ~3,000 steps is ~104 epochs, matching pick-can's epoch count, which is the defensible
  choice -- not 10,000.
* **Actually watch the held-out curve this time**, and snapshot checkpoints: `finetune_libero_ctrl.py:496` writes
  every save to the SAME adapter dir, so each save overwrites the last and the run keeps **no history**.  Copy
  `<run_root>/adapters/<exp_id>` after each save or you cannot pick a step afterwards.
* Note `total_tasks: 4` but only two task strings were legible to me (`press the red button`, `press the blue
  button`).  Read `meta/tasks.parquet` properly (needs pyarrow) and report what the 4 are.

## 5. Registering the dataset

`convert_piper_rlds.py` builds a TFDS builder whose **dataset name comes from the class name** (`PiperPickCan` ->
`piper_pick_can`), and `piper_pick_can` is registered in prismatic by `acdc_la/patches/univla_piper.patch` (image
keys + the `EEF_POS` contract).  For a new dataset, copy the converter to `convert_press_button_rlds.py`, rename the
builder class to `PressButton`, and add a `press_button` entry to the patch's two tables (`OXE_DATASET_CONFIGS` in
`oxe/configs.py` and `OXE_STANDARDIZATION_TRANSFORMS` in `oxe/transforms.py`) -- the Piper entries can be copied
verbatim, both datasets have the same image keys and action contract.

**A gotcha I already hit:** TFDS refuses to build a builder class whose module is `__main__`
("Cannot access resources for '__main__'").  Run it as an import, not as a script:

```bash
PYTHONPATH=.:third_party/UniVLA python -c "
import sys; sys.argv = ['x', '--src', '<lerobot dir>', '--out', 'data/real_rlds']
from acdc_la.realrobot.convert_press_button_rlds import main; main()"
```

The shortcut of reusing the `piper_pick_can` registration under a different `--data_root_dir` works (that is what the
delta copy does) but gives both datasets the same key in `dataset_statistics.json`, which is a real footgun.  Register
a proper name.

## 6. Absolute or delta actions? Do not decide this alone

Two 10,000-step runs are in flight on the workstation **right now** testing exactly this on pick-can, both on delta
actions at global batch 64 (expected to finish ~2026-11-10 22:00 UTC):

* `tc_cont+piper_pick_can+b64x1+...--terra_piper_delta` -- our phase-2 base
* `none+piper_pick_can+b64x1+...--openvla_piper_delta` -- official openvla-7b, no latent mechanism (the baseline)

Why: the absolute parameterisation spends a 164 deg output range on a ~2 deg/step signal (1.2 %), against a measured
error of 2.4-11.5 % of each joint's range -- the signal sits below the error (LEDGER 2026-11-10).  The delta is
`action[:6] - joint_state[:6]`, target minus the step's own **measured** configuration, which is self-correcting at
deployment; `acdc_la/realrobot/make_delta_rlds.py` derives a delta RLDS from an absolute one in ~3 minutes with the
frames bit-identical.

**Recommendation: convert to absolute, immediately derive the delta copy, and hold the training choice** until those
two runs land.  Then use whichever they support -- and use the SAME choice for every arm, per `next_task.md` section 8.
The registered prediction for the delta runs, so you can check it: ~1.0-1.5 deg open-loop (the nulls are 1.20 deg for
"stay where you are" and the demonstration floor is 0.61 deg; the absolute arm is 3.24 deg).

## 7. Commands

```bash
# baseline arm -- runs TODAY with no missing artifacts, openvla-7b downloads from HF
PY=<env>/bin/python INIT=none LATENT=none QUERY_TOKENS=4 \
DS=press_button EXTRA="--data_root_dir $PWD/data/real_rlds --merge False" \
SKIP_EVAL=1 GLOBAL=64 BS=4 STEPS=3000 SAVE=500 \
NOTE=openvla_press_button GPUS="0" RUNROOT=<volume>/vla_runs \
  bash acdc_la/scripts/run_vla_post2_portable.sh

# our arm -- needs the phase-2 base, see section 8
PY=<env>/bin/python INIT=<dir containing vla_full.pt> \
DS=press_button EXTRA="--data_root_dir $PWD/data/real_rlds --merge False" \
SKIP_EVAL=1 GLOBAL=64 BS=4 STEPS=3000 SAVE=500 \
NOTE=terra_press_button GPUS="0" RUNROOT=<volume>/vla_runs \
  bash acdc_la/scripts/run_vla_post2_portable.sh

# merge separately -- NEVER in-run (--merge False above). 30.7 GB peak, 211 s.
<env>/bin/python -m acdc_la.scripts.merge_ctrl_run --run <run dir> [--init <phase-2 dir>]

# offline check, in DEGREES through the deployment image path
PYTHONPATH=.:third_party/UniVLA <env>/bin/python -m acdc_la.realrobot.piper_policy \
    --checkpoint <merged run dir> --replay data/real_rlds --dataset press_button
```

`SKIP_EVAL=1` is required: the LIBERO simulator eval does not apply to real-robot data.

## 8. The blocker you cannot solve yourself

**The phase-2 base is not on HuggingFace.**  `p2fft_20k/vla_full.pt` (15 GB, byte-checked at 15,081,838,026) exists
only on the UCF workstation (`/data/ti874297/p2fft_20k`) and on Newton.  `Potestates/TERRA` on HF holds three
**merged policies** (`piper_pick_can_10k`, `libero_10_full`, `libero_goal_full`, 15.19 GB each) and no base.

So **the baseline arm (section 7, `INIT=none`) can run today and our arm cannot.**  Getting the base to Jetstream2
is a decision for the user (uploading 15 GB of phase-2 weights to a repo that is currently **public**, versus an
rsync from Newton that needs their terminal).  Ask; do not pick one.

## 9. Do not

* Do not change `window_size`, `lora_rank`, `learning_rate`, the global batch, or `FlowHeads`' `n_latents` for this
  arm alone.  They are shared with every LIBERO cell and with the baseline; changing one side voids the comparison.
* Do not run the in-run merge (`--merge False` is in the commands for a reason).
* Do not report `action_l1_1step` from `train_log.jsonl` as a prediction error.  It is **oracle-conditioned**:
  `FlowHeads.loss` computes `(at + (1-t)*va - actions).abs()` where `at` already contains the ground-truth action.
  It is valid for ranking checkpoints under one fixed protocol and for comparing two arms on the SAME dataset and
  normalisation, and for nothing else.  The number a robot would see comes from `piper_policy.py --replay`.
* Do not quote offline L1 as if it predicted success rate.  It does not.
