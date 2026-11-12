# Patches to vendored third-party code

`third_party/` is gitignored, so every modification we make to UniVLA's checkout lives here as a diff and must be
re-applied after a fresh clone.  `REPRODUCE.md` lists the clone step; apply these immediately after it:

    cd third_party/UniVLA && git apply ../../acdc_la/patches/*.patch

Each patch states what it changes and why in its own header comment inside the diff.

## 2026-11-12 — `stack_bowls` registration, applied on the UCF workstation but NOT yet in a patch file

`univla_piper.patch` is written from the Jetstream host and carries entries this workstation's checkout does not
have (`press_button`, `press_button_anc`, `piper_pick_can_anc`, the `piper_multi*` mixtures,
`ActionEncoding.JOINT_ANCHOR`).  Regenerating the patch from this host would **delete** them, so the two insertions
below were applied to the local checkout by hand and are recorded here instead.  Fold them into
`univla_piper.patch` from a host that has both sets.

`prismatic/vla/datasets/rlds/oxe/configs.py`, straight after the `piper_pick_can` entry:

```python
    # Real robot (acdc_la 2026-11-12): AgileX Piper, anhdao69/data_real_4tasks/stack_bowls (100 episodes, one
    # instruction "stack the bowls") via acdc_la/realrobot/convert_press_button_rlds.py --dataset stack_bowls.
    # Same contract as piper_pick_can: 6 ABSOLUTE joint targets + gripper/8.2, EEF_POS declared for prismatic only.
    "stack_bowls": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
```

`prismatic/vla/datasets/rlds/oxe/transforms.py`, next to `piper_pick_can`:

```python
    "stack_bowls": piper_dataset_transform,          # acdc_la 2026-11-12, same Piper contract
```
