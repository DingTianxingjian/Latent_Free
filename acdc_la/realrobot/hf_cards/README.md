# Tracked copies of the Hugging Face model cards

The deployable checkpoints live on `Potestates/TERRA` (public, no token). The cards there are the thing a
collaborator actually reads before putting a policy on the arm, so they are version-controlled here too — HF has no
history a reader of this repo can see, and a card that drifts from the ledger is how a wrong action space or a wrong
chunk length reaches the robot.

| file | HF path |
|---|---|
| `TERRA_root_README.md` | `README.md` (the repo card: one row per model, "which one do I test", the two deployment traps) |
| `piper_pick_can_terra_delta.md` | `piper_pick_can_terra_delta/README.md` |
| `piper_pick_can_openvla_delta.md` | `piper_pick_can_openvla_delta/README.md` |

Cards written on the Jetstream host (`press_button_*`, `piper_multi_anchored_*`, `p2fft_20k`, the LIBERO ones) are
not mirrored here yet. If you edit a card, edit it here and re-upload, not the other way round.
