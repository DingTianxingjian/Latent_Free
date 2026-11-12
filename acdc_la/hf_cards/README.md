# Tracked copies of the Hugging Face model cards

The checkpoints live on `Potestates/TERRA` (public, no token). The cards there are the thing a collaborator actually
reads before putting a policy on the arm or quoting a number, so they are version-controlled here too — HF has no
history a reader of this repo can see, and a card that drifts from the ledger is how a wrong action space, a wrong
chunk length or a withdrawn conclusion reaches the robot or a paper.

| file | HF path | what it is |
|---|---|---|
| `TERRA_root_README.md` | `README.md` | the repo card: "which one do I test", the **ours-vs-controls inventory**, the two deployment traps |
| `piper_pick_can_terra_delta.md` | `piper_pick_can_terra_delta/README.md` | TERRA, pick-can, per-step delta |
| `piper_pick_can_openvla_delta.md` | `piper_pick_can_openvla_delta/README.md` | matched control, pick-can |
| `libero_goal_frac10_terra.md` | `libero_goal_frac10_terra/README.md` | TERRA latent, Goal @ 10 % |
| `libero_goal_frac10_univla_cont.md` | `libero_goal_frac10_univla_cont/README.md` | baseline: UniVLA's latent, same interface |
| `libero_goal_frac10_none.md` | `libero_goal_frac10_none/README.md` | matched control: no latent at all |
| `libero_10_frac10_terra_p2fft20k.md` | `libero_10_frac10_terra_p2fft20k/README.md` | TERRA full stack, Long @ 10 % |

Cards written on the Jetstream host (`press_button_*`, `piper_multi_anchored_*`, `p2fft_20k`) and the two
`libero_*_full` cards written on Newton are not mirrored here yet. If you edit a card, edit it here and re-upload,
not the other way round.
