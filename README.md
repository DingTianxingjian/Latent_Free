# Latent-action diagnostics on LIBERO-spatial

Two no-training diagnostics (plus one optional bonus) probing whether visual
feature *differences* actually encode actions — the empirical question behind
"LAM latents ≈ PCA of frame differences".

Everything runs from frozen encoders + linear probes. No model is trained.

## Layout

| file | what it does |
|---|---|
| `data.py` | LIBERO hdf5 loading, episode-level split, frame/action pairing. Run it alone to print dataset stats. |
| `features.py` | Distractor synthesis + DINOv2 ViT-B/14 CLS feature extraction, cached per condition. |
| `probes.py` | Ridge / logistic probes + metrics, shared by all experiments. |
| `viz.py` | Plot styling and the validated categorical palette. |
| `exp1.py` | Action decodability from Δf vs from a single frame. |
| `exp2.py` | Distractor contamination + PCA / action-subspace analysis. |
| `exp3.py` | *Optional*: probe UniVLA's pretrained stage-1 latent action model. |
| `features_patch.py` | Patch-token features (mean-pooled and 4×4 grid) from the same encoder. |
| `probes_torch.py` | GPU ridge/logistic for the wide patch spaces. `--verify` checks it against `probes.py`. |
| `exp4_patch.py` | *Follow-up*: CLS vs patch tokens × single/diff/pair, and how much of a two-frame score is just single-frame state. |
| `run_all.sh` | Single-job pipeline (stats → features → exp1 ‖ exp2) for the `gpu` queue. |
| `run_features_debug.sh` | GPU stage only, sized for the 30-min `gpu-debug` queue. |
| `run_analysis_cpu.sh` | exp1 ‖ exp2 on the CPU `debug` queue. |
| `run_exp3.sh` | Optional LAM probe, one stride per submission. |

## Environment

```
/anvil/scratch/x-tding2/envs/latent_free      # venv layered on envs/orena
```
It is a `--system-site-packages` venv on top of the pre-existing `orena` conda
env, so `torch 2.7.0+cu118 / torchvision / timm / matplotlib / pandas / cv2` are
inherited and only `scikit-learn, h5py, imageio, sentencepiece,
rotary-embedding-torch` are installed locally. The `orena` env is not modified.

## Data & checkpoints (all pre-fetched; compute nodes run offline)

| what | path | source |
|---|---|---|
| LIBERO `libero_spatial` (10 tasks, 500 demos, 62 250 frames, 6.2 GB) | `data/LIBERO/libero_spatial/*.hdf5` | HF `yifengzhu-hf/LIBERO-datasets` |
| DINOv2 ViT-B/14 weights | `cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` | `dl.fbaipublicfiles.com` |
| DINOv2 hub repo (for offline `torch.hub`) | `cache/torch/hub/facebookresearch_dinov2_main` | GitHub |
| distractor video clip | `data/distractor_video.npy` (200 frames of `imageio:cockatoo.mp4`) | imageio sample data |
| UniVLA stage-1 LAM (exp3) | `cache/univla/lam-stage-1.ckpt` | HF `qwbu/univla-latent-action-model` |
| UniVLA source + t5-base (exp3) | `third_party/UniVLA` | GitHub / HF `google-t5/t5-base` |

The simulator is **not** installed — no robosuite, no mujoco. Only the hdf5 demo
files are used.

## Reproducing

Use the **`ai` partition** (H100). It is by far the least contended GPU queue on
Anvil — when this was run `gpu` had ~370 jobs pending (est. start 5 days out) and
`gpu-debug` ~4 days, while `ai` was starting jobs the same hour.

```bash
sbatch --partition=ai --account=cis250950-ai --cpus-per-task=24 \
       --gpus-per-node=1 run_all.sh    # features for 5 conditions, then exp1 ‖ exp2
sbatch run_exp3.sh                     # optional LAM probe (also on ai)
sbatch run_exp4.sh                     # follow-up: patch tokens + state control
```

Fallbacks if `ai` is ever full — only feature extraction needs a GPU, so the
pipeline can be split across the short queues:

```bash
sbatch run_features_debug.sh   # gpu-debug, 30 min wall; resumable, resubmit
                               # until it prints "still pending: none"
sbatch run_analysis_cpu.sh     # CPU `debug` queue (2 h) -- exp1 ‖ exp2
```

Fixed choices, identical across every experiment:

- **seed 0**, 80/20 **episode-level** split, stratified within each of the 10
  tasks (40 train / 10 val demos per task → 400/100 episodes).
- strides **{1, 4}**; target = sum of actions over `[t, t+stride)` for the 6
  continuous dims, gripper taken at `t+stride-1` as a binary label.
- features and continuous targets standardized with **train statistics only**.
- ridge `alpha ∈ {1e-2, 1e-1, 1, 10, 100}`, logistic `C ∈ {1e-2, 1e-1, 1, 10, 100}`.
  `exp4_patch.py` widens the ridge grid to `1e-4 … 1e5`, because exp 1 selected
  the top of the specified grid and exp 3 the bottom, and its feature spaces span
  768 to 24576 dims — the right shrinkage is not comparable across those rows.
  Caching the gram makes the extra alphas free.

## Verified dataset facts

Printed by `python data.py` at the top of every run:

- `len(actions) == len(agentview_rgb) == T` for **all 500 episodes** — there is no
  off-by-one offset in this LIBERO release. `obs[t]` + `actions[t]` → `obs[t+1]`,
  so the pair `(o_t, o_{t+s})` is explained by `actions[t:t+s]`.
- frames are stored bottom-up (`macros_image_convention == 'opengl'`) and are
  flipped to upright on load — verify with `figs/_check_raw.png` vs
  `figs/_check_flipped.png`.
- `actions[:,6]` (gripper) takes exactly two values, `{-1, +1}`, 45/55 balance.
- translation std ≈ 7× rotation std, which is why rotation R² is expected lower.

## Outputs

`results/exp1_decodability.csv`, `results/exp2_distractor.csv`,
`results/exp2_pca_action_energy.csv`, `results/exp3_lam.csv`,
`figs/exp1_r2_bars.png`, `figs/exp2_r2_clean_vs_distracted.png`,
`figs/exp2_pca_action_energy.png`, `figs/example_*.png`, and `SUMMARY.md`.
