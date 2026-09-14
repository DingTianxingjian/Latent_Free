# Overnight Diagnostic Experiments: Do Visual Feature Differences Encode Actions?

## Context (read first)

We are studying **label-free latent action learning** (LAM-family methods: LAPA, UniVLA, Genie-style).
Theory ("What Do Latent Action Models Actually Learn?", NeurIPS 2025) predicts LAM latents ≈ PCA of
frame differences → they capture the *largest-variance* changes, not necessarily *action-caused* changes.
LAOM (ICML 2025) showed this empirically in toy DMC environments. Nobody has measured this cleanly on
real manipulation data at the action-decoding level.

Tonight's goal: **two fast diagnostic experiments, no model training**, only frozen encoders + linear
probes. Results go into a 3-slide advisor update tomorrow morning. Prioritize getting numbers out over
polish. Everything must run on a single GPU node (A100 40GB or GH200).

## Environment setup

1. Python env: torch, torchvision, timm (or torch.hub), scikit-learn, numpy, h5py, imageio, matplotlib, pandas.
2. Dataset: **LIBERO**, suite `libero_spatial` only (10 tasks, ~50 demos/task, hdf5 files contain
   `agentview_rgb` images and 7-dim actions: 6D EEF delta pose + 1D gripper).
   - First check if LIBERO data already exists on the cluster (search common paths, ask me if unsure).
   - Otherwise download via the official LIBERO repo instructions (datasets are a few GB).
   - We only need the hdf5 demo files, NOT the simulator. Do not install robosuite/mujoco.
3. Encoder: **DINOv2 ViT-B/14** via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`.
   Use the CLS token (dim 768) as the frame feature. Resize frames to 224×224, ImageNet normalization.
   (If DINOv2 download fails on the cluster network, fall back to torchvision ViT-B/16 ImageNet weights.)

## Shared data pipeline

- Load all episodes from libero_spatial. For each episode, extract frames and actions.
- Compute frame features f(o_t) for all frames (batch on GPU, cache to disk as .npy — features for the
  whole suite are small, ~a few GB max).
- Build pairs: Δf_t = f(o_{t+stride}) − f(o_t), target a_t = sum of actions over [t, t+stride)
  (for gripper: take the value at t+stride−1, treat as binary label).
- Run everything for **stride ∈ {1, 4}**.

### CRITICAL correctness rules
- **Split by episode, never by frame.** 80/20 train/val split at the episode level, fixed seed.
  Frame-level splits leak (adjacent frames nearly identical) and invalidate all results.
- Report metrics **per action dimension group**: translation xyz (R²), rotation (R²),
  gripper (balanced accuracy). Also report overall R² on the 6 continuous dims.
- Standardize features and continuous targets using train-set statistics only.

## Experiment 1: Action decodability from feature differences (~3–4 h, must finish)

**Question**: how much ground-truth action is linearly decodable from Δf of a frozen generic encoder?
This is the reference ceiling for what any LAM (which compresses Δf) could contain.

- Probe: ridge regression (sklearn, closed form) for continuous dims; logistic regression for gripper.
  Sweep ridge alpha ∈ {1e-2, 1e-1, 1, 10, 100} on train, report best-on-val (note: small sweep is fine
  for a sanity check; mention alpha chosen).
- Ablation (cheap, do it): probe from single-frame feature f(o_t) alone → shows how much of the
  "decodability" is trivial state correlation rather than transition information. Report side by side.
- Output: `results/exp1_decodability.csv` (rows: stride × feature-type × dim-group, cols: metric),
  plus a bar chart `figs/exp1_r2_bars.png`.

**Expected**: translation R² roughly 0.5–0.8, rotation lower. If everything is ≈0, suspect a pipeline
bug (wrong frame/action alignment, wrong normalization) before concluding anything.

## Experiment 2: Distractor contamination (~2–3 h, must finish; parallelize with Exp 1)

**Question**: how badly does task-irrelevant motion contaminate Δf (and hence any LAM trained on it)?

- Distractor synthesis (image-level, do NOT touch the simulator): overlay a moving natural-video patch
  onto each frame. Implementation: take any short natural video clip (e.g., download one sample clip,
  or use noise-textured moving patches if no video available), crop a 64×64 patch, composite it onto
  the frame border region, moving at constant velocity along the image edge with wrap-around. The
  distractor motion must be **temporally coherent within an episode** (position advances with t) and
  independent of the robot. Also implement a second, stronger condition: replace the outer 20% border
  ring of every frame with a playing video. Conditions: {clean, patch, border}.
- Re-extract features for the two corrupted conditions, re-run the Exp-1 probe (train and eval within
  each condition).
- Additional analysis (this is the theory link, do not skip):
  1. PCA on Δf for clean vs corrupted data (fit on train split).
  2. Compute how much variance the top-k PCs (k = 8, 16, 32) capture, and
  3. Project the corrupted-data top-k PC subspace against the "action subspace" (span of the ridge
     probe weight vectors fitted on clean data): report the fraction of action-subspace energy retained
     in the top-k PCs, clean vs corrupted. This directly tests "distractors occupy the dominant
     variance directions that a LAM bottleneck would keep".
- Output: `results/exp2_distractor.csv`, `figs/exp2_r2_clean_vs_distracted.png` (grouped bars),
  `figs/exp2_pca_action_energy.png` (action-energy-in-top-k-PCs vs k, one line per condition),
  plus 3 example corrupted frames saved for the slides.

## Experiment 3: OPTIONAL, hard 2-hour budget, abandon if blocked

Load a pretrained open-source LAM checkpoint (try LAPA first, then UniVLA), encode the same LIBERO
frame pairs into latent action z, probe z → action with the identical protocol, and add one row to the
Exp-1 table. If checkpoint loading, dependency, or input-format issues eat more than 2 hours total,
stop and skip — this is a bonus, not the main line.

## Deliverables

1. `results/` CSVs + `figs/` PNGs as specified above.
2. A single `SUMMARY.md` with: the final numbers table, 3 bullet takeaways, and any caveats/bugs
   suspected. Written so it can be pasted into slides directly.
3. Log the exact commit/seed/paths so everything is rerunnable.

## Style constraints

- Plain scripts (`exp1.py`, `exp2.py`, shared `data.py`, `features.py`), no framework, no config system.
- Fail loudly: assert frame/action alignment (len(actions) == len(frames) − expected offset per LIBERO
  format — verify against the actual hdf5 structure and print it before assuming).
- Print dataset stats up front (n episodes, n frames, action ranges) so silent data bugs are visible.