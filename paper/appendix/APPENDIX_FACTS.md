# Appendix facts (checked against the repository, 2026-09-25)

Everything below is read from code, checkpoints, logs or result files in this repo.  "NOT IN REPO" means no record
exists here; do not fill it from memory.

## Compute
- Hardware: Anvil `ai` partition, nodes h000-h020, 4 x NVIDIA H100 per node (`AvailableFeatures=H100`, `Gres=gpu:4`).
- Tokenizer, 65k steps (`tc_cont`): 1 x H100, 0.43 s/step as logged at step 65,000 (three jobs sharing the OXE cache),
  i.e. ~7.8 GPU-hours.  Run in two walltime legs.
- EAT continuation, 15k steps (`latp2_inv`): 1 x H100, 5,550 s = 1.5 GPU-hours (0.36 s/step).
  Controls: `latp2_anchor` 5,732 s (1.6 h); `tc_cont_plus15k` 4,131 s (1.1 h).
- VLA post-training on LIBERO (Anvil runs): 4 x H100, 15k steps x batch 64, 5.4-5.6 h per suite (ledger 2026-09-16/18).
- Latent-action VLM pretraining on 10 % of the mixture (the main-table closed loop): **NOT IN REPO** -- run on the
  user's other machine; GPU type and hours must come from there.

## "LAPA-style" / VQ arm
- No LAPA model was run.  The only VQ variant of our tokenizer is `tc_vq256`: codebook 256 entries, 16 codes per
  transition window (5 frames, 4 segments), 64-d codes, 75,000 steps of a 128k cosine schedule, batch 64, lr 3e-4,
  commitment 0.25, dead-code restart every 5k steps (`ckpt/tc_vq256.pt` args).  The LAPA row of the comparison figure
  is a placeholder.

## Closed-loop perturbation protocol
- **NOT IN REPO.**  No perturbed closed-loop evaluation was run; `figures/rollout_compare` is a layout mock-up.
  Delete the protocol paragraph (and any perturbed-success table) unless it is run.

## Linear reader (all probe numbers)
- `acdc_la/common/probes.py::per_source_readouts`, one ridge per source and per target.
- Score split: per source, 1/3 of the EPISODES held out for scoring (numpy RandomState(1)); the other 2/3 train.
- Lambda: chosen per source and target from {0.1, 1, 10, 100, 1e3, 1e4} on an episode-disjoint inner split holding out
  20 % of the training episodes (RandomState(0)); refit on all training episodes with the chosen value.  Dual form
  when dims > rows.
- Features standardised with training mean/sd and clipped to +-10; constant training columns dropped.
- R^2 averaged over target dims that vary on the test rows; NMSE = 1 - R^2.  Aggregate: clip-weighted mean over
  sources with >= 80 test clips (29 sources).  Validation set: 400 clips per source (200 stratified nearest-action
  pairs), T = 5 frames, stride 4.
- Targets from the 4 segment displacements a_k (xyz): net = sum_k a_k; seg1 = a_0; seq = (a_0..a_3), 12-d;
  profile = a_k - mean_k a_k.

## Distractor (moving patch)
- `features.py::apply_distractor`, condition `patch`.  A 64 x 64 crop of a natural video (`data/distractor_video.npy`,
  200 frames, resized to 224 x 224) pasted into each 224 x 224 frame.  **Opaque paste -- there is no alpha / blending
  factor**; if the text mentions an alpha, it does not match the code.
- Crop location, video start frame, start position and speed are drawn once per episode (RandomState(10000 + episode
  id)); the video advances one frame per clip frame, so the patch content changes over time.
- Motion: the patch centre travels along the image perimeter, inset by 32 px, at a constant 4-8 px per clip frame,
  direction random; independent of the robot.
- Drop % = 100 x (1 - R^2_patch / R^2_clean) for the Net target, where the probe fitted on clean codes reads the codes
  of the SAME clips rendered with the patch (ratio of source-aggregated means).

## Confidence intervals
- 2,000 bootstrap resamples of the 29 sources, the same resample for every arm; percentile 95 % intervals.  Paired
  intervals are percentiles of (TERRA - arm) over the shared resamples.  They measure dependence on the source mix,
  NOT seed variance (every arm is one training seed).
- `paper/appendix/nmse_ci.csv` (9 arms x 5 metrics = 45 cells; pick the rows the table uses) and
  `paper/appendix/nmse_ci_paired_terra_minus_arm.csv`.

## Effective rank -- state which definition
- Standard (Roy & Vetterli, 2007), used by the training logs (`train_v2.py::_erank`):
  erank = exp(-sum_i p_i log p_i), p_i = sigma_i / sum_j sigma_j, sigma = singular values of the centred code matrix.
- Variance-spectrum variant: the same with p_i = sigma_i^2 / sum_j sigma_j^2 (entropy of the PCA eigenvalues).
- On the validation codes: `tc_cont` 210.9 / 127.7, `tc_cont_plus15k` 211.5 / 128.4, `latp2_anchor` 203.7 / 90.1,
  `latp2_inv` 165.7 / 24.6 (standard / variance).  The training-log `z_erank` (e.g. 48.3) is the standard definition
  on a 64-clip training batch, so it is capped by the batch size and not comparable to the validation numbers.

## Per-source transport table (Fig. 4b)
- `paper/appendix/transport_by_source.csv`: 29 rows -- source, n_test_clips, recon / EAT transported-action NMSE for
  far donors (mean of Q3 and Q4) and for all donors.  Protocol: `acdc_la/downstream/transport_by_source.py`.
