# ACDC-LA — Preserve, Disentangle, Transport

Latent action model built as three orthogonal stages, each with its own objective and its own metric,
trained on the OXE mixture UniVLA uses (25 sources, UniVLA's sampling weights) with a frozen DINOv2 ViT-B/14.
先保留，再分离，再迁移。

| Stage | Name | What it fixes (as the field states it) | Objective | Metric (this repo) | Evidence status |
|---|---|---|---|---|---|
| I | **Preserve** — AC/DC temporal factorisation | Pooling / endpoint differencing collapses a K-step effect to its mean; the action's temporal shape is gone before any bottleneck sees it | rec + KL on effect cells = 4 quadrants × {DC, AC1} of the frozen feature-difference sequence | temporal-variance kept and action probes (net / seg1 / **prof** / **seq**) per temporal subset, model-free on frozen features and on z; time-reversal parity of z; patch drop | candidate (Bridge: z→action 0.086→0.221; `stage1_preserve/probes.py` on the mixture: see ledger) |
| II | **Disentangle** — TC/TI by intervention | UniVLA's controllable/uncontrollable heads share one loss; the released weights do not separate (audit 2026-08) | Stage I on clean and moving-patch renderings + z-invariance + cross-reconstruction (nuisance must live in u) | **G_sep** = (S_TC − S_TI) + (N_TI − N_TC) vs UniVLA's two heads on the same clips | no evidence yet (code ready, not run) |
| III | **Transport** — LATP | A code that is linearly decodable to the action is not thereby realisable on a new state; pair-based transport needs labels | tokenizer frozen (∇_E L = 0); transport operator T(s, z) trained by swap realism (MMD) + cycle, no labels, no pairs | independent effect reader: transfer R² / cosine / calibrated R², replay, E_correct vs E_wrong, few-label recovery curve | **confirmed** on the controlled tokenizer: decoder-only 0.449 → 0.512 cosine, transfer R² 0.253 → 0.354, clean unchanged (ledger 2026-09-08; reproduced by `stage3_transport/probes.py` on the legacy dumps) |

Ablations (not contributions): relation diff/sym, RALA, swapalg (seed-unstable), grank (a scalar calibration), InfoNCE,
canonical head, full-swap (encoder trained: lower transfer, clean cost).

## Layout

```
acdc_la/
  paths.py                    filesystem constants (caches, results, ckpt, legacy claude_survey paths)
  data/oxe_mix.py             MixDS: multi-source clips, UniVLA sampling, source-pure batches, nearest-action val pairs
  data/distractor.py          moving-patch nuisance rendering + its exact trajectory (Stage II target)
  common/features.py          DINOv2 g4, quadrants, DCT basis (DC = sum, AC_m orthonormal), effect cells, reversal parity
  common/model.py             LAM: tokenizer (E, U) + state-conditioned transformer decoder; loads legacy swaplam_*.pt
  common/pipeline.py          Featurizer: batch -> g4, dF, f0, c, cells (pool = acdc | mean | endpoint, n modes)
  common/loaders.py           train (source-homogeneous) and val (stratified pairs) loaders
  common/train_loop.py        the one training loop; a stage supplies step(); init-from / freeze-encoder / ramp
  common/dump.py              one validation pass -> results/<tag>_val.npz (codes, patch view, reversed view, decodes)
  common/probes.py, readers.py   per-source episode-disjoint ridge, aggregation, legacy-dump loader
  stage1_preserve/{train,probes}.py
  stage2_disentangle/{train,probes}.py
  stage3_transport/{train,probes,recovery}.py
  stage4_portable/train.py            portable core + reconstruction residual on a frozen LATP checkpoint (common/pcrr.py); probes = Stage I/II/III ones
  baselines/univla.py         UniVLA stage-2 codes (TC + TI heads) on the same clips, same dump format
  scripts/run_stage{1,2,3}.sh, run_univla_dump.sh   slurm (ai partition), sequential arms, probes at the end
  results/                    json (tracked) + npz dumps (ignored);  ckpt/, logs/ ignored
```

## Run

```bash
cd /anvil/scratch/x-tding2/latent_free
sbatch acdc_la/scripts/run_stage1.sh                      # s1_acdc, s1_dc, s1_mean, s1_acdc3  + stage1 probes
INIT=s1_acdc sbatch acdc_la/scripts/run_stage2.sh         # s2_tcti, s2_aug, s2_inv, s2_xrec   + G_sep
INIT=s3_latp_1ep sbatch acdc_la/scripts/run_stage4.sh    # s4_pcrr + s3_latp_1ep_cont control + Stage I/II/III probes on both
INIT=s1_acdc sbatch acdc_la/scripts/run_stage3.sh         # s3_latp, s3_ctrl, s3_full          + transport readers, recovery curve
sbatch acdc_la/scripts/run_univla_dump.sh                 # UniVLA baseline dump + its stage1/stage2 probes
# offline, CPU, any dump (this repo's tag, a legacy swaplam tag, or a path):
python -m acdc_la.stage1_preserve.probes rec_src s1_acdc univla_official
python -m acdc_la.stage3_transport.probes rec_src swap_fz s3_latp
```

`RUNS="tag:extra args;tag2:..."`, `STEPS`, `SEED`, `SUFFIX` override the defaults in every script.  Stage III from
the legacy tokenizer (`INIT=rec_src`) reproduces the ledger's `swap_fz`.

## Legacy mapping (claude_survey → here)

| legacy | here |
|---|---|
| `swaplam.py --arm rec` | `stage1_preserve.train --pool acdc --modes 2` |
| `swaplam.py --arm swap --init-from rec_src --freeze-encoder` (swap_fz) | `stage3_transport.train --init-from rec_src` |
| `swaplam.py --arm swap` (full) | `stage3_transport.train --train-encoder` |
| `swaplam.py --arm rec --init-from rec_src --freeze-encoder` (rec_fz) | `stage3_transport.train --lam-real 0 --lam-cyc 0` |
| `swaplam.py --arm pairpatch` | `stage2_disentangle.train` (with the cross-reconstruction made symmetric) |
| `diag_transfer.py` + readout probes | `stage3_transport.probes` |
| `recovery_curve.py` | `stage3_transport.recovery` |
| `results/swaplam_<tag>_readout_arrays.npz` | readable by every probe (`common/readers.load_arrays`) |

Checkpoint attribute names are unchanged, so `claude_survey/ckpt/swaplam_*.pt` load directly (`common.model.load_ckpt`).

## Ledger

Numbers and decisions live in `LEDGER.md` at the repository root (confirmed / provisional / retracted).  Every
run here writes `results/<tag>.json` and `results/<tag>_val.npz`; probes append to `results/stage{1,2,3}_probes.json`
and `results/recovery_curve.json`.
