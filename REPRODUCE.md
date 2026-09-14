# Reproducing the ACDC-LA experiments on a new server

Repository: https://github.com/DingTianxingjian/Latent_Free (branch `main`; `git clone https://github.com/DingTianxingjian/Latent_Free.git latent_free`).
Written 2026-09-13.  Everything below was verified on Anvil (Slurm, H100/A100).  Paths are relative to the
repository root, which must be called `latent_free/` (the code resolves `LF = latent_free/` from `acdc_la/paths.py`
and imports `data.py` from it).

## 0. What lives where

| what | where | how to get it |
|---|---|---|
| our code | `acdc_la/` (+ `data.py` for LIBERO hdf5 access) | this repository |
| UniVLA code | `third_party/UniVLA` | `git clone https://github.com/OpenDriveLab/UniVLA.git && git -C third_party/UniVLA checkout 0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352` (our copy is byte-identical to that commit in `prismatic/`, `latent_action_model/`, `experiments/`, `vla-scripts/`) |
| checkpoints | `acdc_la/ckpt/*.pt`, `claude_survey/ckpt/s2mix_1ep.pt` | `transfer/acdc_la_artifacts.tar` on Anvil (see §2) |
| LIBERO demos (hdf5, 5.9 GB) | `data/LIBERO/libero_spatial/*.hdf5` (10 files) | LIBERO's `libero/scripts/download_libero_datasets.py --datasets libero_spatial` (or copy from Anvil) |
| LIBERO RLDS for the VLA (1.8 GB) | `data/libero_rlds/libero_spatial_no_noops/1.0.0` | `huggingface_hub.snapshot_download("openvla/modified_libero_rlds", repo_type="dataset", allow_patterns=["libero_spatial_no_noops/**"], local_dir="data/libero_rlds")` |
| OpenVLA-7B base (15 GB) | `cache/vla/openvla-7b` | `snapshot_download("openvla/openvla-7b", local_dir="cache/vla/openvla-7b")` |
| DINOv2 weights | `cache/torch/hub/` | auto-downloaded by `torch.hub` on first use (`facebookresearch/dinov2`, `dinov2_vitb14` for us, `dinov2_vitb14_reg` for their LAM); copy the folder if the compute nodes are offline |
| UniVLA released LAM (only for the reference row) | `cache/univla/lam-stage-{1,2}.ckpt` | UniVLA README links (HF `qwbu/univla-latent-action-model`) |
| OXE pretraining caches (1.8 TB) | `cache/oxe/<source>` | NOT needed for the downstream comparison; only for re-pretraining (`phase1/oxe/oxe_convert.py`) |

Filesystem layout the code expects (create the empty dirs):
```
latent_free/
  acdc_la/{ckpt,results,logs}/        # results has the json tables; ckpt gets the .pt files
  cache/{torch/hub,vla,univla}/  cache/index.npz   # LIBERO step index (in the bundle), or `python data.py` rebuilds it
  data/LIBERO/libero_spatial/*.hdf5   data/libero_rlds/
  third_party/UniVLA/
  envs/                               # optional: where the venvs live on Anvil (paths in scripts point to /anvil/scratch/x-tding2/envs/...)
```

## 1. Environments (three; `uv` recommended, Python 3.10)

Frozen package lists are in `acdc_la/envs/*.requirements.txt` (exact versions that ran).

1. **`latent_free`** (pretraining + probes): torch 2.4 cu121, numpy, scipy, h5py, timm, tqdm.
   `uv venv envs/latent_free --python 3.10 && uv pip install -r acdc_la/envs/latent_free.requirements.txt`
   (that file is short; if it fails, `uv pip install torch==2.4.1 torchvision numpy scipy h5py timm tqdm pillow`).
2. **`libero`** (offline LIBERO tables, closed-loop toy policies, VQ gate): torch 2.4.1 cu121, robosuite 1.4.1, mujoco 2.3.7,
   pandas, scikit-learn, rotary-embedding-torch, timm; LIBERO itself is used from a checkout on `PYTHONPATH`
   (`git clone https://github.com/Lifelong-Robot-Learning/LIBERO envs/LIBERO`).  `envs/setup_libero.sh` on Anvil is the
   script that built it; `acdc_la/envs/libero.requirements.txt` is the freeze.
3. **`univla_vla`** (the full-VLA comparison): UniVLA's `pyproject.toml` pins + fixes.
   ```
   uv venv envs/univla_vla --python 3.10
   VIRTUAL_ENV=envs/univla_vla uv pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   cd third_party/UniVLA && VIRTUAL_ENV=../../envs/univla_vla uv pip install -e . && cd ../..
   VIRTUAL_ENV=envs/univla_vla uv pip install "tensorflow_metadata==1.14.0" "protobuf==3.20.3" "wandb==0.16.6" braceexpand webdataset bddl easydict cloudpickle "imageio[ffmpeg]"
   ```
   The three pins after `-e .` are required: tfds 4.9.3 breaks with protobuf 5, wandb >= 0.17 needs protobuf 5.
   Run everything with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false`.
   Check: `python -c "import prismatic.training; from prismatic.vla.datasets import RLDSDataset; print('ok')"`.

Every `acdc_la/scripts/*.sh` hard-codes the Anvil paths (`/anvil/scratch/x-tding2/...`, Slurm account/partition
headers).  On a new server: `sed -i 's|/anvil/scratch/x-tding2/latent_free|/NEW/latent_free|g; s|/anvil/scratch/x-tding2/envs|/NEW/envs|g' acdc_la/scripts/*.sh`
and edit the `#SBATCH` lines (or run the python commands directly; each script is a thin wrapper).

## 2. Checkpoints (transfer bundle on Anvil)

`transfer/acdc_la_artifacts.tar` (~1.2 GB) holds the four checkpoints, the LIBERO step index and the codebooks:
```
rsync -avP <anvil-user>@anvil.rcac.purdue.edu:/anvil/scratch/x-tding2/latent_free/transfer/acdc_la_artifacts.tar .
tar -xf acdc_la_artifacts.tar -C latent_free/        # unpacks acdc_la/ckpt/*.pt, claude_survey/ckpt/s2mix_1ep.pt, cache/index.npz
```
(Optional mirror: `huggingface-cli upload --private Potestates/acdc-la-artifacts transfer/acdc_la_artifacts.tar` from Anvil,
then `snapshot_download` on the new server.  Not done automatically.)
* `acdc_la/ckpt/s1_acdc_1ep.pt`: Stage-I ACDC tokenizer (8 cells x 32, T=5, stride 4), 128k steps x 64 on the capped OXE mixture.
* `acdc_la/ckpt/s3_latp_1ep.pt`: LATP transport decoder on top of it (tokenizer frozen, 10k steps).  `s1_mean_1ep.pt`: pooled ablation.
* `claude_survey/ckpt/s2mix_1ep.pt`: UniVLA Stage-2 recipe trained by us on the same mixture at the same budget (their class
  `ControllableDINOLatentActionModel`; the frozen DINO encoder is not stored, `lang_proj.*` keys are extra; load with
  `strict=False` as `acdc_la/downstream/vla/finetune_libero_ctrl.py:build_labeller` does).
* `acdc_la/ckpt/vq_s1_acdc_1ep_*.npz`: per-cell / product-quantisation codebooks fitted on the OXE validation dump (also in git under `acdc_la/results/codebooks/`).
* `cache/index.npz`: LIBERO step index used by `data.py` (rebuilt by `python data.py` from the hdf5 files if missing).

## 3. The experiments, in priority order

### 3a. Full-VLA controlled comparison (the success-rate result)  — `acdc_la/VLA_DOWNSTREAM_PLAN.md`
Both arms: UniVLA's LIBERO fine-tuning recipe from `openvla/openvla-7b`, only the latent labels differ.
```
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=$PWD/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PY=envs/univla_vla/bin/python
# smoke (1 GPU, ~15 min incl. the merge): 12 steps each
$PY -m torch.distributed.run --standalone --nproc-per-node 1 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent acdc   --max_steps 12 --save_steps 12 --batch_size 2 --grad_accumulation_steps 1 --shuffle_buffer_size 200 --log_every 2 --run_id_note smoke
$PY -m torch.distributed.run --standalone --nproc-per-node 1 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent univla --max_steps 12 --save_steps 12 --batch_size 2 --grad_accumulation_steps 1 --shuffle_buffer_size 200 --log_every 2 --run_id_note smoke
# production (4 x 80 GB GPUs, batch 8 x accum 2 per GPU = 64/step, 15k steps ~ 4-5 h each)
$PY -m torch.distributed.run --standalone --nproc-per-node 4 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent univla --max_steps 15000
$PY -m torch.distributed.run --standalone --nproc-per-node 4 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent acdc   --max_steps 15000 --acdc_codebook vq_s1_acdc_1ep_K64.npz
# -> acdc_la/ckpt/vla_runs/<exp_id>/ : merged model, action_decoder.pt, ctrl_meta.json, dataset_statistics.json, train_log.jsonl
# evaluation (1 GPU, 50 trials x 10 tasks ~ 2-3 h per arm; needs envs/LIBERO on PYTHONPATH and MUJOCO_GL=egl)
PYTHONPATH=$PYTHONPATH:envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY -m acdc_la.downstream.vla.run_libero_eval_ctrl --pretrained_checkpoint acdc_la/ckpt/vla_runs/<exp_id> --task_suite_name libero_spatial --num_trials_per_task 50 --local_log_dir acdc_la/results/vla_eval
```
Slurm wrappers: `acdc_la/scripts/run_vla_smoke.sh`, `run_vla_ctrl.sh` (`LATENT=univla|acdc CODEBOOK=...`), `run_vla_eval.sh` (`RUN=<exp_id>`).
Knobs: `--acdc_codebook` picks the token interface (K16/K64/K256 = 8 tokens; pq2x64 = 16; pq4x16 / pq4x64 = 32; pq8x16 = 64).
Their action decoder reads the last `n_tokens` latent states; the prompt carries the history tokens with p = 0.5 as in their code.

### 3b. VQ gate (is a discrete code of the existing tokenizer a better policy target?)  — `acdc_la/downstream/libero_vqgate.py`
```
PYTHONPATH=$PWD:envs/LIBERO envs/libero/bin/python -m acdc_la.downstream.libero_vqgate --arms ours vq64 vq256 pq4x16 pq2x64 pq4x64 --out libero_vqgate
```
1 GPU, ~1 h (first run extracts LIBERO DINO features, cached to `cache/libero_g4.pt`).  Columns: `l1 oracle` (dequantised
code), `l1 mode` / `l1 expect` (policy-predicted), `code_acc` vs majority.  Gate: a vq arm must beat `ours` chain L1 by >= 0.015.

### 3c. Offline / toy closed-loop tables (already in the ledger; rerun only if needed)
* `acdc_la/downstream/libero_offline.py --tags s1_acdc_1ep s1_mean_1ep --univla-arm s2mix_1ep` (needs `claude_survey/eval_rala2.py` for the UniVLA codes: not in this repo; the numbers are in `acdc_la/results/libero_matched.json`).
* `acdc_la/downstream/libero_policy.py --arms bc ours ours_noeff --rollout 10` (history policy recipe; `results/libero_policy.json`).
* `acdc_la/downstream/libero_closedloop.py` (MLP heads; `results/libero_matched_closedloop.json`).

### 3d. Pretraining probes (Stage I / II / III) on the OXE dumps
`acdc_la/results/*_val.npz` are the validation dumps (not in git, 350 MB each; on Anvil).  `python -m acdc_la.stage1_preserve.probes --code-only s1_acdc_1ep s1_mean_1ep`, `stage2_disentangle.probes`, `stage3_transport.probes` read them.
Re-pretraining needs the OXE caches (`phase1/oxe/oxe_convert.py`, 29 sources, 5,000 episodes each) and `acdc_la/scripts/run_long.sh` (12 h legs, 128k steps x 64 per epoch).

## 4. Results so far (2026-09-13; details and per-task rows in the ledger `SUMMARY.md`)
* Stage I (OXE val, corrected R2): ACDC 0.324 net / 0.264 seg1 / 0.085 prof vs pooled 0.336 / 0.210 / 0.017 vs same-data UniVLA 0.236 / 0.222 / 0.125; identical at 2 epochs.
* LATP transfer R2 0.319 (decoder-only 0.244); same-data UniVLA 0.116 (their reader).
* LIBERO-spatial offline: oracle L1 ours 0.344 < pooled 0.368 < UniVLA 0.377; after latent prediction all 0.42-0.43; no-latent 0.434.
* Toy closed-loop (100 eps/arm): 5-22 %, every arm inside the noise band, in two recipes.  Stage IV (portable core + residual) failed.

## 5. Known pitfalls
* `--init-from` fixes the architecture; flags that disagree now assert (a silent "u64" bug happened once).
* Node h004 on Anvil aborts MuJoCo/EGL rollouts (SIGABRT); excluded in every script.
* R2 convention: constant target dims are excluded from the average (`common/probes.py:_r2`).
* Their evaluator assumes latent ids start at 32001 (`<PAD>` = 32000 in openvla-7b); `finetune_libero_ctrl.py` asserts the tokenizer length before adding `<ACT_i>`.
* TensorFlow prints `AtomicFunction.__del__` tracebacks at interpreter exit; harmless.
