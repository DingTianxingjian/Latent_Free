# Next task: phase-2 latent pretraining at 1/10 of UniVLA's scale

Written 2026-10-07 on Anvil, for running on the workstation (`~/Projects/latent_free`, 2 x RTX PRO 6000 96 GB,
62 GB RAM shared).  Background and every number below: `LEDGER.md`, entries 2026-10-02 and 2026-10-07.

## 0. What and why

UniVLA has three stages: (1) latent action model, (2) **pretraining of the VLM on latent-action pseudo-labels over the
cross-embodiment mixture**, (3) post-training on the robot task.  We have never run (2) at scale; it is where the claim
"a better latent action space transfers better" lives.  UniVLA's stage 2: Prismatic-7B full fine-tuning, global batch
1,024 x 20,000 steps = 20.5M samples, constant lr 2e-5, 960 A100-hours, OXE + GNM + Ego4D.

This task is the **1/10 point** of that curve, with the LoRA recipe registered on 2026-09-20 (30k steps x global batch
64 = 1.92M samples, ~1/10.7 of UniVLA).  Full fine-tuning is deferred to the full-scale run.

Known risk, measured 2026-10-07: from the first frame alone, a linear reader predicts the window's net effect partly
(R^2 .142) and its within-window profile not at all (.009).  The timing information A1 adds may act as target noise for
a single-frame policy.  Arm C below tests exactly that.

## 1. Arms (identical except the labeller)

| arm | labels | `--latent` | extra flags | where |
|---|---|---|---|---|
| A | TERRA (`latp2_inv`, [F0, N, A1]) | `tc_cont` | `--tc_tag latp2_inv` | **workstation** |
| B | UniVLA (`s2mix_1ep`, released TI prior) | `univla_cont` | -- | **workstation** |
| C | endpoint only (`tc_endpoint`, [F0, N], CoMo-style) | `tc_cont` | `--tc_tag tc_endpoint` | Anvil (tokenizer training now, job 21165123) |

Do NOT run A or B on Anvil as well -- Anvil only runs the 20-step smoke (job 21165072), the endpoint tokenizer and C.

Shared configuration (every arm, never changed between arms):

| item | value |
|---|---|
| backbone | `cache/vla/openvla-7b`, LoRA r32 (trainer default) |
| head / interface | `--head flow --query_tokens 4` |
| objective | latent only: `--lam_a 0` (OXE has no common action space) |
| data | `--data_root_dir data/oxe_raw --dataset_name acdc_oxe29` (UniVLA's 29 robot/navigation sources and weights, no Ego4D) |
| window | env `UNIVLA_FIXED_WINDOW=1` (both labellers need 14 frames on every source) |
| budget | `--max_steps 30000`, global batch 64 = 2 GPUs x `--batch_size 8` x `--grad_accumulation_steps 4` |
| lr / schedule | trainer defaults (3.5e-4, StepLR x0.1 at 80 %) |
| checkpoints | `--save_steps 1000 --merge False`; resume a cut run with `--resume True` and the SAME `--max_steps` |
| host RAM | env `CTRL_REITER_BATCHES=2500` (RLDS pipeline leak on 62 GB hosts); if RSS still grows, lower `--shuffle_buffer_size` -- and use the same value for every arm |

## 2. Prerequisites on the workstation

1. `git pull` (branch `main`), then re-apply the vendored-UniVLA patch -- `third_party/` is git-ignored, which is how
   the window fix once existed on one machine only:
   ```
   cd third_party/UniVLA && git apply ../../acdc_la/patches/univla_rlds_acdc.patch && cd ../..
   grep -c "_fixed_window\|acdc_oxe29" third_party/UniVLA/prismatic/vla/datasets/rlds/dataset.py \
        third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/mixtures.py      # both files must report > 0
   ```
2. Checkpoints (all in `transfer/acdc_la_artifacts.tar` on Anvil, or copy directly):
   - `acdc_la/ckpt/latp2_inv.pt` (373 MB) -- arm A labeller
   - `claude_survey/ckpt/s2mix_1ep.pt` (807 MB) -- arm B labeller
   - `cache/vla/openvla-7b` (15 GB), DINOv2 under `cache/torch/hub`
3. **OXE RLDS for the 29 sources: 3.55 TB** at `data/oxe_raw/<dataset>/<version>/` -- the largest blocker.  Needs >= 4 TB
   free disk.  Either rsync from Anvil (`/anvil/scratch/x-tding2/latent_free/data/oxe_raw/<dataset>`) or download from
   the Open X-Embodiment bucket (`gs://gresearch/robotics/<dataset>`).  The 29 names are the entries of `acdc_oxe29` in
   `third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/mixtures.py`.  If the disk cannot hold all of them, stop and
   decide which subset to drop BEFORE training -- every arm must see the same mixture.
4. Environment: the `univla_vla` venv (`acdc_la/envs/*.requirements.txt`).

## 3. Smoke (must pass for both arms before anything long)

```
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=2500
PY=envs/univla_vla/bin/python
for ARM in "tc_cont --tc_tag latp2_inv" "univla_cont"; do
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent $ARM --head flow --query_tokens 4 --lam_a 0 --merge False \
      --data_root_dir data/oxe_raw --dataset_name acdc_oxe29 \
      --batch_size 8 --grad_accumulation_steps 4 --max_steps 20 --save_steps 1000 --run_id_note p2smoke || break
done
```
Pass = both exit 0 and the latent loss is finite.  Record peak GPU memory (`nvidia-smi`) -- it has never been measured.

## 4. Phase 2, arms A and B (~2.5 s/step on 2 GPUs -> ~21 h per arm)

Same command as the smoke with `--max_steps 30000 --save_steps 1000 --run_id_note p2_ours` (arm A) and
`--run_id_note p2_univla` (arm B).  If a run is interrupted, rerun the identical command plus `--resume True`.
Run directory: `acdc_la/ckpt/vla_runs/<exp_id>--p2_ours` (the exp_id encodes latent, data, batch, lr, rank, steps).

## 5. Post-training and evaluation (per arm, Object and Long, 10 % demos)

```
export LIBERO_TRAIN_FRAC=10
P2=acdc_la/ckpt/vla_runs/<phase-2 run dir of the arm>
for SUITE in libero_object libero_10; do
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent <same as phase 2> --head flow --query_tokens 4 --init_from_run $P2 \
      --dataset_name ${SUITE}_no_noops --batch_size 8 --grad_accumulation_steps 4 --max_steps 15000 \
      --save_steps 5000 --run_id_note p2<ours|univla>_${SUITE}_frac10
done
```
Evaluate each post-trained run with the evaluation half of `acdc_la/scripts/workstation_run.sh` (tasks 0-4 on GPU 0,
5-9 on GPU 1, `run_libero_eval_cont`, 50 trials per task), then `python -m acdc_la.downstream.vla.eval_merge
--suite <suite> --note <note> --kind cont`.  500 episodes per suite, one merged JSON each.

## 6. Controls this comparison needs

| control | Object | Long |
|---|---|---|
| no phase 2, same interface (`workstation_run.sh LATENT=tc_cont` / `univla_cont`, FRAC=10, QT=4) | to run | to run |
| no latent at all (`workstation_run.sh LATENT=none`) | to run | to run |

(Goal already has them: 86.8 / 89.4 / 88.6, ledger 2026-09-19/20.)  Without the no-latent row, a phase-2 gain cannot be
attributed to the latent.  Two seeds per cell once the first pass is in.

## 7. Pre-registered reading at this 1/10 point

- Continue to the 1/3 point only if arm A beats its own no-phase-2 cell AND the no-latent cell on both suites by more
  than ~2 binomial SE (~3 pp at these success rates).
- A vs B decides whether TERRA's labels transfer better than UniVLA's; A vs C decides whether A1 helps the VLM at all.
- A tie everywhere is a result too: record it as such.

## 8. Recording (mandatory, same as on Anvil)

- After every run, finished or failed: a `LEDGER.md` entry (config, split, budget, number, control, status, paths,
  next decision).  Mark retractions explicitly; never leave a result only in chat or logs.
- Commit `acdc_la/results/vla_eval/*<note>*`, the merged JSONs and each run's `train_log.jsonl`; push `main`.
- Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01D41cpGfwM76yTZbfdphopa`.
- Never discretise our latent for the VLA.  Never change a shared setting for one arm only.
