# Full-VLA apples-to-apples downstream (plan, 2026-09-13)

Goal: the one closed-loop comparison that can separate latents: UniVLA's own VLA recipe on LIBERO-spatial, same
architecture, same demos, same optimisation, only the latent-token labels swapped (their same-data LAM vs ours).
The toy heads (MLP, 4-frame transformer) put every latent and no latent inside one noise band (ledger 09-13).

## What UniVLA's LIBERO recipe is (`third_party/UniVLA/vla-scripts/finetune_libero.py`)
* Base: `vla_path` = their OXE-pretrained UniVLA-7B (Prismatic-7B + latent-token pretraining).  LoRA r=32 on all linear
  layers, lr 3.5e-4, batch 8 x grad-accum 2 per GPU, 30k steps, README says 8 GPUs (= 3.8M samples, ~70 epochs of
  LIBERO-spatial).  Image aug on.
* Labels: `RLDSBatchTransformLIBERO_withHis` (`prismatic/vla/datasets/datasets.py:84`): for a window of 12 frames the
  LAM encodes (frame 0, frame 11) -> 4 VQ indices (16-way) -> text `<ACT_i>` x 4 appended to the prompt
  "What action should the robot take to {lang}?"; with p = 0.5 a one-step-earlier window's tokens are given as
  "History action".  Loss = next-token CE on the 4 latent tokens + stop token.
* Action head: `ActionDecoder` = MAP block over the hidden states of the last 4 latent tokens + a MAP block over the
  visual tokens -> MLP -> 12 x 7 actions, L1.  Trained jointly.
* Eval: `experiments/robot/libero/run_libero_eval.py`, 50 episodes x 10 tasks, executes the 12-step chunk.

## The controlled version (both arms identical except the token labels)
* Base VLM: `openvla/openvla-7b` for BOTH arms (public, robot-conditioned, the code base is OpenVLA's).  Starting from
  their pretrained UniVLA-7B would favour their tokens (the LLM already speaks that 16-word vocabulary); starting
  from raw Prismatic gives the lowest absolute numbers.  Their own UniVLA-7B result stays as a reference row.
* Arm A (UniVLA, same data): `lam_path` = `claude_survey/ckpt/s2mix_1ep.pt` (their Stage-2 recipe on our capped mixture
  at our budget).  Check first that its state dict loads into `ControllableDINOLatentActionModel` (rala2 wraps it).
* Arm B (ours): tokens = per-cell VQ of `s1_acdc_1ep` (8 cells x Kc codes; Kc from the gate `libero_vqgate`, codebook
  fitted on the OXE validation dump, not on LIBERO).  Window 12 frames -> 5 frames at stride 3 -> DINO g4 -> effect cells
  -> 8 `<ACT_i>` tokens.  `ActionDecoder` reads the last 8 latent-token states instead of 4 (one-line change); prompt,
  history trick, loss, head, optimiser, steps, eval unchanged.
* Budget per arm: 4 x H100 (ai partition, one node), 8 x 2 x 4 = 64 samples per step.  Their recipe is 3.8M samples
  (30k steps x 128); we use 15k steps x 64 = 1M samples (~20 epochs of LIBERO-spatial) for both arms, identical.
  Estimated 4-5 h training per arm; eval 500 rollouts ~2-3 h per arm (parallelisable by task).  ~50-60 GPU-h total,
  balance 2,205 SU on `cis250950-ai`.
* Queue: whole-node jobs on ai have waited 1-3 days this week.  Deadline 09-17.  This is the risk.

## Steps
1. env `envs/univla_vla`: UniVLA's pins + fixes (protobuf 3.20.3, tensorflow_metadata 1.14, wandb 0.16.6, braceexpand,
   webdataset, bddl, easydict).  DONE 09-13; every prismatic module and the LIBERO evaluator import.
2. data: `data/libero_rlds/libero_spatial_no_noops` (432 episodes, 1.8 GB) + `cache/vla/openvla-7b` (15 GB).  DONE 09-13.
3. code: `downstream/vla/{acdc_tokens,datasets_ctrl,finetune_libero_ctrl,run_libero_eval_ctrl}.py`.  DONE 09-13; CPU check
   of one RLDS window -> 8 tokens -> prompt/labels passed; s2mix_1ep loads into their LAM class (frozen DINO from hub cache).
4. smoke on 1 GPU (`scripts/run_vla_smoke.sh`, jobs 20659988 ai / 20659989 gpu, 12 steps per arm + merge) -> queued.
   Then `LATENT=univla sbatch scripts/run_vla_ctrl.sh` and `LATENT=acdc CODEBOOK=<gate winner> sbatch ...` (4 x H100, 15k
   steps), then `RUN=<dir> sbatch scripts/run_vla_eval.sh` (50 trials x 10 tasks).
5. Report: success rate per task (50 x 10), token accuracy during training, and the same offline chain L1 for the two
   VLAs so the toy-head table and the VLA table share one axis.
