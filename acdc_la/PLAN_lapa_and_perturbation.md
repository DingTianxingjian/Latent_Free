# Registered plans (2026-09-25): a LAPA-style baseline, and a closed-loop perturbation protocol

Neither has been run.  Both are blocked on GPU allocation (Anvil accounts at 0 SU).

## 1. LAPA-style baseline, inside our pipeline

LAPA (Ye et al., ICLR 2025), as published: encoder over a frame PAIR (x_t, x_{t+H}), H = 0.6 s on robot video;
discrete latent, vocabulary 8 x sequence 4 (8^4); NSVQ against codebook collapse; decoder reconstructs x_{t+H} pixels
from x_t and the latent (L2); latent pretraining of LWM-7B on the latent tokens, then action finetuning with
equal-mass discretised action bins.

Our version, `tc_lapa`: **LAPA's bottleneck inside the shared pipeline**, so the comparison with TERRA isolates the
latent (evidence + bottleneck), not the backbone.  It is not a reproduction of LAPA and must not be called one.

| | LAPA | `tc_lapa` | TERRA |
|---|---|---|---|
| evidence | (x_t, x_{t+H}) | `--tc endpoint` = [F_0, F_T - F_0], information-equivalent to the pair | [F_0, N, A_1] |
| window | 0.6 s | our 5-frame, stride-4 window (same as TERRA) | same |
| features | pixels, C-ViViT | frozen DINOv2 4x4 grid (shared) | same |
| bottleneck | discrete 8^4 (12 bits) | discrete, 4 tokens x codebook 8 | continuous 4 x 64 |
| quantiser | NSVQ | NSVQ (to add: z_q = z + ||z - q|| e/||e||, ~20 lines) | none |
| decoder target | future pixels | dF in DINO space given F_0 (shared) | same |
| budget | - | tc_cont's: `--steps 128000 --stop-at 65000`, batch 64, lr 3e-4, seed 0 | same |

Readouts: per-token code usage (bits); the NMSE table with paired source bootstrap against TERRA and UniVLA-noinh;
distractor drop.  Downstream: the token interface already used for UniVLA latents (discrete), same VLA recipe --
LAPA's latent pretraining corresponds to our phase-2 VLM latent pretraining.  Cost: ~8 GPU-h tokenizer + dump;
VLA 4 x H100 x ~5.5 h per suite.
Pre-registered: if fewer than half the codes are used at 65k, report it as collapse (NSVQ did not prevent it) rather
than tuning further.

## 2. Closed-loop perturbation protocol

Principles: (a) use the field's benchmark for scene-level perturbations -- LIBERO-Plus (7 dimensions, levels L1-L5) --
rather than inventing them; (b) never test with a family any arm trained on: our invariance term uses crop 0.9 and
global brightness/contrast U(0.8, 1.2), so those are excluded; (c) the same perturbed episodes for every arm.

Conditions (one level each, fixed in advance: LIBERO-Plus L3):
1. light conditions (LIBERO-Plus) -- scene lighting, not our global photometric jitter;
2. background textures (LIBERO-Plus);
3. objects layout / distractor objects (LIBERO-Plus);
4. the moving-patch distractor of the probes (`features.py::apply_distractor`, applied to every policy observation),
   labelled as our intervention -- it ties the closed loop to the probe drop.
Excluded: language perturbations and robot initial state (not what the latent claims); camera viewpoint is optional
and carries no directional prediction.

Arms (identical VLA recipe and checkpoints to the clean table): TERRA; UniVLA with random TI codebook (the fair
baseline); no-latent control.  Suites: LIBERO-Object and LIBERO-Long (the two in the qualitative figure); Spatial and
Goal if budget allows.  Episodes: 10 tasks x 50 trials = 500 per suite per condition, LIBERO's fixed initial states,
perturbation parameters deterministic in (task, trial), same seeds across arms.

Metrics: success; retention = perturbed / clean success of the same arm; Wilson 95 % intervals; paired per-task
differences with a bootstrap over tasks.  Pre-registered: TERRA's retention exceeds UniVLA-noinh's on conditions 3
and 4 (nuisance content), and is not lower on 1 and 2; no prediction for camera.

Qualitative figure (`figures/rollout_compare`): record videos for every episode; the pair shown is chosen by a rule
fixed now -- the lowest (task, trial) index, per suite, where TERRA succeeds and UniVLA-noinh fails under the same
seed -- and the appendix reports failure-type counts over ALL failed episodes of both arms.

Cost: 3 arms x 2 suites x 4 conditions = 24 evaluations x ~2-3 GPU-h (single GPU each) ~ 60 GPU-h.
