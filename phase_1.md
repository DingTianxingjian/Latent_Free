# Phase 1 — train a LAM baseline and PRLA, produce the first controlled comparison

## Goal of this phase, in one sentence

Produce **one table**: a self-trained UniVLA-style LAM baseline vs our method (PRLA), under
**identical data, backbone, compute and evaluation protocol**, measured with the probing
suite we already built. Absolute performance is not the goal in Phase 1 — a clean
controlled contrast is. Scale-up comes in Phase 2 (Bridge V2).

Two models, one pipeline. They share the data loader, the training loop, the evaluation
hooks, and the frozen encoder. **Write it as one codebase with a `--model {lam,prla}` switch**
— do not fork. Everything below assumes that.

---

## Shared setup

**Data.** LIBERO-spatial, existing `data.py`, the same episode-level 80/20 split, seed 0.
Keep every convention already verified (frame flip, `len(actions) == len(frames)`).

**Frozen encoder.** V-JEPA2, spatial pooling `g4` (4×4 grid, keep layout — global pooling is
forbidden, it costs ~50% of Δf decodability across encoders). Cache features to disk once;
both models read the cache. Never fine-tune the encoder in Phase 1.

**Temporal unit.** A window, not a frame pair: window of `4m+1` frames, m=1 (span 4) as the
default, matching the gating experiments. Target for evaluation is the net action over the
window's `4m` steps.

**Budget.** Both models ≤ ~200M trainable params, same optimizer/schedule/steps, same number
of GPU-hours. Log the step count and wall time for both — the claim is "under equal compute".

---

## Model A — LAM baseline (UniVLA-style, self-trained)

Reproduce the standard recipe on our frozen features:

* **IDM**: spatio-temporal transformer over the window's tokens + `n_q` learnable query
  tokens (start with 4, matching UniVLA's 4 code positions).
* **Bottleneck**: VQ, codebook size 16, matching UniVLA. Log codebook usage every epoch
  (we verified 16/16 used on LIBERO for the official weights — collapse would invalidate
  the comparison).
* **FDM**: spatial transformer taking `f(o_t)` (clean, *not* passed through the IDM) + the
  quantized codes, predicting the window-end features `f(o_{t+4m})`. Regression loss in
  feature space.
* **Critical architectural constraint** (this is what makes the bottleneck real): the only
  path from the future frames to the FDM is through the quantized codes. Assert it in code
  — no IDM-processed patch tokens may reach the decoder.

**Reproduction check before anything else.** Probe the trained baseline's `emb` and `z_q`
with our existing suite and compare against the official UniVLA weights we already measured:
state content under shuffle should land near **37–40% (emb)** and **10–26% (z_q)**;
incremental retention emb→z_q near **89–96%**. If our self-trained model is wildly off,
debug the reproduction before proceeding — an unrepresentative baseline makes the whole
comparison meaningless. Report the numbers side by side either way.

---

## Model B — PRLA (the method)

Same encoder, same window, same evaluation. Two stages.

**Stage 1 — passive predictor `P_A`.** Input: the fused latent of the *history* window
`[o_{t-4m..t}]`. Output: prediction of the *future* window's fused latent
`[o_{t+1..t+4m}]`. Loss: feature-space regression only. Train to convergence, then **freeze**.

> **Causality constraint — non-negotiable.** History and future must be encoded as
> **separate clips**. V-JEPA2's temporal attention is bidirectional, so feeding one long
> window would let the history tokens see the future, making `P_A` look artificially strong
> and shrinking the residual for the wrong reason. Two independent encoder calls, no shared
> attention across the boundary. Assert this in code and state it in the report.

`P_A` is trained **before and without `z`** on purpose: with no latent to fall back on, it
has no incentive to under-fit and leave predictable content for `z` to carry.

**Stage 2 — residual encoder `E` + corrector `G`, with `P_A` frozen.**
* `E` sees the true future fused latent and `P_A`'s prediction (or their difference) and
  outputs `(μ, σ)` for a **continuous** low-dimensional `z` (sweep dim ∈ {8, 16, 32}).
* `G` produces the correction: `pred_B = pred_A + G(z, pred_A)`. The residual form is
  hard-coded — `G` only corrects, it never predicts from scratch.
* Loss: `‖pred_B − true‖² + β · KL(q(z|·) ‖ N(0,I))`. **Sweep β over ~4 values spanning
  2 orders of magnitude** and report the rate–distortion curve (KL in nats vs correction
  gain). That curve is a figure in itself: it measures how many bits the decision actually
  needs.
* **No VQ.** Our own measurements make continuous + KL the justified choice.

---

## The comparison table (the deliverable)

Both models, identical probes, on the held-out episodes:

| metric | what it tests | LAM baseline | PRLA | expected |
|---|---|---|---|---|
| **state content under shuffle** (encode `(o_t, random other window)`, probe → action; ratio to the real-pair score; three OOD tiers as before) | does the latent waste capacity re-encoding `o_t`? | ~0.37–0.40 (emb) | **→ 0** | **primary claim** |
| **incremental R²** (`[f(o_t), z]` minus `f(o_t)` alone, per axis) | how much genuine transition does `z` carry beyond the state the policy already sees? | reference | **≥ baseline at equal or smaller capacity** | primary claim |
| latent capacity | bits / dims used to achieve the above | 4×log2(16) = 16 bits | z-dim + KL in nats | supporting |
| decodability under distractor (`patch moving`, existing 5-condition rig) | robustness to exogenous motion | reference | ≥ baseline | supporting |
| leakage check (probe `z` alone → future features, without `o_t`) | is `E` cheating by compressing the future? | both | both | sanity |

**Report per-axis** (xyz / rot / gripper), not just aggregates — rotation behaves differently
everywhere (visual footprint 3.66× smaller than translation) and aggregates hide it.

---

## Verdict rules, stated in advance

* **PRLA's shuffle state content drops far below the baseline's, with incremental R² not
  lower** → pillar ② holds, method is real, proceed to Phase 2 (Bridge V2).
* **State content does not drop** → debug in this order, and report which one it was:
  (1) `P_A` under-fit — check its standalone prediction error and whether the causality
  constraint is actually enforced; (2) β out of range — read it off the rate–distortion
  sweep; (3) `E` short-circuiting — the leakage check above.
* **State content drops but incremental R² collapses too** → `z` is empty. On LIBERO this is
  a *predicted* outcome, not necessarily a failure: scripted single-policy demos leave
  little unpredictable decision content. Say so explicitly and flag Bridge as the decisive
  test rather than declaring failure.

That last case matters — LIBERO is deliberately the least favourable dataset for this
method (deterministic scripted policy, no exogenous motion). Phase 1 tests the **mechanism**;
Phase 2 on Bridge V2 tests whether it **buys anything**.

---

## Practical notes

* Cache the frozen-encoder features once for all windows before training either model;
  extraction should not be repeated per run.
* Smoke-test both models on ~20 episodes and assert the architectural constraints (no future
  path to the FDM; no shared attention across the causal boundary) before submitting the
  full job.
* Log everything needed to claim equal compute: params, steps, wall time, GPU type.
* Commit the comparison table as a CSV plus a short verdict paragraph in SUMMARY, in the
  same style as the diagnostic sections.

## Parallel track (start immediately, independent of the above)

Begin **Bridge V2** data preparation: download, convert to the same interface as `data.py`,
and run the *existing* probing protocol on it (single / pair / pair_shuf / Δf, per-axis).
This produces two numbers we have predicted but never measured — the state-confusion ratio
and the transition budget on multi-operator real data — which are both the last missing
piece of the diagnostic paper and the setup for Phase 2. It needs no training and can run
while the Phase-1 jobs queue.


# Experiment: probe the Wan DiT *internal* features (not the VAE latent)

## Why this experiment decides the method's foundation

Everything we have measured so far leaves us with two candidate spaces for learning `z`,
and **both have a disqualifying weakness**:

* **DINOv2 patch grid** — best distractor robustness (top-128 retains **36%** of action under
  the 8% moving patch, vs Wan VAE's 4%), but it is an *image* encoder: no temporal path, so
  "learned temporal comparison" (our pillar ①) cannot come from the encoder itself.
* **Wan VAE latent** — has a genuine causal temporal path and fusion beats hand-differencing
  (0.858 vs 0.731 at matched width), but it is the *worst* action code under distractors
  (**4%**) because a pixel-reconstruction objective faithfully writes task-irrelevant motion
  into its top variance directions.

**Wan's DiT internal features are the untested third option**, and the only one that could be
good on both axes: they belong to the generative family (so `z` and the generation side stay
in one model), but the DiT is trained to *predict how video evolves*, not to reconstruct
pixels — closer in spirit to a predictive representation. Recent work argues video-generation
backbones are strong general-purpose vision learners (SOTA on depth, normals, camera pose),
which is consistent with that.

**Scope correction this experiment forces**: our current claim "reconstruction-target space is
the most distractor-susceptible (4%)" is about the **VAE latent**, not about video generative
models in general. Whatever this experiment finds, the SUMMARY wording must be tightened to
say *VAE latent* explicitly.

---

## The five things that must not be guessed

### 1. Null text conditioning — try zero context first, do NOT download umT5 yet

Wan's DiT is text-conditioned; we want text to contribute **nothing** so it is not a hidden
variable. Two options; take them in this order:

1. **Zero context** (preferred): pass a zero tensor of the expected context shape
   `[B, L_ctx, D_ctx]`. Read the expected `L_ctx`/`D_ctx` off the model config or the
   cross-attention module rather than assuming.
2. Only if (1) fails — NaNs, shape assertion, or degenerate/constant features — download umT5
   and encode the empty string `""`.

**Validation before running anything else**: encode ~200 frames with option (1) and check the
features are well-behaved — finite, non-constant across frames, per-dim std not collapsed
(compare against the Wan VAE latent's std ≈ 0.46 as a sanity reference). Print the check.
If it fails, fall back to (2) and say so in the report. Do not silently proceed.

### 2. Input: clips, not single frames — and define the fusion readout

The whole point of testing a video model is the temporal axis, so **the primary condition is
a clip**.

* **Clip length**: mirror Exp 5b exactly — a `4m+1`-frame window with **m=1 (span 4)** as the
  default. The VAE compresses 4× temporally, so a 5-frame window yields **2 latent frames**:
  `z0` (context, from `o_t`) and `z1` (fuses `o_{t+1..t+4}`). If time permits, add **m=2
  (span 8, 3 latent frames)** as a secondary row.
* **Target**: the net action over the window's `4m` steps (same as Exp 5b), so the numbers are
  directly comparable.
* **Fusion readout** — extract per-latent-frame features and build these rows:
  * `dit_zlast` — the DiT tokens belonging to the **last latent frame** (the fused motion
    latent). This is the headline row; it must be **width-matched** to the baselines below.
  * `dit_z0` — tokens of the first latent frame alone → the **single-frame / state baseline**.
  * `dit_diff` — `dit_zlast_from_a_separate_single_frame_encode(o_{t+4m}) − dit_z0`, i.e. two
    *independently encoded* single frames differenced. This is the hand-differencing baseline.
  * `dit_pair` — concat of the two independent single-frame encodings.
  * `dit_pair_shuf` — `dit_z0` concat a random other frame's single-frame encoding (the state
    control; keep the same three-tier shuffle discipline if cheap, otherwise the random tier
    alone is fine here).
* **Contract check (mandatory, mirrors Exp 5b)**: when a *single* frame is encoded on its own,
  the resulting feature must match the `z0` extracted from the clip, to a small relative error.
  Print it. If it does not match, the fused-vs-independent comparison is not honest and the
  run must stop.

### 3. Timestep and the noising recipe

Recipe, per window: **VAE-encode the clean frames → add noise at timestep `t` using the model's
own scheduler/sigma convention → one DiT forward → capture block outputs via hooks.** Do not
invent a noise formula; read the scheduler off the pipeline so the sigma convention matches
what the model was trained with. Print the actual sigma used for each `t`.

* **Primary**: the lowest practical noise level — `t` in the bottom ~5% of the schedule
  (near-clean input). Report the concrete `t` and sigma.
* **Sensitivity sweep**: two more values, roughly the ~25% and ~50% points of the schedule.
* **Fixed noise seed** across all conditions and all encoders so that clean/distractor
  comparisons differ only by the image content, not by the noise draw. State the seed.

**Reporting rule**: the headline table uses the primary `t`; the sweep goes in a secondary
table. If the ordering of conclusions changes across `t`, say so plainly — that itself is a
finding about how usable DiT features are.

### 4. Layers: 3–4 of the 30 blocks, early / middle / late, all reported

Capture with forward hooks at **blocks 5, 14, 22, 29** (0-indexed; adjust to the actual depth
if it is not 30 and state what you used). Run the full probe suite at every captured layer and
**report all of them**. Do not select the best layer and report only that — the layer sweep is
part of the result. If capturing four layers is too memory-heavy at 12k+ dims, cut to three
(early / middle / late) rather than dropping the sweep.

### 5. Spatial handling: unpatchify → 4×4 pool, never global pooling

DiT block outputs are a flat token sequence. **Reshape back to the spatial (and temporal) grid
before pooling** — `[B, T_lat, H_p, W_p, C]` — then pool spatially to a **4×4 grid** and
flatten, exactly as `pg4` does for DINOv2. Global mean pooling is forbidden: T3 says it costs
~half the action decodability, consistently across encoders.

Print the recovered grid shape and the final feature dim. If the token layout cannot be
recovered unambiguously from the config, stop and report rather than guessing the reshape.

---

## What to run, and against what

Everything else follows the established protocol: LIBERO-spatial, episode-level 80/20 split,
seed 0, ridge probe with the existing alpha grid, per-axis reporting (xyz / rot / cont6 /
gripper). **Start with a ~100-episode subset** to establish trends; only scale to the full
set if the DiT features look competitive.

Three questions, three tables, each with the DINOv2 `pg4` and Wan VAE columns beside it:

**Q1 — action decodability.** `dit_z0` / `dit_diff` / `dit_pair` / `dit_pair_shuf` /
`dit_zlast`. Reference: DINOv2 pg4 Δf **0.612** (s1) / **0.839** (s4); Wan VAE fused
`zlast` **0.858**.

**Q2 — distractor robustness (the load-bearing one).** Under the **8% moving patch** (the
clean motion-dominated condition; `border` deletes 64% of pixels and is too severe to
discriminate), run the null control: top-k vs random-k (5 seeds) vs bottom-k, refit inside
each, report `top-128 % of full` and `top/rand`. Reference: **DINOv2 36%, Wan VAE 4%**.

**Q3 — temporal fusion.** Does `dit_zlast` beat `dit_diff` at matched width, and is it *not*
explained by frame count? Include the frame-count null from Exp A: concat 3 and 5
**independently encoded** single-frame features, PCA-projected to matched width. Reference:
in Wan VAE, independent frames closed only **18%** of the fusion gap.

---

## Verdicts, written in advance

* **All three pass** (decodability comparable to DINOv2, top-128 retention well above the
  VAE's 4%, fusion beats differencing) → **the method is built on Wan DiT features**: `z` and
  the generation side live in one model family, and pillar ① comes from the encoder itself.
* **Decodability and fusion fine, robustness still ≈ VAE-level** → distractor susceptibility is
  a property of the *generative family*, not just of the VAE. Then **DINOv2 becomes the
  foundation** (best on T4), and pillar ① must instead come from a trainable spatio-temporal
  encoder over multi-frame DINO tokens — i.e. UniVLA's architecture plus window input. Say
  this plainly; it is a legitimate outcome, not a failure.
* **Decodability poor at every layer/timestep** → DiT features may need a non-linear readout,
  which is itself a finding, but they cannot serve as our frozen foundation. Report and move on.

---

## Deliverables

`results/exp7_wan_dit.csv` (one row per layer × timestep × condition × feature), a SUMMARY
section with the three tables side by side against DINOv2 and Wan VAE, and — regardless of the
outcome — the **T4 wording fix**: "reconstruction-target space" → "reconstruction-target **VAE
latent** space", with DiT results stated separately.

Report negatives without softening. A clean "DiT is no better than the VAE under distractors"
is worth as much here as a positive, because it decides where the method gets built.